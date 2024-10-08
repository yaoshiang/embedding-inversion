"""A dense, soft token sequence for trainability."""

from typing import Tuple

import torch
import transformers

CLS = 101
QUERY = 23032
PASSAGE = 6019
COLON = 1024
SEP = 102
PAD = 0


def _check_numerics(tensor) -> torch.Tensor:
    return not (torch.isnan(tensor).any() or torch.isinf(tensor).any())


# The first token is always [CLS] or 101.
# The second is either 'query' 23032 or 'passage' 6019.
# The third is always ':' or 1024.
# After the valid tokens, the penultimate token is [SEP] or 102.
# The final tokens, if any, are always [PAD] or 0.
class TrainableEmbedSequence(torch.nn.Module):
    """A class to implement trainable embeddings / tokens.

    This is engineered for E5 models.
    """

    @staticmethod
    def validate_batch_dict(batch_dict: dict):
        """Validates that a tokenizer's output is a valid comparison for the output of this layer.

        Args:
            batch_dict: the output of a tokenizer.
        """
        assert "input_ids" in batch_dict
        assert "attention_mask" in batch_dict
        assert "token_type_ids" in batch_dict

        B, S = batch_dict["input_ids"].size()
        assert (B, S) == batch_dict["attention_mask"].size()
        assert (B, S) == batch_dict["token_type_ids"].size()

        assert torch.all(batch_dict["input_ids"][:, 0] == CLS), batch_dict["input_ids"][:, 0]
        assert torch.all((batch_dict["input_ids"][:, 1] == QUERY) | (batch_dict["input_ids"][:, 1] == PASSAGE))
        assert torch.all(batch_dict["input_ids"][:, 2] == COLON)

        assert torch.sum(batch_dict["token_type_ids"]) == 0

    def __init__(
        self,
        e5: transformers.models.bert.modeling_bert.BertModel,
        reference_attention_mask: torch.Tensor,
    ):
        """Constructor.

        Args:
            e5: An instance of an E5 model. Key hyperparameters like max_sequence_length will be extracted.
            tokenizer: An instance of an E5 tokenizer.
            reference_attention_mask: The attention_maskof the original inputs.
            dropout (float): The dropout rate.
        """
        super().__init__()
        # Stash some hyperparameters.
        self.register_buffer("word_embeddings", e5.embeddings.word_embeddings.weight.clone().detach().to("cpu"))
        V = e5.embeddings.word_embeddings.weight.size(0)
        E = e5.embeddings.word_embeddings.weight.size(1)
        B = reference_attention_mask.size(0)
        S = reference_attention_mask.size(1)

        assert V == 30522
        assert E == 384
        assert S <= 512, S

        # Construct the input_embeds and mask.
        # emb = torch.randn(B, S, E) * 0.10 * reference_attention_mask.unsqueeze(-1)
        emb = torch.zeros(B, S, E) * reference_attention_mask.unsqueeze(-1)
        mask = reference_attention_mask.detach().clone()

        # Set positions 0, 1, and 2.
        with torch.no_grad():
            assert torch.all(mask[:, 0:3] == 1.0)  # First three should be one: CLS and QUERY/PASSAGE and COLON

            emb[:, 0, :] = self.word_embeddings[CLS]
            emb[:, 1, :] = self.word_embeddings[QUERY]
            emb[:, 2, :] = self.word_embeddings[COLON]

            mask[:, 0:3] = 0

            # Set positions 3 through length of trainable tokens.
            for idx, length in enumerate(reference_attention_mask.sum(dim=1)):
                mask[idx, length - 1 :] = 0.0  # turn off training for the final SEP token.

                emb[:, length - 1 :, :] = self.word_embeddings[PAD]

                assert length <= S
                assert length == S or mask[idx, length] == 0.0, mask[idx, length - 1]

        self.embeddings = torch.nn.Parameter(emb)
        self.register_buffer("mask", mask)
        self.register_buffer("attention_mask", reference_attention_mask.detach().clone())
        self.register_buffer("token_type_ids", torch.zeros(B, S, dtype=torch.int64))
        assert torch.all(self.embeddings[0][0] == self.word_embeddings[CLS])

    @staticmethod
    def normalize(embeddings: torch.Tensor) -> torch.Tensor:
        """Normalize the embeddings to have a standard deviation between 0.0838 and 0.1258.

        These are based on the word_embedding lookup table of BERT.

        Args:
            embeddings: A tensor of shape BSE.

        Returns:
            A tensor of shape BSE with the embeddings normalized.
        """
        assert _check_numerics(embeddings), embeddings

        # These values are generated in tools/word_embeddings.ipynb and
        # represent the min, mean, and max std of each token's embedding vector.
        MIN_STD = 0.0187  # noqa: N806
        MAX_STD = 0.1683  # noqa: N806
        MEAN_STD = 0.1192  # noqa: N806

        B, S, E = embeddings.size()

        std = embeddings.std(dim=-1)  # BSE -> BS
        need_reg_mask = ((std < MIN_STD) | (std > MAX_STD)) & (std != 0.0)  # BS
        need_reg_mask = need_reg_mask.unsqueeze(-1)  # BS
        # If the input is for E5, then the first 3 tokens should be fixed
        # as CLS, QUERY/PASSAGE, and COLON.

        assert _check_numerics(std), std
        assert _check_numerics(need_reg_mask), need_reg_mask

        assert not torch.any(need_reg_mask[:, 0:3]), std[:, 0:3]

        factors = MEAN_STD / std  # () / BS -> BS.
        factors = torch.nan_to_num(factors, nan=0.0, posinf=1.0, neginf=-1.0)

        assert factors.size() == std.size(), factors.size()
        assert _check_numerics(factors), factors

        factors = factors.unsqueeze(-1)  # BS -> BS1
        assert factors.size(0) == embeddings.size(0)
        assert factors.size(1) == embeddings.size(1)

        # normalized = torch.where(need_reg_mask, embeddings * factors, embeddings)
        normalized = embeddings

        assert _check_numerics(normalized), normalized

        return normalized

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Outputs the necessary inputs to a bert model (using input_embeds, not input_ids).

        Returns:
            input_embeds: Tensor of shape BSE, partially trainable.
            token_type_ids: tensor of shape BS, not trainable.
            attention_mask: Tensor of shape BS.
        """
        assert torch.all(self.embeddings[:, 0] == self.word_embeddings[CLS])
        assert _check_numerics(self.embeddings), self.embeddings

        # Apply the mask to control which embeddings are trainable
        trainable_embeds = self.mask.unsqueeze(-1) * self.embeddings
        non_trainable_embeds = (1 - self.mask).unsqueeze(-1) * self.embeddings.detach()
        inputs_embeds = trainable_embeds + non_trainable_embeds
        _check_numerics(inputs_embeds)

        return (
            self.normalize(inputs_embeds),
            self.token_type_ids,
            self.attention_mask,
        )

    # Functional Test:
    # a = torch.tensor([ 1, 2, 3])
    # b = torch.tensor([ 2, 3, 4])
    # assert torch.einsum("a, a -> a", a, b) == [2, 6, 12]

    def get_ids(self):
        embeds, _, _ = self.forward()  # B, S, E
        # self.word_embeddings  # T, E (30522, 384)

        B, S, E = embeds.size()
        T, E_ = self.word_embeddings.size()
        assert E == E_

        MODE = "dot"

        if MODE == "l1":
            distances = torch.cdist(embeds, self.word_embeddings.unsqueeze(0), p=1)  # BSE * 1TE -> BST
        elif MODE == "l2":
            distances = torch.cdist(embeds, self.word_embeddings.unsqueeze(0), p=2)  # BSE * 1TE -> BST
        elif MODE == "cosine":
            embeds_normalized = torch.nn.functional.normalize(embeds, p=2, dim=-1)  # BSE
            word_embeddings_normalized = torch.nn.functional.normalize(self.word_embeddings, p=2, dim=-1)  # TE
            distances = 1.0 - (torch.matmul(embeds_normalized, word_embeddings_normalized.T))  # BST
        elif MODE == "dot":
            distances = - torch.einsum("bse, te -> bst", embeds, self.word_embeddings)
        else:
            assert False, MODE

        # Find the index of the minimum distance (i.e., the closest embedding)
        ids = torch.argmin(distances, dim=-1)  # BS

        # Reshape back to (B, S) for the output
        ids = ids.reshape(B, S)

        return ids
