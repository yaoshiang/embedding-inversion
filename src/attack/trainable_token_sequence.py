"""A dense, soft token sequence for trainability."""

from typing import Self, Sequence

import torch
import torch.nn.functional as F  # noqa: N812
import transformers

QUERY = 23032
PASSAGE = 6019


class TrainableTokenSequence(torch.nn.Module):
    """A class to implement trainable tokens.

    This is engineered for E5 models.

    The first token is always [CLS] or 101.
    The second is either 'query' 23032 or 'passage' 6019.
    The third is always ':' or 1024.
    After the valid tokens, the penultimate token is [SEP] or 102.
    The final tokens, if any, are always [PAD] or 0.
    """

    def __init__(
        self,
        batch_size: int,
        max_sequence_length: int,
        vocab_size: int,
        lengths: Sequence[int],
        types: Sequence[int],
        depth: int,
        dropout: float,
    ):
        """Initialize the TrainableTokenSequence class.

        The lengths and types are per example. Each are sequences that must be
        the same length as batch_size. The purpose is the same as masking:
        we want to train a sequence of a specific length per example.

        Args:
            batch_size (int): The batch size.
            max_sequence_length (int): The maximum sequence length.
            vocab_size (int): The size of the vocabulary.
            lengths (Sequence[int]): The lengths of the sequences.
            types (Sequence[str]): The types of the sequences.
            depth (int): The additional dimension of the token logits. This aids in stochasticity for dropout.
            dropout (float): The dropout rate.

        """
        raise NotImplementedError()

        if batch_size != len(lengths):
            raise ValueError("Batch size must match the lengths.")
        if batch_size != len(types):
            raise ValueError("Batch size must match the types.")
        if depth < 1:
            raise ValueError("Depth must be at least 1.")
        for length in lengths:
            if length > max_sequence_length - 4:
                raise ValueError(
                    "Lengths must be less than or equal to the max sequence length minus 4 (CLS, QUERY/PASSAGE, :, SEP)."
                )

        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.lengths = lengths
        self.types = types
        self.depth = depth

        super().__init__()

        # Setup some one-hot tensors for the special tokens.
        cls_ = torch.tile(F.one_hot(torch.tensor(101), vocab_size).float().reshape((1, -1, 1)), (1, 1, depth))
        query = torch.tile(F.one_hot(torch.tensor(23032), vocab_size).float().reshape((1, -1, 1)), (1, 1, depth))
        passage = torch.tile(F.one_hot(torch.tensor(6019), vocab_size).float().reshape((1, -1, 1)), (1, 1, depth))
        colon = torch.tile(F.one_hot(torch.tensor(1024), vocab_size).float().reshape((1, -1, 1)), (1, 1, depth))
        sep = torch.tile(F.one_hot(torch.tensor(102), vocab_size).float().reshape((1, -1, 1)), (1, 1, depth))
        pad = torch.tile(F.one_hot(torch.tensor(0), vocab_size).float().reshape((1, -1, 1)), (1, 1, depth))

        # Build the variable length trainable token logits.
        def build_logit(length: int, vocab_size: int, depth: int) -> torch.Tensor:
            torch.normal(torch.zeros(length, vocab_size, depth), torch.ones(length, vocab_size, depth))

        # Expand the logits to include the special tokens.
        sequences = []
        masks = []
        for type_, length, logits_ in zip(types, lengths, self.logits):
            sequences.append([])
            masks.append([])
 
            # Token #0
            sequences[-1].append(cls_)
            masks.append()
            # TODO: Continue build up the mask

            # Token #1
            if type_ == QUERY:
                sequences[-1].append(query)
            elif type_ == PASSAGE:
                sequences[-1].append(passage)
            else:
                raise ValueError(f"Invalid type: {type_}")

            # Token #2
            sequences[-1].append(colon)

            # Tokens [3, 3 + length)
            sequences[-1].append(logits_)

            # Token 3 + length
            sequences[-1].append(sep)

            # Tokens [4 + length, max_sequence_length)
            sequences[-1].extend([pad] * (max_sequence_length - length - 4))

            sequences[-1] = torch.concat(sequences[-1], dim=0)
            assert sequences[-1].shape == (max_sequence_length, vocab_size, depth), sequences[-1].shape

        sequences = torch.stack(sequences, dim=0)
        assert sequences.shape == (batch_size, max_sequence_length, vocab_size, depth), self.sequences.shape
        self.sequences = torch.nn.Parameter(sequences)
        assert self.sequences is not None



        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self) -> torch.Tensor:
        """Forward pass.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, vocab_size)
            containing the log-prob of the one-hot representation of
            trainable "input" tokens to a sequence model.
        """
        raise NotImplementedError
        # TODO: Allow backprop from some of the self.sequences parameters via
        # parameters * mask + parameters.detach() * (~mask)

        if self.training:
            y = self.dropout(self.sequences)
        else:
            y = self.sequences



        y = torch.mean(y, dim=-1)
        assert y.shape == (self.batch_size, self.max_sequence_length, self.vocab_size), y.shape

        y = F.log_softmax(y, dim=-1)
        assert y.shape == (self.batch_size, self.max_sequence_length, self.vocab_size), y.shape

        breakpoint()
        return y

    @classmethod
    def create_for_e5(
        cls, e5: transformers.models.bert.modeling_bert.BertModel, batch_dict: dict, depth: int, dropout: float
    ) -> Self:
        """Factory for e5 batch dicts.

        Args:
            e5: Instance of E5 model to grab params like max_seq_length.
            batch_dict: tokenized sequence.
            depth: Internal depth of logits.
            dropout: Dropout rate.

        Returns:
            Instance of TrainableTokenSequence
        """
        assert "input_ids" in batch_dict
        assert "attention_mask" in batch_dict
        assert "token_type_ids" in batch_dict

        B, S = batch_dict["input_ids"].size()
        assert (B, S) == batch_dict["attention_mask"].size()
        assert (B, S) == batch_dict["token_type_ids"].size()

        assert torch.all(batch_dict["input_ids"][:, 0] == 101), batch_dict["input_ids"][:, 0]
        assert torch.all((batch_dict["input_ids"][:, 1] == 23032) | (batch_dict["input_ids"][:, 1] == 6019))
        assert torch.all(batch_dict["input_ids"][:, 2] == 1024)

        assert torch.sum(batch_dict["token_type_ids"]) == 0

        lengths = batch_dict["attention_mask"].sum(dim=1) - 4  # CLS, query/passage, :, SEP
        types = batch_dict["input_ids"][:, 1]

        seq_layer = TrainableTokenSequence(
            batch_size=B,
            max_sequence_length=e5.config.max_position_embeddings,
            vocab_size=e5.config.vocab_size,
            lengths=lengths,
            types=types,
            depth=depth,
            dropout=dropout,
        )
        return seq_layer
