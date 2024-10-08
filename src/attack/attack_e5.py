"""This script runs deep dreams / adversarial attack on an E5 model."""

from pprint import pprint

import torch
import transformers
from torch.nn import functional as F  # noqa: N812
from transformers import AutoTokenizer

from .trainable_embed_sequence import TrainableEmbedSequence

torch.set_printoptions(threshold=float("inf"))


# Check for device availability
def get_device() -> torch.device:
    """Returns the device to use for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")  # Prefer CUDA if available
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Use MPS if CUDA is not available and MPS is
    else:
        return torch.device("cpu")  # Default to CPU if neither CUDA nor MPS is available


# Set the device based on availability
DEVICE = get_device()
print("Using device:", DEVICE)


def _check_numerics(tensor) -> torch.Tensor:
    return not (torch.isnan(tensor).any() or torch.isinf(tensor).any())


# Utility function copied from the E5 documentation.
def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def attack_e5_small(embeddings: torch.Tensor, reference_batch_dict: dict) -> list[str]:
    """Attacks an instance of an E5 pretrained model.

    Args:
        embeddings: The final embeddings generated by E5
        reference_batch_dict: The original batch_dict of the inputs. Only shape information
            will be used, not the actual token information (obviously).
    """
    B = embeddings.size(0)  # noqa: N806
    print("batch_size:", B)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small")

    # Create one_hot version of e5 model.
    # Default is SDPA (scaled dot product attention), does not work with this script.
    # See Pytorch specific parameters at https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/configuration
    e5 = transformers.AutoModel.from_pretrained("intfloat/e5-small", attn_implementation="eager").to(DEVICE).eval()
    assert isinstance(e5, transformers.models.bert.modeling_bert.BertModel)

    input_embed_layer = TrainableEmbedSequence(e5, reference_batch_dict["attention_mask"]).to(DEVICE)

    optimizer = torch.optim.AdamW(input_embed_layer.parameters(), lr=0.3, weight_decay=0.0)
    # optimizer = torch.optim.SGD(input_embed_layer.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, verbose=True)

    # # Microsoft E5 was trained on NCE loss. But we aren't interested in the negatives...
    # # When we already have the target tokens to generate the target embedding,
    # # we want a cosine similarity of 1.0 between the target and the output of the model.
    # # The NCE loss reduces to negative cosine similarity for the positive pairs. No logs.
    
    cosine_embedding_loss = torch.nn.CosineEmbeddingLoss()
    def criterion(output, target):
        return cosine_embedding_loss(output, target, torch.Tensor([1]).to(DEVICE))

    # def criterion(output, target):
    #     assert output.dim() == 2  # BE
    #     l = torch.abs(output - target)  # BE. Error.
    #     l = l**2  # BE. Squared Error.
    #     l = torch.mean(l, dim=-1)  # MSE per batch. B.
    #     l = torch.mean(l, dim=-1)  # Average of MSE per batch. scalar.

    #     return torch.mean(l)

    epochs = 1000

    embeddings = embeddings.to(DEVICE).detach()

    # Quick logs
    print("before training---------------")
    print("original query ids and string---")
    pprint(reference_batch_dict["input_ids"])
    strings = tokenizer.batch_decode(reference_batch_dict["input_ids"])
    for string in strings:
        print(string)

    print("trainable embeddings ids and string ---")
    inputs_embeds, token_type_ids, attention_mask = input_embed_layer()  # BSV, logprobs space
    ids = input_embed_layer.get_ids()
    strings = tokenizer.batch_decode(ids)
    for idx, string in enumerate(strings):
        print(inputs_embeds.shape)
        # print(token_type_ids)
        # print(attention_mask)
        print(ids)
        print(string)
    print("-------")

    for epoch in range(epochs):
        input_embed_layer.train()
        inputs_embeds, token_type_ids, attention_mask = input_embed_layer()  # BSV, logprobs space
        assert _check_numerics(inputs_embeds), inputs_embeds

        outputs = e5(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids, attention_mask=attention_mask)

        pred = average_pool(outputs.last_hidden_state, attention_mask)

        # normalize embeddings
        pred = F.normalize(pred, p=2, dim=1)

        loss = criterion(pred, embeddings)

        if 200 < epoch < 400:
            pass
        elif 400 < epoch < 600:
            pass
        else:
            pass
        optimizer.zero_grad()
        loss.backward()

        assert _check_numerics(input_embed_layer.embeddings)
        assert _check_numerics(input_embed_layer.embeddings.grad)
        optimizer.step()

        assert _check_numerics(input_embed_layer.embeddings)

        scheduler.step(loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss: {loss.item()} LR: {optimizer.param_groups[0]['lr']}")

        if epoch % 100 == 0:
            inputs_embeds, token_type_ids, attention_mask = input_embed_layer()  # BSV, logprobs space
            ids = input_embed_layer.get_ids()
            strings = tokenizer.batch_decode(ids)
            for idx, string in enumerate(strings):
                print(inputs_embeds.shape)
                # print(token_type_ids)
                # print(attention_mask)
                print(ids)
                print(string)
