"""This script runs deep dreams / adversarial attack on an E5 model."""

from pprint import pprint

import torch
import transformers
from torch.nn import functional as F  # noqa: N812
from transformers import AutoTokenizer

from . import masked_lm_regularizer
from .modeling_dense_e5 import DenseE5
from .trainable_token_sequence import TrainableTokenSequence


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


def create_batch_dict(input_ids: torch.Tensor):
    """Creates a batch_dict to match a set of input_ids.

    Args:
        input_ids(torch.Tensor): Tensor of shape BSV.

    Returns:
        Dict[str, list[list[int]]]: A dictionary containing the input_ids, attention_mask, and token_type_ids.
    """
    batch_size, sequence_length = input_ids.shape[0:2]

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")

    # Assume all tokens are valid and use a simple mask (all positions are used)
    attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long).to(DEVICE)

    # Assuming no specific token type ids are needed, use zeros
    token_type_ids = torch.zeros(batch_size, sequence_length, dtype=torch.long).to(DEVICE)

    # Create batch dictionary
    batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

    return batch_dict


def prepare_tensors(batch_dict: dict) -> dict[str, torch.Tensor]:
    """Converts the values in a batch_dict to torch.Tensor.

    Args:
        batch_dict (dict): A dictionary containing the input_ids, attention_mask, and token_type_ids.
        device (torch.device): The device to use.

    Returns:
        dict[str, torch.Tensor]: A dictionary containing the input_ids, attention_mask, and token_type_ids as torch.Tensor.
    """
    for key in batch_dict:
        if not isinstance(batch_dict[key], torch.Tensor):
            batch_dict[key] = torch.tensor(batch_dict[key]).to(DEVICE)

    return batch_dict


def attack_e5_small(embeddings: torch.Tensor, reference_batch_dict: dict) -> list[str]:
    """Attacks an instance of an E5 pretrained model."""

    # Copied from the documentation.
    def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    B = embeddings.size(0)  # noqa: N806
    print("batch_size:", B)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small")

    # Create one_hot version of e5 model.
    # Default is SDPA (scaled dot product attention), does not work with this script.
    # See Pytorch specific parameters at https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/configuration
    original_model = transformers.AutoModel.from_pretrained("intfloat/e5-small", attn_implementation="eager")
    assert isinstance(original_model, transformers.models.bert.modeling_bert.BertModel)
    model = DenseE5(original_model).to(DEVICE)  # These weights will not be updated.
    del original_model

    # Create an "input" tensor that will be backpropped into.
    seq_layer = TrainableTokenSequence(
        batch_size=B,
        # sequence_length=model.config.max_position_embeddings,
        # sequence_length=16,
        sequence_length=64,
        vocab_size=model.config.vocab_size,
        depth=4,
        dropout=0.0,
    )
    seq_layer = seq_layer.to(DEVICE)

    embeddings = embeddings.to(DEVICE).detach()

    optimizer = torch.optim.AdamW(seq_layer.parameters(), lr=0.3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=1000, verbose=True
    )
    # Microsoft E5 was trained on NCE loss. But we aren't interested in the negatives...
    # When we already have the target tokens to generate the target embedding,
    # we want a cosine similarity of 1.0 between the target and the output of the model.
    # The NCE loss reduces to negative cosine similarity for the positive pairs. No logs.
    cosine_embedding_loss = torch.nn.CosineEmbeddingLoss()

    # We regularize with the masked language model loss.
    reg = masked_lm_regularizer.MaskedLMRegularizer(DEVICE)

    def criterion(output, target):
        return cosine_embedding_loss(output, target, torch.Tensor([1]).to(DEVICE))

    epochs = 1000

    for epoch in range(epochs):
        seq_layer.train()
        seq = seq_layer.forward()  # BSV, logprobs space
        batch_dict = create_batch_dict(torch.exp(seq))  # BSV, probs space

        batch_dict = prepare_tensors(batch_dict)
        outputs = model(**batch_dict)

        pred = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

        # normalize embeddings
        pred = F.normalize(pred, p=2, dim=1)

        loss = criterion(pred, embeddings)

        if 200 < epoch < 400:
            seq_layer.dropout.p = 0.1
            reg_loss = 0.1 * reg(seq, loss_type="smooth_l1")
            loss = loss + reg_loss
        elif 400 < epoch < 600:
            seq_layer.dropout.p = 0.1
            reg_loss = 0.1 * reg(seq, loss_type="smooth_l1")
            loss = loss + reg_loss
        else:
            seq_layer.dropout.p = 0.00
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if epoch == 0:
            pprint(reference_batch_dict["input_ids"])
            strings = tokenizer.batch_decode(reference_batch_dict["input_ids"])
            for string in strings:
                print(string)

        if epoch % 100 == 0:
            strings = tokenizer.batch_decode(seq.argmax(dim=2).tolist())
            for idx, string in enumerate(strings):
                print()
                print(string)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss: {loss.item()} LR: {optimizer.param_groups[0]['lr']}")

    predictions = tokenizer.batch_decode(seq.argmax(dim=2).tolist())
    return predictions
