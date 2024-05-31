"""This script runs deep dreams / adversarial attack on an E5 model."""

import typing
from pprint import pprint

import e5_utils
import torch
import transformers
from torch.nn import functional as F  # noqa: N812
from transformers import AutoModel, AutoTokenizer

from .modeling_sparse_e5 import SparseE5
from .trainable_token_sequence import TrainableTokenSequence

x = typing


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


def create_sample_dataset() -> torch.Tensor:
    """Creates a sample dataset for testing.

    Returns:
        List[torch.Tensor]: A list of input texts.
    """
    # input_texts = [
    #     'query: how much protein should a female eat',
    #     'query: summit define',
    #     ("passage: As a general guideline, the CDC's average requirement of "
    #      "protein for women ages 19 to 70 is 46 grams per day. But, as you "
    #      "can see from this chart, you'll need to increase that if you're "
    #      "expecting or training for a marathon. Check out the chart below to "
    #      "see how much protein you should be eating each day."),
    #     ("passage: Definition of summit for English Language Learners. : "
    #      "1  the highest point of a mountain : the top of a mountain. : "
    #      "2  the highest level. : 3  a meeting or series of meetings between "
    #      "the leaders of two or more governments."),
    # ]

    input_texts = [
        "query: What is interesting about the city of Seattle?",
        "query: What are some of the elements of the periodic table?",
        "query: What wars were fought in the 20th century?",
        "passage: SAP is a software company.",
        "passage: ABAP is a programming language.",
        "passage: SuccessFactors is a human capital management company owned by SAP.",
    ]

    model = AutoModel.from_pretrained("intfloat/e5-small")

    embeddings, batch_dict = e5_utils.run_e5(model, input_texts)  # BSV

    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.dim() == 2, embeddings.dim()
    assert embeddings.shape[1] <= model.config.max_position_embeddings

    return embeddings, batch_dict


def create_batch_dict(input_ids: list[list[int]]):
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


def attack_e5_small(embeddings: torch.Tensor, reference_batch_dict: dict) -> None:
    """Attacks an instance of an E5 pretrained model."""

    def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    batch_size = embeddings.size(0)
    print("batch_size:", batch_size)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small")

    # Create one_hot version of e5 model.
    original_model = transformers.AutoModel.from_pretrained("intfloat/e5-small")
    assert isinstance(original_model, transformers.models.bert.modeling_bert.BertModel)
    model = SparseE5(original_model).to(DEVICE)  # These weights will not be updated.
    del original_model

    # Create an "input" tensor that will be backpropped into.
    # If there are 2 input texts, we will create 4 "input" tokens:
    # 2 for the query and 2 for the passage.
    seq = TrainableTokenSequence(
        batch_size=batch_size * 2,
        # sequence_length=model.config.max_position_embeddings,
        sequence_length=16,
        vocab_size=model.config.vocab_size,
        dropout=0.4,
    )

    # Tile the embeddings by 2x to account for the query and passage.
    embeddings = torch.tile(embeddings, (2, 1))
    embeddings = embeddings.to(DEVICE).detach()

    seq = seq.to(DEVICE)

    optimizer = torch.optim.AdamW(seq.parameters(), lr=0.3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=1000, verbose=True
    )
    # Microsoft E5 was trained on NCE loss. But we aren't interested in the negatives...
    # When we already have the target tokens to generate the target embedding,
    # we want a cosine similarity of 1.0 between the target and the output of the model.
    # The NCE loss reduces to negative cosine similarity for the positive pairs. No logs.

    cosine_embedding_loss = torch.nn.CosineEmbeddingLoss()

    def criterion(output, target):
        return cosine_embedding_loss(output, target, torch.Tensor([1]).to(DEVICE))

    epochs = 10000

    for epoch in range(epochs):
        optimizer.zero_grad()

        seq.train()
        input_ids = seq.forward()  # BSV: we got the one hots.

        batch_dict = create_batch_dict(input_ids)

        batch_dict = prepare_tensors(batch_dict)  # Convert to tensors.
        outputs = model(**batch_dict)

        pred = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

        # normalize embeddings
        pred = F.normalize(pred, p=2, dim=1)

        loss = criterion(pred, embeddings)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if epoch % 100 == 0:
            seq.eval()
            token_probs = seq.forward()

            strings = tokenizer.batch_decode(token_probs.argmax(dim=2).tolist())
            for idx, string in enumerate(strings):
                if idx % 3 == 0:
                    print()
                print(string)
            for idx, prob in enumerate(token_probs):
                if idx % 3 == 0:
                    print()
                print(prob.topk(1))

        if epoch == 0:
            pprint(reference_batch_dict["input_ids"])
            strings = tokenizer.batch_decode(reference_batch_dict["input_ids"])
            for string in strings:
                print(string)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss: {loss.item()} LR: {optimizer.param_groups[0]['lr']}")


if __name__ == "__main__":
    embeddings, batch_dict = create_sample_dataset()
    attack_e5_small(embeddings, batch_dict)
