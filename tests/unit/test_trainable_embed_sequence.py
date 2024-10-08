import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from attack.trainable_embed_sequence import TrainableEmbedSequence


def test_init():
    # Arrange
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small")
    query = "query: what?"
    batch_dict = tokenizer(query, return_tensors="pt", padding="max_length", max_length=512)

    e5 = AutoModel.from_pretrained("intfloat/e5-small")
    embed_layer = TrainableEmbedSequence(e5, batch_dict["attention_mask"])

    # Act
    _, token_type_ids, attention_mask = embed_layer.forward()

    # Assert
    TrainableEmbedSequence.validate_batch_dict(batch_dict)  # Raises if error.
    assert attention_mask.size() == (1, 512), attention_mask.size()
    assert token_type_ids.size() == (1, 512), token_type_ids.size()
    assert token_type_ids.sum() == 0
    assert attention_mask.sum() == 6


def test_trainability():
    # Arrange
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small")
    query = "query: what?"
    batch_dict = tokenizer(query, return_tensors="pt", padding=True, max_length=512)
    e5 = AutoModel.from_pretrained("intfloat/e5-small")
    embed_layer = TrainableEmbedSequence(e5, batch_dict["attention_mask"])
    # XXX: This violate the rule of unit tests not poking at internal state... 
    # But we need to simulate a round of backprop on non-zero weights to noise the zero initialization. 
    # 3 and 4 are the trainable parts of the embedding. 
    with torch.no_grad():
        embed_layer.embeddings[:, 3:5, :] += 0.1 * torch.randn(embed_layer.embeddings.size(0), 2, embed_layer.embeddings.size(2))
    input_embeds, _, _ = embed_layer.forward()

    # Act
    loss = (input_embeds * input_embeds).sum()
    loss.backward()

    # Assert
    assert embed_layer.embeddings.grad[0, 0].sum() == 0  # CLS
    assert embed_layer.embeddings.grad[0, 1].sum() == 0  # QUERY/PASSAGE
    assert embed_layer.embeddings.grad[0, 2].sum() == 0  # COLON
    assert (
        embed_layer.embeddings.grad[0, 3].sum()
        == 2.0 * embed_layer.embeddings[0, 3].sum()
    )  # what
    assert (
        embed_layer.embeddings.grad[0, 4].sum()
        == 2.0 * embed_layer.embeddings[0, 4].sum()
    )  # ?
    assert embed_layer.embeddings.grad[0, 3].sum() != 0.0
    assert embed_layer.embeddings.grad[0, 4].sum() != 0.0
    assert embed_layer.embeddings.grad[0, 5].sum() == 0  # SEP


def test_static_ids():
    # Arrange
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small")
    query = "query: what?"
    batch_dict = tokenizer(query, return_tensors="pt", padding="max_length", max_length=512)
    e5 = AutoModel.from_pretrained("intfloat/e5-small")
    embed_layer = TrainableEmbedSequence(e5, batch_dict["attention_mask"])

    # Assert
    ids = embed_layer.get_ids()

    # Act
    input_embeds, _, _ = embed_layer.forward()
    loss = (input_embeds * input_embeds).sum()
    loss.backward()

    # Assert
    ids = embed_layer.get_ids()
    assert ids[0, 0] == 101
    assert ids[0, 2] == 1024
    assert ids[0, 2] != 103

def test_normalization():
    # Arrange
    MIN_STD = 0.0187  # noqa: N806
    MAX_STD = 0.1683  # noqa: N806
    MEAN_STD = pytest.approx(0.1192)  # noqa: N806
    embs = torch.randn((1, 6, 384))
    embs[:, 0, :] *= 0.1 / torch.std(embs[:, 0, :])  # CLS
    embs[:, 1, :] *= 0.1 / torch.std(embs[:, 1, :])  # QUERY/PASSAGE
    embs[:, 2, :] *= 0.1 / torch.std(embs[:, 2, :])  # COLON
    embs[:, 3, :] *= 0.01  # hello
    embs[:, 4, :] *= 0.2  # world
    embs[:, 5, :] *= 0.1 / torch.std(embs[:, 5, :])  # SEP

    # Act
    inputs_embeds = TrainableEmbedSequence.normalize(embs)

    # Assert
    assert inputs_embeds[:, 0, :].std() == pytest.approx(0.1), inputs_embeds[:, 0, :].std()
    assert inputs_embeds[:, 1, :].std() == pytest.approx(0.1), inputs_embeds[:, 1, :].std()
    assert inputs_embeds[:, 2, :].std() == pytest.approx(0.1), inputs_embeds[:, 2, :].std()
    assert inputs_embeds[:, 3, :].std() == pytest.approx(MEAN_STD), inputs_embeds[:, 3, :].std()
    assert inputs_embeds[:, 4, :].std() == pytest.approx(MEAN_STD), inputs_embeds[:, 4, :].std()
    assert inputs_embeds[:, 5, :].std() == pytest.approx(0.1), inputs_embeds[:, 5, :].std()
