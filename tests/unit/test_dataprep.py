import torch

from attack import dataprep


def test_hotpotqa():
    # Arrange
    MAX = 6
    # Act
    texts, reconstructed_texts, batch_dict, embeddings = dataprep.create_hotpotqa_dataset_for_e5(
        max_examples=MAX, max_length=512
    )

    # Assert - Got MAX number of elements in the three returned values.
    assert len(texts) == MAX
    assert embeddings.size(0) == MAX
    for vals in batch_dict.values():
        assert len(vals) == MAX

    # Assert - texts and embeddings
    for text, emb in zip(texts, embeddings):
        assert isinstance(text, str)
        assert text.startswith("query: ") or text.startswith("passage: ")
        assert isinstance(emb, torch.Tensor)

    # Assert - batch_dict
    assert "input_ids" in batch_dict
    assert "token_type_ids" in batch_dict
    assert "attention_mask" in batch_dict
    assert torch.all((batch_dict["input_ids"] == 0) == (batch_dict["attention_mask"] == 0))
    assert 0 == batch_dict["token_type_ids"].sum()
