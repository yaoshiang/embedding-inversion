import torch
from torch import nn
from transformers import AutoTokenizer

from .sparse_embedding import SparseEmbedding


def test_one_hot_embedding_equivalence():  # noqa: D103
    # ---Arrange---
    # Load tokenizer and pretrained embedding layer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    embedding_dim = 768  # Assume the embedding dimension for BERT base

    # Create a pretrained embedding layer and randomly initialize for the sake of testing
    pretrained_embedding = nn.Embedding(vocab_size, embedding_dim)
    pretrained_embedding.weight.data.normal_()

    # Instantiate the custom OneHotEmbedding layer
    custom_embedding = SparseEmbedding(pretrained_embedding)

    # Define a sample input text
    input_texts = ["query: how much protein should a female eat", "query: summit define"]

    # Test each input text
    for text in input_texts:
        # Convert text to token ids
        input_ids = tokenizer.encode(text, return_tensors="pt")

        # Get standard embeddings
        standard_embeddings = pretrained_embedding(input_ids.squeeze()).detach()

        # ---Act---

        # Create one-hot vectors for the same tokens
        one_hot_vectors = torch.nn.functional.one_hot(input_ids, num_classes=vocab_size).squeeze().float()

        # Get custom one-hot embeddings
        one_hot_embeddings = custom_embedding(one_hot_vectors).detach()

        # ---Assert---

        # Assert the equivalence of both embedding outputs
        torch.testing.assert_close(standard_embeddings, one_hot_embeddings, rtol=1e-4, atol=1e-6)
