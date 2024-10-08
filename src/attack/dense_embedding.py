"""Differentiable version of torch.nn.Embedding."""

import torch
from torch import nn


class DenseEmbedding(nn.Module):
    """Differentiable version of torch.nn.Embedding.

    This version takes dense encoded inputs (such as one-hots) and returns the corresponding embeddings.
    This is useful when you want to backprop through the embedding layer to learn the underlying words.
    """

    def __init__(self, pretrained_embedding_layer: nn.Embedding):
        """Initialize the SparseEmbedding class using an existing, trained embedding layer.

        The trained embedding layer's weights will be mutated by this class.

        Args:
            pretrained_embedding_layer (torch.nn.Embedding): A pretrained embedding layer.
        """
        super().__init__()
        # Ensure the embedding weights are not updated
        self.embedding = pretrained_embedding_layer
        self.embedding.weight.requires_grad = False  # Freeze the embedding weights

    def forward(self, dense_vectors):
        """Forward pass.

        Args:
            dense_vectors (torch.Tensor): A tensor of shape (batch_size, sequence_length, vocab_size) containing one-hot or probability distributions of the vocab.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, embedding_dim) containing the embeddings of the inputs.
        """
        # Validate that each one-hot vector is a probability distribution
        assert torch.allclose(
            dense_vectors.sum(dim=-1), torch.ones_like(dense_vectors.sum(dim=-1)), atol=1e-4
        ), f"Input one-hot vectors must sum to 1. Got {dense_vectors.sum(dim=-1)}."
        embedded = torch.matmul(dense_vectors, self.embedding.weight)
        return embedded
