"""Differentiable version of torch.nn.Embedding."""

import torch
from torch import nn


class SparseEmbedding(nn.Module):
    """Differentiable version of torch.nn.Embedding.

    This version takes one-hot encoded inputs and returns the corresponding embeddings.
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

    def forward(self, one_hot_vectors):
        """Forward pass.

        Args:
            one_hot_vectors (torch.Tensor): A tensor of shape (batch_size, sequence_length, vocab_size) containing one-hot encoded vectors.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, embedding_dim) containing the embeddings of the input one-hot vectors.
        """
        # Validate that each one-hot vector is a probability distribution
        assert torch.allclose(one_hot_vectors.sum(dim=-1), torch.ones_like(one_hot_vectors.sum(dim=-1)))

        embedded = torch.matmul(one_hot_vectors, self.embedding.weight)
        return embedded
