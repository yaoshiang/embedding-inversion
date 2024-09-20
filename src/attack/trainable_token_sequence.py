"""A dense, soft token sequence for trainability."""

from typing import Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

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
        cls_ = torch.tile(F.one_hot(torch.tensor(101), vocab_size).float().reshape((1, -1, 1)), (1, depth, 1))
        query = torch.tile(F.one_hot(torch.tensor(23032), vocab_size).float().reshape((1, -1, 1)), (1, depth, 1))
        passage = torch.tile(F.one_hot(torch.tensor(6019), vocab_size).float().reshape((1, -1, 1)), (1, depth, 1))
        colon = torch.tile(F.one_hot(torch.tensor(1024), vocab_size).float().reshape((1, -1, 1)), (1, depth, 1))
        sep = torch.tile(F.one_hot(torch.tensor(102), vocab_size).float().reshape((1, -1, 1)), (1, depth, 1))
        pad = torch.tile(F.one_hot(torch.tensor(0), vocab_size).float().reshape((1, -1, 1)), (1, depth, 1))

        self.register_buffer("cls_", cls_)
        self.register_buffer("query", query)
        self.register_buffer("passage", passage)
        self.register_buffer("colon", colon)
        self.register_buffer("sep", sep)
        self.register_buffer("pad", pad)

        # Build the variable length trainable token logits.
        def build_logit(length: int, vocab_size: int, depth: int) -> torch.Tensor:
            return nn.Parameter(
                torch.normal(torch.zeros(length, vocab_size, depth), torch.ones(length, vocab_size, depth))
            )

        self.logits = nn.ParameterList([build_logit(length, vocab_size, depth) for length in lengths])

        # Expand the logits to include the special tokens.
        sequences = []
        for type_, length, logits_ in zip(types, lengths, self.logits):
            sequences.append([])
            sequences[-1].append(cls_)

            if type_ == QUERY:
                sequences[-1].append(query)
            elif type_ == PASSAGE:
                sequences[-1].append(passage)
            else:
                raise ValueError(f"Invalid type: {type_}")

            sequences[-1].append(colon)
            sequences[-1].append(logits_)
            sequences[-1].append(sep)
            sequences[-1].extend([torch.tile(pad, (depth,))] * (max_sequence_length - length - 4))

            sequences[-1] = torch.concat(sequences[-1], dim=0)
            assert sequences[-1].shape == (max_sequence_length, vocab_size, depth), sequences[-1].shape

        self.sequences = torch.stack(sequences, dim=0)
        assert self.sequences.shape == (batch_size, max_sequence_length, vocab_size, depth), self.sequences.shape

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self) -> torch.Tensor:
        """Forward pass.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, vocab_size)
            containing the log-prob of the one-hot representation of
            trainable "input" tokens to a sequence model.
        """
        if self.training:
            y = self.dropout(self.sequences)
        else:
            y = self.sequences

        y = torch.mean(y, dim=-1)
        assert y.shape == (self.batch_size, self.max_sequence_length, self.vocab_size), y.shape

        y = F.log_softmax(y, dim=-1)
        assert y.shape == (self.batch_size, self.max_sequence_length, self.vocab_size), y.shape

        return y
