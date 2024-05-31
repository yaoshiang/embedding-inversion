# Copyright authors of https://huggingface.co/intfloat/e5-small
"""Utility functions for the Microsoft E5 embedding model."""

from typing import List, Tuple

import torch.nn.functional as F  # noqa: F812
import transformers
from torch import Tensor
from transformers import AutoTokenizer

from .modeling_sparse_e5 import SparseE5


def run_e5(model: transformers.models.bert.modeling_bert.BertModel, input_texts: List[str]) -> Tuple[Tensor, dict]:
    """Run the model on a list of input texts.

    Based on the snippet from https://huggingface.co/intfloat/e5-small

    Args:
        model: A BERT or Distilbert model.
        input_texts (List[str]): A list of input texts.
        Each input text should start with "query: " or "passage: ".
        For tasks other than retrieval, you can simply use the "query: " prefix.

    Returns:
        List[Tensor]: A list of tensors containing the embeddings of the input texts.
    """

    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small")

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # scores = (embeddings[:2] @ embeddings[2:].T) * 100
    # print(scores.tolist())
    return embeddings, batch_dict


def run_one_hot_e5(model: SparseE5, input_texts: List[str]) -> Tuple[Tensor, dict]:
    """Run the model on a list of input texts.

    Adapted from the snippet at https://huggingface.co/intfloat/e5-small.

    Args:
        texts (List[str]): A list of input texts.
        Each input text should start with "query: " or "passage: ".
        For tasks other than retrieval, you can simply use the "query: " prefix.

    Returns:
        List[Tensor]: A list of tensors containing the embeddings of the input texts.
    """
    assert isinstance(model, SparseE5)
    assert input_texts

    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small")

    # Tokenize the input texts
    batch_dict = tokenizer(
        input_texts, max_length=model.config.max_position_embeddings, padding=True, truncation=True, return_tensors="pt"
    )

    batch_dict["input_ids"] = F.one_hot(batch_dict["input_ids"], num_classes=model.config.vocab_size).float()
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # scores = (embeddings[:2] @ embeddings[2:].T) * 100
    # print(scores.tolist())
    return embeddings, batch_dict
