# --------------------------------------------------------------------------------
# This file contains components adapted from Huggingface Transformers, which is licensed under the Apache License, Version 2.0.
# This usage is in compliance with the Apache 2.0 license. However, the modifications and additional proprietary code
# in this project are not disclosed and remain confidential as part of ExampleProject, which is a proprietary product.
#
# Copyright (C) 2024 Yaoshiang Ho
#
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0
#
# NOTICE: The following changes have been made to the original source:
# - One hot implementation of Bert Embeddings to allow backpropagation through the embedding layer.
#
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
#
# The original NOTICE and LICENSE from ExampleLib are included as required by the Apache 2.0 license.
# --------------------------------------------------------------------------------

# Also see the implementation for the attributes to pull out:
# https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/sparse.py

# See the guide to adding models to Transformers here:
# https://huggingface.co/docs/transformers/en/add_new_model

from typing import Optional, Tuple, Union

import torch
import transformers
from torch import nn

from .dense_embedding import DenseEmbedding


class DenseBertEmbeddings(nn.Module):
    """Dense variation of BertEmbeddings layer.

    Includes word, position and token_type embeddings. Takes the weights from
    an existing BertEmbeddings layer and replaces the word embeddings with a
    Dense embedding layer. Reuses the position and type embeddings.

    Args:
        bert_embeddings (transformers.models.bert.modeling_bert.BertEmbeddings):
            A pretrained BertEmbeddings layer.
    """

    def __init__(self, bert_embeddings: transformers.models.bert.modeling_bert.BertEmbeddings):
        super().__init__()
        self.word_embeddings = DenseEmbedding(bert_embeddings.word_embeddings)
        self.position_embeddings = bert_embeddings.position_embeddings
        self.token_type_embeddings = bert_embeddings.token_type_embeddings

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = bert_embeddings.LayerNorm  # pylint: disable=invalid-name
        self.dropout = bert_embeddings.dropout

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = bert_embeddings.position_embedding_type
        self.register_buffer(
            "position_ids", torch.arange(self.position_embeddings.num_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        assert input_ids is not None, "Input IDs must be provided"

        # if input_ids is not None:
        input_shape = input_ids.size()
        # else:
        # input_shape = inputs_embeds.size()[:-1]

        # This is still right becuase the shape went from BS to BSV
        # (Batchsize, Sequence Length) to (Batchsize, Sequence Length, Vocab Size)
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # if inputs_embeds is None:
        # This is the onehot word embedding layer, not the original nn.Embedding layer.
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class DenseE5PreTrainedModel(transformers.modeling_utils.PreTrainedModel):
    pass


# Adapted from https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert
class DenseE5(DenseE5PreTrainedModel):
    """Wraps a pre-trained BertModel.

    Takes an existing pretrained E5 model (an instance of BertModel)
    and replaces the word embeddings with a one-hot embedding layer.

    The purpose of this model is to run "deep dreams" / "adversarial attack" on the Microsoft E5
    embedding model, which uses Bert architecture.

    For simplicity, modes that the general Bert model supports, but are unused by Microsoft E5, are not implemented:
    - Cross-attention
    - Decoder mode
    - etc.

    Args:
        bert_model (transformers.models.bert.modeling_bert.BertModel):
            A pretrained E5 model (an instance of BertModel with a specific configuration).
    """

    def __init__(self, model: transformers.models.bert.modeling_bert.BertModel):  # config, add_pooling_layer=True):
        super().__init__(model.config)

        self.config = model.config

        self.embeddings = DenseBertEmbeddings(model.embeddings)
        self.encoder = model.encoder

        self.pooler = (
            transformers.models.bert.modeling_bert.BertPooler(self.config) if model.pooler is not None else None
        )

        self.attn_implementation = self.config._attn_implementation
        self.position_embedding_type = self.config.position_embedding_type

        # Initialize weights and apply final processing
        self.post_init()

    # def get_input_embeddings(self):
    #     return self.embeddings.word_embeddings

    # def set_input_embeddings(self, value):
    #     self.embeddings.word_embeddings = value

    # def _prune_heads(self, heads_to_prune):
    #     """
    #     Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
    #     class PreTrainedModel
    #     """
    #     for layer, heads in heads_to_prune.items():
    #         self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[
        Tuple[torch.Tensor], transformers.models.bert.modeling_bert.BaseModelOutputWithPoolingAndCrossAttentions
    ]:
        """Forward pass for the model.

        This method does not support many of the args in the original forward pass in Bert,
        as they are not used by the Microsoft E5 model. The Microsoft E5 model only
        uses the following args:

        ['input_ids', 'token_type_ids', 'attention_mask']

        Also, the token_type_ids are always zero. The query vs passage distinction is handled by the input_ids.

        Args:
            input_ids (Optional[torch.Tensor]): The input token ids as one-hot.
            token_type_ids (Optional[torch.Tensor]): The token type ids. Always all zero.
            attention_mask (Optional[torch.Tensor]): The attention mask. Not sure if it's really imporatant.
        """
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_attentions = self.config.output_attentions

        output_hidden_states = self.config.output_hidden_states
        # output_hidden_states = (output_hidden_states
        #                         if output_hidden_states is not None else
        #                         self.config.output_hidden_states)

        assert self.config.is_decoder is False, "Decoder mode not supported"
        use_cache = False
        # if self.config.is_decoder:
        #     use_cache = use_cache if use_cache is not None else self.config.use_cache
        # else:
        #     use_cache = False

        assert input_ids is not None, "Input IDs must be provided"
        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
        assert (
            3 == input_ids.dim()
        ), f"Input shape must be 3D (batch size, sequence length, vocab size), dims: {input_ids.dim()}, shape: {input_ids.shape}"
        assert (
            input_shape[1] <= self.config.max_position_embeddings
        ), "Sequence length must be less-than-equal than max position embeddings"
        assert input_shape[2] == self.config.vocab_size, "Vocab size must match the model's vocab size"

        # if input_ids is not None and inputs_embeds is not None:
        #     raise ValueError(
        #         "You cannot specify both input_ids and inputs_embeds at the same time"
        #     )
        # elif input_ids is not None:
        #     self.warn_if_padding_and_no_attention_mask(input_ids,
        #                                                attention_mask)
        #     input_shape = input_ids.size()
        # elif inputs_embeds is not None:
        #     input_shape = inputs_embeds.size()[:-1]
        # else:
        #     raise ValueError(
        #         "You have to specify either input_ids or inputs_embeds")

        # input_shape is no longer BS, it is now BSV
        # batch_size, seq_length = input_shape
        batch_size, seq_length = input_shape[0], input_shape[1]

        device = input_ids.device
        # device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = 0
        # past_key_values_length = past_key_values[0][0].shape[
        #     2] if past_key_values is not None else 0

        assert token_type_ids is not None, "Token type IDs must be provided"
        # if token_type_ids is None:
        #     if hasattr(self.embeddings, "token_type_ids"):
        #         buffered_token_type_ids = self.embeddings.token_type_ids[:, :
        #                                                                  seq_length]
        #         buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
        #             batch_size, seq_length)
        #         token_type_ids = buffered_token_type_ids_expanded
        #     else:
        #         token_type_ids = torch.zeros(input_shape,
        #                                      dtype=torch.long,
        #                                      device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            # position_ids=position_ids,
            token_type_ids=token_type_ids,
            # inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)

        assert self.attn_implementation != "sdpa", "SDPA not supported"

        # use_sdpa_attention_masks = (self.attn_implementation == "sdpa" and
        #                             self.position_embedding_type == "absolute"
        #                             and head_mask is None
        #                             and not output_attentions)

        # Expand the attention mask
        # if use_sdpa_attention_masks:
        #     # Expand the attention mask for SDPA.
        #     # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
        #     if self.config.is_decoder:
        #         extended_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
        #             attention_mask,
        #             input_shape,
        #             embedding_output,
        #             past_key_values_length,
        #         )
        #     else:
        #         extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
        #             attention_mask, embedding_output.dtype, tgt_len=seq_length)

        # else:
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        assert self.config.is_decoder == False, "Decoder mode not supported"
        # if self.config.is_decoder and encoder_hidden_states is not None:
        #     encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size(
        #     )
        #     encoder_hidden_shape = (encoder_batch_size,
        #                             encoder_sequence_length)
        #     if encoder_attention_mask is None:
        #         encoder_attention_mask = torch.ones(encoder_hidden_shape,
        #                                             device=device)

        #     if use_sdpa_attention_masks:
        #         # Expand the attention mask for SDPA.
        #         # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
        #         encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
        #             encoder_attention_mask,
        #             embedding_output.dtype,
        #             tgt_len=seq_length)
        #     else:
        #         encoder_extended_attention_mask = self.invert_attention_mask(
        #             encoder_attention_mask)
        # else:
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask = self.get_head_mask(head_mask,
        #                                self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            # head_mask=head_mask,
            # encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            # past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return transformers.models.bert.modeling_bert.BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
