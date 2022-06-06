#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertSelfOutput, BertIntermediate, BertOutput
from transformers import PretrainedConfig, AutoConfig, AutoModelForQuestionAnswering
from transformers.models.albert.modeling_albert import (
    AlbertPreTrainedModel,
    AlbertModel,
)
import re
import math
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertModel,
)
from transformers.modeling_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.modeling_outputs import MultipleChoiceModelOutput


class RelationEmbedding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        if config.share_relation_across_head:
            self.relation_embedding = nn.Embedding(config.relation_type_num, self.attention_head_size)
        else:
            self.relation_embedding = nn.Embedding(config.relation_type_num, config.hidden_size)
        self.relation_type_num = config.relation_type_num
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self, evidence_type
    ):
        with torch.no_grad():
            if self.relation_type_num == 16:
                relation_matrix = (evidence_type.unsqueeze(-1) * 4 + evidence_type.unsqueeze(-2)).long()
            elif self.relation_type_num == 10:
                m1 = evidence_type.unsqueeze(-1) * 4 + evidence_type.unsqueeze(-2) - (
                            evidence_type * (evidence_type + 1) / 2).unsqueeze(-1)
                # print(evidence_type.unsqueeze(-1) * 4 + evidence_type.unsqueeze(-2))
                # print((evidence_type * (evidence_type + 1)).unsqueeze(-1))
                # print(m1)
                m2 = evidence_type.unsqueeze(-1) + evidence_type.unsqueeze(-2) * 4 - (
                            evidence_type * (evidence_type + 1) / 2).unsqueeze(-2)
                # print(m2)
                relation_matrix = torch.where(evidence_type.unsqueeze(-1) <= evidence_type.unsqueeze(-2), m1, m2).long()
            elif self.relation_type_num == 7:
                m1 = evidence_type.unsqueeze(-1) * 4 + evidence_type.unsqueeze(-2) - (
                            evidence_type * (evidence_type + 1) / 2).unsqueeze(-1)
                m1 = m1 - evidence_type.unsqueeze(-1)
                m2 = evidence_type.unsqueeze(-1) + evidence_type.unsqueeze(-2) * 4 - (
                            evidence_type * (evidence_type + 1) / 2).unsqueeze(-2)
                m2 = m2 - evidence_type.unsqueeze(-2)
                relation_matrix = torch.where(evidence_type.unsqueeze(-1) <= evidence_type.unsqueeze(-2), m1, m2).long()
                # relation_matrix -= evidence_type.unsqueeze(-1)
                relation_matrix = torch.where(evidence_type.unsqueeze(-1) == evidence_type.unsqueeze(-2), 0,
                                              relation_matrix).long()

        embeddings = self.relation_embedding(relation_matrix)
        # embeddings = self.LayerNorm(embeddings)
        # embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttentionWithRelation(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.relation_encoding_method = config.relation_encoding_method
        self.share_relation_across_head = config.share_relation_across_head

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        relation_embedding_k,
        relation_embedding_v=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.relation_encoding_method in ["key", "key_value"]:
            if self.share_relation_across_head:
                relation_scores = torch.einsum("bhld,blrd->bhlr", query_layer, relation_embedding_k)
            else:
                relation_embedding_k = relation_embedding_k.view(relation_embedding_k.shape[0],
                                                                 relation_embedding_k.shape[1],
                                                                 relation_embedding_k.shape[2],
                                                                 self.num_attention_heads, -1).permute(0, 3, 1, 2, 4)
                relation_scores = torch.einsum("bhld,bhlrd->bhlr", query_layer, relation_embedding_k)
            attention_scores = attention_scores + relation_scores
        elif self.relation_encoding_method == "key_query":
            if self.share_relation_across_head:
                relation_scores_query = torch.einsum("bhld,blrd->bhlr", query_layer, relation_embedding_k)
                relation_scores_key = torch.einsum("bhrd,blrd->bhlr", key_layer, relation_embedding_k)
            else:
                relation_embedding_k = relation_embedding_k.view(relation_embedding_k.shape[0],
                                                                 relation_embedding_k.shape[1],
                                                                 relation_embedding_k.shape[2],
                                                                 self.num_attention_heads, -1).permute(0, 3, 1, 2, 4)
                relation_scores_query = torch.einsum("bhld,bhlrd->bhlr", query_layer, relation_embedding_k)
                relation_scores_key = torch.einsum("bhld,bhlrd->bhlr", key_layer, relation_embedding_k)
            attention_scores = attention_scores + relation_scores_query + relation_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        if self.relation_encoding_method == 'key_value':
            if self.share_relation_across_head:
                context_layer += torch.einsum("bhlr,brld->bhld", attention_probs, relation_embedding_v)
            else:
                relation_embedding_v = relation_embedding_v.view(relation_embedding_k.shape[0],
                                                                 relation_embedding_k.shape[1],
                                                                 relation_embedding_k.shape[2],
                                                                 self.num_attention_heads, -1).permute(0, 3, 1, 2, 4)
                context_layer += torch.einsum("bhlr,bhrld->bhld", attention_probs, relation_embedding_v)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttentionWithRelation(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        relation_embedding_k,
        relation_embedding_v=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            relation_embedding_k,
            relation_embedding_v,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertLayerWithRelation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        relation_embedding_k,
        relation_embedding_v=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            relation_embedding_k,
            relation_embedding_v,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output