#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEncoder, BertLayer
from transformers import PretrainedConfig, AutoConfig, AutoModelForQuestionAnswering
from transformers.models.albert.modeling_albert import (
    AlbertPreTrainedModel,
    AlbertModel,
)
import re
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertModel,
)
from transformers.modeling_outputs import MultipleChoiceModelOutput


class EveEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_eve_layers)])
        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


def convert_mask_to_reality(mask, dtype=torch.float):
    mask = mask.to(dtype=dtype)
    mask = (1.0 - mask) * -10000.0
    return mask


class EveForMultipleChoice(BertPreTrainedModel):

    base_model_prefix = "eve"

    def __init__(self, config):
        super().__init__(config)

        if config.model_type == 'bert':
            self.bert = BertModel(config)
            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
        elif config.model_type == 'albert':
            self.albert = AlbertModel(config)
            classifier_dropout = config.classifier_dropout_prob

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.positive_head_num = sum([int(i) for i in re.findall("p(\d+)", config.eve_att_head)])
        self.negative_head_num = sum([int(i) for i in re.findall("n(\d+)", config.eve_att_head)])
        self.full_head_num = sum([int(i) for i in re.findall("f(\d+)", config.eve_att_head)])
        self.competitive_head_num = sum([int(i) for i in re.findall("c(\d+)", config.eve_att_head)])
        assert self.positive_head_num + self.negative_head_num + self.full_head_num + self.competitive_head_num == config.eve_head_num

        config.num_attention_heads = config.eve_head_num
        if config.num_eve_layers > 0:
            self.eve_layer = EveEncoder(config)
        else:
            self.eve_layer = None

        if config.pooling_type == 'linear_pooling':
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.pooler_activation = nn.Tanh()
        else:
            self.pooler = None
            self.pooler_activation = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        positive_mask=None,
        negative_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        seq_len = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        positive_mask = positive_mask.view(-1, positive_mask.size(-1)) if positive_mask is not None else None
        negative_mask = negative_mask.view(-1, negative_mask.size(-1)) if negative_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        plm_encoder = self.bert if self.config.model_type == 'bert' else self.albert
        outputs = plm_encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        default_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        default_attention_mask = convert_mask_to_reality(default_attention_mask)
        extended_default_mask = default_attention_mask.expand(-1, self.full_head_num, seq_len, -1)
        eve_attention_mask = extended_default_mask

        def _get_extended_evidence_mask(evidence_mask, default_mask, head_num=1):
            reverse_evidence_mask = 1 - evidence_mask
            extended_evidence_mask = evidence_mask.unsqueeze(-1).bmm(evidence_mask.unsqueeze(-2)) \
                                     + reverse_evidence_mask.unsqueeze(-1).bmm(reverse_evidence_mask.unsqueeze(-2))
            extended_evidence_mask = convert_mask_to_reality(extended_evidence_mask)
            extended_evidence_mask = extended_evidence_mask.unsqueeze(1).expand(-1, head_num, -1, -1)
            return extended_evidence_mask + default_mask

        if positive_mask is not None and self.positive_head_num > 0:
            extended_positive_mask = _get_extended_evidence_mask(positive_mask, default_attention_mask, self.positive_head_num)
            eve_attention_mask = torch.cat((eve_attention_mask, extended_positive_mask), dim=1)

        if negative_mask is not None and self.negative_head_num > 0:
            extended_negative_mask = _get_extended_evidence_mask(negative_mask, default_attention_mask, self.negative_head_num)
            eve_attention_mask = torch.cat((eve_attention_mask, extended_negative_mask), dim=1)

        if self.competitive_head_num > 0:
            competitive_mask = torch.logical_or(positive_mask, negative_mask).float()
            extended_competitive_mask = _get_extended_evidence_mask(competitive_mask, default_attention_mask, self.negative_head_num)
            eve_attention_mask = torch.cat((eve_attention_mask, extended_competitive_mask), dim=1)

        if self.eve_layer is not None:
            eve_outputs = self.eve_layer(sequence_output, attention_mask=eve_attention_mask, head_mask=head_mask)[0]
        else:
            eve_outputs = sequence_output

        if self.config.pooling_type == 'linear_pooling':
            pooled_output = self.pooler_activation(
                self.pooler(eve_outputs[:, 0])) if self.pooler is not None else None
        elif self.config.pooling_type == 'sequence_mean':
            pooled_output = torch.mean(eve_outputs, 1)
        elif self.config.pooling_type == 'cls_force':
            pooled_output = eve_outputs[:, 0]
        else:
            raise ValueError("Config.pooling type must be linear_pooling or sequence_mean")

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

