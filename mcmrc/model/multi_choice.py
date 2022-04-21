"""PyTorch OpenAI GPT model."""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers.modeling_outputs import MultipleChoiceModelOutput

from transformers.utils import logging
from transformers.models.albert.modeling_albert import (
    AlbertPreTrainedModel,
    AlbertModel,
    ALBERT_START_DOCSTRING,
    ALBERT_INPUTS_DOCSTRING
)
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertModel,
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING,
)

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_utils import SequenceSummary
logger = logging.get_logger(__name__)

ALBERT_CHECKPOINT_FOR_DOC = "albert-base-v2"
ALBERT_CONFIG_FOR_DOC = "AlbertConfig"
ALBERT_TOKENIZER_FOR_DOC = "AlbertTokenizer"

BERT_CHECKPOINT_FOR_DOC = "bert-base-uncased"
BERT_CONFIG_FOR_DOC = "BertConfig"
BERT_TOKENIZER_FOR_DOC = "BertTokenizer"


@add_start_docstrings(
    """
    Albert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ALBERT_START_DOCSTRING,
)
class AlbertForMultipleChoice(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=ALBERT_TOKENIZER_FOR_DOC,
        checkpoint=ALBERT_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=ALBERT_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        competitive_scores=None,
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
            num_choices-1]`` where `num_choices` is the size of the second dimension of the input tensors. (see
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        outputs = self.albert(
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

        if self.config.pooling_type == 'linear_pooling':
            pooled_output = outputs[1]
        elif self.config.pooling_type == 'sequence_mean':
            sequence_output = outputs[0]
            pooled_output = torch.mean(sequence_output, 1)
        else:
            raise ValueError("Config.pooling type must be linear_pooling or sequence_mean")

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            if competitive_scores is not None:
                loss_fct = CrossEntropyLoss(reduction='none')
                weights = torch.softmax(competitive_scores / self.config.temperature, dim=-1)
                loss = loss_fct(reshaped_logits, labels)
                loss = torch.sum(loss * weights)
            else:
                if self.config.loss_function == "cross_entropy":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(reshaped_logits, labels)
                elif self.config.loss_function == "bce_with_logit":
                    loss_fct = BCEWithLogitsLoss()
                    bce_labels = torch.zeros_like(reshaped_logits, device=reshaped_logits.device)
                    bce_labels.scatter_(-1, labels.unsqueeze(-1), 1)
                    reshaped_bce_labels = bce_labels.view(-1, 1)
                    loss = loss_fct(logits, reshaped_bce_labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    BERT_START_DOCSTRING,
)
class BertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=BERT_TOKENIZER_FOR_DOC,
        checkpoint=BERT_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=BERT_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        competitive_scores=None,
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

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(
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
        if self.config.pooling_type == 'linear_pooling':
            pooled_output = outputs[1]
        elif self.config.pooling_type == 'sequence_mean':
            sequence_output = outputs[0]
            pooled_output = torch.mean(sequence_output, 1)
        else:
            raise ValueError("Config.pooling type must be linear_pooling or sequence_mean")

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            if competitive_scores is not None:
                loss_fct = CrossEntropyLoss(reduction='none')
                weights = torch.softmax(competitive_scores / self.config.temperature, dim=-1)
                loss = loss_fct(reshaped_logits, labels)
                loss = torch.sum(loss * weights)
            else:
                if self.config.loss_function == "cross_entropy":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(reshaped_logits, labels)
                elif self.config.loss_function == "bce_with_logit":
                    loss_fct = BCEWithLogitsLoss()
                    bce_labels = torch.zeros_like(reshaped_logits, device=reshaped_logits.device)
                    bce_labels.scatter_(-1, labels.unsqueeze(-1), 1)
                    reshaped_bce_labels = bce_labels.view(-1, 1)
                    loss = loss_fct(logits, reshaped_bce_labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



