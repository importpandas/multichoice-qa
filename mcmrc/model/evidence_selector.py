import torch
import torch.nn as nn
from torch.nn import KLDivLoss, MSELoss
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
import logging
logger = logging.getLogger(__name__)

_CHECKPOINT_FOR_DOC = "albert-base-v2"
_CONFIG_FOR_DOC = "AlbertConfig"
_TOKENIZER_FOR_DOC = "AlbertTokenizer"


from transformers import AlbertPreTrainedModel, AlbertModel
from transformers.models.albert.modeling_albert import ALBERT_START_DOCSTRING, ALBERT_INPUTS_DOCSTRING

@add_start_docstrings(
    """
    Albert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ALBERT_START_DOCSTRING,
)
class AlbertForEvidenceSelection(AlbertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.albert = AlbertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, 1)
        self.pooling_type = config.pooling_type

        self.init_weights()

    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        sent_bounds=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        device = sequence_output.device
        pos_matrix = torch.arange(sequence_output.size()[1], device=device).view(1, 1, -1)
        if_in_sent = torch.logical_and(sent_bounds[:, :, 1].unsqueeze(-1) <= pos_matrix,
                                           pos_matrix <= sent_bounds[:, :, 2].unsqueeze(-1))

        if self.pooling_type == 'average':
            pooling_matrix = torch.where(if_in_sent, torch.tensor((1), device=device), torch.tensor((0), device=device)).float()
            sent_len = torch.sum(pooling_matrix, 2).unsqueeze(2)
            sent_len[sent_len==0] = 1
            pooling_matrix = pooling_matrix / sent_len
            sentence_hiddens = torch.bmm(sequence_output.transpose(-1, -2), pooling_matrix.transpose(-1, -2)).transpose(-1, -2)
        elif self.pooling_type == 'max':
            pooling_matrix = torch.where(if_in_sent.unsqueeze(-1),  sequence_output.unsqueeze(1), torch.tensor((0.0), device=device)).float()
            sentence_hiddens = torch.max(pooling_matrix, dim=2)[0]
        logits = self.output_layer(sentence_hiddens).squeeze(-1)

        mask = torch.where(sent_bounds[:, :, 0] >= 0, torch.tensor(0.0, device=device), torch.tensor((-10000.0), device=device))
        logits += mask

        loss = None
        if labels is not None:
            loss_fct = KLDivLoss()
            # Only keep active parts of the loss
            loss = loss_fct(F.log_softmax(logits, dim=-1), F.softmax(labels, dim=-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )