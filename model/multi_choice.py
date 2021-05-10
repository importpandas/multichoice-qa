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
    The Original OpenAI GPT Model transformer with a sequence classification head on top (linear layer).
    :class:`~transformers.OpenAIGPTForSequenceClassification` uses the last token in order to do the classification, as
    other causal models (e.g. GPT-2) do. Since it does classification on the last token, it requires to know the
    position of the last token. If a :obj:`pad_token_id` is defined in the configuration, it finds the last token that
    is not a padding token in each row. If no :obj:`pad_token_id` is defined, it simply takes the last value in each
    row of the batch. Since it cannot guess the padding tokens when :obj:`inputs_embeds` are passed instead of
    :obj:`input_ids`, it does the same (take the last value in each row of the batch).
    """,
    OPENAI_GPT_START_DOCSTRING,
)
class OpenAIGPTForMultiChoice(OpenAIGPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = OpenAIGPTModel(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        self.init_weights()

    @add_start_docstrings_to_model_forward(OPENAI_GPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
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

        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    f"unexpected if using padding tokens in conjuction with `inputs_embeds.`"
                )

        pooled_logits = logits[range(batch_size), sequence_lengths]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(pooled_logits.view(-1), labels.to(self.dtype).view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
