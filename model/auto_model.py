from transformers.models.auto.auto_factory import auto_class_factory
from transformers.models.auto.modeling_auto import MODEL_FOR_MULTIPLE_CHOICE_MAPPING
from transformers.models.auto.configuration_auto import OpenAIGPTConfig
from model.multi_choice import OpenAIGPTForMultipleChoice

mapping = MODEL_FOR_MULTIPLE_CHOICE_MAPPING
mapping[OpenAIGPTConfig] = OpenAIGPTForMultipleChoice

AutoModelForMultipleChoice = auto_class_factory(
    "AutoModelForMultipleChoice", mapping, head_doc="multiple choice"
)