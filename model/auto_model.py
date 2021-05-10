from transformers.models.auto.auto_factory import auto_class_factory
from transformers.models.auto.modeling_auto import MODEL_FOR_MULTIPLE_CHOICE_MAPPING
from transformers.models.auto.configuration_auto import OpenAIGPTConfig

AutoModelForMultipleChoice = auto_class_factory(
    "AutoModelForMultipleChoice", MODEL_FOR_MULTIPLE_CHOICE_MAPPING, head_doc="multiple choice"
)