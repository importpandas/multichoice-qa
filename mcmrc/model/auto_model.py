from transformers.models.auto.modeling_auto import MODEL_FOR_MULTIPLE_CHOICE_MAPPING
from transformers.models.auto.auto_factory import _BaseAutoModelClass, \
    _LazyAutoMapping, auto_class_update, model_type_to_module_name, getattribute_from_module
from transformers.models.auto.modeling_auto import CONFIG_MAPPING_NAMES, MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES

from transformers.models.albert.configuration_albert import AlbertConfig
from transformers.models.bert.configuration_bert import BertConfig

import mcmrc.model.multi_choice as mc_models

import importlib


class _LazyAutoMappingModified(_LazyAutoMapping):

    def _load_attr_from_module(self, model_type, attr):
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
        try:
            return getattribute_from_module(mc_models, attr)
        except AttributeError:
            return getattribute_from_module(self._modules[module_name], attr)


MODEL_FOR_MULTIPLE_CHOICE_MAPPING = _LazyAutoMappingModified(CONFIG_MAPPING_NAMES,
                                                             MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES)


class AutoModelForMultipleChoice(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MULTIPLE_CHOICE_MAPPING


AutoModelForMultipleChoice = auto_class_update(AutoModelForMultipleChoice, head_doc="multiple choice")
