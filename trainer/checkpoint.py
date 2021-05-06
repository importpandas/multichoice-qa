
import os
import timeit
import glob
import torch
from transformers.modeling_utils import PreTrainedModel
from transformers import WEIGHTS_NAME

def save_checkpoint(checkpoint_dir, model, tokenizer, model_type=None):
    model_to_save = model.module if hasattr(model, "module") else model
    #if model_type == "bert_transformer":
    #    torch.save(model_to_save, os.path.join(checkpoint_dir, WEIGHTS_NAME))
    #else:
    model_to_save.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

def find_all_checkpoints(checkpoint_dir):
    checkpoints = list(
        os.path.dirname(c)
        for c in sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint-*", WEIGHTS_NAME), recursive=True))
        )
    #logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
    return checkpoints
