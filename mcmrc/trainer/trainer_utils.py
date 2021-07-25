import torch
import numpy as np
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def _grad_sync(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            # Init optimizer state
            torch.distributed.all_reduce(p.grad)


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    example_ids: Optional[Tuple[str]]
    metrics: Optional[Dict[str, float]]


def compute_mc_metrics(eval_predictions, mask=None):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    if mask is None:
        acc = (preds == label_ids).astype(np.float32).mean().item()
    else:
        acc = ((preds == label_ids) & np.array(mask)).sum() / np.array(mask).sum()
    return {"accuracy": acc}
