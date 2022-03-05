import torch
import numpy as np
from sklearn.metrics import f1_score
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
from collections import OrderedDict

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


def compute_mc_metrics(eval_predictions, mask=None, all_example_ids=None):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    if mask is None:
        acc = ((preds == label_ids).astype(np.float32).mean().item()) * 100
    else:
        acc = (((preds == label_ids) & np.array(mask)).sum() / np.array(mask).sum()) * 100
    if all_example_ids is not None:
        high_set_mask = [1 if 'high' in example_id else 0 for example_id in all_example_ids]
        middle_set_mask =[1 if 'middle' in example_id else 0 for example_id in all_example_ids]
        if sum(high_set_mask) > 0:
            assert sum(high_set_mask) + sum(middle_set_mask) == len(all_example_ids)
            high_set_acc = (((preds == label_ids) & np.array(high_set_mask)).sum() / np.array(high_set_mask).sum()) * 100
            middle_set_acc = (((preds == label_ids) & np.array(middle_set_mask)).sum() / np.array(middle_set_mask).sum()) * 100
            return OrderedDict([("accuracy", acc), ('high_accuracy', high_set_acc), ('middle_accuracy', middle_set_acc)])
    return OrderedDict([("accuracy", acc)])


def compute_classification_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    f1 = f1_score(label_ids, preds)
    acc = (preds == label_ids).astype(np.float32).mean().item()
    return {"accuracy": acc, "f1": f1}


def compute_verifier_metrics(answer_probs, labels, verifier_probs, threshold=-1):
    preds_list = []
    label_list = []
    for eid in answer_probs.keys():
        preds_list.append(answer_probs[eid])
        label_list.append(labels[eid])
    orig_preds = np.argmax(preds_list, axis=1)
    num_acc = (orig_preds == np.array(label_list)).astype(np.float32).sum().item()
    cur_score = num_acc
    best_score = cur_score
    best_thresh = 1.0
    score_with_thresh = cur_score
    eid_list = sorted(verifier_probs, key=lambda k: verifier_probs[k], reverse=True)
    for i, eid in enumerate(eid_list):
        if np.argsort(answer_probs[eid])[-1] == labels[eid]:
            diff = -1
        elif np.argsort(answer_probs[eid])[-2] == labels[eid]:
            diff = 1
        else:
            diff = 0
        cur_score += diff
        if 0 <= threshold < verifier_probs[eid]:
            score_with_thresh += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = verifier_probs[eid]

    acc = num_acc / len(label_list)
    best_acc = best_score / len(label_list)
    metrics = {'acc': acc, 'best_acc': best_acc, 'best_thresh': best_thresh}
    if threshold >= 0:
        metrics[f'acc_with_thresh_{threshold}'] = score_with_thresh
    return metrics
