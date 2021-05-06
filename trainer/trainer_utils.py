import torch
import numpy as np


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def _grad_sync(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            # Init optimizer state
            torch.distributed.all_reduce(p.grad)