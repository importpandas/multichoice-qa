
import logging

logger = logging.getLogger(__name__)

def statistic_of_optimizer_parameters(optimizer):
    tmp_sum = 0
    tmp_size = 0
    for g in optimizer.param_groups:
        for p in g['params']:
            tmp_sum += p.sum()
            tmp_size += p.numel()
    return tmp_size, tmp_sum

def statistic_of_model_parameters(model):
    tmp_sum = 0
    tmp_size = 0
    for p in model.parameters():
        tmp_sum += p.sum()
        tmp_size += p.numel()
    return tmp_size, tmp_sum

def iter_parameters_of_optimizer(optimizer):
    """
    Generator expression that iterates over the params owned by ``optimizer``.
    Args:
        optimizer: An optimizer.
    """
    for group in optimizer.param_groups:
        for p in group['params']:
            yield p
