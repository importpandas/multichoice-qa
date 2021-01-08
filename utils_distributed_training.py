# Utils for distributed training in Slurm environment

import re
import os
import torch
import functools

def get_oneNode_addr():
    try:
        nodelist = os.environ['SLURM_STEP_NODELIST']
    except:
        nodelist = os.environ['SLURM_JOB_NODELIST']
    nodelist = nodelist.strip().split(',')[0]
    text = re.split('[-\[\]]',nodelist)
    if('' in text):
        text.remove('')
    return text[0] + '-' + text[1] + '-' + text[2]

def dist_init(host_addr, rank, local_rank, world_size, port=23456):
    host_addr_full = 'tcp://' + host_addr + ':' + str(port)
    try:
        torch.distributed.init_process_group("nccl", init_method=host_addr_full,
                                              rank=rank, world_size=world_size)
    except:
        xprint(f"host addr {host_addr_full}")
        print(f"process id {int(os.environ['SLURM_PROCID'])}")
        exit("distributed training initialization failed")
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    assert torch.distributed.is_initialized()

def stats(output, label):
    if output.size(0) != 0:
        maxval, prediction = output.max(len(output.size()) - 1)
        num_matches = torch.sum(torch.eq(label, prediction)).item()
        num_utts = label.size(0)
    else:
        num_matches, num_utts = 0, 0
    return num_matches, num_utts

def gather_output(inputs, world_size):
    group = torch.distributed.new_group(list(range(world_size)))
    if isinstance(inputs, torch.tensor):
        gather_list = [torch.zeros_like(inputs) for i in range(world_size)]
        torch.distributed.all_gather(gather_list, inputs, group=group)
        return torch.cat(gather_list)
    elif isinstance(inputs, (tuple, list)):
        outputs = []
        for input_tensor in inputs:
            assert isinstance(input_tensor, torch.tensor)
            gather_list = [torch.zeros_like(input_tensor) for i in range(world_size)]
            torch.distributed.all_gather(gather_list, input_tensor, group=group)
            output_tensor = torch.cat(gather_list)
            outputs.append(output_tensor)
        return outputs
    else:
        xprint(f'distributed gather inputs type error')
        return None

#modified print function name
def modified_print(*args, local_rank=-1 ,**kargs):
    if local_rank == 0 or local_rank==-1:
        print(*args,flush=True,**kargs)

#print with flush=True and only print with Slurm_procid=0 or KeyError
try:
    xprint = functools.partial(modified_print, local_rank=int(os.environ['SLURM_PROCID']))
except KeyError:
    xprint = functools.partial(modified_print, local_rank=-1)

