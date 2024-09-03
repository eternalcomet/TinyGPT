import torch.nn.functional as F


def get_act_fn(act_name: str):
    if act_name == 'relu':
        return F.relu
    elif act_name == 'silu':
        return F.silu
    elif act_name == 'gelu':
        return F.gelu
    else:
        raise ValueError

