import torch.nn.functional as F

from .layernorm import layer_norm


def mlp(h, i, weights_dict):
    ln2_w = weights_dict[f'transformer.h.{i}.ln_2.weight']
    ln2_b = weights_dict[f'transformer.h.{i}.ln_2.bias']

    my_ln2 = layer_norm(h, ln2_w, ln2_b)
    my_fc = my_ln2 @ weights_dict[f'transformer.h.{i}.mlp.c_fc.weight'] + weights_dict[f'transformer.h.{i}.mlp.c_fc.bias']
    my_act = F.gelu(my_fc, approximate='tanh')
    my_proj = my_act @ weights_dict[f'transformer.h.{i}.mlp.c_proj.weight'] + weights_dict[f'transformer.h.{i}.mlp.c_proj.bias']

    return h + my_proj
