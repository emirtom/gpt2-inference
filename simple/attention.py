import math

import torch

from .layernorm import layer_norm


def attention(h, i, seq_len, weights_dict):
    ln_w = weights_dict[f'transformer.h.{i}.ln_1.weight']
    ln_b = weights_dict[f'transformer.h.{i}.ln_1.bias']
    my_ln = layer_norm(h, ln_w, ln_b)

    c_attn_w = weights_dict[f'transformer.h.{i}.attn.c_attn.weight']
    c_attn_b = weights_dict[f'transformer.h.{i}.attn.c_attn.bias']
    QKV = my_ln @ c_attn_w + c_attn_b
    Q, K, V = QKV.chunk(3, dim=-1)

    Q = Q.view(Q.size(0), Q.size(1), 12, 64).transpose(1, 2)
    K = K.view(K.size(0), K.size(1), 12, 64).transpose(1, 2)
    V = V.view(V.size(0), V.size(1), 12, 64).transpose(1, 2)

    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(64)
    mask = torch.tril(torch.ones(scores.size(-2), scores.size(-1), device=scores.device))
    mask = mask.masked_fill(mask == 0, float('-inf'))
    weights = torch.softmax(scores + mask, dim=-1)

    out = weights @ V
    out = out.transpose(1, 2).reshape(h.size(0), seq_len, 768)

    cproj_w = weights_dict[f'transformer.h.{i}.attn.c_proj.weight']
    cproj_b = weights_dict[f'transformer.h.{i}.attn.c_proj.bias']
    attn_out = out @ cproj_w + cproj_b

    return h + attn_out
