from .attention import attention
from .mlp import mlp


def block(h, i, seq_len, weights_dict):
    h = attention(h, i, seq_len, weights_dict)
    h = mlp(h, i, weights_dict)
    return h
