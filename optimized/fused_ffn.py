import os

import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = load(
            name='fused_ffn',
            sources=[
                os.path.join(_HERE, 'fused_ffn_kernel.cu'),
                os.path.join(_HERE, 'fused_ffn_host.cpp'),
            ],
            extra_cuda_cflags=['-O3', '--use_fast_math', '-allow-unsupported-compiler'],
            verbose=True,
        )
    return _module


def fused_mlp(h, i, weights_dict):
    """Fused FFN: LayerNorm -> c_fc -> GELU -> c_proj -> +residual.

    Same interface as simple.mlp.mlp().
    """
    device = h.device

    ln2_w = weights_dict[f'transformer.h.{i}.ln_2.weight'].to(device=device, dtype=torch.float32)
    ln2_b = weights_dict[f'transformer.h.{i}.ln_2.bias'].to(device=device, dtype=torch.float32)
    c_fc_w = weights_dict[f'transformer.h.{i}.mlp.c_fc.weight'].to(device=device, dtype=torch.float32)
    c_fc_b = weights_dict[f'transformer.h.{i}.mlp.c_fc.bias'].to(device=device, dtype=torch.float32)
    c_proj_w = weights_dict[f'transformer.h.{i}.mlp.c_proj.weight'].to(device=device, dtype=torch.float32)
    c_proj_b = weights_dict[f'transformer.h.{i}.mlp.c_proj.bias'].to(device=device, dtype=torch.float32)
    h = h.to(dtype=torch.float32).contiguous()

    mod = _get_module()
    return mod.forward(h, ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_w, c_proj_b)
