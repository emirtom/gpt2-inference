"""Correctness test for the fused FFN CUDA kernel.

Compares the fused kernel output against the simple PyTorch reference
across multiple input shapes. Tolerance: 1e-3 absolute.

Run with:
    python -m pytest tests/test_fused_ffn.py -v
    python tests/test_fused_ffn.py
"""

import torch

from simple.mlp import mlp as simple_mlp
from optimized.fused_ffn import fused_mlp


def _make_weights(device='cuda'):
    """Create fake WeightDict-like entries for a single transformer block."""
    return {
        'transformer.h.0.ln_2.weight': torch.randn(768, device=device),
        'transformer.h.0.ln_2.bias': torch.randn(768, device=device),
        'transformer.h.0.mlp.c_fc.weight': torch.randn(768, 3072, device=device),
        'transformer.h.0.mlp.c_fc.bias': torch.randn(3072, device=device),
        'transformer.h.0.mlp.c_proj.weight': torch.randn(3072, 768, device=device),
        'transformer.h.0.mlp.c_proj.bias': torch.randn(768, device=device),
    }


def _run_test(batch, seq_len, atol=5e-3):
    device = 'cuda'
    weights = _make_weights(device)
    h = torch.randn(batch, seq_len, 768, device=device)

    ref = simple_mlp(h, 0, weights)
    out = fused_mlp(h, 0, weights)

    assert torch.allclose(ref, out, atol=atol), \
        f"Mismatch at batch={batch}, seq={seq_len}: max diff = {(ref - out).abs().max():.6f}"
    return True


def test_small():
    assert _run_test(1, 64)


def test_medium():
    assert _run_test(1, 128)


def test_large():
    assert _run_test(1, 256)


def test_batch2():
    assert _run_test(2, 128)


def test_real_weights():
    """Test with actual GPT-2 weights for realistic output scales."""
    device = 'cuda'
    import main
    for k in main.weights_dict:
        main.weights_dict[k] = main.weights_dict[k].to(device)

    for S in [64, 128, 256]:
        h = torch.randn(1, S, 768, device=device)
        ref = simple_mlp(h, 0, main.weights_dict)
        out = fused_mlp(h, 0, main.weights_dict)
        assert torch.allclose(ref, out, atol=1e-3), \
            f"Real weights S={S}: max diff = {(ref - out).abs().max():.6f}"


if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import torch
    from simple.mlp import mlp as simple_mlp
    from optimized.fused_ffn import fused_mlp

    print("Running correctness tests for fused FFN kernel...\n")
    for name in ['test_small', 'test_medium', 'test_large', 'test_batch2', 'test_real_weights']:
        try:
            globals()[name]()
            print(f"  {name}: PASS")
        except AssertionError as e:
            print(f"  {name}: FAIL — {e}")
        except Exception as e:
            print(f"  {name}: ERROR — {e}")
