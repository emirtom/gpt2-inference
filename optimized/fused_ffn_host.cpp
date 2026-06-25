#include <torch/extension.h>

void fused_ffn_cuda(
    torch::Tensor out,
    const torch::Tensor x,
    const torch::Tensor ln_w,
    const torch::Tensor ln_b,
    const torch::Tensor w1,
    const torch::Tensor b1,
    const torch::Tensor w2,
    const torch::Tensor b2,
    int M);

torch::Tensor fused_ffn_forward(
    torch::Tensor x,
    torch::Tensor ln_w,
    torch::Tensor ln_b,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2)
{
    TORCH_CHECK(x.dim() == 3, "x must be 3D (B, S, D)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    int B = x.size(0);
    int S = x.size(1);
    int M = B * S;

    auto x_flat = x.reshape({M, 768}).contiguous();
    auto out_flat = torch::empty({M, 768}, x.options());

    fused_ffn_cuda(out_flat, x_flat, ln_w, ln_b, w1, b1, w2, b2, M);

    return out_flat.reshape({B, S, 768});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_ffn_forward, "Fused FFN (LN + FC + GELU + FC + residual)");
}
