import torch


def layer_norm(x, weight, bias, eps=1e-5):
    mean = torch.mean(x, dim=-1, keepdim=True)
    variance = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    normalized = (x - mean) / torch.sqrt(variance + eps)
    return normalized * weight + bias
