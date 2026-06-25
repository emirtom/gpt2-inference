import torch


def get_embeddings(token_ids, weights_dict):
    wte = weights_dict['transformer.wte.weight']
    wpe = weights_dict['transformer.wpe.weight']
    positions = torch.arange(token_ids.size(1), device=token_ids.device)
    return wte[token_ids] + wpe[positions]
