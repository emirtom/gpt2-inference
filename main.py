import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from simple.layernorm import layer_norm as _layer_norm
from simple.embeddings import get_embeddings as _get_embeddings
from simple.mlp import mlp as _mlp
from simple.attention import attention as _attention
from simple.block import block as _block
from optimized.fused_ffn import fused_mlp as _fused_mlp


# ---- Wrappers that bind the global weights_dict (matching old API) ----

def get_embeddings(token_ids):
    return _get_embeddings(token_ids, weights_dict)


def layer_norm(x, weight, bias, eps=1e-5):
    return _layer_norm(x, weight, bias, eps)


def attention(h, i, seq_len):
    return _attention(h, i, seq_len, weights_dict)


def mlp(h, i):
    return _mlp(h, i, weights_dict)


def block(h, i, seq_len):
    return _block(h, i, seq_len, weights_dict)


def fused_mlp(h, i):
    return _fused_mlp(h, i, weights_dict)


# ---- Full forward passes ----

def forward_pass(token_ids):
    embedding = get_embeddings(token_ids)
    seq_len = token_ids.size(1)

    h = embedding
    for i in range(12):
        h = block(h, i, seq_len)

    h = layer_norm(h, weights_dict['transformer.ln_f.weight'], weights_dict['transformer.ln_f.bias'])
    logits = h @ weights_dict['lm_head.weight'].T
    return logits


def fused_forward_pass(token_ids):
    embedding = get_embeddings(token_ids)
    seq_len = token_ids.size(1)

    h = embedding
    for i in range(12):
        h = attention(h, i, seq_len)
        h = fused_mlp(h, i)

    h = layer_norm(h, weights_dict['transformer.ln_f.weight'], weights_dict['transformer.ln_f.bias'])
    logits = h @ weights_dict['lm_head.weight'].T
    return logits

# ---- Model loading ----

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
weights_dict = model.state_dict()


# ---- Generation ----

def multi_token_generate(text, max_new_tokens=1):
    token_ids = tokenizer.encode(text, return_tensors='pt')
    for _ in range(max_new_tokens):
        logits = forward_pass(token_ids)
        logits = logits[0, -1, :]
        token_id = torch.argmax(logits, dim=-1)
        token_ids = torch.cat([token_ids, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
    return tokenizer.decode(token_ids[0])


def hf_generate(text, max_new_tokens=1):
    token_ids = tokenizer.encode(text, return_tensors='pt')
    out = model.generate(token_ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0])


if __name__ == '__main__':
    text = "His name is Emir"
    token_count = 25

    model.eval()
    torch.manual_seed(42)
    model.config.pad_token_id = tokenizer.eos_token_id

    next_tokens = multi_token_generate(text, max_new_tokens=token_count)
    hf_next_tokens = hf_generate(text, max_new_tokens=token_count)

    print(f"Your token: {next_tokens}")
    print(f"HF token:   {hf_next_tokens}")
    print(f"Match: {next_tokens == hf_next_tokens}")
