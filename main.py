import math

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

def get_embeddings(token_ids, weights_dict):
    wte = weights_dict['transformer.wte.weight']
    wpe = weights_dict['transformer.wpe.weight']
    positions = torch.arange(token_ids.size(1))
    
    embedding = wte[token_ids] + wpe[positions]
    
    
    return embedding

def layer_norm(x, weight, bias, eps=1e-5):
    mean = torch.mean(x, dim=-1, keepdim=True)
    variance = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    normalized = (x - mean) / torch.sqrt(variance + eps)
    y = normalized * weight + bias
    
    
    return y


def block(h, i, seq_len):
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
    mask = torch.tril(torch.ones(scores.size(-2), scores.size(-1)))
    mask = mask.masked_fill(mask == 0, float('-inf'))
    weights = torch.softmax(scores + mask, dim=-1)

    out = weights @ V

    out = out.transpose(1, 2).reshape(1, seq_len, 768)


    cproj_w = weights_dict[f'transformer.h.{i}.attn.c_proj.weight']
    cproj_b = weights_dict[f'transformer.h.{i}.attn.c_proj.bias']

    attn_out = out @ cproj_w + cproj_b


    h = h + attn_out


    ln2_w = weights_dict[f'transformer.h.{i}.ln_2.weight']
    ln2_b = weights_dict[f'transformer.h.{i}.ln_2.bias']

    my_ln2  = layer_norm(h, ln2_w, ln2_b)
    my_fc   = my_ln2 @ weights_dict[f'transformer.h.{i}.mlp.c_fc.weight'] + weights_dict[f'transformer.h.{i}.mlp.c_fc.bias']
    my_act  = torch.nn.functional.gelu(my_fc, approximate='tanh')
    my_proj = my_act @ weights_dict[f'transformer.h.{i}.mlp.c_proj.weight'] + weights_dict[f'transformer.h.{i}.mlp.c_proj.bias']
    
    return h + my_proj



tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
weights_dict = model.state_dict()

text = "Hello, how are you"

def multi_token_generate(text, max_new_tokens=1):
    token_ids = tokenizer.encode(text, return_tensors='pt')
    for _ in range(max_new_tokens):
        embedding = get_embeddings(token_ids, weights_dict)
        seq_len = token_ids.size(1)

        h = embedding
        for i in range(12):
            h = block(h, i, seq_len)

        h = layer_norm(h, weights_dict['transformer.ln_f.weight'], weights_dict['transformer.ln_f.bias'])
        logits = h @ weights_dict['lm_head.weight'].T
        logits = logits[0, -1, :]
        token_id = torch.argmax(logits, dim=-1)
        token_ids = torch.cat([token_ids, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
    
    return tokenizer.decode(token_ids[0])

def hf_generate(text, max_new_tokens=1):
    token_ids = tokenizer.encode(text, return_tensors='pt')
    out = model.generate(token_ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0])


token_count = 25

next_5 = multi_token_generate(text, max_new_tokens=token_count)

model.eval()
torch.manual_seed(42)

# Tie pad token to suppress the warning
model.config.pad_token_id = tokenizer.eos_token_id


hf_next_5 = hf_generate(text, max_new_tokens=token_count)



print(f"Your token: {next_5}")
print(f"HF token:   {hf_next_5}")
print(f"Match: {next_5 == hf_next_5}")


