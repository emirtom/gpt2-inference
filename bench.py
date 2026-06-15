import time
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Import your functions from main
import main

token_count = 25
warmup_iters = 3
bench_iters = 10

main.model.eval()
torch.manual_seed(42)
main.model.config.pad_token_id = main.tokenizer.eos_token_id

# --- Warmup ---
for _ in range(warmup_iters):
    main.multi_token_generate(main.text, max_new_tokens=token_count)
for _ in range(warmup_iters):
    main.hf_generate(main.text, max_new_tokens=token_count)

# --- Benchmark yours ---
t_start = time.perf_counter()
for _ in range(bench_iters):
    my_result = main.multi_token_generate(main.text, max_new_tokens=token_count)
t_my = (time.perf_counter() - t_start) / bench_iters

# --- Benchmark HF ---
t_start = time.perf_counter()
for _ in range(bench_iters):
    hf_result = main.hf_generate(main.text, max_new_tokens=token_count)
t_hf = (time.perf_counter() - t_start) / bench_iters

print(f"Your token: {my_result}")
print(f"HF token:   {hf_result}")
print(f"Match: {my_result == hf_result}")
print(f"\n--- Timing (avg of {bench_iters} runs, {token_count} new tokens) ---")
print(f"Your forward: {t_my*1000:.1f} ms")
print(f"HF generate:  {t_hf*1000:.1f} ms")
print(f"Slowdown:     {t_my/t_hf:.1f}x")
