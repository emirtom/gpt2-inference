#!/usr/bin/env python3
import argparse
import datetime
import json
import time

import torch
import main


def build_inputs(component, batch_size, seq_len, device):
    if component in ('generate', 'forward', 'embed'):
        return torch.randint(0, 50257, (batch_size, seq_len), device=device)
    h = torch.randn(batch_size, seq_len, 768, device=device)
    if component == 'layernorm':
        w = main.weights_dict['transformer.ln_f.weight']
        b = main.weights_dict['transformer.ln_f.bias']
        return h, w.to(device), b.to(device)
    return h


def measure_latency(fn, warmup, bench_iters, device):
    for _ in range(warmup):
        fn()
    if device == 'cuda':
        torch.cuda.synchronize()
        times = []
        for _ in range(bench_iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    else:
        times = []
        for _ in range(bench_iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)
    mu = sum(times) / len(times)
    sd = (sum((t - mu) ** 2 for t in times) / len(times)) ** 0.5
    return mu, sd


def measure_memory(fn, device):
    if device != 'cuda':
        return 0.0
    torch.cuda.reset_peak_memory_stats()
    fn()
    return torch.cuda.max_memory_allocated() / 1e6


def dispatch(component, inputs, device, variant='simple'):
    if component == 'embed':
        return main.get_embeddings(inputs)
    if component == 'layernorm':
        return main.layer_norm(*inputs)
    if component == 'attention':
        return main.attention(inputs, 0, inputs.size(1))
    if component == 'mlp':
        if variant == 'fused':
            return main.fused_mlp(inputs, 0)
        return main.mlp(inputs, 0)
    if component == 'block':
        return main.block(inputs, 0, inputs.size(1))
    if component == 'forward':
        if variant == 'fused':
            return main.fused_forward_pass(inputs)
        return main.forward_pass(inputs)
    raise ValueError(f'Unknown component: {component}')


def run_benchmark(args):
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    main.model.eval()
    main.model.config.pad_token_id = main.tokenizer.eos_token_id

    for k, v in main.weights_dict.items():
        main.weights_dict[k] = v.to(device)

    with torch.no_grad():
        if args.component == 'generate':
            prompt = build_inputs('generate', args.batch_size, args.seq_len, device)
            def fn():
                ids = prompt.clone()
                for _ in range(args.max_new_tokens):
                    logits = main.forward_pass(ids)
                    nxt = logits[:, -1, :].argmax(dim=-1)
                    ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
        else:
            inputs = build_inputs(args.component, args.batch_size, args.seq_len, device)
            def fn():
                dispatch(args.component, inputs, device, args.variant)

        need_latency = args.mode in ('latency', 'throughput', 'all')
        need_memory = args.mode in ('memory', 'all')

        mu = sd = mem = None
        if need_latency:
            mu, sd = measure_latency(fn, args.warmup_iters, args.bench_iters, device)
            mu, sd = round(mu, 3), round(sd, 3)
        if need_memory:
            mem = round(measure_memory(fn, device), 2)

    return {'latency_ms': mu, 'latency_std_ms': sd, 'gpu_mem_mb': mem}, device


def print_summary(args, r):
    print(f"component={args.component}  variant={args.variant}  mode={args.mode}  "
          f"device={'cuda' if torch.cuda.is_available() else 'cpu'}  "
          f"batch={args.batch_size}  seq={args.seq_len}")
    if r['latency_ms'] is not None:
        print(f"  mean: {r['latency_ms']:.3f} ms  \u00b1 {r['latency_std_ms']:.3f} ms")
        if args.component in ('forward', 'generate'):
            tok = args.batch_size * (args.max_new_tokens if args.component == 'generate' else args.seq_len)
            print(f"  throughput: {tok / (r['latency_ms'] / 1000):.1f} tok/s")
    if r['gpu_mem_mb'] is not None:
        print(f"  peak GPU mem: {r['gpu_mem_mb']:.1f} MB")


def log_results(args, r, device):
    record = {
        'timestamp': datetime.datetime.now().isoformat(),
        'component': args.component, 'variant': args.variant, 'mode': args.mode,
        'batch_size': args.batch_size, 'seq_len': args.seq_len,
        'max_new_tokens': args.max_new_tokens,
        'warmup_iters': args.warmup_iters, 'bench_iters': args.bench_iters,
    } | r
    with open(args.output, 'a') as f:
        f.write(json.dumps(record) + '\n')
    print(f"  logged to {args.output}")


def entry():
    p = argparse.ArgumentParser(description='GPT-2 inference benchmark')
    p.add_argument('-c', '--component', required=True,
                   choices=['embed','layernorm','attention','mlp','block','forward','generate'])
    p.add_argument('-v', '--variant', default='simple',
                   choices=['simple', 'fused'],
                   help='Which implementation to benchmark')
    p.add_argument('-m', '--mode', default='all',
                   choices=['latency','throughput','memory','all'])
    p.add_argument('-b', '--batch-size', type=int, default=1)
    p.add_argument('-s', '--seq-len', type=int, default=128)
    p.add_argument('-n', '--max-new-tokens', type=int, default=25)
    p.add_argument('-w', '--warmup-iters', type=int, default=5)
    p.add_argument('-i', '--bench-iters', type=int, default=20)
    p.add_argument('-d', '--device', choices=['cpu','cuda'])
    p.add_argument('-o', '--output', default='bench_results.jsonl')
    args = p.parse_args()

    r, device = run_benchmark(args)
    print_summary(args, r)
    log_results(args, r, device)


if __name__ == '__main__':
    entry()
