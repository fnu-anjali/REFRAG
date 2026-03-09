#!/usr/bin/env python3
"""
Baseline RAG: no chunking, no compression — raw passage text → decoder.
Uses the same input JSON format and produces the same output JSON format as cmd_generate.
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers.cache_utils import Cache, DynamicCache
except Exception:
    Cache = None
    DynamicCache = None


def now_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def parse_original(context_str):
    if isinstance(context_str, dict):
        return context_str.get("Original", "")
    if "Original:" in context_str:
        original = context_str.split("Original:")[1]
        if "\nEntites:" in original:
            original = original.split("\nEntites:")[0]
        return original.strip()
    return context_str.strip()


def _ensure_cache(past_key_values):
    if Cache is None or DynamicCache is None:
        return past_key_values
    if past_key_values is None:
        return DynamicCache()
    if isinstance(past_key_values, Cache):
        return past_key_values
    return DynamicCache.from_legacy_cache(past_key_values)


def baseline_generate(
    tokenizer,
    model,
    question: str,
    passages: list,
    max_ctx_tokens: int = 2048,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    device=None,
) -> dict:
    device = device or now_device()

    # Build raw prompt — no chunking, just concatenate everything
    prompt = (
        f"You are a helpful assistant. Answer the question using only the provided context.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n"
        + "\n".join(passages)
        + "\n\nAnswer:"
    )

    inputs = tokenizer(
        prompt,
        truncation=True,
        max_length=max_ctx_tokens,
        return_tensors="pt",
    ).to(device)

    input_ids = inputs.input_ids
    prompt_len = input_ids.shape[1]

    # Prefill
    cache = DynamicCache() if DynamicCache is not None else None
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True, past_key_values=cache)
    past_key_values = _ensure_cache(out.past_key_values)
    ttft = time.time() - t0

    eos_id = tokenizer.eos_token_id
    generated = []
    ttit_list = []
    last = torch.tensor([[eos_id]], device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            t1 = time.time()
            out = model(input_ids=last, use_cache=True, past_key_values=past_key_values)
            ttit_list.append(time.time() - t1)

            logits = out.logits[:, -1, :]
            past_key_values = _ensure_cache(out.past_key_values)

            if temperature > 0.0:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = cumulative_probs > top_p
                    cutoff[..., 1:] = cutoff[..., :-1].clone()
                    cutoff[..., 0] = False
                    sorted_probs[cutoff] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_id = torch.multinomial(sorted_probs, num_samples=1)
                    next_id = sorted_indices.gather(-1, next_id)
                else:
                    next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            nid = next_id.item()
            if nid == eos_id:
                break
            generated.append(nid)
            last = next_id

    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    throughput = (len(generated) / max(sum(ttit_list), 1e-6)) if ttit_list else 0.0

    return {
        "answer": answer,
        "TTFT_sec": ttft,
        "TTIT_avg_sec": float(np.mean(ttit_list)) if ttit_list else 0.0,
        "throughput_tok_per_sec": throughput,
        "prompt_tokens": prompt_len,
        "generated_tokens": len(generated),
    }


def cmd_baseline(args):
    device = now_device()
    print(f"[baseline] loading model {args.dec} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.dec, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.dec).to(device)
    model.eval()
    print("[baseline] model loaded.")

    with open(args.input_file, "r") as f:
        raw = json.load(f)

    data = raw["data"]
    results = []
    total = len(data) * args.num_runs

    for item in data:
        question = item["question"]
        passages = [parse_original(entry) for entry in item["context"]]

        for run_idx in range(args.num_runs):
            completed = len(results) + 1
            print(f"[baseline] [{completed}/{total}] run {run_idx+1}/{args.num_runs}: {question[:60]}...")

            out = baseline_generate(
                tokenizer=tokenizer,
                model=model,
                question=question,
                passages=passages,
                max_ctx_tokens=args.ctx_max,
                max_new_tokens=args.max_new,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )

            results.append({
                "question": question,
                "run": run_idx + 1,
                "ground_truth": item.get("ground_truth", ""),
                **out,
            })
            print(f"  → answer: {out['answer'][:120]}...")
            print(f"  → TTFT={out['TTFT_sec']:.2f}s  throughput={out['throughput_tok_per_sec']:.1f} tok/s")

    # Summary stats
    latencies = [r["TTFT_sec"] for r in results]
    throughputs = [r["throughput_tok_per_sec"] for r in results]
    print(f"\n[baseline summary]")
    print(f"  total runs      : {len(results)}")
    print(f"  avg TTFT        : {sum(latencies)/len(latencies):.2f}s")
    print(f"  avg throughput  : {sum(throughputs)/len(throughputs):.1f} tok/s")

    if os.path.dirname(args.output_file):
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump({
            "summary": {
                "total_runs": len(results),
                "avg_TTFT_s": round(sum(latencies) / len(latencies), 4),
                "avg_throughput_tok_per_sec": round(sum(throughputs) / len(throughputs), 2),
            },
            "results": results,
        }, f, indent=2)

    print(f"[baseline] saved {len(results)} results to {args.output_file}")


def build_argparser():
    p = argparse.ArgumentParser(description="Baseline RAG — raw passages, no chunking/compression")
    p.add_argument("--dec", type=str, default="meta-llama/Llama-3.2-3B")
    p.add_argument("--input_file", type=str, required=True)
    p.add_argument("--output_file", type=str, required=True)
    p.add_argument("--ctx_max", type=int, default=2048)
    p.add_argument("--max_new", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--num_runs", type=int, default=1)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    cmd_baseline(args)