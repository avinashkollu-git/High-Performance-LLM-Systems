#!/usr/bin/env python3
"""Benchmark LLM inference on Apple Silicon with MLX.

Measures, over several runs and reported as medians:
  * TTFT      - time to first token (prefill latency)
  * decode    - steady-state generation throughput (tokens / second)
  * peak mem  - peak unified-memory footprint

Then compare the decode number to the theoretical ceiling from
`bench/roofline.py` to see how much of the memory bandwidth you are actually
using. Results are written to a CSV for plotting (bench/plot_results.py).

Usage:
    pip install -r requirements.txt
    python bench/benchmark_mlx.py --model mlx-community/Meta-Llama-3-8B-Instruct-4bit
"""
import argparse
import csv
import os
import statistics
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
                    help="HF/MLX model id or local path")
    ap.add_argument("--prompt", default="Explain how a CPU pipeline works in three sentences.")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default="results/mlx_results.csv")
    args = ap.parse_args()

    try:
        import mlx.core as mx
        from mlx_lm import load, stream_generate
    except ImportError:
        sys.exit("Missing dependencies. Install them with:\n"
                 "    pip install -r requirements.txt")

    print(f"Loading {args.model} ...")
    model, tokenizer = load(args.model)

    have_peak = hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory")

    def one_run():
        if have_peak and hasattr(mx.metal, "reset_peak_memory"):
            mx.metal.reset_peak_memory()
        t0 = time.perf_counter()
        first_t = None
        n = 0
        for _ in stream_generate(model, tokenizer, args.prompt, max_tokens=args.max_tokens):
            if first_t is None:            # first yield == prefill done + 1st token
                first_t = time.perf_counter()
            n += 1
        t_end = time.perf_counter()
        ttft = (first_t - t0) if first_t else float("nan")
        decode_s = (t_end - first_t) if first_t else (t_end - t0)
        dec_tps = (n - 1) / decode_s if n > 1 and decode_s > 0 else 0.0
        peak_gb = (mx.metal.get_peak_memory() / 1e9) if have_peak else float("nan")
        return ttft, dec_tps, peak_gb, n

    print("Warmup ...")
    one_run()

    rows = []
    for r in range(args.runs):
        ttft, tps, peak, n = one_run()
        rows.append([r, round(ttft, 4), round(tps, 2), round(peak, 3), n])
        print(f"  run {r}: TTFT {ttft*1000:6.1f} ms | decode {tps:6.1f} tok/s "
              f"| peak {peak:5.2f} GB | {n} tokens")

    ttfts = [r[1] for r in rows]
    tpss = [r[2] for r in rows]
    print(f"\nMedian over {args.runs} runs: "
          f"TTFT {statistics.median(ttfts)*1000:.1f} ms | "
          f"decode {statistics.median(tpss):.1f} tok/s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "ttft_s", "decode_tok_s", "peak_mem_gb", "tokens"])
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
