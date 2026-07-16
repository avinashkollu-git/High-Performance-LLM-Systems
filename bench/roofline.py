#!/usr/bin/env python3
"""Roofline performance model for single-batch LLM inference.

Why this exists
---------------
Autoregressive DECODE (generating one token at a time, batch size 1) is almost
always *memory-bandwidth bound*, not compute bound: to produce each new token the
hardware must stream the entire set of model weights from unified memory exactly
once. So the hard ceiling on decode throughput is

        tokens/sec  <=  memory_bandwidth (bytes/s)  /  weight_bytes

PREFILL (processing the prompt) is different: many tokens are processed together,
so the same weights are reused across the batch and the phase becomes compute
bound. This tool reports the decode ceiling and the arithmetic intensity so you
can see, for a given chip and quantization, what throughput is even *possible*
before you benchmark the real thing (see bench/benchmark_mlx.py).

Everything here is computed from first principles — nothing is measured or made
up. Plug in your chip's real memory bandwidth (Apple publishes it per model).
"""
import argparse

# Apple M4 family unified-memory bandwidth, per Apple's published specs (GB/s).
# These are the memory-bandwidth ceilings; verify against your exact configuration.
APPLE_BW_GBPS = {
    "m4":      120.0,
    "m4-pro":  273.0,
    "m4-max":  410.0,   # 14-core bin; the higher-binned Max is ~546 GB/s
}

# bytes per weight for common quantizations
BYTES_PER_PARAM = {
    "fp16": 2.0,
    "int8": 1.0,
    "q4":   0.5,        # 4-bit (approx; real GGUF/MLX 4-bit ~4.3-4.5 bits/weight)
}


def decode_ceiling(params_billion, bytes_per_param, bw_gbps):
    """Return (weight_GB, max_tokens_per_sec) for batch-1 decode."""
    weight_bytes = params_billion * 1e9 * bytes_per_param
    bw = bw_gbps * 1e9
    return weight_bytes / 1e9, bw / weight_bytes


def write_svg(path, params, bw, label):
    """Render the decode ceilings as a dependency-free SVG bar chart."""
    quants = list(BYTES_PER_PARAM)
    tps = [decode_ceiling(params, BYTES_PER_PARAM[q], bw)[1] for q in quants]
    W, H, pad, base = 560, 340, 60, 270
    top = 70
    vmax = max(tps) * 1.15
    colors = {"fp16": "#f85149", "int8": "#d29922", "q4": "#3fb950"}
    bw_px = 90
    gap = 60
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'font-family="monospace" font-size="13">',
         f'<rect width="{W}" height="{H}" fill="#0d1117"/>',
         f'<text x="{W//2}" y="30" fill="#e6edf3" font-size="15" font-weight="bold" '
         f'text-anchor="middle">Decode ceiling: {params:g}B model @ {label}</text>',
         f'<text x="{W//2}" y="50" fill="#8b949e" font-size="12" text-anchor="middle">'
         f'batch-1, memory-bandwidth bound (tokens / second)</text>',
         f'<line x1="{pad}" y1="{base}" x2="{W-30}" y2="{base}" stroke="#30363d"/>']
    for i, q in enumerate(quants):
        x = pad + 40 + i * (bw_px + gap)
        h = (tps[i] / vmax) * (base - top)
        y = base - h
        s.append(f'<rect x="{x}" y="{y:.1f}" width="{bw_px}" height="{h:.1f}" '
                 f'fill="{colors[q]}" rx="3"/>')
        s.append(f'<text x="{x+bw_px//2}" y="{y-8:.1f}" fill="#e6edf3" '
                 f'text-anchor="middle">{tps[i]:.0f} tok/s</text>')
        s.append(f'<text x="{x+bw_px//2}" y="{base+20}" fill="#8b949e" '
                 f'text-anchor="middle">{q}</text>')
    s.append(f'<text x="{pad}" y="{H-20}" fill="#8b949e" font-size="11">'
             f'4-bit quantization moves 4x fewer bytes/token than fp16 -> 4x the ceiling.</text>')
    s.append('</svg>')
    with open(path, "w") as f:
        f.write("\n".join(s))
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", type=float, default=8.0,
                    help="model size in billions of parameters (default: 8 = Llama-3-8B)")
    ap.add_argument("--chip", choices=list(APPLE_BW_GBPS), default="m4",
                    help="Apple chip (sets memory bandwidth)")
    ap.add_argument("--bandwidth", type=float, default=None,
                    help="override memory bandwidth in GB/s (for any hardware)")
    ap.add_argument("--svg", metavar="PATH", default=None,
                    help="also write a bar-chart SVG of the ceilings to PATH")
    args = ap.parse_args()

    bw = args.bandwidth if args.bandwidth is not None else APPLE_BW_GBPS[args.chip]
    label = f"{bw:.0f} GB/s" + ("" if args.bandwidth else f"  ({args.chip})")

    print(f"Roofline: batch-1 DECODE ceiling for a {args.params:g}B-param model")
    print(f"Memory bandwidth: {label}\n")
    print(f"  {'quant':6}  {'weights':>9}  {'max tok/s':>10}   {'ms/token':>9}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*10}   {'-'*9}")
    for q, bpp in BYTES_PER_PARAM.items():
        gb, tps = decode_ceiling(args.params, bpp, bw)
        print(f"  {q:6}  {gb:7.2f}GB  {tps:10.1f}   {1000.0/tps:9.2f}")

    print("\nNotes:")
    print("  * These are UPPER BOUNDS (100% bandwidth, no overhead). Real")
    print("    throughput is lower; the gap is your optimization headroom.")
    print("  * Decode is memory-bound: arithmetic intensity ~2 FLOP/byte, far")
    print("    below the hardware's FLOP:byte ratio, so bandwidth dominates.")
    print("  * Prefill is compute-bound and scales differently (batched matmuls).")

    if args.svg:
        write_svg(args.svg, args.params, bw, label.strip())


if __name__ == "__main__":
    main()
