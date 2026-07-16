#!/usr/bin/env python3
"""Plot measured decode throughput against the roofline ceiling.

Reads the CSV written by benchmark_mlx.py and draws the median measured
tokens/sec next to the theoretical bandwidth ceiling from roofline.py, so the
gap (== optimization headroom) is visible at a glance.

    python bench/plot_results.py --csv results/mlx_results.csv \
        --params 8 --quant q4 --bandwidth 120
"""
import argparse
import csv
import statistics
import sys

from roofline import decode_ceiling, BYTES_PER_PARAM  # local import


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/mlx_results.csv")
    ap.add_argument("--params", type=float, default=8.0)
    ap.add_argument("--quant", choices=list(BYTES_PER_PARAM), default="q4")
    ap.add_argument("--bandwidth", type=float, default=120.0, help="GB/s")
    ap.add_argument("--out", default="results/roofline_vs_measured.png")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not installed:  pip install matplotlib")

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    measured = statistics.median(float(r["decode_tok_s"]) for r in rows)

    _, ceiling = decode_ceiling(args.params, BYTES_PER_PARAM[args.quant], args.bandwidth)
    util = 100.0 * measured / ceiling if ceiling else 0.0

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["measured", "roofline\nceiling"], [measured, ceiling],
                  color=["#3fb950", "#58a6ff"])
    ax.set_ylabel("decode throughput (tokens / s)")
    ax.set_title(f"{args.params:g}B {args.quant} @ {args.bandwidth:.0f} GB/s\n"
                 f"bandwidth utilization: {util:.0f}%")
    for b, v in zip(bars, [measured, ceiling]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"measured {measured:.1f} tok/s vs ceiling {ceiling:.1f} tok/s "
          f"({util:.0f}% of bandwidth)  ->  {args.out}")


if __name__ == "__main__":
    main()
