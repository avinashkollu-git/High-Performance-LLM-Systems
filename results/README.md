# Results

This folder holds benchmark output generated on **your** machine — it is
intentionally left empty in the repo so that no numbers here are second-hand.

To populate it:

```bash
pip install -r ../requirements.txt
python ../bench/benchmark_mlx.py            # writes mlx_results.csv
python ../bench/plot_results.py             # writes roofline_vs_measured.png
```

`mlx_results.csv` records per-run TTFT, decode tokens/sec, and peak memory.
`roofline_vs_measured.png` shows measured throughput against the theoretical
bandwidth ceiling, i.e. what fraction of memory bandwidth you are using.
