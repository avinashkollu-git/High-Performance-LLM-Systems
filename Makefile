# High-Performance LLM Systems - convenience targets
PARAMS   ?= 8
CHIP     ?= m4
MODEL    ?= mlx-community/Meta-Llama-3-8B-Instruct-4bit

.PHONY: roofline chart bench plot help
help:
	@echo "make roofline   # theoretical decode ceiling (no deps, runs anywhere)"
	@echo "make chart       # regenerate docs/roofline_m4.svg"
	@echo "make bench       # measure real throughput on Apple Silicon (needs MLX)"
	@echo "make plot        # measured-vs-roofline chart (needs matplotlib)"
	@echo "vars: PARAMS=$(PARAMS) CHIP=$(CHIP) MODEL=$(MODEL)"

roofline: ## theoretical bandwidth ceiling for the given model + chip
	python3 bench/roofline.py --params $(PARAMS) --chip $(CHIP)

chart: ## regenerate the committed roofline SVG
	python3 bench/roofline.py --params $(PARAMS) --chip $(CHIP) --svg docs/roofline_m4.svg

bench: ## measure TTFT / decode tok-s / peak memory with MLX
	python3 bench/benchmark_mlx.py --model $(MODEL)

plot: ## chart measured throughput against the roofline
	python3 bench/plot_results.py --params $(PARAMS)
