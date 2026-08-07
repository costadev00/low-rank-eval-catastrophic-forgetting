# Layer-wise LoRA Rank Allocation and Catastrophic Forgetting

This repository provides a reproducible QLoRA pipeline for testing whether
Transformer layers prefer different LoRA ranks and whether reallocating a fixed
parameter budget changes acquisition and retention during sequential
fine-tuning.

The primary model is
`Qwen/Qwen3-4B-Instruct-2507`. LoRA is applied to `q_proj` and `v_proj`;
`q/k/v/o` is available as an ablation. The main experiment compares uniform,
early-heavy, middle-heavy, late-heavy, and spectral rank allocations at the
same trainable-parameter budget.

No scientific results are committed until the corresponding runs complete.
The paper builder intentionally fails when required artifacts are missing.

## Hardware and software

The tested target host has four NVIDIA RTX 4000 Ada Generation GPUs with 20 GB
each. Training uses one NF4-quantized model replica per process. The code never
uses `device_map="auto"` in DDP.

Create the locked Python 3.12 environment:

```bash
uv python install 3.12
uv sync --extra dev
cp .env.example .env
# Edit .env and set HF_TOKEN. The real file is ignored by Git.
uv run accelerate config
```

SDPA is the default attention backend. Flash Attention is optional:

```bash
uv sync --extra flash
```

It requires a compatible CUDA compiler and is not necessary for any main
experiment.

## Data preparation and decontamination

Model and dataset revisions are pinned in `configs/base.yaml`. NuminaMath is
filtered in this order:

1. rows whose `source` identifies GSM8K;
2. normalized exact matches against GSM8K main train and test;
3. approximate character-5-gram matches confirmed above the configured
   threshold;
4. internal exact duplicates.

IFEval-like data is filtered against all official IFEval prompts using the same
hash and approximate-match process. Exclusive removal counts and hashed audit
records are written below `data/processed/<config-fingerprint>/`.

```bash
uv run python scripts/prepare_data.py --config configs/smoke.yaml
uv run python scripts/prepare_data.py --config configs/base.yaml
```

Budgets count tokens whose labels participate in the loss, i.e. assistant
tokens. Prompt and total tokens are also recorded. Calibration and training
splits are disjoint and are created only after decontamination.

## Tests

CPU unit tests cover configuration inheritance, contamination filters, exact
token budgets, rank patterns, parameter budgets, continual-learning metrics,
factorized singular values, spectral allocation, and adapter safetensors
reload:

```bash
uv run pytest
```

The smoke configuration is deliberately non-scientific: it uses 20k assistant
tokens per task, limits training-source preparation, and evaluates eight
examples per benchmark.

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run accelerate launch --num_processes 1 \
  scripts/run_sequential.py \
  --config configs/smoke.yaml \
  --task_order ifeval_to_math \
  --seed 42
```

Four-GPU DDP smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --num_processes 4 \
  scripts/run_sequential.py \
  --config configs/smoke.yaml \
  --task_order math_to_ifeval \
  --seed 42
```

## Full experiment

Prepare the full datasets once, then run the discarded rank-32 spectral pilot:

```bash
uv run python scripts/prepare_data.py --config configs/base.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --num_processes 4 \
  scripts/run_spectral_pilot.py \
  --config configs/base.yaml
```

Run seed 42 for all main configurations and both task orders:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --num_processes 4 \
  scripts/run_experiment_matrix.py \
  --config_dir configs \
  --configs uniform early_heavy middle_heavy late_heavy spectral \
  --orders ifeval_to_math math_to_ifeval \
  --seeds 42
```

Aggregate the initial sweep. `manual_selection.json` uses only calibration NLL
from decontaminated training-derived data:

```bash
uv run python scripts/analyze_results.py --results_dir results
```

Run three seeds for uniform, the selected manual allocation, and spectral:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --num_processes 4 \
  scripts/run_confirmatory.py \
  --config_dir configs \
  --results_dir results \
  --seeds 7 42 123
```

The entire initial matrix, calibration-only selection, confirmatory matrix,
analysis, and guarded paper build can also be run as one resumable command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python scripts/run_full_study.py \
  --num_processes 4
```

Its current step is written to `results/full_study_state.json`. Every nested
training command retains the same checkpoint- and evaluation-level resumption
semantics described above. Generation uses `evaluation.batch_size`, while the
full-vocabulary calibration NLL uses the independently configurable
`evaluation.calibration_batch_size` (default `1`) to bound peak memory.

All scripts are idempotent. A completed stage has `stage_complete.json`;
interrupted Trainer stages resume from the latest `checkpoint-*`. Stage 2
loads the saved stage-1 adapter with `is_trainable=True` and resets optimizer
and scheduler by default.

## Independent evaluation

Base model:

```bash
CUDA_VISIBLE_DEVICES=0 uv run accelerate launch --num_processes 1 \
  scripts/evaluate_checkpoint.py \
  --config configs/uniform.yaml \
  --output_dir results/manual_base_evaluation
```

Saved adapter:

```bash
CUDA_VISIBLE_DEVICES=0 uv run accelerate launch --num_processes 1 \
  scripts/evaluate_checkpoint.py \
  --config configs/uniform.yaml \
  --adapter checkpoints/<run-id>/stage_1/adapter \
  --output_dir results/manual_stage_1_evaluation
```

Generation is greedy with temperature zero. Raw responses and extracted GSM8K
answers are kept as JSONL. IFEval uses the official verifier implementation
shipped by `lm-eval==0.4.12`.

## Outputs and paper

`scripts/analyze_results.py` creates:

- `results/aggregates/all_results.{csv,json}`;
- multi-seed mean, standard deviation, and t-based 95% intervals;
- the Pareto frontier;
- the final comparison table;
- rank-depth, stage-performance, forgetting, parameter-performance, heatmap,
  and spectral-energy figures.

After all mandatory runs complete:

```bash
uv run python scripts/analyze_results.py --results_dir results
uv run python scripts/build_paper.py
```

The first, explicitly limited paper snapshot uses the eight completed manual
protocols (four allocations, both task orders, seed 42):

```bash
uv run python scripts/build_first_paper.py \
  --results_dir results --config_dir configs --paper_dir paper
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

`paper/main.tex` is an English ICML-format manuscript for Overleaf with two
pages of main text, followed by references and appendices. The corrected style
files are already included under `paper/`; `scripts/repair_latex_template.py`
only audits or reconstructs them from the original scrambled bundle. The
first-paper builder rejects a missing manual protocol, partial benchmark,
unequal parameter budget, stale configuration, or unexpected token budget.
It labels the single-seed scope and excludes the unevaluated spectral adapter
from comparative claims.

The original `scripts/build_paper.py` remains the stricter builder for the
pre-registered complete matrix. It refuses to emit result macros until the
spectral protocols and confirmatory seeds exist, so incomplete runs cannot
silently become full-study claims.

## Experimental caution

The complete matrix processes tens of millions of supervised tokens and many
benchmark generations. It can take multiple days even on four GPUs. The
pipeline measures whether rank location matters; it does not assume that a
non-uniform allocation will win.
