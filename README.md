# MoE Distributed Training Pipeline

A from-scratch PyTorch implementation of a Sparse Mixture-of-Experts transformer
with FSDP2 per-parameter sharding, activation checkpointing, and reproducible
benchmarks.

## Architecture

Total parameters: ~26M  
Active parameters per token: ~13M (top-2 of 8 experts per layer)

## Files

| File                   | Purpose                                             |
| ---------------------- | --------------------------------------------------- |
| `config.py`            | All hyperparameters in one place                    |
| `moe/experts.py`       | Single ExpertFFN (SwiGLU, isolated from routing)    |
| `moe/router.py`        | TopKRouter + load-balance auxiliary loss            |
| `moe/layer.py`         | Token dispatch, expert execution, aggregation       |
| `model.py`             | Full SparseMoETransformer, activation checkpointing |
| `train.py`             | FSDP2 + torchrun distributed training loop          |
| `benchmark_routing.py` | Expert utilization benchmark (CPU, M1-safe)         |
| `benchmark_memory.py`  | GPU memory benchmark (requires CUDA / Colab)        |

## Quick Start

```bash
pip install torch
```

### 1. Distributed training with FSDP2 (2 CPU workers)

```bash
torchrun --standalone --nproc_per_node 2 train.py --device cpu
```

Expected output:

```
[FSDP2] Applied per-parameter sharding across 2 ranks
[FSDP2] Each rank holds ~1/2 of each parameter tensor
step   0 | task_loss=8.3141 | aux_loss=0.0098 | total=8.3239
step   5 | task_loss=8.2901 | aux_loss=0.0089 | total=8.2990
...
```

### 2. Routing efficiency benchmark (M1 safe, CPU only)

```bash
python benchmark_routing.py
```

Trains two models (with and without load-balance loss) and measures how uniformly
tokens are distributed across the 8 experts.

### 3. Memory benchmark (requires CUDA GPU)

**On Google Colab** (Runtime → T4 GPU):

```bash
!git clone <your-repo-url>
%cd moe_pipeline
!python benchmark_memory.py
```

## Design Decisions

### Why FSDP2 instead of DDP?

DDP replicates the full model on every rank. With a 26M parameter MoE model at
fp32, each rank holds 26M × 4 bytes ≈ 100MB of parameters alone, plus equal
amounts for gradients and optimizer states.

FSDP2 (`fully_shard`) shards parameters along dim-0 using DTensor. With 2 ranks,
each rank holds ~50MB of parameters. Before each layer's forward pass, parameters
are all-gathered; after backward, gradients are reduce-scattered so each rank
retains only its shard.

FSDP2 is preferred over FSDP1 because it uses DTensor-based per-parameter sharding
instead of flat-parameter sharding, giving a simpler sharded state dict and better
composability with Tensor Parallel.

### Why activation checkpointing at the MoELayer level?

Activation checkpointing is applied per-MoELayer (not globally). The expert FFN
layers are the most activation-heavy parts of the forward pass (8 experts × batch
× seq hidden states), so checkpointing them gives the largest memory reduction per
unit of recompute cost. The embedding and LM head are cheap and left without
checkpointing.

### Load-balance loss formula

```
aux_loss = α × N × Σᵢ( fᵢ × Pᵢ )
```

- `fᵢ`: fraction of tokens dispatched to expert i (discrete, top-1 assignment)
- `Pᵢ`: mean router probability for expert i across all tokens (differentiable)
- `N`: number of experts (scaling factor so α is invariant to expert count)
- `α`: coefficient (0.01 in config)

`fᵢ` provides the signal; `Pᵢ` provides the gradient. The product pushes the
router toward uniform dispatch without interfering with task loss.

## Reproducibility

Both benchmark scripts print a summary section at the end with measured percentages.
Run them on your target hardware — the numbers are not hardcoded anywhere in the project.
