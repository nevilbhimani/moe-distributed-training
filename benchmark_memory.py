# benchmark_memory.py
#
# Measures per-rank peak GPU memory across three conditions:
#
#   Condition 1 — DDP baseline:
#     Single process, full model on one GPU, no activation checkpointing.
#     This is the memory cost DDP incurs on each rank: every rank holds
#     a complete copy of all parameters, gradients, and optimizer states.
#
#   Condition 2 — FSDP2, no checkpointing:
#     Two processes via mp.spawn, FSDP2 per-parameter sharding.
#     Each rank holds ~half the parameters. We measure rank 0's peak memory.
#     All-gather happens transiently during forward but does not persist.
#
#   Condition 3 — FSDP2 + activation checkpointing:
#     Same as condition 2, but intermediate activations are discarded during
#     forward and recomputed during backward via torch.utils.checkpoint.
#     Lowest memory condition — both parameter sharding and activation savings.
#
# On single-GPU Colab: both FSDP2 processes share one physical T4.
# Per-rank memory is still the honest metric — in production each rank
# maps to one physical GPU, so per-rank allocation = per-GPU requirement.
#
# Run on Colab T4: !python benchmark_memory.py

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard

from config import MODEL_CFG, BENCH_CFG
from model import SparseMoETransformer


def bytes_to_mb(n: int) -> float:
    return n / (1024 ** 2)


# ── Condition 1: DDP baseline (single process, full model) ───────────────────

def measure_ddp_baseline(
    batch_size: int,
    seq_len: int,
    warmup: int,
    steps: int,
) -> float:
    """
    Full model on single GPU, no sharding, no checkpointing.
    Represents DDP memory cost per rank: complete parameter replication.
    """
    device = torch.device("cuda")
    torch.manual_seed(42)

    model = SparseMoETransformer(
        MODEL_CFG, use_activation_checkpointing=False
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def step():
        ids     = torch.randint(0, MODEL_CFG.vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, MODEL_CFG.vocab_size, (batch_size * seq_len,), device=device)
        optimizer.zero_grad()
        out  = model(ids)
        loss = F.cross_entropy(out.logits.reshape(-1, MODEL_CFG.vocab_size), targets)
        (loss + out.total_aux_loss).backward()
        optimizer.step()

    for _ in range(warmup):
        step()

    peaks = []
    for _ in range(steps):
        torch.cuda.reset_peak_memory_stats(device)
        step()
        torch.cuda.synchronize(device)
        peaks.append(bytes_to_mb(torch.cuda.max_memory_allocated(device)))

    return sum(peaks) / len(peaks)


# ── Conditions 2 & 3: FSDP2 (two processes via mp.spawn) ─────────────────────

def fsdp2_worker(
    rank: int,
    world_size: int,
    use_activation_checkpointing: bool,
    batch_size: int,
    seq_len: int,
    warmup: int,
    steps: int,
    result_queue: mp.Queue,
) -> None:
    """
    Worker function for FSDP2 benchmark.
    Rank 0 puts its peak memory measurement into result_queue.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0"

    # Use nccl for GPU processes
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.manual_seed(42 + rank)

    model = SparseMoETransformer(
        MODEL_CFG, use_activation_checkpointing=use_activation_checkpointing
    ).to(device)

    # Apply FSDP2 bottom-up: shard each MoELayer, then root model
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def step():
        ids     = torch.randint(0, MODEL_CFG.vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, MODEL_CFG.vocab_size, (batch_size * seq_len,), device=device)
        optimizer.zero_grad()
        out  = model(ids)
        loss = F.cross_entropy(out.logits.reshape(-1, MODEL_CFG.vocab_size), targets)
        (loss + out.total_aux_loss).backward()
        optimizer.step()

    for _ in range(warmup):
        step()

    peaks = []
    for _ in range(steps):
        torch.cuda.reset_peak_memory_stats(device)
        step()
        torch.cuda.synchronize(device)
        peaks.append(bytes_to_mb(torch.cuda.max_memory_allocated(device)))

    if rank == 0:
        result_queue.put(sum(peaks) / len(peaks))

    dist.destroy_process_group()


def measure_fsdp2(
    use_activation_checkpointing: bool,
    batch_size: int,
    seq_len: int,
    warmup: int,
    steps: int,
    world_size: int = 2,
) -> float:
    """Launch FSDP2 workers and return rank-0 peak memory in MB."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=fsdp2_worker,
            args=(rank, world_size, use_activation_checkpointing,
                  batch_size, seq_len, warmup, steps, result_queue),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    return result_queue.get()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Run this on Google Colab with a T4 GPU.")
        print("Runtime → Change runtime type → T4 GPU")
        return

    B       = BENCH_CFG.mem_batch_size
    S       = BENCH_CFG.mem_seq_len
    warmup  = BENCH_CFG.mem_warmup_steps
    steps   = BENCH_CFG.mem_measure_steps

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL_CFG.num_experts} experts x {MODEL_CFG.num_layers} layers | "
          f"d_model={MODEL_CFG.d_model} | expert_hidden={MODEL_CFG.expert_hidden}")
    print(f"Batch: {B} | SeqLen: {S} | Warmup: {warmup} | Measure steps: {steps}")
    print()

    # Condition 1
    print("[1/3] DDP baseline — full model, single process, no checkpointing...")
    mem_ddp = measure_ddp_baseline(B, S, warmup, steps)
    print(f"  Rank-0 peak memory: {mem_ddp:.1f} MB")

    # Condition 2
    print("\n[2/3] FSDP2 — 2 ranks, per-parameter sharding, no checkpointing...")
    mem_fsdp2 = measure_fsdp2(False, B, S, warmup, steps)
    print(f"  Rank-0 peak memory: {mem_fsdp2:.1f} MB")

    # Condition 3
    print("\n[3/3] FSDP2 + activation checkpointing — 2 ranks, sharding + recompute...")
    mem_fsdp2_ckpt = measure_fsdp2(True, B, S, warmup, steps)
    print(f"  Rank-0 peak memory: {mem_fsdp2_ckpt:.1f} MB")

    # Results
    fsdp2_reduction      = (mem_ddp - mem_fsdp2) / mem_ddp * 100
    ckpt_reduction       = (mem_fsdp2 - mem_fsdp2_ckpt) / mem_fsdp2 * 100
    combined_reduction   = (mem_ddp - mem_fsdp2_ckpt) / mem_ddp * 100

    print(f"\n{'='*58}")
    print(f"  Condition                        Peak mem   vs DDP")
    print(f"  {'-'*54}")
    print(f"  DDP baseline (full replication)  {mem_ddp:>7.1f} MB     —")
    print(f"  FSDP2 sharding only              {mem_fsdp2:>7.1f} MB  -{fsdp2_reduction:.1f}%")
    print(f"  FSDP2 + activation checkpointing {mem_fsdp2_ckpt:>7.1f} MB  -{combined_reduction:.1f}%")
    print(f"{'='*58}")
    print(f"\n  FSDP2 sharding alone      : -{fsdp2_reduction:.1f}% vs DDP baseline")
    print(f"  Checkpointing alone       : -{ckpt_reduction:.1f}% vs FSDP2-only")
    print(f"  Combined (FSDP2 + ckpt)   : -{combined_reduction:.1f}% vs DDP baseline")
    print(f"{'='*58}")


if __name__ == "__main__":
    main()
