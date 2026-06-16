# train.py
# Distributed training entry point for SparseMoETransformer.
#
# On Mac (no torchrun): python train.py --device cpu
# On Linux/GPU:         torchrun --standalone --nproc_per_node 2 train.py --device cuda
#
# Uses torch.multiprocessing.spawn to launch 2 worker processes on Mac,
# avoiding torchrun's IPv6 TCP store issue on macOS.

import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard

from config import MODEL_CFG, TRAIN_CFG
from model import SparseMoETransformer


def setup_distributed(rank: int, world_size: int, backend: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )


def build_and_shard_model(device, device_type, rank, world_size):
    torch.manual_seed(TRAIN_CFG.seed + rank)
    model = SparseMoETransformer(MODEL_CFG, use_activation_checkpointing=True).to(device)

    if world_size > 1:
        mesh = init_device_mesh(device_type, (world_size,), mesh_dim_names=("dp",))
        for layer in model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)

        if rank == 0:
            print(f"[FSDP2] Applied per-parameter sharding across {world_size} ranks")
            print(f"[FSDP2] Each rank holds ~1/{world_size} of each parameter tensor")

    return model


def train_worker(rank: int, world_size: int, device_type: str) -> None:
    backend = "gloo" if device_type == "cpu" else "nccl"
    setup_distributed(rank, world_size, backend)

    device = torch.device("cpu") if device_type == "cpu" else torch.device(f"cuda:{rank}")
    torch.manual_seed(TRAIN_CFG.seed)

    model = build_and_shard_model(device, device_type, rank, world_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CFG.lr)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"SparseMoETransformer | {world_size} rank(s) | device={device_type}")
        print(f"  Experts per layer : {MODEL_CFG.num_experts}")
        print(f"  Top-k per token   : {MODEL_CFG.top_k}")
        print(f"  Layers            : {MODEL_CFG.num_layers}")
        print(f"  d_model           : {MODEL_CFG.d_model}")
        print(f"{'='*60}\n")

    def make_batch():
        return torch.randint(
            0, MODEL_CFG.vocab_size,
            (TRAIN_CFG.batch_size, TRAIN_CFG.seq_len),
            device=device,
        )

    model.train()
    for step in range(TRAIN_CFG.num_steps):
        input_ids = make_batch()
        targets   = make_batch()

        optimizer.zero_grad()
        out = model(input_ids)

        task_loss  = torch.nn.functional.cross_entropy(
            out.logits.reshape(-1, MODEL_CFG.vocab_size),
            targets.reshape(-1),
        )
        total_loss = task_loss + out.total_aux_loss
        total_loss.backward()
        optimizer.step()

        if rank == 0 and (step % 5 == 0 or step == TRAIN_CFG.num_steps - 1):
            print(
                f"step {step:3d} | "
                f"task_loss={task_loss.item():.4f} | "
                f"aux_loss={out.total_aux_loss.item():.4f} | "
                f"total={total_loss.item():.4f}"
            )

    if rank == 0:
        print("\nTraining complete.")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--world_size", type=int, default=2)
    args = parser.parse_args()

    mp.spawn(
        train_worker,
        args=(args.world_size, args.device),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
