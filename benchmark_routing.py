# benchmark_routing.py
#
# Measures expert load distribution with and without the auxiliary load-balance loss.
#
# Methodology:
#   Train two models from the same random seed for N steps:
#     - baseline: cross-entropy loss only (alpha=0, routing collapses over time)
#     - balanced:  cross-entropy + aux_loss (alpha=0.01, routing stays uniform)
#
#   After training, run M evaluation batches and collect a per-expert token count
#   histogram. Compare the distributions.
#
#   Primary metric: coefficient of variation (CV = std/mean) of expert load.
#     Lower CV = more uniform dispatch = higher utilization of all experts.
#
#   Secondary metric: max/min load ratio — how much the busiest expert dominates.
#
# Run: python benchmark_routing.py

import torch
import torch.nn.functional as F
import numpy as np
from config import MODEL_CFG, BENCH_CFG, ModelConfig
from model import SparseMoETransformer


TRAIN_STEPS   = 200   # enough for routing collapse to appear without aux loss
EVAL_STEPS    = 50
BATCH_SIZE    = BENCH_CFG.routing_batch_size
SEQ_LEN       = BENCH_CFG.routing_seq_len
DEVICE        = torch.device("cpu")


def train_model(alpha: float, seed: int = 42) -> SparseMoETransformer:
    torch.manual_seed(seed)
    cfg   = ModelConfig(aux_loss_alpha=alpha)
    model = SparseMoETransformer(cfg, use_activation_checkpointing=False).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=3e-3)

    model.train()
    for step in range(TRAIN_STEPS):
        ids     = torch.randint(0, MODEL_CFG.vocab_size, (BATCH_SIZE, SEQ_LEN))
        targets = torch.randint(0, MODEL_CFG.vocab_size, (BATCH_SIZE * SEQ_LEN,))
        opt.zero_grad()
        out  = model(ids)
        loss = F.cross_entropy(out.logits.reshape(-1, MODEL_CFG.vocab_size), targets)
        (loss + out.total_aux_loss).backward()
        opt.step()

        if (step + 1) % 50 == 0:
            tag = f"alpha={alpha}"
            print(f"  [{tag}] step {step+1:3d} | ce={loss.item():.4f} | aux={out.total_aux_loss.item():.5f}")

    return model


def collect_expert_counts(model: SparseMoETransformer) -> np.ndarray:
    counts = np.zeros(MODEL_CFG.num_experts, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for _ in range(EVAL_STEPS):
            ids = torch.randint(0, MODEL_CFG.vocab_size, (BATCH_SIZE, SEQ_LEN))
            x   = model.embedding(ids)
            for layer in model.layers:
                x_norm = layer.norm(x)
                route  = layer.router(x_norm)
                for eid in range(MODEL_CFG.num_experts):
                    counts[eid] += (route.expert_indices == eid).sum().item()
                x, _ = layer(x)
    return counts


def load_stats(counts: np.ndarray) -> dict:
    fracs = counts / counts.sum()
    cv    = fracs.std() / fracs.mean()
    p     = fracs / fracs.sum()
    h     = -np.sum(p * np.log(p + 1e-12))
    return {
        "cv":       cv,
        "entropy":  h / np.log(len(counts)),   # normalized 0-1
        "max_frac": fracs.max(),
        "min_frac": fracs.min(),
        "fracs":    fracs,
    }


def main():
    print(f"Training baseline (no load-balance loss) for {TRAIN_STEPS} steps...")
    model_base = train_model(alpha=0.0)

    print(f"\nTraining balanced model (aux_loss alpha=0.01) for {TRAIN_STEPS} steps...")
    model_bal  = train_model(alpha=0.01)

    print(f"\nCollecting expert counts over {EVAL_STEPS} eval batches...")
    counts_base = collect_expert_counts(model_base)
    counts_bal  = collect_expert_counts(model_bal)

    s_base = load_stats(counts_base)
    s_bal  = load_stats(counts_bal)

    ideal_frac = 1.0 / MODEL_CFG.num_experts

    print(f"\n{'='*62}")
    print(f"Expert Load Distribution  ({MODEL_CFG.num_experts} experts, top-{MODEL_CFG.top_k})")
    print(f"{'='*62}")
    print(f"  {'Expert':>6}  {'No aux loss':>12}  {'With aux loss':>14}  {'Ideal':>8}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*14}  {'-'*8}")
    for i, (b, a) in enumerate(zip(s_base["fracs"], s_bal["fracs"])):
        print(f"  {i:>6}  {b:>11.1%}  {a:>13.1%}  {ideal_frac:>7.1%}")

    print(f"\n  Coefficient of Variation (lower = more balanced):")
    print(f"    No aux loss   : {s_base['cv']:.4f}")
    print(f"    With aux loss : {s_bal['cv']:.4f}")

    cv_improvement = (s_base["cv"] - s_bal["cv"]) / s_base["cv"] * 100

    print(f"\n  Normalized Entropy (higher = more balanced):")
    print(f"    No aux loss   : {s_base['entropy']:.4f}")
    print(f"    With aux loss : {s_bal['entropy']:.4f}")

    ent_improvement = (s_bal["entropy"] - s_base["entropy"]) / s_base["entropy"] * 100

    print(f"\n  Max expert load fraction:")
    print(f"    No aux loss   : {s_base['max_frac']:.3f}")
    print(f"    With aux loss : {s_bal['max_frac']:.3f}")

    print(f"\n  CV improvement     : {cv_improvement:+.1f}%")
    print(f"  Entropy improvement: {ent_improvement:+.1f}%")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
