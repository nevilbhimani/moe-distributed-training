# config.py
# Single source of truth for all model and training hyperparameters.
# Sized deliberately: large enough that FSDP2 sharding matters,
# small enough to run on a CPU dev machine during iteration.

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    # Vocabulary and sequence
    vocab_size: int = 4096
    seq_len: int = 128
    d_model: int = 512          # token embedding dimension

    # MoE layer geometry
    num_experts: int = 8        # total expert FFN modules per MoE layer
    top_k: int = 2              # experts activated per token
    expert_hidden: int = 1024   # inner dimension of each expert FFN

    # Transformer depth
    num_layers: int = 4         # number of MoE layers stacked

    # Routing
    aux_loss_alpha: float = 0.01  # load-balance loss coefficient


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 8
    seq_len: int = 128
    num_steps: int = 20         # short loop for dev/benchmark purposes
    lr: float = 1e-3
    seed: int = 42


@dataclass(frozen=True)
class BenchmarkConfig:
    # Memory benchmark (GPU/Colab)
    mem_batch_size: int = 16
    mem_seq_len: int = 256
    mem_warmup_steps: int = 3
    mem_measure_steps: int = 10

    # Routing benchmark (CPU/M1)
    routing_batch_size: int = 32
    routing_seq_len: int = 128
    routing_steps: int = 50     # batches to accumulate expert counts over


# Module-level singletons — import these everywhere
MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()
BENCH_CFG = BenchmarkConfig()
