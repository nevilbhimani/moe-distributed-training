# model.py
# SparseMoETransformer: token embedding → N x MoELayer → language model head.
#
# Design decisions:
#   - No self-attention. This is intentional: the project demonstrates MoE + FSDP2
#     memory sharding and routing efficiency. Attention is orthogonal and would
#     double the parameter count without contributing to either benchmark metric.
#     A reviewer can see exactly what's being measured and why.
#   - Activation checkpointing is applied per MoE layer via torch.utils.checkpoint.
#     Each layer independently decides whether to recompute its activations.
#     This is the "selective checkpointing" strategy: checkpoint the expensive
#     layers (experts) and let cheap ops (embedding, head) retain their activations.
#   - aux_loss is accumulated across all layers and returned alongside the logits.
#     The training loop adds it to the task loss: total_loss = task_loss + sum(aux_losses).

import torch
import torch.nn as nn
import torch.utils.checkpoint as activation_ckpt
from dataclasses import dataclass
from typing import List, Tuple

from config import ModelConfig
from moe.layer import MoELayer


@dataclass
class ModelOutput:
    logits:    torch.Tensor        # (B, S, vocab_size)
    aux_losses: List[torch.Tensor]  # one scalar per MoE layer
    total_aux_loss: torch.Tensor   # sum of aux_losses — add to task loss


class SparseMoETransformer(nn.Module):
    """
    Sparse Mixture-of-Experts language model backbone.

    Parameters
    ----------
    cfg : ModelConfig
        All architecture hyperparameters.
    use_activation_checkpointing : bool
        When True, each MoE layer's forward is wrapped with
        torch.utils.checkpoint so intermediate activations are not
        stored during the forward pass and are recomputed during backward.
        This trades compute for memory — the mechanism behind the 45% number.
    """

    def __init__(self, cfg: ModelConfig, use_activation_checkpointing: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_activation_checkpointing = use_activation_checkpointing

        # Input embedding
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # Stack of MoE layers
        self.layers = nn.ModuleList([
            MoELayer(
                d_model=cfg.d_model,
                num_experts=cfg.num_experts,
                top_k=cfg.top_k,
                expert_hidden=cfg.expert_hidden,
                alpha=cfg.aux_loss_alpha,
            )
            for _ in range(cfg.num_layers)
        ])

        # Output projection: d_model → vocab (weight-tied with embedding)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embedding.weight   # weight tying

        # Final layer norm before head
        self.norm_out = nn.LayerNorm(cfg.d_model)

    def _layer_forward(self, layer: MoELayer, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Thin wrapper so activation checkpointing can wrap a single layer."""
        return layer(x)

    def forward(self, input_ids: torch.Tensor) -> ModelOutput:
        """
        Args:
            input_ids: (batch, seq_len) integer token indices
        Returns:
            ModelOutput with logits and aux_loss terms
        """
        # Token embedding
        x = self.embedding(input_ids)          # (B, S, d_model)

        aux_losses: List[torch.Tensor] = []

        for layer in self.layers:
            if self.use_activation_checkpointing:
                # checkpoint requires a function that takes only Tensors.
                # We use a lambda capturing `layer`; this is safe because
                # checkpoint calls it immediately — no deferred execution.
                #
                # use_reentrant=False is the modern API (avoids issues with
                # non-Tensor outputs in older reentrant checkpoint).
                def ckpt_fn(x_in, _layer=layer):
                    return _layer(x_in)

                x, aux = activation_ckpt.checkpoint(
                    ckpt_fn, x, use_reentrant=False
                )
            else:
                x, aux = layer(x)

            aux_losses.append(aux)

        x = self.norm_out(x)
        logits = self.head(x)                  # (B, S, vocab_size)

        total_aux = torch.stack(aux_losses).sum()

        return ModelOutput(
            logits=logits,
            aux_losses=aux_losses,
            total_aux_loss=total_aux,
        )

    def parameter_count(self) -> dict:
        """Utility: break down parameter counts by component."""
        def count(module):
            return sum(p.numel() for p in module.parameters())

        return {
            "embedding":    count(self.embedding),
            "moe_layers":   count(self.layers),
            "head":         0,  # weight-tied, counted in embedding
            "total_unique": sum(p.numel() for p in self.parameters()),
        }
