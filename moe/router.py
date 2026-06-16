# moe/router.py
# TopKRouter: maps each token to its top-k experts.
#
# Design decisions:
#   - Router is a plain nn.Linear with no bias (following Switch Transformer).
#   - Returns a named dataclass so callers get named fields, not positional tuple unpacking.
#   - aux_loss lives here, computed from the routing distribution, not from a
#     side-channel global variable (contrast: Megatron-LM uses a global accumulator
#     which causes double-counting under activation checkpointing — see GH #1330).
#
# Load-balance loss formula (Fedus et al., Switch Transformers, 2021):
#   aux_loss = alpha * N * sum_i( f_i * P_i )
#
#   where:
#     N    = num_experts
#     f_i  = fraction of tokens dispatched to expert i   (discrete, non-differentiable)
#     P_i  = mean router probability assigned to expert i (differentiable)
#
#   The product f_i * P_i is the key: f_i provides the signal of actual load imbalance,
#   P_i provides the gradient. Together they push the router toward uniform dispatch.

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class RouterOutput:
    dispatch_weights: torch.Tensor   # (B, S, top_k)  — softmax weights for combining expert outputs
    expert_indices:   torch.Tensor   # (B, S, top_k)  — which experts each token goes to
    aux_loss:         torch.Tensor   # scalar          — load-balance penalty (0 if alpha=0)


class TopKRouter(nn.Module):
    """
    Produces per-token expert assignments via top-k selection over router logits.

    Args:
        d_model:     token embedding dimension
        num_experts: total number of experts
        top_k:       experts activated per token
        alpha:       load-balance loss coefficient (0 = disabled)
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int,
        alpha: float = 0.01,
    ) -> None:
        super().__init__()
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) cannot exceed num_experts ({num_experts})")

        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = alpha

        # Single linear projection: token → expert logits
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            RouterOutput with dispatch_weights, expert_indices, aux_loss
        """
        B, S, D = x.shape
        N = self.num_experts

        # --- Step 1: compute router logits ---
        # x_flat: (B*S, d_model)  →  logits: (B*S, N)
        x_flat = x.reshape(B * S, D)
        logits = self.gate(x_flat)                          # (B*S, N)
        probs  = F.softmax(logits, dim=-1)                  # (B*S, N)

        # --- Step 2: top-k selection ---
        topk_vals, topk_idx = torch.topk(probs, self.top_k, dim=-1)  # (B*S, top_k)

        # Re-normalize the selected top-k weights so they sum to 1 per token.
        # This matches Mixtral's routing: only selected expert weights participate.
        dispatch_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True)  # (B*S, top_k)

        # --- Step 3: load-balance auxiliary loss ---
        # f_i: fraction of tokens routed to expert i  (B*S tokens total)
        # Use one-hot of top-1 only for f_i, following Switch Transformer convention.
        # (Using all top-k assignments would double-count; top-1 is the primary routing signal.)
        top1_idx = topk_idx[:, 0]                          # (B*S,)
        one_hot  = F.one_hot(top1_idx, num_classes=N).float()  # (B*S, N)
        f_i = one_hot.mean(dim=0)                          # (N,) — fraction per expert

        # P_i: mean router probability for each expert (over all tokens, all top-k)
        P_i = probs.mean(dim=0)                            # (N,)

        # aux_loss = alpha * N * dot(f_i, P_i)
        aux_loss = self.alpha * N * (f_i * P_i).sum()      # scalar

        return RouterOutput(
            dispatch_weights=dispatch_weights.reshape(B, S, self.top_k),
            expert_indices=topk_idx.reshape(B, S, self.top_k),
            aux_loss=aux_loss,
        )
