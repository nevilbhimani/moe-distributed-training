# moe/layer.py
# MoELayer: orchestrates routing, dispatch, expert execution, and output aggregation.
#
# Token dispatch strategy: loop over experts.
#
# Design decision — why loop-over-experts instead of batched scatter/gather:
#   Batched dispatch (e.g., via one-hot matmul) is faster in production but
#   obscures what's happening. Loop-over-experts is readable, debuggable, and
#   makes the per-expert execution path explicit — important for demonstrating
#   that each ExpertFFN truly only processes its assigned tokens.
#
#   The loop is over experts (8), not tokens, so it scales with num_experts, not batch*seq.
#
# Activation checkpointing note:
#   torch.utils.checkpoint.checkpoint() is applied at the MoELayer level in model.py,
#   not here. This layer is checkpointing-agnostic — it just does the forward pass.

import torch
import torch.nn as nn
from typing import Tuple

from moe.experts import ExpertFFN
from moe.router import TopKRouter, RouterOutput


class MoELayer(nn.Module):
    """
    One Mixture-of-Experts layer.

    Each input token is routed to top_k experts. The expert outputs are
    weighted by the router's dispatch weights and summed.

    Args:
        d_model:     token dimension
        num_experts: total expert count
        top_k:       active experts per token
        expert_hidden: inner FFN dimension for each expert
        alpha:       load-balance loss coefficient
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int,
        expert_hidden: int,
        alpha: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = TopKRouter(d_model, num_experts, top_k, alpha)
        self.experts = nn.ModuleList(
            [ExpertFFN(d_model, expert_hidden) for _ in range(num_experts)]
        )
        # Layer norm before routing — stabilizes logit scale during early training
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output:   (batch, seq_len, d_model)
            aux_loss: scalar tensor — load-balance penalty
        """
        B, S, D = x.shape

        # Normalize before routing
        x_norm = self.norm(x)

        # Route: get expert assignments and dispatch weights
        route: RouterOutput = self.router(x_norm)
        # expert_indices:   (B, S, top_k)
        # dispatch_weights: (B, S, top_k)

        # Flatten spatial dims for dispatch
        x_flat      = x_norm.reshape(B * S, D)             # (T, D) where T = B*S
        idx_flat    = route.expert_indices.reshape(B * S, self.top_k)    # (T, top_k)
        weight_flat = route.dispatch_weights.reshape(B * S, self.top_k)  # (T, top_k)

        # Accumulate expert outputs
        output_flat = torch.zeros_like(x_flat)              # (T, D)

        for expert_id in range(self.num_experts):
            # Find which (token, k-slot) pairs were assigned to this expert
            # token_mask: (T, top_k) bool
            token_mask = (idx_flat == expert_id)            # (T, top_k)

            # token_sel: indices of tokens that have at least one slot for this expert
            token_sel = token_mask.any(dim=-1).nonzero(as_tuple=True)[0]  # (n_sel,)

            if token_sel.numel() == 0:
                continue  # expert received no tokens this batch

            # Gather the tokens assigned to this expert
            expert_input = x_flat[token_sel]                # (n_sel, D)

            # Run the expert
            expert_out = self.experts[expert_id](expert_input)  # (n_sel, D)

            # For each selected token, sum across its k-slots that hit this expert.
            # A token can activate the same expert in multiple k-slots (rare with top_k=2,
            # but we handle it correctly).
            weights_for_expert = token_mask[token_sel].float()     # (n_sel, top_k)
            # weight_flat rows for these tokens: (n_sel, top_k)
            slot_weights = weight_flat[token_sel]                  # (n_sel, top_k)
            # Effective weight per token = sum of weights where this expert was selected
            eff_weights = (weights_for_expert * slot_weights).sum(dim=-1, keepdim=True)  # (n_sel, 1)

            output_flat[token_sel] += eff_weights * expert_out

        output = output_flat.reshape(B, S, D)

        # Residual connection: expert output added back to the un-normed input
        return x + output, route.aux_loss
