# moe/experts.py
# A single Expert: a two-layer FFN with SwiGLU activation.
#
# Design decision: experts are completely isolated from routing logic.
# The MoELayer owns dispatch; ExpertFFN just transforms a token tensor.
# This mirrors how production MoE code (e.g., Mixtral) separates concerns.
#
# SwiGLU: output = (W1·x ⊙ σ(W3·x)) · W2
# Chosen over vanilla ReLU-FFN because it's what modern MoE LLMs actually use.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    """
    Single sparse expert: SwiGLU FFN.

    Args:
        d_model:  input/output dimension (matches token embedding dim)
        d_hidden: inner projection dimension
    """

    def __init__(self, d_model: int, d_hidden: int) -> None:
        super().__init__()
        # Gate and value projections share the same input — SwiGLU pattern
        self.w_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.w_val  = nn.Linear(d_model, d_hidden, bias=False)
        self.w_out  = nn.Linear(d_hidden, d_model, bias=False)

        # Weight init: scaled normal so expert outputs start near unit variance
        for layer in (self.w_gate, self.w_val, self.w_out):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (num_tokens_routed_here, d_model)
               Only the tokens assigned to this expert are passed in.
        Returns:
            (num_tokens_routed_here, d_model)
        """
        gate = F.silu(self.w_gate(x))   # (T, d_hidden)
        val  = self.w_val(x)             # (T, d_hidden)
        hidden = gate * val              # SwiGLU gating
        return self.w_out(hidden)        # (T, d_model)
