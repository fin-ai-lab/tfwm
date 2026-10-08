"""Shared modeling utilities: RMSNorm."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm is equivalent to T5LayerNorm.

    Constructor signature ``RMSNorm(hidden_size)`` is compatible with
    ``torchvision.ops.MLP``'s ``norm_layer(channels)`` call pattern.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
