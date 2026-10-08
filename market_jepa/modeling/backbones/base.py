"""Abstract base class for time series backbones."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class TimeSeriesBackbone(nn.Module, ABC):
    """Abstract base class for time series backbones.

    All backbones must implement forward() which takes variable-length
    time series data and produces fixed-size embeddings.

    Args:
        n_features: Number of input features per timestep.
        d_embedding: Output embedding dimension.
    """

    def __init__(self, n_features: int, d_embedding: int):
        super().__init__()
        self.n_features = n_features
        self.d_embedding = d_embedding

    @abstractmethod
    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Process time series data.

        Args:
            x: Input tensor of shape (batch, n_features, length).
            lengths: Optional tensor of shape (batch,) indicating actual sequence
                    lengths (for padded batches).

        Returns:
            Embeddings of shape (batch, d_embedding).
        """
        pass
