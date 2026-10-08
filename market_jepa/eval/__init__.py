"""Shared evaluation and supervised training utilities for market-jepa."""

from .tasks import TaskSpec, TASK_REGISTRY, TARGET_TYPES, HORIZONS
from .heads import RegressionHead, create_prediction_head
from .metrics import rank_ic, grouped_rank_ic, pooled_month_ic, delta_ic

__all__ = [
    "TaskSpec",
    "TASK_REGISTRY",
    "TARGET_TYPES",
    "HORIZONS",
    "RegressionHead",
    "create_prediction_head",
    "rank_ic",
    "grouped_rank_ic",
    "pooled_month_ic",
    "delta_ic",
]
