"""Tests for multi-task shared-backbone gradient normalization."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from market_jepa.modeling.modes.supervised import MultiTaskSupervisedModel


TASKS = [
    "return_900",
    "volatility_change_900",
    "spread_change_900",
]

CPU = torch.device("cpu")


class _ToyBackbone(torch.nn.Module):
    d_embedding = 4

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, self.d_embedding)

    def forward(self, x, lengths=None):
        return self.proj(x.mean(dim=-1))


def _configured_model(task_weights=None, tasks=TASKS) -> MultiTaskSupervisedModel:
    # Seed here rather than relying on ambient RNG state: these assertions read
    # gradient magnitudes, which depend on the init, so an unseeded model makes
    # the file order-dependent within a full-suite run.
    torch.manual_seed(31)
    model = MultiTaskSupervisedModel(
        _ToyBackbone(),
        tasks,
        task_weights=task_weights,
        gradient_norm_ema_decay=0.0,
        gradient_norm_min=1.0e-8,
        gradient_norm_max_scale=1.0e6,
    )
    model.target_col_idx = {task: idx for idx, task in enumerate(tasks)}
    return model


def _batch() -> dict:
    generator = torch.Generator().manual_seed(73)
    x = torch.randn(10, 2, 3, generator=generator)
    targets = torch.tensor(
        [
            [-0.9, -0.4, 0.0],
            [-0.5, 0.1, 0.7],
            [-0.1, 0.5, -0.8],
            [0.3, 0.9, -0.3],
            [0.8, -0.8, 0.4],
            [-0.7, -0.1, 0.9],
            [-0.3, 0.3, -0.5],
            [0.1, 0.8, -0.1],
            [0.5, -0.5, 0.3],
            [0.9, 0.0, 0.8],
        ],
        dtype=torch.float32,
    )
    return {
        "buckets": [{
            "views": [x],
            "lengths": [torch.full((len(x),), x.shape[-1], dtype=torch.long)],
            "targets": targets,
        }]
    }


def _flatten_gradients(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([
        parameter.grad.detach().flatten()
        for parameter in module.parameters()
        if parameter.grad is not None
    ])


def _head_output_gradient_norms(
    model: MultiTaskSupervisedModel,
    batch: dict,
) -> list[float]:
    bucket = batch["buckets"][0]
    embeddings = model.backbone(bucket["views"][0], bucket["lengths"][0])
    norms = []
    for task in TASKS:
        predictions = model.heads[task](embeddings)
        targets = bucket["targets"][:, model.target_col_idx[task]]
        valid = ~torch.isnan(targets)
        # cells=None keeps the FLAT path this helper has always measured;
        # the arguments are unmasked because a within-cell mask is built over
        # the whole batch, so _per_task_loss does its own masking now.
        loss, _ = model._per_task_loss(
            predictions,
            targets,
            valid,
            None,
            model.task_specs[task],
            task,
        )
        output_grad = torch.autograd.grad(loss, predictions, retain_graph=True)[0]
        norms.append(float(output_grad.float().norm()))
    return norms


def test_configured_weights_set_normalized_shared_gradient_contributions():
    weights = {TASKS[0]: 1.0, TASKS[1]: 2.0, TASKS[2]: 3.0}
    model = _configured_model(weights)
    expected_head_output_norms = _head_output_gradient_norms(model, _batch())
    result = model.training_step(_batch(), CPU)

    assert result is not None
    metrics = result["metrics"]
    for task_idx, (task, expected_weight) in enumerate(
        zip(TASKS, (1 / 6, 2 / 6, 3 / 6))
    ):
        norm = metrics[f"train/task_gradient_norm_{task}"]
        scale = metrics[f"train/task_gradient_scale_{task}"]
        assert metrics[f"train/task_weight_{task}"] == pytest.approx(expected_weight)
        # ema_decay=0 → the EMA is the current norm, so the rescaled task
        # gradient entering the trunk has exactly the configured magnitude.
        assert norm * scale == pytest.approx(expected_weight, rel=1e-5)
        assert metrics[f"train/task_gradient_contribution_norm_{task}"] == (
            pytest.approx(expected_weight, rel=1e-5)
        )
        assert metrics[f"train/task_head_output_gradient_norm_{task}"] == (
            pytest.approx(expected_head_output_norms[task_idx], rel=1e-5)
        )

    per_task = [metrics[f"train/loss_{task}"] for task in TASKS]
    assert result["loss"] == pytest.approx(sum(per_task) / len(TASKS))


def test_mismatched_loss_scales_are_equalized():
    """Heads reach the trunk with equal magnitude despite unequal residuals.

    Every head now runs the same regression loss on a cross-sectional z-score,
    so the scales agree by construction in normal operation. The imbalance is
    therefore INDUCED — one target is blown up 200x — because that is what the
    normalizer has to survive: a task whose residuals are far larger (a badly
    scaled anchor table, or an early-training head) must not dominate the
    shared backbone.
    """
    batch = _batch()
    batch["buckets"][0]["targets"][:, 0] *= 200.0
    model = _configured_model()
    metrics = model.training_step(batch, CPU)["metrics"]

    norms = [metrics[f"train/task_gradient_norm_{task}"] for task in TASKS]
    assert norms[0] > 10 * max(norms[1:])
    for task in TASKS:
        assert metrics[f"train/task_gradient_contribution_norm_{task}"] == (
            pytest.approx(1 / len(TASKS), rel=1e-5)
        )


def test_task_weights_change_backbone_but_not_independent_head_gradients():
    torch.manual_seed(19)
    equal = _configured_model()
    return_only = _configured_model({TASKS[0]: 1.0, TASKS[1]: 0.0, TASKS[2]: 0.0})
    return_only.load_state_dict(copy.deepcopy(equal.state_dict()))

    assert equal.training_step(_batch(), CPU) is not None
    assert return_only.training_step(_batch(), CPU) is not None

    assert not torch.allclose(
        _flatten_gradients(equal.backbone),
        _flatten_gradients(return_only.backbone),
    )
    for task in TASKS:
        torch.testing.assert_close(
            _flatten_gradients(equal.heads[task]),
            _flatten_gradients(return_only.heads[task]),
        )
    # Zero-weighted tasks still train their own heads.
    assert _flatten_gradients(return_only.heads[TASKS[1]]).norm() > 0
    assert _flatten_gradients(return_only.heads[TASKS[2]]).norm() > 0


def test_missing_task_labels_are_skipped():
    batch = _batch()
    batch["buckets"][0]["targets"][:, 2] = float("nan")
    model = _configured_model()
    metrics = model.training_step(batch, CPU)["metrics"]

    assert f"train/loss_{TASKS[2]}" not in metrics
    assert f"train/task_gradient_scale_{TASKS[2]}" not in metrics
    assert f"train/loss_{TASKS[0]}" in metrics
    assert _flatten_gradients(model.backbone).norm() > 0
    # The head of the fully-masked task receives no gradient at all.
    assert all(p.grad is None for p in model.heads[TASKS[2]].parameters())


def test_all_labels_missing_returns_none():
    batch = _batch()
    batch["buckets"][0]["targets"][:] = float("nan")
    assert _configured_model().training_step(batch, CPU) is None


def test_ema_smooths_the_normalizer_across_steps():
    torch.manual_seed(31)
    model = MultiTaskSupervisedModel(
        _ToyBackbone(), TASKS, gradient_norm_ema_decay=0.9,
    )
    model.target_col_idx = {task: idx for idx, task in enumerate(TASKS)}

    first = model.training_step(_batch(), CPU)["metrics"]
    for task in TASKS:
        # First step initializes the EMA to the observed norm.
        assert first[f"train/task_gradient_norm_ema_{task}"] == pytest.approx(
            first[f"train/task_gradient_norm_{task}"], rel=1e-6
        )

    second = model.training_step(_batch(), CPU)["metrics"]
    for task in TASKS:
        raw = second[f"train/task_gradient_norm_{task}"]
        previous = first[f"train/task_gradient_norm_ema_{task}"]
        assert second[f"train/task_gradient_norm_ema_{task}"] == pytest.approx(
            0.9 * previous + 0.1 * raw, rel=1e-5
        )


def test_frozen_backbone_skips_normalization():
    """No trainable shared parameters -> the heads take plain gradients.

    The backbone is frozen BY HAND here. It used to be reachable from a config
    (``freeze_backbone``), retired 2026-09-16 with the rest of the head-only
    arm; the branch it exercises survives as a guard for exactly this kind of
    caller, so the test constructs that state directly.
    """
    model = MultiTaskSupervisedModel(_ToyBackbone(), TASKS)
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    model.backbone.eval()
    model.target_col_idx = {task: idx for idx, task in enumerate(TASKS)}

    metrics = model.training_step(_batch(), CPU)["metrics"]
    assert not any(k.startswith("train/task_gradient") for k in metrics)
    assert all(p.grad is None for p in model.backbone.parameters())
    for task in TASKS:
        assert _flatten_gradients(model.heads[task]).norm() > 0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"tasks": [TASKS[0], TASKS[0]]}, "unique"),
        ({"task_weights": {"not_a_task": 1.0}}, "not present"),
        ({"task_weights": {TASKS[0]: -1.0}}, "non-negative"),
        ({"task_weights": {task: 0.0 for task in TASKS}}, "at least one"),
        ({"gradient_norm_ema_decay": 1.0}, "ema_decay"),
        ({"gradient_norm_min": 0.0}, "gradient_norm_min"),
        ({"gradient_norm_max_scale": 0.0}, "gradient_norm_max_scale"),
    ],
)
def test_invalid_gradient_balancing_configuration_is_rejected(kwargs, match):
    call_kwargs = {"tasks": TASKS, **kwargs}
    with pytest.raises(ValueError, match=match):
        MultiTaskSupervisedModel(_ToyBackbone(), **call_kwargs)
