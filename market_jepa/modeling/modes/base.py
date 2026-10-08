"""Base classes for training modes."""

import inspect
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


def compute_collapse_metrics(
    embeddings: torch.Tensor,
    prefix: str = "train",
) -> dict[str, float]:
    """Compute collapse-monitoring metrics for JEA/JEPA models.

    Args:
        embeddings: Tensor of shape ``(N, D)`` — one embedding per sample.
        prefix: Metric key prefix (e.g. ``"train"`` → ``"train/emb_std"``).

    Returns:
        Dict of scalar metrics suitable for logging.
    """
    if embeddings.shape[0] < 2:
        return {}

    x = embeddings.float()

    # 1. Per-dimension std, averaged across dimensions.
    #    Near-zero → dimensional or full collapse.
    dim_std = x.std(dim=0)  # (D,)
    mean_std = dim_std.mean().item()
    min_std = dim_std.min().item()

    # 2. Mean pairwise cosine similarity.
    #    → 1.0 means all representations identical up to scale.
    x_norm = torch.nn.functional.normalize(x, dim=-1)
    cos_sim = (x_norm @ x_norm.T).fill_diagonal_(0)
    n = x.shape[0]
    mean_cos = cos_sim.sum().item() / (n * (n - 1))

    # 3. Effective rank (via SVD entropy of singular values).
    #    Low eff_rank → few dimensions carry all the variance.
    #    Center before SVD so rank reflects spread, not mean offset.
    x_centered = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x_centered)
    # Normalize singular values to a probability distribution
    p = s / s.sum().clamp(min=1e-12)
    p = p.clamp(min=1e-12)
    eff_rank = torch.exp(-(p * p.log()).sum()).item()

    return {
        f"{prefix}/repr_std": mean_std,
        f"{prefix}/repr_std_min": min_std,
        f"{prefix}/repr_cos_sim": mean_cos,
        f"{prefix}/repr_eff_rank": eff_rank,
    }


class TrainingModel(nn.Module, ABC):
    """Abstract base class for trainable model wrappers.

    All training models (LeJEPA, IJEPA, SupervisedModel) must subclass this.
    The training harness depends on every method and attribute listed here.

    Required class-level attributes (checked at class definition time):
        uses_multi_view (bool): Whether the model expects multi-view batches.
            LeJEPA uses multi-view; IJEPA and supervised do not.
        mode_label (str): Short label for logging headers (e.g. "LeJEPA",
            "I-JEPA"). May be a ``@property`` for dynamic labels.
        mode_str (str): Descriptive string for the training mode, used in
            run summaries and config dumps. May be a ``@property``.

    Required instance attributes (set in ``__init__``):
        backbone (TimeSeriesBackbone): The underlying backbone encoder.
        d_embedding (int): Output embedding dimension of the backbone.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Skip checks on intermediate abstract subclasses
        if inspect.isabstract(cls):
            return
        for attr in ("mode_label", "mode_str", "uses_multi_view"):
            has_attr = any(
                attr in vars(c)
                for c in cls.__mro__
                if c not in (TrainingModel, object)
            )
            if not has_attr:
                raise TypeError(
                    f"{cls.__name__} must define '{attr}' as a class attribute "
                    f"or property to satisfy the TrainingModel interface"
                )

    _instance_attrs_validated = False

    def _validate_instance_attrs(self):
        """Check that subclass __init__ set required instance attributes (once)."""
        if self._instance_attrs_validated:
            return
        for attr in ("backbone", "d_embedding"):
            if not hasattr(self, attr):
                raise AttributeError(
                    f"{type(self).__name__}.__init__ must set self.{attr}"
                )
        self._instance_attrs_validated = True

    @abstractmethod
    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        """Run the model forward pass. All subclasses must implement this."""
        ...

    def encode(
        self,
        x: torch.Tensor | list[torch.Tensor],
        lengths: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Produce embeddings from input data (no masking, no loss).

        Used by probe evaluation to extract frozen backbone representations.
        Subclasses may override to add extra outputs (e.g. projections).

        Args:
            x: Either a single tensor (batch, n_features, length), a
                multi-view tensor (batch, n_views, n_features, length), or a
                list of per-view tensors.
            lengths: Optional sequence lengths matching the view structure.

        Returns:
            Dictionary containing at minimum
            ``{"embeddings": (batch, n_views, d)}``.
        """
        self._validate_instance_attrs()
        if isinstance(x, torch.Tensor) and x.dim() == 4:
            views = [x[:, v, :, :] for v in range(x.shape[1])]
            view_lengths = [lengths] * x.shape[1] if lengths is not None else [None] * x.shape[1]
        elif isinstance(x, list):
            views = x
            view_lengths = lengths if lengths is not None else [None] * len(views)
        else:
            views = [x]
            view_lengths = [lengths]

        # Fast path: when every view has the same shape, run one batched
        # backbone call instead of one call per view. Essential for the
        # cross_stock augmentation, where a group can hold hundreds of
        # equal-length views (per-view calls would run at batch size
        # n_groups, which can be 1). Gated to the transformer backbone,
        # which is batch-independent (LayerNorm, per-sample attention) —
        # ResNet's BatchNorm would change semantics under concatenation.
        if (
            len(views) > 1
            and type(self.backbone).__name__ == "TransformerBackbone"
            and all(isinstance(v, torch.Tensor) and v.shape == views[0].shape for v in views)
        ):
            stacked = torch.cat(views, dim=0)  # (n_views * batch, F, L)
            if all(vl is not None for vl in view_lengths):
                stacked_lengths = torch.cat(list(view_lengths), dim=0)
            else:
                stacked_lengths = None
            flat = self.backbone(stacked, stacked_lengths)  # (n_views * batch, d)
            embeddings = flat.reshape(len(views), views[0].shape[0], -1).transpose(0, 1)
            return {"embeddings": embeddings}

        embeddings = [self.backbone(view, vl) for view, vl in zip(views, view_lengths)]
        return {"embeddings": torch.stack(embeddings, dim=1)}

    @abstractmethod
    def describe_parameters(self) -> tuple[dict[str, int], str]:
        """Return parameter counts and a human-readable summary.

        Returns:
            (param_counts, summary) where param_counts is a dict mapping
            component names to parameter counts (must include 'total'),
            and summary is a formatted multi-line string for logging.
        """
        ...

    @abstractmethod
    def training_step(
        self, batch: dict, device: torch.device, grad_accum_steps: int = 1,
    ) -> dict[str, object] | None:
        """Run one training step: forward + loss + backward for each bucket.

        Gradients are accumulated but NOT stepped — the caller handles
        ``optimizer.step()``, ``scheduler.step()``, and ``zero_grad()``.

        Args:
            batch: Batch dict with ``"buckets"`` list.
            device: Target device.
            grad_accum_steps: Divide loss by this for gradient accumulation.

        Returns:
            Dict with ``"loss"`` (float for logging) and ``"metrics"``
            (dict of loggable scalars), or None if the batch was empty/invalid.
        """
        ...

    @abstractmethod
    def eval_step(
        self, eval_batches: list[dict], device: torch.device,
    ) -> dict[str, float]:
        """Compute evaluation metrics on pre-cached batches.

        The caller is responsible for calling ``model.train()`` afterwards.

        Args:
            eval_batches: List of batch dicts.
            device: Target device.

        Returns:
            Dict of metric name → value for logging.
        """
        ...

    @abstractmethod
    def default_run_name(self, backbone_type: str, cfg) -> str:
        """Generate a default W&B run name for this training mode.

        Called when ``cfg.wandb.run_name`` is None.

        Args:
            backbone_type: Short backbone identifier (e.g. "transformer", "resnet").
            cfg: Top-level Config object with optimizer/training fields.

        Returns:
            Human-readable run name string.
        """
        ...

    @abstractmethod
    def post_training_step(
        self, completed_steps: int, max_train_steps: int,
    ) -> dict[str, float]:
        """Hook called after each optimizer step (e.g. EMA update).

        Args:
            completed_steps: Number of optimizer steps completed so far.
            max_train_steps: Total number of optimizer steps planned.

        Returns:
            Dict of extra metrics to log (empty dict if none).
        """
        ...
