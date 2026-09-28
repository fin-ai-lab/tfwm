"""LeJEPA (SIGReg) model for time series self-supervised learning."""


from market_jepa.backbone_config import backbone_block
import json
import os
import warnings
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import MLP

from .base import TrainingModel
from ..backbones import backbone_kwargs_from_state_dict, create_backbone
from .utils import RMSNorm


# ---------------------------------------------------------------------------
# SIGReg loss (from LeJEPA, Balestriero & LeCun 2025)
# Reference: https://arxiv.org/abs/2511.08544
# ---------------------------------------------------------------------------


class SIGReg(nn.Module):
    """Signature-based Regularization for JEPA models.

    Uses characteristic function estimation to encourage diverse embeddings
    while avoiding collapse. The implementation leverages the symmetric
    property of the ECF/CF for improved quadrature efficiency.

    Args:
        knots: Number of quadrature points for integration.
        t_max: Maximum value for integration range [0, t_max].
        n_projections: Number of random projections for CF estimation.
    """

    def __init__(self, knots: int = 17, t_max: float = 3.0, n_projections: int = 256):
        super().__init__()
        self._n_projections = n_projections

        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg loss.

        Args:
            proj: Projected embeddings of shape (n_views, batch, proj_dim) or
                  (batch, n_views, proj_dim). Will be normalized internally.

        Returns:
            Scalar SIGReg loss value.
        """
        if proj.dim() == 3 and proj.size(0) > proj.size(1):
            proj = proj.transpose(0, 1)

        A = torch.randn(proj.size(-1), self._n_projections, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))

        x_t = (proj @ A).unsqueeze(-1) * self.t.to(proj.dtype)
        err = (x_t.cos().mean(-3) - self.phi.to(proj.dtype)).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights.to(proj.dtype)) * proj.size(-2)

        return statistic.mean()


class LeJEPA(TrainingModel):
    """LeJEPA (SIGReg) class for time series self-supervised learning.

    Uses interchangeable backbones with SIGReg loss from LeJEPA.

    Input: (batch_size, n_views, n_features, length)
        Note: length can vary per view, handled via nested tensors or padding.

    Output: (batch, n_views, d_embedding)

    Args:
        backbone: A TimeSeriesBackbone instance.
        proj_dim: Dimension of the projection head output.
        proj_hidden: Hidden dimensions for the projection MLP.
        lamb: Lambda parameter balancing SIGReg vs invariance loss.
            PER-PAIRING, and no single value is right for every
            augmentation -- see LeJEPAModeConfig.lamb in schemas.py for
            the swept table (k2 0.2, k2ind 0.1, rrc 0.05, time_warp
            0.001, gaussian_noise 0.3). This default matches that config
            so the two cannot drift; pin lambda explicitly instead of
            inheriting either.
    """

    def __init__(
        self,
        backbone: nn.Module,
        proj_dim: int | None = 128,
        proj_hidden: list[int] | None = None,
        n_projections: int = 256,
        lamb: float = 0.01,
        gradient_checkpointing: bool = False,
    ):
        if gradient_checkpointing:
            raise ValueError(
                "gradient_checkpointing is not supported on LeJEPA. "
                "Use IJEPA for gradient checkpointing support."
            )
        super().__init__()
        self.backbone = backbone
        self.lamb = lamb
        self.d_embedding = backbone.d_embedding

        # Projection head (MLP) — None means SIGReg operates on raw embeddings
        if proj_dim is not None:
            if proj_hidden is None:
                proj_hidden = [backbone.d_embedding * 4, backbone.d_embedding * 4]
            self.proj = MLP(
                backbone.d_embedding,
                [*proj_hidden, proj_dim],
                norm_layer=RMSNorm,
            )
        else:
            self.proj = None

        # SIGReg loss
        self.sigreg = SIGReg(n_projections=n_projections)

    def encode(self, x, lengths=None):
        """Run backbone + projection head, return embeddings and projections."""
        result = super().encode(x, lengths)
        embeddings = result["embeddings"]
        if self.proj is not None:
            batch_size, n_views = embeddings.shape[:2]
            flat_proj = self.proj(embeddings.reshape(-1, self.d_embedding))
            result["projections"] = flat_proj.reshape(batch_size, n_views, -1)
        else:
            result["projections"] = embeddings
        return result

    def compute_loss(self, projections, n_global_views=None, pair_weights=None):
        """Compute SIGReg + invariance loss on the full batch of projections.

        ``pair_weights`` (batch, n_views, n_views) turns the invariance term
        into a weighted pairwise pull — used by the structured-matching
        cross_stock variant, whose dataset emits a 0/1 edge graph
        (global<->global, matched local<->local, local<->own global). Weights
        are normalized to mean 1 over off-diagonal pairs so an all-ones graph
        reduces exactly to the unweighted form (identity: the mean squared
        deviation from the view mean equals the mean pairwise squared
        distance / (2 K^2)).
        """
        batch_size = projections.shape[0]
        proj_transposed = projections.transpose(0, 1)
        sigreg_loss = self.sigreg(proj_transposed)

        pair_w_mean = None
        if pair_weights is not None:
            w = pair_weights
            n_views = projections.shape[1]
            diff2 = (
                projections.unsqueeze(2) - projections.unsqueeze(1)
            ).square().mean(-1)  # (batch, K, K)
            offdiag = ~torch.eye(n_views, dtype=torch.bool, device=w.device)
            w = w * offdiag
            n_offdiag = batch_size * n_views * (n_views - 1)
            pair_w_mean = w.sum() / n_offdiag
            w = w / pair_w_mean.clamp_min(1e-8)
            inv_loss = (w * diff2).sum() / (2 * n_views**2 * batch_size)
        else:
            if n_global_views is not None and n_global_views > 0:
                global_proj = projections[:, :n_global_views, :]
                mu = global_proj.mean(dim=1, keepdim=True)
            else:
                mu = projections.mean(dim=1, keepdim=True)

            inv_loss = (mu - projections).square().mean()

        lejepa_loss = sigreg_loss * self.lamb + inv_loss * (1 - self.lamb)

        # Batch-normalized versions for comparable logging across batch sizes
        sigreg_normalized = sigreg_loss / batch_size
        lejepa_normalized = sigreg_normalized * self.lamb + inv_loss * (1 - self.lamb)

        result = {
            "lejepa_loss": lejepa_loss,
            "sigreg_loss": sigreg_loss,
            "inv_loss": inv_loss,
            "sigreg_loss_normalized": sigreg_normalized,
            "lejepa_loss_normalized": lejepa_normalized,
        }
        if pair_w_mean is not None:
            result["pair_w_mean"] = pair_w_mean
        return result

    def forward(self, x, lengths=None, return_loss=False, n_global_views=None):
        result = self.encode(x, lengths)

        if return_loss:
            result.update(self.compute_loss(result["projections"], n_global_views=n_global_views))

        return result

    mode_label: str = "LeJEPA"
    mode_str: str = "LeJEPA"
    uses_multi_view: bool = True

    def default_run_name(self, backbone_type, cfg):
        return "__".join(
            [
                "mode=lejepa",
                f"bb={backbone_type}",
                f"d_emb={backbone_block(cfg).d_embedding}",
                f"blr={(cfg.optimizer.blr or cfg.mode.training_overrides.blr):.1e}",
                f"bs={cfg.training.per_device_train_batch_size}",
                f"steps={cfg.training.max_train_steps}",
            ]
        )

    def describe_parameters(self):
        from market_jepa.training.utils import count_parameters

        named = {"backbone": self.backbone}
        if self.proj is not None:
            named["projection"] = self.proj
        param_counts = count_parameters(self, **named)
        lines = [
            f"Data dimensions:\n  n_features: {self.backbone.n_features}",
            f"Model parameters:\n  Backbone: {param_counts['backbone']:,}",
        ]
        if "projection" in param_counts:
            lines.append(f"  Projection: {param_counts['projection']:,}")
        lines.append(f"  Total: {param_counts['total']:,}")
        return param_counts, "\n".join(lines)

    def post_training_step(self, completed_steps, max_train_steps):
        return {}

    @staticmethod
    def _combine_bucket_outputs(outputs, sizes):
        """Batch-size-weighted mixture of per-bucket ``compute_loss`` outputs.

        Buckets (one per augmentation config) may differ in view count and
        n_global_views, so each gets its own loss; the batch loss is the
        sample-weighted mean. A single bucket reduces to its outputs
        unchanged.
        """
        total = float(sum(sizes))
        combined = {
            key: sum(o[key] * (n / total) for o, n in zip(outputs, sizes))
            for key in (
                "lejepa_loss", "sigreg_loss", "inv_loss",
                "sigreg_loss_normalized", "lejepa_loss_normalized",
            )
        }
        pw = [(o["pair_w_mean"], n) for o, n in zip(outputs, sizes)
              if "pair_w_mean" in o]
        if pw:
            combined["pair_w_mean"] = (
                sum(v * n for v, n in pw) / float(sum(n for _, n in pw))
            )
        return combined

    def training_step(self, batch, device, grad_accum_steps=1):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = []
            sizes = []
            for bucket in batch["buckets"]:
                views = [v.to(device) for v in bucket["views"]]
                lengths = [l.to(device) for l in bucket["lengths"]]
                enc = self.encode(views, lengths)
                pw = bucket.get("pair_weights")
                outputs.append(self.compute_loss(
                    enc["projections"],
                    n_global_views=bucket.get("n_global_views"),
                    pair_weights=pw.to(device) if pw is not None else None,
                ))
                sizes.append(enc["projections"].shape[0])
            if not outputs:
                return None
            output = self._combine_bucket_outputs(outputs, sizes)
            loss = output["lejepa_loss"]

        if not torch.isfinite(loss):
            return None

        (loss / grad_accum_steps).backward()

        metrics = {
            "train/loss": output["lejepa_loss_normalized"].item(),
            "train/sigreg_loss": output["sigreg_loss_normalized"].item(),
            "train/inv_loss": output["inv_loss"].item(),
        }
        if "pair_w_mean" in output:
            metrics["train/pair_w_mean"] = output["pair_w_mean"].item()

        return {
            "loss": output["lejepa_loss_normalized"].item(),
            "metrics": metrics,
        }

    @torch.no_grad()
    def eval_step(self, eval_batches, device):
        self.eval()
        losses, sigreg_losses, inv_losses = [], [], []

        for batch in eval_batches:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = []
                sizes = []
                for bucket in batch["buckets"]:
                    views = [v.to(device) for v in bucket["views"]]
                    lengths = [l.to(device) for l in bucket["lengths"]]
                    enc = self.encode(views, lengths)
                    # Must mirror training_step's call exactly, pair_weights
                    # included — otherwise a structured-matching run reports a
                    # val loss computed under a different objective than the
                    # one it trains on, and the two curves are incomparable.
                    pw = bucket.get("pair_weights")
                    outputs.append(self.compute_loss(
                        enc["projections"],
                        n_global_views=bucket.get("n_global_views"),
                        pair_weights=pw.to(device) if pw is not None else None,
                    ))
                    sizes.append(enc["projections"].shape[0])
                if not outputs:
                    continue
                loss_out = self._combine_bucket_outputs(outputs, sizes)

            if not torch.isfinite(loss_out["lejepa_loss"]):
                continue
            losses.append(loss_out["lejepa_loss_normalized"].item())
            sigreg_losses.append(loss_out["sigreg_loss_normalized"].item())
            inv_losses.append(loss_out["inv_loss"].item())

        return {
            "eval/jepa_loss": np.mean(losses) if losses else float("nan"),
            "eval/jepa_sigreg_loss": np.mean(sigreg_losses) if sigreg_losses else float("nan"),
            "eval/jepa_inv_loss": np.mean(inv_losses) if inv_losses else float("nan"),
        }

    def save_pretrained(self, path: str) -> None:
        """Save model weights and architecture config to a directory.

        Creates ``config.json`` (architecture) and ``model.pt`` (weights).
        The config contains everything needed to reconstruct the model
        via :meth:`from_pretrained` without any external knowledge.
        """
        os.makedirs(path, exist_ok=True)

        backbone = self.backbone
        backbone_type = {
            "TransformerBackbone": "transformer",
            "ResNetBackbone": "resnet",
        }.get(type(backbone).__name__)

        model_to_save = self._orig_mod if hasattr(self, "_orig_mod") else self
        torch.save(model_to_save.state_dict(), os.path.join(path, "model.pt"))

        if backbone_type is None:
            warnings.warn(
                f"save_pretrained: no architecture config writer for "
                f"{type(backbone).__name__}; wrote weights only. "
                f"from_pretrained will not work for this checkpoint."
            )
            return

        # Infer proj_dim from the last Linear in the projection head
        # (MLP ends with Linear -> Dropout, so the Linear is at [-2])
        proj_dim = self.proj[-2].out_features if self.proj is not None else None

        config = {
            "backbone_type": backbone_type,
            "n_features": backbone.n_features,
            "d_embedding": backbone.d_embedding,
            "proj_dim": proj_dim,
            "n_projections": self.sigreg._n_projections,
            "lamb": self.lamb,
        }

        if backbone_type == "transformer":
            from ..backbones.transformer import TransformerConfig

            config["pool"] = backbone.pool
            config["backbone_config"] = asdict(backbone.config)
        elif backbone_type == "resnet":
            config["pool"] = backbone.pool
            config["resnet_config"] = {
                "variant": next(
                    k for k, (bc, nb) in backbone.CONFIGS.items()
                    if bc == type(backbone.layer1[0]) and nb == [
                        len(backbone.layer1), len(backbone.layer2),
                        len(backbone.layer3), len(backbone.layer4),
                    ]
                ),
                "initial_channels": backbone.stem[0].out_channels,
            }

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "LeJEPA":
        """Load a model from a directory created by :meth:`save_pretrained`.

        Args:
            path: Directory containing ``config.json`` and ``model.pt``.
            **kwargs: Overrides for config values (e.g. ``lamb=0.1``).

        Returns:
            Loaded model with weights restored (on CPU).
        """
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        config.update(kwargs)

        backbone_config = config.pop("backbone_config", None)
        resnet_config = config.pop("resnet_config", None)
        transformer_config = None
        if backbone_config is not None:
            from ..backbones.transformer import TransformerConfig
            transformer_config = TransformerConfig(**backbone_config)

        extra_kwargs = {}
        if resnet_config is not None:
            extra_kwargs.update(resnet_config)

        _sd = torch.load(os.path.join(path, "model.pt"), map_location="cpu")
        # Knobs no save_pretrained ever recorded -- state_token,
        # diff_channels, n_info_channels -- read off the weights. See
        # backbones.backbone_kwargs_from_state_dict.
        extra_kwargs.update(backbone_kwargs_from_state_dict(_sd))
        backbone = create_backbone(
            backbone_type=config["backbone_type"],
            n_features=extra_kwargs.pop("n_features", config["n_features"]),
            d_embedding=config["d_embedding"],
            config=transformer_config,
            pool=config.get("pool", "cls"),
            **extra_kwargs,
        )
        model = cls(
            backbone=backbone,
            proj_dim=config["proj_dim"],
            n_projections=config.get("n_projections", 256),
            lamb=config["lamb"],
        )

        model.load_state_dict(_sd)
        return model
