"""TS2Vec mode for time-series self-supervised learning.

Implements Yue et al. (AAAI 2022, "TS2Vec: Towards Universal Representation
of Time Series") adapted to this codebase's patch-level ViT encoder:

  1. Two overlapping random crops of each sequence (per-sample offsets,
     shared crop geometry within a bucket) are encoded with
     ``backbone.forward_patches``.
  2. TS2Vec's binomial *timestamp masking* becomes binomial *patch masking*:
     masked patches have their content embedding zeroed (position embedding
     kept) via ``forward_patches(zero_mask=...)``.
  3. The hierarchical contrastive loss (instance + temporal terms, max-pooled
     by 2 per level) is applied to the per-patch latents of the overlap
     region — a verbatim port of ``models/losses.py`` from the official repo
     (github.com/zhihanyue/ts2vec), operating on patches instead of raw
     timestamps.
  4. Following the original's use of ``torch.optim.swa_utils.AveragedModel``,
     an equal-weight running average of the backbone is maintained and used
     for downstream ``encode()`` (mirrors BYOL/DINO evaluating the EMA
     teacher).

Deviations from the official repo (documented, all forced by the harness):
  - Encoder is the shared ``TransformerBackbone`` (like the CPC/MAE/DINO/BYOL
    baselines here), not the dilated-conv TSEncoder; the loss operates at
    patch (not timestamp) granularity.
  - Optimizer/schedule come from the harness (AdamW + cosine) rather than
    plain AdamW without a schedule.
  - Crops are sampled inside the valid region (no NaN-padded overhang).
  - Downstream embeddings use the backbone's pooled ``forward`` (config
    ``pool="max"`` mirrors the paper's full-series max-pooling protocol).
  - Crop encoding uses *within-crop* position embeddings
    (``relative_pos=True``), not absolute ones gathered at the global patch
    index — see ``_encode_crop``. The original's dilated-conv encoder has no
    absolute position input, so its contrast can't be solved by position
    matching; absolute codes on the shared ViT reintroduce exactly that
    shortcut.

Only supports ``TransformerBackbone`` (same restriction as I-JEPA / MAE / CPC).
"""

from __future__ import annotations

from market_jepa.backbone_config import backbone_block

import json
import os
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import backbone_kwargs_from_state_dict, create_backbone
from ..backbones.transformer import TransformerBackbone
from .base import TrainingModel, compute_collapse_metrics


# ---------------------------------------------------------------------------
# Hierarchical contrastive loss — port of ts2vec/models/losses.py
# ---------------------------------------------------------------------------


def instance_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Contrast samples against each other at every time step."""
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    return (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2


def temporal_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Contrast time steps against each other within every sample."""
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    return (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2


def hierarchical_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    alpha: float = 0.5,
    temporal_unit: int = 0,
) -> torch.Tensor:
    """Instance + temporal contrast at every scale of a max-pool pyramid."""
    loss = torch.tensor(0.0, device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d


class TS2Vec(TrainingModel):
    """TS2Vec: hierarchical contrastive learning over overlapping crops.

    Args:
        backbone: A ``TransformerBackbone`` with ``pool != "cls"``.
        alpha: Weight of the instance term (1 - alpha goes to the temporal
            term) in the hierarchical loss.
        temporal_unit: Skip temporal contrast below this pyramid level (the
            original's knob for long sequences).
        mask_p: Binomial keep-probability for patch masking (the original's
            ``p=0.5`` binomial timestamp mask).
        swa: Maintain the equal-weight parameter average of the backbone used
            by the original for inference; ``encode()`` evaluates it.
        gradient_checkpointing: Enable gradient checkpointing on the encoder.
    """

    mode_label: str = "TS2Vec"
    mode_str: str = "TS2Vec"
    uses_multi_view: bool = False

    def __init__(
        self,
        backbone: TransformerBackbone,
        alpha: float = 0.5,
        temporal_unit: int = 0,
        mask_p: float = 0.5,
        swa: bool = True,
        gradient_checkpointing: bool = False,
    ):
        if not isinstance(backbone, TransformerBackbone):
            raise ValueError(
                f"TS2Vec only supports TransformerBackbone, got {type(backbone).__name__}"
            )
        if backbone.pool == "cls":
            raise ValueError(
                "TS2Vec requires pool in {'mean', 'max', 'last'} (not 'cls'). "
                "forward_patches operates on raw patch positions without a CLS token."
            )
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not (0.0 < mask_p <= 1.0):
            raise ValueError(f"mask_p must be in (0, 1], got {mask_p}")
        if temporal_unit < 0:
            raise ValueError(f"temporal_unit must be >= 0, got {temporal_unit}")

        super().__init__()
        self.backbone = backbone
        self.d_embedding = backbone.d_embedding
        self.patch_size = backbone.patch_size
        self.n_features = backbone.n_features
        self.alpha = alpha
        self.temporal_unit = temporal_unit
        self.mask_p = mask_p
        self.swa = swa

        if gradient_checkpointing:
            backbone.gradient_checkpointing = True

        if swa:
            # Equal-weight running average (AveragedModel semantics):
            # avg <- avg + (param - avg) / (n_updates + 1)
            import copy

            self.swa_backbone = copy.deepcopy(backbone)
            self.swa_backbone.gradient_checkpointing = False
            for p in self.swa_backbone.parameters():
                p.requires_grad = False
            self.register_buffer("_swa_n", torch.zeros((), dtype=torch.long))
        else:
            self.swa_backbone = None

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------

    def _sample_crops(
        self, B: int, n_valid: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        """Sample TS2Vec's two overlapping crop windows in patch units.

        Port of the fit() crop sampling: overlap ``[crop_left, crop_right)``
        of length ``crop_l``; view 1 spans ``[crop_eleft, crop_right)``, view
        2 spans ``[crop_left, crop_eright)``; a per-sample offset shifts both
        windows together within the valid region.

        Returns (idx1, idx2, crop_l) where idx1/idx2 are (B, len_i) patch
        index tensors, or None when the bucket is too short.
        """
        min_span = 2 ** (self.temporal_unit + 1)
        if n_valid < min_span:
            return None

        crop_l = int(torch.randint(min_span, n_valid + 1, (1,)).item())
        crop_left = int(torch.randint(0, n_valid - crop_l + 1, (1,)).item())
        crop_right = crop_left + crop_l
        crop_eleft = int(torch.randint(0, crop_left + 1, (1,)).item())
        crop_eright = int(torch.randint(crop_right, n_valid + 1, (1,)).item())
        # Per-sample joint shift, constrained to keep both windows in-bounds.
        offset = torch.randint(
            -crop_eleft, n_valid - crop_eright + 1, (B, 1), device=device
        )

        idx1 = offset + crop_eleft + torch.arange(
            crop_right - crop_eleft, device=device
        ).unsqueeze(0)
        idx2 = offset + crop_left + torch.arange(
            crop_eright - crop_left, device=device
        ).unsqueeze(0)
        return idx1, idx2, crop_l

    def _encode_crop(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Encode gathered patch crops with binomial content masking.

        ``relative_pos=True`` is load-bearing: the two crops are aligned on
        the same global patches, so gathering *absolute* position codes gives
        every positive pair in both contrastive terms an identical position
        stamp that negatives (almost) never share — the loss is then solvable
        by echoing the position embedding, and the content masking (which
        keeps the position code) actively teaches that solution. Within-crop
        positions keep order information without identifying the global
        timestamp, matching the original encoder's no-absolute-position
        property.
        """
        B = x.shape[0]
        zero_mask = None
        if self.training and self.mask_p < 1.0:
            keep = torch.rand(B, idx.shape[1], device=x.device) < self.mask_p
            zero_mask = ~keep
        return self.backbone.forward_patches(
            x,
            lengths=None,
            mask_indices=[idx[b] for b in range(B)],
            zero_mask=zero_mask,
            relative_pos=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Two overlapping crops → encode → hierarchical loss on the overlap.

        Returns None when the bucket's valid patch count is below the minimum
        span ``2 ** (temporal_unit + 1)``.
        """
        B, _, T = x.shape
        device = x.device
        n_patches = (T + self.patch_size - 1) // self.patch_size

        if lengths is not None:
            n_valid = int(
                ((lengths + self.patch_size - 1) // self.patch_size).min().item()
            )
        else:
            n_valid = n_patches

        crops = self._sample_crops(B, n_valid, device)
        if crops is None:
            return None
        idx1, idx2, crop_l = crops

        out1 = self._encode_crop(x, idx1)[:, -crop_l:]  # (B, crop_l, H)
        out2 = self._encode_crop(x, idx2)[:, :crop_l]  # (B, crop_l, H)

        loss = hierarchical_contrastive_loss(
            out1, out2, alpha=self.alpha, temporal_unit=self.temporal_unit
        )

        return {
            "ts2vec_loss": loss,
            "crop_l": crop_l,
            "_z": out1.detach(),
        }

    # ------------------------------------------------------------------
    # TrainingModel hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, device, grad_accum_steps=1):
        batch_loss = 0.0
        batch_n = 0
        all_z: list[torch.Tensor] = []

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for bucket in batch["buckets"]:
                x = bucket["views"][0].to(device)
                lengths = bucket["lengths"][0].to(device)
                out = self(x, lengths)
                if out is None:
                    continue
                loss = out["ts2vec_loss"]

                if not torch.isfinite(loss):
                    continue

                (loss / grad_accum_steps).backward()
                batch_loss += loss.item() * x.shape[0]
                batch_n += x.shape[0]
                all_z.append(out["_z"])

        if batch_n == 0:
            return None

        metrics = {"train/loss": batch_loss / batch_n}
        with torch.no_grad():
            if all_z:
                pooled = torch.cat([z.mean(dim=1) for z in all_z], dim=0)
                metrics.update(compute_collapse_metrics(pooled, prefix="train"))

        return {"loss": batch_loss / batch_n, "metrics": metrics}

    @torch.no_grad()
    def eval_step(self, eval_batches, device):
        self.eval()
        losses: list[float] = []
        for batch in eval_batches:
            for bucket in batch["buckets"]:
                x = bucket["views"][0].to(device)
                lengths = bucket["lengths"][0].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = self(x, lengths)
                if out is None:
                    continue
                loss = out["ts2vec_loss"]
                if torch.isfinite(loss):
                    losses.append(loss.item())
        return {
            "eval/ts2vec_loss": float(np.mean(losses)) if losses else float("nan"),
        }

    @torch.no_grad()
    def post_training_step(self, completed_steps, max_train_steps):
        if self.swa_backbone is None:
            return {}
        n = int(self._swa_n.item())
        for p_avg, p in zip(
            self.swa_backbone.parameters(), self.backbone.parameters()
        ):
            p_avg.add_(p.detach() - p_avg, alpha=1.0 / (n + 1))
        for b_avg, b in zip(self.swa_backbone.buffers(), self.backbone.buffers()):
            b_avg.copy_(b)
        self._swa_n += 1
        return {}

    def encode(self, x, lengths=None):
        """Embed with the SWA-averaged backbone (original inference protocol)."""
        if self.swa_backbone is None or int(self._swa_n.item()) == 0:
            return super().encode(x, lengths)

        if isinstance(x, torch.Tensor) and x.dim() == 4:
            views = [x[:, v, :, :] for v in range(x.shape[1])]
            view_lengths = (
                [lengths] * x.shape[1] if lengths is not None else [None] * x.shape[1]
            )
        elif isinstance(x, list):
            views = x
            view_lengths = lengths if lengths is not None else [None] * len(views)
        else:
            views = [x]
            view_lengths = [lengths]

        embeddings = [
            self.swa_backbone(view, vl) for view, vl in zip(views, view_lengths)
        ]
        return {"embeddings": torch.stack(embeddings, dim=1)}

    def describe_parameters(self):
        from market_jepa.training.utils import count_parameters

        param_counts = count_parameters(self, backbone=self.backbone)
        summary = (
            f"Data dimensions:\n  n_features: {self.backbone.n_features}\n"
            f"TS2Vec model parameters:\n"
            f"  Encoder (backbone): {param_counts['backbone']:,}\n"
            f"  Total: {param_counts['total']:,}\n"
            f"  alpha: {self.alpha} | temporal_unit: {self.temporal_unit} | "
            f"mask_p: {self.mask_p} | swa: {self.swa}"
        )
        return param_counts, summary

    def default_run_name(self, backbone_type, cfg):
        bb_cfg = backbone_block(cfg)
        return "__".join(
            [
                "mode=ts2vec",
                f"bb={backbone_type}",
                f"d_emb={bb_cfg.d_embedding}",
                f"alpha={self.alpha}",
                f"blr={(cfg.optimizer.blr or cfg.mode.training_overrides.blr):.1e}",
                f"bs={cfg.training.per_device_train_batch_size}",
                f"steps={cfg.training.max_train_steps}",
            ]
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        backbone_type = {
            "TransformerBackbone": "transformer",
        }.get(type(self.backbone).__name__)
        if backbone_type is None:
            raise ValueError(
                f"Unsupported backbone type for TS2Vec.save_pretrained: "
                f"{type(self.backbone).__name__}"
            )

        config = {
            "class": "TS2Vec",
            "backbone_type": backbone_type,
            "n_features": self.backbone.n_features,
            "d_embedding": self.backbone.d_embedding,
            "pool": self.backbone.pool,
            "backbone_config": asdict(self.backbone.config),
            "alpha": self.alpha,
            "temporal_unit": self.temporal_unit,
            "mask_p": self.mask_p,
            "swa": self.swa,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        model_to_save = self._orig_mod if hasattr(self, "_orig_mod") else self
        torch.save(model_to_save.state_dict(), os.path.join(path, "model.pt"))

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "TS2Vec":
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        config.pop("class", None)
        config.update(kwargs)

        from ..backbones.transformer import TransformerConfig

        backbone_config = config.pop("backbone_config")
        transformer_config = TransformerConfig(**backbone_config)

        extra_kwargs = {}
        _sd = torch.load(os.path.join(path, "model.pt"), map_location="cpu")
        # Knobs no save_pretrained ever recorded -- state_token,
        # diff_channels, n_info_channels -- read off the weights. See
        # backbones.backbone_kwargs_from_state_dict.
        extra_kwargs.update(backbone_kwargs_from_state_dict(_sd))
        # SPLAT THEM IN. Popping n_features and dropping the rest left
        # n_info_channels on the floor, so an information-token
        # checkpoint rebuilt a 20-channel patch embedding with no
        # info_proj and died in load_state_dict -- after training. The
        # same three lines below byol/dino/lejepa already splat it.
        backbone = create_backbone(
            backbone_type=config.pop("backbone_type"),
            n_features=extra_kwargs.pop("n_features", config.pop("n_features")),
            d_embedding=config.pop("d_embedding"),
            config=transformer_config,
            pool=config.pop("pool", "max"),
            **extra_kwargs,
        )
        model = cls(backbone=backbone, **config)
        model.load_state_dict(_sd)
        return model
