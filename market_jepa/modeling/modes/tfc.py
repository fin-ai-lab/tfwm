"""TF-C mode for time-series self-supervised learning.

Implements Zhang et al. (NeurIPS 2022, "Self-Supervised Contrastive
Pre-Training for Time Series via Time-Frequency Consistency") adapted to this
codebase, ported from the official repo (github.com/mims-harvard/TFC-pretraining):

  1. Two encoders: a *time* encoder over the raw sequence and a *frequency*
     encoder over the amplitude spectrum ``|rfft(x)|`` (frequency bins play
     the role of the time axis; channels are unchanged).
  2. Augmentations: jitter in the time domain (``DataTransform_TD``);
     remove-frequency + add-frequency perturbations in the frequency domain
     (``DataTransform_FD``, ported verbatim including the sum of the two
     perturbed copies).
  3. Cross-space projectors map each encoder's embedding to a shared 128-d
     space (Linear→BatchNorm→ReLU→Linear, as in the official ``TFC`` model).
  4. Loss (official ``model_pretrain``): with NTXent-poly at temperature 0.2,
     ``lam * (NTXent(h_t, h_t_aug) + NTXent(h_f, h_f_aug)) + NTXent(z_t, z_f)``
     with ``lam = 0.2``.

Downstream ``encode()`` follows the official fine-tuning protocol: the
concatenation ``[z_t, z_f]`` (2 × proj_dim = 256 dims) — note this means
``d_embedding != backbone.d_embedding`` and offline tools that rip
``backbone.*`` keys out of ``model.pt`` see only the time encoder.

Deviations (forced by the harness / multivariate data): the official
encoders are 2-layer transformers whose feature dim is the (univariate)
sequence length — inapplicable to multivariate inputs — so both branches use
the shared ``TransformerBackbone`` (the frequency branch is an independent
re-initialized copy); optimizer is the harness AdamW + cosine (original:
Adam, lr 3e-4, wd 3e-4).
"""

from __future__ import annotations

from market_jepa.backbone_config import backbone_block

import copy
import json
import os
from dataclasses import asdict

import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import backbone_kwargs_from_state_dict, create_backbone
from ..backbones.transformer import TransformerBackbone
from .base import TrainingModel, compute_collapse_metrics


def nt_xent_poly(
    zis: torch.Tensor,
    zjs: torch.Tensor,
    temperature: float = 0.2,
    use_poly: bool = True,
) -> torch.Tensor:
    """NTXentLoss_poly from the official repo (cosine similarity variant).

    Positives are (i, i) pairs across the two views; every other row of the
    concatenated 2B batch is a negative. The poly term adds
    ``epsilon * (1/B - pt)`` with ``epsilon = B`` (their default).
    """
    B = zis.shape[0]
    representations = torch.cat([zjs, zis], dim=0)  # (2B, D)
    sim = F.cosine_similarity(
        representations.unsqueeze(1), representations.unsqueeze(0), dim=-1
    )  # (2B, 2B)

    l_pos = torch.diag(sim, B)
    r_pos = torch.diag(sim, -B)
    positives = torch.cat([l_pos, r_pos]).view(2 * B, 1)

    diag = torch.eye(2 * B, dtype=torch.bool, device=zis.device)
    pos_mask = (
        torch.diag(torch.ones(B, device=zis.device), B)
        + torch.diag(torch.ones(B, device=zis.device), -B)
    )[: 2 * B, : 2 * B].bool()
    neg_mask = ~(diag | pos_mask)
    negatives = sim[neg_mask].view(2 * B, -1)

    logits = torch.cat([positives, negatives], dim=1) / temperature
    labels = torch.zeros(2 * B, dtype=torch.long, device=zis.device)
    ce = F.cross_entropy(logits, labels, reduction="sum")

    if not use_poly:
        return ce / (2 * B)

    onehot = torch.cat(
        [
            torch.ones(2 * B, 1, device=zis.device),
            torch.zeros(2 * B, negatives.shape[-1], device=zis.device),
        ],
        dim=-1,
    )
    pt = torch.mean(onehot * F.softmax(logits, dim=-1))
    epsilon = float(B)
    return ce / (2 * B) + epsilon * (1.0 / B - pt)


class TFC(TrainingModel):
    """TF-C: time-frequency consistency pre-training.

    Args:
        backbone: A ``TransformerBackbone`` — the time encoder. The frequency
            encoder is an architecture copy with independent weights.
        proj_dim: Output dim of each cross-space projector (128 official).
        proj_hidden: Hidden dim of each projector (256 official).
        temperature: NTXent temperature (0.2 official).
        lam: Weight of the within-space losses (0.2 official).
        jitter_sigma: Std of the time-domain jitter augmentation. The official
            value (2.0) assumes raw EEG scale; our channels are normalized,
            so the default is 0.1.
        freq_perturb_ratio: Fraction of frequency bins removed / added (0.1).
        use_poly_loss: Use the poly variant (official pretrain default).
        gradient_checkpointing: Enable gradient checkpointing on both encoders.
    """

    mode_label: str = "TF-C"
    mode_str: str = "TF-C"
    uses_multi_view: bool = False

    def __init__(
        self,
        backbone: TransformerBackbone,
        proj_dim: int = 128,
        proj_hidden: int = 256,
        temperature: float = 0.2,
        lam: float = 0.2,
        jitter_sigma: float = 0.1,
        freq_perturb_ratio: float = 0.1,
        use_poly_loss: bool = True,
        gradient_checkpointing: bool = False,
    ):
        if not isinstance(backbone, TransformerBackbone):
            raise ValueError(
                f"TFC only supports TransformerBackbone, got {type(backbone).__name__}"
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        super().__init__()
        self.backbone = backbone  # time encoder
        self.patch_size = backbone.patch_size
        self.n_features = backbone.n_features
        self.proj_dim = proj_dim
        self.proj_hidden = proj_hidden
        self.temperature = temperature
        self.lam = lam
        self.jitter_sigma = jitter_sigma
        self.freq_perturb_ratio = freq_perturb_ratio
        self.use_poly_loss = use_poly_loss

        # Probe embeddings are [z_t, z_f].
        self.d_embedding = 2 * proj_dim

        if gradient_checkpointing:
            backbone.gradient_checkpointing = True

        # Frequency encoder: same architecture, independently initialized
        # weights (a fresh build — deepcopy + _init_weights would keep the
        # copied patch-embedding conv and position embeddings).
        n_pos = backbone.position_embeddings.shape[1]
        max_seq_len = n_pos - (1 if backbone.pool == "cls" else 0)
        self.freq_backbone = TransformerBackbone(
            config=copy.deepcopy(backbone.config),
            n_features=backbone.n_features,
            d_embedding=backbone.d_embedding,
            pool=backbone.pool,
            max_seq_len=max_seq_len,
            gradient_checkpointing=gradient_checkpointing,
        )

        def make_projector() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(backbone.d_embedding, proj_hidden),
                nn.BatchNorm1d(proj_hidden),
                nn.ReLU(),
                nn.Linear(proj_hidden, proj_dim),
            )

        self.projector_t = make_projector()
        self.projector_f = make_projector()

    # ------------------------------------------------------------------
    # Augmentations (official augmentations.py, batched on-device)
    # ------------------------------------------------------------------

    def _jitter(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.jitter_sigma

    def _freq_augment(self, x_f: torch.Tensor) -> torch.Tensor:
        """DataTransform_FD: remove_frequency(x) + add_frequency(x)."""
        p = self.freq_perturb_ratio
        keep_mask = torch.rand_like(x_f) > p  # True = keep (1 - p of bins)
        removed = x_f * keep_mask
        add_mask = torch.rand_like(x_f) > (1 - p)  # True on p of bins
        max_amp = x_f.max()
        random_am = torch.rand_like(x_f) * (max_amp * 0.1)
        added = x_f + add_mask * random_am
        return removed + added

    @staticmethod
    def _to_freq(x: torch.Tensor) -> torch.Tensor:
        """Amplitude spectrum along time: (B, C, T) → (B, C, T//2+1)."""
        return fft.rfft(x.float(), dim=-1).abs().to(x.dtype)

    def _crop_to_min(self, x, lengths):
        if lengths is None:
            return x
        min_len = int(lengths.min().item())
        min_len -= min_len % self.patch_size
        return x[:, :, : max(min_len, self.patch_size)]

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------

    def _branches(
        self, x_t: torch.Tensor, x_f: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_t = self.backbone(x_t)
        z_t = self.projector_t(h_t.float())
        h_f = self.freq_backbone(x_f)
        z_f = self.projector_f(h_f.float())
        return h_t, z_t, h_f, z_f

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | None:
        x = self._crop_to_min(x, lengths)
        if x.shape[0] < 2:
            return None

        x_f = self._to_freq(x)
        h_t, z_t, h_f, z_f = self._branches(x, x_f)

        if self.training:
            aug_t = self._jitter(x)
            aug_f = self._freq_augment(x_f)
        else:
            aug_t, aug_f = x, x_f
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = self._branches(aug_t, aug_f)

        loss_t = nt_xent_poly(
            h_t.float(), h_t_aug.float(), self.temperature, self.use_poly_loss
        )
        loss_f = nt_xent_poly(
            h_f.float(), h_f_aug.float(), self.temperature, self.use_poly_loss
        )
        loss_tf = nt_xent_poly(z_t, z_f, self.temperature, self.use_poly_loss)

        loss = self.lam * (loss_t + loss_f) + loss_tf

        return {
            "tfc_loss": loss,
            "tfc_loss_t": loss_t.detach(),
            "tfc_loss_f": loss_f.detach(),
            "tfc_loss_tf": loss_tf.detach(),
            "_z": torch.cat([z_t, z_f], dim=-1).detach(),
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
                loss = out["tfc_loss"]

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
                pooled = torch.cat(all_z, dim=0)
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
                loss = out["tfc_loss"]
                if torch.isfinite(loss):
                    losses.append(loss.item())
        return {
            "eval/tfc_loss": float(np.mean(losses)) if losses else float("nan"),
        }

    def post_training_step(self, completed_steps, max_train_steps):
        return {}

    def encode(self, x, lengths=None):
        """Official fine-tuning protocol: embeddings = [z_t, z_f]."""
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

        embeddings = []
        for view, vl in zip(views, view_lengths):
            v = self._crop_to_min(view, vl)
            x_f = self._to_freq(v)
            _, z_t, _, z_f = self._branches(v, x_f)
            embeddings.append(torch.cat([z_t, z_f], dim=-1))
        return {"embeddings": torch.stack(embeddings, dim=1)}

    def describe_parameters(self):
        from market_jepa.training.utils import count_parameters

        param_counts = count_parameters(
            self,
            backbone=self.backbone,
            freq_backbone=self.freq_backbone,
            projector_t=self.projector_t,
            projector_f=self.projector_f,
        )
        summary = (
            f"Data dimensions:\n  n_features: {self.backbone.n_features}\n"
            f"TF-C model parameters:\n"
            f"  Time encoder: {param_counts['backbone']:,}\n"
            f"  Frequency encoder: {param_counts['freq_backbone']:,}\n"
            f"  Projectors: {param_counts['projector_t'] + param_counts['projector_f']:,}\n"
            f"  Total: {param_counts['total']:,}\n"
            f"  proj_dim: {self.proj_dim} | T: {self.temperature} | lam: {self.lam} | "
            f"jitter_sigma: {self.jitter_sigma}"
        )
        return param_counts, summary

    def default_run_name(self, backbone_type, cfg):
        bb_cfg = backbone_block(cfg)
        return "__".join(
            [
                "mode=tfc",
                f"bb={backbone_type}",
                f"d_emb={bb_cfg.d_embedding}",
                f"lam={self.lam}",
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
                f"Unsupported backbone type for TFC.save_pretrained: "
                f"{type(self.backbone).__name__}"
            )

        config = {
            "class": "TFC",
            "backbone_type": backbone_type,
            "n_features": self.backbone.n_features,
            "d_embedding": self.backbone.d_embedding,
            "pool": self.backbone.pool,
            "backbone_config": asdict(self.backbone.config),
            "proj_dim": self.proj_dim,
            "proj_hidden": self.proj_hidden,
            "temperature": self.temperature,
            "lam": self.lam,
            "jitter_sigma": self.jitter_sigma,
            "freq_perturb_ratio": self.freq_perturb_ratio,
            "use_poly_loss": self.use_poly_loss,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        model_to_save = self._orig_mod if hasattr(self, "_orig_mod") else self
        torch.save(model_to_save.state_dict(), os.path.join(path, "model.pt"))

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "TFC":
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
            pool=config.pop("pool", "mean"),
            **extra_kwargs,
        )
        model = cls(backbone=backbone, **config)
        model.load_state_dict(_sd)
        return model
