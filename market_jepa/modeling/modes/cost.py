"""CoST mode for time-series self-supervised learning.

Implements Woo et al. (ICLR 2022, "CoST: Contrastive Learning of Disentangled
Seasonal-Trend Representations for Time Series Forecasting") adapted to this
codebase's patch-level ViT encoder, ported from the official repo
(github.com/salesforce/CoST):

  1. Two augmented views of each sequence — scale / shift / jitter, each
     applied with probability ``aug_p`` at strength ``sigma`` (the official
     ``PretrainDataset.transform``).
  2. The backbone produces per-patch latents; two heads disentangle them:
     a *trend* head (mean over multi-kernel causal 1D convs) and a *season*
     head (a banded Fourier layer, ``BandedFourierLayer`` ported verbatim
     but made length-robust).
  3. Time-domain loss: MoCo-style InfoNCE on the trend component at one
     random patch position — momentum key encoder (m=0.999) + negative
     queue (K=256), temperature 0.07.
  4. Frequency-domain loss: instance contrast on the amplitude and phase of
     the season component's FFT across the batch, weighted by ``alpha``
     (0.0005), both views through the query encoder.

Downstream ``encode()`` follows the official protocol: concatenation of the
trend and season components at the last (valid) position — so with
``component_dims = d_embedding // 2`` the probe sees ``d_embedding`` dims.

Deviations (forced by the harness, mirroring the other baseline ports):
shared ViT instead of the dilated-conv encoder; harness AdamW + cosine
instead of SGD; queue handles arbitrary batch sizes (ring buffer without the
original's divisibility assert).

Only supports ``TransformerBackbone``.
"""

from __future__ import annotations

from market_jepa.backbone_config import backbone_block

import copy
import json
import math
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
from .ts2vec import instance_contrastive_loss


class BandedFourierLayer(nn.Module):
    """Complex linear map applied to a low-frequency band of the input FFT.

    Port of the official CoST layer (single band). ``length`` fixes the
    maximum sequence length the weights are allocated for; shorter inputs use
    the leading slice of the band (length-robust extension — the original
    hard-required ``t == length``).
    """

    def __init__(self, in_channels: int, out_channels: int, length: int):
        super().__init__()
        self.length = length
        self.num_freqs = length // 2 + 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(
            torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat)
        )
        self.bias = nn.Parameter(
            torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: (B, T, C_in) real
        b, t, _ = input.shape
        n_freqs = t // 2 + 1
        if n_freqs > self.num_freqs:
            raise ValueError(
                f"BandedFourierLayer allocated for length {self.length} "
                f"({self.num_freqs} freqs) but got length {t} ({n_freqs} freqs)"
            )
        input_fft = fft.rfft(input.float(), dim=1)  # (B, n_freqs, C_in)
        output_fft = torch.einsum(
            "bti,tio->bto", input_fft, self.weight[:n_freqs]
        ) + self.bias[:n_freqs]
        return fft.irfft(output_fft, n=t, dim=1).to(input.dtype)


class CoSTHeads(nn.Module):
    """Trend + season disentanglers over per-patch backbone latents."""

    def __init__(
        self,
        hidden_size: int,
        component_dims: int,
        kernels: list[int],
        fourier_length: int,
    ):
        super().__init__()
        self.kernels = kernels
        # Causal convs: left-pad by (k-1), trim the right overhang after.
        self.tfd = nn.ModuleList(
            [
                nn.Conv1d(hidden_size, component_dims, k, padding=k - 1)
                for k in kernels
            ]
        )
        self.sfd = BandedFourierLayer(hidden_size, component_dims, fourier_length)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z: (B, P, H) → trend (B, P, C), season (B, P, C)."""
        zt = z.transpose(1, 2)  # (B, H, P)
        trend = []
        for idx, mod in enumerate(self.tfd):
            out = mod(zt)  # (B, C, P + k - 1)
            if self.kernels[idx] != 1:
                out = out[..., : -(self.kernels[idx] - 1)]
            trend.append(out.transpose(1, 2))
        trend = torch.stack(trend, dim=0).mean(dim=0)  # (B, P, C)
        season = self.sfd(z)
        return trend, season


class CoST(TrainingModel):
    """CoST: disentangled seasonal-trend contrastive learning.

    Args:
        backbone: A ``TransformerBackbone`` with ``pool != "cls"``.
        kernels: Kernel sizes of the trend head's causal-conv experts
            (patch units; the official repo used [1..128] on timestamps).
        alpha: Weight of the frequency-domain (season) loss.
        queue_size: MoCo negative-queue length K.
        ema_momentum: Momentum m of the key encoder update.
        temperature: InfoNCE temperature for the time-domain loss.
        sigma: Augmentation strength for scale / shift / jitter.
        aug_p: Per-transform application probability.
        fourier_length: Max patch count the Fourier layer is allocated for.
        gradient_checkpointing: Enable gradient checkpointing on the encoder.
    """

    mode_label: str = "CoST"
    mode_str: str = "CoST"
    uses_multi_view: bool = False

    def __init__(
        self,
        backbone: TransformerBackbone,
        kernels: list[int] | None = None,
        alpha: float = 0.0005,
        queue_size: int = 256,
        ema_momentum: float = 0.999,
        temperature: float = 0.07,
        sigma: float = 0.5,
        aug_p: float = 0.5,
        fourier_length: int = 256,
        gradient_checkpointing: bool = False,
    ):
        if not isinstance(backbone, TransformerBackbone):
            raise ValueError(
                f"CoST only supports TransformerBackbone, got {type(backbone).__name__}"
            )
        if backbone.pool == "cls":
            raise ValueError(
                "CoST requires pool in {'mean', 'max', 'last'} (not 'cls'). "
                "forward_patches operates on raw patch positions without a CLS token."
            )
        if backbone.d_embedding % 2 != 0:
            raise ValueError(
                f"CoST needs an even d_embedding (trend + season halves), "
                f"got {backbone.d_embedding}"
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        super().__init__()
        self.backbone = backbone
        self.d_embedding = backbone.d_embedding
        self.component_dims = backbone.d_embedding // 2
        self.patch_size = backbone.patch_size
        self.n_features = backbone.n_features
        self.kernels = list(kernels) if kernels is not None else [1, 2, 4, 8, 16, 32, 64]
        self.alpha = alpha
        self.queue_size = queue_size
        self.ema_momentum = ema_momentum
        self.temperature = temperature
        self.sigma = sigma
        self.aug_p = aug_p
        self.fourier_length = fourier_length

        if gradient_checkpointing:
            backbone.gradient_checkpointing = True

        hidden_size = backbone.config.hidden_size
        self.heads = CoSTHeads(
            hidden_size, self.component_dims, self.kernels, fourier_length
        )
        self.head_q = nn.Sequential(
            nn.Linear(self.component_dims, self.component_dims),
            nn.ReLU(),
            nn.Linear(self.component_dims, self.component_dims),
        )

        # Momentum (key) branch: backbone + trend head + projection head.
        self.backbone_k = copy.deepcopy(backbone)
        self.backbone_k.gradient_checkpointing = False
        self.heads_k = copy.deepcopy(self.heads)
        self.head_k = copy.deepcopy(self.head_q)
        for module in (self.backbone_k, self.heads_k, self.head_k):
            for p in module.parameters():
                p.requires_grad = False

        self.register_buffer(
            "queue", F.normalize(torch.randn(self.component_dims, queue_size), dim=0)
        )
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    # ------------------------------------------------------------------
    # Augmentations (official PretrainDataset.transform, batched)
    # ------------------------------------------------------------------

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """scale ∘ shift ∘ jitter on (B, C, T), each with prob ``aug_p``."""
        B, C, _ = x.shape
        device = x.device
        out = x
        # scale: per-sample per-channel factor ~ N(1, sigma)
        apply = (torch.rand(B, 1, 1, device=device) < self.aug_p).float()
        factor = torch.randn(B, C, 1, device=device) * self.sigma + 1.0
        out = out * (apply * factor + (1 - apply))
        # shift: per-sample per-channel offset ~ N(0, sigma)
        apply = (torch.rand(B, 1, 1, device=device) < self.aug_p).float()
        offset = torch.randn(B, C, 1, device=device) * self.sigma
        out = out + apply * offset
        # jitter: elementwise noise ~ N(0, sigma)
        apply = (torch.rand(B, 1, 1, device=device) < self.aug_p).float()
        out = out + apply * torch.randn_like(out) * self.sigma
        return out

    # ------------------------------------------------------------------
    # Momentum / queue
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _momentum_update(self):
        m = self.ema_momentum
        for q_mod, k_mod in (
            (self.backbone, self.backbone_k),
            (self.heads, self.heads_k),
            (self.head_q, self.head_k),
        ):
            for p_q, p_k in zip(q_mod.parameters(), k_mod.parameters()):
                p_k.data.mul_(m).add_(p_q.detach().data, alpha=1 - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Ring-buffer enqueue that tolerates arbitrary batch sizes."""
        keys = keys.detach().float()
        K = self.queue_size
        n = keys.shape[0]
        if n >= K:
            self.queue.copy_(keys[-K:].T)
            self.queue_ptr[0] = 0
            return
        ptr = int(self.queue_ptr.item())
        end = ptr + n
        if end <= K:
            self.queue[:, ptr:end] = keys.T
        else:
            first = K - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, : end - K] = keys[first:].T
        self.queue_ptr[0] = end % K

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------

    def _crop_to_min(self, x, lengths):
        """Crop the bucket to its minimum valid patch span (rectangular)."""
        if lengths is None:
            return x, x.shape[-1] // self.patch_size
        min_len = int(lengths.min().item())
        n_valid = max(1, min_len // self.patch_size)
        return x[:, :, : n_valid * self.patch_size], n_valid

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | None:
        x, n_valid = self._crop_to_min(x, lengths)
        if n_valid < 2 or x.shape[0] < 2:
            return None
        if n_valid > self.fourier_length:
            x = x[:, :, : self.fourier_length * self.patch_size]
            n_valid = self.fourier_length

        x_q = self._augment(x) if self.training else x
        x_k = self._augment(x) if self.training else x

        rand_idx = int(torch.randint(0, n_valid, (1,)).item())

        # Query branch on view q
        z_q = self.backbone.forward_patches(x_q)  # (B, P, H)
        trend_q, season_q = self.heads(z_q)
        q_t = F.normalize(self.head_q(trend_q[:, rand_idx]), dim=-1)

        # Key branch on view k (momentum encoder, no grad)
        with torch.no_grad():
            z_k = self.backbone_k.forward_patches(x_k)
            trend_k, _ = self.heads_k(z_k)
            k_t = F.normalize(self.head_k(trend_k[:, rand_idx]), dim=-1)

        # Time-domain MoCo loss
        l_pos = torch.einsum("nc,nc->n", [q_t, k_t.to(q_t.dtype)]).unsqueeze(-1)
        l_neg = torch.einsum(
            "nc,ck->nk", [q_t, self.queue.clone().detach().to(q_t.dtype)]
        )
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=x.device)
        time_loss = F.cross_entropy(logits, labels)
        if self.training:
            self._dequeue_and_enqueue(k_t)

        # Frequency-domain loss: both views through the *query* encoder
        z_k_q = self.backbone.forward_patches(x_k)
        _, season_k = self.heads(z_k_q)
        q_s = F.normalize(season_q, dim=-1).float()
        k_s = F.normalize(season_k, dim=-1).float()

        q_s_freq = fft.rfft(q_s, dim=1)
        k_s_freq = fft.rfft(k_s, dim=1)
        eps = 1e-6
        q_amp = torch.sqrt((q_s_freq.real + eps).pow(2) + (q_s_freq.imag + eps).pow(2))
        k_amp = torch.sqrt((k_s_freq.real + eps).pow(2) + (k_s_freq.imag + eps).pow(2))
        q_phase = torch.atan2(q_s_freq.imag, q_s_freq.real + eps)
        k_phase = torch.atan2(k_s_freq.imag, k_s_freq.real + eps)

        season_loss = (
            instance_contrastive_loss(q_amp, k_amp)
            + instance_contrastive_loss(q_phase, k_phase)
        )
        loss = time_loss + self.alpha * (season_loss / 2)

        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == 0).float().mean()

        return {
            "cost_loss": loss,
            "cost_time_loss": time_loss.detach(),
            "cost_season_loss": season_loss.detach(),
            "cost_acc": acc,
            "_z": torch.cat([trend_q, season_q], dim=-1).detach(),
        }

    # ------------------------------------------------------------------
    # TrainingModel hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, device, grad_accum_steps=1):
        batch_loss = 0.0
        batch_n = 0
        accs: list[float] = []
        all_z: list[torch.Tensor] = []

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for bucket in batch["buckets"]:
                x = bucket["views"][0].to(device)
                lengths = bucket["lengths"][0].to(device)
                out = self(x, lengths)
                if out is None:
                    continue
                loss = out["cost_loss"]

                if not torch.isfinite(loss):
                    continue

                (loss / grad_accum_steps).backward()
                batch_loss += loss.item() * x.shape[0]
                batch_n += x.shape[0]
                accs.append(out["cost_acc"].item())
                all_z.append(out["_z"])

        if batch_n == 0:
            return None

        metrics = {
            "train/loss": batch_loss / batch_n,
            "train/cost_acc": float(np.mean(accs)) if accs else 0.0,
        }
        with torch.no_grad():
            if all_z:
                pooled = torch.cat([z.mean(dim=1) for z in all_z], dim=0)
                metrics.update(compute_collapse_metrics(pooled, prefix="train"))

        return {"loss": batch_loss / batch_n, "metrics": metrics}

    @torch.no_grad()
    def eval_step(self, eval_batches, device):
        self.eval()
        losses: list[float] = []
        accs: list[float] = []
        for batch in eval_batches:
            for bucket in batch["buckets"]:
                x = bucket["views"][0].to(device)
                lengths = bucket["lengths"][0].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = self(x, lengths)
                if out is None:
                    continue
                loss = out["cost_loss"]
                if torch.isfinite(loss):
                    losses.append(loss.item())
                    accs.append(out["cost_acc"].item())
        return {
            "eval/cost_loss": float(np.mean(losses)) if losses else float("nan"),
            "eval/cost_acc": float(np.mean(accs)) if accs else float("nan"),
        }

    def post_training_step(self, completed_steps, max_train_steps):
        self._momentum_update()
        return {}

    def encode(self, x, lengths=None):
        """Official protocol: concat(trend, season) at the last valid patch."""
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
            v, n_valid = self._crop_to_min(view, vl)
            if n_valid > self.fourier_length:
                v = v[:, :, : self.fourier_length * self.patch_size]
            z = self.backbone.forward_patches(v)
            trend, season = self.heads(z)
            embeddings.append(torch.cat([trend[:, -1], season[:, -1]], dim=-1))
        return {"embeddings": torch.stack(embeddings, dim=1)}

    def describe_parameters(self):
        from market_jepa.training.utils import count_parameters

        param_counts = count_parameters(
            self,
            backbone=self.backbone,
            heads=self.heads,
            head_q=self.head_q,
        )
        summary = (
            f"Data dimensions:\n  n_features: {self.backbone.n_features}\n"
            f"CoST model parameters:\n"
            f"  Encoder (backbone): {param_counts['backbone']:,}\n"
            f"  Trend/season heads: {param_counts['heads']:,}\n"
            f"  Projection head: {param_counts['head_q']:,}\n"
            f"  Total: {param_counts['total']:,}\n"
            f"  kernels: {self.kernels} | alpha: {self.alpha} | K: {self.queue_size} | "
            f"m: {self.ema_momentum} | T: {self.temperature} | sigma: {self.sigma}"
        )
        return param_counts, summary

    def default_run_name(self, backbone_type, cfg):
        bb_cfg = backbone_block(cfg)
        return "__".join(
            [
                "mode=cost",
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
                f"Unsupported backbone type for CoST.save_pretrained: "
                f"{type(self.backbone).__name__}"
            )

        config = {
            "class": "CoST",
            "backbone_type": backbone_type,
            "n_features": self.backbone.n_features,
            "d_embedding": self.backbone.d_embedding,
            "pool": self.backbone.pool,
            "backbone_config": asdict(self.backbone.config),
            "kernels": self.kernels,
            "alpha": self.alpha,
            "queue_size": self.queue_size,
            "ema_momentum": self.ema_momentum,
            "temperature": self.temperature,
            "sigma": self.sigma,
            "aug_p": self.aug_p,
            "fourier_length": self.fourier_length,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        model_to_save = self._orig_mod if hasattr(self, "_orig_mod") else self
        torch.save(model_to_save.state_dict(), os.path.join(path, "model.pt"))

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "CoST":
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
            pool=config.pop("pool", "last"),
            **extra_kwargs,
        )
        model = cls(backbone=backbone, **config)
        model.load_state_dict(_sd)
        return model
