"""TimeMAE mode for time-series self-supervised learning.

Implements Cheng et al. (2023, "TimeMAE: Self-Supervised Representations of
Time Series with Decoupled Masked Autoencoders") adapted to this codebase's
ViT encoder, ported from the official repo (github.com/ustc-time-series/TimeMAE).

The original architecture is already a patch transformer (Conv1d patch
projection with kernel = stride = wave_length, learnable positions, TRM
blocks), so the shared ``TransformerBackbone`` is a direct substitute. The
decoupled-MAE machinery on top is ported faithfully:

  1. A *tokenizer* (Linear → Gumbel-softmax over ``vocab_size`` codewords)
     assigns each patch embedding a discrete token.
  2. ``mask_ratio`` of the patches are masked; the encoder sees only visible
     patches; a *momentum encoder* (m = 0.99) encodes the masked patches to
     produce latent regression targets.
  3. A cross-attention *regressor* (``reg_layers`` blocks, queries = mask
     tokens + positions, keys/values = visible representations) predicts the
     masked representations.
  4. Losses: MSE alignment to the momentum targets (weight ``align_weight``,
     official ``--alpha 5.0``) + codeword classification CE with label
     smoothing 0.2 (weight ``reconstruct_weight``, official ``--beta 1.0``).

Downstream ``encode()`` uses the backbone's pooled ``forward`` (config
``pool="mean"`` mirrors the official mean-over-patches readout).

Deviations: per-sample random masks instead of one batch-wide mask; the
harness AdamW + cosine schedule (original: AdamW, lr 1e-3, no schedule);
bucket sequences are cropped to the bucket-min patch count so the mask split
is rectangular.

Only supports ``TransformerBackbone``.
"""

from __future__ import annotations

from market_jepa.backbone_config import backbone_block

import copy
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


class Tokenizer(nn.Module):
    """Patch embedding → discrete codeword via Gumbel-softmax (official)."""

    def __init__(self, rep_dim: int, vocab_size: int):
        super().__init__()
        self.center = nn.Linear(rep_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, length, dim = x.shape
        probs = self.center(x.reshape(-1, dim).float())
        ret = F.gumbel_softmax(probs)
        indexes = ret.max(-1, keepdim=True)[1]
        return indexes.view(bs, length)


class CrossAttnBlock(nn.Module):
    """Official CrossAttnTRMBlock: post-norm, residual-scaled (a init 1e-8)."""

    def __init__(self, d_model: int, attn_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=attn_heads, batch_first=True, dropout=dropout
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.a1 = nn.Parameter(torch.tensor(1e-8))
        self.a2 = nn.Parameter(torch.tensor(1e-8))

    def forward(
        self, rep_visible: torch.Tensor, rep_mask_token: torch.Tensor
    ) -> torch.Tensor:
        attn_out, _ = self.attn(
            rep_mask_token, rep_visible, rep_visible, need_weights=False
        )
        x = self.norm1(rep_mask_token + self.dropout1(self.a1 * attn_out))
        x = self.norm2(x + self.dropout2(self.a2 * self.ffn(x)))
        return x


class TimeMAERegressor(nn.Module):
    """Stack of cross-attention blocks refining the mask-token queries."""

    def __init__(self, d_model: int, attn_heads: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttnBlock(d_model, attn_heads) for _ in range(layers)]
        )

    def forward(
        self, rep_visible: torch.Tensor, rep_mask_token: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            rep_mask_token = layer(rep_visible, rep_mask_token)
        return rep_mask_token


class TimeMAE(TrainingModel):
    """TimeMAE: decoupled masked autoencoding with codeword targets.

    Args:
        backbone: A ``TransformerBackbone`` with ``pool != "cls"``.
        vocab_size: Codebook size of the tokenizer (192 official).
        mask_ratio: Fraction of patches masked (0.6 official).
        ema_momentum: Momentum of the target-encoder update (0.99 official).
        reg_layers: Number of cross-attention regressor blocks (4 official).
        align_weight: Weight of the latent MSE loss (5.0 official ``alpha``).
        reconstruct_weight: Weight of the codeword CE loss (1.0 official ``beta``).
        gradient_checkpointing: Enable gradient checkpointing on the encoder.
    """

    mode_label: str = "TimeMAE"
    mode_str: str = "TimeMAE"
    uses_multi_view: bool = False

    def __init__(
        self,
        backbone: TransformerBackbone,
        vocab_size: int = 192,
        mask_ratio: float = 0.6,
        ema_momentum: float = 0.99,
        reg_layers: int = 4,
        align_weight: float = 5.0,
        reconstruct_weight: float = 1.0,
        gradient_checkpointing: bool = False,
    ):
        if not isinstance(backbone, TransformerBackbone):
            raise ValueError(
                f"TimeMAE only supports TransformerBackbone, got {type(backbone).__name__}"
            )
        if backbone.pool == "cls":
            raise ValueError(
                "TimeMAE requires pool in {'mean', 'max', 'last'} (not 'cls'). "
                "forward_patches operates on raw patch positions without a CLS token."
            )
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")

        super().__init__()
        self.backbone = backbone
        self.d_embedding = backbone.d_embedding
        self.patch_size = backbone.patch_size
        self.n_features = backbone.n_features
        self.vocab_size = vocab_size
        self.mask_ratio = mask_ratio
        self.ema_momentum = ema_momentum
        self.reg_layers = reg_layers
        self.align_weight = align_weight
        self.reconstruct_weight = reconstruct_weight

        if gradient_checkpointing:
            backbone.gradient_checkpointing = True

        hidden_size = backbone.config.hidden_size

        self.tokenizer = Tokenizer(hidden_size, vocab_size)
        self.mask_token = nn.Parameter(torch.randn(hidden_size))
        self.regressor = TimeMAERegressor(
            hidden_size, backbone.config.num_attention_heads, reg_layers
        )

        # Momentum (target) encoder: exact copy, updated by EMA, never by grad.
        self.momentum_backbone = copy.deepcopy(backbone)
        self.momentum_backbone.gradient_checkpointing = False
        for p in self.momentum_backbone.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | None:
        B, _, T = x.shape
        device = x.device

        if lengths is not None:
            min_len = int(lengths.min().item())
        else:
            min_len = T
        P = min_len // self.patch_size
        mask_len = int(self.mask_ratio * P)
        if mask_len < 1 or P - mask_len < 1:
            return None
        x = x[:, :, : P * self.patch_size]

        # Tokenize every patch (pre-position content embeddings).
        # Through patch_channels, not the raw view: the trailing per-window
        # columns are the information token's and the patch embedding is not
        # shaped for them.
        patch_embeds = self.backbone.patch_embed(
            self.backbone.patch_channels(x))  # (B, P, H)
        tokens = self.tokenizer(patch_embeds)  # (B, P) long

        # Per-sample random visible/masked split.
        noise = torch.rand(B, P, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        v_idx = ids_shuffle[:, : P - mask_len]  # (B, P - mask_len)
        m_idx = ids_shuffle[:, P - mask_len :]  # (B, mask_len)

        rep_visible = self.backbone.forward_patches(
            x, mask_indices=[v_idx[b] for b in range(B)]
        )
        with torch.no_grad():
            rep_mask = self.momentum_backbone.forward_patches(
                x, mask_indices=[m_idx[b] for b in range(B)]
            )

        # Mask-token queries at the masked positions.
        pos = self.backbone.position_embeddings[0]  # (n_pos, H)
        pos_m = pos[m_idx]  # (B, mask_len, H)
        rep_mask_token = self.mask_token.unsqueeze(0).unsqueeze(0) + pos_m

        rep_pred = self.regressor(rep_visible, rep_mask_token)

        align_loss = F.mse_loss(rep_pred.float(), rep_mask.detach().float())

        token_logits = self.tokenizer.center(rep_pred.float())  # (B, mask_len, V)
        target_tokens = torch.gather(tokens, 1, m_idx)  # (B, mask_len)
        reconstruct_loss = F.cross_entropy(
            token_logits.reshape(-1, self.vocab_size),
            target_tokens.reshape(-1),
            label_smoothing=0.2,
        )

        loss = self.align_weight * align_loss + self.reconstruct_weight * reconstruct_loss

        with torch.no_grad():
            token_acc = (
                (token_logits.argmax(dim=-1) == target_tokens).float().mean()
            )

        return {
            "timemae_loss": loss,
            "timemae_align_loss": align_loss.detach(),
            "timemae_reconstruct_loss": reconstruct_loss.detach(),
            "timemae_token_acc": token_acc,
            "_encoded": rep_visible.detach(),
        }

    # ------------------------------------------------------------------
    # TrainingModel hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, device, grad_accum_steps=1):
        batch_loss = 0.0
        batch_n = 0
        accs: list[float] = []
        all_encoded: list[torch.Tensor] = []

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for bucket in batch["buckets"]:
                x = bucket["views"][0].to(device)
                lengths = bucket["lengths"][0].to(device)
                out = self(x, lengths)
                if out is None:
                    continue
                loss = out["timemae_loss"]

                if not torch.isfinite(loss):
                    continue

                (loss / grad_accum_steps).backward()
                batch_loss += loss.item() * x.shape[0]
                batch_n += x.shape[0]
                accs.append(out["timemae_token_acc"].item())
                all_encoded.append(out["_encoded"])

        if batch_n == 0:
            return None

        metrics = {
            "train/loss": batch_loss / batch_n,
            "train/timemae_token_acc": float(np.mean(accs)) if accs else 0.0,
        }
        with torch.no_grad():
            if all_encoded:
                pooled = torch.cat([e.mean(dim=1) for e in all_encoded], dim=0)
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
                loss = out["timemae_loss"]
                if torch.isfinite(loss):
                    losses.append(loss.item())
                    accs.append(out["timemae_token_acc"].item())
        return {
            "eval/timemae_loss": float(np.mean(losses)) if losses else float("nan"),
            "eval/timemae_token_acc": float(np.mean(accs)) if accs else float("nan"),
        }

    @torch.no_grad()
    def post_training_step(self, completed_steps, max_train_steps):
        m = self.ema_momentum
        for p_m, p in zip(
            self.momentum_backbone.parameters(), self.backbone.parameters()
        ):
            p_m.data.mul_(m).add_(p.detach().data, alpha=1 - m)
        return {}

    def describe_parameters(self):
        from market_jepa.training.utils import count_parameters

        param_counts = count_parameters(
            self,
            backbone=self.backbone,
            tokenizer=self.tokenizer,
            regressor=self.regressor,
        )
        summary = (
            f"Data dimensions:\n  n_features: {self.backbone.n_features}\n"
            f"TimeMAE model parameters:\n"
            f"  Encoder (backbone): {param_counts['backbone']:,}\n"
            f"  Tokenizer: {param_counts['tokenizer']:,}\n"
            f"  Regressor: {param_counts['regressor']:,}\n"
            f"  Total: {param_counts['total']:,}\n"
            f"  vocab: {self.vocab_size} | mask_ratio: {self.mask_ratio} | "
            f"m: {self.ema_momentum} | align_w: {self.align_weight}"
        )
        return param_counts, summary

    def default_run_name(self, backbone_type, cfg):
        bb_cfg = backbone_block(cfg)
        return "__".join(
            [
                "mode=timemae",
                f"bb={backbone_type}",
                f"d_emb={bb_cfg.d_embedding}",
                f"mr={self.mask_ratio}",
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
                f"Unsupported backbone type for TimeMAE.save_pretrained: "
                f"{type(self.backbone).__name__}"
            )

        config = {
            "class": "TimeMAE",
            "backbone_type": backbone_type,
            "n_features": self.backbone.n_features,
            "d_embedding": self.backbone.d_embedding,
            "pool": self.backbone.pool,
            "backbone_config": asdict(self.backbone.config),
            "vocab_size": self.vocab_size,
            "mask_ratio": self.mask_ratio,
            "ema_momentum": self.ema_momentum,
            "reg_layers": self.reg_layers,
            "align_weight": self.align_weight,
            "reconstruct_weight": self.reconstruct_weight,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        model_to_save = self._orig_mod if hasattr(self, "_orig_mod") else self
        torch.save(model_to_save.state_dict(), os.path.join(path, "model.pt"))

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "TimeMAE":
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
