"""Masked Autoencoder (MAE) mode for time-series self-supervised learning.

Baseline comparison point for LeJEPA / I-JEPA. Reconstructs masked patches
in input feature space rather than predicting in latent space.

Adapted for 1D financial time series with variable-length padded sequences.
No quantization.
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
from ..backbones.transformer import TransformerBackbone, ViTBlock
from .base import TrainingModel, compute_collapse_metrics


# ---------------------------------------------------------------------------
# Masking + patchify helpers
# ---------------------------------------------------------------------------


def _mae_random_masking(
    n_valid: torch.Tensor,
    n_patches: int,
    mask_ratio: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample random patch masking that respects padding.

    Padded patches receive ``+inf`` noise so they always sort into the tail
    (masked region); ``len_keep`` is clamped to the minimum valid patch count
    in the batch so all samples can produce a rectangular ``(B, len_keep)``
    tensor of visible indices.

    Args:
        n_valid: (B,) number of non-padded patches per sample.
        n_patches: Total padded patch count (uniform across the batch).
        mask_ratio: Fraction of valid patches to mask.
        device: Tensor device.

    Returns:
        ids_keep: (B, len_keep) indices of visible patches (non-padded).
        ids_restore: (B, n_patches) argsort of ids_shuffle (for unshuffling).
        mask: (B, n_patches) float — 1 at masked, 0 at kept.
        valid_mask: (B, n_patches) bool — True at non-padded positions.
    """
    B = n_valid.shape[0]
    min_valid = int(n_valid.min().item())
    len_keep = max(1, int(round(min_valid * (1.0 - mask_ratio))))

    # Noise in [0, 1); set +inf on padded positions so they sort to the tail.
    noise = torch.rand(B, n_patches, device=device)
    patch_pos = torch.arange(n_patches, device=device).unsqueeze(0)  # (1, n_patches)
    valid_mask = patch_pos < n_valid.unsqueeze(1)  # (B, n_patches)
    noise = noise.masked_fill(~valid_mask, float("inf"))

    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small = keep
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]

    # Binary mask: 1 on removed positions, 0 on kept. Built in shuffled order,
    # then unshuffled back to original patch order.
    mask = torch.ones(B, n_patches, device=device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return ids_keep, ids_restore, mask, valid_mask


def _patchify(x: torch.Tensor, patch_size: int, n_patches: int) -> torch.Tensor:
    """Reshape ``(B, n_features, length)`` into ``(B, n_patches, n_features*patch_size)``.

    Pads the time axis to ``n_patches * patch_size`` with zeros before
    reshaping so padding tokens have a well-defined (zero) reconstruction
    target; the loss masks them out anyway.
    """
    B, C, T = x.shape
    padded_T = n_patches * patch_size
    if T < padded_T:
        x = F.pad(x, (0, padded_T - T))
    # (B, C, n_patches, patch_size) → (B, n_patches, C, patch_size) → flatten last
    x = x.reshape(B, C, n_patches, patch_size).permute(0, 2, 1, 3)
    return x.reshape(B, n_patches, C * patch_size)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class MAEDecoder(nn.Module):
    """Lightweight transformer decoder that reconstructs masked patches.

    Mirrors the MAE paper's decoder: project encoder dim to ``decoder_embed_dim``,
    fill masked positions with a learned mask token, unshuffle to the original
    patch order, add learnable positional embeddings, run ``depth`` transformer
    blocks, and project to ``patch_out_dim = n_features * patch_size``.
    """

    def __init__(
        self,
        hidden_size: int,
        decoder_embed_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        max_patches: int,
        patch_out_dim: int,
    ):
        super().__init__()
        self.decoder_embed = nn.Linear(hidden_size, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.randn(1, max_patches, decoder_embed_dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    hidden_size=decoder_embed_dim,
                    num_attention_heads=decoder_num_heads,
                    intermediate_size=decoder_embed_dim * 4,
                    drop_path_rate=0.0,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.pred = nn.Linear(decoder_embed_dim, patch_out_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        encoded: torch.Tensor,
        ids_restore: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct all patches given the encoded visible tokens.

        Args:
            encoded: (B, len_keep, hidden_size) — encoder output for visible patches.
            ids_restore: (B, n_patches) — unshuffling indices.
            valid_mask: (B, n_patches) bool — True at non-padded positions.
                Padded decoder positions are ignored via key_padding_mask.

        Returns:
            pred: (B, n_patches, patch_out_dim) — reconstruction in flattened patch space.
        """
        B, len_keep, _ = encoded.shape
        n_patches = ids_restore.shape[1]

        x = self.decoder_embed(encoded)  # (B, len_keep, D_dec)

        # Append mask tokens for the masked positions, then unshuffle.
        n_mask = n_patches - len_keep
        mask_tokens = self.mask_token.expand(B, n_mask, -1)
        x = torch.cat([x, mask_tokens], dim=1)  # (B, n_patches, D_dec)

        idx = ids_restore.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.gather(x, dim=1, index=idx)

        # Positional embeddings (learnable, matches project convention).
        x = x + self.decoder_pos_embed[:, :n_patches]

        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~valid_mask  # True = ignore

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        x = self.norm(x)
        return self.pred(x)


# ---------------------------------------------------------------------------
# MAE mode
# ---------------------------------------------------------------------------


class MAE(TrainingModel):
    """Masked Autoencoder for time-series self-supervised learning.

    Uses ``TransformerBackbone`` as the encoder (processes only visible patches
    via ``backbone.forward_patches(mask_indices=...)``) and a lightweight
    transformer decoder that reconstructs patches in normalized feature space.

    Only supports ``TransformerBackbone`` (same restriction as I-JEPA — CNN
    backbones cannot drop timesteps without architectural surgery).

    Args:
        backbone: A ``TransformerBackbone`` with ``pool != "cls"``.
        decoder_embed_dim: Decoder hidden dimension.
        decoder_depth: Number of decoder transformer blocks.
        decoder_num_heads: Number of decoder attention heads.
        mask_ratio: Fraction of valid patches to mask. **Sweepable** via config.
        norm_pix_loss: If True, normalize each patch to zero mean / unit variance
            before computing MSE (matches the MAE paper).
        gradient_checkpointing: Enable gradient checkpointing on the encoder.
    """

    mode_label: str = "MAE"
    mode_str: str = "MAE"
    uses_multi_view: bool = False

    def __init__(
        self,
        backbone: TransformerBackbone,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_num_heads: int = 4,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
        gradient_checkpointing: bool = False,
    ):
        if not isinstance(backbone, TransformerBackbone):
            raise ValueError(
                f"MAE only supports TransformerBackbone, got {type(backbone).__name__}"
            )
        if backbone.pool == "cls":
            raise ValueError(
                "MAE requires pool in {'mean', 'max', 'last'} (not 'cls'). "
                "forward_patches operates on raw patch positions without a CLS token."
            )
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")

        super().__init__()
        self.backbone = backbone
        self.d_embedding = backbone.d_embedding
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = backbone.patch_size
        self.n_features = backbone.n_features

        if gradient_checkpointing:
            backbone.gradient_checkpointing = True

        hidden_size = backbone.config.hidden_size
        max_patches = backbone.position_embeddings.shape[1]

        self.decoder = MAEDecoder(
            hidden_size=hidden_size,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            max_patches=max_patches,
            patch_out_dim=self.n_features * self.patch_size,
        )

        # Stash decoder config for save_pretrained.
        self._decoder_embed_dim = decoder_embed_dim
        self._decoder_depth = decoder_depth
        self._decoder_num_heads = decoder_num_heads

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Mask → encode visible patches → decode all patches → reconstruction loss."""
        B, _, T = x.shape
        device = x.device
        n_patches = (T + self.patch_size - 1) // self.patch_size

        if lengths is not None:
            n_valid = (lengths + self.patch_size - 1) // self.patch_size
        else:
            n_valid = torch.full((B,), n_patches, device=device, dtype=torch.long)

        ids_keep, ids_restore, mask, valid_mask = _mae_random_masking(
            n_valid, n_patches, self.mask_ratio, device
        )

        # Encoder sees only visible patches.
        ids_keep_list = [ids_keep[b] for b in range(B)]
        encoded = self.backbone.forward_patches(x, lengths, mask_indices=ids_keep_list)
        # (B, len_keep, hidden_size)

        pred = self.decoder(encoded, ids_restore, valid_mask=valid_mask)
        # (B, n_patches, n_features * patch_size)

        target = _patchify(x, self.patch_size, n_patches)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss_per_patch = ((pred - target) ** 2).mean(dim=-1)  # (B, n_patches)
        weight = mask * valid_mask.float()  # masked + non-padded only
        denom = weight.sum().clamp(min=1.0)
        loss = (loss_per_patch * weight).sum() / denom

        return {
            "mae_loss": loss,
            "pred": pred,
            "mask": mask,
            "valid_mask": valid_mask,
            "_encoded": encoded,  # for collapse monitoring
            "_valid_frac": (valid_mask.float().mean()).detach(),
        }

    # ------------------------------------------------------------------
    # TrainingModel hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, device, grad_accum_steps=1):
        batch_loss = 0.0
        batch_n = 0
        all_encoded = []
        valid_fracs = []

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for bucket in batch["buckets"]:
                x = bucket["views"][0].to(device)
                lengths = bucket["lengths"][0].to(device)
                out = self(x, lengths)
                loss = out["mae_loss"]

                if not torch.isfinite(loss):
                    continue

                (loss / grad_accum_steps).backward()
                batch_loss += loss.item() * x.shape[0]
                batch_n += x.shape[0]
                all_encoded.append(out["_encoded"].detach())
                valid_fracs.append(out["_valid_frac"].item())

        if batch_n == 0:
            return None

        metrics = {"train/loss": batch_loss / batch_n}
        if valid_fracs:
            metrics["train/mae_valid_frac"] = float(np.mean(valid_fracs))

        # Collapse monitoring on mean-pooled encoder output across visible patches.
        with torch.no_grad():
            if all_encoded:
                pooled = torch.cat([e.mean(dim=1) for e in all_encoded], dim=0)
                metrics.update(compute_collapse_metrics(pooled, prefix="train"))

        return {"loss": batch_loss / batch_n, "metrics": metrics}

    @torch.no_grad()
    def eval_step(self, eval_batches, device):
        self.eval()
        losses = []
        for batch in eval_batches:
            for bucket in batch["buckets"]:
                x = bucket["views"][0].to(device)
                lengths = bucket["lengths"][0].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = self(x, lengths)
                    loss = out["mae_loss"]
                if torch.isfinite(loss):
                    losses.append(loss.item())
        return {"eval/mae_loss": float(np.mean(losses)) if losses else float("nan")}

    def post_training_step(self, completed_steps, max_train_steps):
        return {}

    def describe_parameters(self):
        from market_jepa.training.utils import count_parameters

        param_counts = count_parameters(
            self,
            backbone=self.backbone,
            decoder=self.decoder,
        )
        summary = (
            f"Data dimensions:\n  n_features: {self.backbone.n_features}\n"
            f"MAE model parameters:\n"
            f"  Encoder (backbone): {param_counts['backbone']:,}\n"
            f"  Decoder: {param_counts['decoder']:,}\n"
            f"  Total: {param_counts['total']:,}\n"
            f"  mask_ratio: {self.mask_ratio}"
        )
        return param_counts, summary

    def default_run_name(self, backbone_type, cfg):
        bb_cfg = backbone_block(cfg)
        return "__".join(
            [
                "mode=mae",
                f"bb={backbone_type}",
                f"d_emb={bb_cfg.d_embedding}",
                f"mr={self.mask_ratio}",
                f"blr={(cfg.optimizer.blr or cfg.mode.training_overrides.blr):.1e}",
                f"bs={cfg.training.per_device_train_batch_size}",
                f"steps={cfg.training.max_train_steps}",
            ]
        )

    # ------------------------------------------------------------------
    # Save / load (mirrors LeJEPA.save_pretrained; includes "class" field
    # so the offline evals can dispatch without backbone-config guessing)
    # ------------------------------------------------------------------

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        backbone_type = {
            "TransformerBackbone": "transformer",
        }.get(type(self.backbone).__name__)
        if backbone_type is None:
            raise ValueError(
                f"Unsupported backbone type for MAE.save_pretrained: "
                f"{type(self.backbone).__name__}"
            )

        config = {
            "class": "MAE",
            "backbone_type": backbone_type,
            "n_features": self.backbone.n_features,
            "d_embedding": self.backbone.d_embedding,
            "pool": self.backbone.pool,
            "backbone_config": asdict(self.backbone.config),
            "decoder_embed_dim": self._decoder_embed_dim,
            "decoder_depth": self._decoder_depth,
            "decoder_num_heads": self._decoder_num_heads,
            "mask_ratio": self.mask_ratio,
            "norm_pix_loss": self.norm_pix_loss,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        model_to_save = self._orig_mod if hasattr(self, "_orig_mod") else self
        torch.save(model_to_save.state_dict(), os.path.join(path, "model.pt"))

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "MAE":
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
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
            backbone_type=config["backbone_type"],
            n_features=extra_kwargs.pop("n_features", config["n_features"]),
            d_embedding=config["d_embedding"],
            config=transformer_config,
            pool=config.get("pool", "mean"),
            **extra_kwargs,
        )
        model = cls(
            backbone=backbone,
            decoder_embed_dim=config["decoder_embed_dim"],
            decoder_depth=config["decoder_depth"],
            decoder_num_heads=config["decoder_num_heads"],
            mask_ratio=config["mask_ratio"],
            norm_pix_loss=config["norm_pix_loss"],
        )
        model.load_state_dict(_sd)
        return model
