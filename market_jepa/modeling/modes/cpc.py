r"""Contrastive Predictive Coding (CPC) mode for time-series self-supervised learning.

Implements Oord et al. (2018) adapted to 1D market snapshots:

  1. ViT encoder produces per-patch latents ``z_t`` via ``backbone.forward_patches``.
  2. A GRU context network aggregates ``z_1..z_{t_c}`` into a single context
     ``c_{t_c}`` (the GRU's final hidden state).
  3. ``K`` linear predictive heads ``W_k`` map ``c`` to ``\hat z_{t_c+k}``.
  4. InfoNCE classifies the correct target ``z_{t_c+k}`` against in-batch
     cross-sequence negatives (other samples' ``z`` at the same predicted step).

Only supports ``TransformerBackbone`` (same restriction as I-JEPA / MAE).
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


class CPCPredictiveHeads(nn.Module):
    r"""K linear heads mapping context ``c`` to per-step target predictions.

    Matches the CPC paper's density ratio parameterization
    ``f_k(x_{t+k}, c_t) = exp(z_{t+k}^T W_k c_t)`` — each ``W_k`` is a bias-free
    linear map from the GRU hidden size to the encoder hidden size, applied to
    ``c`` to produce ``\hat z_{t+k}``; the InfoNCE logit is then the dot product
    ``\hat z_{t+k} . z_{t+k}'`` for each candidate ``z'``.

    Args:
        context_dim: Dimension of the GRU context vector ``c``.
        z_dim: Dimension of the encoder latents ``z`` (backbone hidden size).
        n_predictions: Number of future steps ``K`` to predict.
    """

    def __init__(self, context_dim: int, z_dim: int, n_predictions: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [nn.Linear(context_dim, z_dim, bias=False) for _ in range(n_predictions)]
        )
        for head in self.heads:
            nn.init.trunc_normal_(head.weight, std=0.02)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Return stacked predictions of shape ``(B, K, z_dim)``."""
        return torch.stack([head(c) for head in self.heads], dim=1)


class CPC(TrainingModel):
    r"""Contrastive Predictive Coding for time-series self-supervised learning.

    Args:
        backbone: A ``TransformerBackbone`` with ``pool != "cls"``. Acts as the
            encoder ``g_enc`` — ``forward_patches`` returns per-patch latents.
        gru_hidden_size: Hidden dim of the GRU context network.
        gru_num_layers: Number of stacked GRU layers.
        n_predictions: Number of future steps ``K`` predicted from ``c``.
        min_context_frac: Lower bound on the context split, as a fraction of the
            minimum valid patch count in the batch. The split ``t_c`` is sampled
            uniformly from ``[max(1, round(min_context_frac * min_valid)), min_valid - K]``.
        temperature: InfoNCE temperature applied to logits (``\hat z . z / T``).
            Only used when ``cosine_logits`` is True — the paper's bilinear
            form has no temperature.
        negative_scope: Which in-batch candidates count as negatives.
            "xticker_xday" (the market adaptation) masks off-diagonal entries
            that share a ticker OR a date with the anchor — a same-ticker or
            same-day window is close to a positive, and InfoNCE with
            near-positives as negatives punishes exactly the invariance being
            trained. "all" is the paper's scheme: every other sample in the
            batch is a negative.
        cosine_logits: True (the market adaptation) L2-normalizes predictions
            and targets so logits are cosine similarities scaled by
            ``1 / temperature``. False is the paper's density-ratio form
            ``f_k = exp(z^T W_k c)`` — raw dot products, ``W_k`` carries the
            scale. Without normalization, magnitude differences across samples
            (price scale, ticker identity) can identify the positive
            regardless of predictive content — that risk is the reason the
            adaptation exists, and turning this off measures it.
        target_encoder: How the future patches are encoded.
            "stopgrad_bidir" (the market adaptation) encodes targets with a
            separate bidirectional no-grad pass, JEPA-style, so they act as a
            fixed regression target. "shared_causal" is the paper's shape: one
            causal pass over context + target patches with the SAME encoder,
            gradients flowing into the targets (in the paper z_t is a local
            conv encoding; a single causal pass is the closest transformer
            analogue that keeps targets blind to their future).
        gradient_checkpointing: Enable gradient checkpointing on the encoder.
    """

    mode_label: str = "CPC"
    mode_str: str = "CPC"
    uses_multi_view: bool = False

    def __init__(
        self,
        backbone: TransformerBackbone,
        gru_hidden_size: int = 256,
        gru_num_layers: int = 1,
        n_predictions: int = 12,
        min_context_frac: float = 0.25,
        temperature: float = 0.1,
        negative_scope: str = "xticker_xday",
        cosine_logits: bool = True,
        target_encoder: str = "stopgrad_bidir",
        gradient_checkpointing: bool = False,
    ):
        if not isinstance(backbone, TransformerBackbone):
            raise ValueError(
                f"CPC only supports TransformerBackbone, got {type(backbone).__name__}"
            )
        if backbone.pool == "cls":
            raise ValueError(
                "CPC requires pool='mean' or pool='max' (not 'cls'). "
                "forward_patches operates on raw patch positions without a CLS token."
            )
        if n_predictions < 1:
            raise ValueError(f"n_predictions must be >= 1, got {n_predictions}")
        if not (0.0 < min_context_frac < 1.0):
            raise ValueError(
                f"min_context_frac must be in (0, 1), got {min_context_frac}"
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if negative_scope not in ("xticker_xday", "all"):
            raise ValueError(
                f"negative_scope must be 'xticker_xday' or 'all', got {negative_scope!r}"
            )
        if target_encoder not in ("stopgrad_bidir", "shared_causal"):
            raise ValueError(
                f"target_encoder must be 'stopgrad_bidir' or 'shared_causal', "
                f"got {target_encoder!r}"
            )

        super().__init__()
        self.backbone = backbone
        self.d_embedding = backbone.d_embedding
        self.patch_size = backbone.patch_size
        self.n_features = backbone.n_features
        self.n_predictions = n_predictions
        self.min_context_frac = min_context_frac
        self.temperature = temperature
        self.negative_scope = negative_scope
        self.cosine_logits = cosine_logits
        self.target_encoder = target_encoder

        if gradient_checkpointing:
            backbone.gradient_checkpointing = True

        hidden_size = backbone.config.hidden_size  # z dim
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers

        self.context_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
        )

        self.predictive_heads = CPCPredictiveHeads(
            context_dim=gru_hidden_size,
            z_dim=hidden_size,
            n_predictions=n_predictions,
        )

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        tickers: list[str] | None = None,
        dates: list[str] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Encode → GRU context → predict future ``z`` → InfoNCE.

        Returns ``None`` when the batch's minimum valid-patch count is too
        small to carve out both a non-empty context prefix and ``K`` future
        target positions — the training step skips these rather than erroring.
        """
        B, _, T = x.shape
        device = x.device
        n_patches = (T + self.patch_size - 1) // self.patch_size

        if lengths is not None:
            n_valid = (lengths + self.patch_size - 1) // self.patch_size
        else:
            n_valid = torch.full((B,), n_patches, device=device, dtype=torch.long)

        K = self.n_predictions
        min_valid = int(n_valid.min().item())
        min_t_c = max(1, int(round(self.min_context_frac * min_valid)))
        max_t_c = min_valid - K
        if max_t_c < min_t_c:
            return None

        if self.training:
            t_c = int(torch.randint(min_t_c, max_t_c + 1, (1,)).item())
        else:
            t_c = max_t_c

        P = self.patch_size

        if self.target_encoder == "shared_causal":
            # --- Paper shape: ONE causal pass over context + target patches,
            # same encoder, gradients flowing into the targets. Each target
            # z_{t_c+k} sees only its own past, never its future.
            z_all = self.backbone.forward_patches(
                x[:, :, : (t_c + K) * P], lengths=None, causal=True,
            )  # (B, t_c + K, H)
            context_z = z_all[:, :t_c]
            targets = z_all[:, t_c:]
        else:
            # --- Context encoder: causal ViT over patches 0..t_c-1 ---
            # All samples have >= t_c+K valid patches by construction, so the
            # prefix slice is fully valid (no padding mask needed).
            context_x = x[:, :, : t_c * P]
            context_z = self.backbone.forward_patches(
                context_x, lengths=None, causal=True,
            )  # (B, t_c, H)

            # --- Target encoder: bidirectional ViT over future patches
            # t_c..t_c+K-1 --- shared weights with the context encoder
            # (initially); stop-gradient so the target representations act as
            # a fixed regression target.
            target_x = x[:, :, t_c * P : (t_c + K) * P]
            with torch.no_grad():
                targets = self.backbone.forward_patches(
                    target_x, lengths=None, causal=False, pos_offset=t_c,
                )  # (B, K, H)

        # --- Context GRU over context_z ---
        # GRU can be numerically finicky under bf16; cast to fp32 for stability.
        _, h_n = self.context_gru(context_z.float())
        c = h_n[-1].to(context_z.dtype)  # (B, gru_hidden_size)

        # --- K-step predictions via W_k c ---
        predictions = self.predictive_heads(c)  # (B, K, H)

        # --- InfoNCE: in-batch cross-sequence negatives, per future step ---
        # For each k: logits[i, j] = <pred[i, k], target[j, k]> / temperature

        if self.cosine_logits:
            # L2-normalize both sides so logits are cosine similarities. Without
            # this, absolute magnitude differences across samples (driven by
            # price scale, ticker identity, etc.) trivially identify the
            # positive regardless of predictive content.
            predictions = F.normalize(predictions, dim=-1)
            targets = F.normalize(targets, dim=-1)
            # (B, K, H) @ (K, H, B) → (B, K, B)
            logits = torch.einsum("bkh,nkh->bkn", predictions, targets) / self.temperature
        else:
            # Paper density-ratio form f_k = exp(z^T W_k c): raw dot products,
            # no temperature — W_k carries the scale.
            logits = torch.einsum("bkh,nkh->bkn", predictions, targets)

        # Restrict negatives to "different ticker AND different day". Off-diagonal
        # entries (i != j) where ticker[i] == ticker[j] OR date[i] == date[j] are
        # masked to -inf so they neither contribute to the softmax denominator nor
        # count as incorrect predictions. The diagonal (positives) is preserved.
        # negative_scope="all" skips the mask: the paper's in-batch scheme.
        valid_mask = torch.ones(B, B, dtype=torch.bool, device=device)
        if (self.negative_scope == "xticker_xday"
                and tickers is not None and dates is not None and B > 1):
            same_ticker = torch.tensor(
                [[t1 == t2 for t2 in tickers] for t1 in tickers],
                dtype=torch.bool, device=device,
            )
            same_date = torch.tensor(
                [[d1 == d2 for d2 in dates] for d1 in dates],
                dtype=torch.bool, device=device,
            )
            forbidden = same_ticker | same_date
            forbidden.fill_diagonal_(False)  # keep positives
            valid_mask = ~forbidden
        # Skip rows whose positive has no valid negatives (no contrastive signal).
        row_has_negative = (valid_mask.sum(dim=-1) > 1)  # >1 because diagonal counts
        if not row_has_negative.any():
            return None

        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~valid_mask.unsqueeze(1), neg_inf)

        labels = torch.arange(B, device=device).unsqueeze(1).expand(B, K)  # (B, K)
        keep = row_has_negative.unsqueeze(1).expand(B, K).reshape(-1)
        flat_logits = logits.reshape(B * K, B)[keep]
        flat_labels = labels.reshape(-1)[keep]
        loss = F.cross_entropy(flat_logits, flat_labels)

        with torch.no_grad():
            acc = (flat_logits.argmax(dim=-1) == flat_labels).float().mean()

        # Concatenate context + target latents along the time axis for downstream
        # collapse / std diagnostics. Detach so monitoring never affects gradients.
        z_full = torch.cat([context_z.detach(), targets.detach()], dim=1)

        return {
            "cpc_loss": loss,
            "cpc_acc": acc,
            "t_c": t_c,
            "_z": z_full,
            "_c": c,
            "_predictions": predictions,
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
                tickers = bucket.get("tickers")
                dates = bucket.get("dates")
                out = self(x, lengths, tickers=tickers, dates=dates)
                if out is None:
                    continue
                loss = out["cpc_loss"]

                if not torch.isfinite(loss):
                    continue

                (loss / grad_accum_steps).backward()
                batch_loss += loss.item() * x.shape[0]
                batch_n += x.shape[0]
                accs.append(out["cpc_acc"].item())
                all_z.append(out["_z"].detach())

        if batch_n == 0:
            return None

        metrics = {
            "train/loss": batch_loss / batch_n,
            "train/cpc_acc": float(np.mean(accs)) if accs else 0.0,
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
                tickers = bucket.get("tickers")
                dates = bucket.get("dates")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = self(x, lengths, tickers=tickers, dates=dates)
                if out is None:
                    continue
                loss = out["cpc_loss"]
                if torch.isfinite(loss):
                    losses.append(loss.item())
                    accs.append(out["cpc_acc"].item())
        return {
            "eval/cpc_loss": float(np.mean(losses)) if losses else float("nan"),
            "eval/cpc_acc": float(np.mean(accs)) if accs else float("nan"),
        }

    def post_training_step(self, completed_steps, max_train_steps):
        return {}

    def describe_parameters(self):
        from market_jepa.training.utils import count_parameters

        param_counts = count_parameters(
            self,
            backbone=self.backbone,
            context_gru=self.context_gru,
            predictive_heads=self.predictive_heads,
        )
        summary = (
            f"Data dimensions:\n  n_features: {self.backbone.n_features}\n"
            f"CPC model parameters:\n"
            f"  Encoder (backbone): {param_counts['backbone']:,}\n"
            f"  Context GRU: {param_counts['context_gru']:,}\n"
            f"  Predictive heads (K={self.n_predictions}): {param_counts['predictive_heads']:,}\n"
            f"  Total: {param_counts['total']:,}\n"
            f"  K: {self.n_predictions} | min_context_frac: {self.min_context_frac} | "
            f"temperature: {self.temperature}\n"
            f"  negatives: {self.negative_scope} | cosine_logits: {self.cosine_logits} | "
            f"target_encoder: {self.target_encoder}"
        )
        return param_counts, summary

    def default_run_name(self, backbone_type, cfg):
        bb_cfg = backbone_block(cfg)
        return "__".join(
            [
                "mode=cpc",
                f"bb={backbone_type}",
                f"d_emb={bb_cfg.d_embedding}",
                f"K={self.n_predictions}",
                f"gru={self.gru_hidden_size}",
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
                f"Unsupported backbone type for CPC.save_pretrained: "
                f"{type(self.backbone).__name__}"
            )

        config = {
            "class": "CPC",
            "backbone_type": backbone_type,
            "n_features": self.backbone.n_features,
            "d_embedding": self.backbone.d_embedding,
            "pool": self.backbone.pool,
            "backbone_config": asdict(self.backbone.config),
            "gru_hidden_size": self.gru_hidden_size,
            "gru_num_layers": self.gru_num_layers,
            "n_predictions": self.n_predictions,
            "min_context_frac": self.min_context_frac,
            "temperature": self.temperature,
            "negative_scope": self.negative_scope,
            "cosine_logits": self.cosine_logits,
            "target_encoder": self.target_encoder,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        model_to_save = self._orig_mod if hasattr(self, "_orig_mod") else self
        torch.save(model_to_save.state_dict(), os.path.join(path, "model.pt"))

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "CPC":
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
            gru_hidden_size=config["gru_hidden_size"],
            gru_num_layers=config["gru_num_layers"],
            n_predictions=config["n_predictions"],
            min_context_frac=config["min_context_frac"],
            temperature=config["temperature"],
            # .get: checkpoints saved before these were knobs used the
            # (adapted) defaults, so that is what absence means.
            negative_scope=config.get("negative_scope", "xticker_xday"),
            cosine_logits=config.get("cosine_logits", True),
            target_encoder=config.get("target_encoder", "stopgrad_bidir"),
        )
        model.load_state_dict(_sd)
        return model