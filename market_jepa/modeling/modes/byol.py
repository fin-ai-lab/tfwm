"""BYOL model for time series self-supervised learning.

Reference: Grill et al., 2020. Bootstrap Your Own Latent: A New Approach to
Self-Supervised Learning. https://arxiv.org/abs/2006.07733

Closer to LeJEPA in spirit: multi-view augmentation (global + local crops)
with a projection MLP, but instead of SIGReg on a shared encoder we use a
BYOL-style asymmetric student/teacher pair. The teacher is an EMA of the
student backbone+projector; only the student carries an extra predictor
head. Loss is normalized MSE (equivalent to ``2 - 2·cos_sim``) between the
student prediction and the stop-gradient teacher projection.
"""


from market_jepa.backbone_config import backbone_block
import json
import os
import warnings
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ema_pytorch import EMA
from torchvision.ops import MLP

from .base import TrainingModel
from ..backbones import backbone_kwargs_from_state_dict, create_backbone
from .utils import RMSNorm


class _BYOLStudent(nn.Module):
    """Owning module for backbone + projector; deepcopied by EMA to form the teacher.

    The predictor lives on the BYOL parent (not here) so the teacher is a
    copy of just (backbone, projector) — preserving BYOL's asymmetry.
    """

    def __init__(self, backbone: nn.Module, projector: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.projector = projector

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        return self.projector(self.backbone(x, lengths))


class BYOL(TrainingModel):
    """BYOL for time series self-supervised learning.

    Multi-view augmentation (global + local) as in LeJEPA, with an EMA
    teacher network as in I-JEPA. The student processes all views and
    runs a predictor on top of its projection; the teacher (EMA of
    student backbone+projector) processes only global views and is the
    stop-gradient target.

    Args:
        backbone: TimeSeriesBackbone used as the student backbone. The
            teacher is a deep copy maintained by EMA.
        proj_dim: Output dimension of the projector / predictor.
        proj_hidden: Hidden dimension of the projector MLP.
        pred_hidden: Hidden dimension of the predictor MLP.
        ema_start: Initial teacher EMA decay (linearly ramped to ``ema_end``).
        ema_end: Final teacher EMA decay.
    """

    def __init__(
        self,
        backbone: nn.Module,
        proj_dim: int = 256,
        proj_hidden: int = 4096,
        pred_hidden: int = 4096,
        ema_start: float = 0.996,
        ema_end: float = 1.0,
        gradient_checkpointing: bool = False,
    ):
        if gradient_checkpointing:
            raise ValueError(
                "gradient_checkpointing is not supported on BYOL."
            )
        super().__init__()
        self.d_embedding = backbone.d_embedding

        projector = MLP(
            backbone.d_embedding,
            [proj_hidden, proj_dim],
            norm_layer=RMSNorm,
        )
        self.student = _BYOLStudent(backbone, projector)

        # Predictor lives only on the student side (BYOL asymmetry).
        self.predictor = MLP(
            proj_dim,
            [pred_hidden, proj_dim],
            norm_layer=RMSNorm,
        )

        # Teacher = EMA copy of (backbone + projector).
        self._ema_start = ema_start
        self._ema_end = ema_end
        self.ema = EMA(
            self.student,
            beta=ema_start,
            update_after_step=0,
            update_every=1,
            include_online_model=False,
        )
        self.ema.ema_model.eval()
        for p in self.ema.ema_model.parameters():
            p.requires_grad = False

        # Stash architectural args for save_pretrained.
        self._proj_dim = proj_dim
        self._proj_hidden = proj_hidden
        self._pred_hidden = pred_hidden

    # -- backbone / projector accessors --------------------------------------
    @property
    def backbone(self) -> nn.Module:
        return self.student.backbone

    @property
    def projector(self) -> nn.Module:
        return self.student.projector

    mode_label: str = "BYOL"
    mode_str: str = "BYOL"
    uses_multi_view: bool = True

    def train(self, mode: bool = True):
        """Keep teacher in eval mode so DropPath etc. stay disabled."""
        super().train(mode)
        self.ema.ema_model.eval()
        return self

    # -- evaluation embeddings -----------------------------------------------
    def encode(self, x, lengths=None):
        """Use the teacher (EMA) backbone to produce embeddings.

        Mirrors I-JEPA / DINO: the EMA target is a more stable
        representation and is the standard choice for downstream eval.
        """
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
            self.ema.ema_model.backbone(view, vl)
            for view, vl in zip(views, view_lengths)
        ]
        return {"embeddings": torch.stack(embeddings, dim=1)}

    def forward(self, x, lengths=None):
        return self.encode(x, lengths)

    # -- BYOL loss -----------------------------------------------------------
    def compute_loss(
        self,
        student_preds: list[torch.Tensor],
        teacher_projs: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Multi-view BYOL loss: ``2 - 2·cos_sim``, averaged over (teacher
        global view, student view) pairs where the views differ.

        Args:
            student_preds: list of (B, D), predictor outputs — one per view (all views).
            teacher_projs: list of (B, D), projector outputs — one per global view (no grad).
        """
        student_normed = [F.normalize(p, dim=-1) for p in student_preds]
        teacher_normed = [F.normalize(t.detach(), dim=-1) for t in teacher_projs]

        n_global = len(teacher_projs)
        total = 0.0
        n_terms = 0
        for ig in range(n_global):
            t = teacher_normed[ig]
            for iv, s in enumerate(student_normed):
                if iv == ig:
                    continue
                # Equivalent to ||s - t||^2 for unit vectors.
                total = total + (2.0 - 2.0 * (s * t).sum(dim=-1).mean())
                n_terms += 1

        loss = total / max(n_terms, 1)
        return {"byol_loss": loss}

    # -- training / eval steps ----------------------------------------------
    def training_step(self, batch, device, grad_accum_steps=1):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            student_preds_all: list[torch.Tensor] = []
            teacher_projs_all: list[torch.Tensor] = []
            for bucket in batch["buckets"]:
                views = [v.to(device) for v in bucket["views"]]
                lengths = [l.to(device) for l in bucket["lengths"]]
                n_global = bucket.get("n_global_views", len(views))

                with torch.no_grad():
                    teacher_projs_all.extend(
                        self.ema.ema_model(views[i], lengths[i])
                        for i in range(n_global)
                    )
                # Student: backbone → projector → predictor.
                student_preds_all.extend(
                    self.predictor(self.student(views[i], lengths[i]))
                    for i in range(len(views))
                )

            if not student_preds_all:
                return None

            output = self.compute_loss(student_preds_all, teacher_projs_all)
            loss = output["byol_loss"]

        if not torch.isfinite(loss):
            return None

        (loss / grad_accum_steps).backward()

        metrics = {"train/loss": loss.item()}
        return {"loss": loss.item(), "metrics": metrics}

    @torch.no_grad()
    def eval_step(self, eval_batches, device):
        self.eval()
        losses = []
        for batch in eval_batches:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                student_preds_all: list[torch.Tensor] = []
                teacher_projs_all: list[torch.Tensor] = []
                for bucket in batch["buckets"]:
                    views = [v.to(device) for v in bucket["views"]]
                    lengths = [l.to(device) for l in bucket["lengths"]]
                    n_global = bucket.get("n_global_views", len(views))
                    teacher_projs_all.extend(
                        self.ema.ema_model(views[i], lengths[i])
                        for i in range(n_global)
                    )
                    student_preds_all.extend(
                        self.predictor(self.student(views[i], lengths[i]))
                        for i in range(len(views))
                    )
                if not student_preds_all or len(teacher_projs_all) == 0:
                    continue
                if len(student_preds_all) <= 1:
                    # Not enough views to form an off-diagonal pair.
                    continue
                out = self.compute_loss(student_preds_all, teacher_projs_all)
            if torch.isfinite(out["byol_loss"]):
                losses.append(out["byol_loss"].item())
        return {"eval/byol_loss": float(np.mean(losses)) if losses else float("nan")}

    # -- run-name / param summary -------------------------------------------
    def default_run_name(self, backbone_type, cfg):
        return "__".join(
            [
                "mode=byol",
                f"bb={backbone_type}",
                f"d_emb={backbone_block(cfg).d_embedding}",
                f"blr={(cfg.optimizer.blr or cfg.mode.training_overrides.blr):.1e}",
                f"bs={cfg.training.per_device_train_batch_size}",
                f"steps={cfg.training.max_train_steps}",
                f"ema={self._ema_start:.4f}",
            ]
        )

    def describe_parameters(self):
        from market_jepa.training.utils import count_parameters

        param_counts = count_parameters(
            self,
            backbone=self.backbone,
            projector=self.projector,
            predictor=self.predictor,
        )
        n_teacher = sum(p.numel() for p in self.ema.ema_model.parameters())
        lines = [
            f"Data dimensions:\n  n_features: {self.backbone.n_features}",
            "BYOL model parameters:",
            f"  Student backbone: {param_counts['backbone']:,}",
            f"  Student projector: {param_counts['projector']:,}",
            f"  Predictor: {param_counts['predictor']:,}",
            f"  Total (trainable): {param_counts['total']:,}",
            f"  Teacher (frozen, EMA): {n_teacher:,}",
        ]
        return param_counts, "\n".join(lines)

    def post_training_step(self, completed_steps, max_train_steps):
        # Linear momentum schedule: ema_start → ema_end over training.
        frac = completed_steps / max(1, max_train_steps - 1)
        momentum = self._ema_start + frac * (self._ema_end - self._ema_start)
        momentum = min(momentum, 1.0)
        self.ema.update_moving_average(
            self.ema.ema_model, self.ema.model, current_decay=momentum
        )
        return {"train/ema_momentum": momentum}

    # -- save / load --------------------------------------------------------
    def save_pretrained(self, path: str) -> None:
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

        config: dict = {
            "class": "BYOL",
            "backbone_type": backbone_type,
            "n_features": backbone.n_features,
            "d_embedding": backbone.d_embedding,
            "proj_dim": self._proj_dim,
            "proj_hidden": self._proj_hidden,
            "pred_hidden": self._pred_hidden,
            "ema_start": self._ema_start,
            "ema_end": self._ema_end,
        }

        if backbone_type == "transformer":
            config["pool"] = backbone.pool
            config["backbone_config"] = asdict(backbone.config)
        elif backbone_type == "resnet":
            config["pool"] = backbone.pool
            config["resnet_config"] = {
                "variant": next(
                    k
                    for k, (bc, nb) in backbone.CONFIGS.items()
                    if bc == type(backbone.layer1[0])
                    and nb
                    == [
                        len(backbone.layer1),
                        len(backbone.layer2),
                        len(backbone.layer3),
                        len(backbone.layer4),
                    ]
                ),
                "initial_channels": backbone.stem[0].out_channels,
            }

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "BYOL":
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        config.pop("class", None)
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
            proj_hidden=config.get("proj_hidden", 4096),
            pred_hidden=config.get("pred_hidden", 4096),
            ema_start=config.get("ema_start", 0.996),
            ema_end=config.get("ema_end", 1.0),
        )
        model.load_state_dict(_sd)
        return model
