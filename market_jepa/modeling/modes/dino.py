"""DINO model for time series self-supervised learning.

Reference: Caron et al., 2021. Emerging Properties in Self-Supervised
Vision Transformers. https://arxiv.org/abs/2104.14294

DINO uses a LeJEPA-style multi-view augmentation pipeline (global + local
crops) combined with an EMA target encoder (the "teacher") as in I-JEPA.
The student processes all views; the teacher processes only the global
views. Loss is the cross-entropy between the teacher's centered + sharpened
softmax distribution and the student's softmax over a learned set of K
prototype directions.
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

from .base import TrainingModel
from ..backbones import backbone_kwargs_from_state_dict, create_backbone


class DINOHead(nn.Module):
    """DINO projection head.

    A small MLP followed by L2-normalization of the bottleneck and a linear
    layer onto K prototype directions. The prototype matrix's rows are
    re-normalized at every forward (matching weight_norm with ``g`` frozen
    to 1, which is what the original DINO uses).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 4096,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        if n_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        with torch.no_grad():
            self.last_layer.weight.copy_(
                F.normalize(self.last_layer.weight, dim=-1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        w = F.normalize(self.last_layer.weight, dim=-1)
        return F.linear(x, w)


class _DINOStudent(nn.Module):
    """Owning module for backbone + head; deepcopied by EMA to form the teacher."""

    def __init__(self, backbone: nn.Module, head: DINOHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        return self.head(self.backbone(x, lengths))


class DINO(TrainingModel):
    """DINO for time series self-supervised learning.

    Multi-view (global + local) augmentation as in LeJEPA, with an
    EMA-momentum teacher network as in I-JEPA. Loss is cross-entropy
    between centered+sharpened teacher probabilities (over global views)
    and student softmax probabilities (over all views, teacher's own
    view excluded).

    Args:
        backbone: TimeSeriesBackbone used as the student backbone. The
            teacher is a deep copy maintained by EMA.
        proj_dim: K — number of prototype directions in the head output.
        proj_hidden: Hidden dimension of the projection MLP.
        proj_bottleneck: Bottleneck dimension before the final prototype layer.
        proj_n_layers: Number of MLP layers in the head (>=1).
        student_temp: Softmax temperature for the student.
        teacher_temp: Softmax temperature for the teacher (lower = sharper).
        center_momentum: EMA momentum for the centering buffer.
        ema_start: Initial teacher EMA decay (linearly ramped to ``ema_end``).
        ema_end: Final teacher EMA decay.
    """

    def __init__(
        self,
        backbone: nn.Module,
        proj_dim: int = 4096,
        proj_hidden: int = 2048,
        proj_bottleneck: int = 256,
        proj_n_layers: int = 3,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
        ema_start: float = 0.996,
        ema_end: float = 1.0,
        gradient_checkpointing: bool = False,
    ):
        if gradient_checkpointing:
            raise ValueError(
                "gradient_checkpointing is not supported on DINO."
            )
        super().__init__()
        self.d_embedding = backbone.d_embedding

        head = DINOHead(
            in_dim=backbone.d_embedding,
            out_dim=proj_dim,
            hidden_dim=proj_hidden,
            bottleneck_dim=proj_bottleneck,
            n_layers=proj_n_layers,
        )
        self.student = _DINOStudent(backbone, head)

        # Stash architectural args for save_pretrained.
        self._proj_dim = proj_dim
        self._proj_hidden = proj_hidden
        self._proj_bottleneck = proj_bottleneck
        self._proj_n_layers = proj_n_layers

        # Teacher = EMA copy of student. We bypass ema-pytorch's internal
        # schedule by passing current_decay= directly in post_training_step.
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

        # Loss config + centering buffer.
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, proj_dim))

    # -- backbone / head accessors -------------------------------------------
    @property
    def backbone(self) -> nn.Module:
        return self.student.backbone

    @property
    def head(self) -> DINOHead:
        return self.student.head

    mode_label: str = "DINO"
    mode_str: str = "DINO"
    uses_multi_view: bool = True

    def train(self, mode: bool = True):
        """Keep teacher in eval mode so DropPath etc. stay disabled."""
        super().train(mode)
        self.ema.ema_model.eval()
        return self

    # -- evaluation embeddings -----------------------------------------------
    def encode(self, x, lengths=None):
        """Use the teacher (EMA) backbone to produce embeddings.

        Following I-JEPA's convention: the EMA target encoder is a more
        stable representation and is the standard choice for downstream
        evaluation in EMA-based SSL.
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

    # -- forward pass --------------------------------------------------------
    def forward(self, x, lengths=None):
        return self.encode(x, lengths)

    # -- DINO loss -----------------------------------------------------------
    def compute_loss(
        self,
        student_logits: list[torch.Tensor],
        teacher_logits: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Multi-view cross-entropy loss.

        For each (teacher_global_view, student_view) pair where the views
        differ, accumulate cross-entropy with the teacher's centered +
        sharpened distribution as target.

        Args:
            student_logits: list of (B, K), one per student view (all views).
            teacher_logits: list of (B, K), one per teacher view (global only).
        """
        # Center + sharpen teacher; detach so no gradient flows through it.
        teacher_probs = [
            F.softmax((tl - self.center) / self.teacher_temp, dim=-1).detach()
            for tl in teacher_logits
        ]
        student_log_probs = [
            F.log_softmax(sl / self.student_temp, dim=-1) for sl in student_logits
        ]

        n_global = len(teacher_logits)
        total = 0.0
        n_terms = 0
        for ig in range(n_global):
            tprobs = teacher_probs[ig]
            for iv, slogp in enumerate(student_log_probs):
                if iv == ig:
                    continue  # skip the same view as the teacher
                total = total + -(tprobs * slogp).sum(dim=-1).mean()
                n_terms += 1

        loss = total / max(n_terms, 1)
        return {
            "dino_loss": loss,
            "_teacher_logits_cat": torch.cat(
                [tl.detach() for tl in teacher_logits], dim=0
            ),
        }

    @torch.no_grad()
    def _update_center(self, teacher_outputs: torch.Tensor) -> None:
        """EMA update of the centering buffer."""
        batch_center = teacher_outputs.float().mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(
            batch_center, alpha=1 - self.center_momentum
        )

    # -- training / eval steps ----------------------------------------------
    def training_step(self, batch, device, grad_accum_steps=1):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            student_logits_all: list[torch.Tensor] = []
            teacher_logits_all: list[torch.Tensor] = []
            for bucket in batch["buckets"]:
                views = [v.to(device) for v in bucket["views"]]
                lengths = [l.to(device) for l in bucket["lengths"]]
                n_global = bucket.get("n_global_views", len(views))

                with torch.no_grad():
                    teacher_logits_all.extend(
                        self.ema.ema_model(views[i], lengths[i])
                        for i in range(n_global)
                    )
                student_logits_all.extend(
                    self.student(views[i], lengths[i]) for i in range(len(views))
                )

            if not student_logits_all:
                return None

            output = self.compute_loss(student_logits_all, teacher_logits_all)
            loss = output["dino_loss"]

        if not torch.isfinite(loss):
            return None

        (loss / grad_accum_steps).backward()

        # Update centering buffer with teacher outputs from this micro-batch.
        self._update_center(output["_teacher_logits_cat"])

        metrics = {"train/loss": loss.item()}
        return {"loss": loss.item(), "metrics": metrics}

    @torch.no_grad()
    def eval_step(self, eval_batches, device):
        self.eval()
        losses = []
        for batch in eval_batches:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                student_logits_all: list[torch.Tensor] = []
                teacher_logits_all: list[torch.Tensor] = []
                for bucket in batch["buckets"]:
                    views = [v.to(device) for v in bucket["views"]]
                    lengths = [l.to(device) for l in bucket["lengths"]]
                    n_global = bucket.get("n_global_views", len(views))
                    teacher_logits_all.extend(
                        self.ema.ema_model(views[i], lengths[i])
                        for i in range(n_global)
                    )
                    student_logits_all.extend(
                        self.student(views[i], lengths[i]) for i in range(len(views))
                    )
                if not student_logits_all:
                    continue
                if len(teacher_logits_all) == 0 or len(student_logits_all) <= 1:
                    # Not enough views to form an off-diagonal pair.
                    continue
                out = self.compute_loss(student_logits_all, teacher_logits_all)
            if torch.isfinite(out["dino_loss"]):
                losses.append(out["dino_loss"].item())
        return {"eval/dino_loss": float(np.mean(losses)) if losses else float("nan")}

    # -- run-name / param summary -------------------------------------------
    def default_run_name(self, backbone_type, cfg):
        return "__".join(
            [
                "mode=dino",
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
            head=self.head,
        )
        n_teacher = sum(p.numel() for p in self.ema.ema_model.parameters())
        lines = [
            f"Data dimensions:\n  n_features: {self.backbone.n_features}",
            "DINO model parameters:",
            f"  Student backbone: {param_counts['backbone']:,}",
            f"  Student head: {param_counts['head']:,}",
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
            "class": "DINO",
            "backbone_type": backbone_type,
            "n_features": backbone.n_features,
            "d_embedding": backbone.d_embedding,
            "proj_dim": self._proj_dim,
            "proj_hidden": self._proj_hidden,
            "proj_bottleneck": self._proj_bottleneck,
            "proj_n_layers": self._proj_n_layers,
            "student_temp": self.student_temp,
            "teacher_temp": self.teacher_temp,
            "center_momentum": self.center_momentum,
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
    def from_pretrained(cls, path: str, **kwargs) -> "DINO":
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
            proj_hidden=config.get("proj_hidden", 2048),
            proj_bottleneck=config.get("proj_bottleneck", 256),
            proj_n_layers=config.get("proj_n_layers", 3),
            student_temp=config.get("student_temp", 0.1),
            teacher_temp=config.get("teacher_temp", 0.04),
            center_momentum=config.get("center_momentum", 0.9),
            ema_start=config.get("ema_start", 0.996),
            ema_end=config.get("ema_end", 1.0),
        )
        model.load_state_dict(_sd)
        return model
