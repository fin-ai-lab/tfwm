"""Prediction head factory for supervised baselines.

The head shape is chosen by the LOSS, not by the task: every task is the same
(target_type, horizon) pair, and the supervised ablation varies only what the
head is trained against.

  - scalar losses (mse, smooth_l1, corr, pairwise) -> :class:`RegressionHead`
  - binned losses (cross_entropy and the expected-bin penalties) ->
    :class:`ClassificationHead` over ``n_bins`` ordered bins

A binned head is still scored by rank IC: its scalar readout is the expected
bin ``E[c] = sum_c c * softmax(logits)_c``, which is what makes a k-way
classifier rankable against a continuous target.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.ops import MLP

from market_jepa.modeling.modes.utils import RMSNorm
from market_jepa.eval.tasks import TaskSpec


class RegressionHead(nn.Module):
    """MLP head that outputs a scalar prediction per sample."""

    def __init__(self, d_embedding: int, hidden: list[int] | None = None):
        super().__init__()
        if hidden is None:
            hidden = [d_embedding * 2]
        self.mlp = MLP(d_embedding, [*hidden, 1], norm_layer=RMSNorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (batch, d_embedding)
        Returns:
            (batch,) scalar predictions
        """
        return self.mlp(x).squeeze(-1)


class LatentPredictorHead(nn.Module):
    """MLP head that outputs another LATENT, not a prediction of the target.

    The world model's transition function (plots/event_conditioning): it maps
    a day's frozen-encoder CLS token to the next day's, and a fixed ridge probe
    decodes that to a return. Same MLP family as :class:`RegressionHead` — one
    hidden layer at ``2 x d``, RMSNorm — so ``EventPredictor`` splits its
    ``mlp[0]`` and injects the event embedding at exactly the same junction;
    only the output width differs.
    """

    def __init__(self, d_embedding: int, hidden: list[int] | None = None,
                 d_in: int | None = None):
        """Args:
            d_embedding: latent width, and the OUTPUT width — the head predicts
                a latent, so the ridge probe reads it unchanged.
            d_in: input width when extra STATE is concatenated to the latent
                (the overnight gap, say). Defaults to ``d_embedding``. Input
                and output widths are allowed to differ because the residual
                path adds ``z_t`` back, and ``z_t`` is ``d_embedding`` wide
                whatever else was fed alongside it.
        """
        super().__init__()
        if hidden is None:
            hidden = [d_embedding * 2]
        self.mlp = MLP(d_in or d_embedding, [*hidden, d_embedding],
                       norm_layer=RMSNorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (batch, d_embedding)
        Returns:
            (batch, d_embedding) the predicted next-day latent
        """
        return self.mlp(x)


class ClassificationHead(nn.Module):
    """MLP head that outputs logits over ``n_classes`` ordered bins."""

    def __init__(
        self, d_embedding: int, n_classes: int, hidden: list[int] | None = None,
    ):
        super().__init__()
        if hidden is None:
            hidden = [d_embedding * 2]
        self.mlp = MLP(d_embedding, [*hidden, n_classes], norm_layer=RMSNorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (batch, d_embedding)
        Returns:
            (batch, n_classes) logits
        """
        return self.mlp(x)


def expected_bin(logits: torch.Tensor) -> torch.Tensor:
    """Scalar readout of a binned head: ``sum_c c * p_c`` over raw bin indices.

    This is the quantity a binned supervised run is ranked by, and the same
    quantity the expected-bin penalties regress. Computed in float32 so it is
    stable under autocast.
    """
    classes = torch.arange(
        logits.shape[-1], device=logits.device, dtype=torch.float32,
    )
    return (torch.softmax(logits.float(), dim=-1) * classes).sum(dim=-1)


def create_prediction_head(
    task_spec: TaskSpec, d_embedding: int, n_bins: int | None = None,
) -> nn.Module:
    """Construct the prediction head for a task.

    ``n_bins`` is None for the scalar losses and an integer for the binned
    ones; the task itself no longer carries a head shape.
    """
    del task_spec  # kept for call-site symmetry; the loss picks the shape
    if n_bins is None:
        return RegressionHead(d_embedding)
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    return ClassificationHead(d_embedding, n_bins)


class SkipRegressionHead(nn.Module):
    """:class:`RegressionHead` plus a LINEAR SKIP, so a ridge probe can be the init.

    WHY A SKIP AND NOT JUST A LINEAR HEAD. The thing we want to start from is a
    ridge probe, which is linear; the head the supervised arms train is an MLP.
    Replacing the MLP with a linear layer would make the ridge init exact but
    would also change the head's capacity, so a finetune could differ from the
    supervised recipe in two things at once. Here the skip carries the ridge and
    the MLP branch starts at ZERO, so::

        step 0:  out == skip(x) == the ridge probe, exactly
        after :  the MLP branch is free to grow whatever the ridge cannot say

    The zero is on the MLP's FINAL layer only. Its earlier layers keep their
    normal random init, so the branch has gradient from the first step (a fully
    zeroed branch would be stuck: every weight's gradient runs through the zero
    output layer). This is the standard zero-init-residual construction.

    WHY `gain` IS A BUFFER AND NOT FOLDED INTO `skip`. The init's output spread
    has to be O(1) for the pairwise loss (see :func:`load_ridge_init`), but the
    ridge direction's own magnitude is O(1) in PARAMETER space, and those two
    facts pull opposite ways. Folding the rescale into `skip.weight` satisfies
    the first and destroys the second: it inflated the skip's RMS to 3.78
    against the backbone's 0.136, and because Adam steps every parameter by
    ~lr REGARDLESS of its magnitude, the head's relative step became 28x
    smaller than the encoder's. Measured consequence, over a full 3,526-step
    run at blr 1e-5 (wave 047986, 93 runs): `cos(w_init, w_final)` = 1.000000
    and the norm change was 0.99837 against 0.998362 predicted by weight decay
    ALONE -- the head's entire net movement was decay, and the gradient did
    nothing. The encoder then drifted 0.0044 out from under a readout that
    could not follow, which is the IC valley.

    A NON-TRAINABLE gain keeps the output spread where the loss wants it and
    the parameters where the optimizer wants them, so one shared LR moves the
    head and the encoder at comparable RELATIVE rates -- which is the regime
    the supervised arm is already in (its head's RMS is 0.43x its backbone's).
    `tests/test_ssl_finetune.py` pins that ratio.

    NOT a drop-in replacement for :class:`RegressionHead`. The skip is extra
    parameters and extra state-dict keys, so a run that switches heads is not
    weight-compatible with one that did not. It is opt-in
    (``mode.init_head_from``) for exactly that reason -- the reported supervised
    arms keep the plain MLP head they were trained with.
    """

    def __init__(self, d_embedding: int, hidden: list[int] | None = None):
        super().__init__()
        if hidden is None:
            hidden = [d_embedding * 2]
        self.mlp = MLP(d_embedding, [*hidden, 1], norm_layer=RMSNorm)
        self.skip = nn.Linear(d_embedding, 1)
        # OUTPUT SCALE LIVES HERE, NOT IN THE WEIGHTS. See the class docstring:
        # folding it into `skip` freezes the head. 1.0 is also exactly right
        # for a checkpoint written before this buffer existed, which has the
        # factor folded in -- so an old state_dict loads and means what it
        # meant, and `strict=False` on a missing key is not a silent rescale.
        self.register_buffer("gain", torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: (batch, d_embedding)
        Returns:
            (batch,) scalar predictions
        """
        return (self.gain * (self.skip(x) + self.mlp(x))).squeeze(-1)

    @torch.no_grad()
    def init_from_ridge(
        self, weight: torch.Tensor, bias: float, gain: float = 1.0,
    ) -> None:
        """Set the skip to a folded ridge probe and zero the MLP branch's output.

        ``weight`` is (d_embedding,) and ``bias`` a scalar, both ALREADY FOLDED
        through the probe's standardizer -- see
        :func:`scripts.eval.fit_ridge_head_init.fold_ridge`. Folding is not
        optional: ``ColumnwiseRidge`` is standardize-then-ridge, and a layer
        built from ``coef_`` alone looks plausible and predicts nothing.
        """
        if weight.shape != self.skip.weight.shape[1:]:
            raise ValueError(
                f"ridge weight is {tuple(weight.shape)}, head expects "
                f"{tuple(self.skip.weight.shape[1:])}"
            )
        self.skip.weight.copy_(weight.reshape(1, -1))
        self.skip.bias.fill_(float(bias))
        self.gain.fill_(float(gain))
        last = [m for m in self.mlp if isinstance(m, nn.Linear)][-1]
        last.weight.zero_()
        last.bias.zero_()


def load_ridge_init(
    path: str, task: str, scale: str = "unit",
) -> tuple[torch.Tensor, float]:
    """Read a folded ridge probe written by ``scripts/eval/fit_ridge_head_init.py``.

    ``path`` is that script's per-month directory; the task's ``<task>.npz``
    inside it holds ``weight`` (d_embedding,), ``bias`` and ``pred_std``.

    ``scale`` decides what the init's OUTPUT SPREAD is, and it matters because
    the objective is ``pairwise`` (RankNet), which reads only DIFFERENCES
    between predictions:

      ``raw``   the probe unchanged. Faithful, and the head's output is then
                bit-identical to the reported probe -- but its spread is
                IC-sized. Measured on pair_warp_6mo/2008-08: 0.0146, 0.0283,
                0.0297 against a target whose own std is 0.2883. Pairs that
                close together sit at softplus' collapse point, so the first
                thing training does is inflate the scale, and until it has, the
                gradient reaching the BACKBONE is proportionally small.
      ``unit``  the same direction rescaled to unit spread on the fit pool.
                The DEFAULT. A rank loss is invariant to a positive rescaling
                of the ordering, so this throws away nothing the ridge knew --
                the probe's content here is entirely its direction -- and it
                starts the head in a well-conditioned part of the loss.

    Returns ``(weight, bias, gain)``. THE GAIN IS NOT MULTIPLIED IN: it is the
    caller's job to hand it to :meth:`SkipRegressionHead.init_from_ridge`,
    which parks it in a non-trainable buffer. Folding it into the weight gives
    the same step-0 predictions and then freezes the head -- see
    :class:`SkipRegressionHead`.

    The bias is carried through both ways for completeness; RankNet ignores it
    outright (it cancels in every difference), so it only matters if this init
    is ever reused under a pointwise loss.
    """
    import numpy as np

    f = Path(path) / f"{task}.npz"
    if not f.exists():
        raise FileNotFoundError(f"init_head_from: no {task}.npz in {path}")
    with np.load(f, allow_pickle=False) as z:
        weight = np.asarray(z["weight"], dtype=np.float32)
        bias = float(z["bias"])
        pred_std = float(z["pred_std"])
        stamp = str(z["readout"]) if "readout" in z else None
    # THE PROBE MUST BE FOR THE FEATURE SPACE THE FINETUNE READS. Prediction
    # is scored at the last token, and a probe fit on MEAN-pooled embeddings
    # describes a different space -- but it is the same shape, so it loads,
    # trains, and yields a plausible curve for the wrong reason. The first
    # wave of head inits was mean-pooled (the probe-breadth sweep read SSL
    # arms at their own training pool until 2026-09-17) and had to be
    # discarded. Unstamped files predate the fix and are refused for the same
    # reason: their readout is unknown, and for an SSL arm it was the mean.
    if stamp != "last":
        raise ValueError(
            f"init_head_from: {f} was fit at readout "
            f"{stamp or 'unstamped/unknown'}, but the finetune reads the "
            f"encoder at 'last'. Refit with "
            f"scripts/eval/fit_ridge_head_init.py against last-pooled "
            f"embeddings."
        )
    if scale == "raw":
        factor = 1.0
    elif scale == "unit":
        if not pred_std > 1e-12:
            raise ValueError(
                f"init_head_from: {f} has pred_std={pred_std:g}; the probe is "
                "constant and cannot be rescaled to unit spread"
            )
        factor = 1.0 / pred_std
    else:
        raise ValueError(f"head_init_scale must be 'raw' or 'unit', got {scale!r}")
    # THE FACTOR IS RETURNED, NOT MULTIPLIED IN. It is an output-scale
    # convention, and a weight is the one place it must not live -- see
    # SkipRegressionHead. The direction and bias stay at the ridge's own
    # magnitude, which is the encoder's.
    return torch.from_numpy(weight), bias, factor
