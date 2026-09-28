"""Fit a prediction head on FROZEN embeddings, offline.

WHY THIS EXISTS. ``market_jepa.modeling.event_predictor.EventPredictor``
splices a text branch into a *trained head's* hidden layer, and a ridge has no
hidden layer to splice into. So the event-conditioning market pathway needs an
actual head module, and it needs one that is not worse than the probe the paper
reports.

SCOPE — WHAT THIS IS NOT FOR. The SSL finetune arm keeps the head its own
training run produced. That arm's claim is about what finetuning a frozen
backbone achieves, so its head has to be the head that training produced;
swapping in one fitted here would change what the number means and break the
symmetry with the supervised arm. Nothing on that path imports this module, and
nothing here is on the reporting path for any arm. This fits heads for event
conditioning, and it answers the readout question below.

And the head the supervised runs already ship is, for ONE task, worse than the
ridge it is supposed to beat. Measured over the 203 months of
``supervised-full-month-*-ce`` (head IC minus probe IC, per month):

    return          -0.0027 +- 0.0007     head wins  79/203
    volatility      +0.0048 +- 0.0009     head wins 142/203
    spread          +0.0391 +- 0.0059     head wins 126/203

So the head is fine on two tasks and loses on return, which is the task whose
signal is smallest and whose z-target is hardest for a pointwise loss (see
``supervised._neg_corr_loss``). A head that cannot match a linear probe on
return makes the event-conditioning return channel start from a handicap.

WHAT THIS MODULE DOES. It fits the same head modules
(``market_jepa.eval.heads``) on embeddings that are already computed, so the
predictor can be swept over objectives and batchings in seconds per fit rather
than per training run. Three axes:

``family``
    ``linear`` (a bare ``Linear(d, 1)``, the SGD-trained analogue of the ridge)
    or ``mlp`` (the real ``RegressionHead``, which is
    what EventPredictor consumes). The linear arm is a CONTROL, not a
    candidate: it separates "the readout family is not expressive enough" from
    "the objective or the optimizer is the problem", but it cannot ship —
    EventPredictor sums ``g(emb)`` into ``mlp[0]``'s output, and a bare
    ``Linear(d, 1)`` has no such junction. Only the ``mlp`` family is
    deployable.

``loss``
    The scalar objectives from ``modeling.modes.supervised``,
    imported rather than reimplemented so the offline fit optimizes exactly
    what a training run would.

``batching``
    ``flat`` draws random rows, which is what the training dataloader does —
    ranking losses then compare stocks ACROSS cross-sections, a surrogate for
    the metric (see ``SupervisedModel._within_cell_loss``). ``cell`` draws
    whole cross-sections, so a ranking loss optimizes the exact quantity the
    reported IC measures. On a cached panel that costs nothing; in training it
    needs a cell-grouped sampler, which is why the shipped heads never had it.

THE RETURNED HEAD CONSUMES RAW EMBEDDINGS. Fitting standardizes the features
(as the ridge probe does), and the standardizer is then FOLDED into the first
Linear so the module that comes back is a drop-in for a checkpoint's own
``head.pt`` — same class, same state-dict keys, same input convention. A head
that needed a separate scaler bolted in front of it would not be.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from market_jepa.eval.heads import RegressionHead
from market_jepa.eval.metrics import grouped_rank_ic
from market_jepa.modeling.modes.supervised import (
    _regression_loss,
)

FAMILIES = ("linear", "mlp")
BATCHINGS = ("flat", "cell")
# Scalar objectives only. smooth_l1 is omitted: it is the second pointwise
# loss and collapses onto the same near-constant solution as mse, so it would
# double the grid to measure the same failure twice. cross_entropy went with
# the binned family on 2026-09-07 -- no supervised run ships a k-way head any
# more, so fitting one here would compare against nothing.
LOSSES = ("mse", "corr", "pairwise")


@dataclass(frozen=True)
class ProbeSpec:
    """One readout configuration: ``family:loss:batching``.

    ``batching`` is only meaningful for the ranking losses — a pointwise loss
    is a sum over rows and does not care how they were grouped — so
    ``cell`` is rejected for the others rather than silently ignored, which
    would put two identical fits in a sweep under different names.
    """

    family: str
    loss: str
    batching: str = "flat"
    hidden_mult: int = 2

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {self.family!r}")
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}, got {self.loss!r}")
        if self.batching not in BATCHINGS:
            raise ValueError(
                f"batching must be one of {BATCHINGS}, got {self.batching!r}")
        if self.batching == "cell" and self.loss not in ("corr", "pairwise"):
            raise ValueError(
                f"batching='cell' is only meaningful for a ranking loss; "
                f"{self.loss!r} is pointwise and would give the same fit as "
                "batching='flat' under a different name."
            )

    @classmethod
    def parse(cls, text: str) -> "ProbeSpec":
        """``'mlp:corr:cell'`` / ``'linear:mse'`` -> a spec."""
        parts = text.split(":")
        if len(parts) == 2:
            parts = [*parts, "flat"]
        if len(parts) != 3:
            raise ValueError(
                f"probe spec must be 'family:loss[:batching]', got {text!r}")
        return cls(parts[0], parts[1], parts[2])

    @property
    def name(self) -> str:
        return f"{self.family}:{self.loss}:{self.batching}"


def build_head(spec: ProbeSpec, d_embedding: int) -> nn.Module:
    """The module a spec trains, on RAW (unstandardized) embeddings.

    ``mlp`` is the shipped head shape verbatim — ``d -> hidden_mult*d -> out``
    with an RMSNorm — so a head fitted here loads into a checkpoint's
    ``head.pt`` slot and into ``EventPredictor`` without translation.
    ``linear`` passes an empty hidden list, which leaves ``mlp[0]`` a bare
    ``Linear(d, out)``; the standardizer fold below only ever touches
    ``mlp[0]``, so both families are handled by one code path.
    """
    hidden = [d_embedding * spec.hidden_mult] if spec.family == "mlp" else []
    return RegressionHead(d_embedding, hidden=hidden)


def _fold_standardizer(head: nn.Module, mu: np.ndarray, sd: np.ndarray) -> None:
    """Rewrite ``mlp[0]`` so the head consumes raw embeddings. Exact, in place.

    ``W((x - mu)/sd) + b == (W/sd) x + (b - W (mu/sd))``, so standardization is
    an affine map that the first Linear can absorb with no change to any
    prediction. Done at the END of fitting: the optimizer sees standardized
    features (which is what makes one learning rate work across encoders),
    and the caller gets a module with the same input convention as every other
    head in the project.
    """
    lin = head.mlp[0]
    if not isinstance(lin, nn.Linear):
        raise TypeError(f"expected mlp[0] to be nn.Linear, got {type(lin).__name__}")
    w = lin.weight.detach()
    scale = torch.as_tensor(sd, dtype=w.dtype, device=w.device)
    shift = torch.as_tensor(mu / sd, dtype=w.dtype, device=w.device)
    with torch.no_grad():
        lin.bias.sub_(w @ shift)
        lin.weight.div_(scale[None, :])


@torch.no_grad()
def head_scores(head: nn.Module, X: np.ndarray, device, batch: int = 65536) -> np.ndarray:
    """Scalar readout of a head over raw embeddings."""
    head = head.to(device).eval()
    out = []
    for i in range(0, len(X), batch):
        x = torch.from_numpy(np.ascontiguousarray(X[i:i + batch])).to(device).float()
        out.append(head(x).float().reshape(-1).cpu().numpy())
    return np.concatenate(out)


def _cell_bounds(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row order that groups cells contiguously, plus each cell's boundaries."""
    order = np.argsort(cells, kind="stable")
    srt = cells[order]
    edges = np.flatnonzero(np.r_[True, srt[1:] != srt[:-1], True])
    return order, edges


def _cell_loss(pred, y, edges_lo, edges_hi, loss_fn: str) -> torch.Tensor:
    """The flat loss applied INSIDE each cross-section, averaged over cells.

    Calls the same ``_regression_loss`` the training path calls, once per
    cell, rather than reimplementing a grouped variant: the two would drift,
    and then "cell batching" would mean something different here than in
    ``SupervisedModel.compute_group_loss``.

    Cells too small to carry an ordering are dropped, not scored as 0 — a
    degenerate cell is missing information, not a perfectly wrong prediction.
    """
    terms = []
    for lo, hi in zip(edges_lo, edges_hi):
        if hi - lo < 2:
            continue
        term = _regression_loss(pred[lo:hi], y[lo:hi], loss_fn, 1.0)
        if torch.isfinite(term):
            terms.append(term)
    if not terms:
        return pred.sum() * 0.0
    return torch.stack(terms).mean()


def _val_ic(head, Xv, yv, cellv, device) -> float:
    """Grouped rank IC on the held-out slice of the FIT month.

    Grouped, not pooled, because that is the reported metric's aggregation and
    a readout selected on a pooled IC can be selected for the wrong thing:
    pooling rewards getting the level right across cross-sections, which the
    metric never looks at.
    """
    s = head_scores(head, Xv, device)
    ic, _, _ = grouped_rank_ic(s, yv, cellv)
    return ic


def fit_head_probe(
    spec: ProbeSpec,
    X: np.ndarray,
    y: np.ndarray,
    cells: np.ndarray,
    val_mask: np.ndarray | None,
    *,
    device=None,
    lrs: tuple[float, ...] = (1e-3, 3e-3, 1e-2, 3e-2),
    weight_decay: float = 0.0,
    epochs: int = 20,
    batch_rows: int = 4096,
    batch_cells: int = 4,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Fit one readout on frozen embeddings; select on a held-out slice.

    Args:
        X: ``(N, d)`` embeddings, raw (unstandardized).
        y: ``(N,)`` cross-sectional z-score. NaN rows must already be dropped.
        cells: ``(N,)`` cell id — one ``(date, anchor)`` cross-section.
        val_mask: ``(N,)`` bool, True for the selection slice. Must be a
            TEMPORAL split (later dates), not a random one: a random split puts
            the same cross-section on both sides and the selection then reads a
            model's ability to interpolate inside a cell it has already seen.

            ``None`` runs the FIXED-RECIPE path: no hold-out, no selection,
            train on every row for exactly ``epochs`` at the single learning
            rate in ``lrs``. This is not a convenience — on a low-signal task
            it is the better estimator. Selecting one hyperparameter on one
            month's 20% slice costs about 0.003 of return IC, measured on the
            ridge, which is the same size as the entire head-vs-probe gap this
            module exists to close; and the hold-out itself costs another
            0.001 in fitting rows. A recipe fixed once on development months
            and applied unchanged pays neither. It is also how the project
            already deploys heads — the SSL finetune sweep bakes HEAD_BLR=1e-3
            into the sweep file rather than tuning per month.

    Returns:
        ``{"head", "lr", "val_ic", "epoch", "trace"}``. ``head`` is on CPU,
        consumes RAW embeddings, and is the state at the best validation epoch.

    The learning rate is swept and selected on ``val_mask`` alone. The eval
    month is never touched here — it is the test set for every arm, and a
    readout tuned on it would report its own selection noise as an improvement.

    The grid starts at 1e-3 because the arms are not equally easy to optimize:
    on a synthetic panel where a ridge reaches 0.749, a linear head needs
    lr=1e-2 to reach 0.750 and gets only 0.63 at 1e-3, while the MLP is already
    at 0.73. Under a narrow grid the linear control would look like evidence
    that a linear readout cannot match a ridge, when it is evidence that it was
    not trained long enough.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fixed = val_mask is None
    if fixed:
        if len(lrs) != 1:
            raise ValueError(
                "the fixed-recipe path (val_mask=None) has nothing to select "
                f"on, so it needs exactly one learning rate; got {lrs}")
        # Every row trains, and the epoch loop below still evaluates on this
        # mask -- pointing it at the training rows makes the logged val_ic an
        # in-sample number. It is reported as such and never selected on.
        val_mask = np.zeros(len(X), dtype=bool)
        tr = np.ones(len(X), dtype=bool)
        if tr.sum() < 100:
            raise ValueError(f"too few rows to fit: {int(tr.sum())}")
    else:
        tr = ~val_mask
        if tr.sum() < 100 or val_mask.sum() < 100:
            raise ValueError(
                f"train/val too small: {int(tr.sum())}/{int(val_mask.sum())} rows")

    mu = X[tr].mean(0)
    sd = X[tr].std(0)
    sd[sd < 1e-8] = 1.0
    Xs = ((X - mu) / sd).astype(np.float32)

    Xtr = torch.from_numpy(Xs[tr]).to(device)
    ytr = torch.from_numpy(y[tr].astype(np.float32)).to(device)
    Xva, yva, cva = Xs[val_mask], y[val_mask], cells[val_mask]

    cells_tr = cells[tr]
    order, edges = _cell_bounds(cells_tr)
    order_t = torch.from_numpy(order).to(device)
    n_cells = len(edges) - 1

    best = {"val_ic": -np.inf, "lr": None, "epoch": -1, "state": None, "trace": []}
    d = X.shape[1]

    for lr in lrs:
        torch.manual_seed(seed)
        head = build_head(spec, d).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        rng = np.random.RandomState(seed)
        trace = []

        for ep in range(epochs):
            head.train()
            if spec.batching == "cell":
                perm = rng.permutation(n_cells)
                batches = [perm[i:i + batch_cells]
                           for i in range(0, n_cells, batch_cells)]
            else:
                perm = torch.from_numpy(rng.permutation(len(Xtr))).to(device)
                batches = [perm[i:i + batch_rows]
                           for i in range(0, len(Xtr), batch_rows)]

            for b in batches:
                if spec.batching == "cell":
                    # Gather the batch's cells into one contiguous block and
                    # remember where each one starts, so the per-cell loss is a
                    # slice rather than a mask.
                    idx, lo, hi, at = [], [], [], 0
                    for c in b:
                        s0, s1 = int(edges[c]), int(edges[c + 1])
                        if s1 - s0 < 2:
                            continue
                        idx.append(order_t[s0:s1])
                        lo.append(at); at += s1 - s0; hi.append(at)
                    if not idx:
                        continue
                    rows = torch.cat(idx)
                else:
                    rows = b

                pred = head(Xtr[rows])
                if spec.batching == "cell":
                    loss = _cell_loss(pred, ytr[rows], lo, hi, spec.loss)
                else:
                    loss = _regression_loss(pred, ytr[rows], spec.loss, 1.0)

                if not torch.isfinite(loss):
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            if fixed:
                # Nothing to select on, so the answer is simply the state after
                # the prescribed number of epochs. Skipping the per-epoch IC
                # also skips a full scoring pass over every row.
                if ep == epochs - 1:
                    best.update(val_ic=float("nan"), lr=float(lr), epoch=ep,
                                state=copy.deepcopy(head.state_dict()))
                continue

            ic = _val_ic(head, Xva, yva, cva, device)
            trace.append({"lr": lr, "epoch": ep, "val_ic": float(ic)})
            if verbose:
                print(f"    {spec.name} lr={lr:.0e} ep{ep:02d} val_ic {ic:+.4f}",
                      flush=True)
            if np.isfinite(ic) and ic > best["val_ic"]:
                best.update(val_ic=float(ic), lr=float(lr), epoch=ep,
                            state=copy.deepcopy(head.state_dict()))
        best["trace"].extend(trace)

    if best["state"] is None:
        raise RuntimeError(f"{spec.name}: no epoch produced a finite validation IC")

    head = build_head(spec, d)
    head.load_state_dict({k: v.cpu() for k, v in best["state"].items()})
    _fold_standardizer(head, mu, sd)
    return {
        "head": head.eval(), "lr": best["lr"], "val_ic": best["val_ic"],
        "epoch": best["epoch"], "trace": best["trace"], "fixed": fixed,
        "n_fit_rows": int(tr.sum()),
        # The selection landing on the last epoch or the edge of the grid means
        # the budget, not the readout, may be what is being measured. Neither
        # is meaningful on the fixed path, where both were prescribed.
        "at_epoch_edge": (not fixed) and best["epoch"] == epochs - 1,
        "at_lr_edge": (not fixed) and best["lr"] in (min(lrs), max(lrs))
        and len(lrs) > 1,
    }
