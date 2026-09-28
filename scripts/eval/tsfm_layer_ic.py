"""Rank IC of every frozen-TSFM layer, from one forward pass per view.

The layer of a pretrained TSFM is a hyper-parameter, and the metric it used to
be chosen against (dAUC) is gone. This re-runs the choice under the reported
rank IC — on the OPTIMIZATION set (``scripts/experiments/holdout_months.py``),
never on the 32 months the paper reports, because picking a layer on the
reported panel is how a sweep quietly turns into a result.

WHY THIS CANNOT BE ``xs_ic_eval.py --run-ids``. Scoring L layers that way is L
independent passes over the panel, each re-paying the tokenizer and every
shared transformer block below its own depth; for TimesFM that is 21 passes
over ~600k views. ``PretrainedTSFM.compute_features_multi`` already captures
every depth from ONE forward, so the whole sweep costs a single pass.

WHAT IT REFUSES TO CHANGE. The panel comes from ``xs_ic_eval.panel_source`` —
the same anchors, the same per-cell resolution, the same crop kernels, the same
z-scores as every other method. That is ``iter_panel`` served from the on-disk
panel cache when ``MJ_PANEL_CACHE`` names one holding this exact key, and a
live ``iter_panel`` decode otherwise; the contract is identical either way, so
a cache hit and a cache miss are the same measurement. Until 2026-09-11 this
script called ``iter_panel`` DIRECTLY and was the last consumer that could not
see the cache — it re-decoded every month pair from the mosaic even where a
built panel was sitting beside it. The predictor is the same ridge on the same
standardized features. Only the encoder differs, which is the point.

THE MEMORY PROBLEM AND ITS ANSWER. Per-channel concat makes d = 9 x d_model =
11,520 for TimesFM. The reported protocol fits the probe on ~36 anchors/day,
~600k rows, so ONE layer's design matrix is 27 GB and the 21-layer sweep is
580 GB — before the eval month. So X is never materialized. Each layer keeps
only its sufficient statistics

    A = sum u u^T      a = sum u      B = sum u y^T      n, sum y

where ``u = (x - m0) / s0`` for a mean/scale estimated from the first
``WARMUP_ROWS`` rows. Working in u rather than x is not cosmetic: the exact
formula ``cov = A - n mu mu^T`` cancels catastrophically in float32 when a
feature's mean dwarfs its spread, and TSFM hidden states are full of such
features. In u-units the subtracted term is tiny and the cancellation is gone,
so a float32 GEMM on the GPU feeds a float64 accumulator with a relative error
of ~3e-7 — two orders below the smallest ridge alpha this sweep tries.

The eval month is then a SECOND pass that never stores embeddings either: the
weights are already solved, so each batch turns straight into predictions.

ONE ALPHA BY DEFAULT, AND IT IS THE PROTOCOL'S. ``--alphas`` defaults to
``xs_ic_eval.RIDGE_ALPHAS`` (10 for all three targets), which is what every
other arm of the paper is quoted at. It is true that 10 was calibrated on d=384
ViT embeddings and that a 6912-11520-wide per-channel concat is a different
ridge problem — that is why ``ALPHA_LADDER`` exists and why a diagnostic sweep
is one flag away. But letting the TSFMs alone pick their own regularizer, when
the supervised encoders, the SSL encoders and the random-init floor cannot,
tilts the comparison in their favour; so the reported runs use the shared
alpha and the only free parameter left is the layer.

The ladder is still cheap when it is wanted: one eigendecomposition per layer
serves every alpha, so ``--alphas $(...)`` costs essentially nothing beyond the
extra solves.

ONE JOB PER MONTH PAIR, NOT PER FAMILY. The panel decode — MDS read, dense
grid, crop, normalize — is a quarter of a Chronos-2 pass and nearly half a
Kronos one, and it is byte-identical for every family. Running the families
together lets one decode feed all four, which on the 32-month panel is ~29 of
the ~94 GPU-hours the sweep would otherwise cost. It also guarantees the four
families see the same panel rather than merely the same panel *definition*.

THE READOUT TOKEN IS A PROTOCOL CONSTANT, AND THIS ARM DID NOT HONOUR IT.
A prediction number in this paper is read at ``xs_ic_eval.PREDICT_POOL`` =
"last": the scorer rewrites every learned encoder's pool to it, the random-init
floor is built at it, and the sup_* families in this script inherit it because
``load_supervised`` loads them at the pool they trained with. The frozen TSFMs
did not, because ``PretrainedTSFM.time_pool`` defaults to "mean" and nothing
ever passed it — so every TSFM delta IC on disk is (TSFM at the MEAN) minus
(ViT at the LAST token), which breaks the rule the rest of the pipeline states
outright: a floor is read the way the models it floors are read. ``--time-pool``
exists to close that. It still DEFAULTS to "mean" so an old run reproduces
byte-for-byte; "last" is the protocol and is what a new reported run wants, and
it must be written to its own ``--out-dir``.

The latent half needs no such flag: ``panel_lib.LATENT_POOL`` is "mean", which
is what the TSFM latent consumers already build, so that side is already on
convention.

Usage:
    uv run scripts/eval/tsfm_layer_ic.py \
        --families chronos2 timesfm sundial kronos \
        --train-month 2009-01 --eval-month 2009-02 \
        --out-dir /data/lab/tsfm_layer_ic

    # the protocol readout; note the separate out-dir
    uv run scripts/eval/tsfm_layer_ic.py --time-pool last \
        --families chronos2 timesfm sundial kronos \
        --train-month 2009-01 --eval-month 2009-02 \
        --out-dir /data/lab/tsfm_layer_ic_lastpool
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/eval"))

from stable_finance import grouped_rank_ic_by_label  # noqa: E402
from stable_finance.dataset import (  # noqa: E402
    ANCHOR_TARGET_TYPES as TARGET_TYPES,
    DEFAULT_TARGET_HORIZONS as HORIZONS,
)
from market_jepa.eval.checkpoints import (  # noqa: E402
    SUP_PROJECTS,
    load_supervised,
)
from market_jepa.modeling.modes.pretrained_tsfm import (  # noqa: E402
    _FAMILIES, PretrainedTSFM,
)
from market_jepa.schemas import BLL01MachineConfig  # noqa: E402
from stable_finance.dataset import AnchorTargetStats as AnchorStats  # noqa: E402
from xs_ic_eval import (  # noqa: E402
    GLOBAL_SEQ_LEN, PREDICT_POOL, TRAIN_ANCHORS_PER_DAY, day_anchors,
    panel_source, ridge_alpha_for,
)

from stable_finance.dataset import MarketSchedule  # noqa: E402  (after ROOT)

# NOT the default any more — see the docstring. Kept because the question it
# answers ("is 10 leaving a lot on the table at d=11520?") is a real one and
# worth being able to re-ask with one flag. The top end is 3e7 because the
# optimum moves UP with both n and d and this problem is large in both: on a
# 16k-row pilot at d=6912, volatility_change wanted 1e5 (+0.200 against +0.122
# at alpha=10), and the production fit is ~30x more rows. Half-decades cost
# nothing: LayerStats.solve eigendecomposes once per layer and every alpha is a
# rescale of that.
ALPHA_LADDER = tuple(
    a * 10 ** k for k in range(8) for a in (1.0, 3.0)
)

# What --alphas defaults to: the one alpha every comparator in the paper is
# scored at. RIDGE_ALPHAS is per target type and is 10 for all three, so this
# is a single value; if that ever stops being true, this becomes the wrong
# shape loudly rather than the wrong number quietly.
_PROTOCOL_ALPHAS = sorted({ridge_alpha_for(t) for t in TARGET_TYPES})

# Rows used to estimate the centering/scaling origin (m0, s0). Only has to be
# accurate enough to keep the float32 GEMM away from cancellation; the exact
# full-sample mean/std is recovered afterwards from (A, a, n).
WARMUP_ROWS = 4096

# GPU bytes to spend on the per-layer chunk buffer before flushing it into the
# float64 accumulators. Bigger = fewer (d x d) device-to-host copies.
CHUNK_BYTES = 6 << 30


class LayerStats:
    """Streaming sufficient statistics for one layer's ridge probe."""

    def __init__(self, d: int, n_targets: int):
        self.d = d
        self.A = np.zeros((d, d), dtype=np.float64)
        self.a = np.zeros(d, dtype=np.float64)
        self.B = np.zeros((d, n_targets), dtype=np.float64)
        self.sy = np.zeros(n_targets, dtype=np.float64)
        self.n = 0
        self.m0: torch.Tensor | None = None
        self.s0: torch.Tensor | None = None

    def set_origin(self, x: torch.Tensor) -> None:
        """Fix (m0, s0) from a warmup block; zero-variance dims get s0 = 1."""
        self.m0 = x.mean(dim=0)
        s0 = x.std(dim=0)
        self.s0 = torch.where(s0 > 1e-12, s0, torch.ones_like(s0))

    def u(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.m0) / self.s0

    def add(self, u: torch.Tensor, y: torch.Tensor) -> None:
        """Fold one chunk of ``u`` rows (GPU, float32) and its targets in."""
        self.A += (u.T @ u).double().cpu().numpy()
        self.a += u.sum(dim=0).double().cpu().numpy()
        self.B += (u.T @ y).double().cpu().numpy()
        self.sy += y.double().sum(dim=0).cpu().numpy()
        self.n += u.shape[0]

    def solve(self, alphas) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Ridge weights at every alpha, in the ORIGINAL feature space.

        Returns ``(mu, sd, W, y_mean, n_clipped)``: standardize an eval row as
        ``(x - mu) / sd`` and predict with ``W[:, alpha, target]`` plus
        ``y_mean``. ``mu``/``sd`` are exactly what
        ``StandardScaler().fit(X_train)`` would have produced (ddof=0), and the
        weights are exactly ``Ridge(alpha, fit_intercept=True)`` on that
        standardization — the estimator ``xs_ic_eval.score`` uses.

        SOLVED BY EIGENDECOMPOSITION, NOT CHOLESKY. Cholesky is ~2x faster and
        it fails: Sundial layer 0 on 2009-01 took down a whole job with
        "1206-th leading minor is not positive definite" at alpha=1. That is
        not a bug to route around — a TSFM's per-channel concat really is
        rank-deficient in places (9 channels x d_model, with hidden units that
        are constant or collinear across the panel), and the standardized
        Gram's diagonal is n ~ 1.6e5, so alpha=1 regularizes it by 6e-6 and
        leaves float32's accumulation error to decide definiteness.

        ``eigh`` handles that head on: eigenvalues below zero are float noise
        around a genuine zero, clipping them is the projection onto the PSD
        cone, and the count is returned so a layer that is mostly rank-deficient
        cannot pass silently. It also makes the alpha ladder free — one
        factorization serves every alpha — which is why the ladder can afford
        half-decades.
        """
        n = float(self.n)
        abar = self.a / n
        ybar = self.sy / n
        # Centered scatter, still in u-units, IN PLACE: A is 1 GB per layer at
        # TimesFM's width and is dead the moment Gs exists. The subtracted term
        # is small by construction (abar ~ 0 because m0 came from the data),
        # which is the whole reason the accumulation is done in u.
        Gs = self.A
        self.A = None
        Gs -= n * np.outer(abar, abar)
        sd_u = np.sqrt(np.maximum(np.diag(Gs) / n, 0.0))
        keep = sd_u > 1e-10          # constant features carry no signal
        inv = np.zeros_like(sd_u)
        inv[keep] = 1.0 / sd_u[keep]

        Gs *= inv[:, None]
        Gs *= inv[None, :]
        Gs += Gs.T.copy()            # kill the float32 GEMM's asymmetry
        Gs *= 0.5
        Bc = (self.B - np.outer(self.a, ybar)) * inv[:, None]

        lam, V = np.linalg.eigh(Gs)
        del Gs                       # 1 GB at TimesFM's width; V is the successor
        n_clipped = int((lam < 0).sum())
        np.maximum(lam, 0.0, out=lam)
        VtB = V.T @ Bc
        W = np.zeros((self.d, len(alphas), Bc.shape[1]), dtype=np.float64)
        for j, alpha in enumerate(alphas):
            W[:, j, :] = V @ (VtB / (lam + alpha)[:, None])
        del V
        W[~keep] = 0.0

        m0 = self.m0.double().cpu().numpy()
        s0 = self.s0.double().cpu().numpy()
        mu = m0 + s0 * abar
        sd = np.where(keep, s0 * sd_u, 1.0)
        return mu, sd, W, ybar, n_clipped


def _targets_at(horizon: int) -> tuple[list[int], list[str]]:
    """Column indices into the panel's flattened (target_type, horizon) z."""
    names = [f"{t}_{h}" for t in TARGET_TYPES for h in HORIZONS]
    cols = [names.index(f"{t}_{horizon}") for t in TARGET_TYPES]
    return cols, [f"{t}_{horizon}" for t in TARGET_TYPES]


def _z_block(metas, cols) -> np.ndarray:
    return np.stack([m[0] for m in metas])[:, cols]


def build_model(family: str, device, tsfm_batch_size: int,
                channel_pool: str = "concat",
                time_pool: str = "mean") -> PretrainedTSFM:
    m = PretrainedTSFM(
        backbone=None, model=family, channels=list(range(9)),
        channel_pool=channel_pool, time_pool=time_pool,
        tsfm_batch_size=tsfm_batch_size,
    )
    m.to(device).eval()
    return m


# Families for which --channel-pool is not a knob at all. Kronos consumes ONE
# multivariate OHLCVA row per sample, so there is no per-channel axis to pool
# over (d = d_model either way); the supervised specialists are ViTs that read
# all nine channels through a patch projection. Running them under
# `--channel-pool mean` would write a file that differs from the concat run
# only in a label, which is worse than not running it.
NO_CHANNEL_POOL = {"kronos", *SUP_PROJECTS}

# Families for which --time-pool is not a knob either. The supervised
# specialists are checkpoints, and load_supervised reads each at the pool it
# TRAINED with -- which for that arm is already PREDICT_POOL ("last"), so they
# need no override and must not receive one.
NO_TIME_POOL = set(SUP_PROJECTS)


class FamilyFit:
    """One family's streaming fit, fed batch by batch from a SHARED panel.

    The panel decode — MDS read, dense grid, crop, normalize — costs about a
    quarter of a Chronos-2 pass and nearly half a Kronos one, and it is
    identical for every family. Running the families as separate jobs paid it
    four times; this class exists so one job can decode once and hand the same
    device tensor to all four. On the 32-month panel that is ~29 GPU-hours of
    the ~94 the sweep would otherwise cost.

    Everything else is unchanged: each family keeps its own (m0, s0) origin,
    its own per-layer sufficient statistics, and its own chunk buffer.
    """

    def __init__(self, family, model, layers, n_targets, chunk_rows):
        self.family = family
        self.model = model
        self.layers = layers
        # Trunk width when the model exposes one (a ViT reads hidden states,
        # not its d_embedding head output); d_embedding otherwise.
        self.d = getattr(model, "feature_dim", None) or model.d_embedding
        self.chunk_rows = chunk_rows
        self.per_layer = {L: LayerStats(self.d, n_targets) for L in layers}
        self._buf_x = {L: [] for L in layers}
        self._buf_y = []
        self._buffered = 0
        self._warm = {L: [] for L in layers}
        self._warm_n = 0

    @property
    def n(self) -> int:
        return self.per_layer[self.layers[0]].n

    def _flush(self):
        if not self._buffered:
            return
        y = torch.cat(self._buf_y)
        for L in self.layers:
            st = self.per_layer[L]
            st.add(st.u(torch.cat(self._buf_x[L])), y)
            self._buf_x[L].clear()
        self._buf_y.clear()
        self._buffered = 0

    def add_batch(self, x, lengths, y):
        """One batch of the shared panel: (B, C, T) views and (B, T) targets."""
        with torch.no_grad(), torch.autocast(device_type=x.device.type,
                                             dtype=torch.bfloat16):
            feats = self.model.compute_features_multi(x, lengths, self.layers)

        if self.per_layer[self.layers[0]].m0 is None:
            # Still collecting the warmup block: hold the rows, set the origin
            # once there are enough, then fold the whole block in at once so no
            # row is counted twice or lost.
            for L in self.layers:
                self._warm[L].append(feats[L].float())
            self._buf_y.append(y)
            self._warm_n += y.shape[0]
            if self._warm_n >= WARMUP_ROWS:
                self._promote_warmup()
            return

        for L in self.layers:
            self._buf_x[L].append(feats[L].float())
        self._buf_y.append(y)
        self._buffered += y.shape[0]
        if self._buffered >= self.chunk_rows:
            self._flush()

    def _promote_warmup(self):
        for L in self.layers:
            block = torch.cat(self._warm[L])
            self.per_layer[L].set_origin(block)
            self._buf_x[L] = [block]
            self._warm[L].clear()
        self._buffered = self._warm_n
        self._flush()

    def finish(self):
        if self.per_layer[self.layers[0]].m0 is None and self._warm_n:
            self._promote_warmup()      # tiny month: never reached WARMUP_ROWS
        else:
            self._flush()


def fit_pass(fits, month_dir, ym, stats, schedule, anchors, device,
             batch_size, cols, stride=1, log_every=50):
    """Pass 1 over the fit month, shared by every family in ``fits``.

    ``ym`` is the month the panel cache is keyed by; ``panel_source`` needs it
    and ``iter_panel`` does not, which is the whole reason it is threaded here.
    """
    dropped = np.zeros(len(cols), dtype=np.int64)
    seen = t_batches = 0
    t0 = time.time()
    for views, metas in panel_source(month_dir, ym, stats, schedule, anchors,
                                     batch_size, num_shards=stride,
                                     # THE FROZEN TSFMs TAKE NINE CHANNELS, and
                                     # say so rather than inheriting it. These
                                     # models tokenize a fixed-width series and
                                     # have nowhere to route an information
                                     # token, so the eleven are deliberately
                                     # withheld -- the asymmetry decided on
                                     # 2026-09-11 and the reason a TSFM delta is
                                     # a lower bound. It read off panel_source's
                                     # default until 2026-09-13, which made a
                                     # deliberate choice indistinguishable from
                                     # an unset argument.
                                     info_norm_stats=False, info_window=False):
        z = _z_block(metas, cols)
        ok = np.isfinite(z).all(axis=1)
        dropped += (~np.isfinite(z)).sum(axis=0)
        seen += len(z)
        t_batches += 1
        if not ok.any():
            continue
        x = torch.from_numpy(views[ok]).permute(0, 2, 1).to(device)
        lengths = torch.full((int(ok.sum()),), GLOBAL_SEQ_LEN,
                             dtype=torch.long, device=device)
        y = torch.from_numpy(z[ok].astype(np.float32)).to(device)
        for f in fits:
            f.add_batch(x, lengths, y)
        del x, y, lengths
        if t_batches % log_every == 0:
            print(f"    fit {seen} rows  {seen / (time.time() - t0):.0f} "
                  f"panel-views/s ({len(fits)} families)", flush=True)
    for f in fits:
        f.finish()
    return seen, dropped


def score_pass(fits, solved, month_dir, ym, stats, schedule, anchors,
               device, batch_size, cols, stride=1, log_every=50):
    """Pass 2 over the eval month, again decoded once for every family.

    ``solved[family][layer]`` is the ``(mu, sd, W, ybar, n_clipped)`` tuple the
    fit produced; predictions are formed on the fly, so no embedding is stored.
    """
    gpu, preds = {}, {}
    for f in fits:
        gpu[f.family] = {}
        preds[f.family] = {L: [] for L in solved[f.family]}
        for L, (mu, sd, W, _ybar, _nc) in solved[f.family].items():
            gpu[f.family][L] = (
                torch.as_tensor(mu, dtype=torch.float32, device=device),
                torch.as_tensor(sd, dtype=torch.float32, device=device),
                torch.as_tensor(W.reshape(W.shape[0], -1), dtype=torch.float32,
                                device=device),
            )
    ys, cells = [], []
    seen = t_batches = 0
    t0 = time.time()
    for views, metas in panel_source(month_dir, ym, stats, schedule, anchors,
                                     batch_size, num_shards=stride,
                                     # THE FROZEN TSFMs TAKE NINE CHANNELS, and
                                     # say so rather than inheriting it. These
                                     # models tokenize a fixed-width series and
                                     # have nowhere to route an information
                                     # token, so the eleven are deliberately
                                     # withheld -- the asymmetry decided on
                                     # 2026-09-11 and the reason a TSFM delta is
                                     # a lower bound. It read off panel_source's
                                     # default until 2026-09-13, which made a
                                     # deliberate choice indistinguishable from
                                     # an unset argument.
                                     info_norm_stats=False, info_window=False):
        z = _z_block(metas, cols)
        x = torch.from_numpy(views).permute(0, 2, 1).to(device)
        lengths = torch.full((len(views),), GLOBAL_SEQ_LEN,
                             dtype=torch.long, device=device)
        for f in fits:
            layers = sorted(solved[f.family])
            with torch.no_grad(), torch.autocast(device_type=device.type,
                                                 dtype=torch.bfloat16):
                feats = f.model.compute_features_multi(x, lengths, layers)
            for L in layers:
                mu, sd, W = gpu[f.family][L]
                preds[f.family][L].append(
                    (((feats[L].float() - mu) / sd) @ W).cpu().numpy())
            del feats
        del x, lengths
        ys.append(z)
        cells.extend(f"{m[1]}@{m[2]}" for m in metas)
        seen += len(z)
        t_batches += 1
        if t_batches % log_every == 0:
            print(f"    eval {seen} rows  {seen / (time.time() - t0):.0f} "
                  f"panel-views/s ({len(fits)} families)", flush=True)

    y = np.concatenate(ys)
    cell = np.array(cells)
    n_t = y.shape[1]
    # The alpha count is PER LAYER once the table is frozen, so the reshape
    # cannot use one global width.
    n_a = {f.family: {L: len(f.alphas_by[L]) for L in solved[f.family]}
           for f in fits}
    out = {fam: {L: np.concatenate(v).reshape(-1, n_a[fam][L], n_t)
                 for L, v in by.items()}
           for fam, by in preds.items()}
    return out, y, cell


def main() -> int:
    machine = BLL01MachineConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--families", nargs="+", required=True,
                   choices=sorted(_FAMILIES) + sorted(SUP_PROJECTS),
                   help="run several in ONE job: the panel decode is shared, "
                        "which is a quarter to a half of a pass")
    p.add_argument("--train-month", required=True, help="YYYY-MM, fits the probe")
    p.add_argument("--eval-month", required=True, help="YYYY-MM, reports the IC")
    p.add_argument("--layers", nargs="*", type=int, default=None,
                   help="default: every hidden state, 0 (patch embedding) .. L")
    p.add_argument("--horizon", type=int, default=900, choices=list(HORIZONS))
    p.add_argument("--alphas", nargs="*", type=float,
                   default=list(_PROTOCOL_ALPHAS),
                   help="ridge alphas to solve at. Defaults to the reported "
                        "protocol's (xs_ic_eval.RIDGE_ALPHAS), which is what "
                        "every comparator is scored at; pass the ladder "
                        "explicitly for a regularization diagnostic, but do "
                        "not report a TSFM at an alpha nothing else got to "
                        "pick.")
    p.add_argument("--alpha-table", default=None,
                   help="JSON of one alpha per (family, task, layer) chosen "
                        "on the OPTIMIZATION set, as {'table': {family: "
                        "{task: {layer: alpha}}}}. When given, --alphas is "
                        "ignored and each layer solves only its own alpha. "
                        "Retired for the reported runs in favour of the "
                        "single protocol alpha above.")
    p.add_argument("--anchors-per-day", type=int, default=8)
    p.add_argument("--train-anchors-per-day", type=int,
                   default=TRAIN_ANCHORS_PER_DAY)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--tsfm-batch-size", type=int, default=64,
                   help="internal row chunk; do NOT raise it for chronos2")
    p.add_argument("--channel-pool", default="concat", choices=("concat", "mean"),
                   help="how the nine per-channel embeddings become one "
                        "feature vector. 'concat' (default, and what every "
                        "reported TSFM number is on) makes d = 9 x d_model; "
                        "'mean' averages them to d = d_model. WRITE THE TWO "
                        "TO DIFFERENT --out-dirs: the filename is keyed by "
                        "(family, train month) alone, so they collide.")
    p.add_argument("--time-pool", default="mean",
                   choices=("mean", "last", "reg"),
                   help="which token of the context the per-channel embedding "
                        "is read at. THE PROTOCOL FOR A PREDICTION TASK IS "
                        f"{PREDICT_POOL!r} (xs_ic_eval.PREDICT_POOL), which is "
                        "what every learned encoder AND the random-init floor "
                        "are re-read at -- including the sup_* families in "
                        "this very script, which load at the pool they "
                        "trained with. 'mean' is the default only because "
                        "every TSFM result currently on disk was produced at "
                        "it, before this flag existed; it is off-protocol. "
                        "WRITE A NON-mean ARM TO A DIFFERENT --out-dir: the "
                        "filename is keyed by (family, train month) alone, so "
                        "the two arms collide. ('reg' is Chronos-2's [REG] "
                        "token, a diagnostic; the settled convention is "
                        "'last' uniformly.)")
    p.add_argument("--mosaic-dir", default=machine.mosaic_dir)
    p.add_argument("--xs-anchor-stats-dir",
                   default="/data/lab/market-jepa-mosaic/xs_anchor_stats_fwdvwap60")
    p.add_argument("--holiday-csv", default=machine.holiday_csv)
    p.add_argument("--shard-stride", type=int, default=1,
                   help="SMOKE TEST ONLY: keep 1 MDS shard in N on both "
                        "months. Anything but 1 is a different (smaller) "
                        "panel and is not the reported measurement.")
    p.add_argument("--out-dir", required=True,
                   help="writes <out-dir>/<family>_<train month>.json")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    # Refuse the combination that would write a differently-labelled duplicate
    # rather than a different measurement (see NO_CHANNEL_POOL).
    if args.channel_pool != "concat":
        inert = sorted(set(args.families) & NO_CHANNEL_POOL)
        if inert:
            print(f"ERROR: --channel-pool {args.channel_pool} does nothing for "
                  f"{inert}; drop them from --families.", file=sys.stderr)
            return 2

    # Same rule for the readout token. Fail here rather than after the panel
    # decode: PretrainedTSFM rejects reg on a non-chronos2 family itself, but
    # only once the family is constructed, an hour into the job.
    if args.time_pool != "mean":
        inert = sorted(set(args.families) & NO_TIME_POOL)
        if inert:
            print(f"ERROR: --time-pool {args.time_pool} does nothing for "
                  f"{inert}; they are checkpoints and load_supervised already "
                  f"reads them at the pool they trained with. Drop them from "
                  f"--families.", file=sys.stderr)
            return 2
    if args.time_pool == "reg":
        bad = sorted(set(args.families) - NO_TIME_POOL - {"chronos2"})
        if bad:
            print(f"ERROR: --time-pool reg is the Chronos-2 [REG] token; "
                  f"{bad} have no such token.", file=sys.stderr)
            return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    families = [f for f in args.families
                if args.overwrite
                or not (out_dir / f"{f}_{args.train_month}.json").is_file()]
    skipped = sorted(set(args.families) - set(families))
    if skipped:
        print(f"already done, skipping: {skipped}", flush=True)
    if not families:
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = MarketSchedule(args.holiday_csv)
    stats_dir = Path(args.xs_anchor_stats_dir)
    mosaic = Path(args.mosaic_dir)

    def month_dir(ym: str) -> Path:
        y, m = ym.split("-")
        return mosaic / y / m

    cols, names = _targets_at(args.horizon)
    alphas = list(args.alphas)
    frozen = (json.loads(Path(args.alpha_table).read_text())["table"]
              if args.alpha_table else None)
    if frozen:
        print(f"═══ alphas FROZEN from {args.alpha_table} — the ladder is not "
              f"swept on this panel", flush=True)

    # The chunk buffer is per family and lives on the GPU, so the budget is
    # split across however many run together — four families each holding a
    # 6 GiB buffer would not fit beside four loaded models.
    per_family_bytes = CHUNK_BYTES // len(families)
    fits = []
    for fam in families:
        if fam in SUP_PROJECTS:
            model = load_supervised(fam, args.train_month, device)
            n_states = model.n_layers + 1
        else:
            model = build_model(fam, device, args.tsfm_batch_size,
                                args.channel_pool, args.time_pool)
            n_states = _FAMILIES[fam]["n_layers"] + 1
        # DEPTH 0 IS DEGENERATE UNDER CLS POOLING. The readout is the CLS
        # token itself, which at depth 0 has not attended to anything -- it is
        # a learned constant, byte-identical for every input (measured: 0.000
        # per-row std). Its ridge features have zero variance, every
        # cross-section is skipped as constant, and the layer reports NaN. A
        # TSFM's depth 0 is the embedded input and carries signal, so this is
        # specific to the pooled-token architectures.
        first = 1 if getattr(model, "pool", None) == "cls" else 0
        layers = (sorted(args.layers) if args.layers
                  else list(range(first, n_states)))
        d = getattr(model, "feature_dim", None) or model.d_embedding
        chunk_rows = max(512, per_family_bytes // (d * 4 * len(layers)))
        fit = FamilyFit(fam, model, layers, len(cols), chunk_rows)
        # Which alphas each layer solves at, and which column each task reads.
        # Frozen: one alpha per (task, layer), so a layer usually needs 1-3
        # distinct values rather than the whole ladder.
        if frozen:
            fit.alphas_by, fit.idx_by = {}, {}
            for L in layers:
                want = [float(frozen[fam][t][str(L)]) for t in names]
                uniq = sorted(set(want))
                fit.alphas_by[L] = uniq
                fit.idx_by[L] = [uniq.index(w) for w in want]
        else:
            fit.alphas_by = {L: alphas for L in layers}
            fit.idx_by = None
        fits.append(fit)
        pool = ("" if fam in NO_CHANNEL_POOL
                else f", channel_pool {args.channel_pool}")
        pool += ("" if fam in NO_TIME_POOL
                 else f", time_pool {args.time_pool}"
                      + ("" if args.time_pool == PREDICT_POOL
                         else f" (OFF-PROTOCOL; xs_ic_eval.PREDICT_POOL "
                              f"is {PREDICT_POOL!r})"))
        print(f"═══ {fam}: {len(layers)} layers, d_embedding {d}{pool}, "
              f"accumulators {len(layers) * d * d * 8 / 2**30:.1f} GiB",
              flush=True)
    total_gib = sum(len(f.layers) * f.d * f.d * 8 for f in fits) / 2**30
    print(f"═══ {args.train_month} -> {args.eval_month}: {len(fits)} "
          f"families sharing one panel decode, {total_gib:.1f} GiB of "
          f"accumulators", flush=True)

    t0 = time.time()
    n_seen, dropped = fit_pass(
        fits, month_dir(args.train_month), args.train_month,
        AnchorStats(stats_dir / f"{args.train_month}.npz"), schedule,
        day_anchors(args.train_anchors_per_day), device, args.batch_size,
        cols, stride=args.shard_stride,
    )
    n_fit = fits[0].n
    print(f"    fit rows {n_fit} of {n_seen} seen "
          f"(shared finite mask; per-target NaN {dict(zip(names, dropped.tolist()))})"
          f"  [{time.time() - t0:.0f}s]", flush=True)

    # ── solve, per family ────────────────────────────────────────────────────
    # solve() frees each layer's (d x d) Gram as it consumes it, so peak memory
    # stays at one sweep's worth rather than doubling. Per-layer try/except
    # because the fit pass is the expensive half and one unsolvable layer must
    # not take the rest down with it — which is exactly what happened when
    # Cholesky met a rank-deficient Sundial layer.
    t1 = time.time()
    solved: dict[str, dict] = {}
    failed: dict[str, dict] = {}
    clipped: dict[str, dict] = {}
    for f in fits:
        if n_fit < 10 * f.d:
            print(f"    WARNING [{f.family}]: {n_fit} rows against d={f.d} — "
                  f"the ridge is under-determined at small alpha", flush=True)
        solved[f.family], failed[f.family] = {}, {}
        for L in f.layers:
            try:
                solved[f.family][L] = f.per_layer[L].solve(f.alphas_by[L])
            except Exception as e:                       # noqa: BLE001
                failed[f.family][L] = repr(e)
                print(f"    [{f.family}] LAYER {L} FAILED TO SOLVE: {e!r}",
                      flush=True)
            f.per_layer[L].A = None
        if not solved[f.family]:
            raise RuntimeError(f"{f.family}: every layer failed to solve")
        clipped[f.family] = {L: solved[f.family][L][4]
                             for L in solved[f.family] if solved[f.family][L][4]}
        if clipped[f.family]:
            print(f"    [{f.family}] negative eigenvalues clipped to 0 "
                  f"(float32 noise around a genuine zero): "
                  f"{clipped[f.family]}", flush=True)
    n_solves = sum(len(f.alphas_by[L]) for f in fits for L in solved[f.family])
    print(f"    solved {sum(len(v) for v in solved.values())} layers "
          f"({n_solves} ridges) [{time.time() - t1:.0f}s]", flush=True)

    t2 = time.time()
    preds, y, cell = score_pass(
        fits, solved, month_dir(args.eval_month), args.eval_month,
        AnchorStats(stats_dir / f"{args.eval_month}.npz"), schedule,
        day_anchors(args.anchors_per_day), device, args.batch_size,
        cols, stride=args.shard_stride,
    )
    print(f"    eval rows {len(y)}  [{time.time() - t2:.0f}s]", flush=True)

    for f in fits:
        layers = sorted(solved[f.family])
        results, alpha_used = {}, {}
        for L in layers:
            per_alpha: dict[str, dict] = {}
            P = preds[f.family][L]
            # Frozen: score only the (task, alpha) pairs the table names, so no
            # unused alpha ever reaches the output. Ladder: score everything.
            wanted = ([(f.idx_by[L][k], k) for k in range(len(names))]
                      if f.idx_by is not None
                      else [(j, k) for j in range(len(f.alphas_by[L]))
                            for k in range(len(names))])
            for j, k in wanted:
                name = names[k]
                alpha = f.alphas_by[L][j]
                ok = np.isfinite(y[:, k]) & np.isfinite(P[:, j, k])
                metric = grouped_rank_ic_by_label(
                    P[ok, j, k], y[ok, k], cell[ok], min_assets=20,
                )
                per_alpha.setdefault(f"{alpha:g}", {})[name] = {
                    "ic": float(metric.mean),
                    "se": float(metric.standard_error),
                    "n_cells": int(metric.observations),
                    "n_rows": int(ok.sum())}
                if f.idx_by is not None:
                    alpha_used.setdefault(str(L), {})[name] = f"{alpha:g}"
            results[str(L)] = per_alpha

        payload = {
            "family": f.family,
            # A TSFM carries its Hub id; a supervised trunk is identified by
            # the project it was trained in.
            "model_id": getattr(f.model, "model_id", None)
                        or SUP_PROJECTS.get(f.family, f.family),
            "d_embedding": f.d,
            "channel_pool": (args.channel_pool
                             if f.family not in NO_CHANNEL_POOL else None),
            # Like channel_pool: the FILENAME does not carry the readout, so
            # the payload has to, or two arms in one directory are
            # indistinguishable after the fact. Files written before this
            # field existed are all time_pool "mean".
            "time_pool": (args.time_pool
                          if f.family not in NO_TIME_POOL else None),
            "layers": layers,
            "alphas": sorted({a for L in layers for a in f.alphas_by[L]}),
            "alpha_source": (f"frozen:{args.alpha_table}" if frozen
                             else "ladder"),
            "alpha_used": alpha_used,
            "protocol_alpha": {n: ridge_alpha_for(n) for n in names},
            "horizon": args.horizon,
            "train_month": args.train_month,
            "eval_month": args.eval_month,
            "train_anchors_per_day": args.train_anchors_per_day,
            "anchors_per_day": args.anchors_per_day,
            "n_fit_rows": int(n_fit),
            "n_fit_rows_seen": int(n_seen),
            "n_eval_rows": int(len(y)),
            "shard_stride": args.shard_stride,
            "eigenvalues_clipped": {str(L): int(v)
                                    for L, v in clipped[f.family].items()},
            "layers_failed": {str(L): v
                              for L, v in failed[f.family].items()},
            "results": results,
        }
        out = out_dir / f"{f.family}_{args.train_month}.json"
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(out)

        for name in names:
            cand = [(L, a) for L in layers for a, per in results[str(L)].items()
                    if name in per for a in [a]]
            best = max(cand, key=lambda t: results[str(t[0])][t[1]][name]["ic"])
            cellv = results[str(best[0])][best[1]][name]
            # Only meaningful when there IS a ladder: with --alphas 10 every
            # row is trivially at both ends of a one-rung ladder.
            edge = ("" if frozen or len(alphas) < 2 else
                    (" EDGE-OF-LADDER"
                     if float(best[1]) in (alphas[0], alphas[-1]) else ""))
            print(f"  [{f.family}] {name:24s} best L{best[0]:<3d} alpha "
                  f"{best[1]:<8s} IC {cellv['ic']:+.4f} +- {cellv['se']:.4f}"
                  f"{edge}", flush=True)
        print(f"  Wrote {out}", flush=True)

    print(f"Done [{time.time() - t0:.0f}s total]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
