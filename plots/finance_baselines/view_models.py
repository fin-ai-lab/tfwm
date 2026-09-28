"""Regressors fitted on the encoder's own input tensor, whole.

Every other baseline here is a hand-built forecasting model: someone decided
that trailing realized vol, or the current spread, is the thing to look at.
These are different, and they answer a different question — how much of the
task is reachable by an off-the-shelf learner reading the view itself, with no
representation learning and no econometrics at all?

They fit the same contract as ``models.Baseline`` (``fit(train_table)`` then
``predict(table, target)``) and are scored by the same ``grouped_rank_ic``, so
a view learner and a GARCH sit on one axis with the encoders.

WHAT THEY SEE. The literal 2048 x 9 view the supervised model is fed, flattened
to 18,432 features. No crop and no pool: those reductions were retired
2026-08-22 because they answered a narrower question than the one being asked
("what can a learner do with a summary of the view") and the full tensor is the
honest comparison.

WHY THAT IS AFFORDABLE AND WHAT IT COST. The AUC-era ``ridge_full`` ran on
~4k sampled rows per month. The synchronized panel is 506k rows at 36 anchors,
which is 37 GB of float32 view tensor — it cannot be cached, and it cannot be
handed to ``sklearn.Ridge`` as one array. So the ridge here is solved from its
normal equations instead, accumulated over ``panel_tables.iter_view_blocks``:
one pass builds the 18,432 x 18,432 Gram (2.7 GB in float64) and the tensor is
never materialized.

THE GRAM IS SHARED ACROSS TARGETS, WHICH IS THE ONLY REASON THIS IS ONE PASS.
Targets are censored at different rates — a 7200 s forward window fits on 36%
of rows, a 300 s one on 100% — so each target fits on a different subset and
naively wants its own Gram. But the censoring patterns are NESTED by horizon:
one month's 506k rows fall into exactly 6 distinct finite-patterns, each a
shell of the next. Accumulating one Gram per shell and summing the shells a
target is finite on gives that target's exact Gram, from a single pass over the
data rather than eighteen.

FLOAT32 STORAGE, FLOAT64 ARITHMETIC. The views are float32 at the source
(``_panel_for_ticker_day`` casts before they leave the panel), so streaming
them as float32 loses nothing. Every accumulation and the solve are float64,
deliberately: sklearn used to warn that this Gram is ill-conditioned at
rcond ~ 1e-7, which is float32's machine epsilon, and at that conditioning the
narrower type does not cost a decimal — it can cost the solve.

GBM CANNOT STREAM. ``HistGradientBoostingRegressor`` needs its rows in memory,
so it fits on a capped, fixed-seed row sample (``HGB_FIT_ROWS``) selected by
index before the pass so the choice does not depend on arrival order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from models import Baseline  # noqa: E402
from panel_tables import (  # noqa: E402
    N_FEATURES, VIEW_TAIL_TOKENS, iter_view_blocks, view_features,
)

# Boosting cost is linear in rows and the panel has ~506k of them per fit
# month, against the ~4k these models were tuned on in the AUC era. Capping the
# fit keeps a 32-month pass tractable; the cap is still 15x the old size, and
# it is applied with a FIXED seed so a rescore reproduces.
HGB_FIT_ROWS = 60_000
FIT_SEED = 0

_HGB_KW = dict(
    max_iter=150, learning_rate=0.06, max_leaf_nodes=15,
    min_samples_leaf=40, l2_regularization=1.0,
    early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
)

# Matches the ENCODER probe: xs_ic_eval.RIDGE_ALPHAS is 10.0 for all three
# target types, measured as the argmax at the ~400k-row production fit size.
# Borrowing the value keeps the probe on both sides of the comparison
# identically regularized rather than making the baseline's handicap a tuning
# artifact. Note the width is NOT matched — 18,432 features against the
# encoder's 384 — so the same alpha is a much lighter penalty per feature here,
# which favours the baseline if anything.
RIDGE_ALPHA = 10.0

# How many censoring shells a month may carry. One Gram per shell at 2.7 GB
# each is what makes this a budget rather than a preference: the COMPUTE is
# flat in the shell count (a row is accumulated into exactly one Gram however
# many there are), the MEMORY is not.
#
# SIX SHELLS WAS THE ORIGINAL DESIGN AND IT NO LONGER HOLDS. Censoring used to
# be nested by horizon alone -- a row is finite up to the last forward window
# that fits inside the session -- which is six shells and an exact partition.
# The forward-VWAP return (2026-08-22) broke the nesting: a VWAP window holding
# no traded volume is NaN whatever the horizon, so a stock can be censored at
# 300 s and finite at 7200 s. 2008-02 has 69 distinct patterns, and 69 Grams is
# 188 GB -- the whole machine.
#
# So the partition is RESTORED BY PROJECTION rather than assumed: see
# ``_patterns``. Eight fits in 22 GB and leaves the buffers and the GBM sample
# room beside it.
MAX_SHELLS = 8

# How much of the panel the projection may cost, as a fraction of the finite
# (row, target) pairs it could have fitted on. The ragged patterns are ~1.5% of
# rows and each keeps most of its targets, so the real number is far under
# this; a month that blows through it has censoring this scheme does not
# describe and should stop the pass rather than quietly fit on a subset.
MIN_PAIR_COVERAGE = 0.97

# Rows buffered per censoring shell before they are folded into that shell's
# Gram. 4096 x 18,432 is 302 MB of float32 buffer (1.2 GB when upcast at
# flush) and drops the number of 2.7 GB Gram round trips over a month from
# ~6,000 to ~130.
BUF_ROWS = 4096

# How much RAM the shell Grams may hold in total, which is what sets how many
# shells a width can afford: see RidgeFullView.begin_fit.
GRAM_BUDGET = 22 * 1024 ** 3

# Ridge variance floor, matching sklearn's StandardScaler behaviour on a
# constant column: scale by 1 rather than by 0.
_VAR_EPS = 1e-12


def _stream_kwargs(table: dict, tail_tokens: int | None = None) -> dict:
    """The arguments that reopen ``table``'s month as a view stream."""
    return dict(anchors_per_day=table["anchors_per_day"],
                tail_tokens=tail_tokens,
                **table.get("stream_kw", {}))


def _stream_tail(learners) -> int | None:
    """The widest tail the pass must stream to serve every learner."""
    tails = [lr.tail for lr in learners]
    return None if any(t is None for t in tails) else max(tails)


class FullViewLearner(Baseline):
    """A regressor fitted on the flattened view, or on its trailing tokens.

    ``tail`` is how many tokens back from the anchor the learner reads, None
    for the whole 2048 (see ``panel_tables.VIEW_TAIL_TOKENS`` for why a tail is
    a legitimate view and what it does NOT claim). Several tails share ONE
    decode pass: the stream runs at the widest tail any learner asked for and
    each learner slices its own out of the block, so measuring 8 against 64
    costs what measuring one of them costs.
    """

    def __init__(self, name: str, label: str, color: str,
                 tail: int | None = VIEW_TAIL_TOKENS):
        super().__init__(name, label, color)
        self.tail = tail
        self.P = view_features(tail)
        self.preds: dict = {}          # cache: table identity -> {target: pred}
        self._state_attrs = ("preds",)

    def _take(self, block: np.ndarray, stream_tail: int | None) -> np.ndarray:
        """This learner's columns out of a block streamed at ``stream_tail``.

        A block is the tail's steps flattened time-major, then the eleven
        information columns ONCE at the end -- so a narrower tail is a
        contiguous slice ending at the last step, plus that same eleven.
        """
        if self.tail is None:
            if stream_tail is not None:
                raise SystemExit(
                    f"{self.name} reads the whole view but the pass streamed a "
                    f"{stream_tail}-token tail. _stream_tail returns None when "
                    "ANY learner wants the full view; this learner was added "
                    "to a fitted group behind its back.")
            return block
        if self.tail == stream_tail:
            return block
        wide = 2048 if stream_tail is None else stream_tail
        lo = (wide - self.tail) * N_FEATURES
        hi = wide * N_FEATURES
        return np.concatenate([block[:, lo:hi], block[:, hi:]], axis=1)

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _patterns(Y: np.ndarray,
                  max_shells: int = MAX_SHELLS,
                  verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """``(row -> shell index, shell finite-masks (K, n_targets))``.

        A shell is a set of rows fitted together, and ``_solve`` reads a
        target's Gram as the sum of the shells that target is finite on. That
        is exact for ANY partition of the rows -- nesting was never the
        correctness condition, only the reason the partition stayed small.

        WHEN THE PATTERNS DO NOT FIT IN ``max_shells``, THE ROWS ARE PROJECTED
        ONTO THE COMMON ONES. The family is the ``max_shells`` most frequent
        patterns; every other row joins the largest family pattern CONTAINED IN
        its own finite mask, so a row is only ever fitted on targets it really
        has. Nothing wrong enters a Gram; what the projection costs is a row
        dropping out of a target it was finite on but its shell is not, and
        ``-1`` for a row containing no family pattern at all.

        The cost is reported, and it is small because the raggedness is:
        98.5% of 2008-02 is already the six horizon shells, and the rest is
        mostly one no-trade VWAP window inside an otherwise finite row.
        """
        ok = np.isfinite(Y)
        uniq, inv, cnt = np.unique(ok, axis=0, return_inverse=True,
                                   return_counts=True)
        inv = np.asarray(inv).ravel()      # numpy 2 returns (N, 1) here
        if len(uniq) <= max_shells:
            return inv, uniq

        # Widest first so a row lands on the family pattern that keeps the most
        # of its targets, and the most common of those breaks a tie -- the
        # frequent shells are the horizon ones, which is where a ragged row
        # belongs.
        fam = uniq[np.argsort(-cnt)[:max_shells]]
        fam_cnt = np.sort(cnt)[::-1][:max_shells]
        fam = fam[np.lexsort((-fam_cnt, -fam.sum(1)))]

        shell_of = np.full(len(ok), -1, dtype=np.int64)
        for k, mask in enumerate(fam):
            todo = np.flatnonzero(shell_of < 0)
            if not len(todo):
                break
            # mask ⊆ row: the row is finite everywhere this shell fits.
            shell_of[todo[~np.any(mask & ~ok[todo], axis=1)]] = k

        kept = fam[np.maximum(shell_of, 0)] & (shell_of[:, None] >= 0)
        coverage = kept.sum() / max(ok.sum(), 1)
        if verbose:
            print(f"    {len(uniq)} censoring patterns -> {len(fam)} shells, "
                  f"{(shell_of < 0).sum()} rows unplaced, "
                  f"{100 * coverage:.2f}% of finite (row, target) pairs kept",
                  flush=True)
        if coverage < MIN_PAIR_COVERAGE:
            raise SystemExit(
                f"projection onto {len(fam)} shells keeps only "
                f"{100 * coverage:.1f}% of the finite (row, target) pairs "
                f"(floor {100 * MIN_PAIR_COVERAGE:.0f}%); this month's "
                "censoring is not a handful of shells plus ragged edges")
        return shell_of, fam

    def _blocks(self, table: dict):
        """Stream ``table``'s month, checking it lines up with the table."""
        for off, block in stream_month(table):
            yield off, block

    # -- stream-driven phases -------------------------------------------
    # A month's decode costs ~15 minutes and is the same work whether the
    # table is 8 anchors or 36, so a learner that opened its own stream would
    # make every added learner cost another pass. These hooks let one pass
    # feed all of them: see ``fit_all`` / ``prime_all``.
    def begin_fit(self, table: dict) -> None:               # pragma: no cover
        raise NotImplementedError

    def accumulate_fit(self, off: int, block: np.ndarray) -> None:
        raise NotImplementedError                            # pragma: no cover

    def end_fit(self, table: dict) -> None:                 # pragma: no cover
        raise NotImplementedError

    def fit(self, table: dict) -> None:
        """Single-learner convenience: one stream, one learner."""
        fit_all([self], table)

    def _key(self, table: dict) -> tuple:
        # A pooled table's identity is its month LIST: two pools ending in the
        # same month are different fits, and the prediction cache is keyed by
        # the table it was primed on, so a scalar ``ym`` would let a six-month
        # pool serve a one-month pool's cached predictions.
        months = table.get("months")
        ym = tuple(m for m, _ in months) if months else table["ym"]
        return (ym, table["anchors_per_day"])

    def _targets(self) -> list[str]:                        # pragma: no cover
        raise NotImplementedError

    def accumulate_predict(self, off: int, block: np.ndarray,
                           cache: dict) -> None:
        raise NotImplementedError                            # pragma: no cover

    def predict(self, table: dict, target: str) -> np.ndarray | None:
        if target not in self._targets():
            return None
        if self._key(table) not in self.preds:
            prime_all([self], table)
        return self.preds[self._key(table)].get(target)


def stream_month(table: dict, tail_tokens: int | None = None):
    """``(row_offset, block)`` over ``table``'s month(s), checked against it.

    A POOLED TABLE IS STREAMED MONTH BY MONTH, IN ITS OWN ROW ORDER.
    ``panel_tables.pool_tables`` concatenates months in chronological order
    and records ``[(ym, n_rows), ...]``; replaying that list reproduces the
    pooled row order exactly, because each month's own stream already matches
    its own table. The per-month row count is checked as it goes, so a stream
    that drifts from its table surfaces on the month that drifted rather than
    as one wrong total at the end.
    """
    months = table.get("months") or [(table["ym"], len(table["Y"]))]
    n = 0
    for ym, want in months:
        m = 0
        for block in iter_view_blocks(ym, **_stream_kwargs(table, tail_tokens)):
            yield n + m, block
            m += len(block)
        if m != want:
            raise SystemExit(
                f"{ym}: view stream gave {m} rows against the table's {want}. "
                "The stream and build_month_table have drifted apart; they "
                "must group batches identically.")
        n += m
    if n != len(table["Y"]):
        raise SystemExit(
            f"{table['ym']}: view stream gave {n} rows against the table's "
            f"{len(table['Y'])}.")


def fit_all(learners, table: dict) -> None:
    """Fit every learner from ONE pass over the month."""
    tail = _stream_tail(learners)
    for lr in learners:
        lr._reset()
        lr.begin_fit(table)
    for off, block in stream_month(table, tail):
        for lr in learners:
            lr.accumulate_fit(off, lr._take(block, tail))
    for lr in learners:
        lr.end_fit(table)


def prime_all(learners, table: dict) -> None:
    """Fill every learner's prediction cache for ``table`` from ONE pass."""
    todo = [lr for lr in learners
            if lr._key(table) not in lr.preds and lr._targets()]
    if not todo:
        return
    n = len(table["Y"])
    tail = _stream_tail(todo)
    caches = [{t: np.empty(n) for t in lr._targets()} for lr in todo]
    for off, block in stream_month(table, tail):
        for lr, cache in zip(todo, caches):
            lr.accumulate_predict(off, lr._take(block, tail), cache)
    for lr, cache in zip(todo, caches):
        lr.preds[lr._key(table)] = cache


class RidgeFullView(FullViewLearner):
    """Ridge on all 18,432 features, solved from streamed normal equations."""

    def __init__(self, *a, alpha: float = RIDGE_ALPHA,
                 buf_rows: int = BUF_ROWS, retain_gram: bool = False, **kw):
        super().__init__(*a, **kw)
        self.alpha = alpha
        self.buf_rows = buf_rows
        # Keep the shell Grams after the solve so ``resolve`` can refit at a
        # different alpha. 16 GB held, against a ~30 minute re-stream per
        # month -- worth it while alpha is being chosen, never in a scoring
        # pass, which is why it is off by default.
        self.retain_gram = retain_gram
        self.coefs: dict = {}          # target -> (w, mu, sigma, y_mean)
        self._state_attrs = ("preds", "coefs")

    def begin_fit(self, table: dict) -> None:
        Y = np.asarray(table["Y"], dtype=np.float64)
        # THE SHELL BUDGET IS A MEMORY BUDGET, SO IT SCALES WITH THE WIDTH.
        # At the full 2048 a Gram is 2.7 GB and eight is all that fits; at a
        # 24-token tail it is 412 KB and the month's every censoring pattern
        # can have its own, which makes the partition EXACT and the projection
        # in ``_patterns`` a no-op. Same code, no approximation to explain.
        self._shell_of, self._shell_ok = self._patterns(
            Y, max_shells=max(1, min(512, int(GRAM_BUDGET // (self.P ** 2 * 8)))))
        K, T = self._shell_ok.shape
        P = self.P
        # Y is zero-filled where censored so one matmul covers every target;
        # the shell's own mask decides which columns of the result are real.
        self._Yf = np.where(np.isfinite(Y), Y, 0.0)
        self._n = np.zeros(K)
        self._s1 = np.zeros((K, P))
        self._sy = np.zeros((K, T))
        # A LIST of Fortran-ordered matrices, not a (K, P, P) array: syrk only
        # accumulates in place when C is column-major, and a slice of a
        # C-ordered 3-D array never is -- scipy would silently copy, leaving
        # every flush after the first writing into a temporary and the Gram
        # holding one block's worth of data.
        # ANNOUNCED BEFORE IT IS ALLOCATED, because this number decides how
        # many scoring processes fit on a machine and it is not knowable from
        # the command line: K is the month's censoring-pattern count, which
        # falls out of the data and the target roster (three targets at one
        # horizon can only make eight patterns, eighteen targets made 69).
        if self.tail is None:
            print(f"    {self.name}: {K} shells x {P}^2 float64 = "
                  f"{K * P * P * 8 / 1024 ** 3:.1f} GB of Gram", flush=True)
        self._G = [np.zeros((P, P), order="F") for _ in range(K)]   # K x 2.7 GB
        self._XtY = np.zeros((K, P, T))
        # One row buffer per shell. THE BUFFER IS THE POINT, not a micro-
        # optimization: every flush reads and writes a 2.7 GB Gram, so
        # updating it once per 512-row stream block would move ~48 TB over the
        # bus across a 506k-row month and swamp the 1.7e14 flops of actual
        # work. Flushing at BUF_ROWS turns ~6,000 of those round trips into
        # ~130.
        self._buf = [np.empty((self.buf_rows, P), np.float32) for _ in range(K)]
        self._bufy = [np.empty((self.buf_rows, T)) for _ in range(K)]
        self._bufn = [0] * K

    def _flush(self, k: int) -> None:
        m = self._bufn[k]
        if not m:
            return
        Xk = self._buf[k][:m].astype(np.float64)
        Yk = self._bufy[k][:m]
        self._n[k] += m
        self._s1[k] += Xk.sum(0)
        # syrk accumulates INTO C (beta=1), so no 2.7 GB temporary is formed
        # per flush. It writes ONE TRIANGLE, and the Gram is never
        # symmetrized: lower=1 writes the same triangle cho_factor(lower=True)
        # reads, so the upper half stays untouched garbage that nothing looks
        # at. Symmetrizing would mean np.triu_indices(18432), which is 2.7 GB
        # of int64 index arrays plus a 1.4 GB gather -- per shell, to produce
        # numbers the solve discards.
        from scipy.linalg.blas import dsyrk
        dsyrk(1.0, Xk, trans=1, beta=1.0, c=self._G[k], overwrite_c=1,
              lower=1)
        self._XtY[k] += Xk.T @ Yk
        self._sy[k] += Yk.sum(0)
        self._bufn[k] = 0

    def accumulate_fit(self, off: int, block: np.ndarray) -> None:
        sl = slice(off, off + len(block))
        shells = self._shell_of[sl]
        Yb = self._Yf[sl]
        for k in np.unique(shells):
            # -1 is a row the projection could not place (``_patterns``). It
            # must be SKIPPED, not indexed: self._G[-1] is the last shell's
            # Gram, so the negative index would fold unrelated rows into it.
            if k < 0:
                continue
            m = shells == k
            rows = np.flatnonzero(m)
            i = 0
            while i < len(rows):
                take = min(self.buf_rows - self._bufn[k], len(rows) - i)
                sel = rows[i:i + take]
                b = self._bufn[k]
                self._buf[k][b:b + take] = block[sel]
                self._bufy[k][b:b + take] = Yb[sel]
                self._bufn[k] = b + take
                i += take
                if self._bufn[k] == self.buf_rows:
                    self._flush(k)

    def end_fit(self, table: dict) -> None:
        for k in range(len(self._G)):
            self._flush(k)
        self._buf = self._bufy = None
        self._solve(table)
        # 16 GB of shell Grams have no use past the solve -- unless alpha is
        # still being chosen, in which case they are the whole month's stream
        # in 16 GB and re-solving costs six Choleskys instead of a re-read.
        if not self.retain_gram:
            self._G, self._XtY, self._Yf = [], None, None

    def resolve(self, table: dict, alpha: float) -> None:
        """Refit every target at a new ``alpha`` from the retained Grams.

        ALPHA IS NOT INHERITABLE HERE. RIDGE_ALPHA is 10, measured for the
        ENCODER probe at 384 features; this model has 18,432, so the same
        constant is a far lighter penalty per feature and there is no reason
        the optimum should coincide. Sweeping it costs six Choleskys per
        value against a 30-minute re-stream, so it is cheap to check and
        expensive to assume.
        """
        if not self._G:
            raise RuntimeError("fit with retain_gram=True to re-solve")
        self.alpha = alpha
        self.coefs.clear()
        self.preds.clear()
        self._solve(table)

    def _solve(self, table: dict) -> None:
        P = self.P
        # ONE FACTORIZATION PER DISTINCT SHELL UNION, not per target. The
        # censoring shells are indexed by horizon, so all three target types
        # at a given horizon are finite on exactly the same rows and therefore
        # share a Gram, a mean and a scale -- everything in the system except
        # the right-hand side. Factoring per target would repeat an 18,432^3
        # Cholesky eighteen times to solve six distinct systems.
        from scipy.linalg import cho_factor, cho_solve

        by_use: dict[tuple, list[int]] = {}
        for ti in range(self._shell_ok.shape[1]):
            key = tuple(np.flatnonzero(self._shell_ok[:, ti]).tolist())
            by_use.setdefault(key, []).append(ti)

        for key, tis in by_use.items():
            use = np.asarray(key, dtype=int)
            if not len(use):
                continue
            nt = self._n[use].sum()
            if nt < 200:
                continue
            mu = self._s1[use].sum(0) / nt
            # order="F" is load-bearing, not style. ndarray.copy() defaults to
            # C order, and every BLAS/LAPACK call below writes or reads ONE
            # TRIANGLE: on a C-ordered buffer they address the transpose, so
            # syr centres the triangle Cholesky does not read and the solve
            # runs on an uncentred Gram -- which does not raise, it just
            # returns coefficients that are quietly ~0.5% wrong. F order also
            # makes cho_factor's overwrite_a real, saving another 2.7 GB copy.
            A = np.array(self._G[use[0]], order="F", copy=True)
            for k in use[1:]:
                A += self._G[k]
            var = np.diag(A) / nt - mu ** 2
            sd = np.sqrt(np.maximum(var, 0.0))
            sd[sd < _VAR_EPS] = 1.0

            # Centre the Gram rather than the 37 GB of rows: (X - mu) has Gram
            # G - n mu mu^T. The SCALING is folded into the penalty instead of
            # applied to the matrix -- with v = w/sd, the standardized system
            # (D^-1 C D^-1 + a I) w = b is exactly (C + a D^2) v = D b -- which
            # saves forming and dividing by a second 2.7 GB outer product, and
            # leaves the prediction as a plain X @ v.
            # syr, not `A -= nt * np.outer(mu, mu)`: the outer product is
            # another 2.7 GB allocation, C-ordered against an F-ordered A, so
            # the subtraction walks it against its stride. syr does the
            # rank-1 update in place, on the same triangle syrk filled.
            from scipy.linalg.blas import dsyr
            dsyr(-nt, mu, a=A, lower=1, overwrite_a=1)
            A.flat[:: P + 1] += self.alpha * sd ** 2
            try:
                chol = cho_factor(A, lower=True, overwrite_a=True)
            except Exception:  # noqa: BLE001 — a month that won't solve is a skip
                del A
                continue
            for ti in tis:
                ybar = self._sy[use, ti].sum() / nt
                b = self._XtY[use, :, ti].sum(0) - nt * mu * ybar
                v = cho_solve(chol, b)
                self.coefs[table["target_names"][ti]] = (v, float(mu @ v), ybar)
            del A, chol


    def _targets(self) -> list[str]:
        return list(self.coefs)

    def accumulate_predict(self, off: int, block: np.ndarray,
                           cache: dict) -> None:
        Xb = block.astype(np.float64)
        for t, (v, mu_v, ybar) in self.coefs.items():
            cache[t][off:off + len(block)] = Xb @ v - mu_v + ybar


class HGBFullView(FullViewLearner):
    """Gradient boosting on all 18,432 features, fitted on a capped sample."""

    def __init__(self, *a, fit_rows: int = HGB_FIT_ROWS, **kw):
        super().__init__(*a, **kw)
        self.fit_rows = fit_rows
        self.models: dict = {}
        self._state_attrs = ("preds", "models")

    def begin_fit(self, table: dict) -> None:
        Y = np.asarray(table["Y"], dtype=np.float64)
        N = len(Y)
        # Chosen by INDEX before the pass, so which rows are kept does not
        # depend on shard arrival order and a rescore reproduces exactly.
        take = np.arange(N)
        if N > self.fit_rows:
            take = np.sort(np.random.default_rng(FIT_SEED).choice(
                N, self.fit_rows, replace=False))
        self._keep = np.zeros(N, dtype=bool)
        self._keep[take] = True
        self._Ys = Y[take]
        self._Xs = np.empty((len(take), self.P), dtype=np.float32)
        self._w = 0

    def accumulate_fit(self, off: int, block: np.ndarray) -> None:
        m = self._keep[off:off + len(block)]
        k = int(m.sum())
        if k:
            self._Xs[self._w:self._w + k] = block[m]
            self._w += k

    def end_fit(self, table: dict) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor
        for ti, target in enumerate(table["target_names"]):
            y = self._Ys[:, ti]
            ok = np.isfinite(y)
            if ok.sum() < 200:
                continue
            est = HistGradientBoostingRegressor(random_state=0, **_HGB_KW)
            # ``self._Xs[ok]`` is a FANCY INDEX, so it materializes a second
            # copy of the sample -- 13 GB at a six-month pool's 180k rows,
            # beside the 13 GB already held and the binned uint8 sklearn adds.
            # When the target is finite everywhere (the common case at one
            # horizon) there is nothing to select, and passing the array
            # itself removes the peak that decides how many of these processes
            # fit on the machine at once.
            Xs = self._Xs if ok.all() else self._Xs[ok]
            try:
                est.fit(Xs, y[ok])
            except Exception:  # noqa: BLE001 — a month that won't fit is a skip
                continue
            self.models[target] = est
        self._Xs = self._Ys = None

    def _targets(self) -> list[str]:
        return list(self.models)

    def accumulate_predict(self, off: int, block: np.ndarray,
                           cache: dict) -> None:
        for t, est in self.models.items():
            cache[t][off:off + len(block)] = est.predict(block)


def default_view_learners(fit_rows: int = HGB_FIT_ROWS
                          ) -> list[FullViewLearner]:
    """Registry. Add a model here and it appears in the scorer and the plot.

    TAILS, NOT THE FULL VIEW, AND WHY. The full-view pair is still defined and
    still correct -- ``full_view_learners()`` below returns it -- but a 2048
    step month costs a 2.7 GB Gram per shell, ~11 minutes of float64 syrk and
    a GBM that has to bin 18,432 features eighteen times. The readout these
    lines are compared against pools at ``last``, so the tail is where the
    comparison actually lives; three of them measure how much of the panel is
    reachable from the final patch, its neighbourhood, and half a view.
    Everything here rides ONE decode pass at the widest tail.

    Read ``panel_tables.VIEW_TAIL_TOKENS`` before quoting these numbers: a
    tail learner is a LOWER BOUND on a full-view one, because the encoder's
    last token attends over all 2048 steps and these do not.
    """
    return [
        RidgeFullView("ridge_tail8", "Ridge (8-token tail)", "#9edae5", tail=8),
        RidgeFullView("ridge_tail24", "Ridge (24-token tail)", "#17becf",
                      tail=24),
        RidgeFullView("ridge_tail64", "Ridge (64-token tail)", "#1f77b4",
                      tail=64),
        HGBFullView("hgb_tail24", "GBM (24-token tail)", "#8c6d31", tail=24,
                    fit_rows=fit_rows),
    ]


def full_view_learners(fit_rows: int = HGB_FIT_ROWS) -> list[FullViewLearner]:
    """The whole 2048 x 9 view. Kept for when the full pass is affordable.

    ``fit_rows`` IS THE EXPENSIVE ARGUMENT HERE, not a detail: the GBM holds
    its sample as float32 at the full width, so 60k rows are 4.4 GB and a
    six-month pool's 360k are 27 GB, beside the ridge's 2.7 GB per shell in
    the same process. Size the concurrency of a pass to that sum, not to the
    core count.
    """
    return [
        RidgeFullView("ridge_full", "Ridge (view)", "#17becf", tail=None),
        HGBFullView("hgb_full", "GBM (view)", "#8c6d31", tail=None,
                    fit_rows=fit_rows),
    ]
