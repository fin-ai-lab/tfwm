"""Fit the ridge probe that a finetune's head STARTS FROM.

The SSL finetuning experiment does not train a head from scratch. A head that
begins at random spends most of a small label budget just learning to read an
embedding it is already handed -- which is the thing the frozen-probe curve in
plots/core/probe_fit_breadth.py measures, and there is no reason to pay
for it twice. So the head starts AT the probe: this script fits
``ColumnwiseRidge``'s arithmetic on the checkpoint's own frozen embeddings,
folds the standardizer in, and writes a weight/bias pair that
:meth:`SkipRegressionHead.init_from_ridge` loads. At step 0 the finetune's head
IS the reported probe, exactly; every label after that goes into the encoder.

THE FIT POOL IS THE FULL SIX-MONTH SPAN, ALWAYS -- it does not shrink with the
finetune's label budget. The swept axis of that experiment is how many labels
the ENCODER is adapted on, not the total label spend of the pipeline, and the
head init is held fixed across the ladder so that axis means one thing. Stated
here because it is the figure's main caveat: a point at x=32768 had a head that
had seen the whole span, so the curve is not a total-label-budget curve.

THE EMBEDDINGS MUST BE READ AT THE PREDICTIVE READOUT, AND THAT IS ENFORCED.
Prediction is scored at the LAST token for every arm (plots/core/readout.py).
Until 2026-09-17 the probe-breadth sweep did not comply: probe_fit_size.py
loaded each checkpoint with its own config, so the SSL/LeJEPA arms came back
MEAN-pooled. A head initialized from a mean-pooled probe is a probe for a
feature space the finetune never sees -- it reads the encoder at `last` -- so
the init is wrong in a way that still trains and still produces a curve. The
first wave of head inits was built that way and had to be thrown out.

So a shard is used only when it is STAMPED ``readout == "last"``, or when the
series is last by construction (readout.LAST_BY_CONSTRUCTION -- the supervised
arms, the random-init floor, and CoST, whose encode() takes the last patch and
never consults ``pool``). UNSTAMPED SHARDS ARE REFUSED for anything else: they
predate the stamp and carry the checkpoint's own training pool, which for an
SSL arm is the mean. There is deliberately no fallback -- a mean-pooled init
that looks plausible is worse than no init, and the failure is silent.

WHAT `z` ACTUALLY HOLDS. The shard key is named `z` and the handoff doc calls
it "z-scored targets". It is neither: it is the UNIFORM cross-sectional
quantile in [0, 1] (measured mean 0.5000, std 0.2883 = 1/sqrt(12)), which is
``DatasetConfig.xs_target='uniform'`` -- the same quantity the finetune trains
against. That agreement is what makes the init transferable at all; if the
dataset's target ever moves off uniform, this script's output becomes a head
initialized for a different question.

THE ARITHMETIC IS ColumnwiseRidge'S, NOT ITS CODE. That class upcasts to
float64, runs an isfinite scan and fits a StandardScaler -- several full-size
copies of a multi-million-row pool per call. Here only the MOMENTS are
accumulated (a 384x384 Gram per target), one pass over the data, so the pool is
never held whole. The scaler is fit on every usable row and each target's ridge
on its own labelled subset, which is exactly what ColumnwiseRidge does; the
targets' NaN patterns really do differ (457,605 vs 460,722 rows on 2008-02), so
one shared Gram would not reproduce it.

    uv run scripts/eval/fit_ridge_head_init.py --series pair_warp_6mo
    uv run scripts/eval/fit_ridge_head_init.py --months 2008-08 --verify
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# THE SINGLE SELECTOR, imported rather than restated: plots/core/readout.py is
# where the predictive-readout rule lives, and two copies of it would drift.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plots/core"))
from readout import (  # noqa: E402
    LAST_BY_CONSTRUCTION, PREDICT_READOUT,
)

STAGING = Path("/data/lab/probe_breadth/embeddings_staging")
MANIFEST_6MO = Path(
    "/data/lab/market-jepa-checkpoints/_scratch/latent_eval/manifest_6mo.json")
DEFAULT_OUT = Path("/data/lab/market-jepa-checkpoints/_scratch/ridge_head_init")

# The paper's three targets at the 15-minute horizon.
TASKS = ("return_900", "volatility_change_900", "spread_change_900")
# alpha=10 is what the reported probe table uses (the sweep fits 1, 10, 100).
DEFAULT_ALPHA = 10.0
# The fit panel: 36 anchors/day over the checkpoint's own training span.
FIT_ANCHORS = 36


def load_manifest(series: str) -> list[dict]:
    """The manifest rows for one series key, oldest eval month first."""
    rows = json.loads(MANIFEST_6MO.read_text())["ckpts"]
    out = [r for r in rows if r["series_key"] == series]
    if not out:
        keys = sorted({r["series_key"] for r in rows})
        raise SystemExit(f"no series {series!r}; have: {', '.join(keys)}")
    return sorted(out, key=lambda r: r["eval_month"])


def shards(run_dir: Path, month: str, anchors: int) -> list[Path]:
    """One month-group's shard files, oldest-name first, TEMPORARIES EXCLUDED.

    The embedding writers land a shard as ``.<name>.<pid>.tmp.npz`` and rename
    it into place, so a killed job leaves a half-written file behind -- two of
    them under run 93hkqfoe on 2026-09-17, 400 MB each and not valid zips.
    SHELL glob would never have matched those, because `*` in the shell does
    not match a leading dot; ``pathlib.glob`` DOES, so the fitter opened one
    and died with BadZipFile after 26 of 31 months.

    Hence the explicit name filter, in the ONE place every caller lists shards
    through. A `.tmp` file is also the wrong thing to count in the
    completeness tie-break of resolve_groups, which would otherwise prefer the
    copy with the most half-written files in it.

    Note this is deliberately narrow: it drops files that are BY NAME
    in-progress, and nothing else. A corrupt file under a real shard name is
    still a hard failure, because that one is not expected and should not be
    skipped quietly.
    """
    return sorted(p for p in run_dir.glob(f"{month}_a{anchors}_*.npz")
                  if not p.name.startswith("."))


def shard_readout(path: Path) -> str | None:
    """The readout stamp on one shard, or None when it predates the stamp."""
    with np.load(path, allow_pickle=False) as z:
        if "readout" not in z:
            return None
        return str(z["readout"])


def group_readout(run_dir: Path, month: str, anchors: int) -> str | None:
    """The readout of a month-group, from its first shard that holds rows.

    Empty markers carry no stamp of their own worth trusting, so they are
    skipped -- a collapsed month is 1 real shard and 31 markers, and reading
    the marker would report every group unstamped.
    """
    for p in shards(run_dir, month, anchors):
        with np.load(p, allow_pickle=False) as z:
            if "empty" in z:
                continue
            return str(z["readout"]) if "readout" in z else None
    return None


def admissible(readout: str | None, series: str) -> bool:
    """Whether embeddings at this readout may initialize a predictive head."""
    if readout == PREDICT_READOUT:
        return True
    # Unstamped rows carry the checkpoint's OWN training pool. That is already
    # the predictive readout for the arms below, and is the MEAN for every
    # SSL/LeJEPA arm -- which is the case this exists to refuse.
    return readout is None and series in LAST_BY_CONSTRUCTION


def resolve_groups(run_id: str, anchors: int = FIT_ANCHORS,
                   series: str = "") -> dict[str, Path]:
    """``{month: dir}`` for one run, deduped across the rescue tree.

    260 month-groups exist twice: a job cancelled on one node and rerun on
    another leaves two copies under different ``<node>/<tag>`` paths with
    IDENTICAL filenames. They are never merged -- a month is either sharded
    (32 real slices) or collapsed (shard 000 holds everything, 001-031 are
    ``empty`` markers), so file-by-file merging can put a real slice next to a
    whole-month cover and count the same rows twice. The copy with the most
    shard files wins, ties broken by path so the choice is reproducible.
    """
    cand: dict[str, list[Path]] = defaultdict(list)
    for d in STAGING.glob(f"*/*/{run_id}"):
        for p in d.glob(f"*_a{anchors}_*.npz"):
            if p.name.startswith("."):    # in-progress write; see shards()
                continue
            month = p.name.split("_a")[0]
            if d not in cand[month]:
                cand[month].append(d)
    out, rejected = {}, defaultdict(set)
    for month, dirs in cand.items():
        # READOUT FIRST, then completeness. Picking the biggest copy and
        # checking the readout afterwards would discard a complete last-pooled
        # group in favour of a complete mean-pooled one.
        ok = []
        for d in sorted(dirs):
            r = group_readout(d, month, anchors)
            if admissible(r, series):
                ok.append(d)
            else:
                rejected[month].add(r or "unstamped")
        if ok:
            out[month] = max(ok, key=lambda d: len(shards(d, month, anchors)))
    if rejected:
        for month in sorted(rejected):
            if month not in out:
                print(f"  {month}: no {PREDICT_READOUT}-pooled copy "
                      f"(found {', '.join(sorted(rejected[month]))})")
    return dict(sorted(out.items()))


def iter_shards(run_dir: Path, month: str, anchors: int, series: str = ""):
    """Yield ``(X, Y, target_names)`` per real shard, skipping empty markers.

    Re-checks the readout on EVERY shard rather than trusting the group probe:
    the rescue tree was assembled from several sweeps, and one directory
    holding shards from two of them would otherwise be averaged together.
    """
    for p in shards(run_dir, month, anchors):
        with np.load(p, allow_pickle=False) as z:
            if "empty" in z:          # collapsed marker or genuinely empty
                continue
            r = str(z["readout"]) if "readout" in z else None
            if not admissible(r, series):
                raise SystemExit(
                    f"{p}: readout {r or 'unstamped'} is not the predictive "
                    f"readout ({PREDICT_READOUT}) and {series} is not last by "
                    f"construction — refusing to mix readouts in one fit.")
            yield z["X"], z["z"], [str(x) for x in z["target_names"]]


class Moments:
    """Per-target ridge moments, accumulated one shard at a time."""

    def __init__(self, d: int, tasks):
        self.d = d
        self.tasks = list(tasks)
        # All usable rows -- this is what the StandardScaler is fit on.
        self.n_all = 0
        self.s1_all = np.zeros(d)
        self.s2_all = np.zeros((d, d))
        # Per target, over that target's labelled rows only.
        self.n = {t: 0 for t in self.tasks}
        self.s1 = {t: np.zeros(d) for t in self.tasks}
        self.gram = {t: np.zeros((d, d)) for t in self.tasks}
        self.xty = {t: np.zeros(d) for t in self.tasks}
        self.sy = {t: 0.0 for t in self.tasks}
        self.syy = {t: 0.0 for t in self.tasks}

    def add(self, X: np.ndarray, Y: np.ndarray, names: list[str]) -> None:
        usable = np.isfinite(X).all(axis=1)
        Xu = X[usable].astype(np.float64)
        Yu = Y[usable]
        self.n_all += len(Xu)
        self.s1_all += Xu.sum(0)
        self.s2_all += Xu.T @ Xu
        for t in self.tasks:
            y = Yu[:, names.index(t)].astype(np.float64)
            lab = np.isfinite(y)
            Xl, yl = Xu[lab], y[lab]
            self.n[t] += len(Xl)
            self.s1[t] += Xl.sum(0)
            self.gram[t] += Xl.T @ Xl
            self.xty[t] += Xl.T @ yl
            self.sy[t] += float(yl.sum())
            self.syy[t] += float(yl @ yl)


def fold_ridge(m: Moments, task: str, alpha: float) -> dict:
    """Solve the standardized ridge and fold the standardizer back out.

    ``ColumnwiseRidge`` is standardize-THEN-ridge, so a layer built from
    ``coef_`` alone is wrong by a per-feature scale and an offset. Dropping the
    scaler is the usual mistake and produces a head that looks plausible and
    predicts nothing. Returned ``weight``/``bias`` act on the RAW embedding:

        pred = x @ weight + bias

    ``pred_std`` is that layer's spread over the whole fit pool, which the head
    loader uses to put the init at a sane scale for the pairwise loss (the
    ridge's own spread is ~IC-sized, and RankNet reads differences).
    """
    d, n = m.d, m.n[task]
    if n < 2:
        raise ValueError(f"{task}: {n} labelled rows")
    mu = m.s1_all / m.n_all
    var = m.s2_all / m.n_all - np.outer(mu, mu)
    sd = np.sqrt(np.maximum(np.diag(var), 0.0))
    sd = np.where(sd > 1e-12, sd, 1.0)

    # Move this target's raw moments onto the standardized basis.
    s1 = m.s1[task]
    zg = (m.gram[task] - np.outer(mu, s1) - np.outer(s1, mu)
          + n * np.outer(mu, mu)) / np.outer(sd, sd)
    zty = (m.xty[task] - mu * m.sy[task]) / sd
    mz = (s1 / n - mu) / sd
    ybar = m.sy[task] / n

    # sklearn's Ridge centers X and y before solving, then recovers the
    # intercept -- the penalty must not touch it.
    gc = zg - n * np.outer(mz, mz)
    cc = zty - n * mz * ybar
    w = np.linalg.solve(gc + alpha * np.eye(d), cc)
    b0 = ybar - w @ mz

    weight = w / sd
    bias = b0 - float((w * mu / sd).sum())

    cov = m.s2_all / m.n_all - np.outer(mu, mu)
    pred_var = float(weight @ cov @ weight)
    # In-sample R on the fit pool: a sanity number, not a result.
    y_var = m.syy[task] / n - ybar ** 2
    return dict(
        weight=weight, bias=float(bias),
        pred_std=float(np.sqrt(max(pred_var, 0.0))),
        n_rows=int(n), alpha=float(alpha),
        r_insample=float(np.sqrt(max(pred_var, 0.0) / max(y_var, 1e-30))),
    )


def fit_one(row: dict, tasks, alpha: float, out_root: Path,
            verify: bool = False) -> dict | None:
    """Fit and write every task's head init for one checkpoint."""
    run_id, month = row["run_id"], row["eval_month"]
    series = row["series_key"]
    groups = resolve_groups(run_id, series=series)
    # The fit panel is the checkpoint's OWN training span, which is what the
    # a36 groups are; the eval month is a8 and must never appear here.
    span = [m for m in groups if row["train_start"] <= m <= row["train_end"]]
    missing = set(groups) - set(span)
    if missing:
        print(f"  {month}: ignoring out-of-span a36 groups {sorted(missing)}")
    if not span:
        # NOT a quiet skip: the usual cause since 2026-09-17 is that this
        # month has no last-pooled copy yet, which is a wave still landing
        # rather than a missing arm. Either way the month is absent from the
        # output and the caller's count says so.
        print(f"  {month}: no admissible a36 groups for {run_id} "
              f"({PREDICT_READOUT}-pooled) -- skipped")
        return None

    mom, names0 = None, None
    for mth in span:
        for X, Y, names in iter_shards(groups[mth], mth, FIT_ANCHORS, series):
            if mom is None:
                mom, names0 = Moments(X.shape[1], tasks), names
            elif names != names0:
                raise SystemExit(f"{month}: target_names differ across shards")
            mom.add(X, Y, names)
    if mom is None or mom.n_all == 0:
        print(f"  {month}: every shard empty -- skipped")
        return None

    out_dir = out_root / month
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(eval_month=month, run_id=run_id, ckpt_dir=row["ckpt_dir"],
                   series_key=series, fit_months=span, readout=PREDICT_READOUT,
                   source_dirs={m: str(groups[m]) for m in span},
                   n_rows_pool=int(mom.n_all), anchors=FIT_ANCHORS, tasks={})
    for t in tasks:
        fit = fold_ridge(mom, t, alpha)
        np.savez(out_dir / f"{t}.npz",
                 weight=fit["weight"].astype(np.float32),
                 bias=np.float32(fit["bias"]),
                 pred_std=np.float32(fit["pred_std"]),
                 n_rows=np.int64(fit["n_rows"]), alpha=np.float32(alpha),
                 # WHICH FEATURE SPACE THIS PROBE IS FOR. The finetune reads
                 # the encoder at `last`; a head initialized from a mean-pooled
                 # probe still trains and still produces a curve, so the only
                 # defence is for the artifact to say what it was fit at.
                 readout=PREDICT_READOUT)
        summary["tasks"][t] = {k: v for k, v in fit.items() if k != "weight"}
        print(f"  {month} {t:24s} n={fit['n_rows']:>9,d} "
              f"pred_std={fit['pred_std']:.4f} R={fit['r_insample']:.4f}")
    if verify:
        _verify(groups, span, mom, tasks, alpha)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _verify(groups, span, mom, tasks, alpha):
    """Check the folded layer against sklearn on one shard's worth of rows."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    X, Y, names = next(iter_shards(groups[span[0]], span[0], FIT_ANCHORS))
    sub = slice(0, 50000)
    Xs, Ys = X[sub], Y[sub]
    small = Moments(X.shape[1], tasks)
    small.add(Xs, Ys, names)
    for t in tasks:
        y = Ys[:, names.index(t)].astype(np.float64)
        usable = np.isfinite(Xs).all(axis=1)
        sc = StandardScaler().fit(Xs[usable].astype(np.float64))
        lab = usable & np.isfinite(y)
        rg = Ridge(alpha=alpha).fit(sc.transform(Xs[lab].astype(np.float64)), y[lab])
        ref = rg.predict(sc.transform(Xs[lab].astype(np.float64)))
        fit = fold_ridge(small, t, alpha)
        got = Xs[lab].astype(np.float64) @ fit["weight"] + fit["bias"]
        print(f"  VERIFY {t:24s} max|folded - sklearn| = "
              f"{np.abs(got - ref).max():.3e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", default="pair_warp_6mo",
                    help="manifest series key (default: the reported "
                         "LeJEPA Time Warping row)")
    ap.add_argument("--months", nargs="*", help="eval months (default: all)")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verify", action="store_true",
                    help="cross-check the fold against sklearn on 50k rows")
    args = ap.parse_args()

    rows = load_manifest(args.series)
    if args.months:
        rows = [r for r in rows if r["eval_month"] in set(args.months)]
    out_root = args.out / args.series
    print(f"{args.series}: {len(rows)} checkpoints -> {out_root}")

    done = []
    for row in rows:
        print(f"[{row['eval_month']}] {row['run_id']}")
        s = fit_one(row, args.tasks, args.alpha, out_root, args.verify)
        if s:
            done.append(s)
    (out_root / "index.json").write_text(json.dumps(done, indent=1))
    print(f"\n{len(done)}/{len(rows)} checkpoints fitted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
