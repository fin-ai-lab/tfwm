"""RankMe (Garrido et al. 2023) over the rescued probe-breadth embeddings.

    RankMe(Z) = exp( -sum_i p_i log p_i ),   p_i = sigma_i / sum_j sigma_j

the exponentiated entropy of the NORMALIZED SINGULAR VALUES of the embedding
matrix. It is a smooth, label-free stand-in for "how many directions does this
representation actually use", and it is the standard diagnostic for
dimensional collapse in SSL.

NOT THE SAME AS THE COVARIANCE EFFECTIVE RANK, and the difference matters when
comparing to any number quoted elsewhere. Participation ratio on the
eigenvalue spectrum uses p_i proportional to sigma_i^2 of CENTERED,
STANDARDIZED features; RankMe uses sigma_i of the RAW embedding. Squaring
concentrates the spectrum, so the covariance version usually reads LOWER.

THEY DO NOT EVEN AGREE ON THE ORDER. Measured across 21 arms on the COMPLETE
31-month panel, Spearman between the two is +0.44 (p=0.047) -- related, but
nowhere near interchangeable, and individual arms move up to 11 places.
RankMe reads the RAW embedding, so a large mean offset or a few wide-scale
coordinates dominate its spectrum; the covariance version removes exactly
those. LeJEPA+warp is 10th by RankMe and LAST by covariance rank (13.2 vs
3.11): its raw spectrum looks healthy while its directions of VARIATION have
collapsed to ~3. Supervised (return) is the mirror image, 18th and 7th. Quote
whichever you mean, never one as evidence for the other.

RANK IS ANTI-CORRELATED WITH FORECASTING SKILL. On the complete panel the
Spearman between an arm's RankMe and its probe IC is NEGATIVE and significant
on all three targets: return -0.66 (p=0.003), volatility -0.51 (p=0.032),
spread -0.52 (p=0.027), n=18 arms. The two widest spectra, TF-C (84.8) and
TimeMAE (63.1), are both struck through on all three targets; the four
narrowest that are not I-JEPA are the four supervised arms, which take every
top-two place in the IC table. Do not read a high RankMe as a healthy
representation here -- on this panel it predicts the opposite.

This file implements the RankMe definition exactly -- including the ``+ 1e-5``
the reference callback adds to p before the log -- so the numbers are
comparable to published ones.

RANKME IS BOUNDED BY THE EMBEDDING WIDTH, so arms of different width are not
directly comparable. tfc_6mo is 256-dim where every other arm is 384; it tops
this table at 84.8 and is near the BOTTOM of the IC table, which is the
cleanest available proof that rank is not quality.

SAME ROWS FOR EVERY ARM. RankMe is a function of the sample as well as the
encoder, and it drifts with N until N >> d. Every arm is therefore read on the
SAME eval panel (the reported month at 8 anchors/day), in shard order, capped
at the same --n-samples. Comparing an arm measured on 25k rows against one
measured on 400k would be measuring the estimator, not the encoder.

    uv run python scripts/eval/rankme.py
    uv run python scripts/eval/rankme.py --n-samples 25600 --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ONE BLAS THREAD PER WORKER. Each task is a 25600x384 SVD, and numpy would
# otherwise open a thread pool per process -- 64 workers x 64 threads is
# thrashing, not parallelism. Set before numpy is imported or it is ignored.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

STAGING = Path("lab/probe_breadth/embeddings_staging")
ROOT = Path("lab/probe_breadth")
INDEXES = ["manifest_all.index.json", "manifest_floor.index.json"]
SHARD = re.compile(r"^(\d{4}-\d{2})_a(\d+)_(\d{3})\.npz$")


def rankme(Z: np.ndarray, eps: float = 1e-5) -> float:
    """Exponentiated entropy of the normalized singular values."""
    s = np.linalg.svd(Z, compute_uv=False)
    p = s / s.sum() + eps
    return float(np.exp(-(p * np.log(p)).sum()))


def cov_effective_rank(Z: np.ndarray) -> float:
    """Participation ratio on the CENTERED, STANDARDIZED covariance spectrum.

    Reported beside RankMe because it is the quantity a PCA-truncation study
    reads, and the two are routinely confused.
    """
    A = Z - Z.mean(0)
    sd = A.std(0)
    A = A / np.where(sd > 1e-12, sd, 1.0)
    ev = np.linalg.svd(A, compute_uv=False) ** 2
    p = ev / ev.sum()
    return float(np.exp(-(p * np.log(p + 1e-300)).sum()))


def load_index() -> dict[tuple[str, str], str]:
    idx = {}
    for f in INDEXES:
        p = ROOT / f
        if p.is_file():
            for r in json.loads(p.read_text()):
                idx[(r["run_id"], r["eval_month"])] = r["series_key"]
    return idx


def collect(run_dir: Path, month: str, anchors: int, n: int):
    """Up to ``n`` rows of this run's panel, in shard order."""
    out, got = [], 0
    for p in sorted(run_dir.glob(f"{month}_a{anchors}_*.npz")):
        try:
            with np.load(p, allow_pickle=False) as z:
                if "empty" in z:
                    continue
                X = z["X"]
        except Exception:                                # truncated / unreadable
            continue
        out.append(X[: n - got])
        got += len(out[-1])
        if got >= n:
            break
    if not out:
        return None
    return np.concatenate(out).astype(np.float64)


def _measure(run_dir: str, month: str, anchors: int, n: int, series: str):
    """One panel, in a worker process."""
    Z = collect(Path(run_dir), month, anchors, n)
    if Z is None or len(Z) < 4 * Z.shape[1]:
        return None
    return {"series": series, "month": month, "run": Path(run_dir).name,
            "n": int(len(Z)), "d": int(Z.shape[1]),
            "rankme": rankme(Z), "cov_effrank": cov_effective_rank(Z)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--staging", default=str(STAGING))
    p.add_argument("--anchors", type=int, default=8,
                   help="8 = the reported eval panel; 36 = the fit panel")
    p.add_argument("--n-samples", type=int, default=25600,
                   help="RankMe's usual sample budget; must match across arms")
    p.add_argument("--json", default=None)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 8) - 4),
                   help="parallel workers; the work is per-panel and "
                        "embarrassingly parallel")
    a = p.parse_args()

    idx = load_index()
    root = Path(a.staging)
    if not root.is_dir():
        raise SystemExit(f"no staging tree at {root}")

    # WORK OUT THE TASK LIST FIRST, THEN FAN OUT. The cost here is npz
    # decompression, not the SVD -- ~9 shards a panel to reach the sample
    # budget -- so it parallelizes cleanly over panels. A run id can appear
    # under several nodes/tags (a part cancelled on one node and rerun on
    # another); they are the same weights, so one reading is enough.
    tasks: dict[tuple[str, str], tuple] = {}
    for node in sorted(x for x in root.iterdir() if x.is_dir()):
        for tag in sorted(x for x in node.iterdir() if x.is_dir()):
            for run in sorted(x for x in tag.iterdir() if x.is_dir()):
                months = {m.group(1) for f in run.iterdir()
                          if (m := SHARD.match(f.name))
                          and int(m.group(2)) == a.anchors}
                for month in sorted(months):
                    series = idx.get((run.name, month))
                    if series is None or (series, month) in tasks:
                        continue
                    tasks[(series, month)] = (str(run), month, a.anchors,
                                              a.n_samples, series)
    print(f"{len(tasks)} (arm, month) panel(s) to measure on {a.jobs} worker(s)",
          file=sys.stderr, flush=True)

    seen: dict[tuple[str, str], dict] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(_measure, *t): k for k, t in tasks.items()}
        for f in as_completed(futs):
            done += 1
            r = f.result()
            if r is not None:
                seen[futs[f]] = r
            if done % 25 == 0 or done == len(futs):
                print(f"  {done}/{len(futs)} panels", file=sys.stderr, flush=True)

    if not seen:
        print("no staged embeddings matched", file=sys.stderr)
        return 1
    rows = list(seen.values())
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))

    by = defaultdict(list)
    for r in rows:
        by[r["series"]].append(r)
    # WIDTHS, PLURAL. Reporting rows[0]["d"] as "the" dimension hid that
    # tfc_6mo is 256-dim among 384-dim arms -- and RankMe is bounded by width.
    dims = sorted({r["d"] for r in rows})
    ns = sorted({r["n"] for r in rows})
    print(f"\nRankMe over {len(rows)} (arm, month) panel(s) at {a.anchors} "
          f"anchors/day\n  embedding width(s): {dims}"
          f"\n  sample size(s): {ns}  (RankMe drifts with n; compare only "
          f"like with like)\n")
    print(f"{'arm':20s} {'d':>4s} {'months':>6s} {'RankMe':>9s} {'min':>8s} "
          f"{'max':>8s} {'cov.effrank':>12s}")
    for key, rs in sorted(by.items(),
                          key=lambda kv: -np.mean([r["rankme"] for r in kv[1]])):
        v = [r["rankme"] for r in rs]
        c = [r["cov_effrank"] for r in rs]
        dd = sorted({r["d"] for r in rs})
        print(f"{key:20s} {(dd[0] if len(dd)==1 else '*'):>4} {len(rs):>6d} "
              f"{np.mean(v):>9.2f} {min(v):>8.2f} "
              f"{max(v):>8.2f} {np.mean(c):>12.2f}")
    print(f"\n(RankMe=d is a perfectly flat spectrum, RankMe=1 total collapse "
          f"to one direction. It is NOT a quality score: see the module "
          f"docstring.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
