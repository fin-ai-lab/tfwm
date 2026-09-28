"""RankMe over the LATENT suite's own full-day embeddings.

WHY NOT plots/metrics/rankme_6mo.json. That file is computed from the
probe-breadth staging, which is the PREDICTIVE panel: its SSL arms are
mean-pooled but its supervised arms and its random-init floor are read at the
LAST token. Putting that column beside the latent table -- every row of which
is mean-pooled by panel_lib.LATENT_POOL -- would compare spectra taken at two
different readouts, which is the same confound the latent suite's pool pin
exists to remove. This reads ff_fullday_cache instead: one readout (mean) for
every arm, including the floor.

SAME ROWS FOR EVERY ARM, BY CONSTRUCTION. Every emb_<series>.npz in a month
holds the same 14,144 (ticker, day) rows in the same order -- they are one
pass over one panel -- so nothing here has to align or subsample them. RankMe
drifts with N until N >> d; at 14k rows against d=384 that ratio is ~37 and is
identical across arms, so the comparison is of encoders, not of estimators.

RANKME IS BOUNDED BY THE EMBEDDING WIDTH, so an arm of a different width is
not on the same scale. tfc_6mo is 256-dim where the rest are 384.

The definition is imported from rankme.py rather than restated, so the two
tables cannot drift apart about what RankMe is.

    uv run python scripts/eval/rankme_fullday.py --json plots/core/rankme_latent.json

THE FROZEN TSFMs, EVERY LAYER, go to a file of their own (--tsfm), read by
plots/tsfm_layers/latent_sweep.py and by the latent table's TSFM block:

    uv run python scripts/eval/rankme_fullday.py --tsfm \\
        --json plots/tsfm_layers/rankme_tsfm.json

The series are the CHANNEL-AVERAGED layers, tsfm_<fam>_cmean_l<L> (the
prediction evals' readout, plots/latent_eval/run_cmean_eval.sh): d_model wide,
768 Chronos-2, 832 Kronos, 1280 TimesFM 3.0 -- still not the 384-wide scale.
(The concatenated layers, 9 x d_model wide, were measured once, 2026-09-24,
and retired with that readout.)

--tsfm RUNS ON THE GPU, and measures RankMe only. numpy's single-threaded fp32
SVD of an 11041 x 11520 panel takes over an hour, so 1,457 of them is days.
The GPU path takes the singular values as the square roots of the eigenvalues
of the smaller Gram matrix in FLOAT64, which reproduces the CPU numbers to four
decimals (floor seed 48.1895 both ways; Kronos L12 78.8227 vs 78.8226) at
~25 s for the widest panel. The floor seeds are not re-measured: their RankMe
is the table's, in rankme_latent.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ONE BLAS THREAD PER WORKER, set before numpy is imported or it is ignored.
# Each task is a 14144x384 SVD; numpy would otherwise open a thread pool per
# process and 64 workers x 64 threads is thrashing, not parallelism. Same
# reason and same list as rankme.py.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from rankme import cov_effective_rank, rankme  # noqa: E402

CACHE = Path("/data/lab/market-jepa-checkpoints/ff_fullday_cache")
FLOOR_SEEDS = re.compile(r"^randvit_s\d+$")
# The three reported frozen TSFMs, every hidden state (--tsfm).
TSFM_SERIES = re.compile(r"^tsfm_(chronos2|kronos|timesfm3)_cmean_l\d+$")


def wanted() -> set[str]:
    """The reported roster's keys, plus the floor seeds.

    AN ALLOWLIST, NOT A SKIP PATTERN. The cache is shared with retired sweeps --
    2009-07 alone still carries h2_* and pd_* from the head/patch grids, and one
    of those entries is a dangling file that raises on open. Excluding "tsfm_
    and grid_meta" let all of that in and made the walk depend on which old
    sweep happened to touch a month.
    """
    sys.path.insert(0, str(ROOT / "plots/task_corr"))
    from task_rank_corr import ROSTER  # noqa: E402

    keys = set()
    for m in ROSTER:
        keys.add(m["lat"])
        # ROSTER is keyed by the one-month generation; the cache holds the
        # six-month wave under the _6mo twin. Accept both.
        if m["lat"].endswith("_final"):
            keys.add(m["lat"][: -len("_final")] + "_6mo")
    return keys


def reported_months() -> list[str]:
    """The eval months the latent table actually reports."""
    p = ROOT / "plots/latent_eval/fixed_panel/fixed_panel_P3S2_6mo.json"
    got: set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "month_rates" and isinstance(v, dict):
                    got.update(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(json.loads(p.read_text()))
    return sorted(got)


def _measure(args):
    """One (series, month) panel. Module-level so it pickles for the pool."""
    path, series, ym = args
    try:
        with np.load(path, allow_pickle=False) as d:
            # float16 on disk; the SVD needs float32 or the small singular
            # values underflow and RankMe reads low.
            Z = d["X_eval"].astype(np.float32)
    except (FileNotFoundError, OSError) as exc:
        # A dangling archive symlink. Reported, and skipped -- silently
        # dropping an arm would leave its RankMe averaged over fewer months
        # than the rest of its row.
        return {"skip": f"{ym} {series}: {type(exc).__name__}: {exc}"}
    return {"series": series, "month": ym, "n": int(Z.shape[0]),
            "d": int(Z.shape[1]), "rankme": rankme(Z),
            "cov_effrank": cov_effective_rank(Z)}


def _rankme_gpu(Z: np.ndarray, eps: float = 1e-5) -> float:
    """rankme() with the spectrum from a float64 Gram on the GPU (see top)."""
    import torch

    X = torch.from_numpy(Z).cuda().double()
    G = X @ X.T if X.shape[0] <= X.shape[1] else X.T @ X
    s = torch.linalg.eigvalsh(G).clamp_min(0).sqrt()
    p = s / s.sum() + eps
    return float(torch.exp(-(p * torch.log(p)).sum()))


def _load(path):
    with np.load(path, allow_pickle=False) as d:
        return d["X_eval"].astype(np.float32)


def run_gpu(tasks, partial=None):
    """Sequential on one GPU, the next panel read while this one computes.

    RESUMABLE. A full sweep is ~3 h, and the first attempt died at 350/1457
    with nothing on disk because rows were only written at the end. Each row
    is appended to ``partial`` (jsonl) as it is measured, and a restart skips
    every (series, month) already there.
    """
    from concurrent.futures import ThreadPoolExecutor

    rows, skipped = [], []
    if partial is not None and partial.is_file():
        rows = [json.loads(l) for l in partial.read_text().splitlines() if l]
        done = {(r["series"], r["month"]) for r in rows}
        tasks = [t for t in tasks if (t[1], t[2]) not in done]
        print(f"  resuming: {len(rows)} done, {len(tasks)} to go", flush=True)
    sink = open(partial, "a") if partial is not None else None
    with ThreadPoolExecutor(max_workers=2) as io:
        futs = [io.submit(_load, t[0]) for t in tasks[:2]]
        for i, (path, series, ym) in enumerate(tasks):
            if i + 2 < len(tasks):
                futs.append(io.submit(_load, tasks[i + 2][0]))
            try:
                Z = futs[i].result()
            except (FileNotFoundError, OSError) as exc:
                skipped.append(f"{ym} {series}: {type(exc).__name__}: {exc}")
                continue
            futs[i] = None
            row = {"series": series, "month": ym, "n": int(Z.shape[0]),
                   "d": int(Z.shape[1]), "rankme": _rankme_gpu(Z)}
            rows.append(row)
            if sink is not None:
                sink.write(json.dumps(row) + "\n")
                sink.flush()
            if (i + 1) % 50 == 0 or i + 1 == len(tasks):
                print(f"  {i + 1}/{len(tasks)}", flush=True)
    if sink is not None:
        sink.close()
    return rows, skipped


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default=str(CACHE))
    p.add_argument("--json", default=None)
    p.add_argument("--months", nargs="*", default=None)
    p.add_argument("--jobs", type=int,
                   default=max(1, (os.cpu_count() or 8) - 4),
                   help="parallel workers; each panel is an independent SVD")
    p.add_argument("--tsfm", action="store_true",
                   help="measure the frozen-TSFM layers (plus the floor "
                        "seeds) instead of the reported roster")
    a = p.parse_args()

    cache = Path(a.cache)
    keep = wanted()
    # THE REPORTED PANEL'S MONTHS, not everything the cache holds. The cache is
    # shared with archived sweeps: 2009-07 is entirely broken symlinks into
    # /data/lab/models-archive, which no longer resolves, so walking the
    # directory listing raises on a month the table never shows.
    months = sorted(a.months or reported_months())
    tasks = []
    for ym in months:
        for f in sorted((cache / ym).glob("emb_*.npz")):
            series = f.name[len("emb_"):-len(".npz")]
            if a.tsfm:
                if not TSFM_SERIES.match(series):
                    continue
            elif FLOOR_SEEDS.match(series):
                pass
            elif series not in keep:
                continue
            tasks.append((str(f), series, ym))
    print(f"{len(tasks)} (series, month) panel(s) on {a.jobs} worker(s)",
          flush=True)

    if a.tsfm:
        rows, skipped = run_gpu(
            tasks, Path(a.json + ".partial") if a.json else None)
    else:
        rows, skipped = [], []
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            futs = [ex.submit(_measure, t) for t in tasks]
            for n, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                if "skip" in r:
                    skipped.append(r["skip"])
                    continue
                rows.append(r)
                if n % 100 == 0 or n == len(futs):
                    print(f"  {n}/{len(futs)}", flush=True)
    rows.sort(key=lambda r: (r["month"], r["series"]))
    for s in skipped:
        print(f"  !! SKIP {s}", file=sys.stderr)
    if skipped:
        print(f"  {len(skipped)} panel(s) skipped", file=sys.stderr)

    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
        print(f"wrote {a.json}  ({len(rows)} panels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
