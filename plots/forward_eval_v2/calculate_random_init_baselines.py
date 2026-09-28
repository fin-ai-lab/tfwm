"""Random-init rank-IC baselines over the forward-decay grid — the ΔIC subtrahend.

Produces, for every (probe-train month, horizon) cell of ``forward_eval_v2``,
the rank IC an UNTRAINED encoder of the same architecture reaches with the same
ridge probe on the same synchronized panel. That is the number every v2 figure
subtracts (see ``market_jepa/eval/metrics.py:delta_ic``): without it a method
can look successful while doing nothing a random projection does not already
do, which on this data is not hypothetical.

Supersedes the ΔAUC baseline this file used to compute. Nothing of that
measurement survives: the metric is rank IC on the cross-sectional z-score, the
targets are the two-forward-window generation (all three changed 2026-08-22),
and the eval crop scale is the unified [0.5, 1.0]. Old ``baseline.json`` values
are not comparable and were deleted rather than migrated.

THE MEASUREMENT IS NOT REIMPLEMENTED HERE. The panel comes from
``xs_ic_eval.embed_month_many`` and the numbers from the same StandardScaler +
``Ridge(ridge_alpha_for(name))`` + ``grouped_rank_ic`` that ``xs_ic_eval.score``
applies, so a baseline cell and a checkpoint cell differ in the encoder and in
nothing else. The one thing this file does differently is WHEN it fits:

    score() fits and evaluates in one call. Here the probe is fit on month 0
    and reused at every horizon, which is the whole point of the v2 scheme --
    the fit depends only on month-0 features, never on H. Calling score() per
    (month, H) pair would refit the identical probe 16 times.

``--verify-against-score`` checks that split against ``score()`` on a real pair
and fails loudly on any drift, so the reordering cannot silently become a
second scorer.

Cost, and why it is shaped this way
-----------------------------------
The decode dominates: building a ticker-day's dense 1 Hz grid is single-threaded
CPU work that a ViT-384 forward is a fraction of. So

  * months are embedded ONCE per (month, anchor grid) and shared across every
    seed in one ``embed_month_many`` call — one decode, N forwards, not N
    decodes;
  * embeddings are cached as npz under the same path ``xs_ic_eval`` uses, so a
    month embedded here is reused by any other IC work on this box (and vice
    versa) instead of being re-decoded;
  * the probe fit is hoisted out of the horizon loop, so a (month, seed) pays 3
    ridge fits rather than 3 x 16.

For the default grid that is 213 month-embeds and 90 ridge fits per seed,
against 639 embeds and 1383 fits for the naive nesting.

Usage:
    uv run plots/forward_eval_v2/calculate_random_init_baselines.py
    uv run plots/forward_eval_v2/calculate_random_init_baselines.py --embed-only
    uv run plots/forward_eval_v2/calculate_random_init_baselines.py --score-only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "eval"))

from stable_finance.dataset import MarketSchedule  # noqa: E402
from market_jepa.eval.checkpoints import (  # noqa: E402
    architecture_signature,
    load_model,
)
from market_jepa.eval.metrics import grouped_rank_ic  # noqa: E402
from market_jepa.schemas import BLL01MachineConfig  # noqa: E402
from stable_finance.dataset import AnchorTargetStats as AnchorStats  # noqa: E402

import xs_ic_eval  # noqa: E402
from xs_ic_eval import (  # noqa: E402
    TRAIN_ANCHORS_PER_DAY,
    day_anchors,
    embed_month_many,
    ridge_alpha_for,
)

sys.path.insert(0, str(_THIS_DIR))
import baseline_common  # noqa: E402

# ─── grid ───────────────────────────────────────────────────────────────────

# Must match forward_eval_v2.HORIZONS: the baseline is looked up by
# (probe_train_month, H) and a cell with no baseline is a cell with no figure.
HORIZONS: tuple[int, ...] = (1, 2, 3, 6, 9, 12, 15, 18, 21, 24,
                             30, 36, 42, 48, 54, 60)

# Last calendar month with mosaic data. 2025/01..03 exist but are sample-less
# stubs, so the real end is 2024-12; cells past it are dropped from the grid
# rather than attempted.
DATA_END_MONTH: str = "2024-12"

# Train months whose ckpts forward_eval_v2 does not plot.
SKIP_TRAIN_YEAR_PREFIXES: tuple[str, ...] = ("2023-",)

EVAL_HORIZON: int = 900
EVAL_TARGET_TYPES: tuple[str, ...] = (
    "return", "volatility_change", "spread_change",
)
TARGET_COLS: tuple[str, ...] = tuple(
    f"{t}_{EVAL_HORIZON}" for t in EVAL_TARGET_TYPES
)

ANCHORS_PER_DAY: int = 8

# The campaign's 32 eval months, single-sourced from the manifest the metrics
# figures read. A train_end is the month BEFORE its eval month, which is what
# v2 fits the probe on.
FAIR_MONTHS_JSON = _REPO_ROOT / "plots" / "metrics" / "noclamp_manifest.json"

# The established ΔIC floor encoders — the same three every other IC table in
# the paper is quoted against ([[randinit_ic_baselines]]). Loaded rather than
# re-drawn from a seed so this figure's subtrahend IS the others', not merely
# the same recipe.
RANDINIT_ROOT = Path("/data/lab/randinit_bb")
DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44)

BASELINE_JSON = _THIS_DIR / "baseline.json"
CACHE_JSON = _THIS_DIR / "calculate_random_init_baselines_cache.json"


# ─── calendar ───────────────────────────────────────────────────────────────


def add_months(ym: str, n: int) -> str:
    """'YYYY-MM' + n calendar months."""
    y, m = map(int, ym.split("-"))
    total = y * 12 + (m - 1) + n
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def build_grid() -> tuple[list[str], list[tuple[str, int, str]]]:
    """``(train_months, [(probe_train_month, H, eval_month)])``.

    ``eval_month = probe_train_month + H`` months, matching
    ``forward_eval_v2.eval_key_for`` (first-of-next-month(train_end) + H-1).
    """
    fair = json.loads(FAIR_MONTHS_JSON.read_text())["fair_months"]
    train_months = sorted(
        ym for ym in {add_months(f, -1) for f in fair}
        if not any(ym.startswith(p) for p in SKIP_TRAIN_YEAR_PREFIXES)
    )
    pairs = [
        (ptm, h, add_months(ptm, h))
        for ptm in train_months for h in HORIZONS
        if add_months(ptm, h) <= DATA_END_MONTH
    ]
    return train_months, pairs


# ─── embedding cache ────────────────────────────────────────────────────────


def cache_path(root: Path, ym: str, sig: str, seed: int, n_anchors: int) -> Path:
    """Exactly ``xs_ic_eval``'s randinit cache path.

    Shared on purpose: a month embedded by either script is reused by the
    other. Diverging here would silently double the GPU cost of every IC
    analysis on this box.
    """
    return root / "xs_ic_cache" / ym / f"randinit_{sig}_s{seed}_a{n_anchors}.npz"


def load_encoders(seeds: tuple[int, ...], device) -> tuple[dict, str]:
    """The random-init backbones, plus the architecture signature they share."""
    models, sig = {}, None
    for seed in seeds:
        d = RANDINIT_ROOT / f"randinit_s{seed}"
        if not (d / "backbone.pt").is_file():
            raise SystemExit(f"missing random-init backbone: {d}/backbone.pt")
        cfg = json.loads((d / "train_meta.json").read_text())["config"]
        this_sig = architecture_signature(cfg)
        if sig is not None and this_sig != sig:
            raise SystemExit(
                f"seed {seed} has architecture {this_sig}, expected {sig} — "
                "the floor is keyed by architecture, so these cannot share a "
                "baseline"
            )
        sig = this_sig
        models[seed] = load_model(str(d), cfg, device).eval()
    return models, sig


def embed_phase(
    needed: list[tuple[str, int]], models: dict, sig: str, root: Path,
    mosaic_dir: Path, stats_dir: Path, schedule, device, batch_size: int,
) -> None:
    """Fill the npz cache. One decode per (month, anchor grid), all seeds."""
    todo = []
    for ym, n_anchors in needed:
        missing = [s for s in models
                   if not cache_path(root, ym, sig, s, n_anchors).is_file()]
        if missing:
            todo.append((ym, n_anchors, missing))
    done = len(needed) - len(todo)
    print(f"embed: {len(todo)} month-grids to do, {done} already cached")

    for i, (ym, n_anchors, missing) in enumerate(todo, 1):
        y, m = ym.split("-")
        t0 = time.time()
        panels = embed_month_many(
            {s: models[s] for s in missing}, mosaic_dir / y / m, ym,
            AnchorStats(stats_dir / f"{ym}.npz"), schedule,
            day_anchors(n_anchors), device, batch_size,
        )
        for seed, panel in panels.items():
            p = cache_path(root, ym, sig, seed, n_anchors)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, **panel)
        rows = len(next(iter(panels.values()))["X"])
        print(f"  [{i}/{len(todo)}] {ym} a{n_anchors} "
              f"seeds={sorted(missing)} rows={rows} "
              f"{time.time() - t0:.0f}s", flush=True)


# ─── probe: score()'s two halves, split across the horizon loop ─────────────


def fit_probes(train_cache: dict, target_cols: tuple[str, ...]) -> tuple:
    """``xs_ic_eval.score``'s fit half, hoisted out of the horizon loop.

    Returns ``(scaler, {col: Ridge})``. Depends only on month-0 features, which
    is why one fit serves every H.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    names = train_cache["target_names"].tolist()
    sc = StandardScaler().fit(train_cache["X"])
    Xtr = sc.transform(train_cache["X"])
    models = {}
    for col in target_cols:
        if col not in names:
            continue
        ytr = train_cache["z"][:, names.index(col)]
        ok = np.isfinite(ytr)
        if ok.sum() < 100:
            continue
        models[col] = Ridge(alpha=ridge_alpha_for(col)).fit(Xtr[ok], ytr[ok])
    return sc, models


def eval_probes(probe: tuple, eval_cache: dict) -> dict:
    """``score``'s eval half: predict, then rank IC within each (date, anchor)."""
    sc, models = probe
    names = eval_cache["target_names"].tolist()
    cells = np.char.add(np.char.add(eval_cache["date"], "@"),
                        eval_cache["anchor"].astype(str))
    Xev = sc.transform(eval_cache["X"])
    out = {}
    for col, m in models.items():
        if col not in names:
            continue
        yev = eval_cache["z"][:, names.index(col)]
        ok = np.isfinite(yev)
        if ok.sum() < 100:
            continue
        mean, se, n_cells = grouped_rank_ic(m.predict(Xev[ok]), yev[ok], cells[ok])
        out[col] = {"ic": float(mean), "se": float(se),
                    "n_cells": int(n_cells), "n_rows": int(ok.sum())}
    return out


def verify_against_score(train_cache: dict, eval_cache: dict) -> None:
    """Fail loudly if the fit/eval split has drifted from ``score()``."""
    ref = xs_ic_eval.score(train_cache, eval_cache)
    got = eval_probes(fit_probes(train_cache, TARGET_COLS), eval_cache)
    for col in TARGET_COLS:
        if col not in ref and col not in got:
            continue
        a, b = ref[col]["ic"], got[col]["ic"]
        if not np.isclose(a, b, rtol=0, atol=1e-12):
            raise SystemExit(
                f"DRIFT from xs_ic_eval.score on {col}: {b:+.10f} vs {a:+.10f}"
            )
    print(f"verify: fit/eval split matches score() on {len(TARGET_COLS)} columns")


# ─── scoring pass ───────────────────────────────────────────────────────────


def score_phase(
    pairs: list[tuple[str, int, str]], seeds, sig: str,
    root: Path, cache: dict, force: bool,
) -> None:
    """Fit each month-0 probe once, then stream the eval months past all of them."""
    by_eval: dict[str, list[tuple[str, int]]] = {}
    for ptm, h, evm in pairs:
        by_eval.setdefault(evm, []).append((ptm, h))

    for seed in seeds:
        want = [(ptm, h, evm) for ptm, h, evm in pairs
                if force or baseline_common.cache_key(ptm, h, "transformer", seed)
                not in cache]
        if not want:
            print(f"seed {seed}: nothing to score")
            continue

        need_ptm = sorted({ptm for ptm, _, _ in want})
        probes = {}
        for ptm in need_ptm:
            p = cache_path(root, ptm, sig, seed, TRAIN_ANCHORS_PER_DAY)
            if not p.is_file():
                print(f"  seed {seed} {ptm}: train embedding missing, skipped")
                continue
            tr = dict(np.load(p, allow_pickle=False))
            probes[ptm] = fit_probes(tr, TARGET_COLS)
            del tr
        print(f"seed {seed}: fitted {len(probes)} month-0 probes "
              f"({len(TARGET_COLS)} targets each)")

        need_eval = sorted({evm for _, _, evm in want})
        for i, evm in enumerate(need_eval, 1):
            p = cache_path(root, evm, sig, seed, ANCHORS_PER_DAY)
            if not p.is_file():
                print(f"  seed {seed} {evm}: eval embedding missing, skipped")
                continue
            ev = dict(np.load(p, allow_pickle=False))
            n = 0
            for ptm, h in by_eval[evm]:
                if (ptm, h, evm) not in want or ptm not in probes:
                    continue
                res = eval_probes(probes[ptm], ev)
                cache[baseline_common.cache_key(ptm, h, "transformer", seed)] = {
                    col: (res[col]["ic"] if col in res else None)
                    for col in TARGET_COLS
                }
                n += 1
            del ev
            if i % 10 == 0 or i == len(need_eval):
                print(f"  [{i}/{len(need_eval)}] {evm} (+{n} cells)", flush=True)
            baseline_common.save_cache(CACHE_JSON, cache)


def aggregate_to_json(pairs, seeds, cache: dict) -> None:
    """Per-seed ICs → ``baseline.json`` aggregates, keyed the way plots read them."""
    results: dict = {}
    for col in TARGET_COLS:
        by_ptm: dict = {}
        for ptm, h, _ in pairs:
            per_seed = [
                (cache.get(baseline_common.cache_key(ptm, h, "transformer", s))
                 or {}).get(col)
                for s in seeds
            ]
            if all(v is None for v in per_seed):
                continue
            by_ptm.setdefault(ptm, {})[h] = per_seed
        if by_ptm:
            results[col] = {"transformer": by_ptm}
    baseline_common.write_baseline_json(BASELINE_JSON, results)
    print(f"wrote {BASELINE_JSON}")


# ─── main ───────────────────────────────────────────────────────────────────


def main() -> None:
    machine = BLL01MachineConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--checkpoint-root", default="/data/lab/market-jepa-checkpoints")
    p.add_argument("--mosaic-dir", default=machine.mosaic_dir)
    # Must match xs_ic_eval's default: the anchor table defines the target
    # generation, and standardizing this panel against a different one is the
    # bug that has already cost a whole sweep (see commit 278282e).
    p.add_argument("--xs-anchor-stats-dir",
                   default="/data/lab/market-jepa-mosaic/xs_anchor_stats_fwdvwap60")
    p.add_argument("--holiday-csv", default=machine.holiday_csv)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--embed-only", action="store_true")
    p.add_argument("--score-only", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="rescore cells already in the per-seed cache")
    p.add_argument("--verify-against-score", action="store_true",
                   help="assert the fit/eval split reproduces xs_ic_eval.score")
    p.add_argument("--limit-months", type=int, default=0,
                   help="embed at most N month-grids (smoke test)")
    args = p.parse_args()

    stats_dir = Path(args.xs_anchor_stats_dir)
    if not stats_dir.is_dir():
        raise SystemExit(f"anchor stats dir not found: {stats_dir}")

    train_months, pairs = build_grid()
    eval_months = sorted({evm for _, _, evm in pairs})
    needed = ([(ym, TRAIN_ANCHORS_PER_DAY) for ym in train_months]
              + [(ym, ANCHORS_PER_DAY) for ym in eval_months])
    missing_stats = [ym for ym, _ in needed
                     if not (stats_dir / f"{ym}.npz").is_file()]
    if missing_stats:
        raise SystemExit(
            f"{len(missing_stats)} months have no anchor table in {stats_dir}: "
            f"{missing_stats[:8]} — build them before scoring, or the panel "
            "would be standardized against a different target generation"
        )

    print(f"grid: {len(train_months)} train months x {len(HORIZONS)} horizons "
          f"= {len(pairs)} cells, {len(eval_months)} distinct eval months")
    print(f"embeds: {len(needed)} month-grids x {len(args.seeds)} seeds "
          f"(one decode each, shared across seeds)")
    print(f"targets: {', '.join(TARGET_COLS)}")
    print(f"anchor stats: {stats_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.checkpoint_root)
    cache = baseline_common.load_cache(CACHE_JSON)

    if not args.score_only:
        models, sig = load_encoders(tuple(args.seeds), device)
        schedule = MarketSchedule(args.holiday_csv)
        todo = needed[:args.limit_months] if args.limit_months else needed
        embed_phase(todo, models, sig, root, Path(args.mosaic_dir), stats_dir,
                    schedule, device, args.batch_size)
        del models
        torch.cuda.empty_cache()
    else:
        _, sig = load_encoders(tuple(args.seeds), torch.device("cpu"))

    if args.embed_only:
        return

    if args.verify_against_score:
        ptm, _, evm = pairs[0]
        tp = cache_path(root, ptm, sig, args.seeds[0], TRAIN_ANCHORS_PER_DAY)
        ep = cache_path(root, evm, sig, args.seeds[0], ANCHORS_PER_DAY)
        if tp.is_file() and ep.is_file():
            verify_against_score(dict(np.load(tp, allow_pickle=False)),
                                 dict(np.load(ep, allow_pickle=False)))
        else:
            print("verify: skipped, first pair not embedded yet")

    score_phase(pairs, args.seeds, sig, root, cache, args.force)
    baseline_common.save_cache(CACHE_JSON, cache)
    aggregate_to_json(pairs, args.seeds, cache)


if __name__ == "__main__":
    main()
