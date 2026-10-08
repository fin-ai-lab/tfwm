"""Shared plumbing for the forward-decay v2 grid and its random-init baselines.

Two scripts sit on top of this module and MUST agree cell for cell:
``forward_eval_v2.py`` (the checkpoint curves) and
``calculate_random_init_baselines.py`` (the floor those curves are read
against). A cell is addressed by ``(probe_train_month, H)``, so a horizon
grid, a month list or a calendar helper that drifted between the two would
produce a figure whose baseline is silently for different months. They live
here for that reason, not for tidiness.

Files
-----
``calculate_random_init_baselines_cache.json``
    Per-seed rank ICs — the unaggregated signal. Keyed
    ``{probe_train_month}/{H}/{backbone}/{seed} → {metric: ic|None}``.
    ``None`` means the cell was attempted and produced nothing usable (too few
    finite rows in the panel) and is retried on the next run.

``baseline.json``
    The aggregate the figures read. ``{metric}`` holds mean/std/n keyed
    ``{probe_train_month: {H: {...}}}``; ``{metric}_by_backbone.{backbone}``
    carries the same aggregates plus a ``per_seed`` list for diagnostics.

``{metric}`` is a ``{target_type}_{horizon}`` column name, matching
``market_jepa.eval.tasks`` and the keys ``xs_ic_eval.score`` emits — e.g.
``return_900``. The old ``_k{k}`` suffix is gone with the binned AUC it named.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ─── the grid both scripts walk ─────────────────────────────────────────────

# Probe-train month is the ckpt's train_end; eval_month = train_end + H
# months. H=1 is the cell the in-job scorer already computed for every
# checkpoint (its xs_ic.json), which is what lets forward_eval_v2 seed that
# column for free.
HORIZONS: tuple[int, ...] = (1, 2, 3, 6, 9, 12, 15, 18, 21, 24,
                             30, 36, 42, 48, 54, 60)

# Last calendar month with mosaic data. 2025/01..03 exist but are sample-less
# stubs, so the real end is 2024-12; cells past it are dropped from the grid
# rather than attempted.
DATA_END_MONTH: str = "2024-12"

# Train months whose ckpts the figures do not plot. The 2023 pair would only
# reach H=21 before running off the data, so they contribute short columns and
# a ragged tail; the grid is ragged enough already.
SKIP_TRAIN_YEAR_PREFIXES: tuple[str, ...] = ("2023-",)

EVAL_HORIZON: int = 900
EVAL_TARGET_TYPES: tuple[str, ...] = (
    "return", "volatility_change", "spread_change",
)
TARGET_COLS: tuple[str, ...] = tuple(
    f"{t}_{EVAL_HORIZON}" for t in EVAL_TARGET_TYPES
)

# Eval months are embedded at the reported 8 anchors/day; probe-fit months at
# TRAIN_ANCHORS_PER_DAY (36), where extra rows ride the same decode. Both
# scripts read these from here so a panel built by one is reused by the other.
ANCHORS_PER_DAY: int = 8

# The campaign's 32 eval months, single-sourced from the manifest the metrics
# figures read. A train_end is the month BEFORE its eval month, which is what
# v2 fits the probe on.
FAIR_MONTHS_JSON = (
    Path(__file__).resolve().parents[2] / "plots" / "metrics" / "noclamp_manifest.json"
)


def add_months(ym: str, n: int) -> str:
    """'YYYY-MM' + n calendar months."""
    y, m = map(int, ym.split("-"))
    total = y * 12 + (m - 1) + n
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def train_months() -> list[str]:
    """The probe-train months of the grid: fair eval months, minus one."""
    fair = json.loads(FAIR_MONTHS_JSON.read_text())["fair_months"]
    return sorted(
        ym for ym in {add_months(f, -1) for f in fair}
        if not any(ym.startswith(p) for p in SKIP_TRAIN_YEAR_PREFIXES)
    )


def eval_months() -> list[str]:
    """The campaign's 32 reported eval months, in order."""
    return sorted(json.loads(FAIR_MONTHS_JSON.read_text())["fair_months"])


def _mi(ym: str) -> int:
    y, m = map(int, ym.split("-"))
    return y * 12 + (m - 1)


def build_pairs(max_gap: int = 60) -> list[tuple[str, str, int]]:
    """``[(train_month, eval_month, gap)]`` over the 32 x 32 panel.

    THE ALL-PAIRS DESIGN (user, 2026-08-31), and why it replaces the rounded
    horizon grid for measuring decay. Every checkpoint is scored on every one
    of the 32 reported eval months that lies 1..``max_gap`` months ahead of its
    training month, so the horizon is whatever the calendar gives -- 59
    distinct values between 1 and 60 rather than 8 chosen ones -- and the
    scatter has structure between the round numbers.

    Two things fall out that the grid could not give:

    * THE REFERENCE IS FREE AND IS THE SAME ARM. Eval month E is one of the 32,
      so the checkpoint trained on ``E - 1`` is in the panel too; the age-1
      model to subtract at E is that checkpoint, not an outside series. The
      gap=1 pairs ARE that reference (train_month = eval_month - 1), which is
      why they are kept in the manifest rather than read from xs_ic.json --
      one run, one code path, and the overlap with xs_ic.json becomes a check.
    * EVERY MONTH IS ALREADY BUILT. 32 probe months at 36 anchors and 32 eval
      months at 8 are exactly what the panel cache and the sweep already hold,
      against 183 mostly-uncached eval months for the grid.

    All 32 train months are usable here, including the 2023 pair the grid drops
    ([[SKIP_TRAIN_YEAR_PREFIXES]]): nothing runs off the end of the data when
    the eval months are drawn from the panel itself.
    """
    evs = eval_months()
    out = []
    for ev in evs:
        for tm in (add_months(e, -1) for e in evs):
            gap = _mi(ev) - _mi(tm)
            if 1 <= gap <= max_gap:
                out.append((tm, ev, gap))
    return sorted(out)


def build_grid() -> tuple[list[str], list[tuple[str, int, str]]]:
    """``(train_months, [(probe_train_month, H, eval_month)])``.

    Cells whose eval month runs past the data are dropped, not attempted, so
    long horizons are simply thinner — the ragged panel the user accepted
    2026-08-03 rather than truncating every column to the shortest one.
    """
    tms = train_months()
    pairs = [
        (ptm, h, add_months(ptm, h))
        for ptm in tms for h in HORIZONS
        if add_months(ptm, h) <= DATA_END_MONTH
    ]
    return tms, pairs


# ─── naming ─────────────────────────────────────────────────────────────────


def metric_name(target_type: str, horizon: int) -> str:
    """Key for this (target, horizon) in the cache + baseline.json.

    Identical to the column name ``xs_ic_eval.score`` returns, so a baseline
    cell and a checkpoint cell are looked up by the same string.
    """
    return f"{target_type}_{horizon}"


def cache_key(probe_train_month: str, h: int, backbone: str, seed: int) -> str:
    return f"{probe_train_month}/{h}/{backbone}/{seed}"


# ─── per-seed cache ─────────────────────────────────────────────────────────


def load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  cache load failed ({e}); starting fresh")
        return {}


def save_cache(path: Path, cache: dict) -> None:
    path.write_text(json.dumps(cache, indent=2, sort_keys=True))


# ─── aggregation → baseline.json ────────────────────────────────────────────


def aggregate(per_seed: list) -> dict:
    arr = np.array([v for v in per_seed if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


ABOUT = (
    "Forward-decay v2 random-init RANK IC baselines — the subtrahend of "
    "delta IC. Probe-train month is fixed at the train_end calendar month "
    "for every H in HORIZONS; eval_month = first_of_next_month(train_end) + "
    "(H-1) months. One ridge probe per (probe-train month, target) is fit on "
    "month 0 and reused at every H. Encoders are the paper's shared floor, "
    "lab/randinit_bb/randinit_s{42,43,44}. Targets are the "
    "two-forward-window generation (xs_anchor_stats_fwdvwap60); numbers from "
    "before 2026-08-22 are a different measurement, not a noisier one. "
    "Schema: `{metric}` is the mean/std/n over seeds, keyed "
    "`{probe_train_month: {H: {...}}}`; `{metric}_by_backbone.transformer` "
    "carries the same aggregates with a per-seed list for diagnostics. "
    "Written by plots/forward_eval_v2/calculate_random_init_baselines.py."
)


def write_baseline_json(
    path: Path,
    results: dict[str, dict[str, dict[str, dict[int, list]]]],
) -> None:
    """Merge ``results[metric][backbone][probe_train_month][h] = per_seed``
    into ``baseline.json``.

    Metrics absent from ``results`` are left untouched, so a run that only
    covers volatility_change does not disturb the return baselines.
    """
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            existing = {}
    existing["__about__"] = ABOUT

    for mname, by_bb in results.items():
        # Merge into existing so re-runs covering a new (ptm, H) subset
        # don't wipe aggregates computed in earlier invocations.
        prior_detail = existing.get(f"{mname}_by_backbone", {})
        detail: dict[str, dict[str, dict[str, dict]]] = {}
        for bb, by_ptm in prior_detail.items():
            if not isinstance(by_ptm, dict):
                continue
            detail[bb] = {}
            for ptm, by_h in by_ptm.items():
                if isinstance(by_h, dict):
                    detail[bb][ptm] = dict(by_h)

        for backbone, by_ptm in by_bb.items():
            detail.setdefault(backbone, {})
            for ptm, by_h in by_ptm.items():
                detail[backbone].setdefault(ptm, {})
                for h, per_seed in by_h.items():
                    agg = aggregate(per_seed)
                    agg["per_seed"] = [None if v is None else float(v) for v in per_seed]
                    detail[backbone][ptm][str(h)] = agg
        existing[f"{mname}_by_backbone"] = detail

        # Flat aggregate, recomputed from merged detail so this stays
        # consistent as new (ptm, H) combos get added over time.
        transformer_detail = detail.get("transformer", {})
        flat: dict[str, dict[str, dict]] = {}
        for ptm in sorted(transformer_detail):
            flat[ptm] = {}
            for h_str in sorted(transformer_detail[ptm], key=lambda x: int(x)):
                per_seed = transformer_detail[ptm][h_str].get("per_seed") or []
                flat[ptm][h_str] = aggregate(per_seed)
        existing[mname] = flat

    path.write_text(json.dumps(existing, indent=2, sort_keys=True))
