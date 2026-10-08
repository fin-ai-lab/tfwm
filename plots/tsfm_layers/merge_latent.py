"""Reassemble the per-month latent jsons into the shape the serial run writes.

`run_tsfm_latent.sh` runs one SLURM job per eval month, so each analysis lands
as N single-month files instead of one pooled file. This merges them.

THE MERGE IS NOT AN APPROXIMATION. Every latent statistic is pooled ACROSS
months at the very end and never within one, so a per-month split loses
nothing — provided the merge redoes the pooling the way the serial run does,
rather than averaging summary numbers:

  fixed panel   `t` is (rate - chance) over MONTH-MEANS, `rank_t` is
                (pctile - 0.5) over month-means. So the per-month rates,
                chances, ranks and pctiles are re-stacked in month order and
                the two t-statistics recomputed from scratch. Averaging the
                per-month `t` values would be wrong and would not even be
                close.
  Pelger        results[key][stat] is already a list aligned with `months`, so
                the merge is a concatenation in month order. Nothing to redo.

Month order is canonical (sorted), not directory order, so the concatenated
lists line up with the merged `months` field for every key.

Usage:
    uv run python plots/tsfm_layers/merge_latent.py \
        --results-dir lab/tsfm_latent_32
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent
REPO = OUT_DIR.parents[1]
# temporal_prediction retired 2026-08-26; its per-month shards remain in
# lab/tsfm_latent_32 but are no longer merged.
PELGER = ("decode_loadings", "subspace_alignment")
# Two tags: _tsfml_ (frozen TSFMs) and _supl_ (supervised).
MONTH_RE = re.compile(r"_(?:tsfml|supl)_(\d{4}-\d{2})\.json$")


def by_month(results: Path, stem: str) -> dict:
    """{month: payload} for one analysis."""
    out = {}
    pats = (f"{stem}_tsfml_*.json", f"{stem}_supl_*.json")
    for p in sorted(q for pat in pats for q in results.glob(pat)):
        m = MONTH_RE.search(p.name)
        if m:
            out[m.group(1)] = json.loads(p.read_text())
    return out


def merge_fixed_panel(shards: dict) -> dict:
    """Re-pool T1-T4 from the per-month series, recomputing both t-statistics."""
    months = sorted(shards)
    first = shards[months[0]]
    models: dict = {}
    # A model/metric is only pooled over the months that actually carry it, so
    # a family missing from one month degrades that cell rather than the file.
    keys = sorted({k for s in shards.values() for k in s["models"]})
    for key in keys:
        for metric in ("metric1", "metric2", "metric3", "metric4"):
            rows, have = [], []
            for ym in months:
                ent = shards[ym]["models"].get(key, {}).get(metric)
                if not ent or "month_chances" not in ent:
                    continue
                rows.append([ent["month_rates"][ym], ent["month_chances"][ym],
                             ent["month_ranks"][ym], ent["month_pctiles"][ym]])
                have.append(ym)
            if len(rows) < 2:      # t needs a spread across months
                continue
            a = np.array(rows, dtype=float)
            rates, chances, ranks, pcts = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
            ex, ex_r = rates - chances, pcts - 0.5
            src = shards[have[0]]["models"][key][metric]
            models.setdefault(key, {})[metric] = {
                "rate": float(rates.mean()),
                "chance": float(chances.mean()),
                "t": float(ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))),
                "mean_rank": float(ranks.mean()),
                "mean_pctile": float(pcts.mean()),
                "rank_t": float(ex_r.mean() / (ex_r.std(ddof=1) / np.sqrt(len(ex_r)))),
                "n_candidates": src["n_candidates"],
                "month_rates": {m: float(v) for m, v in zip(have, rates)},
                "month_ranks": {m: float(v) for m, v in zip(have, ranks)},
                "month_chances": {m: float(v) for m, v in zip(have, chances)},
                "month_pctiles": {m: float(v) for m, v in zip(have, pcts)},
                "n_months": len(have),
            }
    return {"P": first["P"], "S": first["S"], "n_panels": first["n_panels"],
            "metric_names": first["metric_names"], "months": months,
            "models": models}


def merge_pelger(shards: dict, stem: str) -> dict:
    """Concatenate the per-month lists in month order."""
    months = sorted(shards)
    first = shards[months[0]]
    merged: dict = {}
    keys = sorted({k for s in shards.values() for k in s["results"]})
    for key in keys:
        for ym in months:
            entry = shards[ym]["results"].get(key)
            if entry is None:
                continue
            for stat, vals in entry.items():
                if isinstance(vals, list):
                    merged.setdefault(key, {}).setdefault(stat, []).extend(vals)
                else:                       # scalar: keep the first month's
                    merged.setdefault(key, {}).setdefault(stat, vals)
    out = {k: v for k, v in first.items() if k not in ("results", "months")}
    out["months"] = months
    out["results"] = merged
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--geo-out",
                    default=str(OUT_DIR / "fixed_panel_P3S2_tsfmlayers32.json"))
    # latent_sweep.py hardcodes the _tsfmlayers tag, so the merged Pelger files
    # keep that name and live in their OWN directory instead. Pass it as
    # --fs-dir and the reducer needs no change.
    ap.add_argument("--fs-out", default=str(REPO / "plots" / "latent_eval" / "factors"
                                            / "tsfm32"))
    args = ap.parse_args()

    results = Path(args.results_dir)
    if not results.is_dir():
        print(f"no such directory: {results}")
        return 1

    geo = by_month(results, "fixed_panel_P3S2")
    if geo:
        merged = merge_fixed_panel(geo)
        Path(args.geo_out).write_text(json.dumps(merged, indent=1))
        n_cells = sum(len(v) for v in merged["models"].values())
        print(f"fixed panel: {len(geo)} months, {len(merged['models'])} models, "
              f"{n_cells} (model, metric) cells -> {args.geo_out}")
    else:
        print("fixed panel: no per-month files found")

    fs_out = Path(args.fs_out)
    fs_out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for stem in PELGER:
        shards = by_month(results, stem)
        if not shards:
            print(f"{stem}: no per-month files found")
            continue
        merged = merge_pelger(shards, stem)
        dest = fs_out / f"{stem}_tsfmlayers.json"
        dest.write_text(json.dumps(merged, indent=1))
        counts[stem] = len(shards)
        print(f"{stem}: {len(shards)} months, {len(merged['results'])} series "
              f"-> {dest}")

    if len(set(counts.values()) | ({len(geo)} if geo else set())) > 1:
        print("\nWARNING: the analyses do not cover the same months — "
              f"fixed panel {len(geo)}, " +
              ", ".join(f"{k} {v}" for k, v in counts.items()) +
              ". Pooling across a ragged set is not what the serial run does.")

    print(f"\nreduce with:\n  uv run python plots/tsfm_layers/latent_sweep.py \\\n"
          f"      --geo {args.geo_out} --fs-dir {fs_out} \\\n"
          f"      --out {OUT_DIR / 'latent_sweep_32'} "
          f"--json {OUT_DIR / 'latent_sweep_32.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
