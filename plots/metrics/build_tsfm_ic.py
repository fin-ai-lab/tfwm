"""``tsfm_ic.json`` — per-month rank IC for each frozen TSFM at its ONE layer.

The three frozen time-series foundation models (Chronos-2, TimesFM 3.0 and
Kronos-base) are scored by ``scripts/eval/tsfm_layer_ic.py`` at EVERY hidden
state, and ``plots/tsfm_layers/layer_sweep.py`` reduces that into a layer
profile. A breadth figure wants none of that: it wants one number per (family,
target, month), at the single layer the model is reported at.

THREE FAMILIES, NOT FIVE. TimesFM 2.5 and Sundial were dropped from the arm on
2026-09-11 and were never re-scored at the protocol readout below, so there is
nothing on disk to report them from. Their layer profiles survive in
``plots/tsfm_layers/layer_sweep_32.json`` at the retired mean readout.

READ AT THE PREDICTION TOKEN. Every number here is ``--time-pool last``, which
is ``xs_ic_eval.PREDICT_POOL`` — the token the random-init floor subtracted
below is itself read at, and the rule ``checkpoints.py`` states outright: a
floor is read the way the models it floors are read. Until 2026-09-11 this
artifact was built from a mean-pooled sweep against a last-token floor, which
is two readouts in one subtraction; the last-token rescore moved Chronos-2's
spread change from +0.173 to +0.217 and relocated most of the argmax layers, so
this is not a cosmetic difference. ``time_pool`` is CHECKED on every payload
and travels on every record, because the results are ``<family>_<month>.json``
with no readout in the name and the two sweeps are otherwise indistinguishable.

THE LAYER IS A HYPER-PARAMETER AND IS CHOSEN OFF-PANEL. ``--picks-from``
defaults to the layer_sweep json of the 5-month OPTIMIZATION set
(``scripts/experiments/holdout_months.py``), whose per-(family, task) argmax is
the pick; ``--results-dir`` defaults to the 32-month REPORTED panel, which is
what those picks then score. Pointing both at one panel would be selection on
the test set and is refused (see ``--allow-same-panel``).

THE LAYER IS PER TASK, not per family. Chronos-2 reads return off layer 5,
volatility change off layer 2 and spread change off layer 2, because that is
what the optimization set chose for each. So a "Chronos-2" line on a breadth
figure is one model with three readouts, and each panel says which layer it is
drawing (``layer`` travels on every record).

RAW IC, NOT ΔIC. Every other ``json_ic`` artifact in this folder stores the raw
per-month rank IC and lets ``metrics.load_ic_metrics`` subtract the random-init
floor month by month, so that the figure's subtrahend is the same object for
every series on it. This one does the same. ``plots/tsfm_layers`` subtracts its
own copy of that floor (``randinit-fwd3-*``, seeds averaged) and the two agree,
but only one of them should be in the figure's arithmetic. That floor is read
on a 9 + 11 channel panel while the TSFMs see 9 — an accepted asymmetry, NOT a
bug; ``plots/tsfm_layers/README.md`` ("ONE FLOOR FOR EVERY ARM") says how to
read the spread-change column because of it.

ONE HORIZON. The sweep ran at h=900 and nowhere else, so these records exist
only for ``*_900`` and the figure draws them as points rather than lines. Every
payload's ``horizon`` is checked, because a mixed results directory would put
two horizons under one key.

Run:
    uv run plots/metrics/build_tsfm_ic.py
    uv run plots/metrics/build_tsfm_ic.py --allow-same-panel \
        --results-dir lab/tsfm_layer_ic_lastpool \
        --picks-from plots/tsfm_layers/layer_sweep_lastpool.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "plots"))

from style import SERIES_STYLES  # noqa: E402

# The reported panel: the 32 sweep months' worth of (family, train month)
# payloads. The optimization set lives in lab/tsfm_layer_ic_lastpool and
# is where --picks-from reads from, never where the reported numbers come from.
# BOTH are the _lastpool sweeps; the unsuffixed directories beside them are the
# retired mean readout and are refused by the time_pool check below.
RESULTS = Path("lab/tsfm_layer_ic_32_lastpool")
PICKS = REPO / "plots" / "tsfm_layers" / "layer_sweep_lastpool.json"
OUT_JSON = OUT_DIR / "tsfm_ic.json"

FAMILY_ORDER = ["chronos2", "timesfm3", "kronos"]
TASKS = ["return_900", "volatility_change_900", "spread_change_900"]
# The reported protocol's ridge alpha, spelled as the key the payloads use.
# Fixed at xs_ic_eval.RIDGE_ALPHAS for the reason in plots/tsfm_layers/README:
# every arm this figure is read against is quoted at 10, and an arm allowed to
# tune its own regularizer is not on the same axis as the rest.
PROTOCOL_ALPHA = "10"
HORIZON = 900
# xs_ic_eval.PREDICT_POOL, spelled as the payload field. See READ AT THE
# PREDICTION TOKEN: this is the one field that separates the reported sweep
# from the retired one, and the file names do not carry it.
PREDICT_POOL = "last"


def load_payloads(results_dir: Path) -> dict[str, dict[str, dict]]:
    """``{family: {train_month: payload}}``, one json per (family, month pair).

    Smoke runs are dropped exactly as ``layer_sweep.load_sweeps`` drops them:
    a strided run reads a fraction of the shards and its IC is not the panel's.
    """
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for p in sorted(results_dir.glob("*.json")):
        payload = json.loads(p.read_text())
        if payload.get("shard_stride", 1) != 1:
            print(f"  skipping {p.name}: shard_stride="
                  f"{payload['shard_stride']} is a smoke run, not the panel")
            continue
        if payload.get("horizon") != HORIZON:
            raise SystemExit(
                f"{p.name} was scored at h={payload.get('horizon')}, not "
                f"{HORIZON}. Every record this writes is keyed '<target>_900', "
                f"so a second horizon in the same directory would land under "
                f"the same key as the first.")
        # Absent on runs predating 2026-09-11, which are all at the retired
        # mean readout -- so the default here is 'mean' and those files are
        # refused. tsfm_layer_ic.py still DEFAULTS to mean so everything on
        # disk reproduces byte-for-byte; the protocol readout is opt-in, which
        # is exactly why it has to be verified rather than assumed.
        # A readout field is null when the family HAS NO SUCH AXIS -- Kronos
        # embeds a window whole, so channel_pool is meaningless for it and
        # tsfm_layer_ic.py writes None rather than a value it did not honour.
        # Absent is different from null and means the field predates the run:
        # the mean readout for time_pool, concat for channel_pool.
        time_pool = payload.get("time_pool", "mean")
        if time_pool is not None and time_pool != PREDICT_POOL:
            raise SystemExit(
                f"{p.name} was read at time_pool={time_pool!r}"
                + (" (field absent)" if "time_pool" not in payload else "")
                + f", not {PREDICT_POOL!r}. A prediction number in this paper "
                f"is read at xs_ic_eval.PREDICT_POOL and so is the random-init "
                f"floor subtracted from it; mixing readouts would subtract a "
                f"last-token floor from a mean-pooled model. Point "
                f"--results-dir at a --time-pool last sweep "
                f"(lab/tsfm_layer_ic{{,_32}}_lastpool).")
        channel_pool = payload.get("channel_pool", "concat")
        if channel_pool is not None and channel_pool != "concat":
            raise SystemExit(
                f"{p.name} carries channel_pool={channel_pool!r}. The "
                f"reported TSFM numbers are the per-channel CONCAT readout; "
                f"the mean-pool arm is an ablation "
                f"(plots/tsfm_layers/channel_pool.py) and must not be mixed "
                f"into this artifact.")
        out[payload["family"]][payload["train_month"]] = payload
    return out


def load_picks(path: Path) -> dict[str, dict[str, int]]:
    """``{family: {task: layer}}`` — the optimization set's per-task argmax."""
    doc = json.loads(path.read_text())
    return {fam: {t: b["layer"] for t, b in fo.get("best", {}).items()}
            for fam, fo in doc.get("families", {}).items()}


def build(results_dir: Path, picks_path: Path,
          allow_same_panel: bool) -> list[dict]:
    payloads = load_payloads(results_dir)
    if not payloads:
        raise SystemExit(f"no sweep results in {results_dir}")
    picks = load_picks(picks_path)
    if not picks:
        raise SystemExit(f"{picks_path} carries no 'best' layers to pick from")

    # The whole point of the two directories. If the picks were made on the
    # very months being scored here, the argmax is selection on the test set
    # and every number below is inflated by the search.
    pick_months = {m for fo in json.loads(picks_path.read_text())
                   .get("families", {}).values() for m in fo.get("months", [])}
    panel_months = {m for runs in payloads.values() for m in runs}
    if pick_months & panel_months and not allow_same_panel:
        raise SystemExit(
            f"{len(pick_months & panel_months)} of {len(panel_months)} months "
            f"in {results_dir} also chose the layers in {picks_path}: "
            f"{' '.join(sorted(pick_months & panel_months))}\n"
            f"That is selection on the test set. Point --picks-from at the "
            f"optimization set's layer_sweep.json, or pass "
            f"--allow-same-panel and know the numbers are inflated.")

    records: list[dict] = []
    for family in FAMILY_ORDER:
        runs = payloads.get(family)
        if not runs:
            print(f"  {family}: nothing in {results_dir}, skipped")
            continue
        fam_picks = picks.get(family)
        if not fam_picks:
            raise SystemExit(
                f"{picks_path} has no layer picks for {family!r}, but "
                f"{results_dir} holds {len(runs)} of its month pairs. A "
                f"family cannot be reported without a held-out layer.")
        label = SERIES_STYLES[family]["label"]
        for task in TASKS:
            layer = fam_picks.get(task)
            if layer is None:
                raise SystemExit(
                    f"no held-out layer for {family}/{task} in {picks_path}")
            n = 0
            for train_month, payload in sorted(runs.items()):
                cell = (payload["results"].get(str(layer), {})
                        .get(PROTOCOL_ALPHA, {}).get(task))
                if cell is None:
                    # A layer that failed on one month (OOM, a non-PSD Gram
                    # the eigh clip could not rescue) is a hole in the panel,
                    # not a month to quietly drop: the figure averages over
                    # whatever it is given.
                    print(f"  WARN {family}/{task}: no cell for layer {layer} "
                          f"alpha {PROTOCOL_ALPHA} in {train_month} "
                          f"(failed: {payload.get('layers_failed') or 'none'})")
                    continue
                records.append({
                    "eval_month": payload["eval_month"],
                    "train_month": train_month,
                    "target": task,
                    "predictor": label,
                    "model": family,
                    "ic": cell["ic"],
                    "se": cell["se"],
                    "n_cells": cell.get("n_cells"),
                    "layer": layer,
                    "alpha": float(PROTOCOL_ALPHA),
                    "layer_source": str(picks_path.relative_to(REPO)),
                    "channel_pool": payload.get("channel_pool", "concat"),
                    "time_pool": payload.get("time_pool", "mean"),
                })
                n += 1
            print(f"  {family:9s} {task:22s} layer {layer:<3d} {n:2d} months")
    return records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default=str(RESULTS),
                   help="per-(family, month pair) sweep jsons for the "
                        "REPORTED panel (default: the 32 sweep months)")
    p.add_argument("--picks-from", default=str(PICKS),
                   help="layer_sweep json whose per-(family, task) argmax "
                        "layer is the pick. Must be a DIFFERENT panel than "
                        "--results-dir (default: the 5-month optimization set)")
    p.add_argument("--allow-same-panel", action="store_true",
                   help="score the picks on the months that made them. A "
                        "diagnostic; the numbers are inflated by the search.")
    p.add_argument("--out", default=str(OUT_JSON))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    recs = build(Path(args.results_dir), Path(args.picks_from),
                 args.allow_same_panel)
    if not recs:
        raise SystemExit("no records built — nothing to write")
    Path(args.out).write_text(json.dumps(recs, indent=1))
    months = {r["eval_month"] for r in recs}
    print(f"\nwrote {args.out}: {len(recs)} records, "
          f"{len({r['model'] for r in recs})} families, "
          f"{len(months)} eval months")


if __name__ == "__main__":
    main()
