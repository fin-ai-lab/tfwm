"""Per-month supervised-specialist ICs, as a json_ic artifact.

DEFAULT SOURCE: THE LIVE PAIRWISE HEADS (``--source ablation``). One recipe
per target -- ``<task>_pairwise_rep32nost_lr-3e-5_bs256_s42`` under
``supervised-loss-ablation-*`` -- covering the reported 32 months exactly.
These carry the month in the PROJECT suffix, so ``style.iter_ic_runs`` reads
them directly.

WHY IT MOVED. The old default (``--source ce``) walked the three flat
cross-entropy projects ``supervised-full-month-<task>-ce/``. Those were moved
to lab/models-archive and NO LONGER EXIST under the live checkpoint
root, so re-running the old default would have quietly written a near-empty
artifact -- and until then the breadth figure's "Supervised" line was fed by
ARCHIVED models through a checked-in JSON, where no grep for the archive
would ever find it. ``--source ce`` still works: point ``--ckpt-root`` at the
archive.

The record shape is unchanged (``{eval_month, target, ic, model, predictor}``)
so metrics.SERIES_DEFS picks it up with no new loader, and each panel still
takes its own target out of the one file.

TARGET GENERATION. Runs are restricted to one anchor-stat table (``--xs-stats``,
default ``xs_anchor_stats_fwdvwap60``) and THE ARTIFACT NOW RECORDS IT, so a
consumer can tell which generation a checked-in file belongs to instead of
inferring it from the mtime. Both the mid-to-mid and forward-VWAP sweeps live
under the same run names, so without the filter a rerun month contributes two
records for one (eval_month, target) and the definitions average together.

``--readout head`` writes the head's expected-bin IC instead of the probe's,
to ``supervised_head_ic.json`` -- the gray star the breadth figures draw beside
the gray probe line. A head is trained at ONE horizon, so the scorer writes
``xs_ic/head:<task>`` for h=900 and nothing else and the artifact is a third
the size of the probe's; ``supervised_head`` in metrics.SERIES_DEFS carries
``point_only_h`` for exactly that reason.

Usage::

    uv run plots/metrics/build_supervised_ic.py
    uv run plots/metrics/build_supervised_ic.py --readout head
    uv run plots/metrics/build_supervised_ic.py --source ce \\
        --ckpt-root lab/models-archive
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from stable_finance.dataset import next_month  # noqa: E402

CKPT_ROOT = Path("lab/market-jepa-checkpoints")
# --source ce: flat projects, month in the RUN NAME. Archived.
PROJECTS = {
    "supervised-full-month-return": "return",
    "supervised-full-month-vol-change": "volatility_change",
    "supervised-full-month-spread-change": "spread_change",
}
# --source specialists (DEFAULT since 2026-09-13): the live specialists sweep.
#
# ONE PROJECT PER BUNDLE -- "<name>-<commit6>-<train_start>-<train_end>" -- so
# these are prefixes, and the COMMIT PINS THE WAVE. That is the whole pointer:
# to read a newer wave, change SPECIALISTS_COMMIT and nothing else.
#
# Pinning the commit rather than taking every hash is what keeps the panel
# clean. Under these prefixes right now: 3e5087 is the single-month generation
# this campaign replaced, and 62bffe is a six-month wave that was submitted
# through the full-history launcher by mistake and cancelled, so it carries the
# right recipe on the WRONG months. Matching the prefix alone would average all
# three into one figure and nothing in a run name would say so.
SPECIALISTS_COMMIT = "581eb2"
SPECIALISTS_PROJECTS = {
    "supervised-full-month-return": "return",
    "supervised-full-month-vol-change": "volatility_change",
    "supervised-full-month-spread-change": "spread_change",
}

# --source ablation: month in the PROJECT suffix, so iter_ic_runs sees it.
ABLATION_GLOB = "supervised-loss-ablation-*"
ABLATION_RUNS = {
    "return_pairwise_rep32nost_lr-3e-5_bs256_s42": "return",
    "volatility_change_pairwise_rep32nost_lr-3e-5_bs256_s42":
        "volatility_change",
    "spread_change_pairwise_rep32nost_lr-3e-5_bs256_s42":
        "spread_change",
}
HORIZONS = (300, 600, 900, 1800, 3600, 7200)
MONTH_RE = re.compile(r"^(\d{4})-(\d{2})")


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--ckpt-root", default=str(CKPT_ROOT))
    p.add_argument("--source", choices=("specialists", "ablation", "ce"),
                   default="specialists",
                   help="specialists: the live sweep (default). ablation: the "
                        "pairwise loss-ablation projects. ce: the archived "
                        "flat cross-entropy projects -- pass --ckpt-root "
                        "lab/models-archive with it.")
    p.add_argument("--commit", default=SPECIALISTS_COMMIT,
                   help="which wave of the specialists sweep to read "
                        f"(default {SPECIALISTS_COMMIT})")
    p.add_argument("--readout", choices=("probe", "head"), default="probe")
    p.add_argument("--xs-stats", default=None,
                   help="anchor-stat table basename to keep "
                        "(default: checkpoints.SUP_XS_STATS)")
    p.add_argument("--months-from-k2ind", action="store_true", default=True,
                   help="keep only the standard 32 eval months")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    keep = None
    if a.months_from_k2ind:
        # THE CANONICAL 32, from the months script -- not
        # style.standard_k2ind_months. That helper derives the panel from the
        # standard k2ind model's own runs, which was a good guard while that
        # model lived under the live root; it is now ARCHIVED, so the helper
        # globs the live tree, finds zero months and raises. The month list is
        # a property of the SWEEP, not of any one model, and load_sweep_months
        # is the same list the launchers and every ckpt series pin against.
        from style import load_sweep_months
        keep = {next_month(m) for m in load_sweep_months()}

    from market_jepa.eval.checkpoints import SUP_XS_STATS, meta_xs_stats
    want_stats = a.xs_stats or SUP_XS_STATS

    root = Path(a.ckpt_root)
    recs, seen_months = [], set()
    dropped_target = 0

    if a.source == "specialists":
        from style import iter_ic_runs
        n_runs = 0
        for prefix, ttype in SPECIALISTS_PROJECTS.items():
            for run in iter_ic_runs(f"{prefix}-{a.commit}-*", root,
                                    xs_stats=want_stats):
                n_runs += 1
                ev = run.eval_month
                if keep is not None and ev not in keep:
                    continue
                seen_months.add(ev)
                for h in HORIZONS:
                    task = f"{ttype}_{h}"
                    v = (run.probe(task) if a.readout == "probe"
                         else run.ic.get(f"xs_ic/head:{task}"))
                    if v is None:
                        continue
                    recs.append({"eval_month": ev, "target": task,
                                 "predictor": f"Supervised ({a.readout})",
                                 "model": f"supervised_{a.readout}",
                                 "ic": float(v)})
        # A SCORED RUN WITH NO PROBE IS NOT AN EMPTY SWEEP. The campaign runs
        # POST_TRAIN_PROBE=0, so it writes xs_ic/head:<task> and no probe key
        # at all; asking for the probe here finds runs and no numbers, which
        # the generic "found nothing" message would report as a missing sweep.
        if a.readout == "probe" and n_runs and not recs:
            raise SystemExit(
                f"{n_runs} run(s) found at commit {a.commit}, but none carries "
                f"a probe IC: this sweep scores the HEAD only "
                f"(POST_TRAIN_PROBE=0). Use --readout head.")
        _finish(a, recs, seen_months, want_stats, dropped_target, keep)
        return

    if a.source == "ablation":
        from style import iter_ic_runs
        for run in iter_ic_runs(ABLATION_GLOB, root, xs_stats=want_stats):
            ttype = ABLATION_RUNS.get(run.run_name)
            if ttype is None:
                continue
            ev = run.eval_month
            if keep is not None and ev not in keep:
                continue
            seen_months.add(ev)
            for h in HORIZONS:
                task = f"{ttype}_{h}"
                v = (run.probe(task) if a.readout == "probe"
                     else run.ic.get(f"xs_ic/head:{task}"))
                if v is None:
                    continue
                recs.append({"eval_month": ev, "target": task,
                             "predictor": f"Supervised ({a.readout})",
                             "model": f"supervised_{a.readout}",
                             "ic": float(v)})
        _finish(a, recs, seen_months, want_stats, dropped_target, keep)
        return

    for proj, ttype in PROJECTS.items():
        for meta_f in sorted((root / proj).glob("*/train_meta.json")):
            ic_f = meta_f.with_name("xs_ic.json")
            if not ic_f.is_file():
                continue
            try:
                meta = json.loads(meta_f.read_text())
                ic = json.loads(ic_f.read_text())
            except json.JSONDecodeError:
                continue
            if meta_xs_stats(meta) != want_stats:
                dropped_target += 1
                continue
            m = MONTH_RE.match(meta.get("run_name") or "")
            if not m:
                continue
            ev = next_month(f"{m.group(1)}-{m.group(2)}")
            if keep is not None and ev not in keep:
                continue
            seen_months.add(ev)
            for h in HORIZONS:
                task = f"{ttype}_{h}"
                key = (f"xs_ic/{task}" if a.readout == "probe"
                       else f"xs_ic/head:{task}")
                v = ic.get(key)
                if v is None:
                    continue
                recs.append({"eval_month": ev, "target": task,
                             "predictor": f"Supervised ({a.readout})",
                             "model": f"supervised_{a.readout}",
                             "ic": float(v)})

    _finish(a, recs, seen_months, want_stats, dropped_target, keep)


def _finish(a, recs, seen_months, want_stats, dropped_target, keep):
    dupes = len(recs) - len({(r["eval_month"], r["target"]) for r in recs})
    if dupes:
        raise SystemExit(
            f"{dupes} duplicate (eval_month, target) record(s) on "
            f"{want_stats}; two checkpoints claim the same month.")
    if not recs:
        raise SystemExit(
            f"no records: source={a.source} found nothing under "
            f"{a.ckpt_root}. The --source ce projects are ARCHIVED; pass "
            "--ckpt-root lab/models-archive for them.")

    # STAMP THE TARGET GENERATION on every record. A json_ic artifact carries
    # none of the ckpt pins, so without this a stale file is indistinguishable
    # from a current one and would silently mix two target definitions on one
    # axis -- exactly what ``xs_stats`` prevents for checkpoint-backed series.
    for r in recs:
        r["xs_anchor_stats"] = want_stats
        r["source"] = a.source

    out = Path(a.out or _HERE / f"supervised_{a.readout}_ic.json")
    out.write_text(json.dumps(recs, indent=2))
    print(f"{len(recs)} records over {len(seen_months)} eval months "
          f"(source={a.source}, {want_stats}) -> {out}")
    if dropped_target:
        print(f"  dropped {dropped_target} checkpoint(s) trained on a "
              f"different anchor-stat target than {want_stats}")
    if keep is not None and seen_months != keep:
        print(f"  NOTE missing {sorted(keep - seen_months)}")


if __name__ == "__main__":
    main()
