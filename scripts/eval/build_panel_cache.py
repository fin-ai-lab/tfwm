"""Pre-build the synchronized eval panels for the reported + holdout months.

The panel is a function of (month, anchors, geometry) and NOT of the
checkpoint, so every run scored on a month currently rebuilds an identical one
-- ~3028 s per run, against ~2400 s for the training it scores, with ~80 runs
sharing each month in the ssl_ic campaign. This builds each once.

SCOPE IS DELIBERATE: the 32 reported sweep months and holdout set 2, in both
roles a month can play --
    probe-fit role  36 anchors/day  (TRAIN_ANCHORS_PER_DAY)  ~22 GB float16
    eval role        8 anchors/day                            ~4 GB float16
NOT sweeps/full_data_supervised.sh's 203 months: that would be ~5 TB and each
of its months is scored once, so a cache there is pure cost.

Usage:
    MJ_PANEL_CACHE=lab/market-jepa-mosaic/panel_cache \
      uv run scripts/eval/build_panel_cache.py --set all
    ... --set holdout1 --dry-run          # sizes only, builds nothing
    ... --months 2016-03 --roles eval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/eval"))
sys.path.insert(0, str(ROOT / "scripts/experiments"))

import panel_cache as pc  # noqa: E402
from stable_finance.dataset import (  # noqa: E402
    MarketSchedule,
    ViewSpec,
    iter_month_panel,
    next_month,
)
from market_jepa.schemas import LocalMachineConfig, machine_from_env  # noqa: E402
from stable_finance.dataset import AnchorTargetStats as AnchorStats  # noqa: E402
from xs_ic_eval import TRAIN_ANCHORS_PER_DAY, day_anchors  # noqa: E402


def month_set(which: str) -> list[str]:
    # HOLDOUT SET 1 WAS MISSING AND IT IS THE ONE MOST SWEEPS USE. "all" built
    # sweep32 + holdout2 and silently skipped set 1 -- the optimization panel
    # the frozen-TSFM layer sweep, the readout sweep and every other
    # hyper-parameter run are scored on -- so those runs re-decoded all ten of
    # its months from the mosaic on every pass while "the cache" looked
    # complete. Added 2026-09-11, along with set 1 joining "all".
    from holdout_months import HOLDOUT_MONTHS, HOLDOUT_MONTHS_2
    from sample_sweep_months import SWEEP_MONTHS
    if which == "sweep32":
        return list(SWEEP_MONTHS)
    if which == "holdout1":
        return list(HOLDOUT_MONTHS)
    if which == "holdout2":
        return list(HOLDOUT_MONTHS_2)
    return sorted(set(SWEEP_MONTHS) | set(HOLDOUT_MONTHS)
                  | set(HOLDOUT_MONTHS_2))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set",
                   choices=("sweep32", "holdout1", "holdout2", "all"),
                   default="all")
    p.add_argument("--months", nargs="*", default=None, help="override --set")
    p.add_argument("--roles", nargs="*", choices=("probe", "eval"),
                   default=["probe", "eval"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--info-token", choices=("off", "on", "both"), default="off",
                   help="which PANEL GEOMETRY to build. 'off' is the ssl_ic "
                        "campaign contract (9 channels). 'on' is the repo "
                        "default since 2026-08-25 (9 + 8 norm-stat + 3 window "
                        "= 20 channels, so ~2.2x the bytes) and is what the "
                        "lejepa / supervised sweeps run. A checkpoint only "
                        "ever hits the panel matching ITS OWN flags -- a "
                        "mismatch is a miss, never a wrong score.")
    p.add_argument("--jobs", type=int, default=1,
                   help="panels to build CONCURRENTLY. iter_panel's decode is "
                        "single-threaded, so one build pins exactly one core "
                        "and more processes is the only way to use the box. "
                        "~1 GB RSS and its own TMPDIR each.")
    p.add_argument("--xs-anchor-stats-dir",
                   default="lab/market-jepa-mosaic/xs_anchor_stats_fwdvwap60")
    args = p.parse_args()

    # /tmp on the data host is small and chronically full, and open_shard stages every
    # zstd shard through TMPDIR -- an unset TMPDIR fails the build with ENOSPC
    # partway in. Same default run_eval.sh uses.
    import os
    os.environ.setdefault(
        "TMPDIR", "lab/market-jepa-checkpoints/_scratch/panel_tmp")
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    root = pc.cache_root()
    if root is None:
        print("MJ_PANEL_CACHE is not set — nothing to build into", file=sys.stderr)
        return 1
    root.mkdir(parents=True, exist_ok=True)

    machine = machine_from_env(LocalMachineConfig)
    schedule = MarketSchedule(machine.holiday_csv)
    stats_dir = Path(args.xs_anchor_stats_dir)

    train_months = args.months or month_set(args.set)
    # A month is cached in whichever ROLES it is used in. A training month is
    # probe-fit at 36 anchors; its eval month (train+1) is scored at 8.
    todo: list[tuple[str, int]] = []
    for ym in train_months:
        if "probe" in args.roles:
            todo.append((ym, TRAIN_ANCHORS_PER_DAY))
        if "eval" in args.roles:
            todo.append((next_month(ym), 8))
    seen, uniq = set(), []
    for t in todo:
        if t not in seen:
            seen.add(t); uniq.append(t)

    geom = dict(norm_groups=None, fixed_agg=None, seq_len=2048)

    print(f"{len(uniq)} (month, anchors) panels -> {root}")

    if args.jobs > 1 and not args.dry_run:
        return _build_parallel(uniq, root, stats_dir, geom, machine, args)

    built = skipped = 0
    total_gb = 0.0
    for ym, n_anc in uniq:
        key = pc.panel_key(anchors_per_day=n_anc, stats_tag=stats_dir.name,
                           has_rf=False, norm_groups=None, fixed_agg=None,
                           seq_len=2048)
        if pc.is_built(root, ym, key) and not args.overwrite:
            import json
            n = json.loads((pc.panel_dir(root, ym, key) / "meta.json").read_text())["n_rows"]
            gb = n * 2048 * 9 * 2 / (1 << 30)
            total_gb += gb
            print(f"  {ym} a{n_anc:<2d} already built ({n:,} rows, {gb:.1f} GB)")
            skipped += 1
            continue
        if args.dry_run:
            print(f"  {ym} a{n_anc:<2d} WOULD BUILD")
            continue
        y, m = ym.split("-")
        month_dir = Path(machine.mosaic_dir) / y / m
        if not month_dir.is_dir():
            print(f"  {ym} a{n_anc:<2d} SKIP — no mosaic at {month_dir}")
            continue
        stats = AnchorStats(stats_dir / f"{ym}.npz")
        it = iter_month_panel(
            month_dir, stats, schedule, day_anchors(n_anc), args.batch_size,
            view_spec=ViewSpec(sequence_length=geom["seq_len"],
                               aggregation_seconds=(6, 11)),
        )
        d = pc.build(root, ym, key, it, overwrite=args.overwrite)
        import json
        n = json.loads((d / "meta.json").read_text())["n_rows"]
        gb = n * 2048 * 9 * 2 / (1 << 30)
        total_gb += gb
        print(f"  {ym} a{n_anc:<2d} built {n:,} rows, {gb:.1f} GB -> {d}", flush=True)
        built += 1
    print(f"\nbuilt {built}, already present {skipped}, total {total_gb:.0f} GB")
    return 0


def _build_one(job):
    """One panel in its own process. Returns (ym, anchors, rows, err)."""
    import os, json
    ym, n_anc, root, stats_dir, geom, mosaic_dir, holiday_csv, overwrite = job
    try:
        # Per-worker TMPDIR: open_shard stages zstd shards through it, and
        # concurrent builds must not share one scratch directory.
        tmp = Path(os.environ.get(
            "TMPDIR", "lab/market-jepa-checkpoints/_scratch/panel_tmp"))
        tmp = tmp / f"w{os.getpid()}"
        tmp.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(tmp)
        # SETTING THE ENV VAR IS NOT ENOUGH. tempfile caches the resolved
        # directory in the module global `tempfile.tempdir` on first use, and
        # a forked worker inherits the PARENT's cached value -- so workers
        # kept staging zstd shards into the data host's nearly full /tmp and died
        # with ENOSPC while the large scratch volume sat empty. Pin the global.
        import tempfile
        tempfile.tempdir = str(tmp)

        import panel_cache as pc
        from stable_finance.dataset import MarketSchedule, ViewSpec, iter_month_panel
        from stable_finance.dataset import AnchorTargetStats as AnchorStats
        from xs_ic_eval import day_anchors

        key = pc.panel_key(anchors_per_day=n_anc, stats_tag=Path(stats_dir).name,
                           has_rf=False, norm_groups=None, fixed_agg=None,
                           seq_len=2048)
        if pc.is_built(Path(root), ym, key) and not overwrite:
            d = pc.panel_dir(Path(root), ym, key)
            return (ym, n_anc, json.loads((d / "meta.json").read_text())["n_rows"], None)
        md = Path(mosaic_dir) / ym[:4] / ym[5:7]
        if not md.is_dir():
            return (ym, n_anc, 0, f"no mosaic at {md}")
        sch = MarketSchedule(holiday_csv)
        stats = AnchorStats(Path(stats_dir) / f"{ym}.npz")
        it = iter_month_panel(
            md, stats, sch, day_anchors(n_anc), 256,
            view_spec=ViewSpec(sequence_length=geom["seq_len"],
                               aggregation_seconds=(6, 11)),
        )
        d = pc.build(Path(root), ym, key, it, overwrite=overwrite)
        return (ym, n_anc, json.loads((d / "meta.json").read_text())["n_rows"], None)
    except pc.Claimed:
        # Another builder holds it. Not an error and not work for us.
        return (ym, n_anc, -1, None)
    except Exception as e:  # one bad month must not take the whole build down
        return (ym, n_anc, 0, f"{type(e).__name__}: {e}")


def _build_parallel(uniq, root, stats_dir, geom, machine, args):
    """Fan the panel list across processes, LONGEST FIRST.

    A 36-anchor probe panel is ~5.5x an 8-anchor eval panel, so scheduling the
    big ones first keeps the tail short instead of leaving a single half-hour
    build running alone at the end.
    """
    import concurrent.futures as cf
    jobs = [(ym, n, str(root), str(stats_dir), geom, machine.mosaic_dir,
             machine.holiday_csv, args.overwrite)
            for ym, n in sorted(uniq, key=lambda t: -t[1])]
    ok = fail = 0
    total_gb = 0.0
    print(f"  {len(jobs)} panels, {args.jobs}-way parallel (longest first)", flush=True)
    with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for ym, n_anc, rows, err in ex.map(_build_one, jobs):
            if err:
                print(f"  {ym} a{n_anc:<2d} FAILED - {err}", flush=True); fail += 1
            elif rows < 0:
                print(f"  {ym} a{n_anc:<2d} claimed by another builder — skipped",
                      flush=True)
            else:
                gb = rows * 2048 * 9 * 2 / (1 << 30); total_gb += gb
                print(f"  {ym} a{n_anc:<2d} {rows:,} rows, {gb:.1f} GB", flush=True); ok += 1
    print(f"\nbuilt/present {ok}, failed {fail}, total {total_gb:.0f} GB")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
