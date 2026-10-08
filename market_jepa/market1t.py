"""Local copies of the released Market-1T data, fetched from the Hub on demand.

The default machine config (``machine=market1t``) trains from a local copy
under ``<repo>/market1t/``. Before a run reads anything, ``ensure_for_run``
downloads the months it needs, proves every file arrived, decompresses the
day store, and builds the cross-sectional target tables the reported recipe
trains with. A month already on disk costs one listing call.

By hand, e.g. everything the 2020-01 evaluation month needs::

    uv run python -m market_jepa.market1t 2019-07 2020-01
    uv run python -m market_jepa.market1t 2019-07 2020-01 --layouts dense

Only 2019-07 -> 2020-12 is released.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ORG = "fin-ai-lab"
LAYOUTS = {
    "dense": ("Market-1T-1Hz-2019H2-2020-dense", "1Hz_mosaic_mnth"),
    "sparse": ("Market-1T-1Hz-2019H2-2020-sparse", "1Hz_mosaic_mnth_sparse"),
    "daystore": ("Market-1T-1Hz-2019H2-2020-daystore", "1Hz_daystore"),
}
RELEASED = ("2019-07", "2020-12")
TARGETS = "xs_anchor_stats_fwdvwap60"
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = _REPO_ROOT / "market1t"
HOLIDAY_CSV = _REPO_ROOT / "data" / "market_holidays.csv"


def hub(fn, *args, **kw):
    """Call a Hub API function, waiting out HTTP 429 and retrying.

    THE HUB ALLOWS 1000 API REQUESTS PER 5 MINUTES PER ACCOUNT, logged in or
    not, and every file downloaded costs at least one. One evaluation month is
    ~2500 files across the three layouts, so a 429 is expected, not
    exceptional -- and once the window is spent every call fails, listings and
    whoami included, until it rolls over.
    """
    for attempt in range(1, 13):
        try:
            return fn(*args, **kw)
        except Exception as e:  # HfHubHTTPError and requests' HTTPError alike
            resp = getattr(e, "response", None)
            if getattr(resp, "status_code", None) != 429 or attempt == 12:
                raise
            wait = int(resp.headers.get("Retry-After", 0) or 0) or 300
            print(f"rate-limited by the Hub (attempt {attempt}); "
                  f"resuming in {wait} s", flush=True)
            time.sleep(wait)


def months_between(start: str, end: str) -> list[str]:
    """YYYY-MM months from start to end inclusive (dates are cut to the month)."""
    y, m = map(int, str(start)[:7].split("-"))
    ey, em = map(int, str(end)[:7].split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y, m + 1) if m < 12 else (y + 1, 1)
    return out


def _check_released(months: list[str]) -> None:
    outside = [m for m in months if not RELEASED[0] <= m <= RELEASED[1]]
    if outside:
        raise SystemExit(
            f"Market-1T is released for {RELEASED[0]} -> {RELEASED[1]} only; "
            f"this run needs {outside[0]}"
            + (f" .. {outside[-1]}" if len(outside) > 1 else "")
            + ". Pick dates inside the window (a run trains on the six months "
              "before its evaluation month).")


def ensure(layout: str, months: list[str], root: Path = DEFAULT_ROOT) -> Path:
    """Download months of one layout under root, then PROVE nothing is missing.

    A rate-limited snapshot_download can return without error with files
    missing, so the Hub listing is checked against the disk and the download
    repeated until they agree. snapshot_download skips files already on disk,
    so each round fetches only what is left.
    """
    from huggingface_hub import list_repo_files, snapshot_download

    _check_released(months)
    name, sub = LAYOUTS[layout]
    root = Path(root)
    prefixes = tuple(f"{sub}/{m.replace('-', '/')}/" for m in months)
    want = [f for f in hub(list_repo_files, f"{ORG}/{name}", repo_type="dataset")
            if f.startswith(prefixes)]
    missing = [f for f in want if not (root / f).exists()]
    for rnd in range(6):
        if not missing:
            break
        if rnd:
            print(f"{len(missing)} files still missing; retrying in 300 s", flush=True)
            time.sleep(300)
        print(f"downloading {len(missing)} of {len(want)} files from {name} "
              f"into {root}", flush=True)
        hub(snapshot_download, f"{ORG}/{name}", repo_type="dataset", local_dir=root,
            allow_patterns=[p + "*" for p in prefixes], max_workers=4)
        missing = [f for f in want if not (root / f).exists()]
    if missing:
        raise SystemExit(f"{name}: {len(missing)} files still missing, "
                         f"e.g. {missing[0]}. Rerun to resume.")
    if layout == "daystore":
        _decompress_daystore(root / sub, months)
    return root / sub


def _decompress_daystore(store: Path, months: list[str]) -> None:
    """The Hub day store ships features.npy.zst only, and the reader memmaps
    features.npy. Reading through hf:// decompresses on the way in; a local
    copy must be decompressed here or training dies in the dataloader."""
    from stable_finance.dataset.daystore import decompress_day

    days = [d for m in months for d in sorted((store / m.replace("-", "/")).iterdir())
            if (d / "meta.json").is_file() and not (d / "features.npy").is_file()]
    if days:
        print(f"decompressing {len(days)} day-store days", flush=True)
    for d in days:
        decompress_day(d, remove_zst=False)


def ensure_targets(months: list[str], root: Path = DEFAULT_ROOT) -> Path:
    """The per-month cross-sectional target tables, built from the sparse layout.

    These are the tables every released encoder trained and was scored with
    (xs_anchor_stats_fwdvwap60); a rebuild matches the lab's byte for byte.
    """
    out = Path(root) / TARGETS
    todo = [m for m in months if not (out / f"{m}.npz").exists()]
    if todo:
        sparse = ensure("sparse", todo, root)
        cmd = [sys.executable, "-m", "stable_finance.dataset.build_targets",
               "--mosaic-dir", str(sparse), "--out-dir", str(out),
               "--start", todo[0], "--end", todo[-1], "--holiday-csv", str(HOLIDAY_CSV),
               "--workers", str(max(1, (os.cpu_count() or 4) // 4))]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    return out


def ensure_for_run(cfg) -> None:
    """Fetch everything a training run on machine=market1t reads, and point the
    run at the target tables if it was not given its own."""
    from omegaconf import OmegaConf

    sel = lambda k, d=None: OmegaConf.select(cfg, k, default=d)  # noqa: E731
    root = Path(cfg.machine.mosaic_dir).parent
    train = months_between(sel("dataset.train_date_start"), sel("dataset.train_date_end"))
    evalm = months_between(sel("dataset.eval_date_start"), sel("dataset.eval_date_end"))
    dense = list(evalm)
    if str(sel("dataset.backend", "mds")) == "days":
        ensure("daystore", train, Path(cfg.machine.daystore_dir).parent)
    else:
        dense += train
    if sel("training.live_eval", True) and sel("dataset.eval_train_date_start"):
        dense += months_between(sel("dataset.eval_train_date_start"),
                                sel("dataset.eval_train_date_end"))
    ensure("dense", sorted(set(dense)), root)
    if sel("dataset.xs_anchor_stats_dir") is None and sel("machine.xs_anchor_stats_dir"):
        out = ensure_targets(sorted(set(train + evalm)), Path(cfg.machine.xs_anchor_stats_dir).parent)
        # WRITTEN BACK so train_meta records the tables the run trained with.
        cfg.dataset.xs_anchor_stats_dir = str(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("start", help="first month, YYYY-MM")
    p.add_argument("end", help="last month, YYYY-MM (inclusive)")
    p.add_argument("--layouts", nargs="+", choices=list(LAYOUTS),
                   default=["dense", "daystore"])
    p.add_argument("--no-targets", action="store_true",
                   help="skip building the target tables (and the sparse download)")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    a = p.parse_args()
    months = months_between(a.start, a.end)
    for layout in a.layouts:
        print(ensure(layout, months, a.root))
    if not a.no_targets:
        print(ensure_targets(months, a.root))


if __name__ == "__main__":
    main()
