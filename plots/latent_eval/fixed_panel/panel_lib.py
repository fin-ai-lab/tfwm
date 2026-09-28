"""t-SNE of LeJEPA vs supervised-ViT embeddings on curated cross-sector pairs.

Hypothesis we're stress-testing: LeJEPA (SSL) embeddings cluster by ticker
identity, even when two tickers in the panel are near-identical businesses
(same sub-industry); the supervised return-classifier collapses to an
undifferentiated cloud regardless of ticker. The figure is six panels
(2 models × 3 months) showing 6 hand-picked tickers across 3 sectors:

  * Semis: NVDA + AMD          (both fabless GPU/CPU semiconductor cos)
  * Banks: JPM  + BAC          (both diversified money-center banks)
  * Oil:   XOM  + CVX          (both integrated oil supermajors)

Within each sector the two companies are essentially interchangeable from
a business-model standpoint, so a separation between them in the t-SNE has
to be ticker-microstructure, not industry co-movement.

Inputs
------
  * LeJEPA      — wandb regex ``^lejepa_lamb-0\\.01_nloc-6$`` under
                  ``lejepa-lamb-nlocal-*``.
  * Supervised  — wandb regex ``^vit_return_900_k5_lr-1e-4$`` under
                  ``supervised-vit-cf3df6-*``.

Months: the last three sweep months (2022-06, 2023-03, 2023-10), evaluated
on t+1 (Jul 2022, Apr 2023, Nov 2023). With ``n_pairs_per_obs=1`` we emit
one global-view crop per (ticker, trading day), so each ticker gets ~20
windows per month — every dot in the t-SNE is a different day's worth of
1Hz data, not a re-augmentation of the same day.

Efficiency
----------
``TickerFilteredDataset`` short-circuits ``__getitem__`` for any sample
whose ticker isn't in ``CURATED_TICKERS`` — the row is replaced by a
``bucket_key=-1`` zero-sample, which ``collate_bucketed`` drops. We pay
the MDS read for every row but skip augmentation, normalization, target
computation, and the GPU forward for ~99% of the eval-month rows.

Caching
-------
``cache/curated__{series}__eval_{eval_month}.pkl`` per (model, eval-month)
— stores the (96-row) embedding matrix + ticker / return / date arrays.
t-SNE projections cache to
``cache/tsne__{series}__train_{train_month}__seed{seed}__perp{perp}.pkl``
keyed on ticker set + row count, so plot-only tweaks skip the projection.
``--refresh_emb`` drops embedding caches; ``--refresh_tsne`` drops only
the t-SNE caches.

Run
---
    uv run plots/latent_eval/fixed_panel/panel_lib.py
"""
from __future__ import annotations

import argparse
import calendar
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PLOTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PLOTS_DIR))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "eval"))
sys.path.insert(0, str(_REPO_ROOT))

# Reuse the canonical backbone loader so any change to the on-disk checkpoint
# formats flows through automatically — no parallel copy to drift. It used to
# live in the retired mass_eval_world_model.py workflow, which the
# delta-AUC purge deleted; it now lives beside the rest of the checkpoint
# loading, in market_jepa/eval/checkpoints.py.
from market_jepa.eval.checkpoints import (  # noqa: E402
    GLOBAL_SEQ_LEN, N_FEATURES, load_backbone as _load_backbone,
    parse_project_dates as _parse_project_dates,
)

# LATENT EVALS READ THE MEAN, whatever readout the checkpoint trained with.
# The supervised arm trains pool="last"; reading its latent there would make
# every structure number a statement about ONE patch of the day, and would
# confound every supervised-vs-SSL comparison, since LeJEPA is mean-pooled by
# construction. Pooling is a readout applied after the last block, so no
# weight depends on it. See market_jepa.eval.checkpoints.load_backbone.
LATENT_POOL = "mean"
from stable_finance.dataset import MarketSchedule  # noqa: E402
from market_jepa.schemas import BLL01MachineConfig  # noqa: E402
from market_jepa.training.streaming_dataset import (  # noqa: E402
    StreamingMarketDataset, discover_streams,
)
from market_jepa.training.utils import collate_bucketed  # noqa: E402
from style import (  # noqa: E402
    COMPACT_RC_PARAMS, SERIES_STYLES, WIDTH_FULL, WIDTH_HALF,
    add_bottom_legend, apply_style, fetch_wandb_runs, load_sweep_months,
    save_figure,
)

apply_style()

CHECKPOINT_ROOT = Path("/data/lab/market-jepa-checkpoints")
OUT_DIR = Path(__file__).resolve().parent
# Panel windows, cached per (batch, month). NOT small: an information-bearing
# panel is ~2.2x the 9-channel one, and 32 months x 3 batches x both kinds ran
# /localhome (a 525G volume shared with the whole lab) to ZERO bytes free on
# 2026-08-29, which took git down with it. The default now lives beside the
# other big caches on /data; MJ_PANEL_CACHE_DIR overrides it.
CACHE_DIR = Path(os.environ.get(
    "MJ_PANEL_CACHE_DIR",
    "/data/lab/market-jepa-checkpoints/_scratch/latent_eval/fixed_panel_cache"))
API_CACHE = _PLOTS_DIR / "metrics" / "api_cache.json"

BATCH_SIZE = 1024
# One global view per (ticker, date). N=1 means each dot in the t-SNE is
# a distinct trading day rather than one of 4 augmentations of the same
# day — the model sees 19–21 genuinely different windows per ticker per
# month, matching the inference distribution and making the per-ticker
# clusters defensible (no augmentation-induced sub-clusters).
N_PAIRS_PER_OBS = 1
RETURN_BUCKETS = ("down", "flat", "up")  # 33/66 percentiles, eval-window terciles
RETURN_MARKERS = {"down": "v", "flat": "o", "up": "^"}
TARGET_HORIZON = 900   # seconds — return horizon for marker labeling

# Eval window offset (in calendar months) from the training month. 1 = t+1:
# the calendar month immediately after training, which is the smallest
# out-of-sample window we can pick without straddling the training data.
DEFAULT_FWD_MONTHS = 1

# Last three sweep months — picked because the late-2022 / 2023 universe
# contains the obvious cross-sector mega-cap pairs (NVDA/AMD, JPM/BAC,
# XOM/CVX) at high daily presence. Hardcoded rather than derived from
# ``load_sweep_months()[-3:]`` so the figure pins to a specific story.
TRAIN_MONTHS = ["2022-06", "2023-03", "2023-10"]

# Variant catalog: each entry is a (sector_name, [(ticker, hex_color), ...])
# list. Multi-sector variants use sister shades (dark + light from
# tab10/tab20) per pair so sector grouping reads off the color alone.
# Single-sector variants give each of 6 tickers a distinct hue (tab10 C0–C5)
# so the question "does the model separate within-sector tickers?" can be
# answered visually.
_TAB10_6 = (
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
)

# Display name shown alongside the ticker in the legend. Common short forms
# preferred over legal entity names so the legend stays compact in a 3-col
# layout (e.g. "JPMorgan", not "JPMorgan Chase & Co.").
TICKER_NAMES: dict[str, str] = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOG": "Alphabet",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "AMD":  "Adv. Micro Devices",
    "INTC": "Intel",
    "JPM":  "JPMorgan",
    "BAC":  "Bank of America",
    "WFC":  "Wells Fargo",
    "C":    "Citigroup",
    "GS":   "Goldman Sachs",
    "MS":   "Morgan Stanley",
    "XOM":  "ExxonMobil",
    "CVX":  "Chevron",
    "COP":  "ConocoPhillips",
    "EOG":  "EOG Resources",
    "SLB":  "Schlumberger",
    "HAL":  "Halliburton",
}
VARIANTS: dict[str, list[tuple[str, list[tuple[str, str]]]]] = {
    # Single-sector: 6 oil & gas names spanning supermajors (XOM, CVX),
    # E&Ps (COP, EOG), and oilfield services (SLB, HAL).
    "oil": [
        ("Oil", [
            ("XOM", _TAB10_6[0]), ("CVX", _TAB10_6[1]),
            ("COP", _TAB10_6[2]), ("EOG", _TAB10_6[3]),
            ("SLB", _TAB10_6[4]), ("HAL", _TAB10_6[5]),
        ]),
    ],
    # Single-sector: 6 mega-cap tech names — diversified mega-caps
    # (AAPL, MSFT, GOOG, AMZN) plus the GPU/CPU duopoly (NVDA, AMD).
    "tech": [
        ("Tech", [
            ("AAPL", _TAB10_6[0]), ("MSFT", _TAB10_6[1]),
            ("GOOG", _TAB10_6[2]), ("AMZN", _TAB10_6[3]),
            ("NVDA", _TAB10_6[4]), ("AMD",  _TAB10_6[5]),
        ]),
    ],
    # Single-sector: 6 large-cap U.S. banks — money-center (JPM, BAC, WFC,
    # C) and pure-play investment banks (GS, MS).
    "finance": [
        ("Finance", [
            ("JPM", _TAB10_6[0]), ("BAC", _TAB10_6[1]),
            ("WFC", _TAB10_6[2]), ("C",   _TAB10_6[3]),
            ("GS",  _TAB10_6[4]), ("MS",  _TAB10_6[5]),
        ]),
    ],
    # Two of each sector, reusing the "main" dark/light pair color scheme so
    # sector identity reads off color (NVDA/AMD blue, JPM/BAC red, XOM/CVX
    # green) and the panel pairs the single-sector "tech"/"finance"/"oil"
    # plots above.
    "mixed": [
        ("Tech",    [("NVDA", "#1f77b4"), ("AMD", "#aec7e8")]),
        ("Finance", [("JPM",  "#d62728"), ("BAC", "#ff9896")]),
        ("Oil",     [("XOM",  "#2ca02c"), ("CVX", "#98df8a")]),
    ],
}

# Cap = "take every trading day in the month" since N_PAIRS=1. Set above
# the longest possible month so we never trim — collect_curated_windows
# just exits when the loader exhausts (typically 19–21 days/ticker).
SAMPLES_PER_TICKER = 25

# Mutable globals — populated by ``apply_variant`` (called from main()).
_DEFAULT_VARIANT = "mixed"
ACTIVE_VARIANT: str = _DEFAULT_VARIANT
CURATED_INDUSTRIES: list[tuple[str, list[tuple[str, str]]]] = VARIANTS[_DEFAULT_VARIANT]
TICKER_COLOR: dict[str, str] = {}
TICKER_SECTOR: dict[str, str] = {}
ALL_CURATED_TICKERS: list[str] = []


def apply_variant(name: str) -> None:
    """Switch the module-level curated-ticker globals to ``VARIANTS[name]``.

    Called once from ``main()`` after argparse so the rest of the module
    (cache paths, plot helpers) reads from a single source of truth.
    """
    global ACTIVE_VARIANT, CURATED_INDUSTRIES
    global TICKER_COLOR, TICKER_SECTOR, ALL_CURATED_TICKERS
    if name not in VARIANTS:
        raise ValueError(
            f"unknown variant {name!r}; available: {sorted(VARIANTS)}"
        )
    ACTIVE_VARIANT = name
    CURATED_INDUSTRIES = VARIANTS[name]
    TICKER_COLOR = {
        t: c for _, pairs in CURATED_INDUSTRIES for t, c in pairs
    }
    TICKER_SECTOR = {
        t: sector for sector, pairs in CURATED_INDUSTRIES for t, _ in pairs
    }
    ALL_CURATED_TICKERS = list(TICKER_COLOR)


# Module-level apply so tests/smoke imports see a populated set; main()
# overwrites this after argparse.
apply_variant(_DEFAULT_VARIANT)


def _variant_suffix() -> str:
    """``"_<variant>"`` so each variant's caches and output PNG/PDF land in
    their own files (no two variants share the same output path)."""
    return f"_{ACTIVE_VARIANT}"


class TickerFilteredDataset(StreamingMarketDataset):
    """Ticker-filtered streaming dataset.

    Returns ``_zero_sample()`` (with ``bucket_key=-1``) for any sample whose
    ticker isn't in ``target_tickers``. ``collate_bucketed`` drops the -1
    bucket, so non-target rows incur the MDS read but skip augmentation,
    normalization, target computation, and the GPU forward — only the
    handful of curated tickers reach the model.
    """

    def __init__(self, *args, target_tickers=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_tickers = (
            set(target_tickers) if target_tickers is not None else None
        )

    def _getitem_numpy(self, idx, sample):
        if (
            self._target_tickers is not None
            and sample.get("ticker", "") not in self._target_tickers
        ):
            return [self._zero_sample()]
        return super()._getitem_numpy(idx, sample)


def eval_window_t_plus_n(train_month: str, n: int) -> tuple[str, str, str]:
    """Returns ``(eval_month, eval_start, eval_end)`` for ``train_month + n``.

    Pure month math: ``train_month='2008-02'``, ``n=12`` →
    ``('2009-02', '2009-02-01', '2009-02-28')``. Days are pinned to the
    first / last of the eval calendar month, matching the streaming-dataset
    month-aligned convention.
    """
    y, mo = map(int, train_month.split("-"))
    total = y * 12 + (mo - 1) + n
    ny, nm = total // 12, total % 12 + 1
    last = calendar.monthrange(ny, nm)[1]
    return (
        f"{ny:04d}-{nm:02d}",
        f"{ny:04d}-{nm:02d}-01",
        f"{ny:04d}-{nm:02d}-{last:02d}",
    )

# 2 series × 3 months, hardcoded to mirror plots/metrics/metrics.py.
SERIES = [
    {
        "key": "lejepa",
        "label": "LeJEPA",
        "color_for_panel_title": SERIES_STYLES["lejepa"]["color"],
        "project_prefix": "lejepa-lamb-nlocal-cf3df6-",
        "run_name_re": re.compile(r"^lejepa_lamb-0\.01_nloc-6$"),
    },
    {
        "key": "supervised",
        "label": "Supervised",
        "color_for_panel_title": SERIES_STYLES["supervised_return"]["color"],
        "project_prefix": "supervised-vit-cf3df6-",
        "run_name_re": re.compile(r"^vit_return_900_k5_lr-1e-4$"),
    },
]


@dataclass(frozen=True)
class CkptInfo:
    series_key: str
    project: str
    run_id: str
    run_name: str
    train_month: str
    train_start: str
    train_end: str
    eval_month: str          # 'YYYY-MM' — train_month + 1 (calendar wrap)
    eval_start: str          # 'YYYY-MM-DD'
    eval_end: str            # 'YYYY-MM-DD'
    ckpt_dir: Path


def load_series_encoder(skey: str, ev_month: str, device):
    """One loader for every trained-encoder key in the registry.

    Glob-resolved keys pair an eval month with the encoder trained the month
    BEFORE it (the IC-sweep pairing; the fixed-panel MONTHS list is training
    months for the same reason). Manifest keys are stored per eval month
    already. Both go through the canonical loaders, so a checkpoint-format
    change lands here without a parallel copy to drift.

    Returns None when no checkpoint exists for the month — the caller skips,
    matching fixed_panel_metrics' behavior for holes in a series.

    Lives here, not in build_fullday_embs, because every consumer of
    MODEL_ORDER needs it: since the retirement of the pre-IC glob keys
    (2026-09-04) the default model set is manifest-resolved, and a script
    that reaches into ``spec["project_glob"]`` itself now raises KeyError on
    its own defaults.
    """
    # Function-local: fixed_panel_metrics imports this module, so a top-level
    # import would close the cycle.
    from fixed_panel_metrics import _load_manifest_encoder, _manifest_row
    from industry_nn_sweep import MODEL_SPECS, month_dates_suffix, resolve_run
    from market_jepa.eval.checkpoints import prev_month

    spec = MODEL_SPECS[skey]
    if "tsfm_family" in spec or "sup_family" in spec or skey == "random":
        raise SystemExit(
            f"{skey}: not a --series key — use --families / --sup-families / "
            "--randvit-seeds for it")
    if "manifest_series" in spec:
        row = _manifest_row(spec["manifest_series"], ev_month)
        if row is None:
            return None
        return _load_manifest_encoder(row).to(device).eval()
    pglob = spec["project_glob"]
    run_dir = resolve_run(
        pglob.format(dates=month_dates_suffix(prev_month(ev_month))
                     if "{dates}" in pglob else ""),
        spec["run_name"])
    if run_dir is None:
        return None
    return _load_backbone(
        run_dir, run_dir, run_dir.parent.name,
        pool=LATENT_POOL).to(device).eval()


def resolve_ckpts(
    months: list[str], fwd_months: int,
) -> dict[tuple[str, str], CkptInfo]:
    """{(series_key, train_month) -> CkptInfo} for the 2x3 grid.

    fetch_wandb_runs already de-dupes within a project, so for a series
    that only has one cell per month we get exactly one CkptInfo per
    (series, month). Months without a matching ckpt are dropped silently
    — the panel just won't render.
    """
    out: dict[tuple[str, str], CkptInfo] = {}
    for s in SERIES:
        runs = fetch_wandb_runs(
            s["project_prefix"],
            months=months,
            run_name_re=s["run_name_re"],
            apply_baseline=False,
            cache_path=API_CACHE,
            verbose=True,
        )
        for r in runs:
            if r.train_month not in months:
                continue
            d = CHECKPOINT_ROOT / r.project / r.run_id
            if not d.exists():
                print(f"  WARN: wandb has {r.project}/{r.run_id} but no local dir at {d}")
                continue
            key = (s["key"], r.train_month)
            if key in out:
                # Two runs in the same train-month — keep the first; sweep
                # re-runs occasionally produce duplicates and either is fine
                # for a qualitative t-SNE.
                continue
            train_end = r.train_end or ""
            if not train_end or not r.train_month:
                print(f"  WARN: {r.project}/{r.run_id} missing train metadata; skipping")
                continue
            ev_month, ev_start, ev_end = eval_window_t_plus_n(r.train_month, fwd_months)
            out[key] = CkptInfo(
                series_key=s["key"], project=r.project, run_id=r.run_id,
                run_name=r.run_name, train_month=r.train_month,
                train_start=r.train_start or "", train_end=train_end,
                eval_month=ev_month, eval_start=ev_start, eval_end=ev_end,
                ckpt_dir=d,
            )
        n = sum(1 for k in out if k[0] == s["key"])
        print(f"[{s['key']}] resolved {n}/{len(months)} months")
    return out


def build_dataset(
    date_start: str, date_end: str, batch_size: int, seed: int,
    machine: BLL01MachineConfig, schedule: MarketSchedule,
    target_tickers: list[str] | None = None,
    info: bool = False,
):
    """Build the eval-window dataset; pass ``target_tickers`` to short-circuit
    augmentation for non-target rows (see ``TickerFilteredDataset``).

    ``info`` appends the INFORMATION-TOKEN channels -- per-norm-group mean/std
    plus the three window descriptors -- as trailing constant columns, exactly
    as training does. A backbone trained WITH the token strips them into one
    token and its patch embedding never sees them; a backbone trained without
    it must not be handed them at all. Default False keeps every cached panel
    collected before 2026-08-29 valid.
    """
    streams = discover_streams(machine.mosaic_dir, date_start, date_end)
    return TickerFilteredDataset(
        augmentations=[{
            "name": "random_resized_crop",
            "n_global_views": 1,
            "n_local_views": 0,
            "global_seq_len": GLOBAL_SEQ_LEN,
            "global_scale_range": [0.5, 1.0],
        }],
        date_start=date_start,
        date_end=date_end,
        seed=seed,
        n_pairs_per_obs=N_PAIRS_PER_OBS,
        targets={"horizons": [TARGET_HORIZON], "types": ["return"]},
        schedule=schedule,
        streams=streams,
        shuffle=False,
        batch_size=batch_size,
        allow_unsafe_types=True,
        predownload=max(batch_size, 64),
        risk_factor_dir=machine.risk_factor_dir,
        risk_factor_tickers=[],
        risk_factor_columns=None,
        target_tickers=target_tickers,
        info_norm_stats=info,
        info_window=info,
    )


def collect_curated_windows(
    date_start: str, date_end: str,
    target_tickers: list[str], samples_per_ticker: int,
    machine: BLL01MachineConfig, schedule: MarketSchedule,
    seed: int = 42, info: bool = False,
) -> list[dict]:
    """Iterate the eval-window dataset (filtered to ``target_tickers``) and
    keep at most ``samples_per_ticker`` rows per ticker, in the same dict
    shape ``forward_cached`` expects.

    Stops as soon as every target ticker has hit the cap — for our 6 mega-
    caps that all trade every day, this typically means we walk only a few
    days into the month before exiting.
    """
    target_set = set(target_tickers)
    ds = build_dataset(
        date_start, date_end, BATCH_SIZE, seed, machine, schedule,
        target_tickers=target_tickers, info=info,
    )
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=False, drop_last=False, collate_fn=collate_bucketed,
    )

    counts: dict[str, int] = {t: 0 for t in target_tickers}
    cached: list[dict] = []
    for batch in loader:
        kept_buckets = []
        for bucket in batch["buckets"]:
            if not bucket["views"]:
                continue
            tickers = list(bucket.get("tickers", []))
            keep_idx: list[int] = []
            for i, t in enumerate(tickers):
                if t in target_set and counts[t] < samples_per_ticker:
                    keep_idx.append(i)
                    counts[t] += 1
            if not keep_idx:
                continue
            sel = torch.tensor(keep_idx, dtype=torch.long)
            new_bucket = {
                "views": [v.index_select(0, sel) for v in bucket["views"]],
                "lengths": [
                    (l.index_select(0, sel) if l is not None else None)
                    for l in bucket["lengths"]
                ],
                "tickers": [tickers[i] for i in keep_idx],
                "dates": [
                    bucket.get("dates", [""] * len(tickers))[i]
                    for i in keep_idx
                ],
            }
            if "targets" in bucket:
                new_bucket["targets"] = bucket["targets"].index_select(0, sel)
            kept_buckets.append(new_bucket)
        if kept_buckets:
            cached.append({"buckets": kept_buckets})
        if all(c >= samples_per_ticker for c in counts.values()):
            break
    return cached, counts


def _match_width(x, backbone):
    """Trim a panel window to the channel count ``backbone`` was built for.

    The information-token channels are TRAILING constant columns, so an
    info-less backbone reads exactly the same data channels out of an
    info-bearing panel by dropping the tail. That lets ONE collected panel
    serve both kinds of encoder, which is what keeps "fixed panel" true across
    models that disagree about the token. Widening is impossible, so a panel
    that is too narrow stays a loud error.
    """
    want = getattr(backbone, "n_features", None)
    if want is None or x.shape[1] == want:
        return x
    if x.shape[1] < want:
        raise ValueError(
            f"panel window has {x.shape[1]} channels but the backbone wants "
            f"{want} -- collect the panel with info=True")
    return x[:, :want]


def forward_cached(
    backbone, cached_batches: list[dict], device, cap: int,
) -> dict[str, np.ndarray]:
    """Forward `cached_batches` through `backbone`; return aligned arrays.

    Returns a dict with X (N,d) embeddings, tickers (N,) object array,
    return_900 (N,) float (NaN if missing), and dates (N,) object.
    """
    Xs, tks, rets, dates = [], [], [], []
    n = 0
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for batch in cached_batches:
            for bucket in batch["buckets"]:
                views = bucket["views"]
                lengths = bucket["lengths"]
                if not views:
                    continue
                x = views[0].to(device, non_blocking=True)
                x = _match_width(x, backbone)
                lens = lengths[0].to(device, non_blocking=True) if lengths[0] is not None else None
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        emb = backbone(x, lens)
                else:
                    emb = backbone(x, lens)
                Xs.append(emb.float().cpu().numpy())
                tks.append(list(bucket.get("tickers", [""] * x.shape[0])))
                if "targets" in bucket:
                    # Single-column targets dataset: returns the only return_900
                    # column; flatten to 1-D so it lines up with X rows.
                    rets.append(bucket["targets"][:, 0].cpu().numpy())
                else:
                    rets.append(np.full(x.shape[0], np.nan, np.float32))
                dates.append(list(bucket.get("dates", [""] * x.shape[0])))
                n += emb.shape[0]
            if n >= cap:
                break

    if not Xs:
        return {
            "X": np.empty((0, 0), np.float32),
            "tickers": np.empty(0, dtype=object),
            "return_900": np.empty(0, np.float32),
            "dates": np.empty(0, dtype=object),
        }
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    tk = np.asarray([t for chunk in tks for t in chunk], dtype=object)
    rt = np.concatenate(rets, axis=0).astype(np.float32)
    dt = np.asarray([d for chunk in dates for d in chunk], dtype=object)
    if len(X) > cap:
        X, tk, rt, dt = X[:cap], tk[:cap], rt[:cap], dt[:cap]
    return {"X": X, "tickers": tk, "return_900": rt, "dates": dt}


def forward_cached_multi(
    model, cached_batches: list[dict], device, cap: int, layers: list[int],
) -> dict:
    """``forward_cached`` for a frozen TSFM, at every depth from ONE pass.

    Same rows, same order, same autocast as ``forward_cached`` — only the
    readout differs: ``PretrainedTSFM.compute_features_multi`` hangs capture
    hooks on the shallower blocks, so a 21-layer sweep costs one forward
    instead of 21. Returns ``X`` as ``{layer: (N, d)}``; everything else is
    shared across layers and is returned once.
    """
    Xs = {L: [] for L in layers}
    tks, rets, dates = [], [], []
    n = 0
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for batch in cached_batches:
            for bucket in batch["buckets"]:
                views, lengths = bucket["views"], bucket["lengths"]
                if not views:
                    continue
                x = views[0].to(device, non_blocking=True)
                lens = (lengths[0].to(device, non_blocking=True)
                        if lengths[0] is not None else None)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=use_amp):
                    feats = model.compute_features_multi(x, lens, layers)
                for L in layers:
                    Xs[L].append(feats[L].float().cpu().numpy())
                tks.append(list(bucket.get("tickers", [""] * x.shape[0])))
                if "targets" in bucket:
                    rets.append(bucket["targets"][:, 0].cpu().numpy())
                else:
                    rets.append(np.full(x.shape[0], np.nan, np.float32))
                dates.append(list(bucket.get("dates", [""] * x.shape[0])))
                n += x.shape[0]
            if n >= cap:
                break

    if not tks:
        return {"X": {L: np.empty((0, 0), np.float32) for L in layers},
                "tickers": np.empty(0, dtype=object),
                "return_900": np.empty(0, np.float32),
                "dates": np.empty(0, dtype=object)}
    tk = np.asarray([t for chunk in tks for t in chunk], dtype=object)
    rt = np.concatenate(rets, axis=0).astype(np.float32)
    dt = np.asarray([d for chunk in dates for d in chunk], dtype=object)
    X = {L: np.concatenate(v, axis=0).astype(np.float32)[:cap] for L, v in Xs.items()}
    return {"X": X, "tickers": tk[:cap], "return_900": rt[:cap], "dates": dt[:cap]}


def _filter_rows(batches: list[dict], keep_fn) -> list[dict]:
    """Subselect rows of collected batches by ``keep_fn(ticker, date)``."""
    out = []
    for batch in batches:
        kept = []
        for bucket in batch["buckets"]:
            tickers = list(bucket.get("tickers", []))
            dates = list(bucket.get("dates", []))
            idx = [i for i, (t, d) in enumerate(zip(tickers, dates)) if keep_fn(t, d)]
            if not idx:
                continue
            sel = torch.tensor(idx, dtype=torch.long)
            nb = {
                "views": [v.index_select(0, sel) for v in bucket["views"]],
                "lengths": [
                    (l.index_select(0, sel) if l is not None else None)
                    for l in bucket["lengths"]
                ],
                "tickers": [tickers[i] for i in idx],
                "dates": [dates[i] for i in idx],
            }
            if "targets" in bucket:
                nb["targets"] = bucket["targets"].index_select(0, sel)
            kept.append(nb)
        if kept:
            out.append({"buckets": kept})
    return out


def emb_cache_path(series_key: str, eval_month: str) -> Path:
    """``curated[_variant]__`` prefix keeps each variant's caches separate.
    The default ``main`` variant uses ``curated__…`` so existing files stay
    valid across edits; alternates land at ``curated_<variant>__…``."""
    return CACHE_DIR / f"curated{_variant_suffix()}__{series_key}__eval_{eval_month}.pkl"


def compute_or_load_embeddings(
    ckpt: CkptInfo, *, device: torch.device, refresh: bool,
    machine: BLL01MachineConfig, schedule: MarketSchedule,
    cached_batches: list[dict] | None = None,
) -> tuple[dict, list[dict] | None]:
    """Load cached curated embeddings for (series, eval_month) or compute them.

    ``cached_batches`` (optional): the filtered eval-window batches for this
    train month — both series in a column see the same 6 tickers × 16 rows,
    so we collect once per month and reuse for the second series.
    """
    cache_path = emb_cache_path(ckpt.series_key, ckpt.eval_month)
    if cache_path.exists() and not refresh:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        cached_set = set(map(str, data.get("tickers", [])))
        if cached_set >= set(ALL_CURATED_TICKERS) and len(data["X"]) > 0:
            print(f"  cache hit: {cache_path.name} ({len(data['X'])} samples)")
            return data, cached_batches
        print(f"  cache miss (ticker set changed): {cache_path.name}")

    print(f"  computing {ckpt.series_key} / eval={ckpt.eval_month} → {cache_path.name}")
    if cached_batches is None:
        t0 = time.time()
        cached_batches, counts = collect_curated_windows(
            ckpt.eval_start, ckpt.eval_end,
            target_tickers=ALL_CURATED_TICKERS,
            samples_per_ticker=SAMPLES_PER_TICKER,
            machine=machine, schedule=schedule, seed=42,
        )
        print(f"    curated windows collected in {time.time() - t0:.1f}s "
              f"(per-ticker counts: {counts})")

    backbone = _load_backbone(ckpt.ckpt_dir, ckpt.ckpt_dir, ckpt.project,
                              pool=LATENT_POOL)
    backbone = backbone.to(device).eval()
    cap = len(ALL_CURATED_TICKERS) * SAMPLES_PER_TICKER
    try:
        result = forward_cached(backbone, cached_batches, device, cap)
    finally:
        backbone.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result["meta"] = {
        "series_key": ckpt.series_key,
        "train_month": ckpt.train_month,
        "train_start": ckpt.train_start,
        "train_end": ckpt.train_end,
        "eval_month": ckpt.eval_month,
        "eval_start": ckpt.eval_start,
        "eval_end": ckpt.eval_end,
        "project": ckpt.project,
        "run_id": ckpt.run_id,
        "n_features": N_FEATURES,
        "global_seq_len": GLOBAL_SEQ_LEN,
        "samples_per_ticker": SAMPLES_PER_TICKER,
        "curated_tickers": list(ALL_CURATED_TICKERS),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"    wrote {cache_path.name} ({len(result['X'])} samples)")
    return result, cached_batches


def bucketize_returns(rets: np.ndarray, train_rets: np.ndarray | None = None) -> np.ndarray:
    """Tercile-bucket returns into ``RETURN_BUCKETS`` (down/flat/up).

    Edges fit on ``train_rets`` (defaults to ``rets`` itself when None) so
    every panel uses the same bucket definition for the points it shows.
    Out indices: 0=down, 1=flat, 2=up. NaN inputs become -1 (filtered out).
    """
    if train_rets is None:
        train_rets = rets
    valid = train_rets[~np.isnan(train_rets)]
    if valid.size < 6:
        # Too few points to fit terciles — collapse everything into "flat".
        out = np.full(len(rets), 1, dtype=np.int64)
        out[np.isnan(rets)] = -1
        return out
    q33, q66 = np.quantile(valid, [1 / 3, 2 / 3])
    out = np.full(len(rets), -1, dtype=np.int64)
    valid_mask = ~np.isnan(rets)
    out[valid_mask & (rets <= q33)] = 0
    out[valid_mask & (rets > q33) & (rets <= q66)] = 1
    out[valid_mask & (rets > q66)] = 2
    return out


def run_tsne(X: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    """t-SNE with sane defaults; returns (N,2) float."""
    if len(X) < 5:
        return np.zeros((len(X), 2), dtype=np.float32)
    perp = float(min(perplexity, max(5, (len(X) - 1) / 3)))
    tsne = TSNE(
        n_components=2, perplexity=perp, learning_rate="auto",
        init="pca", random_state=seed, max_iter=1000, metric="euclidean",
    )
    return tsne.fit_transform(X).astype(np.float32)


def tsne_cache_path(
    series_key: str, train_month: str, seed: int, perplexity: float,
) -> Path:
    """Per-panel t-SNE cache file. Same (series, train_month, seed,
    perplexity) → same Z, so plot-only tweaks (colors, legend, layout)
    skip the t-SNE solve."""
    return CACHE_DIR / (
        f"tsne{_variant_suffix()}__{series_key}__train_{train_month}"
        f"__seed{seed}__perp{perplexity:g}.pkl"
    )


def compute_or_load_tsne(
    X: np.ndarray, keep_tickers: list[str], *,
    series_key: str, train_month: str,
    perplexity: float, seed: int, refresh: bool,
) -> np.ndarray:
    """Cache-aware wrapper around ``run_tsne``.

    Cached entry stores Z plus the ``keep_tickers`` list and row count it
    was computed on; a mismatch (e.g. someone changed N_TICKERS_PLOT or
    MIN_SAMPLES_PER_TICKER) invalidates the cache automatically.
    """
    cache_path = tsne_cache_path(series_key, train_month, seed, perplexity)
    if cache_path.exists() and not refresh:
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            data = None
        if (
            data is not None
            and data.get("keep_tickers") == list(keep_tickers)
            and data.get("n_samples") == len(X)
            and data.get("Z") is not None
        ):
            print(f"  tsne cache hit: {cache_path.name}")
            return data["Z"]
    Z = run_tsne(X, perplexity=perplexity, seed=seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({
            "Z": Z,
            "keep_tickers": list(keep_tickers),
            "n_samples": len(X),
            "perplexity": float(perplexity),
            "seed": int(seed),
        }, f)
    print(f"  tsne wrote: {cache_path.name}")
    return Z


def _draw_panel(
    ax, cell, *,
    perplexity: float, seed: int, refresh_tsne: bool,
    train_month: str, series_key: str,
) -> bool:
    """Draw one t-SNE scatter into ``ax``; return False if cell is empty / too few rows.

    Caller owns titles and axis labels. Used by both ``plot_grid`` (6-panel)
    and ``plot_select`` (single-month half-width) so the two figures stay
    visually consistent.
    """
    ax.set_xticks([]); ax.set_yticks([])
    if cell is None or len(cell["X"]) < 5:
        return False
    tickers = np.asarray([str(t) for t in cell["tickers"]], dtype=object)
    X = cell["X"]
    buckets = bucketize_returns(cell["return_900"])
    Z = compute_or_load_tsne(
        X, ALL_CURATED_TICKERS,
        series_key=series_key, train_month=train_month,
        perplexity=perplexity, seed=seed, refresh=refresh_tsne,
    )
    for b_idx, b_name in enumerate(RETURN_BUCKETS):
        marker = RETURN_MARKERS[b_name]
        b_mask = buckets == b_idx
        if not b_mask.any():
            continue
        for tname in ALL_CURATED_TICKERS:
            sel = b_mask & (tickers == tname)
            if not sel.any():
                continue
            ax.scatter(
                Z[sel, 0], Z[sel, 1],
                c=[TICKER_COLOR[tname]], marker=marker, s=18,
                edgecolors="none", alpha=0.85,
            )
    ax.set_aspect("equal", adjustable="datalim")
    return True


def _return_marker_handles() -> list:
    """3 grey return-tercile handles in Up → Flat → Down order.

    Visual order ▲ → ● → ▽ mirrors the up/flat/down semantics; the internal
    ``RETURN_BUCKETS`` ordering stays down/flat/up since that's the
    tercile-fitting convention.
    """
    return [
        plt.Line2D(
            [0], [0], marker=RETURN_MARKERS[b], linestyle="",
            markerfacecolor="0.4", markeredgecolor="none",
            markersize=6, label=f"Return: {b.capitalize()}",
        )
        for b in ("up", "flat", "down")
    ]


def plot_grid(
    embeddings: dict[tuple[str, str], dict],
    months: list[str],
    series: list[dict],
    *,
    perplexity: float = 30.0,
    seed: int = 42,
    refresh_tsne: bool = False,
) -> None:
    """6-panel curated figure (2 series × 3 train months).

    Per panel:
      * 6 fixed colors (one per ticker, shared across all columns).
      * Marker shape = realized 900s-return tercile (down/flat/up).

    ``embeddings`` is keyed by ``(series_key, train_month)``; each cell
    already contains exactly the curated tickers (via
    ``collect_curated_windows``), so no per-panel ticker filtering is
    needed here.
    """
    rows = len(series)
    cols = len(months)
    fig, axes = plt.subplots(
        rows, cols, figsize=(WIDTH_FULL, 1.4 * rows + 0.3),
        squeeze=False,
    )
    # Tick-less t-SNE panels can pack right up against each other.
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    for r, s in enumerate(series):
        for c, m in enumerate(months):
            ax = axes[r][c]
            ok = _draw_panel(
                ax, embeddings.get((s["key"], m)),
                perplexity=perplexity, seed=seed, refresh_tsne=refresh_tsne,
                train_month=m, series_key=s["key"],
            )
            if not ok:
                ax.set_title(f"{_month_title(m)}\n(missing)")
                continue
            if r == 0:
                ax.set_title(_month_title(m))
            if c == 0:
                ax.set_ylabel(s["label"], fontsize=10)

    # Bottom legend: 6 ticker handles (colored dots) + 3 return-tercile
    # markers (grey). One row, shared across the figure since the curated
    # ticker set is the same in every panel.
    # Multi-sector variants order by pair_idx-outer / sector-inner so a
    # 3-col legend reads "darker shade row, then lighter shade row" — the
    # tab10/tab20 dark/light pairs are visible by row. Single-sector
    # variants iterate flat (6 distinct hues) and drop the "(sector)"
    # suffix that would otherwise duplicate the panel context.
    n_per_sector = max(len(p) for _, p in CURATED_INDUSTRIES)

    def _ticker_label(t: str) -> str:
        return TICKER_NAMES.get(t, t)

    ticker_handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="",
            markerfacecolor=TICKER_COLOR[pairs[pair_idx][0]],
            markeredgecolor="none", markersize=6,
            label=_ticker_label(pairs[pair_idx][0]),
        )
        for pair_idx in range(n_per_sector)
        for sector_name, pairs in CURATED_INDUSTRIES
        if pair_idx < len(pairs)
    ]
    handles = ticker_handles + _return_marker_handles()
    labels = [h.get_label() for h in handles]

    # Default 3 × 3 layout: 9 handles (6 tickers + 3 return markers) in 3
    # cols × 3 rows. The reduced ``bottom_reserve`` (vs. an earlier 0.32)
    # sits the legend block right under the bottom row of axes — the
    # figure is short enough that higher reserves leave a visible gap.
    add_bottom_legend(fig, handles, labels, ncol=3, bottom_reserve=0.22)
    out = OUT_DIR / f"embedding_geometry{_variant_suffix()}"
    written = save_figure(fig, out)
    plt.close(fig)
    for p in written:
        print(f"Saved {p}")


def plot_select(
    embeddings: dict[tuple[str, str], dict],
    month: str,
    series: list[dict],
    *,
    perplexity: float = 30.0,
    seed: int = 42,
    refresh_tsne: bool = False,
    front_page: bool = False,
    ppt: bool = False,
) -> None:
    """Half-width single-month figure: 1 row × 2 cols (LeJEPA, Supervised).

    Slim companion to ``plot_grid`` for inline / slide use. Reuses the same
    ``_draw_panel`` helper so the two figures stay visually consistent.
    No legend — marker shapes / colors are explained by the full-width
    figure's legend in the paper.

    ``front_page=True`` is the abstract/front-page cut: the same two panels at
    full text width but short, with the series labels on the OUTER edges
    ("LeJEPA" left, "Supervised" right) so the label gutters are symmetric and
    the block centers on the page. Saved to
    ``embedding_geometry_<variant>_frontpage.{png,pdf}``.

    ``ppt=True`` is the slide cut: identical layout to ``front_page`` but an
    inch taller, since the paper banner is too squat for a PowerPoint slide.
    Saved to ``embedding_geometry_<variant>_ppt.{png,pdf}``.
    """
    if front_page or ppt:
        # Two panels, full text width, short — to save vertical space under
        # the abstract. The series labels sit on the OUTER edges (first panel's
        # on the left, last panel's on the right) so the label gutters balance
        # and the block centers naturally. The 0.06/0.94 margins match the axes
        # box of plots/metrics/six_task_absolute.py (same WIDTH_FULL figure);
        # ``bbox_inches=None`` preserves them so the symmetry survives. Aspect
        # is freed to "auto" so the wide-short panels fill rather than leave
        # the equal-aspect t-SNE floating in horizontal whitespace.
        fig, axes = plt.subplots(
            1, len(series), figsize=(WIDTH_FULL, 1.85 if ppt else 1.35),
            squeeze=False,
        )
        fig.subplots_adjust(wspace=0.05, left=0.06, right=0.94,
                            top=0.98, bottom=0.02)
        last = len(series) - 1
        for c, s in enumerate(series):
            ax = axes[0][c]
            ok = _draw_panel(
                ax, embeddings.get((s["key"], month)),
                perplexity=perplexity, seed=seed, refresh_tsne=refresh_tsne,
                train_month=month, series_key=s["key"],
            )
            if ok:
                ax.set_aspect("auto")
            label = s["label"] if ok else f"{s['label']}\n(missing)"
            if c == last and last > 0:
                ax.yaxis.set_label_position("right")
            ax.set_ylabel(label, fontsize=10)
        cut = "ppt" if ppt else "frontpage"
        out = OUT_DIR / f"embedding_geometry{_variant_suffix()}_{cut}"
        written = save_figure(fig, out, bbox_inches=None, pad_inches=0.0)
        plt.close(fig)
        for p in written:
            print(f"Saved {p}")
        return

    # figsize tuned so that after save_figure's bbox_inches="tight" + 0.05
    # pad, the saved PDF lands at ~253.75 x 143.32 pts — exactly matching
    # plots/action_condition/action_condition_grouped_h900_k5.pdf so the
    # two half-width panels sit at identical size in the LaTeX grid. The
    # figure is wider/taller than the action_condition source figsize
    # because this one has no y-label / ticks / legend to extend its
    # artist bbox, so tight-cropping shaves off more.
    fig, axes = plt.subplots(
        1, len(series), figsize=(4.42, 2.21),
        squeeze=False,
    )
    fig.subplots_adjust(wspace=0.05)

    for c, s in enumerate(series):
        ax = axes[0][c]
        ok = _draw_panel(
            ax, embeddings.get((s["key"], month)),
            perplexity=perplexity, seed=seed, refresh_tsne=refresh_tsne,
            train_month=month, series_key=s["key"],
        )
        title = s["label"] if ok else f"{s['label']}\n(missing)"
        ax.set_xlabel(title)

    out = OUT_DIR / "embedding_geometry_select"
    written = save_figure(fig, out)
    plt.close(fig)
    for p in written:
        print(f"Saved {p}")


def _silhouette(Z: np.ndarray, tk: np.ndarray, tickers: list[str]) -> float | None:
    """sklearn silhouette score over the ticker partition. None if degenerate.

    Reported per panel as a quick quantitative signal: higher means tickers
    form tighter, better-separated clusters in the t-SNE projection.
    Computed on Z (the 2-D embedding) rather than the raw features so it
    matches what the eye sees.
    """
    if len(Z) < 3:
        return None
    labels = np.full(len(Z), -1, dtype=np.int64)
    for i, t in enumerate(tickers):
        labels[tk == t] = i
    keep = labels >= 0
    if keep.sum() < 3 or len(np.unique(labels[keep])) < 2:
        return None
    try:
        from sklearn.metrics import silhouette_score
        return float(silhouette_score(Z[keep], labels[keep], metric="euclidean"))
    except Exception:
        return None


def _month_title(month: str) -> str:
    import calendar
    y, m = month.split("-")
    return f"{calendar.month_name[int(m)]} {y}"


def _panel_title(s: dict, m: str) -> str:
    return f"{s['label']} — {_month_title(m)}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--refresh_emb", action="store_true",
        help="Discard cached per-(model,month) embeddings and re-forward.",
    )
    p.add_argument(
        "--refresh_tsne", action="store_true",
        help="Discard cached per-panel t-SNE projections and re-fit.",
    )
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="cuda / cpu / auto")
    p.add_argument(
        "--variant", default=_DEFAULT_VARIANT, choices=sorted(VARIANTS),
        help="Curated-ticker variant. Each variant has its own cache + "
             "output files; see ``VARIANTS`` for the ticker lists.",
    )
    p.add_argument(
        "--fwd_months", type=int, default=DEFAULT_FWD_MONTHS,
        help=f"Eval window offset from train_month, in calendar months "
             f"(default {DEFAULT_FWD_MONTHS} = t+1 — calendar month right "
             f"after training).",
    )
    p.add_argument(
        "--front_page", action="store_true",
        help="Front-page cut: the same two panels as --select (tech variant, "
             "June 2022 — LeJEPA / Supervised) at full text width but short, "
             "with the series labels on the outer edges so the block is "
             "symmetric. Writes to embedding_geometry_<variant>_frontpage."
             "{png,pdf}. Forces --variant=tech; ignored data-wise by --select.",
    )
    p.add_argument(
        "--ppt", action="store_true",
        help="Slide cut: same two panels as --front_page but an inch taller "
             "so the figure isn't squat on a PowerPoint slide. Writes to "
             "embedding_geometry_<variant>_ppt.{png,pdf}. Forces "
             "--variant=tech.",
    )
    p.add_argument(
        "--select", action="store_true",
        help="Render a half-width single-month figure (tech variant, "
             "June 2022) to embedding_geometry_select.{png,pdf}. Forces "
             "--variant=tech and a single-month month list.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.select or args.front_page or args.ppt:
        # --select, --front_page, and --ppt all pin to the tech-variant June-2022
        # column (the two LeJEPA / Supervised panels); --variant is ignored.
        # --select sits half-width next to action_condition_grouped_h900_k5.png
        # in the paper grid; --front_page is the full-width short banner.
        apply_variant("tech")
        if args.select:
            apply_style(extra=COMPACT_RC_PARAMS)
        months = ["2022-06"]
    else:
        apply_variant(args.variant)
        months = list(TRAIN_MONTHS)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Plotting variant={ACTIVE_VARIANT!r}, {len(months)} train months: "
          f"{months} (eval = train + {args.fwd_months} months); "
          f"tickers: {ALL_CURATED_TICKERS}")

    ckpts = resolve_ckpts(months, fwd_months=args.fwd_months)
    print(f"Resolved {len(ckpts)} (series, month) cells")

    machine = BLL01MachineConfig()
    schedule = MarketSchedule(machine.holiday_csv)

    # Embeddings dict is keyed by (series_key, train_month) so the panel
    # title can show the training month directly. Eval-window resolution
    # (train_month + fwd_months) lives inside CkptInfo and the on-disk
    # embedding cache, but the plot grid only sees train_month.
    embeddings: dict[tuple[str, str], dict] = {}
    train_months_ordered: list[str] = []
    for m in months:
        cached_batches: list[dict] | None = None
        ckpts_for_month = [(s, ckpts.get((s["key"], m))) for s in SERIES]
        if not any(ck is not None for _, ck in ckpts_for_month):
            print(f"  skip train_month={m}: no ckpts")
            continue
        train_months_ordered.append(m)
        for s, ck in ckpts_for_month:
            if ck is None:
                continue
            data, cached_batches = compute_or_load_embeddings(
                ck, device=device, refresh=args.refresh_emb,
                machine=machine, schedule=schedule,
                cached_batches=cached_batches,
            )
            embeddings[(s["key"], m)] = data
        # Drop month batches before moving to the next month so we don't
        # accumulate every month's windows in RAM at once.
        del cached_batches

    if not embeddings:
        print("No embeddings — exiting before plot.")
        return
    if args.select or args.front_page or args.ppt:
        plot_select(
            embeddings, months[0], SERIES,
            perplexity=args.perplexity, seed=args.seed,
            refresh_tsne=args.refresh_tsne,
            front_page=args.front_page,
            ppt=args.ppt,
        )
    else:
        plot_grid(
            embeddings, train_months_ordered, SERIES,
            perplexity=args.perplexity, seed=args.seed,
            refresh_tsne=args.refresh_tsne,
        )


if __name__ == "__main__":
    main()
