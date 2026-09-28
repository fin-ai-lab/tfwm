"""Synchronized cross-section rank IC — the reported metric.

For every method the pipeline is IDENTICAL except for one step, the predictor:

    views -> encoder -> embedding -> [PREDICTOR] -> score -> rank IC per cell

An SSL checkpoint's predictor is a ridge fit on the TRAIN month's embeddings
against the same z-scores; a supervised checkpoint's is the head it trained.
Everything before (which anchors, which stocks, which crops, which labels) and
everything after (grouping, Spearman, aggregation) is shared code, so a
difference in IC is a difference in representation and not in protocol.

**Synchronized** is the whole point. A cross-sectional rank correlation is only
meaningful among names observed at the SAME instant, so this walks a fixed grid
of (date, anchor) cells and, at each one, cuts the SAME wall-clock window from
every stock that was quoting. One scale is drawn per cell — seeded from the
cell itself, so every checkpoint and every method sees a byte-identical panel.

The crop kernels are imported from the training path rather than reimplemented:
a view here is what training would have produced had it drawn this (start, agg).

Two phases, both resumable:

  1. **Embed** (GPU) — one pass per (month, checkpoint), cached under
     ``<checkpoint_root>/xs_ic_cache/<YYYY-MM>/``. The per-ticker-day decode is
     amortized across all anchors, so cost is ~1 decode per ticker-day plus one
     forward per (cell, stock).
  2. **Score** (CPU) — ridge or head, then ``grouped_rank_ic``.

Usage:
    uv run scripts/eval/xs_ic_eval.py \
        --run-ids 27n6djvm xeq5rl3e --train-month 2023-01 --eval-month 2023-02
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import os

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from market_jepa.augmentations import encode_view_metadata
from market_jepa.eval.checkpoints import dataset_flag
from market_jepa.backbone_config import backbone_block
from stable_finance.dataset import (
    DEFAULT_TARGET_HORIZONS,
    MARKET_SCHEMA,
    MarketSchedule,
    ViewSpec,
    build_norm_groups,
    build_sample_panel,
    choose_cell_aggregation,
    evenly_spaced_anchors,
)
from stable_finance import ColumnwiseRidge, grouped_rank_ic_by_label
from market_jepa.schemas import BLL01MachineConfig
from market_jepa.training.utils import append_view_info
from stable_finance.dataset.anchors import SESSION_LEN
from stable_finance.dataset.outcomes import ANCHOR_TARGET_TYPES
from stable_finance.dataset.targets import AnchorTargetStats as AnchorStats

GLOBAL_SEQ_LEN = 2048
FEATURE_COLUMNS = list(MARKET_SCHEMA.columns)
HORIZONS = DEFAULT_TARGET_HORIZONS
TARGET_TYPES = tuple(ANCHOR_TARGET_TYPES)
# The training crop's feasible resolutions: agg = round(scale * 23400 / 2048)
# for scale ~ U[0.5, 1.0]. Enumerated rather than redrawn so an eval view is
# always one the encoder saw during training.
AGG_CHOICES = tuple(range(6, 12))

# Ridge alpha per target type, with RIDGE_ALPHA as the fallback for anything
# not listed. Per-task rather than global because the optimum MOVES with the
# number of rows the probe is fit on and it moves by task: on the k2ind
# encoders spread_change wants alpha 1000 at 2k rows and alpha ~1 at 1.2M,
# while volatility_change sits at ~10 throughout. A single constant is only
# correct if every task plateaus in the same place, and they do not.
#
# The probe is the one step the SSL and supervised pipelines are allowed to
# differ in, so tuning it per task does not break the symmetry constraint --
# the supervised side gets a per-task LR and loss for the same reason.
#
# Values are set at the PRODUCTION fit size (see TRAIN_ANCHORS_PER_DAY). Any
# change to that invalidates them, because the alpha optimum is a function of
# n: re-run scripts/eval/probe_fit_size.py before trusting these.
RIDGE_ALPHA = 1.0
RIDGE_ALPHAS: dict[str, float] = {
    # alpha=10 at the ~400k production fit size. On BOTH k2ind encoders
    # measured so far it is the argmax or tied with it:
    #   2017-06 / nr3akk9e @393k   vol +0.1701 (a=1: +0.1679, a=0.1: +0.1532)
    #                              spr +0.0558 (a=1: +0.0538)
    #   2019-04 / glf3h9lu @393k   vol +0.2028 (tied with 0.1 and 1)
    #                              spr +0.0727 (a=1: +0.0705)
    # The 0.1-10 band is nearly flat at the plateau, so this is a low-stakes
    # choice there; it matters at SMALL fit sizes, where 0.1 loses badly and
    # 100-1000 wins. 1000 is wrong everywhere past ~100k rows.
    "volatility_change": 10.0,
    "spread_change": 10.0,
    # return is indistinguishable from zero at every alpha and every fit size
    # (best +0.013 against SE 0.006-0.010), so this is a placeholder for
    # uniformity rather than a measured optimum.
    "return": 10.0,
}


def ridge_alpha_for(name: str) -> float:
    """Alpha for a ``{target_type}_{horizon}`` column."""
    if name in RIDGE_ALPHAS:
        return RIDGE_ALPHAS[name]
    ttype = name.rsplit("_", 1)[0]
    return RIDGE_ALPHAS.get(ttype, RIDGE_ALPHA)


# Anchors per day when embedding the month the PROBE IS FIT ON. Deliberately
# larger than the eval month's 8: at 8 anchors one month yields ~90k rows, and
# the probe's IC is still climbing steeply there -- on the k2ind encoders the
# plateau is around 400k, and 90k leaves ~6% of volatility_change and ~14% of
# spread_change on the table. More anchors is far cheaper than more months
# because _panel_for_ticker_day builds the dense grid ONCE and reuses it
# across anchors, so 8 -> 36 costs 4.5x the encoder forwards and 1x the MDS
# reads and grid decodes, which are the expensive part.
#
# The EVAL month stays at 8. Its anchors define the reported cross-sections,
# so changing it changes the metric rather than the estimator.
TRAIN_ANCHORS_PER_DAY = 36

FFILL_IDX = MARKET_SCHEMA.forward_fill_indices
ZEROFILL_IDX = MARKET_SCHEMA.zero_fill_indices
NORM_GROUPS = build_norm_groups(FEATURE_COLUMNS)


def norm_groups_for(cfg) -> list | None:
    """The normalization a checkpoint was TRAINED with, as iter_panel wants it.

    None means the default per-view standardization; ``[]`` means none at all.
    Read from the run's own config rather than passed by the caller, because a
    scoring script that has to be told is a scoring script that will eventually
    not be — and scoring a ``norm_mode=none`` encoder on normalized views does
    not produce a slightly wrong number, it produces a meaningless one.
    """
    mode = "per_view"
    if cfg is not None:
        get = cfg.get if isinstance(cfg, dict) else (
            lambda k, d=None: getattr(cfg, k, d))
        ds = get("dataset", None) or {}
        dget = ds.get if isinstance(ds, dict) else (
            lambda k, d=None: getattr(ds, k, d))
        mode = str(dget("norm_mode", None) or "per_view")
    if mode == "none":
        return []
    if mode != "per_view":
        raise ValueError(f"unknown dataset.norm_mode: {mode!r}")
    return None


# ONE READER, in market_jepa, because build_untrained_encoder derives the
# random-init floor's width from the same two flags and cannot import from
# scripts/. Two copies of this is how the floor and the panel came to disagree
# about whether there was an information token at all. See its docstring for
# why absence means False rather than the live config default.
_dataset_flag = dataset_flag

def info_norm_stats_for(cfg) -> bool:
    """Whether this checkpoint's information token carries the (mu, sigma).

    Unlike the other two panel knobs this one changes the input WIDTH, so
    getting it wrong fails loudly at load rather than silently at score. It
    still lives here: the panel is what has to change, and it has to change
    from the run's own config like the rest.
    """
    return _dataset_flag(cfg, "info_norm_stats", "norm_stats_channels")


def info_window_for(cfg) -> bool:
    """Whether this checkpoint's information token carries the three descriptors.

    Same class of knob as info_norm_stats_for: it changes the input WIDTH, so a
    mismatch fails at load rather than scoring silently wrong -- but only
    because the backbone strips a FIXED count off the end. Emit them at eval
    for a model trained without them and the count is right while the contents
    are not, which does score silently wrong; hence this lives in
    panel_kwargs_for with the rest.
    """
    return _dataset_flag(cfg, "info_window", "time_info")


def _aug_getter(cfg):
    """A getter for the FIRST global augmentation's fields, or None.

    Shared by every panel accessor so they cannot disagree about which
    augmentation they are reading. Handles the two shapes a checkpoint's config
    arrives in: a whole hydra config, and the flat schema save_train_meta
    writes (which post_train_ic_eval reshapes into this same form).
    """
    if cfg is None:
        return None
    get = cfg.get if isinstance(cfg, dict) else (
        lambda k, d=None: getattr(cfg, k, d))
    ds = get("dataset", None) or {}
    dget = ds.get if isinstance(ds, dict) else (
        lambda k, d=None: getattr(ds, k, d))
    augs = dget("augmentations", None) or {}
    if not augs:
        return None
    first = augs["0"] if isinstance(augs, dict) and "0" in augs else (
        list(augs.values())[0] if isinstance(augs, dict) else augs[0])
    return first.get if isinstance(first, dict) else (
        lambda k, d=None: getattr(first, k, d))


def fixed_agg_for(cfg) -> int | None:
    """The single resolution a checkpoint was TRAINED at, or None if it varied.

    A run pinned to one seconds-per-token saw exactly one resolution for its
    whole training, and scoring it across the usual 6..11 band measures a
    distribution shift instead of the model. So the arm is scored the way it
    was trained -- which is what keeps rank IC an apples-to-apples number: the
    cells, the tickers and the z-scored labels are unchanged, and only the
    input view differs, which IS the ablation.

    ``global_agg_range`` IS THE HONEST KNOB, AND IS READ FIRST.
    ``global_scale_range`` is a fraction of N, the TRIMMED GRID LENGTH OF THAT
    TICKER-DAY, so a stock that starts quoting late gets a different resolution
    from a full-session name in the same cell -- which breaks the one property
    the synchronized panel rests on. A degenerate scale range is still honoured
    (older runs used it) but converted against the STANDARD session, since that
    is the only reading under which a cell has one resolution.

    THE COST IS CELLS, NOT COMPARABILITY. A fixed agg of 8 needs 16,384 s of
    history behind the anchor, so 5 of the 8 eval anchors qualify and 3 drop
    out. A paired comparison must restrict the other arm to the same anchors,
    or it is comparing different times of day.
    """
    aget = _aug_getter(cfg)
    if aget is None:
        return None

    band = aget("global_agg_range", None)
    if band and len(band) == 2 and int(band[0]) == int(band[1]):
        return max(1, int(band[0]))
    if band:
        return None                       # a real band, not a pinned point

    rng = aget("global_scale_range", None)
    if not rng or len(rng) != 2 or float(rng[0]) != float(rng[1]):
        return None
    return max(1, int(round(float(rng[0]) * SESSION_LEN / seq_len_for(cfg))))


def seq_len_for(cfg) -> int:
    """How many tokens the checkpoint's global view had.

    The OTHER way to shorten the context: hold seconds-per-token where the
    control has it and feed fewer tokens, against ``fixed_agg_for``'s way of
    holding the token count and shortening each token. The two separate span
    from resolution, so both have to reach the panel or the arm is scored on a
    view it never trained on -- 2048 tokens against the 256 it saw, eight times
    the context, and no error anywhere.
    """
    aget = _aug_getter(cfg)
    if aget is None:
        return GLOBAL_SEQ_LEN
    v = aget("global_seq_len", None)
    try:
        return max(1, int(v)) if v is not None else GLOBAL_SEQ_LEN
    except (TypeError, ValueError):
        return GLOBAL_SEQ_LEN


def _info_channel_width(panel_kwargs: dict) -> int:
    """How many information-token channels this panel appends.

    Derived from the same flags the panel is built with rather than from a
    constant, so the two cannot drift: 8 normalization values plus 3 window
    values, and 0 when both are off.
    """
    from market_jepa.augmentations import encode_view_metadata
    from stable_finance.dataset.views import ViewMetadata
    if not (panel_kwargs.get("info_norm_stats") or panel_kwargs.get("info_window")):
        return 0
    probe = ViewMetadata(
        start_seconds=0.0, end_seconds=1.0, aggregation_seconds=1.0,
        normalization_means=np.zeros(4), normalization_scales=np.ones(4),
    )
    return int(len(encode_view_metadata(
        probe,
        include_normalization=bool(panel_kwargs.get("info_norm_stats")),
        include_window=bool(panel_kwargs.get("info_window")),
    )))


def panel_kwargs_for(cfg) -> dict:
    """Every way this checkpoint's panel differs from the default one.

    One accessor rather than several, because the failure mode is a caller that
    passes one and forgets the other -- and every one of those failures is
    silent.
    """
    return {"norm_groups": norm_groups_for(cfg), "fixed_agg": fixed_agg_for(cfg),
            "info_norm_stats": info_norm_stats_for(cfg),
            "seq_len": seq_len_for(cfg),
            "info_window": info_window_for(cfg)}


def cell_agg(date_str: str, anchor: int, t_idx_max: int,
             seq_len: int = GLOBAL_SEQ_LEN) -> int | None:
    """The resolution every stock uses at one cell, or None if none fits.

    Seeded from (date, anchor) alone so the panel is a property of the grid,
    not of whichever checkpoint happens to be running. Feasible means the whole
    2048-token window fits before the anchor — the same constraint the training
    sampler enforces, so no eval view is one training could not have drawn.
    """
    return choose_cell_aggregation(
        date_str, anchor, t_idx_max, seq_len, choices=AGG_CHOICES,
    )


def day_anchors(n_per_day: int, seq_len: int = GLOBAL_SEQ_LEN) -> np.ndarray:
    """``n_per_day`` anchors spread evenly over the FEASIBLE band.

    A 2048-token view at the finest trained resolution (6 s/token) already
    spans 12288 s — 3.4 hours — so no crop in this family can end before about
    12:55. Anchors earlier than that admit no view at all, which is a property
    of the augmentation, not of the eval: 100% of training views likewise end
    in the second half of the session. Spreading over the whole grid would
    silently spend half the anchors on empty cells.

    LEAVE ``seq_len`` AT THE DEFAULT unless you mean something unusual. A
    SHORTER view is feasible behind strictly more anchors, so passing an arm's
    own length would hand the short-context arms a different — and larger —
    anchor set than the control, and the two would no longer be scored on the
    same cells. The whole point of a view ablation is that only the view moves.
    Every arm therefore shares the control's grid and a short view simply has
    slack behind it; iter_panel still enforces feasibility per cell.
    """
    return evenly_spaced_anchors(
        n_per_day, seq_len,
        aggregation_choices=AGG_CHOICES,
        horizons=HORIZONS,
    )


def _panel_for_ticker_day(sample, schedule, anchors_tod, stats, hs, rf_merger=None,
                          norm_groups=None, fixed_agg=None, info_norm_stats=False,
                          seq_len: int = GLOBAL_SEQ_LEN,
                          info_window: bool = False):
    """Every (anchor, view, uniform-rank target) this ticker-day contributes.

    Returns lists of (anchor, view (T, C) float32, target (T*H,) float32,
    raw target, quote (2,) float64); empty when the day is unusable. The dense
    grid is built once and reused across anchors, which is what makes a
    full-panel pass affordable.

    The QUOTE is the best bid/ask in natural units at the decision row --
    unnormalized, and therefore not recoverable from the view, whose scale is
    per-observation. It is what the execution stage charges; nothing on the IC
    path reads it.
    """
    out = []
    view_spec = ViewSpec(
        sequence_length=seq_len,
        aggregation_seconds=(fixed_agg, fixed_agg) if fixed_agg else (6, 11),
    )
    observations = build_sample_panel(
        sample, schedule, anchors_tod, stats,
        view_spec=view_spec,
        target_types=TARGET_TYPES,
        horizons=hs,
        target_transform="uniform",
        normalization_groups=(NORM_GROUPS if norm_groups is None else norm_groups),
    )
    for observation in observations:
        view, metadata = observation.view, observation.metadata
        # The three window descriptors, in the SAME trailing position the
        # training path puts them, because the backbone strips a fixed count
        # off the end. `start` is this ticker's grid row, so tod_offset + start
        # is seconds past the standard open -- the identical expression
        # StreamingMarketDataset uses (tod_offset_base + v_start).
        if info_norm_stats or info_window:
            view = append_view_info(
                view,
                encode_view_metadata(
                    metadata,
                    include_normalization=info_norm_stats,
                    include_window=info_window,
                ),
            )
        # Risk-factor channels are appended AFTER the main view is normalized,
        # exactly as in training (_merge_risk_factors normalizes the RF block
        # in its own feature groups). rf_offset is measured from the 09:30 RF
        # grid origin, so it is this ticker's tod_offset plus the view start —
        # the same expression the training path uses.
        if rf_merger is not None:
            view = rf_merger.merge(
                view, observation.date, int(metadata.start_seconds),
                int(len(view) * metadata.aggregation_seconds),
                int(metadata.aggregation_seconds),
            )
        np.nan_to_num(view, copy=False, nan=0.0)
        out.append((
            observation.anchor,
            view.astype(np.float32),
            observation.target,
            observation.raw_target,
            observation.quote,
        ))
    return out


def iter_panel(
    month_dir: Path, stats: AnchorStats, schedule: MarketSchedule,
    anchors_tod: np.ndarray, batch_size: int,
    shard_index: int = 0, num_shards: int = 1, rf_merger=None,
    norm_groups=None, fixed_agg=None, info_norm_stats=False,
    seq_len: int = GLOBAL_SEQ_LEN, info_window: bool = False,
):
    """Yield ``(views (B, T, C) float32, metas)`` — the panel, minus the encoder.

    THE panel. ``embed_month`` is a consumer of this generator and so is the
    TSFM layer sweep, which cannot use ``embed_month`` (it needs every layer's
    embedding from one forward and never materializes X). Two loops that merely
    looked alike would drift, and then the two arms would no longer be scored
    on the same cross-sections — the one property the whole comparison rests on.

    ``metas`` entries are ``(z, date, anchor, ticker, raw, quote)``, aligned
    with the rows of ``views``. ``quote`` is ``[best_bid, best_ask]`` at the
    decision in natural units, for the execution stage; every IC consumer
    ignores it.

    ``norm_groups`` overrides the standardization applied to each view. None
    means the training default (``NORM_GROUPS``); an EMPTY LIST means none at
    all, which is how a ``dataset.norm_mode=none`` checkpoint must be scored —
    normalizing at eval a model that never saw normalization at train is not a
    milder version of the ablation, it is a different model. Passing this
    explicitly (rather than reading a global) is what lets one process score a
    normalized and an unnormalized checkpoint in turn.

    ``fixed_agg`` pins every cell to one resolution instead of drawing it from
    ``AGG_CHOICES``, for a checkpoint trained at a single scale. Cells whose
    anchor is too early for that window are dropped, not truncated.

    ``shard_index``/``num_shards`` split the month's MDS shards across
    processes. Sharding on MDS shards (not on rows) keeps each ticker-day
    whole, which the decode amortization depends on.
    """
    import tempfile

    from stable_finance.dataset.mds import open_shard, shard_list

    hs = list(HORIZONS)
    shards = shard_list(month_dir)
    if num_shards > 1:
        shards = shards[shard_index::num_shards]

    buf_v, buf_meta = [], []
    with tempfile.TemporaryDirectory(prefix="xs_ic_") as tmpdir:
        for shard_meta in shards:
            reader = open_shard(month_dir, shard_meta, tmpdir)
            for i in range(shard_meta["samples"]):
                sample = reader.get_item(i)
                rows = _panel_for_ticker_day(sample, schedule, anchors_tod, stats,
                                             hs, rf_merger, norm_groups, fixed_agg,
                                             info_norm_stats, seq_len,
                                             info_window)
                for anchor, view, target, raw, quote in rows:
                    buf_v.append(view)
                    buf_meta.append(
                        (target, str(sample["date"]), anchor,
                         str(sample["ticker"]), raw, quote)
                    )
                    if len(buf_v) >= batch_size:
                        yield np.stack(buf_v), list(buf_meta)
                        buf_v.clear(); buf_meta.clear()
    if buf_v:
        yield np.stack(buf_v), list(buf_meta)


def panel_source(month_dir, ym, stats, schedule, anchors_tod, batch_size,
                 shard_index=0, num_shards=1, rf_merger=None, norm_groups=None,
                 fixed_agg=None, info_norm_stats=False, seq_len=GLOBAL_SEQ_LEN,
                 info_window=False):
    """``iter_panel``, served from the on-disk cache when one matches.

    Falls through to a live build whenever MJ_PANEL_CACHE is unset, the panel
    is sharded (a shard is a slice of the month, and the cache stores whole
    months), or no cache exists for this exact key. Identical contract either
    way, so callers cannot tell which they got -- see panel_cache.py for why
    the key must cover every field of panel_kwargs_for.
    """
    import panel_cache as _pc

    root = _pc.cache_root()
    if root is not None and num_shards == 1:
        key = _pc.panel_key(
            anchors_per_day=len(anchors_tod),
            stats_tag=_stats_tag(stats),
            has_rf=rf_merger is not None, norm_groups=norm_groups,
            fixed_agg=fixed_agg, seq_len=seq_len)
        if _pc.is_built(root, ym, key):
            # The info flags are NOT part of the key -- one panel stores the 9
            # real channels plus the token's 11 constants, and the reader
            # broadcasts back whichever blocks this checkpoint was trained
            # with. So an info-on and an info-off run share a panel and each
            # still gets exactly the tensor it would have built live.
            print(f"==> panel cache HIT {ym} (key {key['hash']}, "
                  f"info_norm_stats={info_norm_stats} info_window={info_window})",
                  flush=True)
            return _pc.iter_cached(root, ym, key, batch_size,
                                   info_norm_stats=info_norm_stats,
                                   info_window=info_window)
    return iter_panel(month_dir, stats, schedule, anchors_tod, batch_size,
                      shard_index, num_shards, rf_merger, norm_groups,
                      fixed_agg, info_norm_stats, seq_len, info_window)


def _stats_tag(stats) -> str:
    """The anchor-stat TABLE SET a panel's labels came from.

    Part of the cache key because the tables ARE the target: the same month
    under xs_anchor_stats_fwdvwap60 and under the retired mid-to-mid tables is
    two different panels with identical geometry.
    """
    tag = getattr(stats, "table_set", None)
    if not tag:
        # Refuse rather than key a panel on "unknown": a cache that cannot see
        # which tables made its labels will happily serve the wrong target.
        raise RuntimeError(
            "AnchorStats has no table_set; panel caching needs it to key the "
            "target tables (stable_finance.dataset.targets)")
    return str(tag)


def embed_month_many(
    models: dict, month_dir: Path, ym: str, stats: AnchorStats,
    schedule: MarketSchedule, anchors_tod: np.ndarray, device, batch_size: int,
    shard_index: int = 0, num_shards: int = 1, allow_empty: bool = False,
    rf_merger=None, norm_groups=None, fixed_agg=None, info_norm_stats=False,
    seq_len: int = GLOBAL_SEQ_LEN, info_window: bool = False,
):
    """One CPU pass over a month, N encoders: ``{key: panel}``.

    THE DECODE IS THE COST, NOT THE FORWARD. Building a ticker-day's dense 1 Hz
    grid and cutting its views is single-threaded CPU work; a ViT-384 forward
    over the same views is a fraction of it — measured here as a GPU sitting
    near 47% duty while one checkpoint is scored. So scoring M checkpoints on
    the same month one at a time pays the decode M times to change only what
    happens after it.

    Every model sees the SAME views in the same order, which is not merely an
    optimization: the whole comparison rests on the arms being scored on one
    panel, and sharing the generator makes that structural rather than a
    property of two runs agreeing on their seeds.

    Models must share input geometry (channel count and sequence length) since
    they consume one batch. A risk-factor model cannot be grouped with a plain
    one; pass separate calls. The same applies to ``norm_groups`` and
    ``fixed_agg``: a ``norm_mode=none`` checkpoint, or one trained at a single
    resolution, is fed a different tensor from a default one and cannot share
    this pass with it.
    """
    Xs = {k: [] for k in models}
    Z, raw_targets_, dates, ancs, tickers, quotes = [], [], [], [], [], []
    for views, metas in panel_source(
            month_dir, ym, stats, schedule, anchors_tod, batch_size,
            shard_index, num_shards, rf_merger, norm_groups, fixed_agg,
            info_norm_stats, seq_len, info_window):
        x = torch.from_numpy(views).permute(0, 2, 1).to(device)
        lengths = torch.full((len(views),), views.shape[1], dtype=torch.long,
                             device=device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            for k, model in models.items():
                emb = model.encode([x], [lengths])["embeddings"][:, 0, :]
                Xs[k].append(emb.float().cpu().numpy())
        for target, d, a, t, raw, quote in metas:
            Z.append(target); raw_targets_.append(raw)
            dates.append(d); ancs.append(a); tickers.append(t)
            quotes.append(np.full(2, np.nan) if quote is None else quote)

    if not Z:
        if allow_empty:
            return {k: None for k in models}
        raise RuntimeError(f"{ym}: the panel is empty — check the anchor tables")
    shared = {
        "z": np.stack(Z),
        "raw": np.stack(raw_targets_),
        "target_transform": "uniform",
        "date": np.array(dates),
        "anchor": np.array(ancs),
        "ticker": np.array(tickers),
        # (n_rows, 2) natural-unit [best_bid, best_ask] at each decision.
        "quote": np.stack(quotes),
        "target_names": np.array(
            [f"{t}_{h}" for t in TARGET_TYPES for h in HORIZONS]
        ),
    }
    return {k: {"X": np.concatenate(v), **shared} for k, v in Xs.items()}


def embed_month(
    model, month_dir: Path, ym: str, stats: AnchorStats, schedule: MarketSchedule,
    anchors_tod: np.ndarray, device, batch_size: int,
    shard_index: int = 0, num_shards: int = 1, allow_empty: bool = False,
    rf_merger=None, norm_groups=None, fixed_agg=None, info_norm_stats=False,
    seq_len: int = GLOBAL_SEQ_LEN, info_window: bool = False,
):
    """One GPU pass over a month: the full synchronized panel and its labels.

    The per-ticker-day decode is single-threaded CPU work that dominates the
    GPU forward for a ViT-scale encoder, so N workers on one node scale close
    to linearly; each writes its own partial panel and a reducer concatenates.

    A thin wrapper over ``embed_month_many`` rather than its own loop, so the
    one-model and many-model paths cannot drift apart in what they feed the
    encoder.
    """
    return embed_month_many(
        {"_": model}, month_dir, ym, stats, schedule, anchors_tod, device,
        batch_size, shard_index, num_shards, allow_empty, rf_merger,
        norm_groups, fixed_agg, info_norm_stats, seq_len, info_window,
    )["_"]


# WHICH TARGETS GET AN AUC. The ridge probe is fit for all 18 target columns
# because a ridge on 384 features is nearly free; a multinomial logistic on the
# same panel is ~100x that, and the reported horizon is 900 s. Running all 18
# would spend hours per sweep to fill columns no figure reads. Override with
# score(..., auc_tasks=...) when a different horizon is actually being read.
AUC_TASKS = tuple(f"{t}_900" for t in TARGET_TYPES)

# HOW MUCH DATA THE LOGISTIC GETS, AND HOW HARD IT TRIES.
#
# The ridge sees the whole probe-fit month; the logistic sees a seeded
# subsample of it. Not a statistical choice — 150k rows against 384 features
# is already far past where a linear probe's coefficients move — but a cost
# one. Embeddings are strongly collinear (the ridge reports rcond ~ 6e-8), so
# lbfgs does not converge on them and spends its whole max_iter budget; cost
# is then linear in rows x iterations, and the full ~600k-row month runs ~20
# minutes PER FIT. At 300 checkpoints that is weeks.
#
# tol is loosened from sklearn's 1e-4 for the same reason: on an
# ill-conditioned problem the last decade of gradient norm buys nothing a
# ranking metric can see. Both are knobs, and xs_score_many logs the iteration
# count actually used so a run that silently hit the ceiling is visible.
AUC_FIT_ROWS = 150_000
AUC_TOL = 1e-3


def head_readout(model, X: np.ndarray, device) -> dict:
    """``{task: (scores, proba)}`` for a checkpoint's own TRAINED head(s).

    Empty for every SSL arm, which has no head at all. A single-task
    supervised checkpoint returns one entry; a MULTIHEAD returns one per task,
    each scored against its own target -- a head emits one number and
    correlating it with the other five targets produces numbers that look like
    results and mean nothing.

    A BINNED head emits (n, k): its scalar readout is the expected bin -- the
    only thing a rank correlation can consume -- and the full softmax is kept
    as well, because a macro one-vs-rest AUC needs a score per class and the
    expected bin has already collapsed them. A scalar head is its own readout
    and has no distribution to report.

    A RANDOMLY INITIALIZED head returns NOTHING. load_model builds one when a
    supervised checkpoint has no saved head, for callers that only want
    encode(); before 2026-08-29 the only test here was ``model.head is not
    None``, so every multihead checkpoint -- which saves heads.pt and so
    tripped that fallback -- had an untrained head scored and written to
    xs_ic.json as a result. A number no one can distinguish from a real one is
    worse than no number.

    Shared by the in-job hook and the batch re-scorer so the head can never be
    read out two different ways.
    """
    if getattr(model, "_head_is_random", False):
        return {}
    if hasattr(model, "task_specs") and getattr(model, "heads", None) is not None:
        heads = {t: model.heads[t] for t in model.task_specs}
    elif hasattr(model, "task_spec") and getattr(model, "head", None) is not None:
        heads = {model.task_spec.name: model.head}
    else:
        return {}
    out: dict = {}
    with torch.no_grad():
        xt = torch.from_numpy(X).to(device)
        for task, head in heads.items():
            logits = head(xt)
            if logits.ndim == 2 and logits.shape[-1] > 1:
                from market_jepa.eval.heads import expected_bin

                out[task] = (expected_bin(logits).cpu().numpy(),
                             torch.softmax(logits.float(), dim=-1).cpu().numpy())
            else:
                out[task] = (logits.float().cpu().numpy().ravel(), None)
    return out


def macro_ovr_auc(proba: np.ndarray, labels: np.ndarray) -> float:
    """Mean one-vs-rest ROC AUC over the classes actually present.

    The historical ``probe/logistic_auc_return_900_k5`` number, restored: with
    k>2 ordered bins there is no single positive class, so each bin is scored
    against the rest and the AUCs are averaged unweighted. Classes absent from
    the eval split are skipped rather than counted as 0.5, which is what the
    deleted mass_eval_return_900_k5.py did.
    """
    from sklearn.metrics import roc_auc_score

    aucs = []
    for c in np.unique(labels):
        yb = (labels == c).astype(int)
        n = int(yb.sum())
        if n == 0 or n == len(yb) or c >= proba.shape[1]:
            continue
        aucs.append(roc_auc_score(yb, proba[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def _hard_bins(y_fit: np.ndarray, y_apply: np.ndarray, k: int):
    """(fit labels, apply labels) as hard bin indices at k equal-count bins.

    Edges come from the PROBE-FIT month only — the eval month never informs
    its own binning. Ties spread their mass across the bins they straddle, so
    the hard label is the bin holding the most of it.
    """
    from market_jepa.eval import discretize as D

    d = D.fit(y_fit, k)
    if d is None:
        return None, None
    return d.apply(y_fit).argmax(1), d.apply(y_apply).argmax(1)


def raw_targets(cache: dict, stats: AnchorStats) -> np.ndarray:
    """Return raw panel targets, including compatibility with old z caches.

    Format-4 panels store raw values alongside the default empirical-uniform
    labels. Older embedding caches contain only ordinary cross-sectional
    z-scores, which remain exactly invertible through their anchor statistics.
    The AUC needs raw values because a binned head's classes partition the raw
    target rather than either standardized representation.
    """
    if "raw" in cache:
        return np.asarray(cache["raw"], dtype=np.float64)
    return stats.unstandardize_panel(
        cache["z"], cache["date"], cache["anchor"], TARGET_TYPES, HORIZONS,
    )


def _project_proba(proba: np.ndarray, y_fit: np.ndarray, k_to: int):
    """Re-express a k-bin distribution over ``k_to`` bins of the same target.

    A macro one-vs-rest AUC is only comparable across arms when the partition
    is held fixed, and this sweep varies the head's bin count over {5, 11, 21}.
    Left at its native k a head would be scored on a different question in
    every column of the figure — and a harder one, since narrower bins push
    the middle classes' one-vs-rest AUC toward 0.5 no matter how good the
    model is. So the head's distribution is marginalized onto the reported
    partition:

        P(bin_to = c) = sum_j P(bin_k = j) * P(bin_to = c | bin_k = j)

    The conditional is COUNTED on the probe-fit month rather than assumed, so
    it needs no uniformity argument; because both partitions are equal-count
    quantiles of the same variable it comes out near-deterministic (exactly so
    when k_to divides k, one straddling bin otherwise).

    One approximation remains: these edges are refitted on the anchor panel,
    while the head's own were fitted on the training dataloader's crops. Same
    month, same quantile definition, so the edges agree to sampling noise —
    but they are not the identical floats, and the training discretizer is not
    saved with the checkpoint to make them so.
    """
    from market_jepa.eval import discretize as D

    k_from = proba.shape[1]
    if k_from == k_to:
        return proba
    d_from, d_to = D.fit(y_fit, k_from), D.fit(y_fit, k_to)
    if d_from is None or d_to is None:
        return None
    a, b = d_from.apply(y_fit).argmax(1), d_to.apply(y_fit).argmax(1)
    W = np.zeros((k_from, k_to))
    np.add.at(W, (a, b), 1.0)
    rows = W.sum(1, keepdims=True)
    # A fit-month bin no row landed in carries no information about the
    # reported partition; spreading it uniformly is the honest prior and
    # cannot manufacture separation.
    W = np.divide(W, rows, out=np.full_like(W, 1.0 / k_to), where=rows > 0)
    return proba @ W


# THE PROBE IS FIT ON THE TARGET STABLE-FINANCE HANDED US, and that is the
# whole of the policy. ``build_session_panel(target_transform=...)`` chooses
# among stable-finance's four cross-sectional representations (raw | zscore |
# uniform | rank) and computes them from the exact per-cell order statistics;
# this panel asks for "uniform", so ``cache["z"]`` IS the uniform score. There
# is nothing left for the probe to transform.
#
# WHY UNIFORM AND NOT THE MOMENT Z-SCORE. The reported metric is a within-cell
# Spearman rank IC, and Spearman IC IS the Pearson correlation of ranks -- but
# a squared-error fit on a z-score puts a day's biggest movers at 3.5 sigma,
# where they dominate the normal equations while being exactly what the metric
# cannot see. Measured on the day-level panel, every target at unit variance
# so the ridge penalty means the same thing across transforms:
#
#     fit target      alpha=1    alpha=10   alpha=100
#     zscore          +0.7727    +0.7799    +0.7642
#     uniform         +0.8167    +0.8152    +0.7905
#
# Monotone at every penalty, so it is the transform and not extra shrinkage.
#
# FIT AND SCORE ON THE SAME COLUMN, deliberately. Rank IC is invariant to any
# monotone within-cell transform, so scoring on uniform and scoring on zscore
# give the identical number; only the FIT is sensitive to which one it sees.
# Keeping one column means the probe is fit on exactly what it is judged
# against, and the choice lives at the panel build where the definition does,
# rather than as a second transform applied on the way into the ridge.


def _project_proba(proba: np.ndarray, y_fit: np.ndarray, k_to: int):
    """Re-express a k-bin distribution over ``k_to`` bins of the same target.

    A macro one-vs-rest AUC is only comparable across arms when the partition
    is held fixed, and this sweep varies the head's bin count over {5, 11, 21}.
    Left at its native k a head would be scored on a different question in
    every column of the figure — and a harder one, since narrower bins push
    the middle classes' one-vs-rest AUC toward 0.5 no matter how good the
    model is. So the head's distribution is marginalized onto the reported
    partition:

        P(bin_to = c) = sum_j P(bin_k = j) * P(bin_to = c | bin_k = j)

    The conditional is COUNTED on the probe-fit month rather than assumed, so
    it needs no uniformity argument; because both partitions are equal-count
    quantiles of the same variable it comes out near-deterministic (exactly so
    when k_to divides k, one straddling bin otherwise).

    One approximation remains: these edges are refitted on the anchor panel,
    while the head's own were fitted on the training dataloader's crops. Same
    month, same quantile definition, so the edges agree to sampling noise —
    but they are not the identical floats, and the training discretizer is not
    saved with the checkpoint to make them so.
    """
    from market_jepa.eval import discretize as D

    k_from = proba.shape[1]
    if k_from == k_to:
        return proba
    d_from, d_to = D.fit(y_fit, k_from), D.fit(y_fit, k_to)
    if d_from is None or d_to is None:
        return None
    a, b = d_from.apply(y_fit).argmax(1), d_to.apply(y_fit).argmax(1)
    W = np.zeros((k_from, k_to))
    np.add.at(W, (a, b), 1.0)
    rows = W.sum(1, keepdims=True)
    # A fit-month bin no row landed in carries no information about the
    # reported partition; spreading it uniformly is the honest prior and
    # cannot manufacture separation.
    W = np.divide(W, rows, out=np.full_like(W, 1.0 / k_to), where=rows > 0)
    return proba @ W


def _project_proba(proba: np.ndarray, y_fit: np.ndarray, k_to: int):
    """Re-express a k-bin distribution over ``k_to`` bins of the same target.

    A macro one-vs-rest AUC is only comparable across arms when the partition
    is held fixed, and this sweep varies the head's bin count over {5, 11, 21}.
    Left at its native k a head would be scored on a different question in
    every column of the figure — and a harder one, since narrower bins push
    the middle classes' one-vs-rest AUC toward 0.5 no matter how good the
    model is. So the head's distribution is marginalized onto the reported
    partition:

        P(bin_to = c) = sum_j P(bin_k = j) * P(bin_to = c | bin_k = j)

    The conditional is COUNTED on the probe-fit month rather than assumed, so
    it needs no uniformity argument; because both partitions are equal-count
    quantiles of the same variable it comes out near-deterministic (exactly so
    when k_to divides k, one straddling bin otherwise).

    One approximation remains: these edges are refitted on the anchor panel,
    while the head's own were fitted on the training dataloader's crops. Same
    month, same quantile definition, so the edges agree to sampling noise —
    but they are not the identical floats, and the training discretizer is not
    saved with the checkpoint to make them so.
    """
    from market_jepa.eval import discretize as D

    k_from = proba.shape[1]
    if k_from == k_to:
        return proba
    d_from, d_to = D.fit(y_fit, k_from), D.fit(y_fit, k_to)
    if d_from is None or d_to is None:
        return None
    a, b = d_from.apply(y_fit).argmax(1), d_to.apply(y_fit).argmax(1)
    W = np.zeros((k_from, k_to))
    np.add.at(W, (a, b), 1.0)
    rows = W.sum(1, keepdims=True)
    # A fit-month bin no row landed in carries no information about the
    # reported partition; spreading it uniformly is the honest prior and
    # cannot manufacture separation.
    W = np.divide(W, rows, out=np.full_like(W, 1.0 / k_to), where=rows > 0)
    return proba @ W


# THE PROBE IS FIT ON THE TARGET STABLE-FINANCE HANDED US, and that is the
# whole of the policy. ``build_session_panel(target_transform=...)`` chooses
# among stable-finance's four cross-sectional representations (raw | zscore |
# uniform | rank) and computes them from the exact per-cell order statistics;
# this panel asks for "uniform", so ``cache["z"]`` IS the uniform score. There
# is nothing left for the probe to transform.
#
# WHY UNIFORM AND NOT THE MOMENT Z-SCORE. The reported metric is a within-cell
# Spearman rank IC, and Spearman IC IS the Pearson correlation of ranks -- but
# a squared-error fit on a z-score puts a day's biggest movers at 3.5 sigma,
# where they dominate the normal equations while being exactly what the metric
# cannot see. Measured on the day-level panel, every target at unit variance
# so the ridge penalty means the same thing across transforms:
#
#     fit target      alpha=1    alpha=10   alpha=100
#     zscore          +0.7727    +0.7799    +0.7642
#     uniform         +0.8167    +0.8152    +0.7905
#
# Monotone at every penalty, so it is the transform and not extra shrinkage.
#
# FIT AND SCORE ON THE SAME COLUMN, deliberately. Rank IC is invariant to any
# monotone within-cell transform, so scoring on uniform and scoring on zscore
# give the identical number; only the FIT is sensitive to which one it sees.
# Keeping one column means the probe is fit on exactly what it is judged
# against, and the choice lives at the panel build where the definition does,
# rather than as a second transform applied on the way into the ridge.


def _project_proba(proba: np.ndarray, y_fit: np.ndarray, k_to: int):
    """Re-express a k-bin distribution over ``k_to`` bins of the same target.

    A macro one-vs-rest AUC is only comparable across arms when the partition
    is held fixed, and this sweep varies the head's bin count over {5, 11, 21}.
    Left at its native k a head would be scored on a different question in
    every column of the figure — and a harder one, since narrower bins push
    the middle classes' one-vs-rest AUC toward 0.5 no matter how good the
    model is. So the head's distribution is marginalized onto the reported
    partition:

        P(bin_to = c) = sum_j P(bin_k = j) * P(bin_to = c | bin_k = j)

    The conditional is COUNTED on the probe-fit month rather than assumed, so
    it needs no uniformity argument; because both partitions are equal-count
    quantiles of the same variable it comes out near-deterministic (exactly so
    when k_to divides k, one straddling bin otherwise).

    One approximation remains: these edges are refitted on the anchor panel,
    while the head's own were fitted on the training dataloader's crops. Same
    month, same quantile definition, so the edges agree to sampling noise —
    but they are not the identical floats, and the training discretizer is not
    saved with the checkpoint to make them so.
    """
    from market_jepa.eval import discretize as D

    k_from = proba.shape[1]
    if k_from == k_to:
        return proba
    d_from, d_to = D.fit(y_fit, k_from), D.fit(y_fit, k_to)
    if d_from is None or d_to is None:
        return None
    a, b = d_from.apply(y_fit).argmax(1), d_to.apply(y_fit).argmax(1)
    W = np.zeros((k_from, k_to))
    np.add.at(W, (a, b), 1.0)
    rows = W.sum(1, keepdims=True)
    # A fit-month bin no row landed in carries no information about the
    # reported partition; spreading it uniformly is the honest prior and
    # cannot manufacture separation.
    W = np.divide(W, rows, out=np.full_like(W, 1.0 / k_to), where=rows > 0)
    return proba @ W


# THE PREDICTION READOUT IS THE LAST TOKEN, FOR BOTH ARMS.
#
# What a model trains through and what the probe should read are different
# questions, and here they have different answers. LeJEPA computes its
# invariance loss on the MEAN over patch tokens -- that is what shapes a
# whole-day representation -- but reading those same weights at the LAST token
# is worth +0.055 / +0.083 / +0.112 (return / vol / spread, 5/5 months on
# holdout-2) over reading them at the mean. The supervised arm already trains
# pool="last". So both are scored at the last token.
#
# NORMALISING THE CONFIG IS WHY THIS IS ONE CHANGE AND NOT THREE. load_model,
# architecture_signature and build_untrained_encoder all read the pool out of
# the config they are handed, so rewriting it here makes the trained encoder,
# the cache key and the random-init FLOOR agree by construction. A floor read
# at a different token than the model it is subtracted from is not that
# model's floor -- and before this, a mean-trained LeJEPA run got a
# mean-pooled floor while its reported number came from a last-token rescore.
#
# It also retires prep_lastprobe.py, which got the same result by copying a
# run directory and rewriting `pool` in its train_meta.json.
PREDICT_POOL = "last"


def _readout_tag(cfg: dict) -> str:
    """The readout a cached embedding was produced at, for the cache key.

    Just the pool: run_id already pins the weights and therefore the position
    encoding, and the pool is the one thing _predict_readout rewrites.
    """
    from market_jepa.backbone_config import backbone_block
    return str((backbone_block(cfg) or {}).get("pool") or "none")


def _predict_readout(cfg: dict) -> dict:
    """Return ``cfg`` with the backbone readout pinned to PREDICT_POOL."""
    import copy

    cfg = copy.deepcopy(cfg)
    # THE BLOCK THE MODE ACTUALLY TRAINS, via the one resolver. This used to
    # branch on `"IJEPA" in mode_target` and otherwise pin onto the TOP-LEVEL
    # block -- which load_model, architecture_signature and
    # build_untrained_encoder all stopped reading. For every SSL mode that
    # made PREDICT_POOL a NO-OP: the pin landed on a block nothing consulted,
    # the encoder was still read at the mean, and the arm silently never
    # received the readout this constant exists to apply.
    #
    # backbone_block returns a reference INTO the copy, so mutating it here
    # rewrites the config the caller goes on to use.
    bb = backbone_block(cfg)
    if isinstance(bb, dict):
        bb["pool"] = PREDICT_POOL
    return cfg


def score(train_cache, eval_cache, head_scores=None, head_task=None,
          head_proba=None, auc_bins=5, train_stats=None, eval_stats=None,
          auc_tasks=AUC_TASKS, auc_fit_rows=AUC_FIT_ROWS, auc_tol=AUC_TOL,
          auc_extra_bins=(), head_readouts=None, head_only=False):
    """Per-cell rank IC for every target column.

    THE REPORTED NUMBER IS ALWAYS THE RIDGE PROBE, for every arm. A ridge is
    fit on ``train_cache`` embeddings and scored on ``eval_cache``, whether
    the checkpoint came from LeJEPA or from supervised cross-entropy. That is
    the symmetry the whole comparison rests on: the two arms differ in their
    TRAINING OBJECTIVE and in nothing else on the reporting path, so a
    supervised model cannot look better merely because it was handed a trained
    predictor while the SSL model was handed a linear one.

    ``head_scores`` (n_eval,) is therefore ADDITIVE, not a substitution. When
    given, the supervised checkpoint's own head readout is scored as well and
    returned under ``head:<task>``, so "would the head have done better than
    its probe?" is a measured question rather than an assumed answer. For a
    binned head the readout is its expected bin — see
    ``market_jepa.eval.heads.expected_bin``.

    THIS IS THE ONLY SCORER. The SSL arm reaches it offline through this
    module's main(), the supervised arm reaches it in-job through
    scripts/generic/post_train_ic_eval.py. Two implementations that merely
    look alike would silently drift and the comparison would stop meaning
    anything.

    ``head_readouts`` is ``{task: (scores, proba)}`` as head_readout returns
    it -- each head scored against ITS OWN target, because a head emits one
    number and correlating it against the other five would produce numbers
    that look like results and mean nothing. A multihead contributes one entry
    per task. The older singular ``head_scores``/``head_task``/``head_proba``
    trio is the same thing for one head and is folded into the mapping below;
    ``head_task`` is required whenever ``head_scores`` is given.

    AUC RIDES ALONG, ON THE RAW TARGET. Every target also gets the pre-IC
    metric back: bin the RAW target into ``auc_bins`` equal-count bins on the
    probe-fit month, fit StandardScaler + LogisticRegression on the same
    embeddings the ridge sees, and take the macro one-vs-rest AUC — the recipe
    from the deleted mass_eval_return_900_k5.py. ``head_proba`` (n_eval, k)
    adds the head's own softmax, marginalized onto the same ``auc_bins``
    partition so probe and head answer the same question in every cell of the
    figure (_project_proba).

    ``auc_native`` is the same pair WITHOUT the projection, at the head's own
    k, with the probe refit at that k so the number has a reference. Both are
    reported because the projection is a modelling choice, and reporting only
    the projected number would hide it; see the head branch for which of the
    two may be read across k.

    Raw, not z, and this is not a detail: the head's bins are quantiles of the
    raw distribution, and the two partitions disagree on ~50% of rows at k=5
    and ~86% at k=21 (AnchorStats.unstandardize_panel). Scoring the head's
    softmax against z bins would pair class j with a bin it was never trained
    to mean and read out as a broken head. The IC is untouched by this — it is
    computed within a cell, where raw and z are monotone.

    AUC therefore REQUIRES ``train_stats``/``eval_stats``, the anchor tables
    the panel was standardized against. Without them no AUC is emitted, rather
    than a mismatched one. It is also restricted to ``auc_tasks`` (see
    AUC_TASKS) — the logistic is the expensive fit on this path, and the ridge
    still covers every column.

    NOT COMPARABLE TO THE ARCHIVED AUCs. Those were computed on the
    zero-threshold binning (a dedicated +/-0.0005 middle bin) which no longer
    exists, and on the old [0.3, 1.0] eval crop scale. Same recipe, different
    partition of the same targets: read these against each other, not against
    numbers from before 2026-08-14.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if head_scores is not None and head_task is None:
        raise ValueError("head_scores requires head_task — scoring one head "
                         "against every target produces meaningless numbers")
    heads = dict(head_readouts or {})
    if head_scores is not None:
        heads.setdefault(head_task, (head_scores, head_proba))
    if head_only:
        # HEAD ONLY, AND ONLY WHERE THE PROBE IS NOT THE QUESTION. The probe is
        # what makes a supervised arm comparable to an SSL one, so dropping it
        # is never right for a comparison across methods. It IS right for a
        # sweep that reports the trained head and nothing else -- the
        # full-history campaigns and the variance decomposition -- where the
        # probe costs a whole extra month-panel per run: the probe-fit month is
        # embedded at TRAIN_ANCHORS_PER_DAY (36) against the eval month's 8, so
        # it is roughly 4.5x the eval panel's rows and the dominant cost of
        # scoring. No probe means no probe-fit panel to build, stage or embed.
        #
        # AUC goes with it: its bins are quantiles of the RAW target measured on
        # the probe-fit month, so there is nothing to bin against here.
        if not heads:
            raise ValueError("head_only=True but no head readout was given")
        names_ = eval_cache["target_names"].tolist()
        cells_ = np.char.add(np.char.add(eval_cache["date"], "@"),
                             eval_cache["anchor"].astype(str))
        out_: dict[str, dict] = {}
        for task, (scores_, _proba) in heads.items():
            if task not in names_:
                continue
            j_ = names_.index(task)
            y_ = eval_cache["z"][:, j_]
            ok_ = np.isfinite(y_) & np.isfinite(np.asarray(scores_))
            if ok_.sum() < 100:
                continue
            r_ = grouped_rank_ic_by_label(
                np.asarray(scores_)[ok_], y_[ok_], cells_[ok_])
            out_[f"head:{task}"] = {
                "ic": float(r_.mean), "se": float(r_.standard_error),
                "n_cells": int(r_.observations), "n_rows": int(ok_.sum())}
        return out_
    if train_cache is None:
        raise ValueError(
            "score() needs a train_cache: the reported IC is the ridge probe "
            "for every arm, so the probe-fit month must be embedded even for "
            "a supervised checkpoint."
        )

    names = eval_cache["target_names"].tolist()
    # One cell id per (date, anchor): the unit a cross-sectional rank
    # correlation is defined over.
    cells = np.char.add(np.char.add(eval_cache["date"], "@"),
                        eval_cache["anchor"].astype(str))

    sc = StandardScaler().fit(train_cache["X"])
    Xtr, Xev = sc.transform(train_cache["X"]), sc.transform(eval_cache["X"])

    def _ic(pred, y, g):
        result = grouped_rank_ic_by_label(pred, y, g)
        return {"ic": float(result.mean), "se": float(result.standard_error),
                "n_cells": int(result.observations),
                "n_rows": int(len(y))}

    raw_tr = raw_ev = None
    if train_stats is not None and eval_stats is not None:
        raw_tr = raw_targets(train_cache, train_stats)
        raw_ev = raw_targets(eval_cache, eval_stats)

    out = {}
    for j, name in enumerate(names):
        yev = eval_cache["z"][:, j]
        ok_ev = np.isfinite(yev)
        ytr = train_cache["z"][:, j]
        ok_tr = np.isfinite(ytr)
        if ok_tr.sum() < 100 or ok_ev.sum() < 100:
            continue
        probe = ColumnwiseRidge(
            alpha=ridge_alpha_for(name), min_samples=100,
        ).fit(train_cache["X"], ytr[:, None])
        prediction = probe.predict(eval_cache["X"])[ok_ev, 0]
        out[name] = _ic(prediction, yev[ok_ev], cells[ok_ev])

        # The AUC rows are a SUBSET of the IC rows: a row whose cell has a
        # usable sigma always inverts, so this drops nothing in practice, but
        # the masks are kept separate rather than assumed equal.
        want_auc = raw_tr is not None and (auc_tasks is None or name in auc_tasks)
        atr = aev = ltr = lev = None
        if want_auc:
            atr = ok_tr & np.isfinite(raw_tr[:, j])
            aev = ok_ev & np.isfinite(raw_ev[:, j])
            if atr.sum() < 100 or aev.sum() < 100:
                want_auc = False

        def _labels(k):
            """(fit, eval) hard bin labels at k, or (None, None) if unusable."""
            a, b = _hard_bins(raw_tr[atr, j], raw_ev[aev, j], k)
            if a is None or len(np.unique(b)) < 2:
                return None, None
            return a, b

        def _probe_auc(fit_labels, eval_labels):
            X = Xtr[atr]
            if auc_fit_rows and len(X) > auc_fit_rows:
                # Seeded from the row count alone, so the same panel always
                # draws the same subsample and two checkpoints scored on it
                # differ in their embeddings and nothing else.
                idx = np.random.RandomState(len(X)).choice(
                    len(X), auc_fit_rows, replace=False)
                X, fit_labels = X[idx], fit_labels[idx]
            clf = LogisticRegression(max_iter=1000, solver="lbfgs", tol=auc_tol)
            clf.fit(X, fit_labels)
            _probe_auc.n_iter = int(np.max(clf.n_iter_))
            return macro_ovr_auc(clf.predict_proba(Xev[aev]), eval_labels)

        if want_auc:
            ltr, lev = _labels(auc_bins)
        if ltr is not None:
            out[name]["auc"] = _probe_auc(ltr, lev)
            out[name]["auc_bins"] = int(auc_bins)

        # A HEADLESS arm — the random-init baseline above all — has no native
        # k of its own, but it is the subtrahend for arms that do. Without a
        # baseline at k=11 and k=21 the unprojected column of the figure has
        # nothing to subtract, so those k are fit here on request.
        for k_extra in auc_extra_bins:
            if not want_auc or int(k_extra) == int(auc_bins):
                continue
            ltr_x, lev_x = _labels(int(k_extra))
            if ltr_x is not None:
                out[name][f"auc_k{int(k_extra)}"] = _probe_auc(ltr_x, lev_x)

        # The head only speaks to its own target.
        h_scores, h_proba = heads.get(name, (None, None))
        if h_scores is not None:
            out[f"head:{name}"] = _ic(
                h_scores[ok_ev], yev[ok_ev], cells[ok_ev],
            )
            if h_proba is not None and ltr is not None:
                k_head = int(h_proba.shape[1])
                # TWO NUMBERS, DELIBERATELY. ``auc`` marginalizes the head onto
                # the reported partition so every cell of a figure whose axis
                # is k asks one question. ``auc_native`` is the unprojected
                # number on the head's OWN k — what it was actually trained to
                # separate, and the honest thing to show next to a projection
                # rather than instead of it. They are NOT comparable to each
                # other: a k=21 macro one-vs-rest AUC is mechanically nearer
                # 0.5 than a k=5 one, because a narrow middle bin is not
                # separable by any monotone score. Read auc ACROSS k and
                # auc_native only DOWN a column.
                pk = _project_proba(h_proba[aev], raw_tr[atr, j], auc_bins)
                if pk is not None:
                    out[f"head:{name}"]["auc"] = macro_ovr_auc(pk, lev)
                    out[f"head:{name}"]["auc_bins"] = int(auc_bins)
                    out[f"head:{name}"]["head_bins"] = k_head

                if k_head == auc_bins:
                    out[f"head:{name}"]["auc_native"] = out[f"head:{name}"]["auc"]
                    out[f"head:{name}"]["auc_native_bins"] = k_head
                    out[name]["auc_native"] = out[name]["auc"]
                    out[name]["auc_native_bins"] = k_head
                else:
                    ltr_n, lev_n = _labels(k_head)
                    if ltr_n is not None:
                        out[f"head:{name}"]["auc_native"] = macro_ovr_auc(
                            h_proba[aev], lev_n)
                        out[f"head:{name}"]["auc_native_bins"] = k_head
                        # The probe refit at the head's k, so the native
                        # column has its own reference. Without it a k=21
                        # head AUC has nothing to be better or worse than.
                        out[name]["auc_native"] = _probe_auc(ltr_n, lev_n)
                        out[name]["auc_native_bins"] = k_head
    return out


def main():
    machine = BLL01MachineConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--run-ids", nargs="+", required=True)
    p.add_argument("--train-month", required=True, help="YYYY-MM, fits the probe")
    p.add_argument("--eval-month", required=True, help="YYYY-MM, reports the IC")
    p.add_argument("--checkpoint-root", default="/data/lab/market-jepa-checkpoints")
    p.add_argument("--mosaic-dir", default=machine.mosaic_dir)
    p.add_argument("--xs-anchor-stats-dir",
                   default="/data/lab/market-jepa-mosaic/xs_anchor_stats_fwdvwap60")
    p.add_argument("--holiday-csv", default=machine.holiday_csv)
    p.add_argument("--anchors-per-day", type=int, default=8,
                   help="EVAL month; defines the reported cross-sections")
    p.add_argument("--train-anchors-per-day", type=int,
                   default=TRAIN_ANCHORS_PER_DAY,
                   help="probe-fit month; more rows, same decode cost")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--random-init-seeds", nargs="*", type=int, default=[],
                   help="also score untrained encoders at these seeds; the "
                        "subtrahend of delta IC. One baseline per distinct "
                        "architecture, embedded once and reused.")
    p.add_argument("--out", default=None, help="JSON path (default: alongside ckpt)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = MarketSchedule(args.holiday_csv)
    # The probe-fit month is embedded at MORE anchors than the eval month --
    # see TRAIN_ANCHORS_PER_DAY. Two separate grids, so they cannot be mixed up.
    anchors_by_month = {
        args.train_month: (day_anchors(args.train_anchors_per_day),
                           args.train_anchors_per_day),
        args.eval_month: (day_anchors(args.anchors_per_day),
                          args.anchors_per_day),
    }
    stats_dir = Path(args.xs_anchor_stats_dir)
    root = Path(args.checkpoint_root)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from market_jepa.eval.checkpoints import (
        architecture_signature, build_untrained_encoder, fetch_wandb_config,
        load_model,
    )

    results = {}
    archs: dict[str, dict] = {}
    for run_id in args.run_ids:
        run_cfg = _predict_readout(fetch_wandb_config(run_id))
        archs.setdefault(architecture_signature(run_cfg), run_cfg)
        model = load_model(str(root / run_id), run_cfg, device)
        model.eval()
        caches = {}
        for ym in (args.train_month, args.eval_month):
            y, m = ym.split("-")
            anchors_tod, n_anchors = anchors_by_month[ym]
            cache_dir = root / "xs_ic_cache" / ym
            cache_dir.mkdir(parents=True, exist_ok=True)
            # The anchor count is part of the cache IDENTITY: the same run and
            # month embedded at 8 vs 36 anchors are different panels, and a key
            # that omits it silently serves the wrong one.
            #
            # AND SO IS THE READOUT. run_id pins the weights and the month
            # pins the data, but the TOKEN they are read at is chosen here, at
            # scoring time, by _predict_readout -- so it is not implied by the
            # rest of the key. A run embedded at the mean and then re-scored
            # after PREDICT_POOL began reaching it would hit this cache and
            # silently return the mean-pooled embedding under the new label.
            # That is the same shape of bug as the one that made the readout a
            # no-op in the first place, and it costs a rebuild to avoid.
            path = cache_dir / f"{run_id}_p{_readout_tag(run_cfg)}_a{n_anchors}.npz"
            if path.is_file():
                caches[ym] = dict(np.load(path, allow_pickle=False))
                continue
            c = embed_month(
                model, Path(args.mosaic_dir) / y / m, ym,
                AnchorStats(stats_dir / f"{ym}.npz"), schedule,
                anchors_tod, device, args.batch_size,
            )
            np.savez_compressed(path, **c)
            caches[ym] = c

        # The shared readout, not a lookalike: this loop used to build its
        # own head_scores and so missed both the multihead case and the
        # random-head guard head_readout carries.
        results[run_id] = score(
            caches[args.train_month], caches[args.eval_month],
            head_readouts=head_readout(
                model, caches[args.eval_month]["X"], device),
        )
        for k, v in results[run_id].items():
            print(f"  {run_id} {k:26s} IC {v['ic']:+.4f} +- {v['se']:.4f} "
                  f"({v['n_cells']} cells)", flush=True)

    # --- random-init baselines: the subtrahend of delta IC ---
    #
    # Keyed by ARCHITECTURE, not by run: the floor is a property of the
    # encoder shape and the panel, so N runs sharing an architecture share one
    # baseline instead of paying N times to re-measure the same thing.
    for seed in args.random_init_seeds:
        for sig, cfg in sorted(archs.items()):
            key = f"random_init/{sig}/s{seed}"
            # THE FLOOR MUST MATCH THE PANEL IT IS SCORED ON, in architecture
            # and in input. Both halves were wrong: the encoder was built from
            # cfg, which records n_info_channels=0 even for checkpoints whose
            # weights carry an info_proj (pretrain.py takes the width from the
            # dataset), and embed_month was called without panel_kwargs, so
            # the floor saw a 9-channel panel while the models it floors see
            # 9 + 8 norm-stat + 3 window = 20. A floor read off a different
            # architecture on a different panel is not that model's floor.
            floor_kwargs = panel_kwargs_for(cfg)
            n_info = _info_channel_width(floor_kwargs)
            model = build_untrained_encoder(
                cfg, device, seed=seed, n_info_channels=n_info)
            caches = {}
            for ym in (args.train_month, args.eval_month):
                y, m = ym.split("-")
                anchors_tod, n_anchors = anchors_by_month[ym]
                cache_dir = root / "xs_ic_cache" / ym
                cache_dir.mkdir(parents=True, exist_ok=True)
                # _i{n_info} IS PART OF THE KEY. sig is architecture_signature,
                # which reads cfg["n_features"] -- and that records 9 even for a
                # checkpoint whose weights carry an info_proj, which is the very
                # thing the block above exists to correct. So moving the floor
                # from 9 to 20 channels changed nothing sig can see, and every
                # pre-2026-09-13 randinit_*.npz would have been reloaded as if
                # it were the new floor. Same reasoning as _p{readout} on the
                # run cache above.
                path = (cache_dir /
                        f"randinit_{sig}_i{n_info}_s{seed}_a{n_anchors}.npz")
                if path.is_file():
                    caches[ym] = dict(np.load(path, allow_pickle=False))
                    continue
                c = embed_month(
                    model, Path(args.mosaic_dir) / y / m, ym,
                    AnchorStats(stats_dir / f"{ym}.npz"), schedule,
                    anchors_tod, device, args.batch_size,
                    **floor_kwargs,
                )
                np.savez_compressed(path, **c)
                caches[ym] = c
            results[key] = score(caches[args.train_month], caches[args.eval_month])
            for k, v in results[key].items():
                print(f"  {key} {k:26s} IC {v['ic']:+.4f} +- {v['se']:.4f} "
                      f"({v['n_cells']} cells)", flush=True)

    out = Path(args.out) if args.out else root / f"xs_ic_{args.eval_month}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
