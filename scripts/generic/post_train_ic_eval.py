"""Score a just-trained checkpoint on the synchronized panel, in-job.

Runs at the tail of the SLURM job that produced the checkpoint, while the GPU
and the staged mosaic are still there. That is the whole point: the reported
metric needs a full-month cross-sectional panel, which is far too expensive to
recompute later for every checkpoint and far too expensive to run DURING
training on a node with 7 CPUs to one H100.

It replaces the in-training eval, not supplements it. Production sweeps run
``training.live_eval=false``, so the numbers here are the only ones logged --
and they are the better ones: a full month of synchronized cross-sections with
a per-cell standard error, rather than 4096 cached rows scored pooled.

Metrics land on the SAME wandb run as the training, under ``xs_ic/``.

The panel decode (dense-grid build + view cutting per ticker-day) dominates
the GPU forward and is embarrassingly parallel over MDS shards, so it runs
in a spawn pool by default (--decode-procs; measured 2026-08-27 before the
change: eval-phase jobs sat at ~1 CPU with the GPU at 0% for hours).

Usage (from slurm_train_bundle.sh, after training):
    uv run scripts/generic/post_train_ic_eval.py \
        --ckpt-dir /path/to/<run_id> \
        --train-month 2015-04 --eval-month 2015-05 \
        --wandb-run-id <id> --wandb-project <project> \
        --mosaic-dir "$DATA_DIR/1Hz_mosaic_mnth" \
        --xs-anchor-stats-dir "$DATA_DIR/xs_anchor_stats"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/eval"))

from stable_finance.dataset import MarketSchedule  # noqa: E402
from market_jepa.eval.checkpoints import load_model  # noqa: E402
from market_jepa.schemas import BLL01MachineConfig  # noqa: E402
from stable_finance.dataset import AnchorTargetStats as AnchorStats  # noqa: E402
from xs_ic_eval import (  # noqa: E402
    GLOBAL_SEQ_LEN, TRAIN_ANCHORS_PER_DAY, day_anchors, embed_month,
    head_readout, panel_kwargs_for, score,
)


# Bumped whenever the AUC's labelling changes. 2 = equal-count bins of the
# RAW target for both readouts, the head marginalized onto the same partition.
AUC_SCHEMA = 2

# THE HEAD THIS SCORER WRITES IS A REAL ONE, and it has to say so. Mirrors
# merge_score_results.HEAD_SCHEMA: a file with xs_ic/head:* keys and no stamp
# is assumed to predate the 2026-08-29 multihead loader fix, when load_model
# looked for head.pt where a trunk saves heads.pt and every "head" number was
# an untrained readout. Readers therefore require the stamp, and until now only
# the offline merge wrote it -- so every head scored IN THE JOB, by a scorer
# long past that fix, was refused as stale. The full-history multihead figure
# read as empty for exactly this reason on 2026-09-14.
HEAD_SCHEMA = 2


def parse_args():
    machine = BLL01MachineConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--train-month", required=True, help="YYYY-MM, fits the probe")
    p.add_argument("--eval-month", required=True, help="YYYY-MM, reports the IC")
    p.add_argument("--wandb-run-id", default=None)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default="boothai")
    p.add_argument("--mosaic-dir", default=machine.mosaic_dir)
    p.add_argument("--xs-anchor-stats-dir",
                   default="/data/lab/market-jepa-mosaic/xs_anchor_stats")
    p.add_argument("--risk-factor-dir", default=None,
                   help="overrides the machine default; the SLURM job "
                        "passes its staged copy")
    p.add_argument("--holiday-csv", default=machine.holiday_csv)
    p.add_argument("--anchors-per-day", type=int, default=8,
                   help="EVAL month; defines the reported cross-sections")
    p.add_argument("--train-anchors-per-day", type=int,
                   default=TRAIN_ANCHORS_PER_DAY,
                   help="probe-fit month; more rows, same decode cost")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--decode-procs", type=int, default=0,
                   help="processes for the sharded panel decode; 0 = auto "
                        "(job CPUs - 2, capped at 12), 1 = the old serial "
                        "path. The decode is the cost, not the forward -- "
                        "one process leaves the job's other cores and the "
                        "GPU idle for hours (measured 2026-08-27: three "
                        "eval-phase jobs at ~1 CPU each, 7 of 8 GPUs at 0 percent).")
    p.add_argument("--auc-bins", type=int, default=5,
                   help="the AUC's reported partition, on the RAW target "
                        "(the historical metric used 5). The head's softmax "
                        "is marginalized onto it from whatever k it trained "
                        "at, so probe and head are the same question.")
    p.add_argument("--config-json", default=None,
                   help="training config; defaults to <ckpt-dir>/train_meta.json")
    p.add_argument("--head-only", action="store_true",
                   help="score the trained HEAD and skip the ridge probe. The "
                        "probe-fit month is then never embedded -- it is 36 "
                        "anchors/day against the eval month's 8, so it is the "
                        "dominant cost of scoring and its panel need not exist "
                        "at all. Only for sweeps that report the head: the "
                        "probe is what makes a supervised arm comparable to an "
                        "SSL one.")
    p.add_argument("--no-wandb", action="store_true",
                   help="write xs_ic.json beside the checkpoint and stop; "
                        "for offline/local runs that have no W&B row to "
                        "resume. --wandb-run-id/--wandb-project are then "
                        "ignored.")
    return p.parse_args()


def _load_cfg(ckpt_dir: Path, explicit: str | None) -> dict:
    """The config load_model needs, from disk rather than from wandb.

    A compute node may have no wandb API access and certainly should not spend
    a scan over every project to recover something the checkpoint sits next to.

    save_train_meta writes ``task`` at the TOP level, not under ``mode``, so it
    has to be lifted explicitly. Getting this wrong is silent and severe: the
    head is a scalar regardless of task, so a mislabelled model loads happily
    and then scores against the wrong target column under the wrong metric
    name.
    """
    path = Path(explicit) if explicit else ckpt_dir / "train_meta.json"
    meta = json.loads(path.read_text()) if path.is_file() else {}
    cfg = meta.get("config", meta)
    # A REAL backbone block, not merely a present one. `backbone: {}` appears
    # in configs whose mode builds its own encoder, and short-circuiting onto
    # it returns a config with no _target_ and no pool -- worse than
    # synthesizing one below, which at least supplies the defaults every run of
    # that era used.
    bb = cfg.get("backbone") if isinstance(cfg, dict) else None
    if isinstance(bb, dict) and bb.get("_target_") and "mode" in cfg:
        return cfg

    task = meta.get("task") or cfg.get("task")
    if (ckpt_dir / "head.pt").exists() and not task:
        raise SystemExit(
            f"{ckpt_dir} has a trained head but no task recorded in "
            f"{path.name}; refusing to guess which target it predicts."
        )
    # THE INFORMATION TOKEN's two contributions, 8 = 2 stats x 4 normalization
    # groups and 3 window descriptors. They are WIDTHS, not panel knobs: miss
    # one and the backbone is built too narrow, the checkpoint refuses to load,
    # and the run reports nothing rather than reporting a wrong number.
    #
    # BOTH KEY SPELLINGS. norm_stats_channels/time_info are what save_train_meta
    # wrote before the 2026-08-25 rename, and a checkpoint is a historical
    # record -- it has to score the way it trained.
    info_norm_stats = bool(meta.get("info_norm_stats",
                                    meta.get("norm_stats_channels", False)))
    info_window = bool(meta.get("info_window", meta.get("time_info", False)))
    n_features = (9 + (8 if info_norm_stats else 0) + (3 if info_window else 0)
                  + len(meta.get("risk_factor_tickers", []) or [])
                  * len(_rf_columns(meta)))
    # Shape info only — the repo default for every ViT run in these sweeps.
    # n_features MUST include the risk-factor channels or the backbone is built
    # with the wrong input width and the checkpoint load fails (or, worse,
    # partially succeeds).
    # The panel knobs travel in the synthesized `dataset` block so
    # panel_kwargs_for has ONE input shape to understand, whether the meta
    # carried a whole hydra config or the flat schema save_train_meta writes.
    # A checkpoint from before these were recorded yields the defaults, which
    # is right: it was trained on the default panel.
    dataset = {"norm_mode": meta.get("norm_mode", "per_view"),
               "info_norm_stats": info_norm_stats,
               "info_window": info_window}
    aug0 = {}
    if meta.get("global_agg_range"):
        aug0["global_agg_range"] = list(meta["global_agg_range"])
    if meta.get("global_seq_len"):
        aug0["global_seq_len"] = int(meta["global_seq_len"])
    if aug0:
        dataset["augmentations"] = {"0": aug0}
    # A NON-ViT BACKBONE CARRIES ITS OWN IDENTITY. Everything below assumes a
    # ViT-384 because that is all this project had trained until 2026-08-22;
    # a resnet checkpoint rebuilt as a transformer does not mis-score, it
    # fails to load. save_train_meta records the target only when it is NOT
    # the ViT, so checkpoints from before this keep resolving unchanged.
    if meta.get("backbone_target"):
        bb = {"_target_": meta["backbone_target"], "config": {}}
        for k in ("d_embedding", "variant", "pool"):
            if meta.get(f"backbone_{k}") is not None:
                bb[k] = meta[f"backbone_{k}"]
        return {"backbone": bb, "mode": {"task": task} if task else {},
                "n_features": n_features, "dataset": dataset}

    # pool is NOT hardcodable: the backbone has a cls_token and a 2049-row
    # positional embedding under "cls" and neither under "last", so guessing
    # wrong does not mis-score, it fails to load.
    return {"backbone": {"_target_": "market_jepa.modeling.transformer.TransformerBackbone",
                         "d_embedding": 384, "pool": meta.get("pool", "cls"),
                         "causal": bool(meta.get("causal", False)),
                         "state_token": bool(meta.get("state_token", False)),
                         "diff_channels": bool(meta.get("diff_channels", False)),
                         # The INNER TransformerConfig. Empty for every run
                         # that left it at defaults; a knob that changes the
                         # parameter set or the geometry has to be carried
                         # here or the backbone is rebuilt without it and
                         # scores a different model. See save_train_meta.
                         "config": {k: meta[k] for k in
                                    ("pos_embed", "cls_pos", "pos_init_std")
                                    if meta.get(k) is not None}},
            "mode": {"task": task} if task else {}, "n_features": n_features,
            "dataset": dataset}


def _auto_procs() -> int:
    """Decode processes when --decode-procs 0: the job's CPUs minus headroom.

    SLURM_CPUS_PER_TASK is the job's real budget; os.cpu_count() on a shared
    node reports the whole box (208 on the H100 rental) and must not win.
    """
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0) or 0) \
        or (os.cpu_count() or 1)
    return max(1, min(12, cpus - 2))


# Per-process state for the sharded decode, set once by the pool initializer.
_WORKER: dict = {}


def _panel_is_cached(ym: str, anchors_tod, stats_dir: Path, panel_kw: dict,
                     rf_merger, seq_len: int) -> bool:
    """Is this month's panel already on disk for the key the scorer will ask for?

    WHY THIS GATES THE SHARDING. iter_panel_cached consults the cache only
    when ``num_shards == 1`` -- a shard is a slice of a month and the cache
    stores whole months -- and this script shards by default
    (``_auto_procs()`` = CPUs - 2, so 5 on pythia's 7). The two features were
    therefore mutually exclusive: with the pool on, a built cache could never
    be read, and every arm re-decoded a panel that is a function of the MONTH
    and not of the checkpoint. ~3028 s per arm, nine arms to a job.

    A hit makes the decode unnecessary rather than faster -- the cached path
    memmaps views.raw -- so the pool is worth nothing exactly when this is
    true, and worth everything when it is false. Hence: per month, not per
    job. Any failure to answer is answered NO, because a wrong yes would send
    a month down the serial path to a cache that then misses and rebuild it
    one process wide.
    """
    try:
        import panel_cache as pc
        root = pc.cache_root()
        if root is None:
            return False
        key = pc.panel_key(
            anchors_per_day=len(anchors_tod), stats_tag=stats_dir.name,
            has_rf=rf_merger is not None, norm_groups=panel_kw["norm_groups"],
            fixed_agg=panel_kw["fixed_agg"], seq_len=seq_len)
        return bool(pc.is_built(root, ym, key))
    except Exception as error:  # noqa: BLE001 - never fail a run over a cache
        print(f"==> panel cache probe failed for {ym}: {error}", flush=True)
        return False


def _decode_worker_init(ckpt_dir: str, config_json: str | None,
                        risk_factor_dir: str | None, holiday_csv: str) -> None:
    """Runs once per pool process (spawn context): its own model + CUDA ctx.

    Each worker re-loads the checkpoint rather than receiving a pickled
    model — a ViT-384 load is seconds, and spawn + fresh CUDA context is the
    only arrangement that is safe with a GPU in the parent.
    """
    if risk_factor_dir:
        os.environ["MARKET_JEPA_RF_DIR"] = risk_factor_dir
    ckpt = Path(ckpt_dir)
    meta_path = Path(config_json) if config_json else ckpt / "train_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    cfg = _load_cfg(ckpt, config_json)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(str(ckpt), cfg, device)
    model.eval()
    _WORKER.update(
        model=model, device=device, panel_kw=panel_kwargs_for(cfg),
        rf_merger=_build_rf_merger(meta),
        schedule=MarketSchedule(holiday_csv),
    )


def _decode_worker_run(month_dir: str, ym: str, stats_path: str,
                       anchors_tod, batch_size: int,
                       shard_index: int, num_shards: int):
    """One shard of one month; embed_month partitions on MDS shards."""
    stats = AnchorStats(Path(stats_path))
    return embed_month(
        _WORKER["model"], Path(month_dir), ym, stats, _WORKER["schedule"],
        anchors_tod, _WORKER["device"], batch_size,
        shard_index=shard_index, num_shards=num_shards, allow_empty=True,
        rf_merger=_WORKER["rf_merger"], **_WORKER["panel_kw"],
    )


def _merge_shards(parts: list, ym: str) -> dict:
    """Concatenate per-shard panels. shard_index :: num_shards is a
    partition of the month's MDS shards, so the merged row set equals the
    serial pass's; per-row floats can differ at bf16-kernel level because
    batch boundaries fall differently, which scoring does not resolve."""
    parts = [p for p in parts if p is not None]
    if not parts:
        raise RuntimeError(f"{ym}: the panel is empty — check the anchor tables")
    out = {"target_names": parts[0]["target_names"]}
    for k in ("X", "z", "date", "anchor", "ticker"):
        out[k] = np.concatenate([p[k] for p in parts])
    return out


def _rf_columns(meta: dict) -> list[str]:
    """Resolve the risk-factor column list a checkpoint was trained with."""
    from market_jepa.training.streaming_dataset import _RF_PRESETS, FEATURE_COLUMNS

    spec = meta.get("risk_factor_columns")
    if isinstance(spec, str):
        return list(_RF_PRESETS[spec])
    return list(spec or FEATURE_COLUMNS)


def _build_rf_merger(meta: dict):
    """Rebuild the training-time risk-factor merger, or None if unused.

    Without this the eval panel would be 9 channels wide while the model
    expects 9 + n_rf — the reason IWM runs could not be scored at all.
    """
    tickers = meta.get("risk_factor_tickers") or []
    if not tickers:
        return None
    from market_jepa.training.risk_factors import RiskFactorMerger
    from market_jepa.training.streaming_dataset import _AGG_RULES, FEATURE_COLUMNS

    machine = BLL01MachineConfig()
    rf_dir = os.environ.get("MARKET_JEPA_RF_DIR", machine.risk_factor_dir)
    print(f"==> risk factors {tickers} cols={_rf_columns(meta)} from {rf_dir}",
          flush=True)
    return RiskFactorMerger(rf_dir, list(tickers), _rf_columns(meta),
                            FEATURE_COLUMNS, _AGG_RULES)


def main():
    args = parse_args()
    if args.risk_factor_dir:
        os.environ["MARKET_JEPA_RF_DIR"] = args.risk_factor_dir
    ckpt_dir = Path(args.ckpt_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = MarketSchedule(args.holiday_csv)
    # The probe-fit month gets more anchors than the eval month: the dense grid
    # is decoded once per ticker-day and reused across anchors, so extra rows
    # for the ridge are nearly free, while the eval month's 8 anchors define
    # the reported cross-sections and must not move. See TRAIN_ANCHORS_PER_DAY.
    anchors_by_month = {args.eval_month: day_anchors(args.anchors_per_day)}
    if not args.head_only:
        anchors_by_month[args.train_month] = day_anchors(args.train_anchors_per_day)
    stats_dir = Path(args.xs_anchor_stats_dir)
    mosaic = Path(args.mosaic_dir)

    meta_path = (Path(args.config_json) if args.config_json
                 else ckpt_dir / "train_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    rf_merger = _build_rf_merger(meta)

    cfg = _load_cfg(ckpt_dir, args.config_json)
    model = load_model(str(ckpt_dir), cfg, device)
    model.eval()
    if hasattr(model, "task_spec"):
        print(f"==> supervised head, task={model.task_spec.name}", flush=True)
    elif getattr(model, "task_specs", None):
        print(f"==> supervised multihead, tasks="
              f"{list(model.task_specs)}", flush=True)

    # EVERY arm fits the ridge, so every arm embeds the probe-fit month. A
    # supervised checkpoint additionally has a trained head, which is scored
    # alongside the probe rather than instead of it -- the reported number
    # stays the probe for both arms (see xs_ic_eval.score).
    # ASK THE READOUT, not the attribute names: a multihead carries
    # ``heads``/``task_specs`` and a random head carries ``head`` but nothing
    # worth scoring, and both used to answer this question wrong. Cheap --
    # head_readout runs again below on the eval embeddings, this only needs to
    # know whether there is anything to run.
    has_head = bool(
        getattr(model, "task_specs", None)
        or (hasattr(model, "task_spec")
            and getattr(model, "head", None) is not None
            and not getattr(model, "_head_is_random", False))
    )
    months = ([args.eval_month] if args.head_only
              else [args.train_month, args.eval_month])
    if args.head_only and not has_head:
        raise SystemExit(
            f"{ckpt_dir}: --head-only, but this checkpoint has no trained head "
            f"to score. Drop --head-only, or do not score this arm.")
    if has_head:
        print("==> head checkpoint: scoring head readout alongside the probe",
              flush=True)

    panel_kw = panel_kwargs_for(cfg)
    # WHAT COUNTS AS NON-DEFAULT FLIPPED on 2026-08-25: the information token
    # is on by default now, so its ABSENCE is the thing worth announcing. A
    # test that still fired on its presence would print for every run and mean
    # nothing.
    if (panel_kw["norm_groups"] == [] or panel_kw["fixed_agg"] is not None
            or not panel_kw["info_norm_stats"] or not panel_kw["info_window"]):
        # An arm scored on a non-default panel must SAY so: the number lands
        # in the same wandb key as every other run's and is not comparable to
        # them cell for cell.
        print(f"==> non-default panel: norm_mode="
              f"{'none' if panel_kw['norm_groups'] == [] else 'per_view'}, "
              f"fixed_agg={panel_kw['fixed_agg']}, "
              f"info_norm_stats={panel_kw['info_norm_stats']}, "
              f"info_window={panel_kw['info_window']}", flush=True)

    procs = args.decode_procs or _auto_procs()
    # Probed BEFORE the pool exists: each worker spawns a fresh CUDA context
    # and re-loads the checkpoint, so a job whose months are all cached should
    # not pay for five of them to sit idle.
    cached_months = {
        ym: _panel_is_cached(ym, anchors_by_month[ym], stats_dir, panel_kw,
                             rf_merger, GLOBAL_SEQ_LEN)
        for ym in months
    }
    if any(cached_months.values()):
        hit = ", ".join(ym for ym, v in cached_months.items() if v)
        print(f"==> panel cache: serving {hit} from disk (no decode)",
              flush=True)
    pool = None
    if procs > 1 and not all(cached_months.values()):
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(
            procs, initializer=_decode_worker_init,
            initargs=(str(ckpt_dir), args.config_json,
                      args.risk_factor_dir, args.holiday_csv))

    caches, stats_by_month = {}, {}
    try:
        for ym in months:
            y, m = ym.split("-")
            _procs_here = 1 if cached_months.get(ym) else procs
            print(f"==> embedding {ym} (decode procs: {_procs_here}"
                  f"{', panel cache HIT' if cached_months.get(ym) else ''}) ...",
                  flush=True)
            # Kept, not discarded after embedding: the AUC needs the same
            # tables to undo the z-score and recover the raw target its bins
            # live on.
            stats_by_month[ym] = AnchorStats(stats_dir / f"{ym}.npz")
            if pool is not None and not cached_months.get(ym):
                parts = pool.starmap(_decode_worker_run, [
                    (str(mosaic / y / m), ym, str(stats_dir / f"{ym}.npz"),
                     anchors_by_month[ym], args.batch_size, i, procs)
                    for i in range(procs)])
                caches[ym] = _merge_shards(parts, ym)
            else:
                caches[ym] = embed_month(
                    model, mosaic / y / m, ym, stats_by_month[ym],
                    schedule, anchors_by_month[ym], device, args.batch_size,
                    rf_merger=rf_merger, **panel_kwargs_for(cfg),
                )
            print(f"    {len(caches[ym]['X'])} rows", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    tr, ev = caches.get(args.train_month), caches[args.eval_month]
    train_stats = stats_by_month.get(args.train_month)

    # Both arms are REPORTED on the ridge probe fit here; the supervised head
    # is an extra column, not a substitution. The scoring itself is not
    # reimplemented — xs_ic_eval.score is the single scorer both arms call,
    # the SSL arm offline through xs_ic_eval.main and the supervised arm here.
    # Two lookalike implementations would silently drift and the comparison
    # would stop meaning anything.
    head_reads = head_readout(model, ev["X"], device)

    results = score(tr, ev, head_readouts=head_reads,
                    auc_bins=args.auc_bins,
                    train_stats=train_stats,
                    eval_stats=stats_by_month[args.eval_month],
                    head_only=args.head_only)

    metrics: dict[str, float] = {}
    for name, v in results.items():
        metrics[f"xs_ic/{name}"] = v["ic"]
        metrics[f"xs_ic_se/{name}"] = v["se"]
        metrics[f"xs_ic_cells/{name}"] = v["n_cells"]
        metrics[f"xs_ic_rows/{name}"] = v["n_rows"]
        auc_txt = ""
        if "auc" in v:
            metrics[f"xs_auc/{name}"] = v["auc"]
            metrics[f"xs_auc_bins/{name}"] = v["auc_bins"]
            # AUC_SCHEMA marks the labelling, and readers must check it: v1
            # (2026-08-20, never used for a reported figure) binned the
            # Z-SCORE, which mispairs the head's classes with its labels.
            metrics["xs_auc_schema"] = AUC_SCHEMA
            auc_txt = f"  AUC {v['auc']:.4f} (k={v['auc_bins']})"
            if "head_bins" in v:
                metrics[f"xs_auc_head_bins/{name}"] = v["head_bins"]
                auc_txt += f" <- head k={v['head_bins']}"
        if "auc_native" in v:
            metrics[f"xs_auc_native/{name}"] = v["auc_native"]
            metrics[f"xs_auc_native_bins/{name}"] = v["auc_native_bins"]
            if v["auc_native_bins"] != v.get("auc_bins"):
                auc_txt += (f"  native {v['auc_native']:.4f} "
                            f"(k={v['auc_native_bins']})")
        print(f"  {name:26s} IC {v['ic']:+.4f} +- {v['se']:.4f} "
              f"({v['n_cells']} cells){auc_txt}", flush=True)

    # WHICH TARGET THIS SCORE IS OF. The anchor tables ARE the target -- all
    # three became forward-window differences on 2026-08-22
    # (docs/return_bad_calculation.md) -- and xs_ic.json is otherwise identical
    # whichever family produced it. Without this a tree scored across the
    # change cannot be separated after the fact, which is exactly the state the
    # 1269 pre-existing files are in. Same key xs_score_many.py writes, so the
    # in-job hook and the batch re-scorer stay readable by one filter.
    # Stamp only when a head was actually scored: on an SSL arm there is no
    # head and the claim would be empty.
    if any(name.startswith("head:") for name in results):
        metrics["xs_head_schema"] = HEAD_SCHEMA
    metrics["xs_anchor_stats"] = (
        str(stats_dir).rstrip("/").rsplit("/", 1)[-1] if metrics else None)
    (ckpt_dir / "xs_ic.json").write_text(json.dumps(metrics, indent=2))

    if metrics and not args.no_wandb:
        if not (args.wandb_run_id and args.wandb_project):
            raise SystemExit("--wandb-run-id and --wandb-project are required "
                             "unless --no-wandb is given")
        import wandb
        from wandb.sdk.wandb_settings import Settings

        wandb.init(
            settings=Settings(mode="shared"),
            id=args.wandb_run_id, resume="must",
            project=args.wandb_project, entity=args.wandb_entity,
        )
        wandb.log(metrics)
        wandb.finish()
        print(f"Logged {len(metrics)} xs_ic metrics to {args.wandb_project}")
    elif metrics:
        print(f"Wrote {len(metrics)} xs_ic metrics to {ckpt_dir / 'xs_ic.json'} "
              "(--no-wandb)", flush=True)
    else:
        print("No xs_ic metrics produced", flush=True)


if __name__ == "__main__":
    main()
