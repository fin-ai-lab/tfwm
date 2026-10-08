"""Where does the SSL probe stop improving with more fitting data?

The reported metric fits a ridge on the train month's embeddings (~80k rows at
8 anchors/day) and scores the eval month. plots/probe_hpo/train_size.json shows
that curve is still climbing steeply at 100k, so the probe arm is fit-size
limited and its IC is understated by an unknown amount.

That matters because of what it is being compared against: a supervised head
trains on ~1.15M observations (128 x 9000 steps). Fitting its counterpart on
80k rows is a 14x data asymmetry in a comparison whose whole design is that
ONLY the predictor step differs. This sweep pushes the probe panel to a
comparable budget and finds the plateau.

Getting there: the anchor grid admits at most 36 anchors/day inside the
feasible band (a 2048-token view at 6 s/token spans 3.4 h, so nothing can end
before ~12:55), which is ~358k rows/month. Three probe-train months therefore
reach ~1.07M, matching the supervised budget. Those months are taken BEFORE the
encoder's training month so nothing leaks into the eval month.

The eval panel stays at the reported protocol (8 anchors/day, the month after
training) — this changes what the probe is FIT on, never what it is scored on.

Two phases:
  embed   one process per (checkpoint, month, shard); writes a partial .npz
  reduce  concatenates, then fits ridge at each n and reports IC per task

Usage:
    # one worker
    probe_fit_size.py embed --ckpt <dir> --month 2008-01 --shard-index 3 \
        --num-shards 32 --anchors-per-day 36 --out-dir <cache>
    # after all workers
    probe_fit_size.py reduce --out-dir <cache> --json results.json
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
sys.path.insert(0, str(ROOT / "scripts/generic"))

from stable_finance.dataset import MarketSchedule  # noqa: E402
from market_jepa.eval.checkpoints import load_model  # noqa: E402
from stable_finance import ColumnwiseRidge, grouped_rank_ic_by_label  # noqa: E402
from market_jepa.schemas import LocalMachineConfig  # noqa: E402
from stable_finance.dataset import AnchorTargetStats as AnchorStats  # noqa: E402
from xs_ic_eval import (  # noqa: E402
    RIDGE_ALPHA, day_anchors, embed_month, panel_kwargs_for,
)
# The config-from-disk resolution lives with the in-job hook, and is imported
# rather than reimplemented for the reason xs_score_many gives: one definition
# of "what this checkpoint is". See _panel_for below.
from post_train_ic_eval import _load_cfg, _panel_is_cached  # noqa: E402

TASKS = ["return_900", "volatility_change_900", "spread_change_900"]
# Log-spaced, ending at the supervised budget (128 x 9000 = 1,152,000).
FIT_SIZES = [2048, 8192, 32768, 131072, 393216, 786432, 1152000]


# THE READOUT THIS SWEEP EMBEDS AT. Prediction is scored at the LAST token for
# every arm (xs_ic_eval.PREDICT_POOL); the latent suite reads everything at the
# mean (panel_lib.LATENT_POOL). This sweep had NEITHER: it loaded each
# checkpoint with its own config, so the 14 SSL/LeJEPA arms came back mean-
# pooled and only the supervised arms and the random-init floor were read at
# last -- a floor read at a different token than the arm it is subtracted from.
#
# IT CANNOT BE DONE THROUGH THE CONFIG. For a checkpoint that ships
# config.json + model.pt, load_model dispatches to <class>.from_pretrained(),
# which RE-READS config.json off disk and discards the cfg dict it was handed.
# Rewriting `pool` in that dict -- which is exactly what
# xs_ic_eval._predict_readout does -- changes the embedding by 0.0. Setting it
# on the built backbone is bit-identical to a checkpoint that genuinely carries
# the pool, and is safe because pooling is a readout applied after the last
# block: no weight depends on it.
def _apply_readout(model, pool):
    """Set ``model``'s backbone readout. Returns the readout actually used."""
    if not pool:
        bb = getattr(model, "backbone", None)
        return str(getattr(bb, "pool", "none"))
    # EVERY SUB-BACKBONE, NOT JUST .backbone. Measured 2026-09-17 on 2008-04:
    #   TimeMAE   inherits base.encode() -> .backbone            diff 2.75
    #   TF-C      concat(z_t, z_f)       -> + .freq_backbone     diff 1.60
    #   TS2Vec    encode() uses the SWA copy -> + .swa_backbone  diff 0.71
    # Setting only .backbone moved TS2Vec and TF-C by 0.0, which is what made
    # them look unswitchable. They are not; the pool simply lived on a second
    # module the setter never reached.
    #
    # CoST IS THE ONE REAL EXEMPTION. Its encode() takes
    # cat(trend[:, -1], season[:, -1]) straight off backbone.forward_patches(),
    # so it never consults .pool at all -- and it is already reading the LAST
    # valid patch, which is the readout this flag exists to impose.
    # A FROZEN TSFM HAS NO .backbone -- its readout token is its own
    # time_pool, applied to the captured hidden states, so that is what the
    # flag sets. Without this branch it fell through to "encode" and every
    # TSFM row was stamped with a readout no consumer accepts.
    if type(model).__name__ == "PretrainedTSFM":
        model.time_pool = pool
        return str(pool)
    if type(model).__name__ == "CoST":
        print(f"    !! --pool {pool} not applicable: CoST reads "
              f"cat(trend[:,-1], season[:,-1]) off forward_patches() -- it "
              f"never consults backbone.pool, and is already at the last "
              f"patch.", flush=True)
        return "last-by-design"
    touched = []
    for name in ("backbone", "swa_backbone", "freq_backbone"):
        bb = getattr(model, name, None)
        if bb is not None and hasattr(bb, "pool"):
            bb.pool = pool
            touched.append(name)
    if not touched:
        print(f"    !! --pool {pool} ignored: {type(model).__name__} exposes no "
              f"poolable backbone", flush=True)
        return "encode"
    return str(pool)


def _readout_tag_of(d) -> str:
    """The readout an open shard was embedded at.

    Shards written before this stamp existed carry the CHECKPOINT'S OWN pool,
    which is what they were embedded at, so their absence is not staleness.
    """
    return str(d["readout"]) if "readout" in d else ""


def _stats_name(p) -> str:
    """Basename of an anchor-stat directory — the TARGET DEFINITION's name."""
    return str(p).rstrip("/").rsplit("/", 1)[-1]


def _stats_tag_of(d) -> str:
    """The anchor-stat stamp on an open shard, defaulting to the retired family."""
    return str(d["xs_anchor_stats"]) if "xs_anchor_stats" in d else "xs_anchor_stats"


def _num_shards_of(d) -> int | None:
    """The fan-out an open shard was cut at; None for shards predating the stamp."""
    return int(d["num_shards"]) if "num_shards" in d else None


def _shard_is_current(dest: Path, stats_name: str, cover: Path | None = None,
                      readout: str = "", num_shards: int | None = None) -> bool:
    """True if an existing shard was embedded against ``stats_name``.

    A cached shard holds the z-scored TARGET, so it is a function of which
    anchor tables built it -- and the cache key (checkpoint, month, anchors,
    shard) does not mention them. Re-running the same task against a different
    target family would otherwise skip every shard and reduce stale labels into
    a fresh-looking result. Shards written before this stamp existed are
    treated as the retired mid-to-mid family, which is what they are.

    COLLAPSED SHARDS ARE ONLY CURRENT IF THEIR COVER IS. When the panel cache
    holds the month, shard 0 embeds it whole and shards 1..N-1 write a marker
    instead of a slice. That marker was byte-identical to the one a GENUINELY
    empty slice writes, and this function only read the stats tag -- so if such
    a run died after the markers but before shard 0 finished, a later UNCACHED
    run embedded shard 0 as a real 1/N slice, skipped every marker as current,
    and reduced the month holding 1/N of its rows with nothing raising. A
    collapsed marker therefore names its cover (shard 0) and is current only
    while that cover is itself a whole-month embed against the same tables.

    THE FILENAME DOES NOT SAY HOW MANY SHARDS THE MONTH WAS CUT INTO. Shard 3
    of 4 and shard 3 of 32 are both ``..._003.npz``, so a 4-shard pass (the
    examples' --smoke) followed by a 32-shard pass in the same cache skipped
    shards 0-3 as current and reduced a panel of 4/4 + 28/32 of the month:
    58,651 eval rows instead of 31,850, most of them duplicates, scored with
    nothing raising whenever no other checkpoint's count exposed it. A shard
    stamped with a different ``num_shards`` is therefore stale. Unstamped
    shards predate the stamp and are taken as current.
    """
    try:
        with np.load(dest, allow_pickle=False) as d:
            if _stats_tag_of(d) != stats_name:
                return False
            got = _readout_tag_of(d)
            if readout and got and got != readout:
                return False
            got_n = _num_shards_of(d)
            if num_shards and got_n and got_n != num_shards:
                return False
            if "collapsed" not in d:
                return True
    except Exception:                                  # truncated / unreadable
        return False
    if cover is None or not cover.is_file():
        return False
    try:
        with np.load(cover, allow_pickle=False) as d0:
            return "full_month" in d0 and _stats_tag_of(d0) == stats_name
    except Exception:
        return False


def _panel_for(ckpt: Path) -> tuple[dict, dict]:
    """``(cfg, panel_kwargs)`` for one checkpoint.

    THIS USED TO SYNTHESIZE A ViT-384 AND DISCARD EVERYTHING ELSE. It returned
    a hardcoded backbone with 9 input features and no dataset block at all, so:

      * a checkpoint trained with info_norm_stats has 17 input columns and
        could not load here;
      * a non-ViT backbone was rebuilt as a ViT;
      * and, worst because it is silent, every panel knob was dropped. A run
        pinned to one resolution, or to a shorter view, was embedded on the
        DEFAULT 2048-token free-agg panel -- scored on views it never trained
        on, with no error and a plausible number out the other end.

    The in-job hook and xs_score_many both resolve this from ``_load_cfg`` and
    then hand ``panel_kwargs_for`` to the panel builder. This does the same, so
    the cluster scoring path and the local one cannot disagree about what a
    checkpoint is or what it should be shown.
    """
    cfg = _load_cfg(ckpt, None)
    panel = panel_kwargs_for(cfg)
    # DELIBERATE CROSS-EVALUATION ONLY. Everything above exists to stop a
    # checkpoint being scored on a panel it never trained on, so an override
    # has to be explicit, loud, and impossible to set by accident -- it is a
    # different measurement, not a tuning knob. It exists for one question:
    # does a full-view model recover the shortview arm's IC when it is SHOWN a
    # short view? That separates "the weights are better" from "the input is
    # better", and neither the config nor the checkpoint can answer it.
    forced = os.environ.get("MJ_FORCE_SEQ_LEN")
    if forced:
        print(f"    !! MJ_FORCE_SEQ_LEN={forced}: overriding this checkpoint's "
              f"seq_len {panel['seq_len']} -> {int(forced)}. CROSS-EVALUATION; "
              f"the result is NOT this checkpoint's reported score.", flush=True)
        panel["seq_len"] = int(forced)
    return cfg, panel


def cmd_embed(args):
    machine = LocalMachineConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = MarketSchedule(args.holiday_csv or machine.holiday_csv)
    ckpt = Path(args.ckpt)
    out = Path(args.out_dir) / ckpt.name
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{args.month}_a{args.anchors_per_day}_{args.shard_index:03d}.npz"
    stats_name = _stats_name(args.xs_anchor_stats_dir)
    if dest.is_file() and not args.overwrite:
        if _shard_is_current(dest, stats_name, readout=args.pool or "",
                             num_shards=args.num_shards):
            print(f"exists, skipping: {dest}")
            return
        print(f"re-embedding {dest.name}: cached against a different target")

    y, m = args.month.split("-")
    cfg, panel = _panel_for(ckpt)
    model = load_model(str(ckpt), cfg, device)
    model.eval()
    readout = _apply_readout(model, args.pool)
    print(f"    readout={readout}", flush=True)
    cache = embed_month(
        model, Path(args.mosaic_dir) / y / m, args.month,
        AnchorStats(Path(args.xs_anchor_stats_dir) / f"{args.month}.npz"),
        schedule, day_anchors(args.anchors_per_day), device, args.batch_size,
        shard_index=args.shard_index, num_shards=args.num_shards,
        allow_empty=True, **panel,
    )
    if cache is None:                      # this shard held no usable ticker-day
        _atomic_savez(dest, empty=np.array([1]), xs_anchor_stats=stats_name,
                      readout=readout, num_shards=args.num_shards)
        print(f"{dest.name}: empty shard")
        return
    _atomic_savez(
        dest, X=cache["X"].astype(np.float32), z=cache["z"].astype(np.float32),
        date=cache["date"], anchor=cache["anchor"], ticker=cache["ticker"],
        target_names=cache["target_names"], xs_anchor_stats=stats_name,
        readout=readout, num_shards=args.num_shards,
    )
    print(f"{dest.name}: {len(cache['X'])} rows")


def cmd_embed_many(args):
    """Process a strided slice of a task file in ONE process.

    A CUDA context costs ~1.66 GB regardless of how little work the process
    does, so one process per task caps concurrency at ~26 on a 44 GB A40 —
    running 48 killed two thirds of them with OOM. Here the context and the
    loaded model are paid once per worker (and the model once per checkpoint),
    so worker count is set by CPU cores and GPU memory, not by task count.

    Tasks are sorted by checkpoint before striding, so consecutive tasks in a
    worker usually share a checkpoint and skip the reload.
    """
    machine = LocalMachineConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = MarketSchedule(args.holiday_csv or machine.holiday_csv)

    tasks = [ln.split() for ln in Path(args.tasks).read_text().split("\n") if ln.strip()]
    # SHARD 0 FIRST, then everything else. The stride below hands worker i the
    # tasks at i, i+N, i+2N..., and the old sort left each (ckpt, month) group
    # contiguous and 32 long. With 8 workers that is gcd(32, 8) = 8, so group
    # position 0 landed on worker 0 EVERY time -- and when the panel cache
    # hits, position 0 is the only task in the group that does anything: shard
    # 0 embeds the whole month and 1..31 write instant markers. Measured on job
    # 228476: worker 0 held all 28 real month-embeds and workers 1-7 wrote
    # markers and idled, one core busy out of eight, GPU at 0.2%.
    #
    # Sorting the real work to the front makes the stride spread it evenly
    # (28 tasks over 8 workers = 3 or 4 each) and costs nothing when the cache
    # misses, where every shard is real work anyway. Checkpoint order is kept
    # inside each block, so consecutive tasks still usually share a checkpoint
    # and skip the model reload.
    tasks.sort(key=lambda t: (int(t[2]) != 0, t[0], t[1], int(t[2])))
    mine = tasks[args.worker_index :: args.num_workers]
    print(f"worker {args.worker_index}: {len(mine)} of {len(tasks)} tasks", flush=True)

    # SHARDING AND THE PANEL CACHE ARE MUTUALLY EXCLUSIVE, so choose per month.
    # panel_source consults the cache only at num_shards == 1 (a shard is a
    # slice of a month, the cache stores whole months), and this path shards
    # unconditionally -- so a built panel could never be read here and every
    # scoring job re-decoded a month that is a function of the MONTH and not of
    # the checkpoint. Same trap post_train_ic_eval._panel_is_cached exists to
    # avoid, and its helper is reused rather than restated.
    #
    # A cached month collapses to ONE task: shard 0 embeds the whole month from
    # the memmapped panel and the other shards write the empty marker they
    # already use for a genuinely empty slice, which both consumers
    # (_shard_is_current's reader and xs_ic_series) skip on "empty". So the
    # task file, the shard filenames and the reducer are all untouched.
    model, loaded, panel = None, None, {}
    panel_for, cfg = None, None
    readout = ""
    stats_dir = Path(args.xs_anchor_stats_dir)
    stats_name = _stats_name(args.xs_anchor_stats_dir)
    done = failed = skipped = cached_months = 0
    for ckpt_s, month, shard, ap in mine:
        ckpt = Path(ckpt_s)
        out = Path(args.out_dir) / ckpt.name
        out.mkdir(parents=True, exist_ok=True)
        dest = out / f"{month}_a{ap}_{int(shard):03d}.npz"
        # Shard 0 is the cover: it is the one that holds the whole month when
        # the panel cache collapses the fan-out. See _shard_is_current.
        cover = out / f"{month}_a{ap}_000.npz"
        if dest.is_file() and _shard_is_current(
                dest, stats_name, cover=cover, readout=args.pool or "",
                num_shards=args.num_shards):
            skipped += 1
            continue
        try:
            # Panel kwargs BEFORE the model: the cache probe needs them, and a
            # shard the cache makes redundant must not pay a GPU model load.
            if panel_for != ckpt_s:
                cfg, panel = _panel_for(ckpt)
                panel_for = ckpt_s
            is_cached = _panel_is_cached(
                month, day_anchors(int(ap)), stats_dir, panel,
                None, panel["seq_len"])
            if is_cached and int(shard) != 0:
                # NOT an empty slice: these rows are in shard 0. The `collapsed`
                # stamp is what tells a later pass the difference.
                _atomic_savez(dest, empty=np.array([1]),
                              collapsed=np.array([1]),
                              xs_anchor_stats=stats_name,
                              readout=args.pool or readout,
                              num_shards=args.num_shards)
                done += 1
                continue
            if loaded != ckpt_s:
                model = load_model(str(ckpt), cfg, device)
                model.eval()
                readout = _apply_readout(model, args.pool)
                loaded = ckpt_s
            y, m = month.split("-")
            if is_cached:
                cached_months += 1
            cache = embed_month(
                model, Path(args.mosaic_dir) / y / m, month,
                AnchorStats(stats_dir / f"{month}.npz"),
                schedule, day_anchors(int(ap)), device, args.batch_size,
                shard_index=0 if is_cached else int(shard),
                num_shards=1 if is_cached else args.num_shards,
                allow_empty=True, **panel,
            )
            # THE COVER STAMP. In cached mode this shard (always shard 0) holds
            # the WHOLE month, and the collapsed markers beside it are valid
            # only while it does. Stamped on the empty case too, so a genuinely
            # empty cached month still covers its markers.
            cover_stamp = {"full_month": np.array([1])} if is_cached else {}
            cover_stamp["readout"] = readout
            cover_stamp["num_shards"] = args.num_shards
            if cache is None:
                _atomic_savez(dest, empty=np.array([1]),
                              xs_anchor_stats=stats_name, **cover_stamp)
            else:
                _atomic_savez(
                    dest, X=cache["X"].astype(np.float32),
                    z=cache["z"].astype(np.float32), date=cache["date"],
                    anchor=cache["anchor"], ticker=cache["ticker"],
                    target_names=cache["target_names"],
                    xs_anchor_stats=stats_name, **cover_stamp)
            done += 1
        except Exception as exc:                       # noqa: BLE001
            failed += 1
            print(f"  FAIL {ckpt.name} {month} shard {shard}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
    print(f"worker {args.worker_index}: done={done} skipped={skipped} "
          f"failed={failed} cached_months={cached_months}", flush=True)
    # Non-zero exit so the driver cannot mistake a silently-failing pool for
    # a completed one, the way the first 48-worker run did.
    sys.exit(1 if failed else 0)


def _atomic_savez(dest: Path, **arrays) -> None:
    """Write a shard so that its EXISTENCE means it is complete.

    ``np.savez_compressed`` writes straight to the destination, and the resume
    logic here is ``if dest.is_file(): skip``. Those two together are unsafe:
    a write killed partway (ENOSPC, SIGKILL, a full disk) leaves a truncated
    .npz that every later pass silently skips and the reduce then fails on --
    after the whole embed has been paid for again. Staging to a sibling temp
    and renaming makes resume correct by construction, since os.replace is
    atomic within a filesystem.
    """
    tmp = dest.parent / f".{dest.stem}.{os.getpid()}.tmp.npz"
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def _load_group(paths):
    Xs, Zs, ds, as_, ts, names = [], [], [], [], [], None
    # A GROUP MUST BE ONE READOUT. Pooling a mean-pooled month with a
    # last-pooled one would fit a ridge across two different representations
    # and report a plausible number; refuse instead. Shards predating the stamp
    # carry the checkpoint's own pool and read as "".
    #
    # ONLY SHARDS THAT HOLD ROWS GET A VOTE. An `empty` shard -- a genuinely
    # empty slice, or the `collapsed` marker written beside a cached month's
    # shard 0 -- carries no embedding, so it has no readout to be inconsistent
    # about. Counting them refused every cached month outright: markers are
    # written before any model is loaded, so they have no stamp, and the set
    # came out {'', 'last'} for a group whose actual DATA was uniformly 'last'.
    readouts = set()
    for p in sorted(paths):
        d = np.load(p, allow_pickle=False)
        if "empty" in d:
            continue
        readouts.add(_readout_tag_of(d))
        Xs.append(d["X"]); Zs.append(d["z"])
        ds.append(d["date"]); as_.append(d["anchor"]); ts.append(d["ticker"])
        names = [str(x) for x in d["target_names"]]
    if not Xs:
        return None
    if len(readouts) > 1:
        raise SystemExit(
            f"REFUSING TO REDUCE: shards in this group were embedded at "
            f"different readouts {sorted(readouts)} -- pooling them would fit "
            f"one ridge across two representations")
    # EVERY (date, anchor, ticker) ONCE. Shards of one fan-out partition the
    # month, so a repeat means shards from two fan-outs were pooled (see
    # _shard_is_current) -- the panel is then mostly duplicates and the IC is
    # computed on rows weighted by an accident of the cache.
    keys = np.char.add(np.char.add(np.concatenate(ds).astype(str), "@"),
                       np.char.add(np.concatenate(as_).astype(str), "@"))
    keys = np.char.add(keys, np.concatenate(ts).astype(str))
    n_dup = len(keys) - len(np.unique(keys))
    if n_dup:
        raise SystemExit(
            f"REFUSING TO REDUCE: {n_dup:,d} of {len(keys):,d} rows repeat a "
            f"(date, anchor, ticker) -- shards from different --num-shards "
            f"runs are mixed in this cache; delete it and re-embed")
    return dict(X=np.concatenate(Xs), z=np.concatenate(Zs),
                date=np.concatenate(ds), anchor=np.concatenate(as_),
                names=names, readout=readouts.pop() if readouts else "")


def cmd_reduce(args):
    root = Path(args.out_dir)
    ck_dirs = sorted(p for p in root.iterdir() if p.is_dir())

    # The eval panel is the same month at the same anchors for every
    # checkpoint, so its row count MUST match across them. When it does not,
    # shards are missing and the ICs are not comparable — the first run of this
    # sweep reported eval panels of 24,860 / 19,751 / 3,696 rows after 2/3 of
    # the embed tasks died on CUDA OOM, and scored them anyway.
    # Checkpoints with ZERO shards for this eval month are not participants —
    # several encoders can share a cache dir while being scored on different
    # months (their probe windows are chosen off fair_months independently).
    # Only non-empty panels are compared, so a genuinely partial panel is still
    # caught while a disjoint one is simply skipped.
    counts = {p.name: len(list(p.glob(f"{args.eval_month_glob}_a8_*.npz")))
              for p in ck_dirs}
    counts = {k: v for k, v in counts.items() if v > 0}
    if not counts:
        raise SystemExit(f"no checkpoint has an eval panel for "
                         f"{args.eval_month_glob}")
    ck_dirs = [p for p in ck_dirs if p.name in counts]
    if len(set(counts.values())) > 1 or (
            args.expect_shards and set(counts.values()) != {args.expect_shards}):
        msg = ("eval-panel shard counts differ or are short of "
               f"--expect-shards={args.expect_shards}: {counts}")
        if not args.allow_partial:
            raise SystemExit(f"REFUSING TO SCORE: {msg}\n"
                             "Re-run the missing embed tasks, or pass "
                             "--allow-partial if you truly want a partial panel.")
        print(f"WARNING: {msg}")

    results = []
    for ck_dir in ck_dirs:
        ev = _load_group(ck_dir.glob(f"{args.eval_month_glob}_a8_*.npz"))
        tr = _load_group(p for p in ck_dir.glob("*_a*_*.npz")
                         if not p.name.startswith(args.eval_month_glob.rstrip("*"))
                         and "_a8_" not in p.name)
        if ev is None or tr is None:
            print(f"{ck_dir.name}: missing train or eval panel, skipping")
            continue
        if tr["readout"] != ev["readout"]:
            raise SystemExit(
                f"REFUSING TO SCORE {ck_dir.name}: fit pool embedded at "
                f"{tr['readout']!r} but eval panel at {ev['readout']!r}")
        readout = tr["readout"]
        print(f"\n=== {ck_dir.name}: fit pool {len(tr['X']):,d} rows, "
              f"eval {len(ev['X']):,d} rows, readout={readout or 'ckpt-own'} ===")
        cells = np.char.add(np.char.add(ev["date"], "@"), ev["anchor"].astype(str))
        # One shuffle, then NESTED prefixes: every fit size sees a superset of
        # the previous one, so the curve isolates the effect of adding data
        # rather than mixing it with which rows were drawn.
        rng = np.random.RandomState(0)
        perm = rng.permutation(len(tr["X"]))
        alphas = args.alphas or [args.alpha]
        for n in [x for x in FIT_SIZES if x <= len(perm)] + [len(perm)]:
            idx = perm[:n]
            for alpha in alphas:
                row = {"ckpt": ck_dir.name, "n": int(n), "alpha": float(alpha),
                       # WHICH TOKEN THESE EMBEDDINGS WERE READ AT. Without it
                       # the results directory is a pile of jsons keyed only by
                       # (ckpt, n, alpha), and a mean-pooled row and a
                       # last-pooled row for the same checkpoint are
                       # indistinguishable -- probe_fit_table keeps the FIRST
                       # file at max n and probe_fit_breadth keeps the LAST, so
                       # the figure and the table would silently disagree.
                       "readout": readout}
                for task in TASKS:
                    if task not in ev["names"]:
                        continue
                    j = ev["names"].index(task)
                    ytr, yev = tr["z"][idx, j], ev["z"][:, j]
                    ok_tr, ok_ev = np.isfinite(ytr), np.isfinite(yev)
                    if ok_tr.sum() < 100 or ok_ev.sum() < 100:
                        continue
                    probe = ColumnwiseRidge(alpha=alpha, min_samples=100).fit(
                        tr["X"][idx], ytr[:, None],
                    )
                    pred = probe.predict(ev["X"])[ok_ev, 0]
                    metric = grouped_rank_ic_by_label(
                        pred, yev[ok_ev], cells[ok_ev],
                    )
                    row[task] = float(metric.mean)
                    row[task + "_se"] = float(metric.standard_error)
                results.append(row)
                print(f"  n={n:>9,d} a={alpha:<8g} " + "  ".join(
                    f"{t.replace('_900',''):18s} {row.get(t, float('nan')):+.4f}"
                    for t in TASKS))
    Path(args.json).write_text(json.dumps(results, indent=1))
    print(f"\nWrote {len(results)} rows to {args.json}")


def main():
    machine = LocalMachineConfig()
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--month", required=True)
    e.add_argument("--shard-index", type=int, default=0)
    e.add_argument("--num-shards", type=int, default=1)
    e.add_argument("--anchors-per-day", type=int, default=36)
    e.add_argument("--out-dir", required=True)
    e.add_argument("--batch-size", type=int, default=256)
    e.add_argument("--overwrite", action="store_true")
    e.add_argument("--mosaic-dir", default=machine.mosaic_dir)
    e.add_argument("--xs-anchor-stats-dir",
                   default="lab/market-jepa-mosaic/xs_anchor_stats_fwdvwap60")
    e.add_argument("--holiday-csv", default=None)
    e.add_argument("--pool", default=None,
                   help="readout to embed at (e.g. last). Prediction is scored at the LAST token for every arm; omit to use each checkpoint's own training pool -- what this sweep did before 2026-09-17, and why the SSL arms were mean-pooled against a last-pooled floor")
    e.set_defaults(func=cmd_embed)

    em = sub.add_parser("embed-many")
    em.add_argument("--tasks", required=True,
                    help="file of '<ckpt_dir> <month> <shard> <anchors>' lines")
    em.add_argument("--worker-index", type=int, required=True)
    em.add_argument("--num-workers", type=int, required=True)
    em.add_argument("--num-shards", type=int, default=32)
    em.add_argument("--out-dir", required=True)
    em.add_argument("--batch-size", type=int, default=256)
    em.add_argument("--mosaic-dir", default=machine.mosaic_dir)
    em.add_argument("--xs-anchor-stats-dir",
                    default="lab/market-jepa-mosaic/xs_anchor_stats_fwdvwap60")
    em.add_argument("--holiday-csv", default=None)
    em.add_argument("--pool", default=None,
                   help="readout to embed at (e.g. last). Prediction is scored at the LAST token for every arm; omit to use each checkpoint's own training pool -- what this sweep did before 2026-09-17, and why the SSL arms were mean-pooled against a last-pooled floor")
    em.set_defaults(func=cmd_embed_many)

    r = sub.add_parser("reduce")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--json", required=True)
    r.add_argument("--alpha", type=float, default=RIDGE_ALPHA)
    # Alpha and fit size share the same embeddings, so sweeping both costs one
    # extra ridge solve per cell rather than another GPU pass.
    r.add_argument("--alphas", type=float, nargs="*", default=None,
                   help="sweep these ridge alphas instead of just --alpha")
    r.add_argument("--eval-month-glob", required=True,
                   help="YYYY-MM of the eval panel (embedded at 8 anchors/day)")
    r.add_argument("--expect-shards", type=int, default=0,
                   help="required eval-panel shard count per checkpoint")
    r.add_argument("--allow-partial", action="store_true")
    r.set_defaults(func=cmd_reduce)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
