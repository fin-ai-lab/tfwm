"""Full-history supervised: cross-sectional rank IC vs. training month.

Pulls each run's ``xs_ic/<task>`` — the synchronized cross-section IC computed
in the same SLURM job that trained the checkpoint
(``scripts/generic/post_train_ic_eval.py``) — and draws one curve per task over
time. Raw IC, NOT baseline-subtracted.

Projects (one per task; run.name is ``YYYY-MM`` plus a recipe suffix):
    - supervised-full-month-return-ce         → return_900
    - supervised-full-month-vol-change-ce     → volatility_change_900
    - supervised-full-month-spread-change-ce  → spread_change_900

Each project accumulates every generation of the sweep, so runs are filtered
to one target definition by ``XS_STATS_DIR_REQUIRED`` -- see the comment there.

Same figure as the ΔAUC version this replaces; only the metric changed. The
companion probe panel is gone with it: production sweeps run
training.live_eval=false, so no probe AUC is logged any more.

All three targets are now measured between two FORWARD windows (VWAP / mean
spread / realized vol over ``[t, t+60)`` vs ``[t+h, t+h+60)``). The older
definitions each leaked something known at ``t`` -- the quote-bounce term in
the mid-to-mid return, and the ``-spread(t)`` / ``-bwd_vol`` legs of the two
change targets -- which is exactly why runs are filtered by target above.

Only runs whose state is ``finished`` are plotted; in-progress runs are
fetched (so the cache stays current) but filtered out at plot time.

Caching
    A JSON cache lives next to this script (``api_cache.json``). On every
    run, this script:
      1. Lists all runs in each project via the W&B API (always — so new
         runs from in-progress sweeps show up immediately).
      2. Re-fetches summary for any run whose cached state is not
         ``finished`` (so the next run sees up-to-date summaries once
         those runs finish).
      3. Skips API summary calls for runs already cached as ``finished``
         unless ``--refresh`` is passed.

Run:
    uv run plots/full_data_multihead/full_data_multihead.py

    Writes TWO figures from the one W&B pull: ``full_data_multihead`` and
    ``full_data_multihead_highlighted``, the latter greying every month
    outside the 31 reported ones (PANEL_TRAIN_MONTHS). Pass ``--out`` to write
    just one, ``--highlight-panel`` to say which.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "metrics"))
from metrics import SUP_SPAN_NO_SPAN  # noqa: E402
from style import (  # noqa: E402
    DEFAULT_WANDB_ENTITY,
    SERIES_STYLES,
    SERIES_MARKER,
    apply_style,
    load_sweep_months,
    save_figure,
    WIDTH_FULL,
)


HERE = Path(__file__).resolve().parent
CACHE_PATH = HERE / "api_cache.json"

# THE REPORTED PANEL: the 31 eval months every current arm is scored on.
# Seed 42 sampled 32 TRAIN months (scripts/experiments/sample_sweep_months.py);
# the six-month-span recipe drops 2008-02, whose span would start before the
# data begins (metrics.SUP_SPAN_NO_SPAN), so the reported panel is the other 31
# and their eval months. --highlight-panel keeps those in the series colour and
# greys the ~170 months that only this full-history figure covers, so a reader
# can see where the reported panel sits inside the history.
#
# HELD AS TRAIN MONTHS, compared before _train_to_eval_midpoint: the x axis is
# the EVAL midpoint, one month after the training month, so a set of eval
# months cannot be matched against it directly. Train month m IS eval month
# m+1 on that axis, so the 31 below ARE the 31 eval months of
# plots/latent_eval/run_6mo_eval.sh and the metrics table -- derived from the
# same two sources, so the panel cannot drift from them here.
PANEL_TRAIN_MONTHS: frozenset[str] = (
    frozenset(load_sweep_months()) - SUP_SPAN_NO_SPAN
)
GREY_OUT = "#bdbdbd"

# (series style key in style.SERIES_STYLES, wandb project, metric key).
# The -ce projects are the CROSS-ENTROPY recipe (11 bins, expected-bin MSE
# 0.025, tau annealed 0.10*k -> 0.05*k). The -ic projects they replace are the
# regression-era sweep and stopped at 76-78 finished months, which is why the
# old figure was empty from 2016 on.
# Each -ce project now holds TWO generations of runs: the original sweep on the
# mid-to-mid return target, and the 2026-08-22 rerun on the forward-VWAP target
# (docs/return_bad_calculation.md). Both name their runs ``YYYY-MM_...``, so the
# month prefix cannot tell them apart and a month present in both would plot
# twice. The training config stamps which anchor-stat tables the target came
# from, so filter on that -- basename only, because compute nodes see the
# staged /hpc_temp copy while a local run sees /data/lab/market-jepa-mosaic.
XS_STATS_DIR_REQUIRED = "xs_anchor_stats_fwdvwap60"

# ONE PROJECT, THREE METRICS — the full-history baseline became a single
# MULTIHEAD trunk on 2026-08-27 (sweeps/full_data_multihead.sh), so a month is
# now one run that logs all three xs_ic/<task> values rather than three runs
# logging one each.
#
# WHY THE SWITCH. The 32-month W sweep found sharing a trunk costs nothing or
# helps (multihead - specialist: +0.0050 return at W=32, t=+4.01; +0.0028 vol
# at W=8, t=+2.51; nothing significantly negative anywhere), so three models a
# month was three times the GPU for a result one model matches.
#
# AND THE SPECIALIST PROJECTS CANNOT FILL THIS FIGURE. Each of them holds ~203
# runs on the RETIRED mid-to-mid target plus a forward-VWAP rerun that was
# abandoned early: on 2026-08-28 the three had 25, 4 and 1 finished months of
# the 203. XS_STATS_DIR_REQUIRED correctly drops the old generation, which is
# why this plot was almost empty. They are kept here only as provenance:
#   supervised-full-month-return-ce / -vol-change-ce / -spread-change-ce
#
# The multihead sweep pins W=8 where the specialists inherited whatever
# schemas.py defaulted to at submit time (0 before 2026-08-25, 16 after), so
# this series is also the first full-history one where every month is the same
# model.
# Renamed 2026-09-07: the series is pairwise (LTR), not cross_entropy, and
# the "-ce" it carried was left over from the retired binned family.
# ONE PROJECT PER BUNDLE, not one project. run_full_data_supervised.sh resolves
# the sweep's wandb project to "<name>-<commit6>-<train_start>-<eval_end>", so a
# 203-month sweep at 4 months/job lands in ~51 SEPARATE projects and the bare
# name below has never existed as a project of its own. Reading it directly is
# why this figure came back empty: api.runs("boothai/supervised-full-month-
# multihead") raises, fetch_project_runs swallows it, and every task returns [].
# FULL_DATA_PROJECT is therefore a PREFIX, expanded at run time by
# resolve_full_data_projects.
#
# NOT A SUBSTRING MATCH. "supervised-full-month-multihead-ce" is the RETIRED
# binned generation (202 finished months, run names like
# "2014-05_multi_k11_mse0.025_tau0.5_recency8_s42") and a plain startswith()
# would pull all 202 of them into a figure that is meant to be pairwise/LTR
# only. The suffix pattern -- six hex digits then two YYYY-MM pairs -- excludes
# it structurally rather than by blacklisting the one name we happen to know.
FULL_DATA_PROJECT = "supervised-full-month-multihead-67e219"

_BUNDLE_PROJECT_RE = re.compile(
    rf"^{re.escape(FULL_DATA_PROJECT)}-"
    rf"\d{{4}}-\d{{2}}-\d{{4}}-\d{{2}}$"
)


def resolve_full_data_projects(api, entity: str) -> list[str]:
    """Every per-bundle project belonging to the full-history multihead sweep.

    Listed from the API rather than reconstructed from a month list: the
    bundling (--months-per-job) is a submit-time choice, so the ranges in the
    project names are not derivable here, and a wave resubmitted at a different
    bundle size creates a different set of them.
    """
    return sorted(p.name for p in api.projects(entity)
                  if _BUNDLE_PROJECT_RE.match(p.name))


# THE SPAN IS THE RECIPE BOUNDARY. The same project pattern carries the
# single-month generation this campaign replaced, and a project name says
# nothing about the recipe, so a figure reading the pattern alone silently
# averages two different experiments. Keep only runs whose own train window is
# the span schemas.py currently defines -- resolved, never written here, so it
# cannot go stale when the span moves.
def recipe_span_months() -> int:
    from market_jepa.schemas import DatasetConfig
    return int(DatasetConfig.train_span_months)


def run_span_months(record: dict) -> int:
    """Months covered by a run's own training window, 0 if unreadable."""
    a = str(record.get("train_date_start") or "")
    b = str(record.get("train_date_end") or "")
    if len(a) < 7 or len(b) < 7:
        return 0
    try:
        return ((int(b[:4]) * 12 + int(b[5:7]))
                - (int(a[:4]) * 12 + int(a[5:7]))) + 1
    except ValueError:
        return 0


def keep_current_recipe(records: list[dict]) -> tuple[list[dict], int]:
    """Drop runs whose training window is not the current span."""
    want = recipe_span_months()
    kept = [r for r in records if run_span_months(r) == want]
    return kept, len(records) - len(kept)


def dedupe_by_train_month(records: list[dict]) -> tuple[list[dict], int]:
    """One record per TRAIN month, newest run wins; returns (kept, dropped).

    Bundle projects overlap whenever a wave is resubmitted under a new commit:
    2008-01..2008-04 exist under both -8067e9- and -ca70db-, the same month
    trained twice by the same recipe. Without this every such month plots two
    points and any per-month join silently becomes many-to-many.
    """
    best: dict[str, dict] = {}
    dropped = 0
    for r in records:
        ym = _train_month(r.get("run_name") or "")
        if ym is None:
            continue
        prev = best.get(ym)
        if prev is None:
            best[ym] = r
        else:
            dropped += 1
            if str(r.get("created_at") or "") > str(prev.get("created_at") or ""):
                best[ym] = r
    return list(best.values()), dropped

# THE HEAD KEY, because the campaign scores head-only (POST_TRAIN_PROBE=0) and
# writes no xs_ic/<task> at all. Reading the probe key found the runs, dropped
# nothing, and reported "no runs with xs_ic" -- a sweep that had scored looked
# exactly like a sweep that had not. A multihead run carries one head key per
# task, which is what this figure has always been about.
TASKS: list[tuple[str, str, str]] = [
    ("supervised_return", FULL_DATA_PROJECT, "xs_ic/head:return_900"),
    ("supervised_vol",    FULL_DATA_PROJECT, "xs_ic/head:volatility_change_900"),
    ("supervised_spread", FULL_DATA_PROJECT, "xs_ic/head:spread_change_900"),
]


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True))


def _fetch_run_summary(
    run,
    summary_keys: tuple[str, ...],
    history_keys: tuple[str, ...] = (),
) -> dict:
    """Pull metric values from ``run.summary`` plus optional history scans.

    ``summary_keys`` are read straight from ``run.summary`` (single API call).
    ``history_keys`` are pulled with ``scan_history(keys=...)`` and reduced to
    the last non-null value — this covers metrics that are logged from a
    sibling process via ``wandb.init(mode="shared")`` (e.g. probe AUC), which
    end up in history but never in ``run.summary``. Network-bound; thread-safe.
    """
    out = {k: run.summary.get(k) for k in summary_keys}
    for k in history_keys:
        last_val = None
        try:
            for row in run.scan_history(keys=[k]):
                v = row.get(k)
                if v is not None:
                    last_val = v
        except Exception:
            last_val = None
        out[k] = last_val
    return out


def _resolve_state(api, entity: str, project: str, run_id: str, listing_state: str,
                   *, state_verified: bool) -> tuple[str, bool]:
    """Return ``(true_state, verified)`` for a run.

    The W&B project listing (``api.runs(project)``) sometimes returns a stale
    ``state`` field — runs that are still ``running`` can show up as
    ``finished`` for a window. Direct ``api.run(path)`` is consistent. If we
    accept the listing's word, we end up plotting in-progress runs.

    Policy: a run is treated as ``finished`` only after a direct fetch has
    confirmed it. Once verified, finished is sticky (we don't re-verify).
    If a previously-verified run shows non-finished in the listing (e.g. it
    was resumed), we re-verify directly.
    """
    if state_verified and listing_state == "finished":
        return "finished", True
    if listing_state != "finished" and not state_verified:
        # Listing says not finished and we never verified anything else;
        # cheapest path: trust it. We don't plot non-finished runs anyway,
        # and a stale-running listing only causes a missed point, not a
        # wrong one.
        return listing_state, False
    try:
        return api.run(f"{entity}/{project}/{run_id}").state, True
    except Exception:
        return listing_state, False


def fetch_project_runs(
    api,
    entity: str,
    project: str,
    cache: dict,
    summary_keys: tuple[str, ...],
    history_keys: tuple[str, ...],
    *,
    refresh: bool,
    workers: int,
    verbose: bool,
) -> list[dict]:
    """Return the latest run records for one project, refreshing the cache.

    A record is ``{run_id, run_name, state, summary}``. Runs whose cached
    state is not ``finished`` (or all runs if ``refresh=True``) get a fresh
    summary pull; the rest read straight from the cache.

    Cached ``finished`` runs are also re-fetched if their summary is missing
    any of ``summary_keys`` or ``history_keys`` — this handles the schema-bump
    case where new metric keys (e.g. probe AUC) are added after the cache was
    first built. Once a run is fetched, missing keys are stored as ``None`` so
    the check is a no-op on subsequent runs.

    State is resolved via :func:`_resolve_state` — the project listing alone
    is not trusted for ``finished``.
    """
    try:
        api_runs = list(api.runs(f"{entity}/{project}"))
    except Exception as e:
        if verbose:
            print(f"  !! failed to list runs for {project}: {e}")
        return []

    # Resolve true state for every run (parallel, but only does a direct
    # fetch when needed — see _resolve_state).
    def _resolve_one(run):
        cache_key = f"{project}/{run.id}"
        entry = cache.get(cache_key) or {}
        prev_verified = bool(entry.get("state_verified"))
        return _resolve_state(
            api, entity, project, run.id, run.state,
            state_verified=prev_verified,
        )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        resolved = list(ex.map(_resolve_one, api_runs))

    pending: list = []          # API run objects that need a summary pull
    cached_records: list[dict] = []  # records we'll keep as-is from the cache
    verified_directly = 0

    for run, (true_state, verified) in zip(api_runs, resolved):
        cache_key = f"{project}/{run.id}"
        entry = cache.get(cache_key) or {}
        prev_state = entry.get("state")
        is_terminal = true_state == "finished"
        cached_summary = entry.get("summary") or {}
        history_keys_fetched = set(entry.get("history_keys_fetched") or [])
        # Summary keys: present-in-dict is enough (None = legit "not logged").
        # History keys: must have actually been scan_history'd before; we can't
        # tell a None-because-not-logged apart from a None-because-never-fetched
        # without an explicit marker, so use the marker.
        missing_summary_key = any(k not in cached_summary for k in summary_keys)
        missing_history_key = any(k not in history_keys_fetched for k in history_keys)
        missing_key = missing_summary_key or missing_history_key

        meta = {
            "project": project,
            "run_id": run.id,
            "run_name": run.name,
            "state": true_state,
            "state_verified": verified and is_terminal,
            # Comes free with the listing (no extra API call) and is refreshed
            # on every invocation, so old caches need no invalidation.
            "xs_stats_dir": ((run.config or {}).get("dataset") or {})
                             .get("xs_anchor_stats_dir"),
            # THE TRAINING WINDOW, for the same reason and at the same price:
            # free from the listing, refreshed every invocation. It is what
            # separates this campaign from the single-month generation under
            # the same project pattern -- see keep_current_recipe.
            "train_date_start": ((run.config or {}).get("dataset") or {})
                                 .get("train_date_start"),
            "train_date_end": ((run.config or {}).get("dataset") or {})
                               .get("train_date_end"),
            # Free from the listing; the tiebreak in dedupe_by_train_month.
            "created_at": str(getattr(run, "created_at", "") or ""),
        }
        if verified and not entry.get("state_verified"):
            verified_directly += 1

        if refresh or not is_terminal or "summary" not in entry or missing_key:
            pending.append((cache_key, run, meta))
        else:
            entry.update(meta)
            cache[cache_key] = entry
            cached_records.append(entry)

    if verbose:
        print(
            f"[{project}]  api={len(api_runs)}  "
            f"refresh={len(pending)}  cached={len(cached_records)}  "
            f"newly_verified={verified_directly}"
        )

    if pending:
        def _work(item):
            cache_key, run_obj, meta = item
            summary = _fetch_run_summary(run_obj, summary_keys, history_keys)
            return cache_key, meta, summary

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_work, item) for item in pending]
            for fut in as_completed(futs):
                cache_key, meta, summary = fut.result()
                entry = cache.get(cache_key, {})
                entry.update(meta)
                merged_summary = dict(entry.get("summary") or {})
                merged_summary.update(summary)
                entry["summary"] = merged_summary
                already_fetched = set(entry.get("history_keys_fetched") or [])
                already_fetched.update(history_keys)
                entry["history_keys_fetched"] = sorted(already_fetched)
                cache[cache_key] = entry
                cached_records.append(entry)

    return cached_records


def _train_month(run_name: str) -> str | None:
    """``'2009-01_k11_mse0.025_tau0.5'`` → ``'2009-01'`` (the TRAIN month)."""
    m_ = re.match(r"^(\d{4}-\d{2})(?:_|$)", run_name or "")
    return m_.group(1) if m_ else None


def _train_to_eval_midpoint(run_name: str) -> pd.Timestamp | None:
    """Map a ``YYYY-MM`` train month to the middle (day 15) of the next month.

    The supervised baselines are evaluated on the calendar month immediately
    following training; placing the marker mid-eval-month visualises that
    the AUC summarises a full month of out-of-sample data, not a single
    point in time.
    """
    # The month is a PREFIX, not the whole name: the cross-entropy sweep pins
    # its knobs in the run name (``2009-01_k11_mse0.025_tau0.5``) so an
    # archived run says which arm it belongs to. Splitting the whole string on
    # "-" silently dropped every one of those runs.
    m_ = re.match(r"^(\d{4})-(\d{2})(?:_|$)", run_name or "")
    if not m_:
        return None
    y, m = int(m_.group(1)), int(m_.group(2))
    eval_y, eval_m = (y + 1, 1) if m == 12 else (y, m + 1)
    try:
        return pd.Timestamp(year=eval_y, month=eval_m, day=15)
    except ValueError:
        return None


CKPT_ROOT = Path("/data/lab/market-jepa-checkpoints")

# Mirrors merge_score_results.HEAD_SCHEMA. A checkpoint scored before the
# 2026-08-29 multihead loader fix carries xs_ic/head:* keys that are an
# UNTRAINED head's readout, so the presence of a head key proves nothing --
# only this stamp, written by the merge, says the value came from the real
# head. The guard that used to refuse the whole figure is now per checkpoint,
# which is what lets the head panel fill in as the re-score lands.
HEAD_SCHEMA = 2


def fill_from_ckpt(records: list[dict], project: str, metric_key: str,
                   ckpt_root: Path = CKPT_ROOT) -> tuple[int, int]:
    """Backfill ``metric_key`` into each record's summary from its xs_ic.json.

    The head's readout is written to every checkpoint's ``xs_ic.json`` by
    post_train_ic_eval, but it never reached the W&B summaries -- the
    cross-entropy sweep logged the probe keys and not ``xs_ic/head:*``, so
    the API cache has 0 of 203 runs carrying it per task while the checkpoint
    tree has 203 of 203. Rather than re-log a finished 609-run sweep, read
    the number from where it actually lives. The probe path is untouched and
    still comes from W&B, so the two figures are not silently sourced
    differently: this only fires for keys the summary lacks.

    THAT PARAGRAPH DESCRIBES THE RETIRED SPECIALIST PROJECTS. The figure now
    reads the MULTIHEAD trunk, and until it is re-scored each of its 202
    checkpoints carries exactly one head key, ``xs_ic/head:return_900``, whose
    value is an untrained head (eval/checkpoints.load_model looked for
    ``head.pt`` where a trunk saves ``heads.pt``). Those are refused by the
    HEAD_SCHEMA stamp below rather than plotted, so the head panel shows only
    months that have actually been re-measured and gains the rest as the
    re-score merges. The probe path (the default, and the checked-in figure)
    reads none of this and is unaffected.

    Returns ``(n_filled, n_stale)`` -- the second being checkpoints whose head
    predates the loader fix and was therefore refused.
    """
    n = stale = 0
    is_head = metric_key.startswith("xs_ic/head:")
    for r in records:
        summary = r.setdefault("summary", {})
        # THE W&B SUMMARY IS NOT AN INDEPENDENT SOURCE FOR A HEAD. The same
        # post_train_ic_eval run that wrote the untrained readout beside the
        # checkpoint ALSO logged it to the run summary, so a summary hit is
        # the same bad number rather than a second opinion -- and taking it
        # short-circuits the stamp check below. That is exactly how 202 months
        # of noise reached the return panel on the first cut of this gate
        # (return is the one task the broken loader produced a head for, which
        # is why vol and spread looked correctly empty and hid it).
        #
        # In head mode the STAMPED CHECKPOINT FILE IS THE ONLY SOURCE, and a
        # month without one is blanked so records_to_dataframe drops it.
        if summary.get(metric_key) is not None and not is_head:
            continue
        # HEAD MODE PREFERS A STAMPED CHECKPOINT AND FALLS BACK TO W&B.
        #
        # It used to blank the summary here and accept the stamped file as the
        # ONLY source, because the eval that wrote an untrained readout beside
        # a checkpoint had also logged that same number to the run summary --
        # so a summary hit was the same bad number, not a second opinion.
        #
        # That premise no longer holds for these projects. Every pre-fix run
        # was DELETED on 2026-09-08 and the wave was retrained; all 203
        # finished runs here were created 2026-09-09, after the 2026-08-29
        # loader fix. The remaining check is that nothing in the training path
        # writes xs_head_schema at all -- only rescore_heads.py and
        # merge_score_results.py do -- so requiring the stamp refused every
        # in-job eval and drew an empty figure.
        #
        # A stamped file still WINS where one exists; the summary is used only
        # when there is no stamped checkpoint to prefer. Runs predating the fix
        # are recognisable and should not be re-admitted by this: the broken
        # loader produced a head for return_900 ONLY, so a multihead run
        # carrying fewer than three xs_ic/head:* keys is refused below.
        # THE STAMP IS THE TEST, AND IT IS READ FIRST. The three-head rule
        # below is a heuristic for telling a pre-fix run from a real one WHEN
        # NOTHING ELSE CAN; a stamped checkpoint says so outright and must not
        # be gated behind it. It was: the rule ran first, on the W&B SUMMARY,
        # and these runs log no metrics to W&B at all (every xs_ic key in the
        # summary is None), so a scored, stamped, three-head checkpoint was
        # refused as "pre-2026-08-29" and the figure read as empty.
        f = ckpt_root / project / str(r.get("run_id", "")) / "xs_ic.json"
        blob = None
        if f.is_file():
            try:
                blob = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                blob = None
        stamped = bool(blob) and blob.get("xs_head_schema", 0) >= HEAD_SCHEMA
        if is_head and not stamped:
            head_keys = [k for k in summary
                         if str(k).startswith("xs_ic/head:")
                         and summary.get(k) is not None]
            if len(head_keys) < 3:
                summary[metric_key] = None
                stale += 1
                continue
            # Unstamped file: leave whatever the summary already gave us
            # rather than overwriting it with an unverifiable number.
            continue
        if blob is None:
            continue
        v = blob.get(metric_key)
        if v is not None:
            summary[metric_key] = v
            n += 1
    return n, stale


def candidate_midpoints(records: list[dict]) -> list[pd.Timestamp]:
    """Every eval month the sweep covers, whether or not it has a value yet.

    The same state / target / run-name filters records_to_dataframe applies,
    minus the metric check -- so a month whose head is still being re-scored
    still contributes its x position.
    """
    out = []
    for r in records:
        if r.get("state") != "finished":
            continue
        if PurePosixPath(str(r.get("xs_stats_dir"))).name != XS_STATS_DIR_REQUIRED:
            continue
        ts = _train_to_eval_midpoint(r.get("run_name") or "")
        if ts is not None:
            out.append(ts)
    return sorted(out)


def records_to_dataframe(
    records: list[dict], series_key: str, metric_key: str,
) -> pd.DataFrame:
    """Filter to runs with a parseable ``YYYY-MM`` run name + a non-null AUC.

    Runs built against a different anchor-stat table than
    ``XS_STATS_DIR_REQUIRED`` are dropped: their target is a different
    quantity and mixing the two generations in one curve is meaningless.
    A run with no stamp at all predates the field and is likewise dropped.
    """
    rows: list[dict] = []
    dropped_target = 0
    for r in records:
        if r.get("state") != "finished":
            continue
        stamped = r.get("xs_stats_dir")
        if PurePosixPath(str(stamped)).name != XS_STATS_DIR_REQUIRED:
            dropped_target += 1
            continue
        run_name = r.get("run_name") or ""
        ts = _train_to_eval_midpoint(run_name)
        if ts is None:
            continue
        summary = r.get("summary") or {}
        auc = summary.get(metric_key)
        if auc is None:
            continue
        rows.append({
            "series": series_key,
            "run_name": run_name,
            "eval_midpoint": ts,
            "auc": float(auc),
            "state": r.get("state"),
            "in_panel": _train_month(run_name) in PANEL_TRAIN_MONTHS,
        })
    if dropped_target:
        print(f"  [{series_key}] dropped {dropped_target} run(s) on a "
              f"different anchor-stat target")
    if not rows:
        return pd.DataFrame(
            columns=["series", "run_name", "eval_midpoint", "auc", "state",
                     "in_panel"]
        )
    df = pd.DataFrame(rows).sort_values("eval_midpoint").reset_index(drop=True)
    dupes = df["eval_midpoint"].duplicated().sum()
    if dupes:
        raise SystemExit(
            f"{series_key}: {dupes} month(s) have two runs on "
            f"{XS_STATS_DIR_REQUIRED}; the figure would plot each twice. "
            f"Offenders: {sorted(df.loc[df['eval_midpoint'].duplicated(keep=False), 'run_name'])}"
        )
    return df


def _plot_curves(
    df: pd.DataFrame, out_path: Path,
    full_x: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    highlight_panel: bool = False,
) -> None:
    """Render the three task curves as a 1×3 stacked subplot and save.

    One row per task with its own y-axis so the within-task variation isn't
    dominated by the across-task level differences. Each row shows two y-ticks
    and is identified by a label in the bottom-right corner.
    """
    fig, axes = plt.subplots(
        len(TASKS), 1,
        figsize=(WIDTH_FULL, WIDTH_FULL * 0.18 * len(TASKS)),
        sharex=True,
    )

    # The ΔAUC version pinned each row to a hand-picked band around 0.5-0.95.
    # IC is signed and centred on zero, so rows autoscale and the ticks are
    # chosen from the data; zero gets its own rule because "better than
    # predicting nothing" is the question this figure is asked.
    YLIMS = {k: (None, None) for k, _, _ in TASKS}
    # HOW MANY LOW OUTLIERS TO CROP OUT OF THE VIEW, per row. return's 2021-02
    # sits at -0.026 while the next lowest month is -0.008 -- a 3x gap from one
    # point, which stretched the row's axis over empty space and squashed the
    # 200 months that carry the signal.
    #
    # A COUNT, NOT A LIMIT. A hardcoded ylim would silently start clipping more
    # months as the sweep fills in; cropping the n lowest and PRINTING what
    # went missing cannot hide a second outlier appearing later.
    YCROP_LOW = {"supervised_return": 1}
    cropped: list[tuple[str, str, float]] = []
    for ax, (series_key, _project, _probe_key) in zip(axes, TASKS):
        style = SERIES_STYLES[series_key]
        sub = df[df["series"] == series_key]
        if not sub.empty and highlight_panel:
            # TWO LAYERS, ONE SERIES. The months outside the reported panel
            # still carry the shape of the history, so they stay -- in grey
            # and behind -- while the 31 keep the task colour. COLOUR ALONE
            # SEPARATES THEM: every dot is the 3.5 of the un-highlighted
            # figure (user, 2026-09-15), so the two versions overlay exactly
            # and a highlighted month is not also a visually heavier one.
            # Everything downstream (mu, the crop, the ticks) still reads the
            # FULL sub: this is a change of ink, not of what the row
            # summarises.
            for mask, colour, size, z in (
                (~sub["in_panel"], GREY_OUT, 3.5, 1),
                (sub["in_panel"], style["color"], 3.5, 3),
            ):
                part = sub[mask]
                if part.empty:
                    continue
                ax.plot(
                    part["eval_midpoint"], part["auc"],
                    linestyle="none", marker=SERIES_MARKER, markersize=size,
                    color=colour, zorder=z,
                )
        elif not sub.empty:
            ax.plot(
                sub["eval_midpoint"],
                sub["auc"],
                linestyle="none",
                marker=SERIES_MARKER,
                markersize=3.5,
                color=style["color"],
            )
        ylim, base_ticks = YLIMS[series_key]
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.axhline(0.0, color="#888", linewidth=0.6, zorder=0)
        if not sub.empty:
            mu = float(sub["auc"].mean())
            ax.plot(
                [sub["eval_midpoint"].min(), sub["eval_midpoint"].max()],
                [mu, mu],
                color="black", linestyle="--", linewidth=0.8, zorder=0,
            )
            hi = float(sub["auc"].max())
            n_crop = YCROP_LOW.get(series_key, 0)
            if n_crop and len(sub) > n_crop + 1:
                ordered = sub["auc"].sort_values()
                lo = float(ordered.iloc[n_crop])
                pad = 0.08 * max(hi - lo, 1e-9)
                ax.set_ylim(lo - pad, hi + pad)
                for _, r in sub[sub["auc"] < lo - pad].iterrows():
                    cropped.append((series_key,
                                    r["eval_midpoint"].strftime("%Y-%m"),
                                    float(r["auc"])))
            else:
                lo = float(sub["auc"].min())
            # mu sits wherever the data puts it, so a fixed tick can land on
            # top of it and the two labels overprint. Keep mu and drop any
            # neighbour closer than 12% of the row's range.
            #
            # 6% was not enough once the -ce projects filled the history: on
            # vol_change the row minimum (0.021) sits 6.0% of the span from
            # the zero line, cleared the test by a rounding margin, and
            # printed on top of it. A row here is 0.18 * WIDTH_FULL tall, so
            # 6% of it is a few pixels -- the threshold has to be generous in
            # DATA units because the PIXEL budget is small.
            # MEASURED ON THE AXIS, NOT THE DATA. This guard exists to stop
            # two tick labels printing on top of each other, which is a
            # question about pixels -- and pixels map to the axis range, which
            # matplotlib may have padded well beyond hi-lo. Using the data
            # span broke exactly when the figure is sparsest: with one month
            # plotted hi == lo, the span collapsed to 1e-9, every candidate
            # tick cleared a threshold of 1.2e-10, and mu printed through the
            # numeric tick beside it.
            y0, y1 = ax.get_ylim()
            span = max(y1 - y0, 1e-9)
            # THREE TICKS: zero, mu, and one ROUND number near the top. The
            # row minimum is not labelled -- it is the least interesting value
            # on the axis and it cost a third of a 3-decimal tick budget.
            #
            # The top tick is hi FLOORED to two decimals (0.045 -> 0.04, 0.121
            # -> 0.12, 0.255 -> 0.25), not rounded: floor keeps it inside the
            # data so the axis never advertises a value no month reached.
            # Two decimals throughout, because these rows differ by an order of
            # magnitude and a shared 3-decimal format spent its extra digit on
            # return alone.
            top = math.floor(hi * 100.0) / 100.0
            tick_vals = [mu]
            for t in (0.0, top):
                # Skip a top tick that floored into the zero line (hi < 0.01),
                # and keep the 12%-of-span clearance that stops a fixed tick
                # printing on top of mu -- a row is 0.18 * WIDTH_FULL tall, so
                # the pixel budget is small and the guard has to be generous in
                # DATA units.
                if all(abs(t - u) > 0.12 * span for u in tick_vals):
                    tick_vals.append(t)
            tick_vals = sorted(tick_vals)
            tick_labels = [
                r"$\mu$" if abs(t - mu) < 1e-12 else f"{t:.2f}"
                for t in tick_vals
            ]
            ax.set_yticks(tick_vals)
            ax.set_yticklabels(tick_labels)
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.text(
            0.98, 0.04, style["label"],
            transform=ax.transAxes, ha="right", va="bottom",
        )

    for series_key, ym, val in cropped:
        print(f"  [{series_key}] cropped out of view: {ym} at {val:+.4f}")

    bottom = axes[-1]
    # SPAN EVERY MONTH THE SWEEP COVERS, not just the months that have a value
    # yet. At completion the two are identical, so the finished figure is
    # unchanged; while a re-score is landing it keeps the axis still, so points
    # appear where they belong instead of the axis rescaling under them -- and
    # it shows honestly how much of the history is still missing.
    span_src = full_x if full_x else (
        (df["eval_midpoint"].min(), df["eval_midpoint"].max())
        if not df.empty else None
    )
    if span_src:
        margin = pd.Timedelta(days=30)
        # SNAP THE LEFT EDGE TO ITS OWN JANUARY. YearLocator(2) ticks even
        # years at Jan 1, but the first eval midpoint is mid-February, so a
        # 30-day margin put the axis at ~Jan 16 and the opening year's tick
        # fell just outside it -- the history started at 2008 and the first
        # label read 2010. Extending to Jan 1 of that year costs six weeks of
        # empty axis and labels the year the data actually starts in.
        left = min(span_src[0] - margin,
                   pd.Timestamp(year=span_src[0].year, month=1, day=1))
        bottom.set_xlim(left, span_src[1] + margin)
        # THE GAP BEFORE THE FIRST POINT IS THE TRAINING PERIOD, not missing
        # data: the axis starts at January of the first year so the opening
        # tick has a label, but the first checkpoint cannot be evaluated until
        # its training span has elapsed. Shade that lead-in in each row's own
        # colour at low opacity -- the caption names it -- so the empty stretch
        # reads as history spent rather than months the sweep never ran.
        for ax, (series_key, _project, _probe_key) in zip(axes, TASKS):
            sub = df[df["series"] == series_key]
            if sub.empty:
                continue
            ax.axvspan(
                left, sub["eval_midpoint"].min(),
                color=SERIES_STYLES[series_key]["color"],
                alpha=0.12, linewidth=0, zorder=0,
            )
    bottom.xaxis.set_major_locator(mdates.YearLocator(2))
    bottom.xaxis.set_minor_locator(mdates.YearLocator(1))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    out_paths = save_figure(fig, out_path)
    for p in out_paths:
        print(f"Wrote {p}")
    plt.close(fig)


def _print_diagnostics(df: pd.DataFrame, label: str) -> None:
    """Print coverage + pairwise correlations for one metric flavor."""
    print(f"\n=== {label} ===")
    summary_rows = (
        df.groupby("series")
          .agg(n_runs=("auc", "size"),
               mean_auc=("auc", "mean"),
               first_eval=("eval_midpoint", "min"),
               last_eval=("eval_midpoint", "max"))
          .reindex([k for k, _, _ in TASKS])
    )
    print("Coverage:")
    print(summary_rows.to_string())

    wide = (
        df.pivot_table(index="eval_midpoint", columns="series", values="auc")
          .reindex(columns=[k for k, _, _ in TASKS])
    )
    series_keys = [k for k, _, _ in TASKS]
    print("Pairwise correlations (aligned by eval month):")
    for i in range(len(series_keys)):
        for j in range(i + 1, len(series_keys)):
            a, b = series_keys[i], series_keys[j]
            pair = wide[[a, b]].dropna()
            n = len(pair)
            if n < 2:
                print(f"  {a} vs {b}: n={n} — not enough overlap")
                continue
            pearson = pair[a].corr(pair[b], method="pearson")
            spearman = pair[a].corr(pair[b], method="spearman")
            print(
                f"  {a} vs {b}: n={n}  "
                f"pearson={pearson:+.3f}  spearman={spearman:+.3f}"
            )


def _print_head_vs_probe_correlation(
    sup_df: pd.DataFrame, probe_df: pd.DataFrame,
) -> None:
    """For each task, correlate the supervised-head AUC against the probe AUC.

    Joined on (series, eval_midpoint) so we only compare runs of the same
    training month for the same task. Expectation: the head should track its
    matched probe tightly (>0.9) — if it doesn't, the head and probe are
    measuring different things.
    """
    print("\n=== Supervised head vs. matched probe (per task) ===")
    sup = sup_df.rename(columns={"auc": "auc_head"})[
        ["series", "eval_midpoint", "auc_head"]
    ]
    probe = probe_df.rename(columns={"auc": "auc_probe"})[
        ["series", "eval_midpoint", "auc_probe"]
    ]
    merged = sup.merge(probe, on=["series", "eval_midpoint"], how="inner")
    for series_key, _project, _probe_key in TASKS:
        pair = merged[merged["series"] == series_key]
        n = len(pair)
        if n < 2:
            print(f"  {series_key}: n={n} — not enough overlap")
            continue
        pearson = pair["auc_head"].corr(pair["auc_probe"], method="pearson")
        spearman = pair["auc_head"].corr(pair["auc_probe"], method="spearman")
        print(
            f"  {series_key}: n={n}  "
            f"pearson={pearson:+.3f}  spearman={spearman:+.3f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--entity", default=DEFAULT_WANDB_ENTITY,
        help=f"W&B entity (default: {DEFAULT_WANDB_ENTITY})",
    )
    ap.add_argument(
        "--refresh", action="store_true",
        help="Re-pull summaries for every run, including finished ones.",
    )
    ap.add_argument(
        "--cache", type=Path, default=CACHE_PATH,
        help=f"Path to the API cache JSON (default: {CACHE_PATH}).",
    )
    ap.add_argument(
        "--workers", type=int, default=16,
        help="Thread-pool size for parallel summary pulls.",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Output figure path stem; written as {stem}.{png,pdf}. Giving "
             "one selects a SINGLE figure -- plain, or the highlighted one "
             "if --highlight-panel is also passed -- because a stem cannot "
             "name two. Default (no --out): BOTH are written, to "
             "full_data_multihead[_head] and ..._highlighted.",
    )
    ap.add_argument(
        "--verbose-projects", action="store_true",
        help="Print the per-project api/refresh/cached line for every bundle "
             "project. The sweep spans ~51 of them at 4 months/job, so three "
             "tasks would emit ~150 such lines; off by default and the "
             "per-project backfill/skip warnings still print either way.",
    )
    ap.add_argument(
        "--highlight-panel", "--highlight-sweep32", action="store_true",
        dest="highlight_panel",
        help="With --out, write the HIGHLIGHTED figure rather than the plain "
             "one (without --out both are written anyway, so the flag is "
             "only needed to pick one). Draw the 31 REPORTED months "
             "(sample_sweep_months.py minus "
             "metrics.SUP_SPAN_NO_SPAN) in the task colour and grey out every "
             "other month. The row's mu, y-crop and ticks are unchanged -- "
             "they still summarise the full history, so this figure is the "
             "same numbers with the reported panel picked out.",
    )
    ap.add_argument(
        "--readout", choices=("probe", "head"), default="probe",
        help="probe = ridge on the frozen embedding (xs_ic/<task>), the "
             "figure's historical content and the only readout the checked-in "
             "figure has ever shown. head = the model's OWN 11-bin softmax "
             "collapsed to its expected bin (xs_ic/head:<task>); they are "
             "different estimators, not two views of one number. A month "
             "whose checkpoint has not been re-scored since the 2026-08-29 "
             "multihead loader fix contributes NO point -- see HEAD_SCHEMA.",
    )
    args = ap.parse_args()
    # ONE FETCH, BOTH FIGURES. The plain and the highlighted versions are the
    # same numbers with different ink, and the W&B pull in front of them is
    # the whole cost of this script -- so a bare run writes both and they can
    # never be checked in a generation apart. --out names a single stem, so
    # it necessarily picks one (--highlight-panel says which).
    stem = HERE / ("full_data_multihead"
                   + ("_head" if args.readout == "head" else ""))
    if args.out is not None:
        outputs = [(args.out, args.highlight_panel)]
    else:
        outputs = [(stem, False),
                   (stem.with_name(stem.name + "_highlighted"), True)]

    import wandb  # local import — keeps dry-run / --help fast

    cache = _load_cache(args.cache)
    api = wandb.Api(timeout=600)

    # Expand the prefix once and reuse it for all three tasks -- a multihead
    # month is ONE run carrying all three xs_ic/<task> keys, so the same
    # projects are listed for each series and the API call should not be
    # repeated three times.
    projects = resolve_full_data_projects(api, args.entity)
    if not projects:
        print(f"No bundle projects found under {args.entity} matching "
              f"'{FULL_DATA_PROJECT}-<commit6>-<range>'. Nothing to plot.")
        return
    print(f"Found {len(projects)} bundle project(s) for {FULL_DATA_PROJECT}")

    sup_frames: list[pd.DataFrame] = []
    full_x: tuple[pd.Timestamp, pd.Timestamp] | None = None
    for series_key, _project, probe_key in TASKS:
        metric_key = (probe_key if args.readout == "probe"
                      else probe_key.replace("xs_ic/", "xs_ic/head:"))
        # IN HEAD MODE, FETCH ALL THREE HEAD KEYS, not just this series'.
        # fill_from_ckpt refuses a run carrying fewer than three -- the
        # signature of the pre-2026-08-29 loader, which produced a head for
        # return_900 alone -- and it can only count keys the record actually
        # holds. Requesting one would make that check unsatisfiable and blank
        # every month.
        if args.readout == "head":
            want = tuple(pk.replace("xs_ic/", "xs_ic/head:")
                         for _, _, pk in TASKS)
        else:
            want = (metric_key,)
        records: list[dict] = []
        for project in projects:
            recs = fetch_project_runs(
                api, args.entity, project, cache,
                summary_keys=want,
                history_keys=(),
                refresh=args.refresh, workers=args.workers,
                verbose=args.verbose_projects,
            )
            filled, stale = fill_from_ckpt(recs, project, metric_key)
            if filled:
                print(f"  {project}: filled {filled} runs' {metric_key} "
                      f"from the checkpoint tree")
            if stale:
                print(f"  {project}: {stale} checkpoint(s) carry a "
                      f"pre-2026-08-29 head and were SKIPPED — those months "
                      f"have no point until the re-score merges "
                      f"(merge_score_results.py)")
            records.extend(recs)

        records, off_recipe = keep_current_recipe(records)
        if off_recipe:
            print(f"  {series_key}: dropped {off_recipe} run(s) not trained on "
                  f"the current {recipe_span_months()}-month span")
        records, dup = dedupe_by_train_month(records)
        if dup:
            print(f"  {series_key}: dropped {dup} duplicate month-run(s) "
                  f"across bundle projects (kept the newest per month)")
        cand = candidate_midpoints(records)
        if cand:
            full_x = (cand[0], cand[-1])
        sup_frames.append(records_to_dataframe(records, series_key, metric_key))

    _save_cache(args.cache, cache)

    sup_df = pd.concat(sup_frames, ignore_index=True)
    if sup_df.empty:
        print("No runs with xs_ic found yet — nothing to plot.")
        return

    apply_style()
    if any(highlight for _, highlight in outputs):
        hit = sorted({_train_month(n) for n in
                      sup_df.loc[sup_df["in_panel"], "run_name"]})
        missing = sorted(PANEL_TRAIN_MONTHS - set(hit))
        print(f"Highlighting {len(hit)}/{len(PANEL_TRAIN_MONTHS)} reported "
              f"panel months present in this figure")
        if missing:
            print(f"  panel months with no full-history run yet (train "
                  f"month): {', '.join(missing)}")
    for out_path, highlight in outputs:
        _plot_curves(sup_df, out_path=out_path, full_x=full_x,
                     highlight_panel=highlight)
    _print_diagnostics(sup_df, (
        "Supervised ridge probe — cross-sectional rank IC (xs_ic/<task>)"
        if args.readout == "probe" else
        "Supervised HEAD expected-bin — cross-sectional rank IC "
        "(xs_ic/head:<task>, the model's own 11-bin softmax)"))


if __name__ == "__main__":
    main()
