"""Shared plotting style + wandb fetch helpers for scripts under plots/.

Sibling scripts (PEP 723) can import this via:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from style import (
        apply_style, save_figure, load_baselines, MODEL_COLORS,
        fetch_wandb_runs, attach_local_pkl, runs_to_dataframe, WandbRun,
    )

``fetch_wandb_runs`` is the canonical downloader: given a project-name
prefix, it enumerates W&B projects whose names end in a ``-YYYY-MM-01-…``
month suffix, filters to a requested month list (or the sweep months from
``scripts/python/data/sample_sweep_months.py``), pulls histories / summaries
with caching, parses train/eval months and run-name config fields, and
optionally subtracts the per-eval-month random-init baseline.
"""
from __future__ import annotations

import calendar
import importlib.util
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Iterable, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt  # noqa: F401  (type-only)
    import pandas as pd


# Canonical figure widths for paper figures. We fix only the width so
# matplotlib renders fonts at the same point size across plots (nothing has
# to be rescaled in the LaTeX source); each figure picks its own height.
WIDTH_FULL: float = 7.0   # spans the full text width
WIDTH_HALF: float = 3.5   # wrapped / two-per-row


RC_PARAMS = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
}

# Smaller-fonts override for compact figures (e.g. WIDTH_HALF subfigures).
# Use ``apply_style(extra=COMPACT_RC_PARAMS)`` so half-width plots keep a
# single source of truth and stay consistent with each other.
COMPACT_RC_PARAMS = {
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
}


MODEL_COLORS = {
    "vit": "#1f77b4",
    "transformer": "#1f77b4",
    "resnet": "#d62728",
    "effnet": "#2ca02c",
    "efficientnet": "#2ca02c",
}


# Canonical {color, label} per model family. Reference these from any plot
# script's SERIES list so the same family always renders the same way across
# paper figures (LeJEPA always blue, Multihead always red, etc.). Markers are
# intentionally not stored here — every series uses a plain dot ('o') except
# per-panel head-overlay stars, which are wired up at the call site.
SERIES_STYLES: dict[str, dict[str, str]] = {
    # Cross-paper model families (used by both panels of plots/metrics).
    "lejepa":            {"color": "tab:blue",   "label": "LeJEPA RRC"},
    # The untrained floor, drawn as a SERIES on absolute-metric figures where
    # there is no subtraction to make it implicit. Black dashed rather than a
    # tab10 hue: it is a reference level, not another method, and gray is
    # already the supervised specialist.
    "randinit":          {"color": "black", "label": "Random-init ViT",
                          "linestyle": "--"},
    # Model-free reference: the change target's own level, reverting. Dotted
    # for the same reason randinit is dashed — it is not a learned method.
    "mean_reversion":    {"color": "tab:orange",
                          "label": "Mean reversion (view)", "linestyle": ":"},
    "supervised_return": {"color": "tab:purple", "label": "Return"},
    "supervised_vol":    {"color": "tab:green",  "label": "Vol Change"},
    "supervised_spread": {"color": "tab:orange", "label": "Spread Change"},
    "multihead":         {"color": "tab:red",    "label": "Multihead"},
    # Baseline families use plain tab10 hues, assigned per figure. Colors ARE
    # reused across figures (e.g. TS2Vec and TimesFM are both tab:orange) —
    # only tab:gray (supervised specialist) and tab:blue (LeJEPA) are
    # reserved paper-wide; every figure just has to be internally distinct.
    # MAE stays tab:brown because plots/concat draws it beside the per-task
    # supervised purple/green/orange.
    "mae":               {"color": "tab:brown",  "label": "MAE"},
    "cpc":               {"color": "tab:orange", "label": "CPC"},
    "ijepa":             {"color": "tab:green",  "label": "I-JEPA"},
    "dino":              {"color": "tab:purple", "label": "DINO"},
    "byol":              {"color": "tab:red",    "label": "BYOL"},
    # Reviewer-rebuttal time-series SSL baselines (sweeps/baselines_final.sh).
    "ts2vec":            {"color": "tab:orange", "label": "TS2Vec"},
    "cost":              {"color": "tab:green",  "label": "CoST"},
    "tfc":               {"color": "tab:red",    "label": "TF-C"},
    "timemae":           {"color": "tab:purple", "label": "TimeMAE"},
    # Frozen pretrained TSFMs (mode=pretrained_tsfm, sweeps/tsfm_final.sh).
    # Chronos-2 gets green (not orange) so the hero — which carries both
    # TS2Vec and Chronos-2 — stays internally distinct.
    "chronos2":          {"color": "tab:green",  "label": "Chronos-2"},
    "timesfm":           {"color": "tab:orange", "label": "TimesFM 2.5"},
    # TimesFM 3.0 is a SHADE of 2.5's orange, not a fifth hue: the two are one
    # model family, and no tab10 leftover survives a five-way CVD check beside
    # cyan/orange/red/purple (the best, tab:olive, collapses to dE 3.5 under
    # deuteranopia). This deep burnt orange leaves the existing four's floors
    # untouched -- normal 18.6 and deutan 14.4 both unchanged, protan 14.9 --
    # and clears 11.9:1 contrast on white.
    "timesfm3":          {"color": "#5c2800",   "label": "TimesFM 3.0"},
    "sundial":           {"color": "tab:red",    "label": "Sundial"},
    "kronos":            {"color": "tab:purple", "label": "Kronos-base"},
    # forward_eval_v2 + market_shocks lambda variants + supervised probe/head
    # split. All lines solid; variants disambiguate by *shade* (LeJEPA: lighter
    # blue for λ=0.005, darker blue for λ=0.1) and by *marker* (Supervised:
    # dots for the probe, stars for the head — the star marker is applied at
    # the call site since SERIES_STYLES intentionally doesn't store markers,
    # see SERIES_MARKER above). Neither lambda uses the exact ``tab:blue``
    # owned by the cross-paper ``lejepa`` entry, to keep that color reserved.
    "lejepa_lamb_low":   {"color": "#6fa8dc",    "label": r"$\lambda$=0.005",    "linestyle": "-"},
    "lejepa_lamb_high":  {"color": "#1a3d6e",    "label": r"$\lambda$=0.1",      "linestyle": "-"},
    "supervised_probe":  {"color": "tab:purple", "label": "Supervised w/ Probe", "linestyle": "-"},
    "supervised_head":   {"color": "tab:purple", "label": "Supervised w/ Head",  "linestyle": "-"},
    # LeJEPA POSITIVE-PAIR variants (all_metrics_augs_ic): the cross-stock
    # pair keeps the orange/red it has in the embedding-geometry figures; the
    # same-stock augmentations take the tab10 hues those figures leave free.
    # ``lejepa_rrc`` is the same-stock CONTROL — two crops of one ticker-day
    # and nothing else — so it gets the reserved LeJEPA blue: warp and noise
    # compose with that crop rather than replacing it, and the whole figure
    # is LeJEPA, so blue marks the arm the others are read against.
    # volj / pricej / chdrop were never re-run under the IC metric; their
    # entries survive only as colors for plots/latent_eval/factors.
    "lejepa_rrc":        {"color": "tab:blue",   "label": "Same Stock Views"},
    "cross_stock_k2":    {"color": "tab:orange", "label": "Cross-Stock"},
    "cross_stock_k2ind": {"color": "tab:red",    "label": "C-S Same Ind."},
    "aug_warp":          {"color": "tab:green",  "label": "Time Warp"},
    "aug_noise":         {"color": "tab:brown",  "label": "Noising"},
    "aug_volj":          {"color": "tab:pink",   "label": "Volume noise"},
    "aug_pricej":        {"color": "tab:cyan",   "label": "Price jitter"},
    "aug_chdrop":        {"color": "tab:olive",  "label": "Channel drop"},
}

# Default marker for every series line. Stars (used as head-overlay points
# in plots/metrics) are the only deliberate exception, applied at the call site.
SERIES_MARKER: str = "o"


METRIC_LABEL = r"Mean $\Delta$AUC"


# Shared placement for the centered horizontal legend that sits beneath the
# axes (used by forward_eval.py + plots/metrics/metrics.py so the legend lands at the
# same vertical offset regardless of how many panels the figure has).
#
# A 2-row legend extends higher up than a 1-row legend, so each needs its own
# bottom reserve to keep a consistent gap between the xlabel and the legend's
# top edge. ``add_bottom_legend`` infers the row count from labels / ncol and
# picks the right default; pass ``bottom_reserve`` to override.
LEGEND_BOTTOM_RESERVE_1ROW: float = 0.25
LEGEND_BOTTOM_RESERVE_2ROW: float = 0.31
LEGEND_BOTTOM_ANCHOR: float = -0.01  # bbox_to_anchor y for fig.legend
                                     # (negative = below figure; bbox_inches="tight" includes it)


def add_bottom_legend(
    fig,
    handles=None,
    labels=None,
    *,
    ncol: int | None = None,
    frameon: bool = False,
    bottom_reserve: float | None = None,
    **legend_kw,
):
    """Centered horizontal legend below the axes, consistent across figures.

    Reserves bottom space via ``subplots_adjust`` then anchors a figure-level
    legend at the shared offset. Call AFTER ``fig.tight_layout()`` so the
    bottom reserve isn't overwritten.

    If ``handles``/``labels`` are omitted, gathers them across every axis in
    ``fig`` and dedupes by label. ``bottom_reserve`` defaults to
    ``LEGEND_BOTTOM_RESERVE_1ROW`` for single-row legends and
    ``LEGEND_BOTTOM_RESERVE_2ROW`` for 2+-row legends (inferred from
    ``len(labels) / ncol``); pass an explicit value to override.

    Extra ``legend_kw`` (e.g. ``columnspacing``, ``handletextpad``) pass
    straight through to ``fig.legend`` — use them to tighten a wide single-row
    legend without touching the shared defaults.
    """
    if handles is None or labels is None:
        seen: dict[str, object] = {}
        for ax in fig.axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    seen[l] = h
        handles = list(seen.values())
        labels = list(seen.keys())
    if not labels:
        return None
    if ncol is None:
        ncol = len(labels)
    if bottom_reserve is None:
        nrows = math.ceil(len(labels) / ncol)
        bottom_reserve = (
            LEGEND_BOTTOM_RESERVE_2ROW if nrows >= 2 else LEGEND_BOTTOM_RESERVE_1ROW
        )
    fig.subplots_adjust(bottom=bottom_reserve)
    return fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, LEGEND_BOTTOM_ANCHOR),
        ncol=ncol, frameon=frameon, **legend_kw,
    )


def apply_style(extra: dict | None = None) -> None:
    """Apply the shared rcParams. Pass `extra` to override/add keys."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC_PARAMS)
    if extra:
        plt.rcParams.update(extra)


# Y-tick step ladder: the ONLY spacings a shared-style axis may use. Every
# entry is a whole number of hundredths, so no tick can ever need a third
# decimal to print exactly -- which is the whole point. The plain 1/2/5 ladder
# matplotlib uses is what produced the 0.000/0.025/0.050/0.075 axis on the
# Volatility Change panel: 0.025 is a 2.5-step, and 0.005 would be a 1/2/5
# step at a scale finer than we ever want to read.
Y_TICK_STEPS: tuple[float, ...] = (
    0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0,
)


def set_two_decimal_yticks(ax, nbins: int = 5) -> None:
    """Put ``ax``'s y ticks on a grid no finer than 0.01, labeled to match.

    A drop-in for ``ax.locator_params(axis="y", nbins=nbins)``, which chooses
    from matplotlib's 1/2/5/2.5 ladder at whatever scale fits and so happily
    lands on 0.025 or 0.005 — three decimals of axis furniture on figures
    whose panels are two inches wide.

    ``nbins`` is a ceiling on INTERVALS, and wants to be one or two higher
    than the same request to matplotlib: this ladder has no step between 0.02
    and 0.05, so a tight budget rounds up hard and leaves a tall-ish panel
    with two ticks on it.

    The step is chosen at DRAW time, not here: several callers add an
    ``axhline`` or an autoscale after styling an axis, and a locator that had
    frozen its ticks against the earlier limits would quietly disagree with
    the data. The label format is fixed here instead, since it depends only
    on the ladder — two decimals below a 0.1 step, one at or above it.
    """
    import matplotlib.ticker as mticker

    class _Locator(mticker.Locator):
        def __call__(self):
            return self.tick_values(*self.axis.get_view_interval())

        def tick_values(self, vmin, vmax):
            vmin, vmax = sorted((vmin, vmax))
            span = vmax - vmin
            step = next(
                (s for s in Y_TICK_STEPS if span / s <= nbins),
                Y_TICK_STEPS[-1],
            )
            # A hair of tolerance on each end: a limit sitting exactly on a
            # tick is a float compare away from dropping that tick entirely.
            lo = math.ceil(vmin / step - 1e-9)
            hi = math.floor(vmax / step + 1e-9)
            return [i * step for i in range(lo, hi + 1)]

    ax.yaxis.set_major_locator(_Locator())
    # Decided off the SPAN rather than off the chosen step, so the two panels
    # of one figure cannot end up with differently-formatted labels when
    # their limits differ slightly. Same ladder, same question.
    span = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    step = next((s for s in Y_TICK_STEPS if span / s <= nbins), Y_TICK_STEPS[-1])
    ax.yaxis.set_major_formatter(
        mticker.FormatStrFormatter("%.2f" if step < 0.1 else "%.1f")
    )


def save_figure(
    fig,
    path: str | Path,
    *,
    dpi: int = 300,
    formats: tuple[str, ...] = ("png", "pdf"),
    bbox_inches: str = "tight",
    pad_inches: float = 0.05,
) -> list[Path]:
    """Save `fig` to `path` in each of `formats` at a consistent high DPI.

    ``pad_inches=0.05`` (default) leaves a thin margin around the outermost
    artist on every side. Override per-call if a figure needs more or less
    breathing room outside the artists.

    The suffix on `path` is ignored; one file is written per entry in `formats`.
    Returns the list of written paths.
    """
    base = Path(path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for fmt in formats:
        out = base.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)
        out_paths.append(out)
    return out_paths


def model_color(name: str, default: str = "#7f7f7f") -> str:
    """Look up a model color by short name (case-insensitive)."""
    return MODEL_COLORS.get(name.lower(), default)


BASELINE_PATH = Path(__file__).resolve().parent / "baseline.json"


def load_baselines(target: str = "return_900_k5") -> dict[str, float]:
    """Per-eval-month random-init probe AUC, {YYYY-MM: mean}.

    The baseline is the mean probe AUC across random-init ViT (transformer)
    seeds for each eval month — one universal baseline applied to every
    model.

    Scored no-clamp, the only regime the codebase runs: a row counts only if
    its forward window fits before the close (slack >= h). There is no
    clamped counterpart to pick between — the separate baseline_noclamp.json
    was folded into this file on 2026-08-19, since the subtraction is only
    meaningful when baseline and model share a row filter.

    This is a delta-AUC artifact and has no live producer: the AUC pipeline
    was retired with the IC switch, and the IC-era floor is measured per run
    by ``scripts/eval/xs_ic_eval.py --random-init-seeds``.
    """
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"{BASELINE_PATH} not found; it is a checked-in delta-AUC "
            "artifact with no producer left in the tree."
        )
    data = json.loads(BASELINE_PATH.read_text())
    if target not in data:
        have = [k for k in data if not k.startswith("__")]
        raise KeyError(f"target {target!r} not in baseline.json; have: {have}")
    return {em: float(b["mean"]) for em, b in data[target].items()}


# ─── W&B project / run helpers ───────────────────────────────────────────────

DEFAULT_WANDB_ENTITY = os.environ.get("WANDB_ENTITY")  # None: your default W&B entity

# A sweep project name ends in "-YYYY-MM-01-YYYY-MM-{last}". The leading prefix
# may contain dashes and an optional commit-hash segment (e.g.
# ``lejepa-projector-ablations-{hash}-2023-10-01-2023-10-31``).
_MONTH_SUFFIX_RE = re.compile(
    r"(?P<year>\d{4})-(?P<mon>\d{2})-01-(?P<ey>\d{4})-(?P<em>\d{2})-(?P<ed>\d{2})$"
)


# A bundled run names its own month: "2008-03_multi_pairwise_s42".
_RUN_MONTH_RE = re.compile(r"(?P<ym>\d{4}-\d{2})_")


def load_sweep_months() -> list[str]:
    """Canonical cross-month sweep months from ``scripts/experiments/sample_sweep_months.py``."""
    here = Path(__file__).resolve().parent
    script = here.parent / "scripts" / "experiments" / "sample_sweep_months.py"
    spec = importlib.util.spec_from_file_location("sample_sweep_months", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return list(mod.SWEEP_MONTHS)


def month_suffix(ym: str) -> str:
    """``'2023-10'`` → ``'-2023-10-01-2023-10-31'`` (calendar-last for the month)."""
    y, m = map(int, ym.split("-"))
    last = calendar.monthrange(y, m)[1]
    return f"-{y:04d}-{m:02d}-01-{y:04d}-{m:02d}-{last:02d}"


def project_train_month(project: str) -> str | None:
    """Extract the training month (``'YYYY-MM'``) from a sweep project name.

    Returns None if the project name doesn't end with the expected
    ``-YYYY-MM-01-YYYY-MM-DD`` suffix.
    """
    m = _MONTH_SUFFIX_RE.search(project)
    if not m:
        return None
    return f"{int(m.group('year')):04d}-{int(m.group('mon')):02d}"


def eval_month_from_train(train_ym: str) -> str:
    """Train month + 1 (wraps year). Eval month = month immediately after training."""
    y, m = map(int, train_ym.split("-"))
    m += 1
    if m > 12:
        m = 1
        y += 1
    return f"{y:04d}-{m:02d}"


@dataclass
class WandbRun:
    """One W&B run, post-fetch. ``history`` is None if ``history_keys`` was empty."""

    project: str
    run_id: str
    run_name: str
    state: str
    train_month: str | None           # 'YYYY-MM' from project suffix
    eval_month: str | None            # train_month + 1
    train_start: str | None           # 'YYYY-MM-DD' start of training window
    train_end: str | None             # 'YYYY-MM-DD' end of training window
    config: dict[str, Any]            # parsed from run_name_re named groups
    summary: dict[str, Any]           # slice of run.summary keyed by summary_keys
    history: "pd.DataFrame | None"    # obs_seen + requested history_keys
    baseline: float | None = None     # subtracted from metric if apply_baseline=True


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


def _parse_run_name(
    run_name: str, pattern: "re.Pattern[str] | None",
) -> dict[str, Any] | None:
    if pattern is None:
        return {}
    m = pattern.match(run_name)
    if not m:
        return None
    out: dict[str, Any] = {}
    for k, v in m.groupdict().items():
        if v is None:
            out[k] = None
            continue
        # Auto-coerce numeric strings.
        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(v)
            continue
        except ValueError:
            pass
        out[k] = v
    return out


def _get_dotted(d: Any, path: str) -> Any:
    """Return ``d["a"]["b"]...`` for ``path="a.b..."``, or None if missing.

    Tolerates dict / list / object-with-attrs nesting, returning None on the
    first segment that can't be resolved (rather than raising).
    """
    cur = d
    for seg in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            if seg not in cur:
                return None
            cur = cur[seg]
            continue
        try:
            idx = int(seg)
            cur = cur[idx]
            continue
        except (ValueError, TypeError, IndexError, KeyError):
            pass
        cur = getattr(cur, seg, None)
    return cur


def _fetch_one_run_payload(
    run,
    history_keys: Sequence[str],
    summary_keys: Sequence[str],
    config_keys: Sequence[str],
    history_outer: bool,
) -> dict:
    """Pull history + summary + config slices. Network-bound; thread-safe.

    ``history_outer=False`` (default): one ``scan_history(keys=[obs_seen, *H])``
    call — inner-joins across all keys, so rows missing ANY key are dropped.
    Right when all keys are logged at the same cadence (e.g. probe AUC variants).

    ``history_outer=True``: one scan per key, merged on ``obs_seen`` with
    missing entries as None. Right for mixed-cadence streams like
    dense ``train/loss_avg`` + sparse ``probe/logistic_auc_*``.
    """
    payload: dict = {"summary": {}, "config": {}}
    for key in summary_keys:
        payload["summary"][key] = run.summary.get(key)
    cfg = dict(run.config or {}) if config_keys else {}
    for key in config_keys:
        payload["config"][key] = _get_dotted(cfg, key)
    if history_keys:
        try:
            if not history_outer:
                hist_rows = list(
                    run.scan_history(keys=list({"obs_seen", *history_keys}))
                )
            else:
                merged: dict[int, dict] = {}
                for k in history_keys:
                    for row in run.scan_history(keys=["obs_seen", k]):
                        obs = row.get("obs_seen")
                        val = row.get(k)
                        if obs is None or val is None:
                            continue
                        merged.setdefault(int(obs), {"obs_seen": int(obs)})[k] = val
                hist_rows = [merged[o] for o in sorted(merged)]
        except Exception as e:
            payload["history_error"] = f"{type(e).__name__}: {e}"
            hist_rows = []
        payload["history"] = hist_rows
    return payload


def fetch_wandb_runs(
    project_prefix: str,
    *,
    entity: str | None = DEFAULT_WANDB_ENTITY,
    months: Sequence[str] | str | None = None,
    run_name_re: "str | re.Pattern[str] | None" = None,
    history_keys: Sequence[str] = (),
    history_outer: bool = False,
    summary_keys: Sequence[str] = (),
    config_keys: Sequence[str] = (),
    state_filter: Sequence[str] | None = ("finished",),
    apply_baseline: bool = True,
    baseline_target: str = "return_900_k5",
    baseline_metric: str = "probe/logistic_auc_return_900_k5",
    cache_path: str | Path | None = None,
    dedupe_by_config: bool = True,
    workers: int = 16,
    verbose: bool = True,
    expected_runs_per_project: int | None = None,
) -> list[WandbRun]:
    """Download every matching run under ``{entity}/{project_prefix}*``.

    Matches projects whose names start with ``project_prefix`` and end with
    a ``-YYYY-MM-01-YYYY-MM-DD`` month suffix. ``months`` is a list of
    ``YYYY-MM`` train months; ``None`` → canonical sweep months; pass
    ``"all"`` to accept every suffix-valid project.

    ``run_name_re`` is compiled and matched against each run's ``display_name``
    — runs that don't match are dropped, and named groups populate
    ``WandbRun.config``. Groups that look numeric are auto-coerced to int/float.

    ``config_keys`` is a list of dotted paths into ``run.config`` (e.g.
    ``["optimizer.blr", "mode.target_scale.0"]``). Resolved values land in
    ``WandbRun.config`` under the full dotted path as the key; missing paths
    yield ``None``. Combine with ``run_name_re`` freely — named groups and
    config keys share ``WandbRun.config``.

    If ``apply_baseline`` is True, the per-eval-month baseline (from
    ``plots/baseline.json[baseline_target]``) is subtracted from
    ``baseline_metric`` in-place on each run's ``history`` column and the
    matching ``summary`` key; runs whose ``eval_month`` has no baseline entry
    are dropped with a warning.

    ``cache_path`` enables a JSON cache keyed by ``{project}/{run_id}``. The
    W&B API is always queried for the project/run list (so new runs are
    picked up automatically), but heavy data (history, summary, config) is
    only re-pulled for runs missing from the cache or whose cached entry
    lacks a requested field. To force a full re-pull, delete the cache file.

    ``dedupe_by_config`` (default True): keep one run per ``(project, config)``.
    Re-running a sweep — e.g. forward + backward halves to fill in crashes —
    can produce >1 run with identical hyperparameters in the same project;
    the first run encountered (in API / cache iteration order) wins.

    ``expected_runs_per_project`` (default None): when set, projects whose
    cache already holds at least this many *complete* entries (after
    ``state_filter`` / ``run_name_re``) skip the per-project ``api.runs()``
    listing call entirely. The dominant cost of a warm-cache refresh is
    one round-trip per project (×N projects), and once a sweep cell is
    full it never gains more runs — so for plot scripts that re-run
    against fixed-cell-count sweeps (e.g. 32 hyperparameter combinations
    per month), passing the cell count here turns repeat invocations
    from O(N) round-trips to ~zero. Leave None to keep the default
    "always discover new runs" behavior.
    """
    import pandas as pd  # noqa: F401  — for history DataFrame below
    import wandb

    if isinstance(run_name_re, str):
        run_name_re = re.compile(run_name_re)

    if months is None:
        months = load_sweep_months()
    if months != "all":
        month_set = set(months)
    else:
        month_set = None  # accept any suffix-valid project

    baselines = load_baselines(baseline_target) if apply_baseline else {}

    cache_file = Path(cache_path) if cache_path else None
    cache = _load_cache(cache_file) if cache_file else {}

    # Index cached entries by project so the per-project listing call can
    # be skipped when ``expected_runs_per_project`` is met.
    cached_by_project: dict[str, list[tuple[str, dict]]] = {}
    for _key, _entry in cache.items():
        if "/" not in _key:
            continue
        _proj_name, _ = _key.split("/", 1)
        cached_by_project.setdefault(_proj_name, []).append((_key, _entry))

    def _entry_complete(entry: dict | None) -> bool:
        if entry is None or "meta" not in entry:
            return False
        if history_keys and "history" not in entry:
            return False
        entry_summary = entry.get("summary") or {}
        if any(k not in entry_summary for k in summary_keys):
            return False
        entry_config = entry.get("config") or {}
        if any(k not in entry_config for k in config_keys):
            return False
        return True

    # Hit the W&B API to discover the project/run list so newly-created
    # runs are picked up automatically. Heavy data (history, summary,
    # config) is hydrated incrementally below: cached runs are reused,
    # only missing or under-populated entries get a fresh pull. Set
    # ``expected_runs_per_project`` to bypass the per-project ``api.runs()``
    # listing call once a project's cache has enough complete entries (the
    # dominant cost when refreshing a warm cache). Hard refresh = delete
    # the cache file.
    runs_meta: list[dict] = []
    api = wandb.Api()
    entity = entity or api.default_entity
    try:
        all_projects = [p.name for p in api.projects(entity=entity)]
    except Exception as e:
        raise RuntimeError(f"Failed to list wandb projects for {entity!r}: {e}") from e

    matched_projects: list[tuple[str, str, str, str]] = []  # (name, train_month, train_start, train_end)
    for name in all_projects:
        if not name.startswith(project_prefix):
            continue
        m = _MONTH_SUFFIX_RE.search(name)
        if m is None:
            continue
        train_month = f"{int(m.group('year')):04d}-{int(m.group('mon')):02d}"
        if month_set is not None and train_month not in month_set:
            continue
        train_start = f"{int(m.group('year')):04d}-{int(m.group('mon')):02d}-01"
        train_end = f"{int(m.group('ey')):04d}-{int(m.group('em')):02d}-{int(m.group('ed')):02d}"
        matched_projects.append((name, train_month, train_start, train_end))

    matched_projects.sort()
    if verbose:
        print(f"[fetch_wandb_runs] matched {len(matched_projects)} projects "
              f"with prefix {project_prefix!r}")

    pending: list[tuple[Any, str, str, str]] = []  # (run, project, train, cache_key)
    cache_dirty = False
    n_projects_skipped = 0

    for project, train_month, train_start, train_end in matched_projects:
        eval_month = eval_month_from_train(train_month)

        # Fast path: skip the per-project listing call when the cache
        # already holds enough complete entries (filtered by state_filter
        # / run_name_re). One round-trip × N projects is the dominant
        # cost of a warm-cache refresh, and once a sweep cell is full it
        # never gains more runs.
        if expected_runs_per_project is not None:
            usable: list[tuple[str, dict, dict]] = []
            for cache_key, entry in cached_by_project.get(project, []):
                if not _entry_complete(entry):
                    continue
                mb = entry["meta"]
                if state_filter is not None and mb.get("state") not in state_filter:
                    continue
                cfg = _parse_run_name(mb.get("run_name", ""), run_name_re)
                if cfg is None:
                    continue
                usable.append((cache_key, entry, cfg))
            if len(usable) >= expected_runs_per_project:
                n_projects_skipped += 1
                for cache_key, entry, cfg in usable:
                    mb = entry["meta"]
                    runs_meta.append({
                        "project": mb["project"],
                        "run_id": mb["run_id"],
                        "run_name": mb["run_name"],
                        "state": mb["state"],
                        "train_month": mb["train_month"],
                        "eval_month": mb["eval_month"],
                        "train_start": mb["train_start"],
                        "train_end": mb["train_end"],
                        "config": cfg,
                        "cache_key": cache_key,
                        "run_obj": None,
                    })
                continue

        try:
            project_runs = list(api.runs(f"{entity}/{project}"))
        except Exception as e:
            if verbose:
                print(f"  !! failed to list runs for {project}: {e}")
            continue

        for run in project_runs:
            if state_filter is not None and run.state not in state_filter:
                continue
            config = _parse_run_name(run.name, run_name_re)
            if config is None:
                continue
            cache_key = f"{project}/{run.id}"
            meta = {
                "project": project,
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "train_month": train_month,
                "eval_month": eval_month,
                "train_start": train_start,
                "train_end": train_end,
                "config": config,
                "cache_key": cache_key,
                "run_obj": run,
            }
            runs_meta.append(meta)

            entry = cache.get(cache_key)
            meta_block = {
                "project": project,
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "train_month": train_month,
                "eval_month": eval_month,
                "train_start": train_start,
                "train_end": train_end,
            }
            if entry is None or entry.get("meta") != meta_block:
                cache.setdefault(cache_key, {})["meta"] = meta_block
                cache_dirty = True

            need_history = bool(history_keys) and (entry is None or "history" not in entry)
            need_summary = any(
                k not in (entry.get("summary") or {} if entry else {})
                for k in summary_keys
            )
            need_config = any(
                k not in (entry.get("config") or {} if entry else {})
                for k in config_keys
            )
            if need_history or need_summary or need_config or entry is None:
                pending.append((run, project, train_month, cache_key))

    if verbose:
        msg = (
            f"[fetch_wandb_runs] {len(runs_meta)} runs matched "
            f"({len(pending)} need fresh pulls; "
            f"{len(runs_meta) - len(pending)} cache hits)"
        )
        if n_projects_skipped:
            msg += (
                f"; skipped api.runs() for {n_projects_skipped} full projects "
                f"(>= {expected_runs_per_project} cached entries each)"
            )
        print(msg)

    if pending:
        def _work(triple):
            run, _, _, key = triple
            return key, _fetch_one_run_payload(
                run, history_keys, summary_keys, config_keys, history_outer,
            )

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_work, t): t[3] for t in pending}
            for fut in as_completed(futs):
                key, payload = fut.result()
                existing = cache.get(key, {})
                existing.setdefault("summary", {}).update(payload.get("summary", {}))
                existing.setdefault("config", {}).update(payload.get("config", {}))
                if "history" in payload:
                    existing["history"] = payload["history"]
                if "history_error" in payload:
                    existing["history_error"] = payload["history_error"]
                cache[key] = existing
        cache_dirty = True

    if cache_dirty and cache_file is not None:
        _save_cache(cache_file, cache)

    # Hydrate meta["config"] with cached config_keys before dedup. Without
    # this, runs whose run_name parses to {} (i.e. callers that pass no
    # run_name_re) all hash to the same dedup key and collapse to one run
    # per project — see scripts/sweeps cells keyed by optimizer.blr /
    # mode.target_scale, which only live in the wandb config blob.
    for meta in runs_meta:
        entry_config = (cache.get(meta["cache_key"], {}) or {}).get("config") or {}
        for k in config_keys:
            meta["config"].setdefault(k, entry_config.get(k))

    # Dedupe within each project: keep first run per unique config. Sweep
    # re-runs (forward + backward halves to backfill crashes) can produce
    # duplicate cells in the same project; "first encountered" is fine —
    # the user just wants one of them.
    if dedupe_by_config:
        seen: set[tuple[str, str]] = set()
        deduped: list[dict] = []
        n_dropped = 0
        for meta in runs_meta:
            ck = (
                meta["project"],
                json.dumps(meta["config"], sort_keys=True, default=str),
            )
            if ck in seen:
                n_dropped += 1
                continue
            seen.add(ck)
            deduped.append(meta)
        if verbose and n_dropped:
            print(f"[fetch_wandb_runs] deduped {n_dropped} duplicate runs "
                  f"(same project + config; kept first encountered)")
        runs_meta = deduped

    # Assemble WandbRun objects
    out: list[WandbRun] = []
    n_missing_baseline = 0
    for meta in runs_meta:
        entry = cache.get(meta["cache_key"], {})
        entry_summary = entry.get("summary") or {}
        summary = {k: entry_summary.get(k) for k in summary_keys}

        history_df = None
        if history_keys:
            rows = entry.get("history", [])
            if rows:
                import pandas as pd
                history_df = pd.DataFrame(rows)
                keep = [c for c in history_df.columns if c in {"obs_seen", *history_keys}]
                history_df = history_df[keep]

        baseline_val: float | None = None
        if apply_baseline:
            em = meta["eval_month"]
            if em not in baselines:
                n_missing_baseline += 1
                continue
            baseline_val = baselines[em]
            if history_df is not None and baseline_metric in history_df.columns:
                history_df = history_df.copy()
                history_df[baseline_metric] = history_df[baseline_metric] - baseline_val
            if baseline_metric in summary and summary[baseline_metric] is not None:
                summary[baseline_metric] = summary[baseline_metric] - baseline_val

        out.append(WandbRun(
            project=meta["project"],
            run_id=meta["run_id"],
            run_name=meta["run_name"],
            state=meta["state"],
            train_month=meta["train_month"],
            eval_month=meta["eval_month"],
            train_start=meta["train_start"],
            train_end=meta["train_end"],
            config=meta["config"],
            summary=summary,
            history=history_df,
            baseline=baseline_val,
        ))

    if verbose and apply_baseline and n_missing_baseline:
        print(f"[fetch_wandb_runs] dropped {n_missing_baseline} runs "
              f"whose eval_month has no baseline in {baseline_target!r}")

    return out


def attach_local_pkl(
    runs: Sequence[WandbRun],
    *,
    checkpoint_root: "str | Path",
    pkl_name: str,
    fields: Sequence[str] | None = ("auc",),
    apply_baseline: bool = True,
    baseline_target: str = "return_900_k5",
    baseline_field: str = "auc",
    drop_missing_pkl: bool = True,
    verbose: bool = True,
) -> list[WandbRun]:
    """Load ``{checkpoint_root}/{project}/{run_id}/{pkl_name}`` per run.

    Extracted values land in ``run.summary`` under their original key (e.g.
    ``auc``). If ``fields`` is None, the entire pkl dict is merged into
    ``summary`` (shallow). If a field clashes with an existing wandb-side key,
    the pkl value wins — this is what lets plots swap a wandb-history AUC for
    a pkl-eval AUC without touching any downstream code.

    If ``apply_baseline`` is True, the per-eval-month baseline is subtracted
    from ``baseline_field`` in-place and stored in ``run.baseline``. Runs
    whose ``eval_month`` has no baseline entry are dropped with a warning.

    ``drop_missing_pkl=True`` (default) drops runs whose pkl file doesn't
    exist or fails to unpickle; set False to keep them (``run.summary``
    won't have the pkl fields in that case).
    """
    import pickle

    root = Path(checkpoint_root)
    baselines = load_baselines(baseline_target) if apply_baseline else {}

    out: list[WandbRun] = []
    n_missing_pkl = 0
    n_missing_baseline = 0

    for r in runs:
        pkl_path = root / r.project / r.run_id / pkl_name
        if not pkl_path.exists():
            n_missing_pkl += 1
            if drop_missing_pkl:
                continue
            out.append(r)
            continue
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            if verbose:
                print(f"  !! {pkl_path}: read error ({e})")
            if drop_missing_pkl:
                continue
            out.append(r)
            continue

        new_summary = dict(r.summary)
        if fields is None:
            if isinstance(data, dict):
                new_summary.update(data)
        else:
            for k in fields:
                new_summary[k] = data.get(k) if isinstance(data, dict) else None

        baseline_val: float | None = r.baseline
        if apply_baseline:
            em = r.eval_month
            if em is None or em not in baselines:
                n_missing_baseline += 1
                continue
            baseline_val = baselines[em]
            v = new_summary.get(baseline_field)
            if v is not None:
                try:
                    new_summary[baseline_field] = float(v) - baseline_val
                except (TypeError, ValueError):
                    pass

        out.append(WandbRun(
            project=r.project,
            run_id=r.run_id,
            run_name=r.run_name,
            state=r.state,
            train_month=r.train_month,
            eval_month=r.eval_month,
            train_start=r.train_start,
            train_end=r.train_end,
            config=r.config,
            summary=new_summary,
            history=r.history,
            baseline=baseline_val,
        ))

    if verbose:
        if n_missing_pkl:
            action = "dropped" if drop_missing_pkl else "kept (no pkl data)"
            print(f"[attach_local_pkl] {n_missing_pkl} runs with missing pkl — {action}")
        if n_missing_baseline:
            print(f"[attach_local_pkl] dropped {n_missing_baseline} runs "
                  f"whose eval_month has no baseline in {baseline_target!r}")
    return out


def render_latex_tabular(
    rows: Sequence[Sequence[str]],
    *,
    header_rows: Sequence[Sequence[str]] = (),
    col_spec: str | None = None,
    booktabs: bool = True,
) -> str:
    """Just the ``\\begin{tabular}...\\end{tabular}`` part — no outer table env.

    Use this when you want to drop the same tabular into multiple contexts
    (e.g. inside ``subtable`` blocks of a compound paper figure).
    """
    def _n_cols(seq: Sequence[Sequence[str]]) -> int:
        return max((len(r) for r in seq), default=0)

    n_cols = max(_n_cols(rows), _n_cols(header_rows))
    if n_cols == 0:
        raise ValueError("render_latex_tabular: need at least one row or header row")
    if col_spec is None:
        col_spec = "l" + "c" * (n_cols - 1)

    top = r"\toprule" if booktabs else r"\hline"
    mid = r"\midrule" if booktabs else r"\hline"
    bot = r"\bottomrule" if booktabs else r"\hline"

    lines: list[str] = [r"\begin{tabular}{" + col_spec + "}", top]
    for hdr in header_rows:
        lines.append(" & ".join(hdr) + r" \\")
    if header_rows:
        lines.append(mid)
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [bot, r"\end{tabular}"]
    return "\n".join(lines)


# CELL MARKS FOR THE PAPER TABLES, by colour instead of \sout/bold: a struck
# number is hard to read and a reader has to be told what the strike means.
# Red is "no better than the untrained floor", green is "best in its row".
# Both are darkened so they stay legible in print; bold still rides with the
# green, so the best cells survive greyscale and red-green colour blindness.
TEX_WORSE = "red!75!black"
TEX_BEST = "green!45!black"


def tex_mark(num: str, *, best: bool = False, worse: bool = False) -> str:
    """One number as ``$...$``, coloured. ``worse`` wins: a cell below the
    floor is not evidence of anything, so it is never also marked best."""
    if worse:
        return r"\textcolor{" + TEX_WORSE + "}{$" + num + "$}"
    if best:
        return r"\textcolor{" + TEX_BEST + r"}{$\mathbf{" + num + "}$}"
    return "$" + num + "$"


def tex_escape(s: str) -> str:
    """Escape the LaTeX specials that occur in a row label.

    Deliberately NOT applied to anything the caller means as markup: a
    \\citep or a math group must be appended AFTER this, or the backslash
    becomes \\textbackslash{} and the macro prints instead of running.
    """
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("_", r"\_")):
        s = s.replace(a, b)
    return s


def render_latex_table(
    rows: Sequence[Sequence[str]],
    *,
    header_rows: Sequence[Sequence[str]] = (),
    col_spec: str | None = None,
    caption: str | None = None,
    label: str | None = None,
    title: str | None = None,
    small: bool = True,
    size: str | None = None,
    pre: Sequence[str] = (),
    post: Sequence[str] = (),
    caption_above: bool = False,
    booktabs: bool = True,
) -> str:
    """Render a LaTeX ``table`` environment from pre-formatted string cells.

    Wraps ``render_latex_tabular`` with ``table``/``centering``/``caption``
    boilerplate. ``size`` overrides ``small`` with a named size macro and
    ``pre`` adds lines just after it (e.g. a narrower ``\\tabcolsep``) and
    ``post`` adds lines just before ``\\end{table}`` (e.g. a ``\\vspace``). Cells are emitted as-is — format numbers, escape LaTeX
    specials, and wrap math (``$...$``) in the caller.
    """
    # CAPTION ABOVE, for a table. The float's caption counter does not care,
    # but the convention for tables is above and for figures below, and a
    # caption emitted after the tabular has to be moved by hand every time the
    # file is regenerated. It goes before \centering so it sets at full width.
    lines: list[str] = [r"\begin{table}[t]"]
    if caption_above and caption:
        lines.append(r"\caption{" + caption + "}")
        if label:
            lines.append(r"\label{" + label + "}")
    lines.append(r"\centering")
    # ``size`` names the LaTeX size macro outright ("footnotesize", "scriptsize"
    # ...) and overrides ``small``; ``pre`` is any further setup that has to sit
    # inside the float, such as \setlength{\tabcolsep}. Both default to the
    # old behaviour, so existing callers render byte-identically.
    if size:
        lines.append("\\" + size)
    elif small:
        lines.append(r"\small")
    lines.extend(pre)
    if title:
        lines.append(title + r" \\[2pt]")
    lines.append(render_latex_tabular(
        rows, header_rows=header_rows, col_spec=col_spec, booktabs=booktabs,
    ))
    if caption and not caption_above:
        lines.append(r"\caption{" + caption + "}")
    if label and not (caption_above and caption):
        lines.append(r"\label{" + label + "}")
    lines.extend(post)
    lines.append(r"\end{table}")
    return "\n".join(lines)


def runs_to_dataframe(
    runs: Sequence[WandbRun], *, include_history: bool = True,
) -> "pd.DataFrame":
    """Flatten runs into one long DataFrame.

    If ``include_history`` and runs have histories, emits one row per
    history checkpoint with run metadata repeated. Otherwise emits one row
    per run (summary-only). Config fields land in columns prefixed with
    ``cfg_``; summary fields in columns prefixed with ``sum_``.
    """
    import pandas as pd

    rows: list[dict] = []
    for r in runs:
        base = {
            "project": r.project,
            "run_id": r.run_id,
            "run_name": r.run_name,
            "state": r.state,
            "train_month": r.train_month,
            "eval_month": r.eval_month,
            "baseline": r.baseline,
        }
        base.update({f"cfg_{k}": v for k, v in r.config.items()})
        base.update({f"sum_{k}": v for k, v in r.summary.items()})

        if include_history and r.history is not None and len(r.history):
            for _, hist_row in r.history.iterrows():
                row = dict(base)
                row.update(hist_row.to_dict())
                rows.append(row)
        else:
            rows.append(base)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rank-IC readers (the ΔAUC → ΔIC migration).
#
# IC is NOT fetched from wandb. ``post_train_ic_eval`` writes ``xs_ic.json``
# next to the checkpoint it scored, and the checkpoint tree rsyncs back from
# the cluster — so the file on disk is the record, and it is there whether or
# not the run's wandb sync succeeded. (Wandb's listing state also lies in both
# directions, which is a second reason not to route this through it.)
# ---------------------------------------------------------------------------

CKPT_ROOT = Path("lab/market-jepa-checkpoints")
SCORE_RESULTS = Path("lab/score_results")
# WHICH GENERATION OF RANDOM-INIT FLOOR load_randinit_ic reads. The tag is the
# only thing distinguishing a forward-window floor from a retired midpoint one
# -- the result files themselves carry no target stamp. See load_randinit_ic.
RANDINIT_TAG = "randinit-fwd3"

IC_METRIC_LABEL = r"Mean $\Delta$IC"

# Mirrors merge_score_results.HEAD_SCHEMA / rescore_heads.HEAD_SCHEMA. Stamped
# by whichever path re-scored a checkpoint's head after the 2026-08-29
# multihead loader fix. See ICRun.head for why only MULTIHEAD checkpoints are
# held to it.
HEAD_SCHEMA = 2


def _next_month(ym: str) -> str:
    """'2013-12' -> '2014-01'."""
    y, m = (int(x) for x in ym.split("-")[:2])
    return f"{y + (m == 12):04d}-{(m % 12) + 1:02d}"


@dataclass
class ICRun:
    """One scored checkpoint: its identity, its metadata, and its ICs."""

    project: str
    run_id: str
    path: Path
    train_month: str          # the month the encoder trained on
    eval_month: str           # the month it was scored on (train + 1)
    meta: dict[str, Any]      # train_meta.json
    ic: dict[str, Any]        # xs_ic.json

    @property
    def run_name(self) -> str:
        return str(self.meta.get("run_name", ""))

    def probe(self, task: str, metric: str = "ic") -> float | None:
        """Ridge-probe IC (or logistic-probe AUC) for ``task``."""
        return _finite(self.ic.get(f"xs_{metric}/{task}"))

    @property
    def is_multihead(self) -> bool:
        """Does this checkpoint save ``heads.pt`` (a multi-task trunk)?

        The distinction matters because the 2026-08-29 loader bug was specific
        to that layout: a single-task run saves ``head.pt`` and always loaded
        correctly, so its head numbers were never wrong and must not be gated.
        """
        return (self.path / "heads.pt").is_file()

    def head(self, task: str, metric: str = "ic") -> float | None:
        """The head's own readout, for checkpoints that have a head.

        ``ic`` is the expected-bin rank IC. ``auc`` is the macro one-vs-rest
        AUC of the head's softmax marginalized onto the reported 5-bin
        partition, so it may be read across a figure whose axis is k;
        ``auc_native`` is the same softmax scored at the head's OWN k, which
        may not.

        A MULTIHEAD CHECKPOINT WITHOUT THE SCHEMA STAMP RETURNS None. Until
        2026-08-29 load_model rebuilt every ``heads.pt`` trunk as a single-task
        SupervisedModel, found no ``head.pt``, RANDOMLY INITIALIZED the head,
        and let the scorer write the result out -- so the head keys sitting
        beside an un-re-scored trunk are noise, on the ``return_900`` default
        only. That is what once put a multihead star below the zero line on
        the Return panel. Key presence cannot tell the two apart, so the stamp
        is the only test, and it is applied HERE rather than per figure
        because every consumer of a head number reaches it through this one
        accessor. Single-task checkpoints are exempt -- see ``is_multihead``.
        """
        if self.is_multihead and int(
                self.ic.get("xs_head_schema") or 0) < HEAD_SCHEMA:
            return None
        return _finite(self.ic.get(f"xs_{metric}/head:{task}"))


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _train_end_month(meta: dict) -> str | None:
    """The YYYY-MM a run's training window ENDS on, from its own config.

    Authoritative for a span run, where neither the project suffix nor a bare
    month elsewhere says which end is meant. None when the config does not
    record the window (older trees), so callers fall back.
    """
    cfg = meta.get("config", meta)
    ds = cfg.get("dataset", {}) if isinstance(cfg, dict) else {}
    end = str(ds.get("train_date_end") or "")
    return end[:7] if len(end) >= 7 and end[4] == "-" else None


def iter_ic_runs(
    project_glob: str,
    ckpt_root: Path | str = CKPT_ROOT,
    xs_stats: str | None = None,
    xs_target: str | None = None,
    train_months: Collection[str] | None = None,
) -> list[ICRun]:
    """Every scored run under ``<ckpt_root>/<project_glob>/<run_id>/``.

    Runs with no ``xs_ic.json`` are skipped, so a sweep still in flight simply
    yields fewer records — callers report coverage rather than waiting.

    A RUN NAME DOES NOT PIN THE RECIPE, which is what the filters below are
    for. Run names encode the arm and its hyperparameter but not the recipe,
    and a project glob matches every commit hash, so a run trained under a
    superseded recipe survives under the same glob and the same name. A
    pre-``eb22cac`` ``k2_lamb0.005_bs256_s42`` carrying ``xs_target=zscore``
    simply BECAME 2009-06's lambda=0.005 number: it moved that cell from
    +0.0000 to −0.0019 on return and flipped a significance call on spread.
    Filter on the config, not the name.

    ``xs_stats`` restricts to one TARGET DEFINITION, by the anchor-stat table
    basename ``save_train_meta`` records. All three targets became
    forward-window differences on 2026-08-22 (docs/return_bad_calculation.md),
    so a glob that spans the change would average two different quantities into
    one cell. A checkpoint written before the stamp existed reports
    ``xs_anchor_stats``, which is what it in fact used.

    ``xs_target`` restricts to one TRAINING TARGET (``zscore`` | ``rank`` |
    ``raw``). Recorded only when it is not ``zscore``, so a missing key means
    ``zscore``, matching the default everything else reads.

    ``train_months`` restricts to a set of ``YYYY-MM`` training months. Pass it
    whenever a sweep shares run names across panels: the 32-month runs of
    k2@0.2, k2ind@0.1 and rrc@0.05 carry the SAME run name as their holdout-2
    counterparts, so without a month filter a completed reported-month run
    silently joins the optimization-set table and moves exactly the cells
    lambda was chosen from.
    """
    want_months = None if train_months is None else set(train_months)
    out: list[ICRun] = []
    for proj in sorted(Path(ckpt_root).glob(project_glob)):
        m = _MONTH_SUFFIX_RE.search(proj.name)
        # A BUNDLED project spans SEVERAL months and its suffix is a range,
        # so the project cannot name the month and each RUN must. The
        # month-per-project assumption holds for one-job-per-month sweeps
        # (run_all_months_sweep.sh) and silently drops everything submitted
        # with --months-per-job N > 1: the 203-month multihead campaign lands
        # in projects ending "-2008-01-2008-04", which this regex does not
        # match at all, so every one of its 51 projects was skipped and the
        # series read as zero runs rather than as an error.
        proj_month = f"{m.group('year')}-{m.group('mon')}" if m else None
        # Filter here only when the project names ONE month; otherwise the
        # filter has to wait until each run has named its own.
        if proj_month is not None and want_months is not None \
                and proj_month not in want_months:
            continue
        for run in sorted(p for p in proj.iterdir() if p.is_dir()):
            meta_p, ic_p = run / "train_meta.json", run / "xs_ic.json"
            if not (meta_p.is_file() and ic_p.is_file()):
                continue
            try:
                meta = json.loads(meta_p.read_text())
                ic = json.loads(ic_p.read_text())
            except json.JSONDecodeError:
                continue
            if xs_stats is not None and (
                str(meta.get("xs_anchor_stats") or "xs_anchor_stats") != xs_stats
            ):
                continue
            # Written only when it differs from the value a reader assumes,
            # so absence is a value here, not a gap.
            if xs_target is not None and (
                str(meta.get("xs_target") or "zscore") != xs_target
            ):
                continue
            # THE MONTH IS THE ONE THE TRAINING WINDOW ENDS ON, and only the
            # run's own config knows it. This used to read the PROJECT's
            # month, which is the window's START: a six-month span lands in
            # "...-2008-02-01-2008-07-31", so every span run was filed five
            # months early and its eval month with it, and a panel keyed on
            # eval month then matched almost nothing -- 31 scored runs came
            # back as 5. Config first, then the run name, then the project.
            train_month = _train_end_month(meta)
            if train_month is None:
                rm = _RUN_MONTH_RE.match(str(meta.get("run_name") or ""))
                train_month = rm.group("ym") if rm else proj_month
            if train_month is None:
                continue
            if want_months is not None and train_month not in want_months:
                continue
            out.append(ICRun(
                project=proj.name, run_id=run.name, path=run,
                train_month=train_month, eval_month=_next_month(train_month),
                meta=meta, ic=ic,
            ))
    return out


# ── THE STANDARD LeJEPA MODEL ───────────────────────────────────────────────
#
# Read this before writing any glob that mentions k2ind.
#
# The standard model is LeJEPA on cross_stock K=2 within the focal stock's
# FF49 industry at lamb=0.01, and its 32-month series is the c7ff4e
# generation. There are three OTHER things on disk whose names contain
# "k2ind" or "cross-stock-ind", and every one of them has been mistaken for
# it at least once:
#
#   k2ind-lamb-0ebfed-*       the LAMBDA SWEEP: 43 runs over 24 months at
#                             lamb in {0.005, 0.01, 0.025, 0.05}. Filtering
#                             it to lamb=0.01 leaves THIRTEEN months, which
#                             looks like a plausible series and is not one.
#                             ARCHIVED 2026-08-21 to lab/models-archive
#                             so a glob against the checkpoint root cannot
#                             reach it; plots/lambda_sweep reads it there.
#   cross-stock-ind-c8dd09-*  an earlier generation, 14 months, NO xs_ic.json.
#                             iter_ic_runs skips it, but it matches an
#                             unpinned "cross-stock-ind-*" — which is why the
#                             glob below carries the hash.
#   k2ind-scaling-9eec75-*    a model-size sweep on the 5-month optimization
#                             set.
#
# Use standard_k2ind_runs(). It pins the hash, pins lamb, and refuses to
# return a short series, because the failure mode here is not an exception —
# it is a figure that renders beautifully over the wrong months.
K2IND_GLOB = "cross-stock-ind-c7ff4e-*"
K2IND_LAMB = 0.01
K2IND_N_MONTHS = 32

# Where retired-but-cited checkpoint generations go. See [[models_archive]].
MODELS_ARCHIVE = Path("lab/models-archive")


def standard_k2ind_runs(
    ckpt_root: Path | str = CKPT_ROOT, require_full: bool = True,
) -> list[ICRun]:
    """Every scored run of the standard k2ind LeJEPA series.

    Args:
        ckpt_root: checkpoint tree to search.
        require_full: raise unless all :data:`K2IND_N_MONTHS` months are
            present. Pass ``False`` only for a deliberately partial check —
            never for anything that gets plotted or reported.

    Raises:
        SystemExit: if the series is short. A missing month is nearly always
            a sync that has not finished or a glob that found the wrong
            project, and both produce a number averaged over the wrong panel
            rather than an error.
    """
    runs = [r for r in iter_ic_runs(K2IND_GLOB, ckpt_root)
            if r.meta.get("lamb") == K2IND_LAMB]
    months = {r.eval_month for r in runs}
    if require_full and len(months) < K2IND_N_MONTHS:
        raise SystemExit(
            f"standard k2ind series has {len(months)} of {K2IND_N_MONTHS} "
            f"eval months under {Path(ckpt_root) / K2IND_GLOB}.\n"
            f"Found: {' '.join(sorted(months))}\n"
            "Do NOT fall back to k2ind-lamb-* — that is the lambda sweep, "
            "and at lamb=0.01 it covers only 13 months."
        )
    return runs


def standard_k2ind_months(kind: str = "eval", **kw) -> list[str]:
    """Sorted ``eval`` (default) or ``train`` months of the standard series."""
    if kind not in ("eval", "train"):
        raise ValueError(f"kind must be 'eval' or 'train', got {kind!r}")
    return sorted({getattr(r, f"{kind}_month") for r in standard_k2ind_runs(**kw)})


def load_randinit_ic(
    task: str,
    results_dir: Path | str = SCORE_RESULTS,
    metric: str = "ic",
    bins: int | None = None,
) -> dict[str, float]:
    """{eval_month: random-init encoder IC for ``task``}, averaged over seeds.

    ``metric="auc"`` reads the logistic-probe AUC column instead, which is the
    ΔAUC subtrahend. Both come from the same scoring pass, so a month has one
    without the other only while a rescore is still in flight.

    ``bins`` selects WHICH AUC. A macro one-vs-rest AUC is a property of the
    partition as much as of the model — a k=21 number is mechanically nearer
    0.5 than a k=5 one — so a baseline at the wrong k is not a baseline. The
    reported partition (k=5) is the default; pass 11 or 21 to subtract from a
    head scored at its own bin count.

    The ΔIC subtrahend. Keyed by ARCHITECTURE rather than by run, so one
    baseline serves every checkpoint sharing an encoder shape; the files hold
    seeds 42/43/44 and this returns their mean.

    ONLY ``randinit-fwd3-*`` IS READ. A floor is the IC of a random encoder
    against a PARTICULAR target, and all three targets became forward-window
    differences on 2026-08-22; a randinit run against the retired midpoint
    tables is the floor of a different quantity. The files carry no stamp --
    only the tag says which generation a run belongs to -- so this used to glob
    ``randinit-*.json`` and average the two generations together, silently.
    That is not a small effect: on the set-2 holdout months the pooled floor
    read +0.0047 and came ENTIRELY from retired-target runs, because no
    forward-window floor existed for those months at all. A missing floor must
    surface as a missing month, not as a plausible number borrowed from the
    definition it replaced.
    """
    if metric == "ic":
        col = task
    elif bins in (None, 5):
        col = f"{task}_auc"
    else:
        col = f"{task}_auc_k{int(bins)}"
    by: dict[str, list[float]] = {}
    for f in sorted(Path(results_dir).glob(f"{RANDINIT_TAG}-*.json")):
        try:
            recs = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        for r in recs:
            v = _finite(r.get(col))
            if v is not None:
                by.setdefault(str(r["eval_month"]), []).append(v)
    return {m: sum(v) / len(v) for m, v in by.items()}
