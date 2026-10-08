"""Forward decay: how much rank IC a checkpoint loses as it ages.

THE FIGURE THIS FILE EXISTS FOR is ``forward_eval_v2_decay.png`` -- three
panels, one per target, each a scatter of (months between training and
evaluation, IC relative to retraining) with the fitted decay line. Nothing
else is drawn here any more; the horizon-grid curves, the age scatters and
the per-target diagnostics were deleted on 2026-09-14 (user) because none of
them controlled for the eval month, and a raw forward curve mostly traces
which months happened to be easy.

WHY THIS ONE CONTROLS FOR THE MONTH. A checkpoint trained on month T and
scored on month E is compared against THE SAME ARM'S checkpoint trained on
E-1 -- the model you would have if you retrained every month -- scored on
that identical panel. The month's difficulty, the arm's level and the
target's scale all cancel in the subtraction, so what is left is the price of
the checkpoint being old and nothing else. The other figures subtracted a
random-init floor instead, which moves with the month and so measured the
month.

THE ALL-PAIRS DESIGN. Every checkpoint is scored on every reported eval month
that lies 1..60 months ahead of its training month, so the horizon is
whatever the calendar gives -- 59 distinct values rather than 8 chosen ones --
and the reference is free, because the age-1 model for eval month E is
already in the panel (see ``baseline_common.build_pairs``).

THE ARMS ARE THE THREE SUPERVISED SPECIALISTS at the locked recipe (six-month
span, 12 passes, blr 2e-4), read through their OWN TRAINED HEAD. LeJEPA is
not on this figure: those models have not been retrained at the locked recipe
yet (user, 2026-09-14). When they are, add an Arm and give it
``variants=("probe",)`` -- it carries no head -- and the panel will draw two
lines per target with no other change.

HEAD-ONLY SCORING, which is what makes the campaign affordable. The gap=1
reference comes from each checkpoint's own ``xs_ic.json`` when the scored row
has not landed, and ``--head`` on the launcher keeps ``head.pt`` and skips
the ridge entirely.

WHERE THE WORK RUNS. This script does not embed anything: it writes the
manifest our cluster's scoring launcher (not included) consumes and collects
the result jsons those jobs push back. The measurement is
``xs_ic_eval.score`` either way; only the machine moves.

Usage:
    # 1. write the all-pairs manifest
    uv run plots/forward_eval_v2/forward_eval_v2.py pairs-manifest
    # 2. submit it (the command is printed, --head keeps the trained heads)
    # 3. collect what has landed, and draw
    uv run plots/forward_eval_v2/forward_eval_v2.py pairs-collect
    uv run plots/forward_eval_v2/forward_eval_v2.py pairs-plot

``pairs-collect`` is incremental and reports coverage, so the figure can be
drawn off a partial grid while the queue drains.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLOTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_PLOTS_DIR))

from style import (  # noqa: E402
    COMPACT_RC_PARAMS, RC_PARAMS, SERIES_STYLES, WIDTH_FULL,
    add_bottom_legend, apply_style, save_figure, set_two_decimal_yticks,
)

sys.path.insert(0, str(_THIS_DIR))
import baseline_common as bc  # noqa: E402

CKPT_ROOT = Path("lab/market-jepa-checkpoints")
SCORE_RESULTS = Path("lab/score_results")


# ─── the arms ───────────────────────────────────────────────────────────────

# Axis title per target. Wording matches plots/metrics/metrics.WM_PANEL_TITLES
# so a target reads the same way across paper figures.
TARGET_TITLE: dict[str, str] = {
    "return": "Return",
    "volatility_change": "Volatility Change",
    "spread_change": "Spread Change",
}



@dataclass(frozen=True)
class Arm:
    """One encoder family, and which figures it appears on.

    ``project`` carries the COMMIT HASH deliberately. An unpinned
    ``supervised-loss-ablation-*`` matches four sweeps that share a directory
    prefix and measure different things, and the failure mode is never an
    exception — it is a figure that renders beautifully over the wrong
    checkpoints.

    ``run_re`` then picks the arm out of its sweep.
    """

    key: str                    # results.json key + SERIES_STYLES lookup
    project: str                # checkpoint project prefix, hash included
    run_re: re.Pattern
    targets: tuple[str, ...]    # target COLUMNS this arm is plotted on
    variants: tuple[str, ...]   # "probe", and "head" where one was trained
    style: dict = field(default_factory=dict)




# THE SIX-MONTH-SPAN SPECIALISTS (2026-09-13). One encoder per target at the
# locked recipe, read through its own trained head -- the same wave
# plots/metrics/metrics.py draws, pinned the same way and for the same reason.
#
# THE COMMIT IS IN THE PROJECT PREFIX. Three waves answer to
# ``supervised-full-month-return-*``: 3e5087 (the single-month generation this
# replaced), 62bffe (this recipe on the wrong months, cancelled) and 581eb2.
# All three carry run names that differ only in the blr token, so a prefix
# match would put three generations on one figure with nothing in a run name
# to say so. To read a newer wave, change this hash and nothing else.
SPAN_COMMIT = "581eb2"

# THE TARGET'S OWN HUE, from the one table that owns it -- return purple, vol
# green, spread orange, the same colours the task wears in
# plots/variance_decomp and plots/full_data_multihead. An arm's ``key`` IS its
# SERIES_STYLES key, so this reads the shared table rather than restating it.
#
# WAS tab:gray labelled "Supervised Specialist (15 Min)", which was right
# while the gray had to mean "specialist" against other families sharing an
# axis. Here each panel draws exactly one arm -- the one trained on that
# panel's target -- so the colour is free to carry the target instead, and the
# legend it needed goes with it (see ``plot_pairs_panel``).
def _head_style(key: str) -> dict:
    """``key``'s paper colour and label, looked up rather than copied."""
    return {"color": SERIES_STYLES[key]["color"],
            "label": SERIES_STYLES[key]["label"]}

ARMS: tuple[Arm, ...] = (
    Arm(
        key="supervised_return",
        project=f"supervised-full-month-return-{SPAN_COMMIT}",
        run_re=re.compile(r"^\d{4}-\d{2}_pairwise_blr0\.0002$"),
        targets=("return_900",),
        variants=("head",),
        style={"head": _head_style("supervised_return")},
    ),
    Arm(
        key="supervised_vol",
        project=f"supervised-full-month-vol-change-{SPAN_COMMIT}",
        run_re=re.compile(r"^\d{4}-\d{2}_pairwise_blr0\.0002$"),
        targets=("volatility_change_900",),
        variants=("head",),
        style={"head": _head_style("supervised_vol")},
    ),
    Arm(
        key="supervised_spread",
        project=f"supervised-full-month-spread-change-{SPAN_COMMIT}",
        run_re=re.compile(r"^\d{4}-\d{2}_pairwise_blr0\.0002$"),
        targets=("spread_change_900",),
        variants=("head",),
        style={"head": _head_style("supervised_spread")},
    ),
)

ARM_BY_KEY: dict[str, Arm] = {a.key: a for a in ARMS}


def arms_for(target_col: str) -> list[Arm]:
    """The arms that appear on ``target_col``'s figure."""
    return [a for a in ARMS if target_col in a.targets]


# ─── checkpoint discovery ───────────────────────────────────────────────────


def find_ckpt(arm: Arm, train_month: str) -> Path | None:
    """The run directory for ``arm`` whose training window ENDS on that month.

    A sweep project directory is ``<project>-<start>-<end>``, and this used to
    glob on the START -- the same month, and so the same answer, only while a
    run trained one calendar month. Under the locked recipe a run trains
    ``DatasetConfig.train_span_months`` ending at the eval-adjacent month, so
    the project for 2008-07 is ``...-2008-02-01-2008-07-31`` and a glob on the
    start would have silently found nothing, or worse, found the wave that
    still names its months the old way.

    THE CONFIG IS AUTHORITATIVE and the suffix only narrows the search: the
    suffix is written by the launcher, the window is what training used, and
    only ``train_date_end`` records it. ``run_re`` then picks the arm out of
    its sweep -- run_name is the only place the trained task is recorded (W&B
    flattens ``backbone`` to a parameter count).
    """
    y, m = train_month.split("-")
    for d in sorted(CKPT_ROOT.glob(f"{arm.project}-*-{y}-{m}-*")):
        for run in sorted(p for p in d.iterdir() if p.is_dir()):
            meta = run / "train_meta.json"
            if not meta.is_file():
                continue
            try:
                rec = json.loads(meta.read_text())
            except (OSError, ValueError):
                continue
            cfg = rec.get("config", rec)
            end = str((cfg.get("dataset") or {}).get("train_date_end") or "")
            if end[:7] != train_month:
                continue
            if arm.run_re.match(str(rec.get("run_name", ""))):
                return run
    return None


def inventory(months: list[str] | None = None) -> dict[tuple[str, str], Path]:
    """``{(arm_key, train_month): run_dir}`` for every cell that exists."""
    out: dict[tuple[str, str], Path] = {}
    for arm in ARMS:
        for ptm in (months if months is not None else pair_train_months()):
            p = find_ckpt(arm, ptm)
            if p is not None:
                out[(arm.key, ptm)] = p
    return out


def pair_train_months() -> list[str]:
    """The 32 train months of the all-pairs design (eval months minus one)."""
    return sorted({bc.add_months(e, -1) for e in bc.eval_months()})


PAIRS_MANIFEST = _THIS_DIR / "manifest_pairs.tsv"
PAIRS_JSON = _THIS_DIR / "results_pairs.json"
# THE TAG NAMES THE WAVE. Results land in a shared directory as
# ``<tag>-partNN.json``, and ``pairs-collect`` globs the tag -- so a rerun
# against a different generation of checkpoints under the same tag would
# collect both. The commit that trained the arms is in the tag for the same
# reason it is in the project prefix.
PAIRS_TAG = f"fwdv2pairs-{SPAN_COMMIT}"


def cmd_pairs_manifest(args) -> None:
    """The 32 x 32 all-pairs manifest — the design decay is measured on.

    Rows are ordered by train month so run_score_ckpts.sh's chunker puts one
    probe-fit month and a run of its eval months in each job, the same reason
    the grid manifest is.
    """
    months = pair_train_months()
    have = inventory(months)
    missing = [(a.key, m) for a in ARMS for m in months if (a.key, m) not in have]
    if missing:
        print(f"WARNING: {len(missing)} (arm, month) cells have no checkpoint:")
        for k in missing[:12]:
            print(f"    {k[0]:20s} {k[1]}")

    pairs = bc.build_pairs(args.max_gap)
    lines = []
    for tm, ev, _gap in pairs:
        for arm in ARMS:
            run = have.get((arm.key, tm))
            if run is not None:
                lines.append(f"{run}\t{tm}\t{ev}\n")
    PAIRS_MANIFEST.write_text("".join(lines))
    gaps = sorted({g for _, _, g in pairs})
    print(f"wrote {PAIRS_MANIFEST}: {len(lines)} rows "
          f"({len(pairs)} pairs x {len(ARMS)} arms)")
    print(f"{len(months)} train months x 32 eval months, gap 1..{args.max_gap}; "
          f"{len(gaps)} distinct horizons realised")
    print(f"{len({m for p in pairs for m in (p[0], p[1])})} distinct months "
          f"(all already panel-cached)")
    print()
    print("submit with:")
    print("    XS_STATS_DIR=xs_anchor_stats_fwdvwap60 MONTHS_PER_JOB=17 "
          "SCORE_AUC=0 \\")
    print("      PARTITION=<h100-partition> N_WORKERS=14 \\")
    print('      SBATCH_EXTRA="--gres=gpu:2 --cpus-per-task=14 --mem=128G" \\')
    print(f"        <your run_score_ckpts launcher> \\")
    print(f"          --manifest {PAIRS_MANIFEST.relative_to(_REPO_ROOT)} "
          f"--tag {args.tag} --head")


def cmd_pairs_collect(args) -> None:
    """Gather the all-pairs result jsons into ``results_pairs.json``."""
    have = inventory(pair_train_months())
    run_to_cell = {p.name: k for k, p in have.items()}
    valid = {(tm, ev) for tm, ev, _ in bc.build_pairs(args.max_gap)}

    cells: dict[str, dict] = {}
    if PAIRS_JSON.exists() and not args.rebuild:
        cells = json.loads(PAIRS_JSON.read_text()).get("cells", {})
    n_files = n_rows = 0
    for f in sorted(SCORE_RESULTS.glob(f"{args.tag}-part*.json")):
        n_files += 1
        for rec in json.loads(f.read_text()):
            k = run_to_cell.get(rec.get("ckpt", ""))
            if k is None:
                continue
            arm_key, tm = k
            ev = rec.get("eval_month", "")
            if rec.get("fit_month") != tm or (tm, ev) not in valid:
                continue
            ics = _record_ics(rec, ARM_BY_KEY[arm_key].targets)
            if not ics:
                continue
            cells[f"{arm_key}/{tm}/{ev}"] = {"eval_month": ev, **ics}
            n_rows += 1
    PAIRS_JSON.write_text(json.dumps(
        {"__about__": PAIRS_ABOUT, "cells": dict(sorted(cells.items()))}, indent=2))
    print(f"{n_rows} cells from {n_files} result file(s); "
          f"wrote {PAIRS_JSON} ({len(cells)} total)")
    want = len(valid) * len(ARMS)
    print(f"coverage: {len(cells)}/{want} "
          f"({100 * len(cells) / want:.0f}%)")


PAIRS_ABOUT = (
    "Forward-decay all-pairs rank ICs. One record per (arm, train month, eval "
    "month) over the 32 x 32 reported panel, restricted to eval months 1..60 "
    "months AHEAD of the training month. A ridge probe is fit once on the "
    "arm's train month at 36 anchors/day and scored on the eval month at the "
    "reported 8, through xs_ic_eval.score. The gap=1 records are the age-1 "
    "reference every other record is read against, and duplicate what each "
    "checkpoint's own xs_ic.json holds -- kept so one run produces both sides "
    "of the subtraction. Produced by plots/forward_eval_v2/forward_eval_v2.py "
    "pairs-collect."
)


# ─── collect ────────────────────────────────────────────────────────────────

ABOUT = (
    "Forward-decay v2 rank ICs. One record per (arm, probe-train month, H): "
    "a ridge probe fit ONCE on the arm's train_end month at 36 anchors/day "
    "and scored on train_end + H months at the reported 8 anchors/day, "
    "through xs_ic_eval.score. `head` is present only for the supervised "
    "specialists, which carry a trained head; it is that head's own readout "
    "on the same eval panel. Targets are the two-forward-window generation "
    "(xs_anchor_stats_fwdvwap60) -- numbers from before 2026-08-22 measure "
    "something else. Produced by plots/forward_eval_v2/forward_eval_v2.py "
    "collect, from lab/score_results/<tag>-part*.json (cluster) and "
    "each checkpoint's own xs_ic.json (the H=1 column)."
)


def _record_ics(rec: dict, targets: tuple[str, ...]) -> dict:
    """``{"probe": {col: {...}}, "head": {col: {...}}}`` from one result record.

    The WITHIN-month SE and cell count are kept alongside the IC even though
    no curve uses them: the reported SE is over months, not cells, and the
    only way to show what that clustering cost is to have both
    (``pooled_month_ic``'s inflation factor). Re-collecting to add them later
    would mean re-reading result files that may have been cleaned up.
    """
    out: dict[str, dict[str, dict]] = {"probe": {}, "head": {}}
    for col in targets:
        for variant, key in (("probe", col), ("head", f"head:{col}")):
            v = rec.get(key)
            if v is None or not np.isfinite(v):
                continue
            cell = {"ic": float(v)}
            se, n = rec.get(f"{key}_se"), rec.get(f"{key}_cells")
            if se is not None and np.isfinite(se):
                cell["se"] = float(se)
            if n is not None:
                cell["n_cells"] = int(n)
            out[variant][col] = cell
    return {k: v for k, v in out.items() if v}



# ─── the all-pairs decay estimator ──────────────────────────────────────────


_PAIR_INV: dict | None = None


def _pair_inventory() -> dict:
    """``inventory`` over the all-pairs train months, built once per process."""
    global _PAIR_INV
    if _PAIR_INV is None:
        _PAIR_INV = inventory(pair_train_months())
    return _PAIR_INV


def pairs_points(pcells: dict, arm_key: str, variant: str, target_col: str
                 ) -> list[tuple[int, float, str, str]]:
    """``(gap, y, train_month, eval_month)`` over the 32 x 32 panel.

    ``y`` is this checkpoint's IC on an eval month MINUS the IC of the same
    arm's checkpoint trained on that eval month's own preceding month — an
    age-1 model of the identical family, scored on the identical panel. The
    month's difficulty, the arm's level, and the target's scale all cancel in
    the subtraction; what is left is the price of the checkpoint being old.

    GAP=1 IS DROPPED. There the checkpoint IS its own reference, so y is
    exactly 0 with no sampling error, and 32 noiseless zeros would pin the
    intercept and shrink every standard error. The fit's intercept is instead
    left free and reported: it should land near zero on its own, and that is
    a check rather than an assumption.
    """
    def gap(tm: str, ev: str) -> int:
        return bc._mi(ev) - bc._mi(tm)

    # THE REFERENCE, PREFERRING THIS RUN'S OWN gap=1 ROW. Falling back to the
    # checkpoint's xs_ic.json when that row has not landed is not a
    # compromise: the two are the same measurement through the same scorer,
    # and the 36 cells where both exist agree to a signed mean of +5.5e-06
    # (sd 9.0e-05, t=0.37 -- noise, not drift), which is three orders of
    # magnitude under the slope SE. Without the fallback a partial campaign
    # can produce no estimate at all, since one missing reference row
    # disqualifies every pair sharing its eval month.
    ref: dict[str, float] = {}
    ref_src = {"run": 0, "xs_ic": 0}
    have = _pair_inventory()
    for ev in bc.eval_months():
        tm0 = bc.add_months(ev, -1)
        c = pcells.get(f"{arm_key}/{tm0}/{ev}")
        e = (c or {}).get(variant, {}).get(target_col)
        if e is not None:
            ref[ev] = e["ic"]
            ref_src["run"] += 1
            continue
        run = have.get((arm_key, tm0))
        if run is None or not (run / "xs_ic.json").is_file():
            continue
        try:
            ic = json.loads((run / "xs_ic.json").read_text())
        except (OSError, ValueError):
            continue
        key = f"xs_ic/{'head:' if variant == 'head' else ''}{target_col}"
        v = ic.get(key)
        if v is not None and np.isfinite(v):
            ref[ev] = float(v)
            ref_src["xs_ic"] += 1
    pairs_points.last_ref_src = dict(ref_src)

    out = []
    for key, c in pcells.items():
        a, tm, ev = key.split("/")
        if a != arm_key or ev not in ref:
            continue
        e = c.get(variant, {}).get(target_col)
        g = gap(tm, ev)
        if e is None or g <= 1:
            continue
        out.append((g, e["ic"] - ref[ev], tm, ev))
    return out


def _meat_by_code(S: np.ndarray, code: np.ndarray, n_groups: int) -> np.ndarray:
    """``sum_g (sum_{i in g} S_i)(...)^T`` — segment sums, not a Python loop."""
    acc = np.zeros((n_groups, S.shape[1]))
    np.add.at(acc, code, S)
    return acc.T @ acc


def _cgm_se_fast(X, XtXi, u, ct, nt, ce, ne) -> float:
    """Cameron-Gelbach-Miller two-way cluster-robust SE on the slope.

    ``V_train + V_eval - V_intersection``. Both dependencies are real here and
    each is strong: residual intra-class correlation runs 0.29-0.73 by
    checkpoint and 0.51-0.74 by eval month, so one-way clustering removes only
    one of them.

    THE INTERSECTION TERM IS PLAIN HC0. A point is one (checkpoint, eval
    month) pair and no two points share both, so the intersection clusters are
    all singletons and their meat is ``X' diag(u^2) X``.
    """
    S = X * u[:, None]
    M = (_meat_by_code(S, ct, nt) + _meat_by_code(S, ce, ne) - S.T @ S)
    V = XtXi @ M @ XtXi
    return float(np.sqrt(max(V[1, 1], 0.0)))


def wild_two_way(x: np.ndarray, y: np.ndarray, gt: np.ndarray, ge: np.ndarray,
                 n_boot: int = 1999, seed: int = 5) -> dict:
    """``y = a + b x`` with a CGM two-way SE and a wild-bootstrap-t p-value.

    See ``pairs_slope`` for why the t must not be read against 1.96.
    """
    ct, ot = np.unique(gt, return_inverse=True)[::-1]
    ce, oe = np.unique(ge, return_inverse=True)[::-1]
    nt, ne = len(ot), len(oe)
    X = np.column_stack([np.ones_like(x), x])
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ (X.T @ y)
    u = y - X @ beta
    se = _cgm_se_fast(X, XtXi, u, ct, nt, ce, ne)
    t_obs = float(beta[1] / se) if se else float("nan")

    rng = np.random.default_rng(seed)
    ar = float(y.mean())                       # restricted fit: slope == 0
    ur = y - ar
    wt = rng.choice([-1.0, 1.0], size=(n_boot, nt))
    we = rng.choice([-1.0, 1.0], size=(n_boot, ne))
    Ys = ar + (wt[:, ct] * we[:, ce]) * ur
    Bs = Ys @ (X @ XtXi)
    Us = Ys - Bs @ X.T
    ts = [Bs[k, 1] / se_k for k in range(n_boot)
          if (se_k := _cgm_se_fast(X, XtXi, Us[k], ct, nt, ce, ne)) > 0]
    ts = np.abs(np.asarray(ts))
    ss, tss = (u ** 2).sum(), ((y - y.mean()) ** 2).sum()
    return {
        "a": float(beta[0]), "b": float(beta[1]), "se": se, "t": t_obs,
        "n": len(y),
        "p": float((ts >= abs(t_obs)).mean()) if len(ts) else float("nan"),
        "crit95": float(np.percentile(ts, 95)) if len(ts) else float("nan"),
        "r2": float(1 - ss / tss) if tss else float("nan"),
        "per_60": float(beta[1] * 60.0), "se_60": se * 60.0,
        "clusters": nt, "clusters_eval": ne,
    }


_SLOPE_CACHE: dict = {}


def pairs_slope(pcells: dict, arm_key: str, variant: str, target_col: str,
                n_boot: int = 1999, seed: int = 5) -> dict | None:
    """Decay slope over the all-pairs panel, x60, with a two-way wild-bootstrap p.

    THE t IS NOT NORMAL HERE AND MUST NOT BE READ AGAINST 1.96. There are 32
    checkpoints and 32 eval months, and at cluster counts that small the
    bootstrap-t reference distribution has critical values around 2.2-2.5 — so
    a t of 2.0-2.2 is NOT significant. Reading them against 1.96 is what made
    an earlier version of this report spread/LeJEPA as a result. ``p`` comes
    from a two-way WILD cluster bootstrap with the null imposed: Rademacher
    weights per checkpoint times per eval month, applied to the restricted
    residuals, refit, and the resulting |t| compared to the observed one.

    Wild rather than a resampling bootstrap on purpose. Drawing checkpoints
    and eval months with replacement and keeping the intersection retains only
    ~40% of the points per draw, which inflates the SE for a mechanical reason
    unrelated to the dependence being modelled.
    """
    ck = (id(pcells), arm_key, variant, target_col, n_boot, seed)
    if ck in _SLOPE_CACHE:
        return _SLOPE_CACHE[ck]
    pts = pairs_points(pcells, arm_key, variant, target_col)
    if len(pts) < 20:
        _SLOPE_CACHE[ck] = None
        return None
    out = wild_two_way(
        np.array([p[0] for p in pts], float),
        np.array([p[1] for p in pts], float),
        np.array([p[2] for p in pts]), np.array([p[3] for p in pts]),
        n_boot=n_boot, seed=seed,
    )
    _SLOPE_CACHE[ck] = out
    return out



def plot_pairs_panel(pcells: dict, outdir: Path, *,
                     name: str = "forward_eval_v2_decay") -> None:
    """The paper figure: all three targets side by side, full text width.

    ONE ROW, THREE PANELS, SHARED LEGEND. The targets share an x axis (months
    between training and evaluation) and nothing else -- spread change's cloud
    is seven times taller than return's -- so each panel keeps its own y scale
    and only the leftmost is labelled.

    THE SCATTER STAYS AT FULL RANGE. R^2 is 0.003-0.044 here: the slopes are
    real and they explain almost none of the variance, because month-to-month
    dispersion dwarfs aging. Clipping the y limits to make the lines look
    steeper would misrepresent exactly the thing this figure exists to show,
    so the points are drawn over their whole range and the fitted lines are
    left as shallow as they are. The per-panel text block carries the slope
    and its t, which is where the result actually lives.
    """
    # COMPACT fonts, not RC_PARAMS: three panels across the text width give
    # each axis ~2 inches, where the full-size 12pt labels crowd out the data.
    #
    # EVERY ARM THE PANEL'S TARGET HAS, in the order ARMS declares them --
    # not a hand-written list of two. That list named ``lejepa_tw`` by key and
    # would have raised the day the arm was dropped rather than drawn one
    # line; the arms carry their own style now, so adding LeJEPA back is an
    # Arm entry and nothing here.
    #
    # THE LINE IS THE HEAD, not a ridge probe -- user's call 2026-08-31, and
    # it is now the only readout scored (the campaign runs ``--head``). Worth
    # knowing when this sits beside the talk figure, whose gray line IS the
    # probe. The two no longer look alike -- these lines wear the target's
    # hue -- but they still answer to the same words in a caption.
    with plt.rc_context({**RC_PARAMS, **COMPACT_RC_PARAMS}):
        fig, axes = plt.subplots(
            1, len(bc.TARGET_COLS), figsize=(WIDTH_FULL, 2.25), sharex=True)
        for ax, target_col in zip(axes, bc.TARGET_COLS):
            target_type = target_col.rsplit("_", 1)[0]
            xs_line = np.linspace(1, 60, 100)
            panel_series = [
                (arm, variant, arm.style[variant])
                for arm in arms_for(target_col)
                for variant in arm.variants if variant in arm.style
            ]
            for arm, variant, style in panel_series:
                pts = pairs_points(pcells, arm.key, variant, target_col)
                r = pairs_slope(pcells, arm.key, variant, target_col)
                if not pts or r is None:
                    continue
                # SCATTERED AT FULL RANGE, so the line sits against the points
                # it was actually fit on. A fitted line shown against a
                # tighter cloud than produced it reads as better determined
                # than it is.
                ax.scatter([q[0] for q in pts], [q[1] for q in pts],
                           s=5, alpha=0.16, linewidths=0,
                           color=style["color"], zorder=1)
                ax.plot(xs_line, r["a"] + r["b"] * xs_line,
                        color=style["color"], linestyle="-",
                        linewidth=1.8, zorder=3, label=style["label"])
            ax.axhline(0, color="black", linewidth=0.6, alpha=0.45, zorder=2)
            ax.set_title(TARGET_TITLE[target_type], pad=4)
            ax.set_xlim(0, 61)
            ax.set_xticks([0, 12, 24, 36, 48, 60])
            set_two_decimal_yticks(ax, nbins=5)
            ax.grid(False)
            ax.margins(y=0.08)
            # NO IN-PANEL SLOPE/p BLOCK. It invited the two questions the
            # figure cannot answer in its own margin -- what the number is a
            # slope OF, and how the p was computed (a two-way wild cluster
            # bootstrap, because the t here is not normal). Both belong in the
            # caption or the text; `pairs-plot` still prints them.
        # NOT "delta IC": [[delta_ic_definition]] reserves that for the
        # random-init subtraction, and the subtrahend here is a TRAINED
        # model -- the one you would have if you retrained every month.
        axes[0].set_ylabel("Rank IC Relative\nto Retraining")
        fig.tight_layout(w_pad=1.0)
        # NO LEGEND WHILE EVERY PANEL HAS ONE LINE (user, 2026-09-15). The
        # colour now says which target, and the panel title says it again --
        # a legend naming the single arm spent a fifth of the figure height
        # repeating the one thing already unambiguous.
        #
        # IT COMES BACK BY ITSELF when a panel gains a second arm, which is
        # the documented way to add LeJEPA (see this module's docstring): the
        # legend is what distinguishes two lines sharing an axis, so it is
        # drawn exactly when there are two. Keep this conditional rather than
        # deleting the call -- adding the arm must stay a one-line change.
        multi = max(len(a.get_legend_handles_labels()[1]) for a in axes) > 1
        fig.supxlabel("Months Between Training and Evaluation",
                      y=0.135 if multi else 0.035,
                      fontsize=COMPACT_RC_PARAMS["axes.labelsize"])
        if multi:
            add_bottom_legend(fig, ncol=3, bottom_reserve=0.285)
        else:
            fig.subplots_adjust(bottom=0.20)
        written = save_figure(fig, outdir / name)
        print(f"Saved {', '.join(str(p) for p in written)}")
        plt.close(fig)



def cmd_pairs_plot(args) -> None:
    if not PAIRS_JSON.exists():
        raise SystemExit(f"{PAIRS_JSON} not found — run `pairs-collect` first")
    pcells = json.loads(PAIRS_JSON.read_text())["cells"]
    apply_style(COMPACT_RC_PARAMS)
    outdir = Path(args.outdir) if args.outdir else _THIS_DIR
    print("DECAY over the 32x32 all-pairs panel: IC lost per 5 years of "
          "checkpoint age, against the same arm's model trained the month "
          "before the eval month. SE = max(cluster by checkpoint, cluster by "
          "eval month).")
    for target_col in bc.TARGET_COLS:
        print(f"\n{target_col}")
        for arm in arms_for(target_col):
            for variant in arm.variants:
                r = pairs_slope(pcells, arm.key, variant, target_col)
                if r is None:
                    continue
                print(f"  {arm.key}/{variant:6s} {r['per_60']:+.4f} ± "
                      f"{r['se_60']:.4f}  t={r['t']:+.1f}  p={r['p']:.3f}"
                      f"  (|t| crit95={r['crit95']:.2f})   "
                      f"n={r['n']}  R2={r['r2']:.3f}  intercept={r['a']:+.4f}")
        print(f"    reference rows: {pairs_points.last_ref_src}")
    plot_pairs_panel(pcells, outdir)



# ─── main ───────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("pairs-manifest",
                        help="all-pairs manifest over the 32x32 panel")
    pm.add_argument("--tag", default=PAIRS_TAG)
    pm.add_argument("--max-gap", type=int, default=60)
    pm.set_defaults(func=cmd_pairs_manifest)

    pc = sub.add_parser("pairs-collect", help="gather the all-pairs results")
    pc.add_argument("--tag", default=PAIRS_TAG)
    pc.add_argument("--max-gap", type=int, default=60)
    pc.add_argument("--rebuild", action="store_true")
    pc.set_defaults(func=cmd_pairs_collect)

    pp = sub.add_parser("pairs-plot", help="the decay figure + slope table")
    pp.add_argument("--outdir", default=None)
    pp.set_defaults(func=cmd_pairs_plot)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
