"""Do the nine evaluations rank the models the same way? A rank-rank matrix.

Eighteen METHODS and the Random ViT floor are scored by nine different
evaluations in this repo, and the tables that report them live apart: three
rank-IC probes (plots/core/probe_fit_table), four organization tasks and two
factor-structure analyses (plots/core/fixed_panel_table). Each is read on its
own, so nothing says whether they are nine views of one quality or nine
different questions. This ranks every entry 1..19 on each task and correlates
the RANKINGS.

A METHOD IS NOT AN ENCODER. Every arm here is retrained per eval month, so
one row of the rank table summarizes one separately trained encoder per month
of the shared panel, each scored on the month after its training span, then
averaged. n = 19 for every correlation because 19 is the number of things
being ranked; the months are what makes each of those 19 numbers stable
rather than one draw.

THE NINE TASKS (rows/columns of the matrix, in group order):

    Return                  rank IC, forward VWAP return,   h = 900 s (15 min)
    Volatility Change       rank IC, volatility change,     h = 900 s
    Spread Change           rank IC, spread change,         h = 900 s
    Partner Matched View    T1: pooled NN is the partner's same-day view
    Own-Day Centroid        T2: nearest centroid is the focal day's
    Own-Firm Centroid       T3: nearest centroid is the focal firm's
    Partner Firm Centroid   T4: centroid NN is the partner's centroid
    Decoded Loadings        ridge decode of Pelger factor loadings 1-4
    Subspace Alignment      canonical overlap of embedding PCs and loadings

ONE FIGURE, ONE CONFIGURATION. There are no variant flags: every choice
below is made once, here, so there is exactly one matrix to quote and no way
to quote the wrong one.

  * THE ROSTER IS THE 18 TRAINED METHODS PLUS THE RANDOM ViT -- 19 ranked
    entries (METHODS). The floor is ranked like any method, so a task where
    it places well is visibly a task that measures little. It is kept OUT of
    ROSTER itself, because both paper tables read ROSTER as their method list
    and render the floor in a block of their own.

  * THE SOURCES ARE THE PAPER TABLES' OWN LOADERS, so a number here is the
    number in the table (2026-09-26 rebuild, after the six-month retrain):
      forecasting   plots/core/probe_fit_table.load_results -- the ridge probe
                    (alpha 10) at the full six-month fit pool, predictive
                    readout; the Random ViT is the randinit seeds averaged
                    within a month.
      organization  fixed_panel_P3S2{tag}.json over fixed_panel_table's
                    DEFAULT_TAGS; the floor is its seeded ``random`` model.
      factors       decode_loadings / subspace_alignment{tag}.json over the
                    same tags; the floor is the five randvit seeds averaged
                    within a month.
    ROSTER is keyed by the one-month generation (``<x>_final``); every SSL and
    LeJEPA row is read under its six-month key (``<x>_6mo``), exactly as the
    two tables map it.

  * THE SUPERVISED ARMS ARE READ THROUGH THEIR OWN HEAD on the target(s) they
    trained for (plots/metrics/probe_head_ic.json), probe everywhere else.
    See ic_scores().

  * T1-T4 ARE RANKED ON BOTH READOUTS, averaged. The rank by TOP-1 RATE
    (higher better) with the rank by MEAN RANK of the target among the
    candidates (lower better): the rate saturates once a model is good and
    says nothing about how it fails, the mean rank does, so neither alone is
    the task. The average is re-ranked, so every column is a permutation of
    1..N before anything is correlated.

  * THE FACTOR COLUMNS ARE ONE NUMBER EACH -- the mean OOS Pearson r over
    total loadings 1-4, and rho-bar. The other number each analysis reports
    is deliberately unused. Permutation z only calibrates rho ("read rho; z
    only guards it" -- summarize.alignment), so averaging it in adds nothing.
    Factor-model R^2 is a different question (a stock's communality, not its
    position in factor space) and belongs in prose, not in this column.

ONE MONTH PANEL FOR EVERYTHING. A rank is only meaningful if every entry
earned it on the same months, so every score -- all nine tasks, all 19
entries -- is averaged over the eval months EVERY source has for EVERY entry
(``shared_panel``). As of 2026-09-26 that is all 31 months (the multihead's
last six head, organization and factor months were filled in that day). The
run prints the panel and, if it shrinks, which entries limit it.

ABSOLUTE vs DELTA IC. The IC columns are absolute rank IC. On a shared month
panel this changes no ordering -- mean dIC = mean IC - mean floor, one
constant per column -- so the figure is the same either way.

Outputs, in this folder:

    task_rank_corr.{png,pdf}   the lower-triangle 9x9 Spearman matrix
    task_rank_corr.json        the 19x9 rank table + the correlations

Run:
    uv run plots/task_corr/task_rank_corr.py
    uv run plots/task_corr/task_rank_corr.py --from-json   # replot only
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
_PLOTS = HERE.parent
sys.path.insert(0, str(_PLOTS))
sys.path.insert(0, str(HERE))
# `paths` (FACTORS/PANELS) lives in plots/core; this module moved out of it.
sys.path.insert(0, str(_PLOTS / "core"))
# _plot_ic() does a function-local `import metrics` (plots/metrics/metrics.py,
# for SERIES_DEFS/load_ic_metrics). It is lazy, so dropping this path does NOT
# fail at import time -- only when that code path runs.
sys.path.insert(0, str(_PLOTS / "metrics"))

from style import (  # noqa: E402
    COMPACT_RC_PARAMS, SERIES_STYLES, WIDTH_FULL, apply_style, save_figure,
)

from paths import FACTORS as FS  # noqa: E402
H = 900          # the paper's headline horizon, 15 minutes
ALPHA = 10.0     # the ridge alpha probe_fit_table reports

# ONE CANVAS FOR BOTH FIGURES. The two are shown on consecutive slides, so
# they must occupy the same footprint or the deck jumps between them. That
# rules out ``bbox_inches="tight"``, whose output size depends on how long
# the labels happen to be (it had them at 6.92x5.34 and 4.99x4.18 in). Both
# now save on a FIXED canvas with explicit margins instead; the margins are
# what has to be right, since nothing outside them is rendered.
#
# THE SIZE IS plots/core/probe_fit_breadth.png's AS SAVED -- 2055 x 740 px at
# 300 dpi. That figure is WIDTH_FULL x 0.36 cropped tight, so these sit on the
# page at the same footprint and, with COMPACT_RC_PARAMS, the same type size.
FIG_SIZE = (2055 / 300, 740 / 300)

# THE ROSTER, and the two names each model answers to: ``lat`` in the
# latent-eval JSONs (fixed_panel and factors agree on it), ``ic`` in
# plots/metrics.SERIES_DEFS. The two trees grew separately, so the join is by
# hand; a model missing from either side raises rather than being ranked on
# half the tasks.
FAMILY_CITE = {"LeJEPA": "balestriero2025lejepaprovablescalableselfsupervised"}

# LATEX-ONLY SIZE OVERRIDES for row names that do not fit their column.
# Applied by BOTH table renderers, AFTER escaping, and keyed by the label as a
# row actually prints it (family prefix already trimmed).
#
# NOT part of `label`. ROSTER's labels also name points in the matplotlib
# figures below and rows in rankme_table.py's plain-text output, where a LaTeX
# macro would print literally rather than run.
LATEX_LABEL_SIZE = {
    "Same Stock, Diff. View": "scriptsize",
}


def latex_size_label(escaped, plain):
    """``escaped`` wrapped in its size macro, if ``plain`` has one."""
    size = LATEX_LABEL_SIZE.get(plain)
    return ("\\" + size + "{" + escaped + "}") if size else escaped


ROSTER: list[dict] = [
    {"label": "LeJEPA Same Stock, Diff. View", "fam": "LeJEPA",
     "lat": "pair_rrc_final",   "ic": "pair_rrc", "style": "lejepa_rrc"},
    {"label": "LeJEPA Time Warping",    "fam": "LeJEPA",
     "lat": "pair_warp_final",  "ic": "pair_warp", "style": "aug_warp"},
    {"label": "LeJEPA Gaussian Noising", "fam": "LeJEPA",
     "lat": "pair_noise_final", "ic": "pair_noise", "style": "aug_noise"},
    {"label": "LeJEPA Cross Stock",     "fam": "LeJEPA",
     "lat": "pair_k2_final",    "ic": "pair_k2", "style": "cross_stock_k2"},
    {"label": "LeJEPA C-S, Same Industry", "fam": "LeJEPA",
     "lat": "pair_k2ind_final", "ic": "pair_k2ind", "style": "cross_stock_k2ind"},
    # THE SUPERVISED ARMS ARE READ THROUGH THEIR OWN TRAINED HEAD on the
    # target(s) they were trained for; the probe still supplies their other
    # targets, and every unsupervised arm is probe everywhere because a head
    # is not a thing it has. A head is the model's actual output -- scoring a
    # specialist by linear probe measures a readout nobody would deploy, and
    # understates it on its own task (+0.0019 return, +0.0039 vol, +0.0482
    # spread).
    #
    # THE MULTIHEAD WAS BLOCKED UNTIL 2026-08-31. Its heads were written by a
    # loader that never loaded the head weights, so 31 of 32 months carried an
    # UNTRAINED head whose return IC averaged -0.0015 against a +0.0145 probe.
    # The full-history re-score landed 2026-08-31 03:16 and all 32 runs are now
    # stamped with style.HEAD_SCHEMA, which ICRun.head enforces -- an unstamped
    # multihead checkpoint returns None rather than that noise, so this cannot
    # silently regress. See [[supervised_head_collapse]].
    #
    # THE ASYMMETRY IS REAL AND BELONGS IN THE CAPTION: on the three
    # forecasting columns four methods are read through a trained head and
    # fifteen (the Random ViT included) through a linear probe. On the
    # 18-method, one-month-span campaign it changed little -- one cell moved
    # by >= 0.05, mean |rho| 0.349 -> 0.355 -- but it handed the multihead
    # first place on forecasting, ahead of MAE.
    {"label": "Supervised (return)",    "fam": "Supervised",
     "lat": "sup_return_w8",    "ic": "sup_return_w8",
     "style": "supervised_return", "head": ("return",)},
    {"label": "Supervised (vol)",       "fam": "Supervised",
     "lat": "sup_vol_w8",       "ic": "sup_vol_w8",
     "style": "supervised_vol", "head": ("volatility_change",)},
    {"label": "Supervised (spread)",    "fam": "Supervised",
     "lat": "sup_spread_w8",    "ic": "sup_spread_w8",
     "style": "supervised_spread", "head": ("spread_change",)},
    {"label": "Supervised (multihead)", "fam": "Supervised",
     "lat": "sup_multi_w8",     "ic": "sup_multi_w8",
     "style": "multihead",
     "head": ("return", "volatility_change", "spread_change")},
    {"label": "BYOL", "fam": "SSL", "lat": "byol_final", "cite": "grill2020bootstraplatentnewapproach",    "ic": "ssl_byol", "style": "byol"},
    {"label": "CoST",    "fam": "SSL", "lat": "cost_final", "cite": "woo2022costcontrastivelearningdisentangled",    "ic": "ssl_cost", "style": "cost"},
    {"label": "CPC",     "fam": "SSL", "lat": "cpc_final", "cite": "oord2019representationlearningcontrastivepredictive",     "ic": "ssl_cpc", "style": "cpc"},
    {"label": "DINO", "fam": "SSL", "lat": "dino_final", "cite": "caron_etal_dino",    "ic": "ssl_dino", "style": "dino"},
    {"label": "I-JEPA",  "fam": "SSL", "lat": "ijepa_final", "cite": "assran2023selfsupervisedlearningimagesjointembedding",   "ic": "ssl_ijepa", "style": "ijepa"},
    {"label": "MAE",     "fam": "SSL", "lat": "mae_final", "cite": "he_etal_2022",     "ic": "ssl_mae", "style": "mae"},
    {"label": "TF-C",    "fam": "SSL", "lat": "tfc_final", "cite": "zhang2022selfsupervisedcontrastivepretrainingtime",     "ic": "ssl_tfc", "style": "tfc"},
    {"label": "TimeMAE", "fam": "SSL", "lat": "timemae_final", "cite": "Cheng_2026", "ic": "ssl_timemae", "style": "timemae"},
    {"label": "TS2Vec",  "fam": "SSL", "lat": "ts2vec_final", "cite": "yue2022ts2vec",  "ic": "ssl_ts2vec", "style": "ts2vec"},
]

# THE RANDOM ViT, ranked as a 19th entry. NOT in ROSTER: ROSTER is the
# method list both paper tables render, and they give the floor its own block.
# ``lat``/``ic`` are unused for it -- each loader resolves the floor from its
# own seeds (see the module docstring).
FLOOR_ENTRY = {"label": "Random ViT", "fam": "Untrained", "lat": "random",
               "ic": "randinit", "style": "randinit", "point": "Random ViT"}
METHODS: list[dict] = ROSTER + [FLOOR_ENTRY]


def series_key(m: dict) -> str:
    """The key this entry's CURRENT results are stored under.

    ROSTER is keyed by the one-month generation; the reported wave is the
    six-month retrain, which the SSL and LeJEPA arms carry as ``<x>_6mo``.
    The supervised ``*_w8`` keys were never renamed.
    """
    k = m["lat"]
    return k[:-len("_final")] + "_6mo" if k.endswith("_final") else k


# DINO/BYOL ARE TIME-WARP VIEWS, and the table no longer says so in the row
# name (dropped 2026-09-17 for width). Both pin dataset_overrides to draw the
# positive pair as two time warps of ONE window -- the same view LeJEPA's
# 'Time Warping' row uses -- so a reader who takes these as image-style crops
# has them wrong. If the name does not carry it, the prose must.

# (task label, group). The group drives the block gaps, the tick colors and
# the left-hand brackets.
#
# THE LABELS ARE THE DIAGRAM PANEL TITLES, VERBATIM. T1-T4 are the four
# ``CLEAN_TITLES`` of fixed_panel/metric_diagram.py, and the two factor tasks
# are panels 3 and 4 of factors/factor_diagram.py (``--clean``); the three
# probes are the ``WM_PANEL_TITLES`` of plots/metrics/metrics.py. A talk shows
# the diagram and then this matrix, so a task must not change its name in
# between -- "T3" and "Own-Firm Centroid" being the same thing is not
# something an audience can be asked to hold in their head.
# The forecasting family's name, spelled ONCE: it is a group label on the
# figure and the filter plot_split splits on, and those two drifting apart
# would silently put every task on one side of the scatter.
PRED_GROUP = "Prediction"

# (label, group, source). ``source`` is what build_table dispatches on, so a
# task is declared ONCE: renaming a label cannot silently desynchronize a
# column from its data the way a second hard-coded name list can.
TASKS: list[tuple[str, str, str]] = [
    ("Return",                PRED_GROUP,      "ic:return"),
    ("Volatility Change",     PRED_GROUP,      "ic:volatility_change"),
    ("Spread Change",         PRED_GROUP,      "ic:spread_change"),
    ("Partner Matched View",  "Organization",      "org:T1"),
    ("Own-Day Centroid",      "Organization",      "org:T2"),
    ("Own-Firm Centroid",     "Organization",      "org:T3"),
    ("Partner Firm Centroid", "Organization",      "org:T4"),
    ("Decoded Loadings",      "Factor Structure",  "fac:decode"),
    ("Subspace Alignment",    "Factor Structure",  "fac:align"),
]
GROUP_COLOR = {PRED_GROUP: "tab:blue", "Organization": "tab:orange",
               "Factor Structure": "tab:green"}


# ── scores ────────────────────────────────────────────────────────────────
#
# Every loader returns PER-MONTH values, {task: {label: {month: value}}}, and
# nothing is averaged until build_table has the shared panel. A score that is
# missing for an entry raises rather than being dropped: a rank table with a
# hole in it silently re-ranks everything below it.

def _nanmean(xs) -> float:
    xs = [float(x) for x in xs if x is not None and float(x) == float(x)]
    return st.fmean(xs) if xs else float("nan")


def _core():
    """The two paper-table modules, imported lazily (they pull the registry)."""
    for p in (_PLOTS.parent, _PLOTS / "latent_eval" / "fixed_panel"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import fixed_panel_table as fpt
    import probe_fit_table as pft
    return pft, fpt


def ic_scores() -> dict[str, dict[str, dict[str, float]]]:
    """{target: {label: {month: rank IC at h=900}}}.

    READOUT. Every entry is read through the ridge probe on its frozen
    embedding, EXCEPT that a supervised arm is read through its own trained
    head on the target(s) it was trained for: that head is the model's actual
    output, and scoring it by probe measures a readout nobody would deploy.
    Off its own target a supervised arm is probe, like everyone else. Say so
    wherever the ranking is quoted -- on its own column a specialist is a
    trained head against linear probes.
    """
    pft, _ = _core()
    ics, _ns, nfiles, _unk = pft.load_results(ALPHA)
    head = pft.load_head()
    seeds = sorted(k for k in ics if k.startswith("randinit_"))
    if not seeds:
        raise SystemExit("no randinit_* floor in the probe results")
    print(f"  [ic] {nfiles} result files; floor seeds {', '.join(seeds)}")

    res: dict[str, dict[str, dict[str, float]]] = {}
    for task, _ in pft.TASKS:
        target = task[:-len(f"_{H}")]
        col = {}
        for m in METHODS:
            if m is FLOOR_ENTRY:
                months = {mo for s_ in seeds for mo in ics[s_].get(task, {})}
                col[m["label"]] = {
                    mo: _nanmean([ics[s_].get(task, {}).get(mo)
                                  for s_ in seeds]) for mo in months}
                continue
            key = series_key(m)
            if target in (m.get("head") or ()):
                per = head.get((key, task))
                if not per:
                    raise SystemExit(f"no head {task} for {key} in "
                                     f"{pft.HEAD_JSON.name}")
            else:
                per = ics.get(key, {}).get(task)
                if not per:
                    raise SystemExit(f"no probe {task} for {key}")
            col[m["label"]] = dict(per)
        res[target] = col
    return res


def organization_scores() -> dict[str, dict[str, dict[str, tuple]]]:
    """{'T1'..'T4': {label: {month: (top-1 rate, mean rank)}}}.

    Merged over fixed_panel_table.DEFAULT_TAGS the way that table merges
    them (later tags win a repeated key). Every tag is one panel geometry, so
    a month's rate and rank are comparable across entries.
    """
    _, fpt = _core()
    geo, tags = fpt.load_geo(list(fpt.DEFAULT_TAGS), required=False)
    print(f"  [org] tags {' '.join(tags)}")
    res: dict[str, dict[str, dict[str, tuple]]] = {}
    for mi in ("1", "2", "3", "4"):
        col = {}
        for m in METHODS:
            key = fpt.FLOOR if m is FLOOR_ENTRY else series_key(m)
            d = (geo.get(key) or {}).get(f"metric{mi}")
            if not d:
                raise SystemExit(f"no T{mi} result for {key}")
            rates, ranks = d["month_rates"], d["month_ranks"]
            col[m["label"]] = {
                mo: (float(rates[mo]), float(ranks[mo])) for mo in rates
                if mo in ranks and rates[mo] == rates[mo]
                and ranks[mo] == ranks[mo]}
        res[f"T{mi}"] = col
    return res


def _factor_months(tags, stem: str, field) -> dict[str, dict[str, float]]:
    """{series: {month: value}} from one factor stage, merged over ``tags``.

    ``field`` maps a result cell to its per-month list, which is aligned with
    ``months_used[series]`` (else the file's ``months``).
    """
    out: dict[str, dict[str, float]] = {}
    for tag in tags:
        f = FS / f"{stem}{tag}.json"
        if not f.is_file():
            continue
        j = json.loads(f.read_text())
        for key, cell in j["results"].items():
            series = key.split("|")[0]
            if stem == "decode_loadings" and not key.endswith("|emb"):
                continue
            vals = field(cell)
            months = j.get("months_used", {}).get(series, j["months"])
            if vals is None or len(vals) != len(months):
                continue
            out[series] = {mo: v for mo, v in zip(months, vals) if v == v}
    return out


def factor_scores() -> dict[str, dict[str, dict[str, float]]]:
    """{'decode': {label: {month: mean r over tot1-4}},
        'align':  {label: {month: rho-bar}}}."""
    _, fpt = _core()
    tags = list(fpt.DEFAULT_TAGS)

    def decode(cell):
        cols = [cell.get(f"tot{i}") for i in (1, 2, 3, 4)]
        if any(c is None for c in cols):
            return None
        return [_nanmean(v) for v in zip(*cols)]

    per = {"decode": _factor_months(tags, "decode_loadings", decode),
           "align": _factor_months(tags, "subspace_alignment",
                                   lambda c: c.get("rhobar"))}
    res: dict[str, dict[str, dict[str, float]]] = {}
    for kind, by in per.items():
        col = {}
        for m in METHODS:
            if m is FLOOR_ENTRY:
                seeds = [by[k] for k in fpt.FLOOR_SEEDS if k in by]
                if not seeds:
                    raise SystemExit(f"no {kind} floor seeds")
                months = set.intersection(*(set(s_) for s_ in seeds))
                col[m["label"]] = {mo: _nanmean([s_[mo] for s_ in seeds])
                                   for mo in months}
                continue
            key = series_key(m)
            if key not in by:
                raise SystemExit(f"no {kind} result for {key}")
            col[m["label"]] = by[key]
        res[kind] = col
    return res


def shared_panel(sources: dict[str, dict[str, dict]]) -> list[str]:
    """The eval months EVERY source carries for EVERY entry.

    Prints the panel and, for each entry that is short of the widest month
    set, what it is missing -- so a shrinking panel is visible, not silent.
    """
    per_entry: dict[str, set[str]] = {}
    for col in sources.values():
        for label, by_month in col.items():
            got = set(by_month)
            per_entry[label] = (per_entry[label] & got
                                if label in per_entry else got)
    panel = set.intersection(*per_entry.values())
    widest = set.union(*per_entry.values())
    print(f"  [panel] {len(panel)} shared months of {len(widest)} seen")
    for label, got in per_entry.items():
        if len(got) < len(widest):
            print(f"    {label}: {len(got)} months "
                  f"(missing {' '.join(sorted(widest - got))})")
    if not panel:
        raise SystemExit("no eval month is shared by every entry and task")
    return sorted(panel)


# ── ranking ───────────────────────────────────────────────────────────────

def _rank(values: list[float], higher_better: bool) -> np.ndarray:
    """1 = best, ties get the average rank."""
    v = np.asarray(values, dtype=float)
    return rankdata(-v if higher_better else v, method="average")


def build_table() -> dict:
    labels = [m["label"] for m in METHODS]

    print("loading scores")
    ic = ic_scores()
    org = organization_scores()
    fac = factor_scores()

    sources = {f"ic:{k}": v for k, v in ic.items()}
    sources.update({f"org:{k}": v for k, v in org.items()})
    sources.update({f"fac:{k}": v for k, v in fac.items()})
    panel = shared_panel(sources)

    def mean_on_panel(by_month, idx=None):
        vals = [by_month[mo] if idx is None else by_month[mo][idx]
                for mo in panel]
        return st.fmean(vals)

    # One column per TASKS row, built from that row's ``source``. Only the
    # four organization tasks have two components (see the docstring); the
    # rest are a single measure, higher-is-better.
    ranks: dict[str, np.ndarray] = {}
    for label, _group, source in TASKS:
        col = sources.get(source)
        if col is None:
            raise SystemExit(f"unknown task source {source!r}")
        if source.startswith("org:"):
            comps = [([mean_on_panel(col[l], 0) for l in labels], True),
                     ([mean_on_panel(col[l], 1) for l in labels], False)]
        else:
            comps = [([mean_on_panel(col[l]) for l in labels], True)]
        # Where there are two components the average of their ranks is the
        # score; re-ranking it is what makes the column a permutation of
        # 1..N, which keeps every column on one scale.
        avg = np.mean([_rank(v, hb) for v, hb in comps], axis=0)
        ranks[label] = rankdata(avg, method="average")

    names = [c for c, _, _ in TASKS]
    R = np.column_stack([ranks[c] for c in names])
    rho, pval = spearmanr(R)
    return {
        "labels": labels, "families": [m["fam"] for m in METHODS],
        "tasks": names, "groups": [g for _, g, _ in TASKS],
        "ranks": R.tolist(),
        "rho": np.asarray(rho).tolist(), "p": np.asarray(pval).tolist(),
        "n_methods": len(labels), "months": panel,
    }


# ── output ────────────────────────────────────────────────────────────────

def crit_rho(n: int, alpha: float = 0.05) -> float:
    """Two-sided ``alpha`` critical |rho| for ``n`` models, t approximation."""
    from scipy.stats import t as _t
    tc = _t.ppf(1 - alpha / 2, n - 2)
    return float(tc / np.sqrt(n - 2 + tc ** 2))


def print_table(d: dict) -> None:
    names, labels = d["tasks"], d["labels"]
    R = np.asarray(d["ranks"])
    w = max(len(l) for l in labels)
    head = " " * (w + 2) + "".join(f"{n[:9]:>11}" for n in names)
    print(f"\nRANKS (1 = best of {d['n_methods']}; the four organization "
          "columns are the mean of the rate rank and the mean-rank rank, "
          "re-ranked)\n")
    print(head)
    print("-" * len(head))
    for i, l in enumerate(labels):
        row = "".join(f"{R[i, j]:>11.1f}" for j in range(len(names)))
        print(f"{l:<{w}}  {row}")

    rho = np.asarray(d["rho"])
    print("\nSPEARMAN RANK CORRELATION between tasks\n")
    print(head)
    print("-" * len(head))
    for i, n in enumerate(names):
        row = "".join(f"{rho[i, j]:>11.2f}" for j in range(len(names)))
        print(f"{n:<{w}}  {row}")

    off = ~np.eye(len(names), dtype=bool)
    print(f"\nmean |rho| off-diagonal: {np.abs(rho[off]).mean():.3f}   "
          f"median: {np.median(rho[off]):+.3f}   "
          f"range: {rho[off].min():+.2f} .. {rho[off].max():+.2f}")
    n = d["n_methods"]
    crit = crit_rho(n)
    pairs = [(rho[i, j], names[i], names[j])
             for i in range(len(names)) for j in range(i + 1, len(names))]
    sig = [p for p in pairs if abs(p[0]) >= crit]
    print(f"|rho| >= {crit:.2f} is p < 0.05 at n = {n}: "
          f"{len(sig)} of {len(pairs)} pairs")
    for r, a, b in sorted(pairs, reverse=True)[:5]:
        print(f"    strongest  {r:+.2f}  {a} <-> {b}")
    for r, a, b in sorted(pairs)[:3]:
        print(f"    weakest    {r:+.2f}  {a} <-> {b}")


def plot(d: dict, out: Path) -> None:
    """The lower triangle only, sized and labelled for a slide.

    The matrix is symmetric with a constant diagonal, so a full 9x9 grid
    shows every correlation twice and nine cells that carry no information.
    Dropping the mirror halves what the eye has to scan -- which matters more
    on a projector than on paper.

    The axes are TRIMMED as well as masked: the first row and the last column
    of the full matrix are entirely upper-triangle, so they are cut rather
    than left as an empty band of axis furniture. Rows are therefore tasks
    2..9 and columns tasks 1..8.

    Style comes from plots/style.apply_style (the shared serif rcParams) at
    WIDTH_FULL, exactly as the paper figures do: a talk slide showing this
    beside the two task diagrams should not switch typeface halfway through.
    """
    import matplotlib.pyplot as plt

    apply_style(extra=COMPACT_RC_PARAMS)
    names, groups = d["tasks"], d["groups"]
    rho = np.asarray(d["rho"])
    n = len(names)
    crit = crit_rho(d["n_methods"])

    # M[i, j] is rho between task i+1 (row) and task j (column); everything
    # with j > i is the mirror image and is masked out.
    M = np.ma.masked_array(rho[1:, :-1],
                           mask=np.triu(np.ones((n - 1, n - 1), bool), k=1))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(alpha=0.0)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    # Margins, not a tight bbox: the left has to hold the row labels AND the
    # rotated family brackets outside them, the bottom the 30-degree column
    # labels. The canvas is far wider than tall, so the cells are NOT square
    # (aspect="auto") -- square cells would leave the triangle 1.4 in wide.
    fig.subplots_adjust(left=0.330, right=0.895, bottom=0.345, top=0.985)
    im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    for i in range(n - 1):
        for j in range(i + 1):
            v = M[i, j]
            # Bold marks the pairs that clear p < 0.05 at this n, so the ones
            # that survive are visible without the printed table beside it.
            # White ink only where the cell is genuinely dark. The cut has
            # to sit clear of the values actually on the figure: at 0.6 it
            # fell between +0.59 and +0.61, which print on indistinguishable
            # reds and so looked like an encoding rather than a threshold.
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.75 else "#111111",
                    fontweight="bold" if abs(v) >= crit else "normal")

    ax.set_xticks(range(n - 1))
    ax.set_yticks(range(n - 1))
    ax.set_xticklabels(names[:-1], rotation=30, ha="right",
                       rotation_mode="anchor", fontsize=8)
    ax.set_yticklabels(names[1:], fontsize=8)
    for tick, g in zip(ax.get_xticklabels(), groups[:-1]):
        tick.set_color(GROUP_COLOR[g])
    for tick, g in zip(ax.get_yticklabels(), groups[1:]):
        tick.set_color(GROUP_COLOR[g])

    # Family gaps. An outline around each family's own block was tried and
    # removed: the blocks are triangular, so every outline has a diagonal
    # edge that reads as a chart element of its own.
    for e in [i for i in range(1, n) if groups[i] != groups[i - 1]]:
        ax.plot([e - 0.5, e - 0.5], [e - 1.5, n - 1.5], color="white", lw=2.5)
        ax.plot([-0.5, e - 0.5], [e - 1.5, e - 1.5], color="white", lw=2.5)

    # Family brackets down the left margin. The row groups are contiguous
    # (2 / 4 / 2), so one bracket per family names the block a slide is
    # pointing at without the audience matching tick colors to a legend.
    #
    # Placed AFTER a draw, from the measured extent of the row labels: x is
    # in axes fraction, and how far left the labels reach depends on the
    # longest one, so a hardcoded offset silently prints the bracket on top
    # of "Partner Firm Centroid" the first time a task is renamed.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(rend)
    lab_x0 = min(t.get_window_extent(rend).x0 for t in ax.get_yticklabels())
    edge = (lab_x0 - ax_bb.x0) / ax_bb.width          # labels' left edge
    tr = ax.get_yaxis_transform()
    rows = groups[1:]
    starts = [i for i in range(len(rows)) if i == 0 or rows[i] != rows[i - 1]]
    for a in starts:
        b = next((i for i in range(a + 1, len(rows) + 1)
                  if i == len(rows) or rows[i] != rows[a]), len(rows))
        c = GROUP_COLOR[rows[a]]
        ax.plot([edge - 0.012, edge - 0.012], [a - 0.42, b - 1 + 0.42],
                transform=tr, color=c, lw=1.6, clip_on=False,
                solid_capstyle="butt")
        # HORIZONTAL, not rotated: at this canvas height a two-row block is
        # ~0.4 in tall, shorter than "Prediction" set sideways.
        ax.text(edge - 0.025, (a + b - 1) / 2, rows[a].replace(" ", "\n"),
                transform=tr, color=c, fontsize=8, ha="right", va="center",
                linespacing=1.0)

    ax.set_xticks(np.arange(-0.5, n - 1, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n - 1, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    # NO CAPTION BLOCK. The upper triangle is left empty on purpose: the
    # figure is shown on a slide that carries its own heading, and the one
    # thing the annotation explained -- that bold marks p < 0.05 -- is said
    # out loud instead. Reinstate from git if the figure ever has to stand
    # alone on a page.

    # Explicit colorbar axes: ``fig.colorbar(ax=ax)`` steals width from ax,
    # which would fight the fixed margins set above.
    cax = fig.add_axes([0.912, 0.345, 0.012, 0.64])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label(r"Rank Correlation $\rho$", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_visible(False)

    paths = save_figure(fig, out, bbox_inches=None)
    plt.close(fig)
    print("\n[wrote " + ", ".join(str(p) for p in paths) + "]")


# The scatter's point labels come from plots/style.SERIES_STYLES, so a method
# is named on this figure exactly as it is named on every other paper figure
# ("Time Warp", "Same Stock Views", "Multihead", ...). ROSTER["label"] stays
# the longer form because the PRINTED table has no color to carry the family
# and "Return" alone would read as a task there.
FAM_COLOR = {"LeJEPA": "tab:blue", "Supervised": "tab:gray",
             "SSL": "tab:orange", "Untrained": "black"}


def point_label(entry: dict) -> str:
    return entry.get("point") or SERIES_STYLES[entry["style"]]["label"]


def plot_split(d: dict, out: Path, y_groups: tuple[str, ...] | None = None,
               ylabel: str | None = None) -> None:
    """Average rank on the three forecasting tasks vs on the other six.

    The matrix says the nine evaluations disagree; this says what that costs
    anyone who wants one ordering. Each method is one point, and the two
    coordinates are the two things the nine tasks turn out to measure.

    BOTH AXES ARE INVERTED so rank 1 sits top and right: the eye reads
    up-and-right as better, and a rank axis running N -> 1 is the only way
    to get that without plotting a negated rank nobody can read off. The
    axis labels say "1 = best" so the inversion never has to be guessed.

    Deliberately bare: no y = x line, no median crosshairs, no quadrant
    captions, no title. Every one of those was drawn at some point and every
    one competed with the point labels, which are the content.

    Point labels and family colors come from plots/style so a method is named
    and coloured here exactly as on every other paper figure.

    ``y_groups`` narrows the y axis to a subset of the non-prediction groups.
    The default (None = all six) is the figure the matrix is about and the one
    to quote. The single exception drawn is ``("Organization",)`` -- the four
    fixed-panel tasks alone -- for a poster whose latent section shows only
    those panels, so its scatter cannot promise a factor axis the poster never
    draws. Two similar scatters is exactly the quote-the-wrong-one hazard this
    module was built to avoid, so they are given DIFFERENT y-axis labels and
    different stems, and each prints its own rho.
    """
    import matplotlib.pyplot as plt

    apply_style(extra=COMPACT_RC_PARAMS)
    R = np.asarray(d["ranks"])
    groups = d["groups"]
    pred = [i for i, g in enumerate(groups) if g == PRED_GROUP]
    if y_groups is None:
        other = [i for i, g in enumerate(groups) if g != PRED_GROUP]
    else:
        other = [i for i, g in enumerate(groups) if g in y_groups]
        if not other:
            raise SystemExit(f"no task in groups {y_groups}")
    x, y = R[:, pred].mean(1), R[:, other].mean(1)
    n = len(d["labels"])
    if ylabel is None:
        ylabel = f"Latent-Structure Rank  (1 = best of {n})"
    rho_xy, p_xy = spearmanr(x, y)
    print(f"{out.name}: y = mean rank over "
          f"{', '.join(d['tasks'][i] for i in other)}\n"
          f"    rho(forecasting, y) = {rho_xy:+.3f}  (p = {p_xy:.2f}, "
          f"n = {n})")

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    # Same canvas as the matrix (see FIG_SIZE). The legend sits OUTSIDE the
    # axes on the right: at this height it would cover a quarter of the plot.
    fig.subplots_adjust(left=0.100, right=0.835, bottom=0.175, top=0.975)
    for fam, c in FAM_COLOR.items():
        m = [i for i, f in enumerate(d["families"]) if f == fam]
        ax.scatter(x[m], y[m], s=28, c=c, edgecolor="white", linewidth=0.6,
                   label=fam, zorder=3)

    texts = [ax.annotate(point_label(METHODS[i]), (x[i], y[i]),
                         textcoords="offset points", xytext=(0, 5),
                         ha="center", va="bottom", fontsize=7.5, zorder=4,
                         color=FAM_COLOR[d["families"][i]])
             for i in range(n)]

    lo, hi = 0.2, n + 0.8
    ax.set_xlim(hi, lo)          # inverted: rank 1 on the RIGHT
    ax.set_ylim(hi, lo)          # inverted: rank 1 at the TOP
    ax.set_xlabel(f"Forecasting Rank  (1 = best of {n})", fontsize=9)
    # Two lines: at this canvas height the one-line title overruns the axis.
    ax.set_ylabel(ylabel.replace("  (", "\n("), fontsize=9)
    ax.set_xticks([1, 5, 10, 15, n])
    ax.set_yticks([1, 5, 10, 15, n])
    ax.tick_params(labelsize=8)
    leg = ax.legend(fontsize=8, frameon=False, loc="center left",
                    bbox_to_anchor=(1.01, 0.5), handletextpad=0.3,
                    borderpad=0.5, labelspacing=0.4)
    leg.set_zorder(5)

    # Label placement: N points on an NxN grid collide by
    # construction, so each label picks a SLOT. Candidates are tried in
    # preference order (above first, it reads best) and scored on how much
    # they overlap the markers, the labels already placed, and the legend;
    # the first clean slot wins, else the least-bad one. A push-apart
    # relaxation was tried first and oscillated -- two labels shoving each
    # other back and forth until the iteration budget ran out.
    from matplotlib.transforms import Bbox

    CANDIDATES = [(0, 5), (0, -6), (7, 1), (-7, 1), (7, -4), (-7, -4),
                  (0, 12), (0, -13), (10, 4), (-10, 4), (10, -8), (-10, -8)]
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    PAD = 4.0                                   # px halo around each marker
    pts = ax.transData.transform(np.column_stack([x, y]))
    marks = [Bbox([[px - PAD, py - PAD], [px + PAD, py + PAD]])
             for px, py in pts]
    obstacles = marks + [leg.get_window_extent(rend)]
    ax_bb = ax.get_window_extent(rend)

    def cost(box, placed, dx, dy):
        c = 3.0 * sum(box.overlaps(o) for o in obstacles)
        c += 3.0 * sum(box.overlaps(b) for b in placed)
        c += 2.0 * (box.x0 < ax_bb.x0 or box.x1 > ax_bb.x1
                    or box.y0 < ax_bb.y0 or box.y1 > ax_bb.y1)
        # Prefer the slot NEAREST the marker among the ones that fit. Without
        # this the search is indifferent between every collision-free slot and
        # takes whichever comes first in the list, which parked "Same-stock
        # crops" a centimetre from its own dot and next to somebody else's.
        c += 0.35 * ((dx ** 2 + dy ** 2) ** 0.5) / 10.0
        return c

    def apply(t, dx, dy):
        t.set_position((dx, dy))
        t.set_ha("center" if dx == 0 else ("left" if dx > 0 else "right"))
        t.set_va("bottom" if dy >= 0 else "top")

    placed: list = []
    # Densest neighbourhoods first: those have the fewest workable slots, so
    # letting an isolated label take a good one early is what strands them.
    order = sorted(range(n), key=lambda i: -sum(
        1 for j in range(n)
        if j != i and abs(x[i] - x[j]) < 2.5 and abs(y[i] - y[j]) < 2.5))
    for i in order:
        best, best_c = CANDIDATES[0], None
        for dx, dy in CANDIDATES:
            apply(texts[i], dx, dy)
            c = cost(texts[i].get_window_extent(rend), placed, dx, dy)
            if best_c is None or c < best_c:
                best, best_c = (dx, dy), c
        apply(texts[i], *best)
        placed.append(texts[i].get_window_extent(rend))

    paths = save_figure(fig, out, bbox_inches=None)
    plt.close(fig)
    print("[wrote " + ", ".join(str(p) for p in paths) + "]")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-json", action="store_true",
                   help="replot from the saved table; no score loading")
    p.add_argument("--out", default=str(HERE / "task_rank_corr"),
                   help="output stem")
    a = p.parse_args()

    jpath = Path(a.out).with_suffix(".json")
    if a.from_json:
        d = json.loads(jpath.read_text())
    else:
        d = build_table()
        jpath.write_text(json.dumps(d, indent=1))
    print_table(d)
    plot(d, Path(a.out))
    stem = Path(a.out)
    plot_split(d, stem.with_name(stem.name + "_split"))
    plot_split(d, stem.with_name(stem.name + "_split_org"),
               y_groups=("Organization",),
               ylabel=f"Organization Rank  (1 = best of {d['n_methods']})")
    print(f"[wrote {jpath}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
