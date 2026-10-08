"""Appendix table: the classical baselines beside the supervised arms.

The question this answers is the one a reader asks after Table~\\ref{tab:probe_fit}
-- "fine, but how good is that in absolute terms?" -- and it is not answerable
from the main table, which compares pretraining methods to each other and to an
untrained network. Here the same supervised numbers sit beside AR(p), ARMA(p,q),
HAR-RV, GARCH(1,1), Ridge ARDL and a ridge/GBM read off the raw view.

ONE FIT BUDGET ON BOTH SIDES, WHICH IS THE WHOLE POINT. The supervised probe in
Table~\\ref{tab:probe_fit} is fit on the six-month pool the head itself trained
on (~2.5M rows at 36 anchors/day). The finance baselines were historically fit
on ONE month (``finance_panel_ic.json``, 358k rows), so putting the two on one
axis compared fit-set size as much as model class. This table reads
``finance_panel_ic_6mo.json`` instead -- the same baselines refit on the same
six-month pool, written by::

    uv run plots/finance_baselines/calculate_panel_ic.py --months <32> \\
        --fit-span 6 --horizons 900 --json plots/metrics/finance_panel_ic_6mo.json

It is worth knowing what that refit was worth, and ``--one-month-delta``
prints it rather than leaving it in this docstring to go stale. On the
reported panel it is not uniform: on RETURN it moved Ridge ARDL +0.0112 ->
+0.0177,
ARMA +0.0069 -> +0.0134 and AR +0.0068 -> +0.0128 (+58% to +94%), while on
volatility and spread every model moved by less than +0.0023. The noisiest
target is where one month of coefficient estimation was the binding
constraint, and it is not a cosmetic difference: at one month Ridge ARDL's return
cell sits BELOW the untrained floor, at six months it clears it.

THE SUPERVISED ROWS AND THE FLOOR ARE NOT RECOMPUTED HERE. They come from
``plots/core/probe_fit_table``'s own loaders, at its alpha and on its month
restriction, so a cell in this table and the same cell in Table~\\ref{tab:probe_fit}
cannot disagree. Import, don't re-derive: this table used to be a place where
the supervised numbers could go stale independently.

THE FROZEN TSFMs ARE THE OTHER "OFF-THE-SHELF" ANSWER. Chronos-2, Kronos and
TimesFM 3.0 are pretrained forecasters nobody here trained, so they sit with
the baselines rather than among the pretraining methods of
Table~\\ref{tab:probe_fit}. They are rows of the probe-breadth sweep itself
(scripts/eval/build_probe_breadth_manifest.py --tsfm): the same six-month pool
at 36 anchors/day, the same eval panel at 8, the same ridge at the same alpha,
read at their LAST layer and last token with the nine per-channel states
averaged (Kronos has no channel axis). So they come from ``load_results`` like
the supervised rows and cannot disagree with the sweep. What they do not get
is the encoder's eleven info-token channels -- the 2026-09-11 asymmetry -- which
is where the spread leak lives.

EVERY ROW IS READ ON ONE PANEL. ``comparable_months`` gives the months where
both the floor and some method scored, and the baselines are restricted to that
same per-task set before averaging -- otherwise the classical rows would
average over 32 months (they include 2008-03) against the floor's 31 and the
column would not be a comparison. The ``n`` column appears by itself if any row
still lands on a different count.

Usage::

    uv run plots/finance_baselines/baseline_table.py            # markdown
    uv run plots/finance_baselines/baseline_table.py --latex \\
        --out plots/finance_baselines/baseline_table.tex
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_REPO), str(_REPO / "plots"), str(_REPO / "plots" / "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from probe_fit_table import (  # noqa: E402
    TASKS, build_rows, comparable_months, floor_row, load_head, load_results,
    trim_family,
)

# Written by calculate_panel_ic.py --fit-span 6 --horizons 900. It lives beside
# the other per-month IC artifacts rather than here, which is where that script
# has always written and where plots/core/paths.py expects the metrics JSONs.
SIX_MONTH_JSON = _REPO / "plots" / "metrics" / "finance_panel_ic_6mo.json"
ONE_MONTH_JSON = _REPO / "plots" / "metrics" / "finance_panel_ic.json"

# The supervised arms this table shows for context. NOT the whole roster: the
# LeJEPA and SSL arms are Table~\ref{tab:probe_fit}'s subject and repeating
# fourteen of them here would bury the comparison this table is for.
SUP_FAMILY = "Supervised"

# A view learner is named for the crop it reads; everything else in the
# artifact is a hand-built forecasting model. Keyed on the name PREFIX rather
# than a list, so a tail added to view_models.default_view_learners() lands in
# the right family without a second registry here.
# CAREFUL: this is a prefix match on the model KEY, so a CLASSICAL model
# keyed "ridge_*" silently files itself under the raw-view family. That is
# why the ridge-penalised ARDL is keyed "ardl" and not "ridge_ardl".
VIEW_PREFIXES = ("ridge_", "hgb_")
CLASSICAL_FAMILY = "Classical forecasting models"
VIEW_FAMILY = "Learners on the raw view"

# REPORTING ORDER, mirroring the two registries: models.default_baselines()
# and view_models.default_view_learners(). Spelled out rather than imported
# because importing those pulls panel_tables -> xs_ic_eval -> torch, which is
# a heavy dependency for a script that only formats numbers. A model absent
# from here still renders -- it sorts after these, alphabetically -- so the
# list going stale costs row order, never a missing row.
MODEL_ORDER = ("mean_reversion", "ar_p", "arma_pq", "har_rv", "garch11",
               "ardl", "ridge_tail8", "ridge_tail24", "ridge_tail64",
               "hgb_tail24")

# Bold marks the best two per target WITHIN EACH SIDE -- the supervised block
# and the baselines -- rather than the best two overall. Two marked pairs per
# column is the comparison the table is for: the supervised pair is always the
# column's own specialist and the multihead, the baseline pair is whatever
# came closest, and the gap between the two pairs is the result. Bolding the
# best two overall would mark two supervised rows and stop there, which says
# nothing a reader cannot already see from the column order.
#
# Table~\ref{tab:probe_fit} bolds only its self-supervised field, and that is
# not an inconsistency: nearly every cell there is struck, so the supervised
# rows need no mark to stand out. Nothing is struck here.
N_BOLD = 2

# Series keys as build_probe_breadth_manifest.py --tsfm writes them into
# manifest_tsfm.index.json, which plots/core/paths.py lists among INDEXES.
TSFM_FAMILY = "Frozen time-series foundation models"
TSFM_ARMS = [("tsfm_chronos2", "Chronos-2"),
             ("tsfm_kronos", "Kronos"),
             ("tsfm_timesfm3", "TimesFM 3.0")]

FAMILY_ORDER = [SUP_FAMILY, CLASSICAL_FAMILY, VIEW_FAMILY, TSFM_FAMILY]

DEFAULT_LABEL = "tab:finance_baselines"


def default_caption():
    return (
        r"Forecasting rank IC at the 15-minute horizon for classical "
        r"baselines, beside the supervised arms of "
        r"Table~\ref{tab:probe_fit}, averaged over evaluation months. "
        r"The baselines are read off the "
        r"\emph{normalised view tensor the encoder receives}, not the raw "
        r"order book. Based on the results on the Optimization Set, the frozen "
        r"TSFMs were nearly uniformly optimal at the last layer for all three "
        r"tasks. Supervised cells give the ridge probe with that arm's own "
        r"head IC in parentheses. \textcolor{green!45!black}{\textbf{Green}} "
        r"marks the best two strongest supervised arms and the two strongest "
        r"baselines.")


def load_baselines(path: Path, restrict=None):
    """``[(model, label, family, {task: (ic, None, n_months)})]``.

    Shaped like ``probe_fit_table.build_rows``' rows so one renderer serves
    both -- the fourth slot is that function's ``(probe, head, n)`` triple
    with no head, because a GARCH has no second readout.
    """
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. Write it with:\n"
            f"  uv run plots/finance_baselines/calculate_panel_ic.py "
            f"--months <fit months> --fit-span 6 --horizons 900 \\\n"
            f"      --json {path}")
    recs = json.loads(path.read_text())

    spans = {r.get("n_fit_months") for r in recs}
    by: dict[str, dict] = {}
    labels: dict[str, str] = {}
    for r in recs:
        if r["target"] not in {t for t, _ in TASKS}:
            continue
        labels[r["model"]] = r["predictor"]
        by.setdefault(r["model"], {}).setdefault(r["target"], {})[
            r["eval_month"]] = float(r["ic"])

    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    rows = []
    for model in sorted(by, key=lambda m: (rank.get(m, len(rank)), m)):
        fam = (VIEW_FAMILY if model.startswith(VIEW_PREFIXES)
               else CLASSICAL_FAMILY)
        cells = {}
        for task, _ in TASKS:
            per = by[model].get(task, {})
            months = sorted(per)
            if restrict is not None:
                months = [m for m in months if m in restrict.get(task, ())]
            if not months:
                cells[task] = (float("nan"), None, 0)
                continue
            cells[task] = (sum(per[m] for m in months) / len(months), None,
                           len(months))
        rows.append((model, labels[model], fam, set(), cells, ""))
    return rows, spans


def tsfm_rows(ics, restrict=None):
    """The frozen-TSFM rows, in ``build_rows``' tuple shape, off the sweep."""
    rows = []
    for series, label in TSFM_ARMS:
        cells = {}
        for task, _ in TASKS:
            per = ics.get(series, {}).get(task, {})
            months = sorted(m for m in per
                            if restrict is None or m in restrict.get(task, ()))
            cells[task] = ((sum(per[m] for m in months) / len(months), None,
                            len(months)) if months
                           else (float("nan"), None, 0))
        if any(c[2] for c in cells.values()):
            rows.append((series, label, TSFM_FAMILY, set(), cells, ""))
    return rows


def one_month_delta(restrict, out=sys.stderr):
    """What the six-month refit bought, per model and target, on THIS panel.

    The comparison the table rests on but does not show: the same models, the
    same eval months, fit on one month (``finance_panel_ic.json``) against six
    (``finance_panel_ic_6mo.json``). Printed rather than written into the
    caption, because it is a claim about two artifacts and only one of them is
    this table's subject.
    """
    if not ONE_MONTH_JSON.exists():
        print(f"  no {ONE_MONTH_JSON.name}; skipping the one-month delta",
              file=out)
        return
    one = json.loads(ONE_MONTH_JSON.read_text())
    six = json.loads(SIX_MONTH_JSON.read_text())

    def avg(recs, model, task):
        per = {r["eval_month"]: r["ic"] for r in recs
               if r["model"] == model and r["target"] == task}
        months = [m for m in per if m in restrict.get(task, ())]
        return (sum(per[m] for m in months) / len(months)) if months else None

    print("  one-month -> six-month fit pool, on the reported panel:",
          file=out)
    for task, label in TASKS:
        for model in MODEL_ORDER:
            a, b = avg(one, model, task), avg(six, model, task)
            if a is None or b is None:
                continue
            print(f"    {label:11s} {model:16s} {a:+.4f} -> {b:+.4f}  "
                  f"({b - a:+.4f})", file=out)


def _fmt(v, digits=4):
    return f"{v:.{digits}f}"


def _mark(rows, floor):
    """``{task: {model}}``: the best ``N_BOLD`` per target on EACH side.

    Ranked on the probe number, never the head. A head is a different readout,
    and ranking the two together would put a supervised arm above its own
    block for having a head rather than for its features -- probe_fit_table's
    rule, for probe_fit_table's reason.
    """
    sides = ([r for r in rows if r[2] == SUP_FAMILY],
             [r for r in rows if r[2] != SUP_FAMILY])
    top = {}
    for task, _ in TASKS:
        marked = set()
        for side in sides:
            cand = [(r[4][task][0], r[0]) for r in side
                    if not math.isnan(r[4][task][0])]
            marked |= {k for _, k in sorted(cand, reverse=True)[:N_BOLD]}
        top[task] = marked
    return top


def render_latex(rows, floor, *, show_n, caption, label, size, tabcolsep):
    from style import render_latex_table, tex_escape as _tex_escape, tex_mark

    top = _mark(rows, floor)

    def cell(r, task):
        # NO FLOOR-RELATIVE MARKING. This table is read for absolute context
        # -- "how good is 0.0317, really?" -- and the floor row answers that
        # by sitting at the bottom of the column where a reader can compare
        # against it directly. Striking every cell beneath it would put a
        # dozen rules through a table whose whole subject is the rows'
        # magnitudes, and it would assert a binary (beats/does not) on gaps
        # like Ridge ARDL's +0.0002 on return that the numbers do not support.
        ic, head, _n = r[4][task]
        if math.isnan(ic):
            return "--"
        # GREEN ONLY: never red, for the reason above.
        txt = tex_mark(_fmt(ic), best=r[0] in top[task])
        if head is not None and not math.isnan(head):
            txt = txt + r"\,($" + f"{head:.4f}" + r"$)"
        return txt

    ncol = 1 + len(TASKS) + (1 if show_n else 0)
    body = []
    by_fam: dict[str, list] = {}
    for r in rows:
        by_fam.setdefault(r[2], []).append(r)
    for fam in FAMILY_ORDER + [f for f in by_fam if f not in FAMILY_ORDER]:
        if fam not in by_fam:
            continue
        body.append([r"\addlinespace \multicolumn{" + str(ncol)
                     + r"}{l}{\emph{" + _tex_escape(fam) + r"}}"])
        for r in by_fam[fam]:
            line = [_tex_escape(trim_family(r[1], fam))]
            line += [cell(r, t) for t, _ in TASKS]
            if show_n:
                line.append(str(max(c[2] for c in r[4].values())))
            body.append(line)
    if floor:
        body.append([r"\addlinespace \multicolumn{" + str(ncol)
                     + r"}{l}{\emph{Untrained floor}}"])
        line = [_tex_escape(floor[1])] + [cell(floor, t) for t, _ in TASKS]
        if show_n:
            line.append(str(max(c[2] for c in floor[4].values())))
        body.append(line)

    hdr = [""] + [h for _, h in TASKS] + (["n"] if show_n else [])
    return (render_latex_table(
                body, header_rows=[hdr],
                col_spec="l" + "c" * (len(TASKS) + (1 if show_n else 0)),
                caption=caption, label=label if caption else None,
                caption_above=True, small=bool(size), size=size or None,
                pre=([r"\setlength{\tabcolsep}{" + f"{tabcolsep:g}" + "pt}"]
                     if tabcolsep > 0 else []),
            ))


def render_markdown(rows, floor, *, show_n):
    hdr = ["model"] + [h for _, h in TASKS] + (["n"] if show_n else [])
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    top = _mark(rows, floor)
    for r in (rows + ([floor] if floor else [])):
        line = [trim_family(r[1], r[2])]
        for task, _ in TASKS:
            ic, head, _n = r[4][task]
            if math.isnan(ic):
                line.append("--")
                continue
            s = _fmt(ic)
            if r[0] in top.get(task, ()):
                s = f"**{s}**"
            if head is not None and not math.isnan(head):
                s += f" ({head:.4f})"
            line.append(s)
        if show_n:
            line.append(str(max(c[2] for c in r[4].values())))
        out.append("| " + " | ".join(line) + " |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", default=str(SIX_MONTH_JSON),
                   help="the six-month baseline artifact")
    p.add_argument("--alpha", type=float, default=10.0,
                   help="ridge alpha for the SUPERVISED rows; must match "
                        "probe_fit_table.py's, or the two tables disagree")
    p.add_argument("--latex", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--caption", default=None)
    p.add_argument("--label", default=DEFAULT_LABEL)
    p.add_argument("--size", default="",
                   help="LaTeX size macro without the backslash; empty emits "
                        "none")
    p.add_argument("--tabcolsep", type=float, default=0.0)
    p.add_argument("--one-month-delta", action="store_true",
                   help="also print what the six-month refit bought, against "
                        "finance_panel_ic.json on the same eval months")
    p.add_argument("--show-n", dest="show_n", action="store_const", const=True,
                   default=None, help="force the month-count column on")
    p.add_argument("--no-n", dest="show_n", action="store_const", const=False,
                   help="force it off")
    a = p.parse_args()

    ics, ns, nfiles, unknown = load_results(a.alpha)
    if not ics:
        raise SystemExit("no probe-breadth results; the supervised rows and "
                         "the floor come from there")
    head = load_head()
    restrict = comparable_months(ics, TASKS)
    sizes = {t: len(v) for t, v in restrict.items()}
    print("  floor-covered panel: "
          + ", ".join(f"{t.replace('_900', '')} {n}" for t, n in sizes.items())
          + " month(s)", file=sys.stderr)

    sup = [r for r in build_rows(ics, ns, head, min_months=1, fair=False,
                                 restrict=restrict)
           if r[2] == SUP_FAMILY]
    base, spans = load_baselines(Path(a.json), restrict=restrict)
    if spans - {6}:
        print(f"  NOTE: baseline records carry fit spans {sorted(spans)}; a "
              f"span below 6 is a month the mosaic could not fill (it starts "
              f"2008-01) and is averaged in as itself.", file=sys.stderr)
    floor = floor_row(ics, min_months=1, restrict=restrict)

    tsfm = tsfm_rows(ics, restrict=restrict)
    print("  frozen TSFMs: " + (", ".join(
        f"{r[1]} {max(c[2] for c in r[4].values())} month(s)" for r in tsfm)
        or "no results yet"), file=sys.stderr)

    rows = sup + base + tsfm
    counts = {max(c[2] for c in r[4].values()) for r in rows}
    if floor:
        counts.add(max(c[2] for c in floor[4].values()))
    show_n = len(counts) > 1 if a.show_n is None else a.show_n

    if a.one_month_delta:
        one_month_delta(restrict)

    caption = a.caption if a.caption is not None else default_caption()
    txt = (render_latex(rows, floor, show_n=show_n, caption=caption,
                        label=a.label, size=a.size, tabcolsep=a.tabcolsep)
           if a.latex else render_markdown(rows, floor, show_n=show_n))
    if a.out:
        # The paper copy ends on a blank line; keep regenerations byte-equal.
        Path(a.out).write_text(txt + "\n\n")
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
