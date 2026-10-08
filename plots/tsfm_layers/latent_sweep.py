"""Latent organization of the three frozen TSFMs, layer by layer.

Chronos-2, Kronos and TimesFM 3.0 at every hidden state, on the six columns of
the latent table (plots/core/fixed_panel_table.tex), over its 31 eval months:
T1-T4 from the fixed panel, F1 (loading decode) and F2 (factor subspace
captured) from the Pelger factor stages.

Data: the ``_cmean`` wave (plots/latent_eval/run_cmean_eval.sh), which reads
each TSFM at the latent protocol's MEAN over valid patches with the nine
per-channel states AVERAGED -- the prediction evals' readout. The stars come
from the same readout on the Optimization Set (``_cmean_opt``, --make-picks).

TWO REFERENCE LINES PER PANEL, both read through fixed_panel_table's own
loaders so neither can disagree with the table:
  black dashed  the untrained Random ViT (the table's floor row)
  gray dashed   the best trained method in that column of the table, named
                in the panel -- what a frozen TSFM would have to reach

TWO VERSIONS, one per readout of T1-T4 the table prints:
  --metric multiple  top-1 rate as a multiple of chance
  --metric pctile    mean percentile rank of the target (lower is better; the
                     axis is inverted so up is better on every panel)
F1 and F2 have one readout each and are identical in both.

A THIRD FIGURE, --metric rankme, is the layer-wise RankMe on its own (the
appendix RankMe table's figure, plots/latent_eval/rankme/rankme_table.py): the
effective rank of each layer's full-day embedding (scripts/eval/
rankme_fullday.py --tsfm -> rankme_tsfm.json). No star -- nothing was picked
on it -- and no best-in-table line: RankMe is reference, never ranked. It is
bounded by the embedding width, and the TSFMs' widths (768 / 832 / 1280) are
not the floor's 384, so the floor line is a level, not a bar.

THE STAR IS THE OPTIMIZATION-SET PICK (optimization_picks.json), never this
panel's argmax -- the argmax on the reported months would be selection on the
test set. The picks were made on the percentile readout, so on the multiple
figure a star need not sit at that curve's peak.

Usage::

    uv run python plots/tsfm_layers/latent_sweep.py --metric multiple
    uv run python plots/tsfm_layers/latent_sweep.py --metric pctile
    uv run python plots/tsfm_layers/latent_sweep.py --metric rankme
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (REPO, REPO / "plots", REPO / "plots/core",
           REPO / "plots/latent_eval/fixed_panel"):
    sys.path.insert(0, str(_p))

import fixed_panel_table as fpt  # noqa: E402
from style import (  # noqa: E402
    COMPACT_RC_PARAMS, SERIES_STYLES, WIDTH_FULL, WIDTH_HALF,
    add_bottom_legend, apply_style, save_figure, set_two_decimal_yticks,
)

# THE CHANNEL-AVERAGED WAVE (plots/latent_eval/run_cmean_eval.sh): each TSFM's
# nine per-channel states averaged, the prediction evals' readout. The reported
# 31 months are _cmean; the Optimization Set the stars are picked on is
# _cmean_opt. The series are tsfm_<fam>_cmean_l<L>.
TAG = "_cmean"
OPT_TAG = "_cmean_opt"
SERIES_SUFFIX = "_cmean"
PICKS = HERE / "optimization_picks.json"
RANKME = HERE / "rankme_tsfm.json"

# family -> number of transformer layers (hidden states 0..n)
FAMILIES = {"chronos2": 12, "kronos": 12, "timesfm3": 20}
# (panel title, optimization_picks task key, table column)
PANELS = [("T1: Partner NN", "T1 partner NN", "metric1"),
          ("T2: Day Centroid", "T2 day centroid", "metric2"),
          ("T3: Own-Firm Centroid", "T3 own-firm centroid", "metric3"),
          ("T4: Partner-Firm Centroid", "T4 partner centroid", "metric4"),
          ("F1: Loading Decode", "LOAD decode", "r"),
          ("F2: Factor Subspace", "SUBSP align", "rhobar")]
YLABEL = {"multiple": "Top-1 Rate / Chance",
          "pctile": "Mean Percentile Rank (%)",
          "r": "Mean Decode r",
          "rhobar": "Subspace Captured"}
FLOOR_STYLE = SERIES_STYLES["randinit"]
BEST_COLOR = "tab:gray"


def skey(fam: str, L: int) -> str:
    """The series key of one family's layer in this wave."""
    return f"tsfm_{fam}{SERIES_SUFFIX}_l{L}"


def geo_value(ent, metric):
    """(value, se) of one fixed-panel cell, on the table's own scale.

    The file carries each statistic's t over month means, so the SE is
    derived from it rather than re-estimated: for the hit rate t tests
    rate - chance, for the percentile t tests pctile - 0.5.
    """
    if metric == "multiple":
        c = ent["chance"]
        v = ent["rate"] / c
        se = abs(ent["rate"] - c) / abs(ent["t"]) / c if ent["t"] else np.nan
        return v, se
    v = ent["mean_pctile"]
    se = abs(v - 0.5) / abs(ent["rank_t"]) if ent["rank_t"] else np.nan
    return 100 * v, 100 * se


def factor_value(dec, sub, key, col):
    """(value, se) of F1/F2 for one series, month mean and its SE.

    The same aggregation fixed_panel_table.load_factors uses for the table:
    F1 averages the four total loadings per month, then months.
    """
    if col == "r":
        cell = dec["results"][f"{key}|emb"]
        a = np.mean([cell[t] for t in fpt.DECODE_COLS], axis=0)
    else:
        a = np.asarray(sub["results"][key]["rhobar"], float)
    a = a[np.isfinite(a)]
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size))


def load_rankme():
    """{series: (mean, se, n_months)} per TSFM layer, plus the table's floor.

    The TSFM layers come from rankme_tsfm.json; the floor is NOT re-measured
    there, it is the latent table's own Random ViT RankMe (rankme_latent.json,
    seeds collapsed per month first), so the line is the table's number.
    """
    from collections import defaultdict
    per = defaultdict(dict)
    for r in json.loads(RANKME.read_text()):
        per[r["series"]][r["month"]] = float(r["rankme"])
    out = {}
    for k, months in per.items():
        v = np.array([x for _, x in sorted(months.items())])
        out[k] = (float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)),
                  len(v))
    mu, se, n, _ws = fpt.load_rankme()[fpt.FLOOR]
    out["floor"] = (mu, se, n)
    return out


def table_refs(metric):
    """{column: (floor, best value, best label)} from the latent table itself."""
    geo, seen = fpt.load_geo(fpt.DEFAULT_TAGS, required=False)
    rows = fpt.select_rows(geo, seen)
    fac = fpt.load_factors(seen)
    roster = fpt.latex_roster()
    # The rows the paper table prints: its roster, minus the floor.
    keys = [k for k, _ in rows if k in roster and k != fpt.FLOOR]
    out = {}
    for _, _, col in PANELS:
        if col.startswith("metric"):
            val = {k: geo_value(geo[k][col], metric)[0] for k in keys}
            pick = min if metric == "pctile" else max
            floor = geo_value(geo[fpt.FLOOR][col], metric)[0]
        else:
            val = {k: fac[k][col] for k in keys if col in fac.get(k, {})}
            pick = max
            floor = fac[fpt.FLOOR][col]
        best = pick(val, key=val.get)
        out[col] = (floor, val[best], roster[best][0])
    return out


def _style_axis(ax):
    ax.grid(False)
    ax.tick_params(axis="both", which="both", length=2.5, width=0.6, pad=1.5)
    ax.set_xticks([0, 0.5, 1])
    ax.margins(x=0.04)


def _legend_entries(with_best: bool):
    handles = [Line2D([], [], marker="o", ms=2.4, lw=1.1,
                      color=SERIES_STYLES[f]["color"]) for f in FAMILIES]
    labels = [SERIES_STYLES[f]["label"] for f in FAMILIES]
    if with_best:
        handles.append(Line2D([], [], marker="*", ms=7, color="0.35",
                              ls="none"))
        labels.append("Optimization-Set Pick")
    handles.append(Line2D([], [], ls="--", lw=1.1,
                          color=FLOOR_STYLE["color"]))
    labels.append("Random ViT")
    if with_best:
        handles.append(Line2D([], [], ls="--", lw=1.1, color=BEST_COLOR))
        labels.append("Best in Table")
    return handles, labels


def draw_rankme(out: str, n_months: int) -> None:
    """The layer-wise RankMe, one panel."""
    rk = load_rankme()
    short = {k: n for k, (_, _, n) in rk.items() if n != n_months}
    if short:
        raise SystemExit(f"RankMe months short of {n_months}: {short}")
    apply_style(extra=COMPACT_RC_PARAMS)
    fig, ax = plt.subplots(figsize=(WIDTH_HALF, WIDTH_HALF * 0.78))
    prof = {}
    for fam, n in FAMILIES.items():
        st = SERIES_STYLES[fam]
        xs = np.arange(n + 1) / n
        ys, es = np.array([rk[skey(fam, L)][:2]
                           for L in range(n + 1)]).T
        prof[fam] = (ys, es)
        ax.fill_between(xs, ys - es, ys + es, color=st["color"], alpha=0.16,
                        lw=0)
        ax.plot(xs, ys, marker="o", ms=2.4, lw=1.1, color=st["color"],
                zorder=3)
    ax.axhline(rk["floor"][0], ls="--", lw=1.1, color=FLOOR_STYLE["color"],
               zorder=2)
    _style_axis(ax)
    ax.set_xlabel("Relative Depth")
    ax.set_ylabel("RankMe")
    fig.tight_layout()
    handles, labels = _legend_entries(with_best=False)
    add_bottom_legend(fig, handles, labels, ncol=2, bottom_reserve=0.3,
                      columnspacing=1.2, handletextpad=0.5)
    save_figure(fig, out)
    plt.close(fig)
    Path(out + ".json").write_text(json.dumps(
        {"metric": "rankme", "floor": rk["floor"][0],
         "families": {f: {"value": ys.tolist(), "se": es.tolist()}
                      for f, (ys, es) in prof.items()}}, indent=1))
    print(f"wrote {out}.{{png,pdf,json}}  ({n_months} months)")


def make_picks() -> None:
    """Re-make the stars: per (family, task) argmax layer on the Optimization Set.

    Scored the way the 2026-09-06 picks were: T1-T4 on the percentile readout
    (lowest mean rank percentile), F1 on the month-mean decode r, F2 on the
    month-mean canonical correlation. No floor is subtracted -- a constant
    per task cannot move an argmax.
    """
    geo = json.loads((fpt.PANELS / f"fixed_panel_P3S2{OPT_TAG}.json").read_text())
    models, months = geo["models"], geo["months"]
    dec = json.loads((fpt.FACTORS / f"decode_loadings{OPT_TAG}.json").read_text())
    sub = json.loads((fpt.FACTORS / f"subspace_alignment{OPT_TAG}.json").read_text())
    picks = {}
    for fam, n in FAMILIES.items():
        picks[fam] = {}
        for _, task, col in PANELS:
            if col.startswith("metric"):
                vals = [-geo_value(models[skey(fam, L)][col], "pctile")[0]
                        for L in range(n + 1)]
            else:
                vals = [factor_value(dec, sub, skey(fam, L), col)[0]
                        for L in range(n + 1)]
            picks[fam][task] = int(np.argmax(vals))
    PICKS.write_text(json.dumps({
        "_source": (f"latent_sweep.py --make-picks: the per-(family, task) "
                    f"argmax layer on the Optimization Set ({OPT_TAG}, eval "
                    f"months {' '.join(months)}), T1-T4 on the lowest mean "
                    f"rank percentile, LOAD on the mean total-loading decode "
                    f"r, SUBSP on the mean canonical correlation. Readout: "
                    f"time mean over valid patches, nine channels AVERAGED "
                    f"(the {TAG} series)."),
        "picks": picks}, indent=1))
    print(f"wrote {PICKS} from {len(months)} Optimization-Set months")
    for fam, d in picks.items():
        print(f"  {fam:10s} " + "  ".join(f"{t.split()[0]} L{L}"
                                           for t, L in d.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--make-picks", action="store_true",
                    help="re-make optimization_picks.json from the "
                         "Optimization-Set wave, then exit")
    ap.add_argument("--metric", choices=("multiple", "pctile", "rankme"),
                    default="multiple")
    ap.add_argument("--out", default=None,
                    help="figure basename (default latent_sweep_<metric>)")
    a = ap.parse_args()
    if a.make_picks:
        make_picks()
        return 0
    out = a.out or str(HERE / f"latent_sweep_{a.metric}")

    geo = json.loads((fpt.PANELS / f"fixed_panel_P3S2{TAG}.json").read_text())
    models, months = geo["models"], geo["months"]
    if a.metric == "rankme":
        draw_rankme(out, len(months))
        return 0
    dec = json.loads((fpt.FACTORS / f"decode_loadings{TAG}.json").read_text())
    sub = json.loads((fpt.FACTORS / f"subspace_alignment{TAG}.json").read_text())
    for name, x in (("decode", dec), ("subspace", sub)):
        if list(x["months"]) != list(months):
            raise SystemExit(f"{name} months differ from the fixed panel's")
    picks = json.loads(PICKS.read_text())["picks"]
    refs = table_refs(a.metric)

    prof = {}
    for fam, n in FAMILIES.items():
        for _, _, col in PANELS:
            pts = []
            for L in range(n + 1):
                key = skey(fam, L)
                pts.append(geo_value(models[key][col], a.metric)
                           if col.startswith("metric")
                           else factor_value(dec, sub, key, col))
            prof[fam, col] = np.array(pts)

    apply_style(extra=COMPACT_RC_PARAMS)
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH_FULL, WIDTH_FULL * 0.62))
    for ax, (title, task, col) in zip(axes.ravel(), PANELS):
        for fam, n in FAMILIES.items():
            st = SERIES_STYLES[fam]
            xs = np.arange(n + 1) / n
            ys, es = prof[fam, col].T
            ax.fill_between(xs, ys - es, ys + es, color=st["color"],
                            alpha=0.16, lw=0)
            ax.plot(xs, ys, marker="o", ms=2.4, lw=1.1, color=st["color"],
                    zorder=3)
            L = picks[fam][task]
            ax.plot([L / n], [ys[L]], marker="*", ms=9, color=st["color"],
                    mec="white", mew=0.6, ls="none", zorder=5)
        floor, best, best_label = refs[col]
        ax.axhline(floor, ls="--", lw=1.1, color=FLOOR_STYLE["color"],
                   zorder=2)
        ax.axhline(best, ls="--", lw=1.1, color=BEST_COLOR, zorder=2)
        # Which method the gray line is -- it changes from column to column.
        ax.annotate(best_label, xy=(1.0, best), xycoords=("axes fraction",
                    "data"), xytext=(-2, 2), textcoords="offset points",
                    ha="right", va="bottom", fontsize=6.5, color="0.35")
        # HEADROOM FOR THE NAME. The best line can sit at the edge of the data
        # (T3's is far above every TSFM), and its label is drawn just above
        # it on screen, so reserve room past it in the "better" direction --
        # which, on the inverted percentile panels, is the low end.
        lo, hi = ax.get_ylim()
        pad = 0.12 * (hi - lo)
        if a.metric == "pctile" and col.startswith("metric"):
            ax.set_ylim(min(lo, best - pad), hi)
        else:
            ax.set_ylim(lo, max(hi, best + pad))
        ax.set_title(title, pad=4)
        _style_axis(ax)
        set_two_decimal_yticks(ax, nbins=5)
        if a.metric == "pctile" and col.startswith("metric"):
            ax.invert_yaxis()
    ylab = YLABEL[a.metric]
    axes[0][0].set_ylabel(ylab)
    axes[1][0].set_ylabel(ylab)
    axes[1][1].set_ylabel(YLABEL["r"])
    axes[1][2].set_ylabel(YLABEL["rhobar"])
    axes[1][1].set_xlabel("Relative Depth")

    handles, labels = _legend_entries(with_best=True)
    fig.tight_layout()
    add_bottom_legend(fig, handles, labels, ncol=3, bottom_reserve=0.16,
                      columnspacing=1.2, handletextpad=0.5)
    save_figure(fig, out)
    plt.close(fig)

    summary = {"metric": a.metric, "months": months, "picks": picks,
               "table": {c: {"floor": f, "best": b, "best_label": l}
                         for c, (f, b, l) in refs.items()},
               "families": {fam: {col: {"value": prof[fam, col][:, 0].tolist(),
                                        "se": prof[fam, col][:, 1].tolist()}
                                  for _, _, col in PANELS}
                            for fam in FAMILIES}}
    Path(out + ".json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {out}.{{png,pdf,json}}  ({len(months)} months)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
