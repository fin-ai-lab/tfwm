"""The frozen-TSFM figure: three pretrained forecasters, one rank-IC panel set.

    all_metrics_tsfm_ic_minus_random   ΔIC vs a random-init encoder (default)
    all_metrics_tsfm_ic                absolute rank IC, with the untrained
                                       floor drawn as its own line (--absolute)

THE THREE ARMS are time-series foundation models read FROZEN — Chronos-2,
TimesFM 3.0 and Kronos-base — with the same ridge probe, on the same 32
months, against the same three targets as every other arm in the paper.
Nothing about them is trained here; what varies across the three is the
pretraining corpus and architecture of an off-the-shelf forecaster. TimesFM
2.5 and Sundial were dropped from the arm on 2026-09-11, which also removes
the 3.0-vs-2.5 pair — the one comparison in this folder that held the
architecture fixed and varied only the pretraining generation.

READ AT THE PREDICTION TOKEN. Every number here is the last token of the
context window, ``xs_ic_eval.PREDICT_POOL``, which is the token the random-init
floor below is itself read at. Until 2026-09-11 this figure was a mean-pooled
sweep minus a last-token floor, breaking the rule ``checkpoints.py`` states
outright — a floor is read the way the models it floors are read — and the
correction is not cosmetic: Chronos-2's raw spread change went +0.173 -> +0.217
and most of the nine argmax layers moved, so the picks were redone under it.
The latent half of this arm (``plots/tsfm_layers/latent_sweep.py``) reads at
``panel_lib.LATENT_POOL = "mean"`` and always did; the two constants disagree
because the tasks do.

ONE LAYER PER (FAMILY, TARGET), CHOSEN OFF-PANEL. A frozen model has a hidden
state at every depth and they do not score alike, so the layer is a
hyper-parameter. It is the argmax on the 5-month optimization set
(``scripts/experiments/holdout_months.py``, disjoint from these 32 in both the
training and the eval role) and it is re-scored here — see
``plots/tsfm_layers/README.md``. It varies BY TARGET, so a single family reads
a different depth in each panel (Chronos-2: return L5, vol L2, spread L2); the
summary table prints which. ``build_tsfm_ic.py`` refuses to make the picks and
score them on one panel.

THE FLOOR SEES MORE CHANNELS THAN THE MODELS DO, and that is an accepted
asymmetry rather than a bug (``plots/tsfm_layers/README.md``, "ONE FLOOR FOR
EVERY ARM"). It is templated off a real trained checkpoint, so it carries the
11-channel information token — log-scale per normalization group — on top of
the 9 the TSFMs see, while every family RevINs level and scale away. So a
POSITIVE ΔIC here is a lower bound, and the spread-change column needs the
caveat: the floor's +0.161 is mostly the mechanical ``-spread(t)`` leak
arriving through a channel the TSFMs are not given, which is why Kronos reads
-0.115 there and why that is not a claim that Kronos is worse than noise.

ONE HORIZON, SO POINTS AND NOT LINES. ``tsfm_layer_ic.py`` takes a --horizon
and the sweep ran at h=900 alone, so each arm is a single marker with a ±1 SE
bar. The three are dodged ±8% around the 15-minute tick purely so they do not
stack — the Return panel puts Chronos-2 and TimesFM 3.0 0.0013 apart, a quarter
of that panel's range — and all three are measured at exactly h=900.

READ AGAINST THE ON-TASK SUPERVISED SPECIALIST, in gray across all six
horizons — the arm trained on the very target the probe scores. LeJEPA (the
standard model, in the reserved blue) belongs here as the second reference and
is NOT currently drawn: ``lejepa-k2-lambda-*`` survives only in
lab/models-archive, retired 2026-09-06, where the ALiBi parameters make
every checkpoint fail strict ``load_state_dict`` and the scored ICs are against
the pre-``uniform`` cross-sectional target. main() drops a reference with no
checkpoints and says so rather than failing; the line returns when the arm is
re-trained and re-scored, and nothing else on the figure has to change.

The model-free mean-reversion line is deliberately NOT drawn — it appears as a
table row instead, which is where it matters, since on spread change the frozen
models and a one-scalar baseline are the comparison that decides whether the
pretraining bought anything.

COLOR. The families keep the assignment ``plots/tsfm_layers/families.py`` uses,
so a reader moving between the layer profiles and this figure tracks a model by
its hue. Two deviations, both inherited: Chronos-2 is cyan rather than its
usual tab:green, because green against TimesFM's orange is an OKLab ΔE of 0.7
under the Machado protanopia simulation, i.e. the same color for a protanopic
reader; Kronos moves off tab:purple to a darker #6a3d9a because this figure
adds the reserved LeJEPA tab:blue, which tab:purple collides with. TimesFM 3.0
keeps the deep burnt orange #5c2800 it was given to sit beside 2.5's
tab:orange, even though 2.5 is gone — the hue is not load-bearing here (the
binding pair on all three CVD simulations is Kronos against LeJEPA, ΔE 7.1
deutan, and dropping a family cannot improve it), and matching the layer-sweep
figures is worth more than reclaiming the canonical TimesFM orange. Marker
shape carries the family too, so nothing rests on hue alone.

Reads ``tsfm_ic.json`` (built by ``build_tsfm_ic.py`` off the layer sweep),
``xs_ic.json`` beside the LeJEPA checkpoints, the checked-in
``supervised_probe_ic.json``, and the random-init floor from ``randinit-fwd3-*``
via ``style.load_randinit_ic``.

Run:
    uv run plots/metrics/build_tsfm_ic.py      # the artifact, first
    uv run plots/metrics/tsfm_ic.py
    uv run plots/metrics/tsfm_ic.py --absolute
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR))

import metrics  # noqa: E402
from metrics import (  # noqa: E402
    SERIES_DEFS,
    TARGET_TYPES,
    WM_PANEL_TITLES,
    legend_bottom_reserve,
    legend_keys,
    load_ic_metrics,
    load_json_ic,
    plot_all_metrics,
)

# Ordered by the reported-panel result, strongest first, so the legend reads
# top-to-bottom like the panels do.
TSFM_KEYS = ["tsfm_chronos2", "tsfm_timesfm3", "tsfm_kronos"]
# The two reference LINES, drawn across all six horizons. cross_stock_k2ind is
# the standard model; supervised_specialist is the on-task cross-entropy
# encoder. Both are read from artifacts/checkpoints the other breadth figures
# use, so the numbers are line-for-line the ones on all_metrics_ic.
REFERENCE_KEYS = ["cross_stock_k2ind", "supervised_specialist"]
# Table-only: the model-free level term each change target subtracts, read off
# the view the encoder is fed. It has no return row and it costs a hue the
# palette would rather not spend (see COLOR), but on spread change it is the
# number the frozen models have to be read against, so it belongs in the
# printed comparison even when it is not on the axes.
TABLE_ONLY_KEYS = ["mean_reversion"]

# See COLOR. Per-figure hues, exactly as ssl_baselines_ic.py does it.
# TimesFM 3.0 is absent on purpose: it keeps style.py's #5c2800 unchanged, so
# it needs no per-figure entry (see COLOR).
TSFM_COLORS: dict[str, dict] = {
    "tsfm_chronos2": {"color": "tab:cyan"},
    "tsfm_kronos":   {"color": "#6a3d9a"},
    # The reserved LeJEPA blue, free on this figure. Relabelled from the
    # pairing-figure "K=2 same-ind." because here the pairing is not what is
    # being compared — this is the paper's standard model against four
    # off-the-shelf forecasters, and the arm needs the name it is known by.
    "cross_stock_k2ind": {"color": "tab:blue", "label": "LeJEPA"},
}

TABLE_H = 900   # the only horizon the TSFM sweep ran, and the paper's headline
ARTIFACT = OUT_DIR / "tsfm_ic.json"


def apply_palette() -> None:
    """Override the shared hues/labels for THIS figure only (see COLOR)."""
    for k, style in TSFM_COLORS.items():
        SERIES_DEFS[k].update(style)


def layers_used() -> dict[tuple[str, str], int]:
    """``{(family, target): layer}`` as recorded in the artifact.

    The layer is per (family, target), so a figure that did not say which
    depth each panel draws would be three results wearing one label.
    """
    if not ARTIFACT.is_file():
        return {}
    return {(r["model"], r["target"]): r["layer"]
            for r in json.loads(ARTIFACT.read_text())}


def summary_table(panel: set[str] | None, refs: list[str]) -> None:
    """ΔIC at ``TABLE_H`` per arm, with the layer each number was read at.

    ``refs`` is the reference lines that still have data — see main().
    """
    layers = layers_used()
    print(f"Rank IC at h={TABLE_H // 60} min, minus the random-init floor"
          + (f", over the {len(panel)} months every arm shares:" if panel
             else " (each arm over its OWN months):"))
    head = "".join(f"{WM_PANEL_TITLES[t]:>24s}" for t in TARGET_TYPES)
    print(f"  {'arm':22s}{head}{'n':>5s}")
    for key in TSFM_KEYS + refs + TABLE_ONLY_KEYS:
        cells, ns = [], []
        for t in TARGET_TYPES:
            task = f"{t}_{TABLE_H}"
            floor = metrics.load_randinit_ic(task)
            ic = load_json_ic(key, task) if SERIES_DEFS[key].get("json_ic") \
                else {r.eval_month: r.probe(task)
                      for r in metrics._select_runs(SERIES_DEFS[key])
                      if r.probe(task) is not None}
            use = sorted(panel & set(ic)) if panel is not None else sorted(ic)
            d = [ic[m] - floor[m] for m in use if m in floor]
            ns.append(len(d))
            fam = SERIES_DEFS[key].get("json_ic_model")
            # The depth, glued to the number it produced. Only the TSFM rows
            # have one; the two reference lines are whole encoders.
            tag = f" (L{layers[(fam, task)]})" if (fam, task) in layers else ""
            cells.append(f"{st.fmean(d):+.4f}{tag:>7s}".rjust(24) if d
                         else f"{'--':>24s}")
        print(f"  {SERIES_DEFS[key]['label']:22s}"
              + "".join(cells) + f"{(max(ns) if ns else 0):>5d}")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default=str(OUT_DIR))
    p.add_argument("--absolute", action="store_true",
                   help="Plot absolute rank IC and draw the random-init "
                        "encoder as its own line instead of subtracting it.")
    p.add_argument("--own-months", action="store_true",
                   help="Draw each arm over every month it has, rather than "
                        "over the months they all share. A coverage check, "
                        "not a comparison: the arms then average different "
                        "panels.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not ARTIFACT.is_file():
        raise SystemExit(
            f"{ARTIFACT.name} not built yet — run "
            f"uv run plots/metrics/build_tsfm_ic.py")
    apply_palette()
    fair = not args.own_months

    # A REFERENCE LINE WITH NO CHECKPOINTS IS DROPPED, not fatal. The fair
    # panel is an intersection, so one empty checkpoint series empties it and
    # the figure would refuse to draw arms that are themselves complete. That
    # is the state LeJEPA is in: lejepa-k2-lambda-* lives only in
    # lab/models-archive, whose README (2026-09-06) retires the whole
    # tree -- the ALiBi parameters make every checkpoint fail strict
    # load_state_dict, and its scored ICs are against the pre-``uniform``
    # cross-sectional target. It is not a path to repoint; the arm has to be
    # re-trained and re-scored. The TSFM arms, the supervised specialist
    # (checked-in json_ic) and the floor are all current, so they are drawn.
    refs = [k for k in REFERENCE_KEYS
            if SERIES_DEFS[k].get("json_ic")
            or metrics._select_runs(SERIES_DEFS[k])]
    for k in REFERENCE_KEYS:
        if k not in refs:
            print(f"  WARN: reference series {k!r} has no scored checkpoint "
                  f"under {metrics.CKPT_ROOT} — dropped from the figure and "
                  f"the shared panel. The TSFM arms are unaffected; this line "
                  f"comes back when the arm is re-scored on current code.")

    keys = TSFM_KEYS + refs \
        + (["randinit_vit"] if args.absolute else [])
    data, cov = load_ic_metrics(keys, absolute=args.absolute, fair_months=fair)
    for sk, (n_runs, n_months) in cov.items():
        print(f"  [ic] {sk}: {n_runs} runs over {n_months} eval months")

    # The panel the TABLE reports over: the months every drawn arm carries.
    # The figure gets this restriction from load_ic_metrics (json_ic series
    # are cut to the checkpoint series' months, and ``fair_months`` cuts the
    # checkpoint series to each other); the table has to intersect it itself.
    per_arm = []
    for key in TSFM_KEYS + refs:
        if SERIES_DEFS[key].get("json_ic"):
            per_arm.append(set(load_json_ic(key, f"return_{TABLE_H}")))
        else:
            per_arm.append({r.eval_month
                            for r in metrics._select_runs(SERIES_DEFS[key])})
    shared = set.intersection(*per_arm) if per_arm else set()
    summary_table(shared if fair else None, refs)

    drawn = [k for k in keys if data.get(k)]
    if not any(k in drawn for k in TSFM_KEYS):
        raise SystemExit(
            "no IC data for any TSFM arm — rebuild tsfm_ic.json with "
            "uv run plots/metrics/build_tsfm_ic.py")
    missing = [k for k in TSFM_KEYS if k not in drawn]
    if missing:
        print(f"  WARN: no data for {', '.join(missing)} — the artifact is "
              f"short a family; rerun build_tsfm_ic.py")

    # SIZED OFF WHAT IS ACTUALLY KEYED, because that count now moves: a
    # dropped reference takes a legend slot with it. Four or fewer entries go
    # in ONE row -- matplotlib fills a legend column-major, so four entries in
    # three columns strands the fourth alone in a second row beneath the
    # first, which is how this figure read the day LeJEPA dropped out. Beyond
    # four, split into two balanced rows rather than widening: ssl_baselines_ic
    # measured that a legend wider than the axes makes bbox_inches="tight"
    # stretch the SAVED figure, shrinking every panel once it is placed at
    # text width (a sixth column cost 9%).
    n_keyed = len(legend_keys(drawn))
    ncol = n_keyed if n_keyed <= 4 else -(-n_keyed // 2)

    plot_all_metrics(
        data, Path(args.outdir), baseline_subtracted=not args.absolute,
        series_keys=drawn, out_name="all_metrics_tsfm_ic",
        legend_ncol=ncol,
        bottom_reserve=legend_bottom_reserve(n_keyed, ncol),
        sharey=False,
    )


if __name__ == "__main__":
    main()
