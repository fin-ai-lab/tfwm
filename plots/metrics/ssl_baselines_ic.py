"""The SSL-baseline figure: nine re-optimized baselines, one rank-IC panel set.

    all_metrics_ssl_ic_minus_random   ΔIC vs a random-init encoder (default)
    all_metrics_ssl_ic                absolute rank IC, with the untrained
                                      floor drawn as its own line (--absolute)

THE NINE ARMS are the SSL baselines re-optimized under rank IC (the ssl_ic
campaign; every prior selection was made on the retired ΔAUC metric). Each is
its own objective on the SAME encoder, data and budget, so what varies across
the arms is the pretext task and nothing else:

    BYOL / DINO      two augmented views of one series, EMA teacher
    I-JEPA / MAE     masked reconstruction, in latent / input space
    CPC / TF-C       contrastive, over time / over the time-frequency pair
    CoST / TS2Vec    contrastive with a seasonal-trend or hierarchical view
    TimeMAE          masked reconstruction with an alignment term

SELECTION WAS ON HOLDOUT SET 2 (5 months), NEVER on the 32 reported here, and
the two panels disagree hard enough to be worth a caption: MAE was SIXTH of
nine on holdout-2 and leads this figure, while TF-C took the largest round-2
gain there and is the only arm that lands BELOW the untrained floor here.
Read the holdout-2 ordering as a tuning artifact, not a preview.

CONFIGS: five arms (byol, cpc, dino, ijepa, tfc) ship round-2 winners; four
(cost, mae, timemae, ts2vec) ship ROUND-1 winners because round 2 was stopped
early once its measured gain had collapsed to ~+0.0005. Each choice, and the
number behind it, is recorded per method in sweeps/ssl_ic/final32.sh.

MONTHS. The reported panel is the 32 sweep months. Lines are drawn over the
months every arm shares, printed on every run, for the reason given in
lejepa_augs.py: averaging one arm over 32 months and another over 18 makes
part of the gap a difference in which months were sampled. ``--own-months``
draws each arm over everything it has — a coverage check, not a comparison.

THE SUPERVISED SPECIALIST is drawn alongside them in gray: the on-task
cross-entropy model, per panel, which is what the nine are worth reading
against — it is trained on the very target the probe scores. Not an arm, and
it takes no part in the shared panel; it is cut to the months the arms share.

COLOR. plots/style.SERIES_STYLES gives these nine only five distinct hues
(cpc/ts2vec, ijepa/cost, dino/timemae, byol/tfc all pair up), which is fine
where they never share an axis and wrong here. This figure therefore carries
its OWN assignment: the eight tab10 hues that are not reserved paper-wide
(tab:blue is LeJEPA, tab:gray the supervised specialist, which this figure
now draws), plus one dashed repeat. The repeat is deliberately MAE/TF-C — the
top and bottom arm — so the one reused hue is the pair a reader can never
confuse.

Reads ``xs_ic.json`` beside each checkpoint (no producer of its own; the
in-job scorer writes all 18 (type, horizon) probes), and the random-init floor
from ``randinit-fwd3-*`` via ``style.load_randinit_ic``.

Run:
    uv run plots/metrics/ssl_baselines_ic.py
    uv run plots/metrics/ssl_baselines_ic.py --absolute
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR))

import metrics  # noqa: E402
from metrics import (  # noqa: E402
    SERIES_DEFS,
    SSL_IC_N_MONTHS,
    TARGET_TYPES,
    WM_PANEL_TITLES,
    _select_runs,
    legend_bottom_reserve,
    legend_keys,
    load_ic_metrics,
    load_json_ic,
    plot_all_metrics,
)

# Ordered by the reported-panel result so the legend reads top-to-bottom.
SSL_KEYS = ["ssl_mae", "ssl_byol", "ssl_dino", "ssl_cost", "ssl_ijepa",
            "ssl_timemae", "ssl_cpc", "ssl_ts2vec", "ssl_tfc"]
# Drawn last, in gray: the on-task supervised specialist, which is not an SSL
# arm at all but the supervised reference every breadth figure carries. Two
# readouts of the one model — the ridge probe as a line, and its own trained
# head as a star at h=900, the one horizon a head is trained at. Both are
# json_ic series (see SERIES_DEFS), so they take no part in the fair panel —
# load_ic_metrics cuts them to the months the nine arms share.
REFERENCE_KEYS = ["supervised_specialist", "supervised_head"]

# See COLOR above. Eight non-reserved tab10 hues; MAE/TF-C share brown, split
# by linestyle, because they are the extreme arms and cannot be mistaken.
SSL_COLORS: dict[str, dict] = {
    "ssl_mae":     {"color": "tab:brown"},
    "ssl_byol":    {"color": "tab:red"},
    "ssl_dino":    {"color": "tab:purple"},
    "ssl_cost":    {"color": "tab:green"},
    "ssl_ijepa":   {"color": "tab:olive"},
    "ssl_timemae": {"color": "tab:cyan"},
    "ssl_cpc":     {"color": "tab:orange"},
    "ssl_ts2vec":  {"color": "tab:pink"},
    "ssl_tfc":     {"color": "tab:brown", "linestyle": "--"},
}

TABLE_H = 900   # the paper's headline horizon, as in lejepa_augs.py


def apply_palette() -> None:
    """Override the shared hues for THIS figure only (see COLOR above)."""
    for k, style in SSL_COLORS.items():
        SERIES_DEFS[k].update(style)


def coverage_report() -> dict[str, set[str]]:
    months = {k: {r.eval_month for r in _select_runs(SERIES_DEFS[k])}
              for k in SSL_KEYS}
    print(f"Coverage (reported panel is {SSL_IC_N_MONTHS} eval months):")
    for k in SSL_KEYS:
        n = len(months[k])
        flag = "" if n >= SSL_IC_N_MONTHS else "   <-- STILL FILLING IN"
        print(f"  {SERIES_DEFS[k]['label']:16s} {n:2d}/{SSL_IC_N_MONTHS}{flag}")
    live = [m for m in months.values() if m]
    shared = set.intersection(*live) if live else set()
    print(f"  shared panel: {len(shared)} months\n")
    return months


def summary_table(fair: bool) -> None:
    """ΔIC at ``TABLE_H`` per arm, over the shared panel (or its own months)."""
    runs = {k: {r.eval_month: r for r in _select_runs(SERIES_DEFS[k])}
            for k in SSL_KEYS}
    live = [set(v) for v in runs.values() if v]
    panel = sorted(set.intersection(*live)) if (fair and live) else None
    print(f"Rank IC at h={TABLE_H // 60} min, minus the random-init floor"
          f"{'' if fair else ' (each arm over its OWN months)'}:")
    head = "".join(f"{WM_PANEL_TITLES[t]:>22s}" for t in TARGET_TYPES)
    print(f"  {'arm':16s}{head}{'n':>5s}")
    for k in SSL_KEYS:
        cells, ns = [], []
        for t in TARGET_TYPES:
            task = f"{t}_{TABLE_H}"
            floor = metrics.load_randinit_ic(task)
            use = panel if panel is not None else sorted(runs[k])
            d = [runs[k][m].probe(task) - floor[m] for m in use
                 if m in runs[k] and runs[k][m].probe(task) is not None
                 and m in floor]
            ns.append(len(d))
            cells.append(f"{st.fmean(d):>+22.4f}" if d else f"{'--':>22s}")
        print(f"  {SERIES_DEFS[k]['label']:16s}"
              + "".join(cells) + f"{(max(ns) if ns else 0):>5d}")

    # The gray reference rows, over the same panel: json_ic series, so their
    # months come from the artifact rather than from _select_runs. Both
    # readouts of the one supervised specialist -- ridge probe, then its own
    # trained head, which exists only at h=900 and so is a row here and a
    # star on the figure.
    for ref in REFERENCE_KEYS:
        cells, ns = [], []
        for t in TARGET_TYPES:
            task = f"{t}_{TABLE_H}"
            floor = metrics.load_randinit_ic(task)
            ic = load_json_ic(ref, task)
            use = panel if panel is not None else sorted(ic)
            d = [ic[m] - floor[m] for m in use if m in ic and m in floor]
            ns.append(len(d))
            cells.append(f"{st.fmean(d):>+22.4f}" if d else f"{'--':>22s}")
        print(f"  {SERIES_DEFS[ref]['label']:16s}"
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
                        "over the months all nine share. A coverage check, "
                        "not a comparison: the lines then average different "
                        "panels.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_palette()
    fair = not args.own_months
    coverage_report()
    summary_table(fair)

    keys = SSL_KEYS + REFERENCE_KEYS \
        + (["randinit_vit"] if args.absolute else [])
    data, cov = load_ic_metrics(keys, absolute=args.absolute, fair_months=fair)
    for sk, (n_runs, n_months) in cov.items():
        print(f"  [ic] {sk}: {n_runs} runs over {n_months} eval months")
    drawn = [k for k in keys if data.get(k)]
    if not drawn:
        raise SystemExit("no IC data for any SSL arm — check SSL_IC_GLOB in "
                         "metrics.SERIES_DEFS against the checkpoint tree")

    plot_all_metrics(
        data, Path(args.outdir), baseline_subtracted=not args.absolute,
        series_keys=drawn, out_name="all_metrics_ssl_ic",
        # Ten KEYED entries in FIVE columns -> two rows (the head star is
        # drawn but not keyed, so it is not counted). Five is the widest
        # legend that still fits inside the panels: a sixth column is wider
        # than the axes, and bbox_inches="tight" then stretches the saved
        # figure by 9%. So the column count is fixed and any further row is
        # paid for in the reserve.
        legend_ncol=5,
        bottom_reserve=legend_bottom_reserve(len(legend_keys(drawn)), 5),
        sharey=False,
    )


if __name__ == "__main__":
    main()
