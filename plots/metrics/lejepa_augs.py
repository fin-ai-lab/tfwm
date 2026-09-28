"""The LeJEPA augmentation figure: five positive pairings, one rank-IC panel set.

    all_metrics_augs_ic_minus_random   ΔIC vs a random-init encoder (default)
    all_metrics_augs_ic                absolute rank IC, with the untrained
                                       floor drawn as its own line (--absolute)

WHAT VARIES ACROSS THE FIVE ARMS is what counts as a positive pair, and
nothing else — one recipe (blr 6e-5, batch 256, 200 epochs, wd 5e-2, seed 42,
pool=cls, information token on, forward-VWAP targets):

    Same-stock crops     two crops of ONE ticker-day and nothing else
    Crops + time warp    the same, plus one view locally time-rescaled
    Crops + noise        the same, plus iid noise on every channel of one view
    Cross-stock K=2      two DIFFERENT tickers, same wall-clock window
    Cross-stock same-ind the same, partner drawn from the focal's FF49 industry

Same-stock crops IS THE CONTROL for the two same-stock augmentations: warp and
noise compose with the crop, they do not replace it, so "does warping help" is
only answerable against the arm that crops and stops.

A SIXTH LINE, IN GRAY, IS NOT A PAIRING: the on-task cross-entropy supervised
specialist, per panel, drawn as the reference the family is read against. It
is left out of the paired-margin table for that reason, and takes no part in
choosing the shared panel — it is simply cut to the months the arms share.

LAMBDA IS PART OF THE ARM. Each series is pinned to its own winner on holdout
set 2 (rrc 0.05, time_warp 0.001, gaussian_noise 0.3, k2 0.2, k2ind 0.1) —
never chosen on the months this figure reports. Two of those picks carry
caveats worth repeating in any caption: rrc's 0.05 is a mid-grid choice on a
flat, non-monotone curve rather than an argmax, and time_warp's 0.001 sits on
the GRID EDGE, still descending, so it is a bound rather than a located
optimum. The pins themselves live in ``metrics.SERIES_DEFS`` (``pair_*``).

MONTHS. The reported panel is the 32 sweep months, and the two cross-stock
arms are still filling in. Lines are therefore drawn over the months every arm
shares, printed on every run: averaging one arm over 32 months and another
over 18 makes part of the gap a difference in which months were sampled, and
months differ by more here than most of the effects being compared. Pass
``--own-months`` to draw each arm over everything it has, for a coverage
check — not for a comparison. When the sweep lands, the shared panel IS the
32 and the two agree.

Reads ``xs_ic.json`` beside each checkpoint (no producer of its own; the
in-job scorer writes all 18 (type, horizon) probes), and the random-init floor
from ``randinit-fwd3-*`` via ``style.load_randinit_ic``.

Run:
    uv run plots/metrics/lejepa_augs.py
    uv run plots/metrics/lejepa_augs.py --absolute
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
    PAIRING_N_MONTHS,
    SERIES_DEFS,
    TARGET_TYPES,
    WM_PANEL_TITLES,
    _select_runs,
    legend_keys,
    load_ic_metrics,
    load_json_ic,
    plot_all_metrics,
)

# Control first, then what composes with it, then the cross-stock pair — the
# legend reads as the argument the figure makes.
AUG_KEYS = ["pair_rrc", "pair_warp", "pair_noise", "pair_k2", "pair_k2ind"]
CONTROL = "pair_rrc"
# Drawn last, in gray: the on-task supervised specialist, in both readouts —
# the ridge probe as a line, and its own trained head as a star at h=900, the
# one horizon a head is trained at. Not a pairing — it is the supervised
# reference the whole family is read against, and it is excluded from the
# paired margin below for the same reason. Both are json_ic series, so they
# take no part in choosing the shared panel; load_ic_metrics cuts them to the
# months the five arms share.
REFERENCE_KEYS = ["supervised_specialist", "supervised_head"]
# The horizon the summary table reports. 15 min is the paper's headline
# horizon (it is the one the supervised heads are trained at), and the table
# is a reading aid for the figure, which draws all six.
TABLE_H = 900


def coverage_report() -> dict[str, set[str]]:
    """Print, and return, each arm's eval months. Loud about short arms."""
    months = {k: {r.eval_month for r in _select_runs(SERIES_DEFS[k])}
              for k in AUG_KEYS}
    print(f"Coverage (reported panel is {PAIRING_N_MONTHS} eval months):")
    for k in AUG_KEYS:
        n = len(months[k])
        flag = "" if n >= PAIRING_N_MONTHS else "   <-- STILL FILLING IN"
        print(f"  {SERIES_DEFS[k]['label']:24s} {n:2d}/{PAIRING_N_MONTHS}{flag}")
    shared = set.intersection(*months.values())
    print(f"  shared panel: {len(shared)} months\n")
    return months


def summary_table(fair: bool) -> None:
    """ΔIC at ``TABLE_H``, and each arm's paired margin over the control.

    Paired over months, which is the unit of independence here: the same month
    scored twice differs only by the arm, so the difference is far tighter
    than either level. n is the shared panel, so read the ordering and treat
    the t as indicative — twelve-ish months is the whole independence budget.
    """
    runs = {k: {r.eval_month: r for r in _select_runs(SERIES_DEFS[k])}
            for k in AUG_KEYS}
    panel = sorted(set.intersection(*(set(v) for v in runs.values()))) if fair \
        else None
    print(f"Rank IC at h={TABLE_H // 60} min, minus the random-init floor"
          f"{'' if fair else ' (each arm over its OWN months)'}:")
    head = "".join(f"{WM_PANEL_TITLES[t]:>22s}" for t in TARGET_TYPES)
    print(f"  {'arm':24s}{head}{'n':>5s}")
    for k in AUG_KEYS:
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
        print(f"  {SERIES_DEFS[k]['label']:24s}"
              + "".join(cells) + f"{max(ns):>5d}")

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
        print(f"  {SERIES_DEFS[ref]['label']:24s}"
              + "".join(cells) + f"{(max(ns) if ns else 0):>5d}")

    if panel is None:
        return
    print(f"\nPaired margin over {SERIES_DEFS[CONTROL]['label']!r} "
          f"(same {len(panel)} months, h={TABLE_H // 60} min):")
    print(f"  {'arm':24s}{head}")
    for k in AUG_KEYS:
        if k == CONTROL:
            continue
        cells = []
        for t in TARGET_TYPES:
            task = f"{t}_{TABLE_H}"
            d = [runs[k][m].probe(task) - runs[CONTROL][m].probe(task)
                 for m in panel]
            mu = st.fmean(d)
            se = st.stdev(d) / len(d) ** 0.5 if len(d) > 1 else float("nan")
            cells.append(f"{mu:>+13.4f} t={mu / se:>+5.1f}" if se == se
                         else f"{mu:>+22.4f}")
        print(f"  {SERIES_DEFS[k]['label']:24s}" + "".join(cells))
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default=str(OUT_DIR))
    p.add_argument("--absolute", action="store_true",
                   help="Plot absolute rank IC and draw the random-init "
                        "encoder as its own line instead of subtracting it.")
    p.add_argument("--own-months", action="store_true",
                   help="Draw each arm over every month it has, rather than "
                        "over the months all five share. A coverage check, "
                        "not a comparison: the lines then average different "
                        "panels.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fair = not args.own_months
    coverage_report()
    summary_table(fair)

    keys = AUG_KEYS + REFERENCE_KEYS \
        + (["randinit_vit"] if args.absolute else [])
    data, cov = load_ic_metrics(keys, absolute=args.absolute, fair_months=fair)
    for sk, (n_runs, n_months) in cov.items():
        print(f"  [ic] {sk}: {n_runs} runs over {n_months} eval months")
    drawn = [k for k in keys if data.get(k)]
    if not drawn:
        raise SystemExit("no IC data for any pairing arm — check the pins in "
                         "metrics.SERIES_DEFS against the checkpoint tree")

    plot_all_metrics(
        data, Path(args.outdir), baseline_subtracted=not args.absolute,
        series_keys=drawn, out_name="all_metrics_augs_ic",
        # Half the KEYED entries per column -> exactly two legend rows (3+3
        # here, 4+3 under --absolute). The head star is drawn but not keyed
        # (``no_legend``), so it must not be counted. One wide row is wider
        # than the three panels at these label lengths, and
        # bbox_inches="tight" would stretch the saved figure to the legend
        # rather than to the axes; three rows would overrun the bottom
        # reserve onto the shared x-label.
        legend_ncol=-(-len(legend_keys(drawn)) // 2), sharey=False,
    )


if __name__ == "__main__":
    main()
