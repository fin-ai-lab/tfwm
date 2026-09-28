"""The two deliverables of the IC-vs-Sharpe section.

ONE: the choice space -- a column per place the pipeline makes a decision
between forward return and order, a row per option at that decision.

TWO: the two correlations the section is about, taken across METHODS.

    corr(avg Sharpe, IC)   average each method's Sharpe over every config,
                           then correlate those averages with IC.
    avg corr(Sharpe, IC)   correlate across methods inside ONE config, then
                           average that correlation over configs.

They answer different questions and the gap between them is the finding: the
first asks whether a better forecast wins ON AVERAGE over implementations, the
second asks whether it wins in the single implementation you actually shipped.

IC IS A PROPERTY OF THE FORECAST, NOT THE BOOK. Only ``universe`` changes it,
by changing which rows are scored (verified: within a month an arm has exactly
one IC per screen, and 4080/3 configs share each). The headline IC is therefore
the unscreened one, so the x-axis is forecast quality and every implementation
choice lives on the y-axis.

RAGGED MONTHS. The random-init floor was never embedded for 12 of the 31 eval
months, so averaging each method over its OWN months would compare the floor's
19 months against everything else's 31 -- the months differ more than the
methods do. The reported numbers use the month intersection where every method
is present; the all-months variant is printed beside them so the choice is
visible rather than buried.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

# ``cost_model`` and ``annualization`` are gone. Cost is now EFQ, one measured
# continuous quantity rather than five named models; annualization is pinned
# to ``daily`` because it is pure accounting -- the same trades folded into
# different periods -- and a spread inflated by it is not a spread any model
# has an opinion about.
AXES = ("universe", "selection", "weighting", "risk_model", "efq", "rebalance")
AXIS_LABELS = {
    "universe": "Universe screen", "selection": "Selection",
    "weighting": "Weighting", "risk_model": "Risk model",
    "efq": "Execution quality (EFQ)", "rebalance": "Rebalance",
}

# THE PAPER TABLE IS TYPESET, NOT DUMPED. The sweep's option names are Python
# identifiers; printing them raw gives a table full of escaped underscores that
# has to be hand-fixed after every regeneration, and a hand-fixed generated file
# loses its edits the next time this script runs. The display name and the
# reading order therefore live here, beside the generator, so `make` reproduces
# the polished table rather than the raw one.
AXIS_HEADERS = {   # shorter than AXIS_LABELS: every column must fit one line
    "universe": "Universe", "selection": "Selection", "weighting": "Weighting",
    "risk_model": "Risk model", "efq": "EFQ", "rebalance": "Rebalance",
}
# Ordered as a reader would want them -- widest screen first, cheapest cost
# first -- not alphabetically, which interleaves "cross+0.5bps" with "mid" and
# puts the 25% screen above the 50% one.
OPTION_ORDER = {
    "universe": ("all", "tight_50pct", "tight_25pct"),
    "selection": ("all", "quintile", "decile", "decile_long_only"),
    "weighting": ("equal", "inverse_vol", "min_var", "erc", "rank",
                  "mean_variance"),
    "risk_model": ("diagonal", "shrunk_sample", "ledoit_wolf", "embedding"),
    "efq": (0.0, 0.25, 0.50, 0.70, 0.88, 1.00),   # filtered by REPORTED_EFQ
    "rebalance": ("every_decision", "daily"),
    "annualization": ("decisions", "daily"),
}
OPTION_LABELS = {
    "all": "All", "tight_50pct": "Tightest 50\\%", "tight_25pct": "Tightest 25\\%",
    "quintile": "Quintile", "decile": "Decile", "decile_long_only": "Long-only",
    "equal": "Equal", "inverse_vol": "Inverse vol.", "min_var": "Min.\\ variance",
    "erc": "ERC", "rank": "Rank", "mean_variance": "Mean--variance",
    "diagonal": "Diagonal", "shrunk_sample": "Shrunk sample",
    "ledoit_wolf": "Ledoit--Wolf", "embedding": "Embedding",
    "every_decision": "Every decision", "daily": "Daily",
    # EFQ is a number, and reads as one: a percent, as Levy (2022) quotes it.
    **{efq: f"${round(efq * 100)}\\%$"
       for efq in (0.0, 0.25, 0.50, 0.70, 0.88, 1.00)},
}
# THE CAPTION LIVES HERE, not in a hand-edited wrapper, because the generator
# emits the whole float. Every citation is checked against what the code
# actually does, which is narrower than the option names suggest:
#   * `shrunk_sample` shrinks the sample covariance toward its own DIAGONAL at
#     a fixed intensity. It is not a single-index model, so Sharpe (1963) does
#     not apply and is deliberately absent.
#   * `ledoit_wolf` is sklearn's LedoitWolf: shrinkage toward a SCALED IDENTITY
#     at a data-chosen intensity, i.e. the 2004 JMVA estimator, not the 2003
#     single-index-target one.
#   * `rank` has no standard antecedent and is left uncited rather than
#     attached to a paper that describes something else.
#: ``{axes}`` and ``{configs}`` are filled from the sweep that was actually
#: run, so the caption cannot drift from the table above it -- which it did,
#: silently, when the space changed from seven axes to six.
CAPTION = r"""{axes} of the many choices that lie between a forward return
    prediction and a Sharpe ratio, and the options we sweep at each. Their
    product, less combinations that are degenerate by construction, gives
    the ${configs}$ configurations we use to demonstrate how wide a spread
    these choices induce. Execution quality is EFQ, the effective half-spread
    actually paid over the half-spread quoted when the order was sent
    \citep{{levy2022}}, at $0\%$ a midpoint fill and at $100\%$ the whole
    quoted spread. Annualization is not swept: every Sharpe here is one
    non-overlapping daily observation annualized at $252$, because folding
    the same trades into different periods changes the number without
    changing anything a model has an opinion about."""
NUMBER_WORDS = {4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight"}
TABLE_LABEL = "tab:ic-sharpe-choices"

# CITATIONS GO IN THE CELL, beside the method they describe. Only the two
# columns with a standard literature are cited: `rank`, `embedding` and
# `shrunk_sample` have no specific antecedent (the last shrinks toward its own
# diagonal at a fixed intensity, which is not any published estimator), and
# attaching a paper that describes something else is worse than a gap.
#
# WIDTH DEPENDS ON THE BIB STYLE. These render as "[7]" under a numeric style
# and as "(Ledoit and Wolf, 2004)" under author-year, which will not fit seven
# columns on one line. Use a numeric style, or move these to the caption.
OPTION_CITATIONS = {
    "equal": ("demiguel2009naive",),
    "inverse_vol": ("maillard2010erc",),
    "erc": ("maillard2010erc",),
    "min_var": ("markowitz1952portfolio",),
    "mean_variance": ("markowitz1952portfolio", "michaud1989enigma"),
    "ledoit_wolf": ("ledoit2004wellconditioned", "ledoit2004honey"),
}

#: The execution qualities REPORTED. The sweep also ran $88\%$ (Levy's NYSE
#: figure); it is dropped here because it sits close enough to $100\%$ that
#: the row lengthens the table without changing anything a reader takes from
#: it. Filtering at load() rather than per table keeps the choice-space table,
#: the configuration count and the harness table from disagreeing about which
#: grid they describe.
REPORTED_EFQ = (0.0, 0.25, 0.50, 0.70, 1.00)

SHARPES = ("sharpe_net", "sharpe_mid")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--results-dir", default="/data/lab/ic_sharpe_configs/results")
    p.add_argument("--tex-out", default="plots/ic_vs_sharpe/ic_sharpe_choices.tex")
    p.add_argument("--json-out", default="plots/ic_vs_sharpe/ic_sharpe_correlations.json")
    p.add_argument("--efq", type=float, default=0.70,
                   help="execution quality the harness-spread table is held "
                        "at; 0.70 sits between Levy (2022)'s direct-access "
                        "55%% and NYSE 88%%")
    p.add_argument("--tables-only", action="store_true",
                   help="write the choice table from a single result file and "
                        "stop; the axis space does not depend on the sweep, so "
                        "formatting it need not re-read every result")
    return p.parse_args()


def load(results_dir: str):
    """(arm, month) -> {config key -> record}, plus each arm's unscreened IC."""
    sharpe: dict[tuple[str, str], dict[tuple, dict]] = {}
    ic: dict[tuple[str, str], float] = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        result = json.loads(path.read_text())
        key = (result["arm"], result["eval_month"])
        if result["n_configs_scored"] != result["n_configs_requested"]:
            raise SystemExit(
                f"{path.name}: {result['n_configs_scored']} of "
                f"{result['n_configs_requested']} configs scored -- refusing to "
                f"average over a ragged config set")
        sharpe[key] = {tuple(c[a] for a in AXES): c for c in result["configs"]
                       if c["efq"] in REPORTED_EFQ}
        unscreened = [c["information_coefficient"] for c in result["configs"]
                      if c["universe"] == "all"]
        ic[key] = float(np.mean(unscreened))
    return sharpe, ic


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok] - np.mean(a[ok]), b[ok] - np.mean(b[ok])
    denominator = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else np.nan


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return pearson(rankdata(a[ok]), rankdata(b[ok])) if ok.sum() > 2 else np.nan


def correlations(sharpe, ic, arms, months, field, *, config_keys=None):
    """How well the IC predicts a Sharpe, averaged over configs versus inside one.

    Two numbers, and the gap between them is the argument. ``corr_of_average``
    averages each method's Sharpe over every configuration and then correlates
    those averages with the IC across methods -- what the IC predicts once the
    harness is averaged away. ``average_corr`` correlates across methods inside
    ONE configuration and then averages that correlation over configurations --
    what the IC predicts to somebody who ran a single backtest.

    ``configs_ranking_backwards`` counts the configurations in which the
    correlation is NEGATIVE: harnesses under which a better forecast scores
    worse. It is the blunt version of the same point.
    """
    if config_keys is None:
        config_keys = sorted(next(iter(sharpe.values())).keys())
    ic_by_arm = np.array([
        float(np.mean([ic[(arm, month)] for month in months])) for arm in arms])
    # (method, configuration), each cell averaged over the shared months, so a
    # method is never credited for a month another method did not run.
    grid = np.array([[float(np.mean([sharpe[(arm, month)][key][field]
                                     for month in months]))
                      for key in config_keys] for arm in arms])
    per_config = np.array([pearson(ic_by_arm, grid[:, j])
                           for j in range(grid.shape[1])])
    per_config_rank = np.array([spearman(ic_by_arm, grid[:, j])
                                for j in range(grid.shape[1])])
    return {
        "n_methods": len(arms), "n_months": len(months),
        "n_configs": len(config_keys),
        "corr_of_average_pearson": pearson(ic_by_arm, grid.mean(axis=1)),
        "corr_of_average_spearman": spearman(ic_by_arm, grid.mean(axis=1)),
        "average_corr_pearson": float(np.nanmean(per_config)),
        "average_corr_spearman": float(np.nanmean(per_config_rank)),
        "per_config_pearson_sd": float(np.nanstd(per_config, ddof=1)),
        "per_config_pearson_min": float(np.nanmin(per_config)),
        "per_config_pearson_max": float(np.nanmax(per_config)),
        "configs_ranking_backwards": int(np.sum(per_config < 0)),
    }


EFQ_INDEX = AXES.index("efq")


def keys_at_efq(sharpe, efq):
    """The configuration keys whose execution quality is ``efq``."""
    return sorted(k for k in next(iter(sharpe.values())) if k[EFQ_INDEX] == efq)


def efq_correlation_rows(sharpe, ic, arms, months, field="sharpe_net"):
    """The two correlations, one row per execution quality.

    WHY THIS IS A TABLE AND NOT A NUMBER. The headline pair was computed on
    the FRICTIONLESS Sharpe, and it does not survive costs unchanged: the IC
    orders methods well when nobody pays a spread, and progressively less well
    as they do, because the spread bill depends on what a book trades rather
    than on how well it forecasts. Reporting the pair at one cost hides that;
    reporting it down an axis of measured execution qualities states it.

    Each row is 408 configurations at that EFQ, so the rows are comparable --
    the same harness throughout, only the execution quality differing.
    """
    rows = []
    for efq in OPTION_ORDER["efq"]:
        keys = keys_at_efq(sharpe, efq)
        if not keys:
            continue
        stats = correlations(sharpe, ic, arms, months, field, config_keys=keys)
        rows.append((efq, stats))
    return rows


#: The order the spread columns appear in, widest effect first. Fixed rather
#: than sorted per row: the merged table compares the SAME choice down an EFQ
#: column, which a per-row sort would scramble.
SPREAD_AXES = ("rebalance", "selection", "universe", "weighting", "risk_model")


def spread_at_efq(sharpe, arms, months, efq, field="sharpe_net"):
    """How far each remaining choice moves the Sharpe, at ONE execution quality.

    Holding EFQ fixed is what makes this an argument. With the cost axis free a
    reader can dismiss the spread as "they included a free backtest and an
    expensive one"; with every configuration paying the same measured
    execution quality, what is left is the harness alone.

    Also returns the comparison the spread is only meaningful against: the
    variation a method sees ACROSS harnesses beside the variation between
    methods at a FIXED harness. If the first is not at least comparable to the
    second, the harness is a nuisance rather than a confound.
    """
    keys = keys_at_efq(sharpe, efq)
    grid = np.array([[float(np.mean([sharpe[(a, m)][k][field] for m in months]))
                      for k in keys] for a in arms])

    spreads = {}
    for axis in SPREAD_AXES:
        index = AXES.index(axis)
        marginals = collections.defaultdict(list)
        for column, key in enumerate(keys):
            marginals[key[index]].extend(grid[:, column].tolist())
        means = {opt: float(np.mean(v)) for opt, v in marginals.items()}
        spreads[axis] = {"options": len(means),
                         "range": max(means.values()) - min(means.values())}

    within = float(np.mean(np.std(grid, axis=1, ddof=1)))   # across harnesses
    between = float(np.mean(np.std(grid, axis=0, ddof=1)))  # across methods
    return spreads, {
        "efq": efq, "n_configs": len(keys), "n_methods": len(arms),
        "within_method_sd": within, "between_method_sd": between,
        "ratio": within / between if between > 0 else float("inf"),
        "config_min": float(grid.min()), "config_max": float(grid.max()),
        "config_median": float(np.median(grid)),
    }


def write_harness_tex(rows, headline, path):
    r"""Deliverables 2 and 3 as ONE table, one row per execution quality.

    Both halves are indexed by EFQ, so they are one table and not two: the
    correlations say how well the IC orders methods at that execution quality,
    and the spreads say how much the harness moves the score it is ordering.
    Reading across a row is the whole argument -- as execution worsens the IC
    predicts less and the harness matters more.

    The two halves are separated by a gutter and their own header rules rather
    than a vertical line: they are different quantities in different units,
    and booktabs tables say that with space.
    """
    heads = " & ".join(rf"\textbf{{{AXIS_LABELS[a].split(' (')[0]}}}"
                       for a in SPREAD_AXES)
    lines = [
        r"% Generated by plots/ic_vs_sharpe/ic_sharpe_deliverables.py -- do not edit.",
        r"% Edit HARNESS_CAPTION / AXIS_LABELS in that script instead;",
        r"% this file is overwritten wholesale on every run.",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{l rr @{\hspace{14pt}} rrrrr}",
        r"  \toprule",
        r"  & \multicolumn{2}{c}{\textbf{IC vs.\ Sharpe}}"
        r" & \multicolumn{5}{c}{\textbf{Sharpe range across each choice}} \\",
        r"  \cmidrule(lr){2-3} \cmidrule(l){4-8}",
        r"  \textbf{EFQ} & $r(\mathrm{IC},\bar{S})$ & $\bar{r}(\mathrm{IC},S)$ & "
        + heads + r" \\",
        r"  \midrule",
    ]
    for efq, stats, spreads in rows:
        cells = " & ".join(f"{spreads[a]['range']:.1f}" for a in SPREAD_AXES)
        lines.append(
            f"  {OPTION_LABELS[efq]} & {stats['corr_of_average_pearson']:+.2f} & "
            f"{stats['average_corr_pearson']:+.2f} & " + cells + r" \\")

    caption = HARNESS_CAPTION
    for token, value in (
        ("@EFQ@", f"{headline['efq']:.0%}".replace("%", r"\%")),
        ("@METHODS@", str(headline["n_methods"])),
        ("@CONFIGS@", str(headline["n_configs"])),
        ("@WITHIN@", f"{headline['within_method_sd']:.1f}"),
        ("@BETWEEN@", f"{headline['between_method_sd']:.1f}"),
        ("@RATIO@", f"{headline['ratio']:.0f}"),
    ):
        caption = caption.replace(token, value)
    if "@" in caption:
        raise RuntimeError(f"unfilled placeholder in caption: {caption}")

    lines += [
        r"  \bottomrule",
        r"  \end{tabular}",
        r"  \caption{" + caption + "}",
        rf"  \label{{{HARNESS_LABEL}}}",
        r"\end{table}",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


# PLACEHOLDERS ARE @TOKENS@, NOT str.format braces. A LaTeX caption is mostly
# braces, so .format() would need every one of them doubled -- a silent trap
# the next time anybody edits this string.
HARNESS_CAPTION = r"""A Sharpe ratio is a property of the harness as much as of
    the forecast, and both halves of this table are read down the same axis of
    execution quality. EFQ is the effective half-spread actually paid over the
    half-spread quoted when the order was sent \citep{levy2022}, at $0\%$ a
    midpoint fill and at $100\%$ the whole quoted spread.
    \textbf{Left:} $r(\mathrm{IC},\bar{S})$ correlates each method's IC with
    its Sharpe averaged over every harness at that execution quality;
    $\bar{r}(\mathrm{IC},S)$ correlates them inside one harness and then
    averages that correlation over harnesses. The IC predicts how well a
    method does once implementation choices are averaged away, much less so
    inside any one implementation, and both weaken as execution worsens --- so
    the frictionless pair is the most favourable reading rather than the
    general one. \textbf{Right:} how far each remaining choice moves the
    annualized Sharpe, over @METHODS@ methods and the @CONFIGS@ harnesses at
    each execution quality. At $\mathrm{EFQ}=@EFQ@$ a single method's Sharpe
    varies across harnesses with a standard deviation of $@WITHIN@$, against
    $@BETWEEN@$ across methods at a \emph{fixed} harness: the choice of
    backtest moves the score $@RATIO@\times$ harder than the choice of model."""
HARNESS_LABEL = "tab:ic-sharpe-harness"


def choice_table(sample: dict[tuple, dict]) -> dict[str, list[str]]:
    """The options actually present on each axis, in reading order.

    Ordering comes from OPTION_ORDER rather than sorted(), and anything the
    sweep produced that is not listed there is appended and reported, so a new
    option shows up in the table instead of being silently dropped.
    """
    options: dict[str, list[str]] = {}
    for axis in AXES:
        seen = {record[axis] for record in sample.values()}
        known = [o for o in OPTION_ORDER[axis] if o in seen]
        extra = sorted(seen - set(OPTION_ORDER[axis]), key=str)
        if extra:
            print(f"  NOTE: unlisted {axis} option(s) {extra} -- appended "
                  f"unstyled; add them to OPTION_ORDER/OPTION_LABELS")
        options[axis] = known + extra
    return options


def _cell(option: str) -> str:
    """One table cell: the display label, then any citation for that option."""
    label = str(OPTION_LABELS.get(option, option)).replace("_", r"\_")
    keys = OPTION_CITATIONS.get(option)
    return f"{label}~\\citep{{{', '.join(keys)}}}" if keys else label


def write_tex(options, n_configs, path):
    r"""The complete float: table, tabular, caption and label in one file.

    \scriptsize and \tabcolsep are set inside the float, so they apply to the
    tabular without leaking into the surrounding document.
    """
    rows = max(len(v) for v in options.values())
    header = " & ".join(f"\\textbf{{{AXIS_HEADERS[a]}}}" for a in AXES)
    lines = [
        r"% Generated by plots/ic_vs_sharpe/ic_sharpe_deliverables.py -- do not edit.",
        r"% Edit OPTION_LABELS / AXIS_HEADERS / CAPTION in that script instead;",
        r"% this file is overwritten wholesale on every run.",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{6pt}",
        r"  \begin{tabular}{" + "l" * len(AXES) + "}",
        r"  \toprule",
        "  " + header + r" \\",
        r"  \midrule",
    ]
    for i in range(rows):
        cells = [_cell(options[a][i]) if i < len(options[a]) else ""
                 for a in AXES]
        lines.append("  " + " & ".join(cells) + r" \\")
    lines += [
        r"  \bottomrule",
        r"  \end{tabular}",
        r"  \caption{" + CAPTION.format(
            axes=NUMBER_WORDS.get(len(AXES), str(len(AXES))),
            configs=f"{n_configs:,}".replace(",", "{,}")) + "}",
        rf"  \label{{{TABLE_LABEL}}}",
        r"\end{table}",
        rf"% {n_configs} distinct configurations after removing degenerate pairs",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.tables_only:
        one = sorted(Path(args.results_dir).glob("*.json"))[0]
        result = json.loads(one.read_text())
        sample = {tuple(c[a] for a in AXES): c for c in result["configs"]}
        options = choice_table(sample)
        write_tex(options, len(sample), args.tex_out)
        print(f"choice space from {one.name} -> {args.tex_out}")
        for axis in AXES:
            print(f"  {AXIS_HEADERS[axis]:<14} {len(options[axis])}  "
                  f"{', '.join(str(OPTION_LABELS.get(o, o)) for o in options[axis])}")
        return
    sharpe, ic = load(args.results_dir)

    months_by_arm = collections.defaultdict(set)
    for arm, month in sharpe:
        months_by_arm[arm].add(month)
    arms = sorted(months_by_arm)
    shared = sorted(set.intersection(*months_by_arm.values()))
    every = sorted(set.union(*months_by_arm.values()))
    complete = [m for m in every
                if all(m in months_by_arm[a] for a in arms)]

    print(f"{len(arms)} methods, {len(every)} eval months, "
          f"{len(shared)} shared by every method")
    ragged = [a for a in arms if len(months_by_arm[a]) < len(every)]
    if ragged:
        print(f"  ragged: {', '.join(f'{a} ({len(months_by_arm[a])})' for a in ragged)}")

    options = choice_table(next(iter(sharpe.values())))
    n_configs = len(next(iter(sharpe.values())))
    write_tex(options, n_configs, args.tex_out)
    print(f"\n=== choice space -> {args.tex_out} ===")
    width = max(len(AXIS_LABELS[a]) for a in AXES)
    for axis in AXES:
        print(f"  {AXIS_LABELS[axis]:<{width}}  {len(options[axis])}  "
              f"{', '.join(str(o) for o in options[axis])}")
    print(f"  {'':<{width}}  -> {n_configs} distinct configurations")

    print(f"\n=== corr(Sharpe, IC) across {len(arms)} methods ===")
    # THE FLOOR IS THE RAGGED ONE, so the contrast that matters is "drop the
    # three random-init seeds and you get back the months they were missing" --
    # not "let every method average over whatever it happens to have".
    floorless = [a for a in arms if not a.startswith("randinit")]
    floorless_months = sorted(set.intersection(
        *(months_by_arm[a] for a in floorless)))

    out = {}
    variants = [("all 21 methods", arms, complete)]
    if len(floorless_months) > len(complete):
        variants.append((f"{len(floorless)} methods, no floor",
                         floorless, floorless_months))
    for label, subset, months in variants:
        for field in SHARPES:
            stats = correlations(sharpe, ic, subset, months, field)
            out[f"{label}|{field}"] = stats
            print(f"\n  {label}, {field}  "
                  f"({stats['n_methods']} methods x {stats['n_months']} months "
                  f"x {stats['n_configs']} configs)")
            print(f"    corr(avg Sharpe, IC)   "
                  f"{stats['corr_of_average_pearson']:+.3f} pearson   "
                  f"{stats['corr_of_average_spearman']:+.3f} spearman")
            print(f"    avg corr(Sharpe, IC)   "
                  f"{stats['average_corr_pearson']:+.3f} pearson   "
                  f"{stats['average_corr_spearman']:+.3f} spearman")
            print(f"    per-config spread      sd {stats['per_config_pearson_sd']:.3f}, "
                  f"range [{stats['per_config_pearson_min']:+.3f}, "
                  f"{stats['per_config_pearson_max']:+.3f}], "
                  f"{stats['configs_ranking_backwards']} of {stats['n_configs']} negative")

    efq_stats = dict(efq_correlation_rows(sharpe, ic, floorless,
                                          floorless_months))
    rows, summaries = [], {}
    for efq in OPTION_ORDER["efq"]:
        if efq not in efq_stats:
            continue
        spreads, summary = spread_at_efq(sharpe, floorless, floorless_months, efq)
        rows.append((efq, efq_stats[efq], spreads))
        summaries[efq] = summary
    headline = summaries[args.efq]

    harness_path = str(Path(args.tex_out).with_name("ic_sharpe_harness.tex"))
    write_harness_tex(rows, headline, harness_path)
    print(f"\n=== harness table -> {harness_path} ===")
    print("    EFQ  r(IC,avgS)  avg r(IC,S)  " +
          "  ".join(f"{a[:9]:>9}" for a in SPREAD_AXES))
    for efq, stats, spreads in rows:
        print(f"  {efq:>5.2f} {stats['corr_of_average_pearson']:>+11.2f} "
              f"{stats['average_corr_pearson']:>+12.2f}  " +
              "  ".join(f"{spreads[a]['range']:>9.2f}" for a in SPREAD_AXES))
    summary = headline
    print(f"\n  at EFQ {args.efq:.0%}: across {summary['n_configs']} harnesses "
          f"x {summary['n_methods']} "
          f"methods: Sharpe {summary['config_min']:+.2f} to "
          f"{summary['config_max']:+.2f}, median {summary['config_median']:+.2f}")
    print(f"  sd across harnesses within a method {summary['within_method_sd']:.3f} "
          f"vs across methods at a fixed harness {summary['between_method_sd']:.3f} "
          f"-> ratio {summary['ratio']:.2f}")

    # DELIVERABLE 3: the two correlations, one row per execution quality.
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(
        {"choice_space": options, "n_configs": n_configs,
         "correlations": out,
         "spread_at_efq": {str(k): v for k, v in summaries.items()},
         "harness_rows": {str(efq): {"correlations": st, "spreads": sp}
                          for efq, st, sp in rows},
         }, indent=2))
    print(f"\nwrote {harness_path}")
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
