"""How far a ridge probe on frozen features gets as the fit pool grows.

One panel per target. Each curve is a nested-prefix ladder over the SAME
embeddings: the probe is fit on the first n rows of one shuffle of the
six-month pool the supervised head itself trained on, so a point differs from
the one before it only by rows added, not by which rows were drawn. The dotted
line is the supervised specialist's TRAINED HEAD on the same eval month --
what the probe is being asked to catch.

WHAT CHANGED FROM THE OLD FIGURE. This used to ask "one month deep or six
months broad"; that question is answered and its script is kept beside this
one as probe_span_deep_vs_broad.py. The question here is whose FEATURES carry
the signal: the supervised trunk that was trained on the target, against
LeJEPA with time warping -- the strongest self-supervised arm on all three
targets in Table~\\ref{tab:probe_fit}, and the current default augmentation.
(LeJEPA same-stock crops, the old default, is not drawn: it is shortcut
learning.)

COLOR AND FURNITURE ARE THE PAPER'S. A target owns a color wherever it
appears -- purple/green/orange for return/volatility/spread, as in
forward_eval_v2 and variance_decomp -- so the supervised specialist takes its
target's color. The rest is those figures' shared helpers rather than local
choices: ``set_two_decimal_yticks`` (their y ticks never go past two
decimals), ``add_bottom_legend`` at the shared offset, ``save_figure`` for png
AND pdf. No suptitle and one x label under the middle panel, both as there.

EVERY POINT OVER THE SAME MONTHS. A month's pool size depends on its
cross-section, so the largest fit sizes exist for only some of them; averaging
whichever months reach each n would put a change of month SET into the shape
of the curve, which is the one thing the curve is supposed to isolate. A point
is drawn only where every month of that arm has it, and the curve ends where
its smallest month does. The full pool is drawn separately, hollow, at the
MEAN of the per-month maxima, because no single n is common to every month
there.

IC IS RAW, not floor-subtracted, and the comparison here is between arms on
one panel. The untrained floor has since landed on the full 31-month panel and
``--with-floor`` draws it as a fourth curve (black dashed, seed-averaged) into
a SEPARATE file, so the reported figure keeps the three arms it was written
for and the floor version is an additional read rather than a replacement.

Data is the probe-breadth sweep (scripts/eval/probe_fit_size.py reduce, one
json per job and eval month). PARTIAL IS EXPECTED -- the panel fills in as
jobs land, and the title says how many months each curve rests on.

    uv run plots/core/probe_fit_breadth.py [--alpha 10]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plots"))
sys.path.insert(0, str(ROOT / "plots/core"))
from paths import FT_JSON, HEAD_JSON, RESULTS, index_rows, result_files  # noqa: E402,F401
import readout as _readout  # noqa: E402
from style import (COMPACT_RC_PARAMS, SERIES_STYLES, WIDTH_FULL,  # noqa: E402
                   add_bottom_legend, apply_style, save_figure,
                   set_two_decimal_yticks)


# (task, panel title, the supervised specialist that owns it, its style key).
# The specialist's style key is what carries the target's color, so the panel,
# its supervised curve and its head line cannot drift apart.
# The last field is the legend corner. Pinned rather than "best": the curves
# rise left-to-right and the head lines sit across the top, so matplotlib's
# choice landed the Return box on top of the two dotted head lines. Each panel
# names the quadrant its own curves leave empty.
PANELS = [
    ("return_900", "Return", "sup_return_w8", "supervised_return",
     "lower right"),
    ("volatility_change_900", "Volatility Change", "sup_vol_w8",
     "supervised_vol", "lower right"),
    ("spread_change_900", "Spread Change", "sup_spread_w8",
     "supervised_spread", "center right"),
]
# LeJEPA + time warp: the default augmentation and the strongest SSL arm.
# tab:blue is LeJEPA's color in the paper.
LEJEPA_KEY = "pair_warp_6mo"
LEJEPA_COLOR = "tab:blue"
LEJEPA_LABEL = "LeJEPA (time warping)"

# THE SAME ARM, FINETUNED. A synthetic series key so the finetune rides the
# same _series() machinery as everything else -- the month-matching, the
# common-rung rule and the hollow full-pool marker are not worth a second
# implementation that can drift from this one.
#
# WHAT ITS X MEANS, AND HOW THAT DIFFERS FROM THE SOLID CURVES. On the frozen
# curves x is rows the RIDGE was fit on. Here it is rows the ENCODER was
# finetuned on; the readout is the standard full-pool probe, as in the
# reported table. Both are labelled rows spent, which is what makes one axis
# legitimate, but they are spent in different places -- that IS the question
# the panel now asks.
#
# IT IS READ OFF THE HEAD, not a probe. The finetune scores head-only, which
# is what makes it cheap: the a36 probe-fit month is never embedded. So these
# points are the model's own forecast rather than a ridge refit on its
# embeddings, while the solid curves beside them are probes. The head init is
# what makes them commensurable -- at step 0 the head IS that ridge probe --
# but they are not the same estimator, and the dotted head references on each
# panel are the like-for-like comparison. The rungs are also UN-ANNEALED: each
# sits in a WSD stable phase, so it is a lower bound on a decayed model there.
#
# AND THE CAVEAT. The finetune's head starts at a ridge probe fit on the FULL
# six-month span whatever x says (scripts/eval/fit_ridge_head_init.py), so
# this curve does not fall to zero on the left: it decays toward the full-pool
# probe, which is where its own init sits. It is not a total-label-budget
# curve and must not be read as one.
FT_KEY = "pair_warp_6mo_ft"
FT_LABEL = "LeJEPA Time Warping (finetuned)"

# ONE CURVE PER LEARNING RATE, because the LR is the thing under test. The
# 1e-5 wave lost a third of its IC in the first 32 steps from a head init that
# reproduces the probe exactly, so the arm was re-run at 1e-6; drawing only one
# of them would hide the comparison the second wave exists to make. A row's LR
# comes from the collector (`blr`), not from the directory name -- see
# collect_ssl_finetune_breadth.py.
FT_KEY_FMT = "pair_warp_6mo_ft_{}"


def _ft_label(blr) -> str:
    return f"{FT_LABEL}, blr {blr:g}" if blr else FT_LABEL


def _fmt_blr(blr) -> str:
    return f"{blr:g}" if blr else "unknown"

# THE MULTIHEAD'S HEAD ON EVERY PANEL. It trained a head on all three targets
# off ONE trunk, so unlike the specialists it gives a single model's reference
# line in all three -- the bar a general-purpose supervised trunk sets. Red is
# its color everywhere in the paper (SERIES_STYLES["multihead"]).
MULTI_KEY = "sup_multi_w8"

# THE UNTRAINED FLOOR AS A CURVE (--with-floor). probe_fit_table.py reports it
# as one number and strikes every cell that does not beat it; on this axis it
# answers a question that table cannot -- whether an arm's margin over a random
# encoder holds at every fit size, or only appears once the ridge has enough
# rows to exploit features the floor also has.
#
# SEEDS ARE AVERAGED PER (MONTH, N) BEFORE _series AVERAGES OVER MONTHS, which
# is floor_row()'s order in probe_fit_table.py. A seed is a draw from the same
# untrained architecture, so the three are exchangeable and one seed's noise
# must not set the level a whole panel is read against.
#
# Black dashed is SERIES_STYLES["randinit"], the paper's convention wherever
# the floor is drawn as a series rather than subtracted: it is a reference
# level, not another method. It cannot collide with the finetune's dashed blue.
FLOOR_KEY = "randinit"
FLOOR_COLOR = SERIES_STYLES["randinit"]["color"]
FLOOR_LABEL = "Random ViT"

# THE SPECIALIST'S HEAD LINE TAKES ITS TARGET'S COLOR, like its probe curve --
# the two are the same model read two ways, and the target owns the color
# everywhere else in the paper. Only the shared legend's SWATCH is neutral,
# since one key cannot be purple, green and orange at once; the dotted stroke
# is what identifies it, and the panel supplies the color.
HEAD_GRAY = "#555555"   # legend swatch and the hollow full-pool key only


def floor_by_month(alpha: float = 10.0) -> dict:
    """``{(task, eval_month): Random ViT IC}`` -- Table 1's floor row.

    The three random-init seeds' full-pool probe, averaged, read through
    plots/core/probe_fit_table.py's own loader so the two cannot disagree.
    Used ONLY for the band (--floor-se): the curve stays raw IC.
    """
    import probe_fit_table as pft  # noqa: E402
    ics, *_ = pft.load_results(alpha)
    acc: dict[tuple, list] = {}
    for seed in ("randinit_s42", "randinit_s43", "randinit_s44"):
        for task, by_month in ics.get(seed, {}).items():
            for ym, v in by_month.items():
                acc.setdefault((task, ym), []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def band_se(ys, months, task, floor) -> float:
    """s.e. across months of ``ys``, or of ``ys`` minus each month's floor.

    WHY THE FLOOR-ADJUSTED BAND. Most of the spread across months is the
    month being easy or hard for ANY encoder -- the Random ViT tracks the
    trained model's month means at r = +0.92 to +0.97 (variance_decomp) --
    and that shared difficulty is not uncertainty about the curve. The curve
    is still the raw mean IC; only its band is taken on the paired
    differences, which removes the month-difficulty term and leaves the
    variation in what the model adds.
    """
    ys = np.asarray(ys, dtype=float)
    if len(ys) < 2:
        return 0.0
    if floor is not None:
        missing = [m for m in months if (task, m) not in floor]
        if missing:
            raise SystemExit(f"no Random ViT for {task} {missing}")
        ys = ys - np.array([floor[(task, m)] for m in months])
    return float(ys.std(ddof=1) / np.sqrt(len(ys)))


def load_curves(alpha):
    """{series_key: {eval_month: {n: ic_by_task}}} over every result json."""
    idx = {(r["run_id"], r["eval_month"]): r["series_key"]
           for r in index_rows()}

    # COLLECT, THEN CHOOSE THE READOUT -- the same rule probe_fit_table uses,
    # from the same module. Assigning straight into `out` let the last file
    # read win, and `sorted()` puts "pball-*" (mean) before "pblast-*" (last),
    # so this figure and that table would silently have shown different
    # readouts for the same arm. See readout.py.
    seen: dict[tuple, list] = {}
    for name, rows in result_files(RESULTS):
        ym = Path(name).stem[-7:]
        if len(ym) != 7 or ym[4] != "-":
            continue
        for row in rows:
            if abs(float(row.get("alpha", -1)) - alpha) > 1e-9:
                continue
            series = idx.get((row["ckpt"], ym))
            if series is None:
                continue
            seen.setdefault((series, ym), []).append(row)

    out: dict[str, dict[str, dict[int, dict]]] = {}
    blank: set[str] = set()
    for (series, ym), rows in seen.items():
        kept, used = _readout.pick(rows, series)
        if used != _readout.PREDICT_READOUT:
            blank.add(series)
        for row in kept:
            out.setdefault(series, {}).setdefault(ym, {})[int(row["n"])] = row
    _readout.report(blank, sys.stderr)
    return out


def load_head():
    """{(series_key, task): {month: ic}} -- the trained-head reference."""
    out: dict[tuple, dict[str, float]] = {}
    if HEAD_JSON.exists():
        for r in json.loads(HEAD_JSON.read_text()):
            if r.get("ic") is not None:
                out.setdefault((r["series_key"], r["target"]), {})[
                    r["eval_month"]] = float(r["ic"])
    return out


def load_finetune(curves, field="ic", files=None, blrs=None):
    """Fold the finetune sweep into ``curves`` under :data:`FT_KEY`.

    Shaped exactly like the probe-breadth rows so ``_series`` can read it: the
    collector already resolves a full-pool run's ``rows-0`` to that month's
    real pool size, so each month contributes its own top rung and the hollow
    marker lands where it does for every other arm.
    """
    # EVERY WAVE FILE, not just the default one. The collector refuses to pool
    # two commits into one file, so each wave is collected to its own
    # ssl_finetune_breadth*.json and they are read back together here.
    #
    # ``files`` NARROWS THAT, and the reason is not tidiness. Two waves can
    # carry the same LR on the same months at different commits -- 1e-05 is in
    # both the 047986 wave and the 653736 pilot -- and one of those commits
    # predates the head-scale fix, so its head never moved. Pooled under one
    # key they would differ only in which file was read last. The collision
    # guard below refuses that; ``files`` is how a caller says which wave it
    # means.
    rows = []
    for f in (sorted(FT_JSON.parent.glob("ssl_finetune_breadth*.json"))
              if files is None else files):
        rows.extend(json.loads(Path(f).read_text()))
    if not rows:
        return
    # WHICH ESTIMATOR THE CURVE IS. ``ic`` is the finetune's own head; a wave
    # run with POST_TRAIN_PROBE=1 also carries ``ic_probe``, a fresh ridge on
    # the finetuned encoder. Against the frozen-probe curves beside it only
    # the probe is like for like -- see the FT_KEY note on commensurability --
    # so a months-restricted read of a probe-bearing wave should ask for it.
    # A wave without probes simply contributes nothing under field="ic_probe",
    # which is the honest outcome rather than a silent fallback to the head.
    for r in rows:
        if r.get(field) is None or not r.get("rows"):
            continue
        if blrs is not None and r.get("blr") not in blrs:
            continue
        # EVERY LADDER RUNG IS A POINT. Since the WSD rewrite these are
        # stable-phase checkpoints of ONE run at constant LR, so they were all
        # trained under the same optimizer state and differ only in how many
        # observations they consumed -- which is the axis. (Under the earlier
        # cosine wave each budget was its own run and only the endpoint
        # counted; that is a different commit and the collector refuses to
        # pool the two.)
        #
        # THE RUN ROOT IS KEPT, as that curve's FULL-POOL point. It is the
        # decayed endpoint -- the one properly annealed model on the arm -- and
        # every month's lands at a different budget, so it can never be a
        # shared rung and _series draws it exactly where it draws every other
        # arm's: hollow, at the mean of the per-month maxima. Dropping it threw
        # away the most defensible point on the curve to avoid a mismatch the
        # figure already has an idiom for.
        # THE X IS THE NOMINAL RUNG, NOT THE MEASURED COUNT. A checkpoint
        # lands on a whole number of steps, so two months asked for 32,768
        # observations report 32,512 and 33,024; _series draws only the
        # budgets EVERY month reached, so keying on the measured count leaves
        # one month per rung and no line at all. The collector does the
        # binning (and refuses to bin an annealed root, whose budget is that
        # month's own pool) -- here the root simply keeps its own count, which
        # is what makes it the hollow full-pool marker.
        n = r.get("rung") or int(r["rows"])
        key = FT_KEY_FMT.format(_fmt_blr(r.get("blr")))
        slot = (curves.setdefault(key, {}).setdefault(r["eval_month"], {})
                .setdefault(int(n), {}))
        # TWO WAVES, ONE SLOT. Same LR, month and rung from different commits
        # is not a duplicate to be deduplicated -- it is two different models,
        # and picking one by file order would put the head-scale bug into a
        # curve at random. Say which wave you meant.
        prev = slot.get(r["task"])
        if prev is not None and abs(prev - float(r[field])) > 1e-12:
            raise SystemExit(
                f"two waves both claim blr {_fmt_blr(r.get('blr'))} "
                f"{r['eval_month']} rung {int(n)} {r['task']} "
                f"({prev:g} vs {float(r[field]):g}); pass --ft-file to say "
                f"which one this figure means")
        slot[r["task"]] = float(r[field])


def fold_floor(curves):
    """Average the randinit seeds into one synthetic :data:`FLOOR_KEY` arm.

    Only the three panel targets are pooled: a result row also carries ``n``,
    ``alpha`` and ``ckpt``, and averaging those would be meaningless even
    though ``_series`` would never read them.
    """
    seeds = sorted(k for k in curves if k.startswith("randinit_s"))
    if not seeds:
        return
    tasks = [t for t, *_ in PANELS]
    acc: dict[str, dict[int, dict[str, list]]] = {}
    for s in seeds:
        for ym, rows in curves[s].items():
            for n, row in rows.items():
                for task in tasks:
                    v = row.get(task)
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        continue
                    (acc.setdefault(ym, {}).setdefault(int(n), {})
                     .setdefault(task, []).append(float(v)))
    curves[FLOOR_KEY] = {
        ym: {n: {t: float(np.mean(vs)) for t, vs in by_task.items()}
             for n, by_task in by_n.items()}
        for ym, by_n in acc.items()}


def restrict_months(curves, months):
    """Drop every month outside ``months`` from every arm, in place.

    THE FILTER GOES HERE, NOT IN ``_series``. Every rule downstream is stated
    over "the months this arm has" -- the common-rung intersection, the hollow
    full-pool marker at the mean of the per-month maxima, the head line that
    month-matches to its own arm's curve. Narrowing the set before any of them
    runs means they all keep meaning what they say; filtering inside the
    drawing code would leave each of them with its own idea of which months
    the figure is about.

    An arm with none of these months is removed rather than left empty, so it
    is absent from the legend instead of appearing as a curve that is nowhere.
    """
    want = set(months)
    for key in list(curves):
        kept = {m: v for m, v in curves[key].items() if m in want}
        if kept:
            curves[key] = kept
        else:
            del curves[key]


def _series(curves, key, task, floor=None):
    """(ns, mean, se, months, tops) for one arm on one task.

    ``ns`` are only the ladder rungs EVERY month of this arm reached; ``tops``
    are the per-month maxima, which is the full pool and is drawn on its own.
    """
    per_month = curves.get(key, {})
    by_n: dict[int, dict[str, float]] = {}
    tops: list[tuple[int, float]] = []
    months = []
    for ym, rows in per_month.items():
        vals = {n: r.get(task) for n, r in rows.items()
                if r.get(task) is not None
                and not (isinstance(r.get(task), float)
                         and math.isnan(r[task]))}
        if not vals:
            continue
        months.append(ym)
        for n, v in vals.items():
            by_n.setdefault(n, {})[ym] = float(v)
        mx = max(vals)
        tops.append((mx, float(vals[mx])))
    if not months:
        return None
    ns = sorted(n for n, d in by_n.items() if len(d) == len(months))
    if not ns:
        return None
    arr = [np.array([by_n[n][m] for m in months]) for n in ns]
    mu = np.array([float(a.mean()) for a in arr])
    se = np.array([band_se(a, months, task, floor) for a in arr])
    return ns, mu, se, sorted(months), tops


# One dash pattern per LR, so two finetune curves in the same blue are still
# told apart in print. Anything not listed falls back to a plain dash.
FT_DASHES = {"1e-05": (0, (1, 1.2)), "1e-06": "--"}


def draw(curves, head, alpha, out: Path, with_floor=False, floor=None,
         n_se=1.0):
    # THE PAPER'S FULL-TEXT WIDTH AND ITS COMPACT FONTS. Fixing the width is
    # what makes point sizes match across figures without LaTeX rescaling
    # anything; 0.36 is the three-panel-row aspect variance_decomp uses.
    apply_style(extra=COMPACT_RC_PARAMS)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_FULL, WIDTH_FULL * 0.36))
    any_data = False
    # WHICH MONTHS EACH ARM ACTUALLY RESTS ON. Every curve is a mean over its
    # own months, and the figure says nothing about which; on the full panel
    # they agree and the caption carries the count. On a narrowed panel they
    # need not agree -- an arm missing one of five months is a fifth of a
    # different comparison -- so the arms are reported to stderr, and a
    # disagreement is called out rather than left to be read off the caption.
    coverage: dict[str, set] = {}
    # The finetune is an overlay and lands job by job. Its key goes in the
    # legend only once a panel has actually drawn it -- a standing entry for a
    # curve that is not there reads as a curve that is zero everywhere.
    drew_ft = False
    drew_floor = False
    # Which finetune LRs actually landed. Sorted descending so the LOWEST LR
    # is drawn last and sits on top: it is the corrected arm.
    ft_blrs = sorted(
        {k[len(FT_KEY_FMT.format("")):] for k in curves
         if k.startswith(FT_KEY_FMT.format(""))},
        key=lambda t: -float(t) if t != "unknown" else 0.0)
    ft_blrs = [None if b == "unknown" else float(b) for b in ft_blrs]
    for ax, (task, title, sup_key, sup_style, loc) in zip(axes, PANELS):
        color = SERIES_STYLES[sup_style]["color"]
        arms = [(sup_key, "Supervised (probe)", color, "-"),
                (LEJEPA_KEY, LEJEPA_LABEL, LEJEPA_COLOR, "-")]
        # Dashed, same blue: one arm read two ways, not two arms. One entry
        # per learning rate present, cheapest LR last so the arm that is meant
        # to be read draws on top of the one it replaces.
        for blr in ft_blrs:
            arms.append((FT_KEY_FMT.format(_fmt_blr(blr)), _ft_label(blr),
                         LEJEPA_COLOR, FT_DASHES.get(_fmt_blr(blr), "--")))
        if with_floor:
            arms.append((FLOOR_KEY, FLOOR_LABEL, FLOOR_COLOR, "--"))
        n_months = {}
        for key, label, c, ls in arms:
            got = _series(curves, key, task, floor)
            if got is None:
                continue
            any_data = True
            ns, mu, se, months, tops = got
            n_months[label] = len(months)
            drew_ft = drew_ft or key.startswith(FT_KEY_FMT.format(""))
            drew_floor = drew_floor or key == FLOOR_KEY
            # linestyle by KEYWORD: a dash pattern is a tuple, which is not a
            # valid positional format string.
            coverage.setdefault(label, set()).update(months)
            ax.plot(ns, mu, marker="o", linestyle=ls, color=c, ms=2.4, lw=1.1,
                    label=label, zorder=3)
            ax.fill_between(ns, mu - n_se * se, mu + n_se * se, color=c,
                            alpha=0.16, lw=0)
            # THE FULL POOL, which the shared grid cannot show. The solid
            # curve stops at the last ladder rung EVERY month reached
            # (1,152,000); each month's actual pool is larger and a different
            # size, so no single x exists for it. Drawn hollow at the MEAN of
            # the per-month maxima -- an average x, not a shared one -- and
            # joined to the curve by a dotted stub so it reads as that curve's
            # endpoint rather than a stray marker. It is the largest fit
            # actually run, and the one closest to the head.
            if len(tops) == len(months):
                fx = float(np.mean([t[0] for t in tops]))
                fy = float(np.mean([t[1] for t in tops]))
                ax.plot([ns[-1], fx], [mu[-1], fy], ":", color=c, lw=0.9,
                        alpha=0.7, zorder=3)
                ax.plot([fx], [fy], "o", ms=4.5, mfc="none", mec=c, mew=1.1,
                        zorder=4)
        # THE TRAINED HEAD, month-matched to the supervised curve so the line
        # and the curve describe the same months.
        for hkey, hcolor, hlabel in (
                (sup_key, color, "Supervised (head)"),
                (MULTI_KEY, SERIES_STYLES["multihead"]["color"],
                 "Multihead (head)")):
            hd = head.get((hkey, task), {})
            # Month-matched to that arm's OWN curve, so the line and the arm
            # describe the same months. The multihead covers 25 of the 31
            # months, so its line is not over the same set as the specialist's
            # and must not borrow it.
            hv = [hd[m] for m in curves.get(hkey, {}) if m in hd]
            if hv:
                ax.axhline(float(np.mean(hv)), ls=":", lw=1.2, color=hcolor,
                           zorder=2, label=hlabel)
        # The month count lives in the caption, not the panel: it is the same
        # for all three and changes every time the sweep lands another job.
        ax.set_title(title, pad=4)
        ax.set_xscale("log")
        ax.grid(False)
        ax.tick_params(axis="both", length=2.5, width=0.6, pad=1.5)
        set_two_decimal_yticks(ax, nbins=5)
        # No per-panel legend: see the figure-level one below.
    axes[0].set_ylabel("Rank IC")
    # One x label, under the middle panel: the axis is the same on all three
    # and three copies of it is furniture, not information.
    axes[1].set_xlabel("Rows the Ridge Was Fit On")
    # ONE LEGEND FOR THE FIGURE. Everything in it means the same thing in all
    # three panels; the solid colored curve is the supervised specialist and
    # takes its target's color, which is the convention the rest of the paper
    # sets, so it is the panel that names it rather than a key.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], ls=":", lw=1.2, color=HEAD_GRAY),
        Line2D([], [], ls=":", lw=1.2,
               color=SERIES_STYLES["multihead"]["color"]),
        Line2D([], [], ls="-", lw=1.1, marker="o", ms=2.4,
               color=LEJEPA_COLOR),
    ]
    labels = ["Supervised Specialist (Head)", "Multihead (Head)",
              "LeJEPA Time Warping"]
    if drew_floor:
        handles.append(Line2D([], [], ls="--", lw=1.1, marker="o", ms=2.4,
                              color=FLOOR_COLOR))
        labels.append(FLOOR_LABEL)
    if drew_ft:
        # ONE ENTRY PER LEARNING RATE, matching the dash each was drawn with.
        # With a single LR present the label stays the plain one; the moment
        # there are two, an unlabelled pair of blue dashed curves would be
        # unreadable, and the LR is the whole difference between them.
        for blr in ft_blrs:
            handles.append(Line2D([], [], lw=1.1, marker="o", ms=2.4,
                                  linestyle=FT_DASHES.get(_fmt_blr(blr), "--"),
                                  color=LEJEPA_COLOR))
            labels.append("LeJEPA Time Warping (Finetuned)" if len(ft_blrs) < 2
                          else f"LeJEPA TW (Finetuned, blr {blr:g})")
    handles.append(Line2D([], [], marker="o", ms=4.5, mfc="none",
                          mec=HEAD_GRAY, mew=1.1, ls="none"))
    labels.append("Full Pool (Mean N)")
    fig.tight_layout()
    # LEGEND_BOTTOM_RESERVE_1ROW, the shared default -- the x label sits in
    # this reserved band, so trimming it crowds the label against the legend.
    # Only the intra-legend spacing is tightened: at WIDTH_FULL these four
    # entries run close to the full text width.
    # ONE ROW WHILE THE ENTRIES FIT. add_bottom_legend picks the bottom
    # reserve from len(labels)/ncol, so a hardcoded 4 would send five entries
    # to a second row with one lonely key in it.
    ncol = len(labels) if len(labels) <= 5 else 3
    add_bottom_legend(fig, handles, labels, ncol=ncol,
                      columnspacing=1.2, handletextpad=0.5)
    save_figure(fig, out)
    plt.close(fig)
    return any_data, coverage


def main():
    global RESULTS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=str(RESULTS))
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--out", default=None)
    p.add_argument("--with-floor", action="store_true",
                   help="draw the untrained Random ViT floor as a fourth "
                        "curve, seed-averaged; writes _floor.png unless --out "
                        "says otherwise, so the reported figure is untouched")
    p.add_argument("--with-finetune", action="store_true",
                   help="draw the SSL finetune as an extra curve; writes "
                        "_ft.png unless --out says otherwise, so the reported "
                        "figure is untouched. Partial is expected -- it draws "
                        "from whatever months have landed.")
    p.add_argument("--months", default=None,
                   help="comma-separated eval months; every arm is narrowed "
                        "to these before any curve is built. --out is "
                        "required with it, so the reported figure cannot be "
                        "overwritten by a subset of its own panel.")
    p.add_argument("--ft-readout", choices=("head", "probe"), default="head",
                   help="which finetune number to draw: its own head "
                        "(default, and what the reported _ft figure shows) or "
                        "a fresh ridge on the finetuned encoder. Only "
                        "'probe' is like-for-like against the frozen-probe "
                        "curves beside it; waves run head-only contribute "
                        "nothing under it.")
    p.add_argument("--ft-blr", default=None,
                   help="comma-separated finetune learning rates to draw; "
                        "default draws every one present")
    p.add_argument("--ft-file", action="append", default=None,
                   help="a collected wave json to read instead of every "
                        "ssl_finetune_breadth*.json; repeatable. Needed "
                        "whenever two waves share an LR and a month.")
    a = p.parse_args()
    if a.months and a.out is None:
        raise SystemExit("--months needs --out: a narrowed panel must not be "
                         "written over the reported figure")
    if a.out is None:
        stem = Path(__file__).with_suffix("")
        suffix = (("_floor" if a.with_floor else "")
                  + ("_ft" if a.with_finetune else ""))
        a.out = str(stem.with_name(stem.name + suffix).with_suffix(".png"))

    RESULTS = Path(a.results)
    curves = load_curves(a.alpha)
    if not curves:
        raise SystemExit(f"no results under {RESULTS} (alpha {a.alpha:g})")
    # Folded in AFTER the emptiness check: the finetune is an overlay on this
    # figure, not a reason for it to exist, and a run of it with no probe
    # results at all is a broken invocation whatever the finetune has landed.
    #
    # OPT-IN, like the floor. The reported figure is the frozen-probe
    # comparison and must not acquire a curve read off a different estimator
    # (the finetune scores its HEAD) just because a sweep happens to have
    # landed some months.
    if a.with_finetune:
        load_finetune(
            curves,
            field="ic" if a.ft_readout == "head" else "ic_probe",
            files=a.ft_file,
            blrs=({float(x) for x in a.ft_blr.split(",")}
                  if a.ft_blr else None))
    if a.with_floor:
        fold_floor(curves)
    # AFTER every arm is folded in, so the floor and the finetune are narrowed
    # by the same rule as the probe curves rather than each at its own loader.
    if a.months:
        restrict_months(curves, [m.strip() for m in a.months.split(",")])
        if not curves:
            raise SystemExit(f"no arm has any of {a.months}")
    ok, coverage = draw(curves, load_head(), a.alpha, Path(a.out),
                        with_floor=a.with_floor,
                        # Band = 2 s.e. of each month's IC minus its Random
                        # ViT (Table 1's floor); the curves stay raw IC.
                        floor=floor_by_month(a.alpha), n_se=2.0)
    if not ok:
        raise SystemExit("results found but no arm drawn -- wrong keys?")
    months = {m for d in curves.values() for m in d}
    print(f"wrote {a.out} ({len(curves)} arm(s) over {len(months)} month(s))")
    for label, ms in sorted(coverage.items()):
        print(f"  {label}: {len(ms)} month(s) {' '.join(sorted(ms))}",
              file=sys.stderr)
    if len({frozenset(v) for v in coverage.values()}) > 1:
        print("  WARNING: the arms above do not rest on the same months, so "
              "the panel compares means over different sets.", file=sys.stderr)


if __name__ == "__main__":
    main()
