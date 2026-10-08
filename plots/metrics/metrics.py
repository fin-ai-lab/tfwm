"""Breadth figures: rank IC across three targets x six forward horizons.

Outputs (in this folder, per figure x {png, pdf}):

  * ``all_metrics_ic_minus_random``  — the standard LeJEPA encoder against the
    model-free mean-reversion line, as a per-month difference from a
    random-init encoder (the default).
  * ``all_metrics_ic``               — the same series in ABSOLUTE rank IC,
    with the untrained floor drawn as its own line (``--absolute``).
  * ``all_metrics_presentation_minus_random`` — the talk cut: the time-warp
    LeJEPA arm and the supervised specialist (with its head star) only, and
    no reference line (``--out-name``, below). It draws a DIFFERENT LeJEPA
    arm from the figure above — see PRESENTATION_TWINS.

The series a figure draws are named on the command line; what each one IS
lives in ``SERIES_DEFS``. The classical finance suite is drawn by this same
script (see plots/finance_baselines/README.md) so the econometric models and
the encoders land on one axis, and the LeJEPA augmentation figure has its own
entry point in ``plots/metrics/lejepa_augs.py``.

Input: ``<CKPT_ROOT>/<project>/<run_id>/xs_ic.json`` — the 18 (type, horizon)
ridge probes the in-job scorer writes beside every checkpoint — plus the
random-init floor from ``randinit-fwd3-*`` and the checked-in per-month
artifacts (``finance_panel_ic.json``, ``mean_reversion_ic.json``,
``supervised_probe_ic.json``, ``supervised_head_ic.json``, ``tsfm_ic.json``)
that other scripts in the repo produce.

METRIC. Rank IC, cross-sectionally, synchronized across tickers. There is no
AUC path here any more: probe AUC and its ΔAUC difference were retired from
the project on 2026-08-19, and every producer, artifact and figure that spoke
that metric is gone (the delta-AUC pipeline, ``metrics_families.py``'s family
figures, ``metrics_hero.py``, ``baseline_variation.py``, the
``finance_raw_results*.json`` snapshots). A number in a commit older than that
is not comparable with anything this script draws.

Run:
    uv run plots/metrics/metrics.py
    uv run plots/metrics/metrics.py --absolute \
        --series cross_stock_k2ind randinit_vit mean_reversion

    # all_metrics_supervised_ic -- the THREE specialists, each with its head
    # star. Recorded here because this figure has no script of its own. The
    # stars carry no legend key (``no_legend``), so the legend is the three
    # lines and the caption explains the star; they still go last, to render on
    # top.
    #
    # THE MULTIHEAD IS NOT IN IT AND THE COMMAND ABOVE USED TO SAY IT WAS,
    # which made the recorded invocation exit 1. sup_multi_w8 yields a value
    # for 0 of 32 eval months, and the arm it names is retired twice over: the
    # W=8 wording is a leftover of the ALiBi recency-window campaign (the
    # recency prior itself was deleted with no backward compatibility), and the
    # multihead checkpoints behind it were dropped when MultiTaskSupervisedModel
    # turned out never to read xs_cell (57c671c, b7e8948). Add it back only if
    # a re-run covers the reported panel; the `_w8` suffixes in the series keys
    # mean nothing now and are kept only because renaming them moves every
    # caller.
    uv run plots/metrics/metrics.py --out-name all_metrics_supervised_ic \
        --series sup_return_w8 sup_vol_w8 sup_spread_w8 \
                 sup_return_w8_head sup_vol_w8_head sup_spread_w8_head

    # all_metrics_presentation -- the two-line talk cut: the time-warp LeJEPA
    # arm against the supervised specialist, plus that specialist's head star
    # (drawn, never keyed). No third reference line. Also has no script of
    # its own. Every series is a ``pres_*`` twin of a paper entry (see
    # PRESENTATION_TWINS), so the numbers are ones a paper figure already
    # draws -- but READ THAT BLOCK: the LeJEPA line here is pair_warp, not
    # the cross_stock_k2ind arm the standard figure draws.
    uv run plots/metrics/metrics.py --out-name all_metrics_presentation \
        --series pres_lejepa pres_supervised pres_supervised_head

The finance-suite command lives in plots/finance_baselines/README.md; the
augmentation, SSL and frozen-TSFM family figures have their own entry points
(``lejepa_augs.py``, ``ssl_baselines_ic.py``, ``tsfm_ic.py``).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLOTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PLOTS_DIR))
# K2IND_GLOB / K2IND_LAMB / K2IND_N_MONTHS and MODELS_ARCHIVE are deliberately
# NOT imported: no series here reads the archive any more (2026-08-29), and an
# import of them is an invitation to point a figure back at a retired
# generation. ``ckpt_root`` remains a supported pin -- see _select_runs -- so a
# series that genuinely needs the archive re-imports it and says why.
from style import (  # noqa: E402
    CKPT_ROOT, IC_METRIC_LABEL,
    LEGEND_BOTTOM_RESERVE_1ROW, LEGEND_BOTTOM_RESERVE_2ROW,
    SERIES_MARKER, SERIES_STYLES, WIDTH_FULL, _get_dotted,
    add_bottom_legend, apply_style, iter_ic_runs, load_randinit_ic,
    load_sweep_months, save_figure, set_two_decimal_yticks,
)


OUT_DIR = Path(__file__).resolve().parent
# Live wandb cache, shared with five other plot scripts. Nothing in THIS file
# reads it any more — the AUC-era wandb series went with the metric — but the
# file is this folder's by convention and is still written elsewhere.
API_CACHE = OUT_DIR / "api_cache.json"

HORIZONS: tuple[int, ...] = (300, 600, 900, 1800, 3600, 7200)
TARGET_TYPES: tuple[str, ...] = ("return", "volatility_change", "spread_change")

WM_PANEL_TITLES = {
    "return":            "Return",
    "volatility_change": "Volatility Change",
    "spread_change":     "Spread Change",
}


# ─── the canonical SERIES dict ─────────────────────────────────────────────
#
# Color + label come from plots/style.SERIES_STYLES (canonical paper palette)
# so every figure renders the same family the same way; markers are uniformly
# SERIES_MARKER ('o').

# Per-month rank ICs for every classical / view-learner baseline, one
# record per (eval_month, target, model), written by
# plots/finance_baselines/calculate_panel_ic.py.
FIN_IC = "finance_panel_ic.json"

# What rebuilds each ``json_ic`` artifact, printed when one is missing. Every
# series here is fed by a script somewhere else in the repo, and "not built
# yet" is only actionable with the command attached.
JSON_IC_PRODUCERS = {
    FIN_IC: "plots/finance_baselines/calculate_panel_ic.py --months-from-k2ind",
    "mean_reversion_ic.json":
        "plots/metrics/mechanical_baseline.py --months-from-k2ind",
    "supervised_probe_ic.json": "plots/metrics/build_supervised_ic.py",
    "supervised_head_ic.json":
        "plots/metrics/build_supervised_ic.py --readout head",
    "tsfm_ic.json": "plots/metrics/build_tsfm_ic.py",
}

# Shared by the five LeJEPA pairing series below: what makes a checkpoint
# under ``lejepa-*-lambda-*`` a member of the REPORTED series rather than of
# the holdout sweep that chose its lambda, or of a superseded recipe. See the
# block comment beside ``pair_rrc``.
PAIRING_PINS: dict = {
    "months": set(load_sweep_months()),
    "xs_stats": "xs_anchor_stats_fwdvwap60",
    "meta_pins": {"config.dataset.xs_target": "rank"},
}
# What a COMPLETE pairing series looks like. Not spelled as ``expect_months``:
# that key raises, and two of the five arms are still filling in on the 40 GB
# box. plots/metrics/lejepa_augs.py reports coverage against this instead, and
# draws every arm over the months they all share.
PAIRING_N_MONTHS = 32

# Shared by the nine ssl_* series below. The commit is pinned in the glob for
# the reason given beside ``ssl_byol``; ``xs_stats`` pins the TARGET GENERATION
# (all three targets became forward-window differences on 2026-08-22, so a glob
# spanning that change would average two different quantities into one cell).
# No ``meta_pins``: unlike the pairing arms, these projects hold nothing but
# the reported series — the HPO rounds live under ``ssl-ic-<method>-*``.
SSL_IC_GLOB = "ssl-ic-final-222d99-*"
SSL_IC_PINS: dict = {
    "months": set(load_sweep_months()),
    "xs_stats": "xs_anchor_stats_fwdvwap60",
}
SSL_IC_N_MONTHS = 32

# Shared by the four supervised series.
#
# NOT "W=8" any more: three specialists and the multihead now all read the
# full-history pairwise recipe, where the recency/W axis does not exist. The
# name SUP_PINS and the W=8 wording are leftovers from the retired campaign.
#
# THE WINDOW IS ONLY IN THE RUN NAME. The comment here used to claim
# ``meta_pins`` re-checks the window against the recorded config "so a rename
# cannot quietly widen it" -- SUP_PINS carries no meta_pins and never did, and
# nothing else checks the window either. What actually happened is the
# opposite of a quiet widening: the arms gained a ``recency<N>`` token
# (recency0/4/8/16/32 all exist under supervised-loss-ablation-*), the four
# regexes below still spelled the pre-token name, and every one of them
# matched ZERO checkpoints -- so the figure rendered empty rather than wrong.
# The regexes named recency8 explicitly, which is what W=8 meant. That is now
# history: all four series read the full-history recipe and carry no W token.
SUP_PINS: dict = {
    "months": set(load_sweep_months()),
    "xs_stats": "xs_anchor_stats_fwdvwap60",
}

# ── THE SIX-MONTH-SPAN WAVE (2026-09-13) ───────────────────────────────────
#
# The three specialists were retrained under the locked recipe: six months of
# data ending at the eval month, 12 passes, blr 2e-4, 256 cells/step. That is
# a different generation from the single-month 3e5087 wave the series below
# used to read, so THE COMMIT IS PINNED IN THE GLOB rather than left to the
# run name. Two other waves answer to the same prefix and the same
# ``blr0.0002`` name: 3e5087 (single month, blr 1e-5) and 62bffe, which is
# this exact recipe submitted through the full-history launcher by mistake and
# cancelled -- right recipe, WRONG months. Matching the prefix alone would
# average generations together with nothing in a run name to say so. To read a
# newer wave, change this hash and nothing else.
SUP_SPAN_COMMIT = "581eb2"

# ONE MONTH SHORT OF THE REPORTED PANEL, STRUCTURALLY. A six-month span ending
# at 2008-02 starts 2007-09, before the data begins, so the launcher drops that
# month and no rerun will ever produce it. The panel for this wave is the other
# 31 sampled months; ``expect_months`` below is derived from this set so a
# genuinely missing month still raises.
SUP_SPAN_NO_SPAN = {"2008-02"}
SUP_SPAN_PINS: dict = {
    "months": set(load_sweep_months()) - SUP_SPAN_NO_SPAN,
    "xs_stats": "xs_anchor_stats_fwdvwap60",
}

SERIES_DEFS: dict[str, dict] = {
    # A series gets its ICs from ONE of three sources, and says which by the
    # key it carries: ``ckpt_glob`` reads xs_ic.json out of the checkpoint
    # tree, ``json_ic`` reads a checked-in per-month artifact built by another
    # script, and ``random_init_baseline`` builds the untrained floor out of
    # the very numbers every other series is differenced against.
    #
    # THE STANDARD LeJEPA MODEL. The glob, the lamb pin and the month count
    # all come from style.py -- see the comment above standard_k2ind_runs
    # there for why this must never be spelled out by hand. It used to read
    # "k2ind-lamb-*" + lamb 0.01, which is the LAMBDA SWEEP filtered to one
    # arm: 13 months, not 32, and no error anywhere to say so.
    # ARCHIVED 2026-08-14 with the IC switch, hence ckpt_root: the tree it
    # used to sit in no longer holds it, and a glob that finds nothing here
    # would be a figure short of its whole reference line. Its xs_ic.json
    # carries the CURRENT forward-VWAP targets (the stamp is inside the IC
    # file rather than in train_meta for this generation, which is why the
    # series cannot also pin xs_stats).
    "cross_stock_k2ind": {
        # SWITCHED OFF THE ARCHIVE 2026-08-29. This read K2IND_GLOB +
        # K2IND_LAMB out of MODELS_ARCHIVE (cross-stock-ind-c7ff4e-* at
        # lambda 0.01), a RETIRED generation -- a materially different model,
        # not a relocation of this one. On the reported panel that scores
        # +0.0039 return dIC at h=900 against this arm's +0.0121, so the
        # breadth figure was advertising a 3x weaker "standard LeJEPA" than
        # the augmentation figure drew, under the same name.
        #
        # Defined exactly like ``pair_k2ind`` below -- same glob, same
        # run-name pin, same PAIRING_PINS -- because it IS that arm: k2ind at
        # its holdout-2 lambda winner 0.1 (2026-08-27 per-pairing sweep). Kept
        # as its own key so the two figures can style it differently; the
        # SELECTION must stay identical to pair_k2ind.
        #
        # style.K2IND_GLOB / K2IND_LAMB still point at the archive and are
        # still used by standard_k2ind_months() to enumerate the 32 sweep
        # months -- a month list, not a drawn series, and both generations
        # cover the same months.
        "ckpt_glob": "lejepa-k2-lambda-*",
        "run_name_re": re.compile(r"^k2ind_lamb0\.1_bs256_s42$"),
        **PAIRING_PINS,
        **SERIES_STYLES["cross_stock_k2ind"], "marker": SERIES_MARKER,
    },
    # Not a checkpoint series: built from the random-init probe results that
    # every OTHER series is differenced against. Only meaningful on an
    # absolute figure -- under the default difference it is identically zero,
    # which the axhline already shows -- so load_ic_metrics drops it there.
    "randinit_vit": {
        "random_init_baseline": True,
        **SERIES_STYLES["randinit"], "marker": SERIES_MARKER,
    },
    # Model-free: -spread(t) and -bwd_vol, the level each change target
    # subtracts, read off THE VIEW THE ENCODER IS FED rather than off the raw
    # grid. That restriction is the whole point of drawing it here: after
    # normalize_numpy standardizes the price group by each view's own mean and
    # std, the absolute spread level is gone, and a line computed on raw units
    # would be a baseline no model on this panel could reach. Scored by
    # plots/metrics/mechanical_baseline.py on the same synchronized panel, so
    # it is comparable line-for-line with the encoders.
    # No return series: that target has no analogous level term.
    "mean_reversion": {
        "json_ic": "mean_reversion_ic.json",
        **SERIES_STYLES["mean_reversion"], "marker": SERIES_MARKER,
    },
    # ── the five LeJEPA POSITIVE PAIRINGS, each at its own lambda ──────────
    #
    # What changes across these five is WHAT COUNTS AS A POSITIVE PAIR, and
    # nothing else: one recipe (blr 6e-5, batch 256, 200 epochs, wd 5e-2,
    # seed 42, pool=cls, information token on, forward-VWAP targets) built so
    # the arms are readable against each other.
    #
    # LAMBDA IS PART OF THE ARM, not a shared constant. Each pin below is
    # that pairing's winner on holdout set 2 (5 months, never the reported
    # 32), from the 2026-08-27 sweep; the arms split into two regimes, with
    # everything cross-stock or noise-based wanting lambda >= 0.1 and
    # time_warp wanting the grid minimum. Two caveats travel with the picks:
    # rrc's 0.05 is a MID-GRID choice on a flat, non-monotone curve rather
    # than an argmax, and time_warp's 0.001 is a GRID EDGE still descending —
    # not a located optimum.
    #
    # These are ``lejepa-*-lambda-*`` projects, which also hold the holdout
    # sweeps themselves under the SAME run names: the name carries arm and
    # lambda, not the month set. Hence ``months`` (the reported 32) on every
    # entry, and ``meta_pins`` for the recipe, which the name does not carry
    # either — one surviving 2009-06 k2 run answers to the same name with the
    # retired z-score target.
    "pair_rrc": {
        "ckpt_glob": "lejepa-samestock-lambda-*",
        "run_name_re": re.compile(r"^random_resized_crop_lamb0\.05_bs256_s42$"),
        **PAIRING_PINS,
        **SERIES_STYLES["lejepa_rrc"], "marker": SERIES_MARKER,
    },
    "pair_warp": {
        "ckpt_glob": "lejepa-samestock-lambda-*",
        "run_name_re": re.compile(r"^time_warp_lamb0\.001_bs256_s42$"),
        **PAIRING_PINS,
        **SERIES_STYLES["aug_warp"], "label": "Crops + time warp",
        "marker": SERIES_MARKER,
    },
    # REPOINTED 2026-08-29 to the post-fix rerun. Until then gaussian_noise
    # was the one pairing whose augmentation also perturbed the INFORMATION
    # TOKEN -- the 11 per-window constants (norm stats + window descriptors)
    # are appended as channels before augmentation, so N(0, 0.75^2) landed on
    # them too, at 0.7x-8.1x each channel's natural spread (tod_start, a
    # fraction of the session and so bounded in [0, 1], reached 2.717). Fixed
    # in streaming_dataset.py by slicing those rows out of the noise draw;
    # tests/test_time_warp_noise.py pins it.
    #
    # These are ``lejepa-noisefix-lambda-*`` and the run name is
    # ``noisefix_*``, NOT ``gaussian_noise_*`` -- deliberately disjoint from
    # the old project so both halves coexist and the fix has a matched
    # before/after. lambda stayed 0.3: on the 19 months carrying both, the
    # paired difference was return -0.0010 (t = -1.4) with vol and spread
    # flat, so the fix is ~free rather than the cost an early 4-month read
    # suggested.
    #
    # The lambda 0.2/0.4 holdout arms live in this same project under their
    # own names, and its 5 holdout-2 months are dropped by PAIRING_PINS
    # ["months"] -- neither leaks in.
    "pair_noise": {
        "ckpt_glob": "lejepa-noisefix-lambda-*",
        "run_name_re": re.compile(r"^noisefix_lamb0\.3_bs256_s42$"),
        **PAIRING_PINS,
        **SERIES_STYLES["aug_noise"], "label": "Crops + noise",
        "marker": SERIES_MARKER,
    },
    "pair_k2": {
        "ckpt_glob": "lejepa-k2-lambda-*",
        "run_name_re": re.compile(r"^k2_lamb0\.2_bs256_s42$"),
        **PAIRING_PINS,
        **SERIES_STYLES["cross_stock_k2"], "marker": SERIES_MARKER,
    },
    "pair_k2ind": {
        "ckpt_glob": "lejepa-k2-lambda-*",
        "run_name_re": re.compile(r"^k2ind_lamb0\.1_bs256_s42$"),
        **PAIRING_PINS,
        **SERIES_STYLES["cross_stock_k2ind"], "label": "Cross-stock same-ind.",
        "marker": SERIES_MARKER,
    },
    # ── the four SUPERVISED HEADS ─────────────────────────────────────────
    #
    # One recipe across all four (pairwise, rep32nost, lr 3e-5, bs 256, seed
    # 42), so the heads are readable against each other and against the
    # pairing arms above. All four cover the reported 32 months exactly.
    #
    # NOT ``supervised_specialist`` (above): that is the cross-entropy
    # specialist read from a json_ic artifact, a different loss and a
    # different recipe. And NOT supervised-full-month-multihead-ce, which is
    # the 202-month CE multihead; the multihead here is the 32-month pairwise
    # arm, matched to the other three.
    "sup_return_w8": {
        # THE SIX-MONTH-SPAN SPECIALIST (repointed 2026-09-14). Pairwise on
        # the uniform target at the locked recipe -- see SUP_SPAN_PINS for why
        # the commit is in the glob and why the panel is 31 months, not 32.
        #
        # SCORED HEAD-ONLY. This wave ran under POST_TRAIN_PROBE=0, so the
        # checkpoints carry ``xs_ic/head:<task>`` at h=900 and NO probe keys:
        # the ``_head`` twin below draws, and the probe line drawn from this
        # entry is empty until the checkpoints are re-scored with the ridge.
        "ckpt_glob": f"supervised-full-month-return-{SUP_SPAN_COMMIT}-*",
        "run_name_re": re.compile(r"^\d{4}-\d{2}_pairwise_blr0\.0002$"),
        **SUP_SPAN_PINS,
        **SERIES_STYLES["supervised_return"], "label": "Supervised (return)",
        "marker": SERIES_MARKER,
    },
    "sup_vol_w8": {
        # THE SIX-MONTH-SPAN SPECIALIST (repointed 2026-09-14). Pairwise on
        # the uniform target at the locked recipe -- see SUP_SPAN_PINS for why
        # the commit is in the glob and why the panel is 31 months, not 32.
        #
        # SCORED HEAD-ONLY. This wave ran under POST_TRAIN_PROBE=0, so the
        # checkpoints carry ``xs_ic/head:<task>`` at h=900 and NO probe keys:
        # the ``_head`` twin below draws, and the probe line drawn from this
        # entry is empty until the checkpoints are re-scored with the ridge.
        "ckpt_glob": f"supervised-full-month-vol-change-{SUP_SPAN_COMMIT}-*",
        "run_name_re": re.compile(r"^\d{4}-\d{2}_pairwise_blr0\.0002$"),
        **SUP_SPAN_PINS,
        **SERIES_STYLES["supervised_vol"], "label": "Supervised (vol)",
        "marker": SERIES_MARKER,
    },
    "sup_spread_w8": {
        # THE SIX-MONTH-SPAN SPECIALIST (repointed 2026-09-14). Pairwise on
        # the uniform target at the locked recipe -- see SUP_SPAN_PINS for why
        # the commit is in the glob and why the panel is 31 months, not 32.
        #
        # SCORED HEAD-ONLY. This wave ran under POST_TRAIN_PROBE=0, so the
        # checkpoints carry ``xs_ic/head:<task>`` at h=900 and NO probe keys:
        # the ``_head`` twin below draws, and the probe line drawn from this
        # entry is empty until the checkpoints are re-scored with the ridge.
        "ckpt_glob": f"supervised-full-month-spread-change-{SUP_SPAN_COMMIT}-*",
        "run_name_re": re.compile(r"^\d{4}-\d{2}_pairwise_blr0\.0002$"),
        **SUP_SPAN_PINS,
        **SERIES_STYLES["supervised_spread"], "label": "Supervised (spread)",
        "marker": SERIES_MARKER,
    },
    "sup_multi_w8": {
        # THE FULL-HISTORY MULTIHEAD, matching the three specialists above.
        # This was "supervised-multihead-*" against a run name carrying
        # rep32nost/recency8 -- a project prefix nothing writes and a name no
        # sweep in the tree can emit (`grep -rn "recency8\|rep32nost" scripts/`
        # returns nothing). So while the three specialists were repointed at
        # the full-history recipe, this one was left describing the retired
        # W-axis campaign and matched zero checkpoints: the same empty-figure
        # failure the comment above says it had just fixed.
        #
        # full_data_multihead.sh sets SWEEP_NAME=supervised-full-month-multihead
        # and delegates to supervised_multihead.sh, whose name is
        # ${YM}_multi_${LOSS_TAG}${TAG}_s${SEED} -- pairwise, no tag, seed 42.
        "ckpt_glob": "supervised-full-month-multihead-*",
        "run_name_re": re.compile(r"^\d{4}-\d{2}_multi_pairwise_s42$"),
        **SUP_PINS,
        **SERIES_STYLES["multihead"], "label": "Supervised (multi)",
        "marker": SERIES_MARKER,
    },
    # ── the nine SSL BASELINES, IC-re-optimized (the ssl_ic campaign) ─────
    #
    # One entry per method, all reading the SAME 32-month series that
    # sweeps/ssl_ic/final32_pred.sh trained: project
    # ``ssl-ic-final-222d99-<month>``, one run per method named
    # ``<method>-final``. The glob pins the commit deliberately — every wave of
    # that series was submitted with ``--commit-hash 222d99`` precisely so all
    # nine land in one project per month, and a bare ``ssl-ic-final-*`` would
    # silently absorb any later re-run under a different hash.
    #
    # CONFIGS ARE NOT RE-DERIVABLE FROM THE RUN NAME: every method answers to
    # ``<method>-final`` whatever hyper-parameters it carries, so what each one
    # IS lives in sweeps/ssl_ic/final32.sh, where each line records its winner
    # and the holdout-2 number that chose it. Five methods (byol, cpc, dino,
    # ijepa, tfc) ship round-2 winners; four (cost, mae, timemae, ts2vec) ship
    # ROUND-1 winners because round 2 was stopped early — see that file.
    #
    # SELECTION WAS ON HOLDOUT SET 2, NOT ON THESE MONTHS, and the two panels
    # disagree sharply: mae was 6th of 9 on holdout-2 and leads here, while tfc
    # took the largest round-2 gain there and is the only negative arm here.
    # Treat the holdout-2 ordering as a tuning artifact, not a preview.
    "ssl_byol": {
        "ckpt_glob": SSL_IC_GLOB, "run_name_re": re.compile(r"^byol-final$"),
        **SSL_IC_PINS, **SERIES_STYLES["byol"], "marker": SERIES_MARKER,
    },
    "ssl_cost": {
        "ckpt_glob": SSL_IC_GLOB, "run_name_re": re.compile(r"^cost-final$"),
        **SSL_IC_PINS, **SERIES_STYLES["cost"], "marker": SERIES_MARKER,
    },
    "ssl_cpc": {
        "ckpt_glob": SSL_IC_GLOB, "run_name_re": re.compile(r"^cpc-final$"),
        **SSL_IC_PINS, **SERIES_STYLES["cpc"], "marker": SERIES_MARKER,
    },
    "ssl_dino": {
        "ckpt_glob": SSL_IC_GLOB, "run_name_re": re.compile(r"^dino-final$"),
        **SSL_IC_PINS, **SERIES_STYLES["dino"], "marker": SERIES_MARKER,
    },
    "ssl_ijepa": {
        "ckpt_glob": SSL_IC_GLOB, "run_name_re": re.compile(r"^ijepa-final$"),
        **SSL_IC_PINS, **SERIES_STYLES["ijepa"], "marker": SERIES_MARKER,
    },
    "ssl_mae": {
        "ckpt_glob": SSL_IC_GLOB, "run_name_re": re.compile(r"^mae-final$"),
        **SSL_IC_PINS, **SERIES_STYLES["mae"], "marker": SERIES_MARKER,
    },
    "ssl_tfc": {
        "ckpt_glob": SSL_IC_GLOB, "run_name_re": re.compile(r"^tfc-final$"),
        **SSL_IC_PINS, **SERIES_STYLES["tfc"], "marker": SERIES_MARKER,
    },
    "ssl_timemae": {
        "ckpt_glob": SSL_IC_GLOB, "run_name_re": re.compile(r"^timemae-final$"),
        **SSL_IC_PINS, **SERIES_STYLES["timemae"], "marker": SERIES_MARKER,
    },
    "ssl_ts2vec": {
        "ckpt_glob": SSL_IC_GLOB, "run_name_re": re.compile(r"^ts2vec-final$"),
        **SSL_IC_PINS, **SERIES_STYLES["ts2vec"], "marker": SERIES_MARKER,
    },
    # ── the three FROZEN TSFMs, each at its held-out layer ────────────────
    #
    # Chronos-2, TimesFM 3.0 and Kronos-base read frozen, with a ridge probe
    # on one hidden state. WHICH hidden state is a hyper-parameter, chosen per
    # (family, target) on the 5-month optimization set and re-scored on the
    # reported 32 — so the layer varies across the three panels of one family
    # (Chronos-2: return L5, vol L2, spread L2) and every record carries the
    # layer it came from.
    #
    # THREE, NOT FIVE. TimesFM 2.5 and Sundial were dropped from the arm on
    # 2026-09-11 and never re-scored at the readout below, so there is nothing
    # on disk to draw them from; their series defs went with them rather than
    # sit here resolving to an artifact that no longer names them. The retired
    # mean-readout profiles are still in plots/tsfm_layers/layer_sweep_32.json.
    #
    # READ AT THE PREDICTION TOKEN, like every other prediction number on
    # these figures and like the random-init floor that gets subtracted from
    # them (xs_ic_eval.PREDICT_POOL = "last"). Until 2026-09-11 the artifact
    # was a mean-pooled sweep minus a last-token floor — two readouts in one
    # subtraction — and fixing it moved Chronos-2's raw spread change from
    # +0.173 to +0.217. build_tsfm_ic.py verifies the payload field.
    #
    # POINTS, NOT LINES. scripts/eval/tsfm_layer_ic.py takes a --horizon and
    # the sweep ran at 900 only, so these series exist at one x and are drawn
    # as a marker with an SE bar, like ``supervised_head``. A line here would
    # be four fifths interpolation.
    #
    # ``point_dx`` dodges the three families around h=900 so their markers do
    # not stack: on the Return panel Chronos-2 and TimesFM 3.0 sit 0.0013
    # apart, a quarter of that panel's whole range, and at one x the later
    # series would simply hide the earlier one. The
    # dodge is cosmetic and small (+-8% of a decade, against the 50% gap to
    # the next tick), log-symmetric about the tick, and the figure caption
    # says the horizon is one value.
    #
    # From ``tsfm_ic.json`` (build_tsfm_ic.py) rather than a ckpt_glob:
    # a frozen TSFM has no checkpoint in the tree and no xs_ic.json beside
    # one — it is scored by a standalone pass that streams sufficient
    # statistics instead of ever materializing a d=9*d_model embedding.
    "tsfm_chronos2": {
        "json_ic": "tsfm_ic.json", "json_ic_model": "chronos2",
        **SERIES_STYLES["chronos2"], "marker": "o", "markersize": 6,
        "markeredgecolor": "black", "markeredgewidth": 0.4,
        "point_only_h": 900, "point_dx": 0.925,
    },
    "tsfm_timesfm3": {
        "json_ic": "tsfm_ic.json", "json_ic_model": "timesfm3",
        **SERIES_STYLES["timesfm3"], "marker": "s", "markersize": 5.5,
        "markeredgecolor": "black", "markeredgewidth": 0.4,
        "point_only_h": 900, "point_dx": 1.000,
    },
    "tsfm_kronos": {
        "json_ic": "tsfm_ic.json", "json_ic_model": "kronos",
        **SERIES_STYLES["kronos"], "marker": "D", "markersize": 5,
        "markeredgecolor": "black", "markeredgewidth": 0.4,
        "point_only_h": 900, "point_dx": 1.081,
    },
    # Per-panel on-task supervised specialist, trained at h=900 and probed at
    # all six horizons; gray so no family baseline has to share its hue. The
    # vision figures keep their per-task supervised colors instead.
    #
    # NOT THE CROSS-ENTROPY SPECIALISTS ANY MORE. This comment described the
    # three flat ``supervised-full-month-<task>-ce`` projects until 2026-08-30,
    # long after they stopped feeding the line: those runs moved to
    # lab/models-archive on 2026-08-29 and build_supervised_ic.py
    # defaulted to ``--source w8`` the same day. The checked-in artifact
    # records which it is -- every record in supervised_probe_ic.json carries
    # ``source: w8`` -- so CHECK THE FILE rather than this comment. What the
    # line actually draws is the W=8 pairwise arm per target
    # (``<task>_pairwise_rep32nost_lr-3e-5_bs256_s42``), the same
    # runs ``sup_return_w8`` and its two siblings draw.
    #
    # Still a json_ic artifact rather than a ckpt_glob, though the w8 runs do
    # carry the month in the project suffix: the record shape was kept when
    # the source moved so SERIES_DEFS needed no new loader. One series covers
    # all three panels -- the file carries every target, and each panel takes
    # its own.
    "supervised_specialist": {
        "color": "tab:gray", "label": "Supervised",
        "json_ic": "supervised_probe_ic.json",
        "json_ic_model": "supervised_probe",
        "marker": SERIES_MARKER,
    },
    # The same specialist read out through its OWN trained head instead of a
    # fresh ridge probe -- the gray star every family figure used to carry,
    # restored on rank IC (it went out with the AUC path on 2026-08-19).
    #
    # ONE POINT, NOT A LINE. A supervised head is trained at a single horizon,
    # so the scorer writes ``xs_ic/head:<task>`` for h=900 and nothing else,
    # and ``point_only_h`` draws it where it exists rather than letting a
    # one-horizon series look like a line that fell off the axis. Gray and
    # star-marked so it reads as the same model as the "Supervised" line
    # below it: same encoder, same months, different readout.
    #
    # Built by ``build_supervised_ic.py --readout head`` off the same W=8
    # pairwise runs the probe artifact comes from, so the two differ ONLY in
    # the readout.
    #
    # NO LEGEND KEY (``no_legend``), ON ANY FIGURE. This is a project-wide
    # standard, not a per-figure call: the head star is never keyed anywhere
    # it is drawn, including the talk cut. A key would spend a legend slot --
    # and on the SSL figure a whole third row -- restating what the caption
    # already says, and a star that means one thing on one figure and carries
    # its own key on the next is worse than either. Every derived entry
    # (SUP_HEAD_TWINS, PRESENTATION_TWINS) inherits the flag rather than
    # re-deciding it; nothing should override it back to False.
    "supervised_head": {
        "color": "tab:gray", "label": "Supervised (head)",
        "json_ic": "supervised_head_ic.json",
        "json_ic_model": "supervised_head",
        "marker": "*", "markersize": 11,
        "markeredgecolor": "black", "markeredgewidth": 0.4,
        "point_only_h": 900, "no_legend": True,
    },
    # Classical finance baselines + view learners, from
    # plots/finance_baselines (models.default_baselines /
    # view_models.default_view_learners). Every predictor they read is a
    # function of the normalized view the encoder is fed, so these lines are
    # comparable with the encoders row for row -- see panel_tables.py.
    # Colors are tab10 in palette order, skipping tab:blue (LeJEPA) and
    # tab:gray (the specialist), because this figure draws cross_stock_k2ind
    # (tab:red) and mean_reversion (tab:orange) too. Two series sharing a hue
    # on a shared legend is not a style question -- the Return panel once had
    # two red lines and no way to tell which was the encoder.
    # json_ic_model selects one model's records out of the shared file;
    # without it a series would collect every model at once.
    #
    # hgb_full moved off tab:olive because Ridge ARDL owns it, and the two were
    # drawn on one legend.
    "ar_p":       {"color": "tab:cyan",   "label": "AR(p)",
                   "json_ic": FIN_IC, "json_ic_model": "ar_p",
                   "marker": SERIES_MARKER},
    "arma_pq":    {"color": "tab:green",  "label": "ARMA(p,q)",
                   "json_ic": FIN_IC, "json_ic_model": "arma_pq",
                   "marker": SERIES_MARKER},
    "ardl":       {"color": "tab:olive",  "label": "Ridge ARDL",
                   "json_ic": FIN_IC, "json_ic_model": "ardl",
                   "marker": SERIES_MARKER},
    "har_rv":     {"color": "tab:purple", "label": "HAR-RV",
                   "json_ic": FIN_IC, "json_ic_model": "har_rv",
                   "marker": SERIES_MARKER},
    "garch11":    {"color": "tab:brown",  "label": "GARCH(1,1)",
                   "json_ic": FIN_IC, "json_ic_model": "garch11",
                   "marker": SERIES_MARKER},
    "ridge_full": {"color": "tab:pink",   "label": "Ridge (view)",
                   "json_ic": FIN_IC, "json_ic_model": "ridge_full",
                   "marker": SERIES_MARKER},
    "hgb_full":   {"color": "#8c6d31",    "label": "GBM (view)",
                   "json_ic": FIN_IC, "json_ic_model": "hgb_full",
                   "marker": SERIES_MARKER},
    # THE TAIL LEARNERS, which are what the finance pass actually writes.
    # ``ridge_full``/``hgb_full`` above read the whole 2048-step view and have
    # never been scored on the forward-VWAP targets: that pass wants a 2.7 GB
    # Gram per censoring shell and a GBM that bins 18,432 features eighteen
    # times, and the 2026-08-30 run skipped it. These read the LAST tokens
    # before the anchor instead -- full resolution, no pool -- which is where
    # the encoder's ``last`` readout sits.
    #
    # A TAIL IS A LOWER BOUND ON THE FULL VIEW and the caption must say so:
    # the encoder's final token attends over all 2048 steps, so beating it
    # with a tail is a stronger result and losing to one proves nothing. See
    # panel_tables.VIEW_TAIL_TOKENS.
    #
    # Three widths because they cost one decode pass between them, not three:
    # 8 is exactly the readout patch (config patch_size), 24 and 64 its
    # neighbourhood. Draw ONE of them beside the classical arms -- the figure
    # command names tail24 -- and keep the other two for the question "how
    # much context does the tail need", which they answer on their own axes.
    "ridge_tail8":  {"color": "#f7b6d2", "label": "Ridge (8-token tail)",
                     "json_ic": FIN_IC, "json_ic_model": "ridge_tail8",
                     "marker": SERIES_MARKER},
    "ridge_tail24": {"color": "tab:pink", "label": "Ridge (24-token tail)",
                     "json_ic": FIN_IC, "json_ic_model": "ridge_tail24",
                     "marker": SERIES_MARKER},
    "ridge_tail64": {"color": "#ad494a", "label": "Ridge (64-token tail)",
                     "json_ic": FIN_IC, "json_ic_model": "ridge_tail64",
                     "marker": SERIES_MARKER},
    "hgb_tail24":   {"color": "#8c6d31", "label": "GBM (24-token tail)",
                     "json_ic": FIN_IC, "json_ic_model": "hgb_tail24",
                     "marker": SERIES_MARKER},
}


# ── each supervised arm, read through its TRAINED HEAD ─────────────────────
#
# The star on all_metrics_supervised_ic: the same checkpoints as the four
# lines above, scored through the head they were trained with instead of a
# fresh ridge probe. Declared off the arm rather than written out, so a pin
# added to the line can never fail to reach its star.
#
# ONE POINT, ON ITS OWN PANEL. A head is trained at one target and one
# horizon, so ``sup_return_w8_head`` has a number only in the Return panel at
# h=900 and is simply absent everywhere else -- which is why the star needs no
# per-panel wiring, unlike the AUC-era HEAD_OVERLAYS table it replaces.
#
# NO MULTIHEAD TWIN, DELIBERATELY. Every ``head:*`` number sitting beside a
# supervised-multihead-* checkpoint today is NOISE FROM AN UNTRAINED HEAD:
# multihead runs save ``heads.pt`` and their config carries ``tasks``, and
# until 2026-08-29 eval/checkpoints.load_model knew neither -- it rebuilt them
# as single-task SupervisedModels on a ``return_900`` default, found no
# head.pt, randomly initialized the head, and let the scorer write it out.
# That is what put a multihead star BELOW the zero line on the Return panel.
#
# RESTORED 2026-08-31, and it is now safe to DECLARE before the data exists.
# ICRun.head refuses a multihead head that lacks ``xs_head_schema``, so an
# un-re-scored trunk contributes nothing rather than noise, and the
# ``expect_months`` below turns "nothing" into a loud failure instead of a
# star quietly averaged over a handful of months. Unlike the three
# specialists this one earns a star on ALL THREE panels: a multihead trunk
# has a trained head per task, so it has a number in each.
SUP_HEAD_TWINS = {
    "sup_return_w8_head": ("sup_return_w8", "Return (head)"),
    "sup_vol_w8_head":    ("sup_vol_w8",    "Vol (head)"),
    "sup_spread_w8_head": ("sup_spread_w8", "Spread (head)"),
    "sup_multi_w8_head":  ("sup_multi_w8",  "Multi (head)"),
}
#
# NO LEGEND KEYS either (``no_legend``), for the project-wide reason given
# beside ``supervised_head``: a head star is never keyed. Here it would also
# cost something concrete -- the stars share the color of the arm they are
# drawn on, so three more keys would push the four lines onto a second row.
for _twin, (_arm, _label) in SUP_HEAD_TWINS.items():
    SERIES_DEFS[_twin] = {
        **SERIES_DEFS[_arm], "readout": "head", "label": _label,
        "marker": "*", "markersize": 11,
        "markeredgecolor": "black", "markeredgewidth": 0.4,
        "point_only_h": 900, "no_legend": True,
        # ALL 32 OR NOTHING. A star sits beside lines drawn over the full
        # sweep panel and is read as the same panel; one averaged over a
        # subset is not comparable to the line it is drawn on, and nothing in
        # the rendering would reveal it. See the value-bearing month count in
        # load_ic_metrics.
        # The specialists read the six-month-span wave, whose panel is 31
        # months (SUP_SPAN_NO_SPAN); the multihead still reads SUP_PINS.
        "expect_months": len(SERIES_DEFS[_arm]["months"]),
    }


# ── the TALK cut (all_metrics_presentation_minus_random) ───────────────────
#
# Two lines and the specialist's head star, and nothing else: the standard
# LeJEPA encoder against the on-task supervised specialist. A slide cannot
# carry the eleven-key legend the paper figures do, so this cut exists to be
# projected — same numbers, same panel, fewer things to read.
#
# DERIVED FROM THE PAPER ENTRIES, never re-specified: each one spreads the
# entry it IS and overrides only styling. Spelling the glob and pins out a
# second time is exactly how ``cross_stock_k2ind`` spent two weeks drawing a
# retired generation under the standard model's name -- a talk figure
# disagreeing with the paper figure beside it is worse than no talk figure.
#
# STYLED FOR THE TALK, not for the family figure it comes from. LeJEPA takes
# the reserved paper-wide blue rather than the tab:green it wears in
# all_metrics_augs_ic, where green distinguishes it from four sibling pairings
# that are not on this figure; the specialist keeps its reserved gray. The
# labels are the spoken names rather than the recipe names the paper legend
# uses -- but they still NAME THE ARM: "LeJEPA + Time Warping" is
# ``pair_warp`` and nothing else, and the day this line is repointed the
# label moves with it.
#
# THE TALK LINE IS NOT THE PAPER'S STANDARD ARM. ``cross_stock_k2ind`` (the
# cross-stock same-industry pairing at lambda 0.1) is what
# all_metrics_ic_minus_random draws; this figure draws the time-warp pairing
# instead, chosen 2026-08-30. Both are complete over the same 32 months, and
# at h=900 the warp arm leads on all three targets (return +0.0107 vs
# +0.0094, vol +0.0583 vs +0.0478, spread +0.0423 vs +0.0200) -- so a reader
# holding the slide against the paper figure is looking at two different
# models, not at a rendering difference.
PRESENTATION_TWINS = {
    "pres_lejepa": ("pair_warp",
                    {"color": SERIES_STYLES["lejepa"]["color"],
                     "label": "LeJEPA + Time Warping"}),
    # "(15 Min)" is the TRAINING horizon, not an eval restriction: the line
    # is drawn at all six. A supervised head is trained at one horizon, and
    # supervised_head_ic.json carries only ``*_900`` records, which is what
    # makes this label a fact rather than a description. It is also why the
    # gray line peaks near the 15-minute tick and decays away from it -- the
    # point of showing it beside a general encoder.
    "pres_supervised": ("supervised_specialist",
                        {"color": "tab:gray",
                         "label": "Supervised Specialist (15 Min)"}),
    # The same specialist read through its OWN trained head instead of a
    # fresh ridge probe: gray star at h=900, the one horizon a head is
    # trained at, inheriting ``point_only_h``, the star and ``no_legend``
    # from ``supervised_head`` so it needs no per-figure wiring.
    #
    # NOT KEYED -- see the standard beside ``supervised_head``. This entry
    # briefly overrode ``no_legend`` to False on the theory that a slide has
    # no caption to explain a star; it is the same star as every other
    # figure's and gets no key here either.
    "pres_supervised_head": ("supervised_head", {"color": "tab:gray"}),
}
for _key, (_src, _overrides) in PRESENTATION_TWINS.items():
    SERIES_DEFS[_key] = {**SERIES_DEFS[_src], **_overrides}


def _select_runs(spec: dict) -> list:
    """The runs one ``ckpt_glob`` series is made of, after every pin.

    Each pin exists because some OTHER checkpoint on disk answers to the same
    glob. Skipping one does not raise — it silently widens the series:

      ``ckpt_root``   WHICH TREE. A retired-but-cited generation lives in
                      lab/models-archive, where a glob against the live
                      checkpoint root cannot reach it.
      ``xs_stats``    the TARGET GENERATION. All three targets became
                      forward-window differences on 2026-08-22, and a run
                      scored against the retired midpoint tables is a
                      different quantity under the same key.
      ``lamb`` /      the arm. In the pairing-lambda sweeps the run NAME
      ``run_name_re`` carries arm and lambda but not the recipe or the month.
      ``months``      the PANEL. The 32-month runs of a sweep's chosen arm
                      carry the same run_name as its holdout counterparts and
                      live under the same project glob, so without this the
                      optimization months join the reported series.
      ``meta_pins``   the recipe, read out of the config rather than the name:
                      ``{dotted path in train_meta: required value}``. A
                      pre-rename run can match name, lambda and month while
                      carrying a different encoder or target.
    """
    runs = iter_ic_runs(spec["ckpt_glob"], spec.get("ckpt_root", CKPT_ROOT),
                        xs_stats=spec.get("xs_stats"))
    lamb = spec.get("lamb")
    if lamb is not None:
        runs = [r for r in runs if r.meta.get("lamb") == lamb]
    rx = spec.get("run_name_re")
    if rx is not None:
        runs = [r for r in runs if rx.match(r.run_name)]
    months = spec.get("months")
    if months is not None:
        runs = [r for r in runs if r.train_month in months]
    for path, want in (spec.get("meta_pins") or {}).items():
        runs = [r for r in runs if _get_dotted(r.meta, path) == want]
    return runs


def load_json_ic(series_key: str, task: str) -> dict[str, float]:
    """``{eval_month: rank IC}`` for one ``json_ic`` series and one task.

    load_ic_metrics reads the same artifacts for the figure, but it returns
    lists with the months stripped off. A summary table needs the months back:
    it differences against the random-init floor month by month and restricts
    to the same shared panel the lines are drawn over. Returns ``{}`` when the
    artifact has not been built, which is how the figure treats it too.
    """
    spec = SERIES_DEFS[series_key]
    path = Path(__file__).resolve().parent / spec["json_ic"]
    if not path.is_file():
        return {}
    model = spec.get("json_ic_model")
    return {r["eval_month"]: r["ic"] for r in json.loads(path.read_text())
            if r["target"] == task
            and (model is None or r.get("model") == model)}


def load_ic_metrics(
    series_keys: list[str], absolute: bool = False,
    fair_months: bool = False,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, tuple[int, int]]]:
    """``{series: {"<type>_<h>": [delta IC per eval month]}}`` from xs_ic.json.

    This needs no producer of its own: post_train_ic_eval fits the ridge probe
    for ALL 18 (type, horizon) columns on every checkpoint it scores, so the
    breadth these figures want is already sitting beside each checkpoint — one
    value per eval month, minus the random-init probe on that same month.

    A series joins by adding ``ckpt_glob`` to its SERIES_DEFS entry, plus
    whichever pins :func:`_select_runs` needs to cut that glob down to one
    arm — or, for something not scored from a checkpoint at all, by pointing
    ``json_ic`` at a per-month artifact another script writes.

    ``absolute`` reports the raw IC instead of the difference, and turns on any
    ``random_init_baseline`` series — the untrained floor drawn as its own
    line. That line is restricted to the months the checkpoint series actually
    contributed, so the two are averaged over the SAME panel of months; the
    baseline file covers 46 months and a checkpoint series usually covers far
    fewer, and averaging them over different months would make the gap partly
    a difference in which months were sampled.

    ``fair_months`` extends that restriction to the checkpoint series
    themselves: every one is cut to the INTERSECTION of their eval months, so
    a comparison between two lines is a comparison between two models rather
    than partly between two panels. Pass it whenever a figure draws several
    checkpoint series that are still filling in — a sweep mid-flight otherwise
    contributes only the months it happens to have finished, and months differ
    enough here to reorder arms on their own (a 32-month series and a
    15-month one differ by more than most of the effects being compared).
    """
    bases: dict[str, dict[str, float]] = {}

    def base(task: str) -> dict[str, float]:
        # One disk scan per target, not per (series, run, target).
        if task not in bases:
            bases[task] = load_randinit_ic(task)
        return bases[task]

    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    coverage: dict[str, tuple[int, int]] = {}
    months_used: set[str] = set()
    ckpt_keys = [k for k in series_keys if SERIES_DEFS[k].get("ckpt_glob")]
    runs_by_key = {k: _select_runs(SERIES_DEFS[k]) for k in ckpt_keys}

    # The fair panel is computed BEFORE anything is accumulated: a month
    # counts only if it carries a scored run of every checkpoint series on
    # the figure, and every series is then cut to it.
    panel: set[str] | None = None
    if fair_months and runs_by_key:
        panel = set.intersection(*(
            {r.eval_month for r in rs} for rs in runs_by_key.values()
        ))
        own = {k: len({r.eval_month for r in rs}) for k, rs in runs_by_key.items()}
        print(f"  [ic] fair panel: {len(panel)} eval months shared by all "
              f"{len(runs_by_key)} checkpoint series (own coverage: "
              + ", ".join(f"{k}={n}" for k, n in own.items()) + ")")
        if not panel:
            raise SystemExit(
                "no month is covered by every series — drop a series or pass "
                "own-months scoring and accept that the lines average "
                "different panels"
            )

    for sk in ckpt_keys:
        spec = SERIES_DEFS[sk]
        runs = runs_by_key[sk]
        if panel is not None:
            runs = [r for r in runs if r.eval_month in panel]
        # Collected per MONTH, not per run. A month can carry two scored runs
        # of one arm -- a resubmission lands under a new commit-hash project
        # while the first is still on disk, and the same glob matches both --
        # and a per-run list would then weight that month twice in the mean
        # and twice again in the SE band. One month is one observation, so
        # duplicates are averaged and named out loud: which of the two is
        # wanted is a question about the runs, not about this figure.
        by_month: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list))
        # ``readout``: which number beside the checkpoint the series IS. The
        # ridge probe exists for all 18 (type, horizon) columns; a trained
        # head speaks only to its own target at its own horizon (see
        # ``name == head_task`` in mass_eval/xs_ic_eval.py), so a head series
        # simply finds nothing in the other 17 cells and draws one point.
        readout = spec.get("readout", "probe")
        for r in runs:
            for t in TARGET_TYPES:
                for h in HORIZONS:
                    task = f"{t}_{h}"
                    v = r.probe(task) if readout == "probe" else r.head(task)
                    if v is None:
                        continue
                    # THE FLOOR IS REQUIRED ONLY WHEN IT IS SUBTRACTED.
                    # ``b is None`` used to skip the cell in BOTH modes, which
                    # coupled the absolute figure to data it does not use: with
                    # lab/score_results empty (no randinit-fwd3-* scores)
                    # every cell was dropped and --absolute rendered nothing at
                    # all, for probe and head alike, even though raw IC needs no
                    # baseline. Under --absolute the month now stands on its own
                    # value; the delta path is unchanged and still requires the
                    # floor, so a differenced figure can never quietly become a
                    # raw one.
                    b = base(task).get(r.eval_month)
                    if absolute:
                        by_month[r.eval_month][task].append(v)
                    elif b is not None:
                        by_month[r.eval_month][task].append(v - b)
        dupes = sorted(m for m, tv in by_month.items()
                       if any(len(vs) > 1 for vs in tv.values()))
        if dupes:
            print(f"  WARN: {sk} has more than one scored run in "
                  f"{len(dupes)} month(s) — averaged: {' '.join(dupes)}")
            for m in dupes:
                for r in runs:
                    if r.eval_month == m:
                        print(f"        {m}  {r.project}/{r.run_id}")
        months = set(by_month)
        for m, tv in by_month.items():
            for task, vs in tv.items():
                out[sk][task].append(sum(vs) / len(vs))
        # A series may declare how many eval months it MUST have. Short is
        # not an exception anywhere else in this function -- it is a figure
        # that renders cleanly over the wrong panel, which is exactly how
        # the lambda sweep spent a while impersonating the standard model.
        # Under a fair panel the check is on the series' OWN coverage: the
        # intersection is allowed to be short, the series is not.
        #
        # COUNTED ON MONTHS THAT PRODUCED A VALUE, not on months a checkpoint
        # merely exists for. The two differ exactly when a number is missing
        # or refused beside a checkpoint that is otherwise a fine match --
        # which is the live case for a HEAD series, where ICRun.head returns
        # None for a multihead trunk that has not been re-scored since the
        # 2026-08-29 loader fix. Counting matches would let a star be drawn as
        # the mean of whatever subset happened to be re-scored, on a figure
        # whose lines use all 32, with nothing on the page to say so.
        expect = spec.get("expect_months")
        own_months = {r.eval_month for r in runs_by_key[sk]}
        if expect is not None and len(months) < expect:
            raise SystemExit(
                f"series {sk!r} produced a value for {len(months)} of "
                f"{expect} eval months under {spec['ckpt_glob']!r} "
                f"({len(own_months)} checkpoint(s) matched).\n"
                f"With a value: {' '.join(sorted(months)) or '(none)'}\n"
                f"Missing:      {' '.join(sorted(own_months - months)) or '(none)'}"
            )
        months_used |= months
        coverage[sk] = (len(runs), len(months))

    # Months the json_ic series bring, unioned into ``months_used`` only
    # AFTER the loop below. Accumulating inside it would make the restriction
    # order-dependent: the second artifact series would be cut to the first
    # one's months for no reason, since the restriction exists to match the
    # CHECKPOINT panel, not to match a sibling artifact.
    json_months: set[str] = set()
    for sk in series_keys:
        src = SERIES_DEFS[sk].get("json_ic")
        if not src:
            continue
        path = Path(__file__).resolve().parent / src
        if not path.is_file():
            print(f"  [ic] {sk} skipped: {src} not built yet "
                  f"(uv run {JSON_IC_PRODUCERS.get(src, str(path))})")
            continue
        recs = json.loads(path.read_text())
        # One file can hold many models; a series takes exactly its own.
        model = SERIES_DEFS[sk].get("json_ic_model")
        if model is not None:
            recs = [r for r in recs if r.get("model") == model]
        # Same month restriction as the random-init floor, and for the same
        # reason: a line averaged over different months is not comparable.
        keep = [r for r in recs
                if not months_used or r["eval_month"] in months_used]
        for r in keep:
            b = base(r["target"]).get(r["eval_month"])
            if b is None and not absolute:
                continue
            out[sk][r["target"]].append(r["ic"] if absolute else r["ic"] - b)
        coverage[sk] = (len(keep), len({r["eval_month"] for r in keep}))
        json_months |= {r["eval_month"] for r in keep}

    # An artifact series is a drawn panel too. Until 2026-09-12 only the
    # checkpoint series fed ``months_used``, so the floor below had nothing to
    # borrow on a figure made entirely of json_ic arms and --absolute exited
    # rather than drawing -- which is where tsfm_ic.py landed the moment its
    # one checkpoint reference (LeJEPA, retired to models-archive) dropped
    # out, with three complete frozen-TSFM arms and a supervised specialist
    # still on the page. The union is deliberate: the floor must cover every
    # month any drawn series contributed, or the line it draws is over a
    # narrower panel than the series it is being compared to.
    months_used |= json_months

    for sk in series_keys:
        if not SERIES_DEFS[sk].get("random_init_baseline"):
            continue
        if not absolute:
            print(f"  [ic] {sk} skipped: it is the subtrahend, so under "
                  f"minus-random it is identically zero")
            continue
        if not months_used:
            raise SystemExit(
                f"{sk} has no months to be drawn over — it borrows them from "
                f"the other series on the figure, so ask for at least one "
                f"series that has data"
            )
        for t in TARGET_TYPES:
            for h in HORIZONS:
                task = f"{t}_{h}"
                b = base(task)
                out[sk][task] = [b[m] for m in sorted(months_used) if m in b]
        coverage[sk] = (len(months_used), len(months_used))
    return out, coverage


def legend_keys(series_keys: list[str]) -> list[str]:
    """The drawn series that actually take a legend slot.

    A series with ``no_legend`` is drawn but never keyed — the supervised
    head stars, which the captions explain. Column counts and the bottom
    reserve must be sized off THIS list, not off the drawn one: counting the
    silent series is how the SSL figure paid for a third legend row that had
    nothing in it.
    """
    return [k for k in series_keys if not SERIES_DEFS[k].get("no_legend")]


def legend_bottom_reserve(n_labels: int, ncol: int) -> float | None:
    """Bottom space a legend of ``n_labels`` in ``ncol`` columns needs.

    ``add_bottom_legend`` reserves for one row, or for two, and stops there:
    a THIRD row lands on top of the shared x-label. Widening the legend is not
    the fix — ``bbox_inches="tight"`` grows the SAVED figure to fit a legend
    wider than the axes, so every panel shrinks once the figure is placed at
    text width (a sixth SSL column cost 9%). Add a row instead, and pay for it
    here. Returns None when the default already clears the legend.
    """
    # NOTHING TO KEY. A stars-only cut draws no legend at all (every head
    # star is ``no_legend``), so there is no row to reserve for -- and the
    # ncol=0 the caller derives from it would divide by zero here.
    if n_labels < 1 or ncol < 1:
        return None
    rows = -(-n_labels // ncol)
    if rows <= 2:
        return None
    per_row = LEGEND_BOTTOM_RESERVE_2ROW - LEGEND_BOTTOM_RESERVE_1ROW
    return LEGEND_BOTTOM_RESERVE_2ROW + (rows - 2) * per_row


def _marker_kw(spec: dict) -> dict:
    """Marker size / edge a series declares, defaulted to the line style.

    Only the point-only overlays set these: a star at the size of a line
    marker is unreadable, and without an edge it dissolves into the line it
    sits on. A series that says nothing renders exactly as before.
    """
    kw = {"markersize": spec.get("markersize", 4)}
    for k in ("markeredgecolor", "markeredgewidth"):
        if k in spec:
            kw[k] = spec[k]
    return kw


def plot_all_metrics(
    series_data: dict[str, dict[str, list[float]]],
    outdir: Path, *, baseline_subtracted: bool,
    series_keys: list[str],
    out_name: str,
    legend_ncol: int,
    figsize: tuple[float, float] = (WIDTH_FULL, 3.0),
    bottom_reserve: float | None = None,
    panel_notes: dict[str, str] | None = None,
    sharey: bool = True,
    point_only_series: dict[str, int] | None = None,
    annotate_values: bool = False,
) -> None:
    """Three panels (one per target type) x six forward horizons.

    ``baseline_subtracted`` says which quantity ``series_data`` holds: the
    per-month difference against the random-init encoder, or the raw rank IC.
    It only picks the y-label and the zero line — the subtraction happens in
    load_ic_metrics, one month at a time, because a mean-of-differences and a
    difference-of-means are not the same number when the two sides cover
    different months.

    ``annotate_values`` prints each drawn point's value ± SE beside it. Off by
    default and meant for a READING figure -- a cut with one or two markers per
    panel, where the number is the whole point -- not for the family figures,
    where six horizons x five series of text would bury the lines.
    """
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=sharey)

    # A series that exists at ONE horizon says so in SERIES_DEFS
    # (``point_only_h``) rather than at each of the four call sites, so every
    # figure that draws it gets the overlay without opting in. The explicit
    # argument still wins, for a figure that wants a normal series pinned to
    # one horizon.
    point_only = {k: SERIES_DEFS[k]["point_only_h"] for k in series_keys
                  if SERIES_DEFS[k].get("point_only_h") is not None}
    point_only.update(point_only_series or {})
    for ax, t in zip(axes, TARGET_TYPES):
        deferred = []
        for sk in series_keys:
            s = SERIES_DEFS[sk]
            # ``point_only_series`` maps a series to the ONE horizon it is
            # drawn at: a single marker (+ SE bar) instead of a line, held
            # back to `deferred` so it renders last, on top of every other
            # series and of the head star. Its legend key is drawn here (as
            # marker-only, matching what's rendered) so legend ordering is
            # unaffected — unless the series is ``no_legend``, in which case
            # nothing is keyed at all.
            only_h = point_only.get(sk)
            # A DODGE, applied to the drawn x only. Several point-only series
            # at the same horizon land on the same pixel, and one hides the
            # rest whenever their values are close. Multiplicative because the
            # axis is log, so a factor is a constant visual offset wherever it
            # is applied; only point-only series may set it, since nudging a
            # LINE would misstate the horizon it was measured at.
            dx = s.get("point_dx", 1.0) if only_h is not None else 1.0
            xs, ys, ses = [], [], []
            for h in HORIZONS if only_h is None else [only_h]:
                vals = np.asarray(
                    series_data.get(sk, {}).get(f"{t}_{h}", []), dtype=float)
                if vals.size == 0:
                    continue
                xs.append(h * dx / 60.0)
                ys.append(float(vals.mean()))
                ses.append(
                    float(vals.std(ddof=1) / np.sqrt(vals.size))
                    if vals.size > 1 else 0.0
                )
            if not xs:
                continue
            xs_a = np.asarray(xs)
            ys_a = np.asarray(ys)
            ses_a = np.asarray(ses)
            label = "_nolegend_" if s.get("no_legend") else s["label"]
            if only_h is not None:
                ax.plot(
                    [], [], marker=s["marker"], color=s["color"],
                    linestyle="none", label=label, **_marker_kw(s),
                )
                deferred.append((xs_a, ys_a, ses_a, s))
                continue
            ax.plot(
                xs_a, ys_a, marker=s["marker"], color=s["color"],
                linestyle=s.get("linestyle", "-"),
                linewidth=1.5, markersize=4, label=label,
            )
            ax.fill_between(
                xs_a, ys_a - ses_a, ys_a + ses_a,
                color=s["color"], alpha=0.15, linewidth=0,
            )

        value_notes: list[tuple[str, str]] = []
        for xs_a, ys_a, ses_a, s in deferred:
            ax.errorbar(
                xs_a, ys_a, yerr=ses_a,
                fmt=s["marker"], color=s["color"],
                ecolor=s["color"], elinewidth=1.0, capsize=2,
                zorder=7, **_marker_kw(s),
            )
            if annotate_values:
                value_notes.extend((f"{y:+.4f} ± {se:.4f}", s["color"])
                                   for y, se in zip(ys_a, ses_a))
        # HEADROOM FIRST. The notes sit at the top of the panel in axes
        # coordinates, so without this they land on the upper cap of the very
        # error bar they describe.
        if value_notes:
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + (hi - lo) * (0.10 + 0.09 * len(value_notes)))
        # IN AXES COORDINATES, STACKED AT THE TOP. Anchored to the marker the
        # text ran off the right edge on two of the three panels -- these
        # panels are ~1.7in wide and "+0.0274 ± 0.0033" is most of that -- and
        # bbox_inches="tight" would have paid for it by shrinking every panel.
        for i, (txt, color) in enumerate(value_notes):
            ax.annotate(
                txt, xy=(0.5, 0.95 - 0.09 * i), xycoords="axes fraction",
                ha="center", va="top", fontsize=7, color=color, zorder=8,
            )
        if panel_notes and t in panel_notes:
            ax.annotate(
                panel_notes[t], xy=(0.97, 0.03), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=7, color="0.35",
            )
        ax.set_title(WM_PANEL_TITLES[t])
        ax.set_xlabel("")
        ax.set_xscale("log")
        ax.set_xticks([h / 60.0 for h in HORIZONS])
        ax.set_xticklabels([
            "" if int(h / 60) == 10 else str(int(h / 60)) for h in HORIZONS
        ])
        ax.minorticks_off()
        ax.grid(False)
        # NOT ``locator_params(axis="y", nbins=4)``, which is free to pick a
        # 2.5- or 0.005-step and did: every figure this function draws is a
        # rank IC, so the interesting range is a few hundredths and the plain
        # ladder kept labeling it 0.000/0.025/0.050/0.075. Three decimals of
        # axis furniture on a two-inch panel, on all four figures.
        #
        # FIVE bins, not the four that call asked for. The coarse ladder can
        # only step 0.01/0.02/0.05, so a strict four-interval budget rounds a
        # 0.09-tall panel all the way up to a 0.05 step and labels it 0.00 and
        # 0.05 -- two ticks. Five buys the 0.02 step back on exactly those
        # panels and changes nothing on the others.
        set_two_decimal_yticks(ax, nbins=5)
        if baseline_subtracted:
            ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)

    axes[0].set_ylabel(IC_METRIC_LABEL if baseline_subtracted else "Rank IC")
    for ax in axes[1:]:
        ax.set_ylabel("")
    axes[1].set_xlabel("Forward Horizon (Minutes)")

    fig.tight_layout()
    # Tighter inter-entry spacing than the mpl default (2.0) so the wide
    # single-row legend (up to 5 entries) isn't oversized for the panels.
    # A FIGURE MAY KEY NOTHING. Every head star is ``no_legend`` (see the
    # standard beside ``supervised_head``), so a stars-only cut has zero
    # entries -- and ``ncol=0`` is not a legend with nothing in it, it is a
    # ZeroDivisionError inside mpl's legend layout. Nothing to key means no
    # legend and no bottom reserve; the panel titles already name the task.
    if legend_ncol < 1:
        pass
    elif bottom_reserve is None:
        add_bottom_legend(fig, ncol=legend_ncol, columnspacing=1.3)
    else:
        add_bottom_legend(
            fig, ncol=legend_ncol, bottom_reserve=bottom_reserve,
            columnspacing=1.3,
        )

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / (f"{out_name}_minus_random" if baseline_subtracted else out_name)
    written = save_figure(fig, out)
    print(f"Saved {', '.join(str(p) for p in written)}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default=str(OUT_DIR))
    p.add_argument("--absolute", action="store_true",
                   help="Plot absolute rank IC instead of the per-month "
                        "difference against a random-init encoder. Turns on "
                        "any random_init_baseline series, which is dropped "
                        "from the difference figure because there it is "
                        "identically zero.")
    p.add_argument("--series", nargs="+",
                   default=["cross_stock_k2ind", "randinit_vit",
                            "mean_reversion"],
                   help="SERIES_DEFS keys to draw, in legend order.")
    p.add_argument("--fair-months", action="store_true",
                   help="Cut every checkpoint series to the months they ALL "
                        "cover, so a comparison between two lines is not "
                        "partly a comparison between two panels of months. "
                        "Worth passing whenever a drawn sweep is still "
                        "filling in.")
    p.add_argument("--annotate-values", action="store_true",
                   help="Print each point-only marker's value ± SE beside it. "
                        "For a reading cut (one or two markers a panel); on a "
                        "family figure the text buries the lines.")
    p.add_argument("--out-name", default="all_metrics_ic",
                   help="stem for the figure. A DIFFERENT series set (the "
                        "finance baselines, say) must pass its own or it "
                        "silently overwrites the standard figure.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    unknown = [k for k in args.series if k not in SERIES_DEFS]
    if unknown:
        raise SystemExit(
            f"not in SERIES_DEFS: {unknown}\nknown: {' '.join(SERIES_DEFS)}")

    data, coverage = load_ic_metrics(
        args.series, absolute=args.absolute, fair_months=args.fair_months,
    )
    for sk, (n_runs, n_months) in coverage.items():
        print(f"  [ic] {sk}: {n_runs} runs over {n_months} eval months")
    if not any(data.values()):
        raise SystemExit(
            "no IC data for " + ", ".join(args.series)
            + " — a checkpoint series needs a ckpt_glob and scored "
              "checkpoints with an xs_ic.json; a json_ic series needs its "
              "artifact built"
        )
    drawn = [k for k in args.series if data.get(k)]
    for sk in drawn:
        print(f"  [ic] {sk} counts:")
        for t in TARGET_TYPES:
            for h in HORIZONS:
                tk = f"{t}_{h}"
                print(f"    {tk:24s} n={len(data.get(sk, {}).get(tk, []))}")

    # The two family scripts pay for their own legend rows; this entry point
    # did not, so a series list long enough for a THIRD row (the finance
    # suite, at eleven-plus entries) drew that row straight over the shared
    # x-label. legend_bottom_reserve returns None for one or two rows, which
    # is the default add_bottom_legend already applies.
    n_keys = len(legend_keys(drawn))
    ncol = min(n_keys, 4)
    plot_all_metrics(
        data, Path(args.outdir), baseline_subtracted=not args.absolute,
        series_keys=drawn, out_name=args.out_name,
        legend_ncol=ncol, bottom_reserve=legend_bottom_reserve(n_keys, ncol),
        sharey=False, annotate_values=args.annotate_values,
    )


if __name__ == "__main__":
    main()
