"""Month-pooled P-portfolio / S-stock nearest-neighbor industry test,
swept over the 32 canonical sweep months.

For every trading day of each month's eval month (train+1): draw
``--n_portfolios`` random FF49 industries and ``--n_stocks`` random member
stocks of each (fresh draw every day, members must be present in the
mosaic that day) and take ONE random global view per stock.

PRIMARY metric (month-pooled): pool the month's ~n_days*P*S points and
score the fraction whose nearest neighbor AMONG OTHER STOCKS' points
(own-stock views excluded, so identity islands cannot score) is in the
same industry — against the empirical same-industry share of each
point's pool (~3% at P=3/S=2). This rewards persistent firm-level
industry structure, including matching the partner on a different day.

Secondary (within-day): NN among the same day's other P*S-1 points only,
chance (S-1)/(P*S-1) = 20% at 3x2 — industry proximity given the same
market moment.

Models per month: MODEL_ORDER, the 18 reported IC-era encoders (five
LeJEPA pairings, four supervised heads, nine SSL baselines) plus the
random-init floor. Those resolve per EVAL month through the noclamp
manifest; the retired pre-IC entries in LEGACY_GLOB_KEYS resolve by
TRAINING month from checkpoint train_meta.json instead.

Prints a per-model table (pooled rate, chance, t vs chance across month
means) and writes industry_nn_sweep_P{p}S{s}.{png,pdf} plus a results
json (for the later portfolio/stock scaling comparison).

Run:
    uv run plots/latent_eval/fixed_panel/industry_nn_sweep.py
    uv run plots/latent_eval/fixed_panel/industry_nn_sweep.py --n_portfolios 6 --n_stocks 4
    uv run plots/latent_eval/fixed_panel/industry_nn_sweep.py --models pair_k2ind_final,sup_return_w8
"""
from __future__ import annotations

import argparse
import calendar
import os
import json
import pickle
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import panel_lib as eg
from stable_finance.dataset import MarketSchedule
from market_jepa.schemas import LocalMachineConfig
from market_jepa.training.streaming_dataset import ensure_ticker_date_sidecar
from style import save_figure

# THE REPORTED PANEL: the 32 canonical sweep months, one definition shared
# with every other cross-month figure (scripts/experiments/sample_sweep_months.py).
# Until 2026-09-04 this was a hardcoded 14-month list — the post-2015 subset
# the pre-IC variant encoders had been swept over. It outlived them: the
# reported latent-eval JSONs (_augsup, _sslic) have been on all 32 since
# 2026-08-30, run by passing MONTHS explicitly, while this default still
# named 14 months of a model set whose checkpoints no longer exist.
#
# TRAINING months; every panel is built on train+1. LATENT_TRAIN_MONTHS
# overrides them so a model family trained off the reported panel (the
# holdout-2 readout grid, say) can be evaluated without --months, which
# glob-resolved keys cannot use -- they need a training month to resolve a
# checkpoint, and --months only names eval months.
MONTHS = os.environ.get("LATENT_TRAIN_MONTHS", "").split() or eg.load_sweep_months()

# data/ moved out of scripts/ in 6150ad1 ("Retire legacy data and evaluation
# workflows"); this path was not updated with it and every latent-eval run
# since has died in pick_panels on a missing parquet.
INDUSTRY_MAP = eg.OUT_DIR.parents[2] / "data" / "industry_map.parquet"

# project_glob is formatted with the month's -{start}-{end} date suffix.
MODEL_SPECS = {
    "k2": {
        "label": "Cross-stock K=2",
        "project_glob": "cross-stock-ind-*{dates}",
        "run_name": "k2",
    },
    "k2ind": {
        "label": "K=2 same-industry",
        "project_glob": "cross-stock-ind-*{dates}",
        "run_name": "k2ind",
    },
    "lejepa": {
        "label": "LeJEPA (standard)",
        "project_glob": "lejepa-lamb-nlocal-cf3df6{dates}",
        "run_name": "lejepa_lamb-0.01_nloc-6",
    },
    "sup": {
        "label": "Supervised (return)",
        "project_glob": "supervised-vit-cf3df6{dates}",
        "run_name": "vit_return_900_k5_lr-1e-4",
    },
    "supmulti": {
        "label": "Supervised (multi)",
        "project_glob": "multi-supervised-vit-cf3df6{dates}",
        "run_name": "vit_multi_lr-1e-4",
    },
    # Same-stock invariance family at their Jan-2023 HPO winners
    # (sweeps/samestock_augs_14mo.sh; same 14-month panel).
    "warp": {
        "label": "Time warp",
        "project_glob": "samestock-augs-*{dates}",
        "run_name": "warp",
    },
    "noise": {
        "label": "Gaussian noise",
        "project_glob": "samestock-augs-*{dates}",
        "run_name": "noise",
    },
    "volj": {
        "label": "Volume noise",
        "project_glob": "samestock-augs-*{dates}",
        "run_name": "volj",
    },
    "pricej": {
        "label": "Price jitter",
        "project_glob": "samestock-augs-*{dates}",
        "run_name": "pricej",
    },
    "chdrop": {
        "label": "Channel drop",
        "project_glob": "samestock-augs-*{dates}",
        "run_name": "chdrop",
    },
    "supvol": {
        "label": "Supervised (vol)",
        "project_glob": "supervised-vit-cf3df6{dates}",
        "run_name": "vit_volatility_change_900_k5_lr-1e-4",
    },
    "supspread": {
        "label": "Supervised (spread)",
        "project_glob": "supervised-vit-cf3df6{dates}",
        "run_name": "vit_spread_change_900_k5_lr-1e-4",
    },
    # Aug-mix 14-month winners (sweeps/aug_mix_14mo.sh) and the
    # 0-local-view RRC ablation (sweeps/lejepa_nolocal_14mo.sh).
    # Not in MODEL_ORDER — opt in via --models.
    "mix2": {
        "label": "Mix RRC25/Ind75",
        "project_glob": "aug-mix-454262{dates}",
        "run_name": "rrc25_ind75",
    },
    "mix4": {
        "label": "Mix4 RRC50",
        "project_glob": "aug-mix-454262{dates}",
        "run_name": "mix4_rrc50",
    },
    "nolocal": {
        "label": "RRC no locals",
        "project_glob": "lejepa-nolocal-536904{dates}",
        "run_name": "lejepa_lamb-0.01_nloc-0",
    },
    # SSL baselines (the backtesting-manifest winners; run_name is stable
    # across months). Not in MODEL_ORDER — opt in via --models.
    "ijepa": {
        "label": "I-JEPA",
        "project_glob": "ijepa-final-cf3df6{dates}",
        "run_name": "pythia-lr-5e-4-scale-0.05-0.10",
    },
    "mae": {
        "label": "MAE",
        "project_glob": "mae-lr-patch-mask-cf3df6{dates}",
        "run_name": "pythia-p-8",
    },
    # TS-SSL series (appendix): resolved from
    # plots/metrics/noclamp_manifest.json by (series_key, eval month) and
    # loaded mode-aware — see fixed_panel_metrics. NOT in MODEL_ORDER: these
    # are the pre-re-optimization HPO winners, superseded by the *_final rows
    # below. Opt in via --models / --series.
    # The frozen TSFMs are NOT here; they are generated below, one entry per
    # layer, because a TSFM has no checkpoint to resolve.
    "dino": {"label": "DINO", "manifest_series": "dino_ema0.99"},
    "byol": {"label": "BYOL", "manifest_series": "byol_ema0.996"},
    "cpc": {"label": "CPC", "manifest_series": "cpc_temp0.1"},
    "ts2vec_cb028b": {"label": "TS2Vec", "manifest_series": "ts2vec_cb028b"},
    "cost_4c05e0": {"label": "CoST", "manifest_series": "cost_4c05e0"},
    "tfc_4c05e0": {"label": "TF-C", "manifest_series": "tfc_4c05e0"},
    "timemae_4c05e0": {"label": "TimeMAE", "manifest_series": "timemae_4c05e0"},
    # The IC-era re-optimized SSL baselines (sweeps/ssl_ic/, 2026-08): the
    # final 32-month series at each method's holdout-set-2 winner. Resolved
    # per EVAL month through the manifest like the rows above; the final32
    # bundle job writes a job-local manifest (MJ_NOCLAMP_MANIFEST) for the
    # checkpoints it just trained, so the latent stages run in-job. These are
    # in MODEL_ORDER — nine of the 18 reported encoders.
    "byol_final": {"label": "BYOL", "manifest_series": "byol_final"},
    "cost_final": {"label": "CoST", "manifest_series": "cost_final"},
    "cpc_final": {"label": "CPC", "manifest_series": "cpc_final"},
    "dino_final": {"label": "DINO", "manifest_series": "dino_final"},
    "ijepa_final": {"label": "I-JEPA", "manifest_series": "ijepa_final"},
    "mae_final": {"label": "MAE", "manifest_series": "mae_final"},
    "tfc_final": {"label": "TF-C", "manifest_series": "tfc_final"},
    "timemae_final": {"label": "TimeMAE", "manifest_series": "timemae_final"},
    "ts2vec_final": {"label": "TS2Vec", "manifest_series": "ts2vec_final"},
    # ── The IC-era LeJEPA PAIRINGS and SUPERVISED heads (2026-08-29) ──────
    #
    # These replace the stale glob entries above (``lejepa``, ``warp``, ``k2``,
    # ``k2ind``, ``sup*``), which still point at the PRE-IC-MIGRATION runs:
    # lejepa_lamb-0.01_nloc-6, samestock-augs-*, cross-stock-ind-*,
    # supervised-vit-cf3df6. Those are a different lambda regime, a different
    # target generation and a different supervised recipe — do not mix them
    # with anything below.
    #
    # MANIFEST-RESOLVED, not glob-resolved, because selecting these correctly
    # needs more than a run_name: each pairing is pinned to ITS OWN lambda
    # (rrc 0.05, time_warp 0.001, k2 0.2, k2ind 0.1 — holdout-set-2 winners,
    # 2026-08-27) AND to a meta pin (xs_target rank)
    # that separate the reported arm from the lambda sweep sharing its name.
    # metrics.SERIES_DEFS[pair_*] holds that logic; the manifest is built from
    # it so there is exactly one definition of what each arm is.
    #
    # gaussian_noise is DELIBERATELY ABSENT: the info token was being noised
    # until 2026-08-29, so every gaussian_noise number predates the fix and
    # the arm needs retraining before it can be compared.
    "pair_rrc_final": {"label": "Same-stock crops", "manifest_series": "pair_rrc_final"},
    "pair_warp_final": {"label": "Crops + time warp", "manifest_series": "pair_warp_final"},
    "pair_k2_final": {"label": "Cross-stock K=2", "manifest_series": "pair_k2_final"},
    "pair_k2ind_final": {"label": "Cross-stock same-ind.", "manifest_series": "pair_k2ind_final"},
    # The four supervised heads, one recipe
    # (pairwise, rep32nost, lr 3e-5, bs 256, seed 42) across all four so the
    # heads are readable against each other. All four cover the core 32 sweep
    # months exactly. NOTE the multihead here is the 32-month PAIRWISE arm
    # (supervised-multihead-*), NOT supervised-full-month-multihead-ce, which
    # is a cross-entropy model over 202 months.
    # gaussian_noise, RETRAINED after the info-token fix (2026-08-29). The
    # original arm noised the information token itself, so every number it
    # produced is on a different model than the one the recipe describes --
    # it is excluded from the pair_* set above for that reason. This is the
    # corrected run (project lejepa-noisefix-lambda-*, lambda 0.3, the same
    # holdout-2 winner), so it is the one to compare with.
    "pair_noise_final": {"label": "LeJEPA +noise (fixed)",
                         "manifest_series": "pair_noise_final"},
    # ── THE SIX-MONTH-SPAN WAVE (2026-09-15) ─────────────────────────────
    #
    # Every SSL and LeJEPA arm retrained on a 6-month span x 12 passes, the
    # budget the supervised specialists already use, so the latent table stops
    # comparing a 1-month SSL encoder with a 6-month supervised one. Resolved
    # per EVAL month through a MACHINE-LOCAL manifest --
    # plots/latent_eval/build_6mo_manifest.py, written beside the caches --
    # because one project per (arm, span) fits no glob in this registry.
    #
    # THE _6mo SUFFIX IS LOAD-BEARING. The *_final and pair_*_final rows above
    # are the SAME METHODS on a ONE-MONTH span; pooling the two into one cell
    # would average two different budgets. They are separate keys for that
    # reason, and the archived ones stay archived.
    #
    # PAIRING, for the three joint-embedding arms: pair_warp_6mo, dino_6mo and
    # byol_6mo all draw their positive pair as two time warps of ONE window
    # (schemas.DINOModeConfig/BYOLModeConfig pin dataset_overrides), so those
    # three differ only in the objective. pair_k2*_6mo take a partner ticker.
    "pair_rrc_6mo": {"label": "Same-stock crops (6mo)", "manifest_series": "pair_rrc_6mo"},
    "pair_warp_6mo": {"label": "Crops + time warp (6mo)", "manifest_series": "pair_warp_6mo"},
    "pair_noise_6mo": {"label": "LeJEPA +noise (6mo)", "manifest_series": "pair_noise_6mo"},
    "pair_k2_6mo": {"label": "Cross-stock K=2 (6mo)", "manifest_series": "pair_k2_6mo"},
    "pair_k2ind_6mo": {"label": "Cross-stock same-ind. (6mo)", "manifest_series": "pair_k2ind_6mo"},
    "byol_6mo": {"label": "BYOL (6mo)", "manifest_series": "byol_6mo"},
    "cost_6mo": {"label": "CoST (6mo)", "manifest_series": "cost_6mo"},
    "cpc_6mo": {"label": "CPC (6mo)", "manifest_series": "cpc_6mo"},
    "dino_6mo": {"label": "DINO (6mo)", "manifest_series": "dino_6mo"},
    "ijepa_6mo": {"label": "I-JEPA (6mo)", "manifest_series": "ijepa_6mo"},
    "mae_6mo": {"label": "MAE (6mo)", "manifest_series": "mae_6mo"},
    "tfc_6mo": {"label": "TF-C (6mo)", "manifest_series": "tfc_6mo"},
    "timemae_6mo": {"label": "TimeMAE (6mo)", "manifest_series": "timemae_6mo"},
    "ts2vec_6mo": {"label": "TS2Vec (6mo)", "manifest_series": "ts2vec_6mo"},
    "sup_return_w8": {"label": "Supervised (return)", "manifest_series": "sup_return_w8"},
    "sup_vol_w8": {"label": "Supervised (vol)", "manifest_series": "sup_vol_w8"},
    "sup_spread_w8": {"label": "Supervised (spread)", "manifest_series": "sup_spread_w8"},
    "sup_multi_w8": {"label": "Supervised (multi)", "manifest_series": "sup_multi_w8"},
    # Random-init ViT (the dAUC baseline model, torch.manual_seed(0)) — no
    # checkpoint; consumers that don't special-case it skip it via the
    # unresolvable glob. fixed_panel_metrics builds it in-process.
    "random": {
        "label": "Random ViT",
        "project_glob": "random-init-vit-no-checkpoint{dates}",
        "run_name": "random",
    },
}

# ── Frozen TSFM layers ────────────────────────────────────────────────────────
#
# One entry per hidden state, generated rather than listed: a PretrainedTSFM is
# fully determined by (family, layer) — there are no trained weights, so there
# is no checkpoint to resolve and no manifest row to look up. Consumers that
# see "tsfm_family" build the model directly and read the layer straight out of
# a multi-layer capture (fixed_panel_metrics), so asking for all 60 of these
# costs four forward passes, not sixty.
#
# Layer 0 is the patch embedding; the last layer carries the model's own final
# norm where the architecture has one.
TSFM_FAMILIES = {
    "chronos2": ("Chronos-2", 12),
    "timesfm": ("TimesFM 2.5", 20),
    # TimesFM 3.0 is the same depth and width as 2.5 and differs in reading
    # the nine channels as one multivariate group; 2.5's label gains its
    # version here so the two are told apart in a legend.
    "timesfm3": ("TimesFM 3.0", 20),
    "sundial": ("Sundial", 12),
    # The three reported families again with the nine per-channel states
    # AVERAGED rather than concatenated -- the prediction evals' readout
    # (resolve_family in pretrained_tsfm). Separate keys, tsfm_<fam>_cmean_l<L>,
    # so they never share a cache file or a result row with the concat ones.
    "chronos2_cmean": ("Chronos-2", 12),
    "kronos_cmean": ("Kronos", 12),
    "timesfm3_cmean": ("TimesFM 3.0", 20),
    "kronos": ("Kronos", 12),
}

TSFM_KEYS: dict[str, list[str]] = {}
for _fam, (_label, _n) in TSFM_FAMILIES.items():
    TSFM_KEYS[_fam] = []
    for _L in range(_n + 1):
        _key = f"tsfm_{_fam}_l{_L}"
        MODEL_SPECS[_key] = {
            "label": f"{_label} L{_L}",
            "tsfm_family": _fam,
            "tsfm_layer": _L,
        }
        TSFM_KEYS[_fam].append(_key)

ALL_TSFM_KEYS = [k for keys in TSFM_KEYS.values() for k in keys]

# The full-history supervised specialists, swept for depth like the TSFMs.
# DEPTH 0 IS OMITTED: these pool the CLS token, which at depth 0 has not
# attended to anything and is a learned constant identical for every input --
# zero-variance features, no usable cross-section. A TSFM's depth 0 is the
# embedded input and does carry signal, which is why they start at 0 and these
# start at 1.
SUP_FAMILIES = {"sup_return": ("Supervised return", 12),
                "sup_vol": ("Supervised vol", 12),
                "sup_spread": ("Supervised spread", 12)}

SUP_KEYS: dict[str, list[str]] = {}
for _fam, (_label, _n) in SUP_FAMILIES.items():
    SUP_KEYS[_fam] = []
    for _L in range(1, _n + 1):
        _key = f"{_fam}_l{_L}"
        MODEL_SPECS[_key] = {
            "label": f"{_label} L{_L}",
            "sup_family": _fam,
            "sup_layer": _L,
        }
        SUP_KEYS[_fam].append(_key)

ALL_SUP_KEYS = [k for keys in SUP_KEYS.values() for k in keys]

# ── The readout grid, per month (sweeps/readout_grid_h2.sh) ────────────────
# Arm keys are the ARCHIVED run names, which still carry the _w16/_w0 suffixes
# from when the recency window was an axis. The knob is retired; these rows
# point at checkpoints on disk, so they keep the names those runs were given.
#
# One checkpoint per (arm, month) over holdout set 2, so these resolve through
# {dates} exactly like the rest of the registry. Use with
# LATENT_TRAIN_MONTHS set to the holdout-2 months; the reported 14 have no
# checkpoints in this project.
for _k, _rn, _lab in (
    ("h2_cls_w16",  "cls_w16",  "H2 CLS, W=16"),
    ("h2_cls_w0",   "cls_w0",   "H2 CLS, W=0"),
    ("h2_mean_w16", "mean_w16", "H2 mean pool, W=16"),
    ("h2_mean_w0",  "mean_w0",  "H2 mean pool, W=0"),
):
    MODEL_SPECS[_k] = {"label": _lab,
                       "project_glob": "readout-grid-h2-8ec432{dates}",
                       "run_name": _rn}
H2_KEYS = ["h2_cls_w16", "h2_cls_w0", "h2_mean_w16", "h2_mean_w0"]

# ── The projector-width grid (sweeps/lejepa_projdim_h2.sh) ─────────────────
# proj_dim {8, 32, 64, 128, 256} x pairing {k2ind, time_warp}, one checkpoint
# per (arm, month) over holdout set 2 -- the SAME five months, and therefore
# the same eval months and cached return panel, as the readout grid above.
#
# WHY THESE ARE HERE AT ALL: the IC sweep found width almost inert on
# xs_ic/return_900 (the whole k2ind column spans 0.0032, and only width 8
# cleared noise), which is a statement about what a ridge probe can read off
# the encoder -- not about how the latent space is ORGANIZED. T1-T4 and the
# factor analyses ask the second question, and a knob that does not move a
# probe can still move the geometry.
#
# 64 is the incumbent default and is in the grid under both pairings, so each
# curve carries its own control. Use with LATENT_TRAIN_MONTHS set to the
# holdout-2 training months.
for _pairing, _tag in (("k2ind", "k2ind"), ("timewarp", "warp")):
    for _w in (8, 32, 64, 128, 256):
        _key = f"pd_{_tag}_{_w}"
        MODEL_SPECS[_key] = {
            "label": f"projdim {_w} ({'K=2 same-ind' if _pairing == 'k2ind' else 'time warp'})",
            "project_glob": "lejepa-projdim-h2-ca70db{dates}",
            "run_name": f"lejepa_{_pairing}_proj-{_w}",
        }
PROJDIM_KEYS = [f"pd_{t}_{w}" for t in ("k2ind", "warp")
                for w in (8, 32, 64, 128, 256)]

# ── The full-day readout arms (sweeps/fullday_readout.sh) ───────────────────
#
# These are the ONLY encoders in the registry whose pooled readout is not the
# CLS token, which makes them the first models here whose "full-day" embedding
# is actually a full day. Every other entry is W=16 with a CLS anchored at the
# last patch, so a 12-layer rollout puts 90% of the readout in the final ~100
# minutes ([[recency_prior_fullday_gap]]) -- the latent-structure numbers for
# the rest of this table are computed on a last-hour view of a whole-day panel.
#
# ONE CHECKPOINT, NO {dates}: trained once on 2018-2022 rather than per month,
# so resolve_run finds them for any eval month. Score them on 2023+ months
# only; the training span covers most of the reported panel.
_FULLDAY = "lejepa-5yr-fullday-readout-8ec432-2018-01-01-2022-12-31"
for _k, _rn, _lab in (
    ("fd_mean_w16", "fullday_mean_w16", "Full-day mean pool, W=16"),
    ("fd_mean_w0",  "fullday_mean_w0",  "Full-day mean pool, W=0"),
    ("fd_cls_w0",   "fullday_cls_w0",   "Full-day CLS, W=0"),
):
    MODEL_SPECS[_k] = {"label": _lab, "project_glob": _FULLDAY, "run_name": _rn}
# The production control: same recipe, CLS readout, W=16 (run_lejepa_5yr_fullday.sh).
MODEL_SPECS["fd_cls_w16"] = {
    "label": "Full-day CLS, W=16",
    "project_glob": "lejepa-5yr-fullday-2018-01-01-2022-12-31",
    "run_name": "time_warp_lamb0.001_fullday_bs256_s42",
}
FULLDAY_KEYS = ["fd_cls_w16", "fd_cls_w0", "fd_mean_w16", "fd_mean_w0"]

# ── What a bare run evaluates ────────────────────────────────────────────────
#
# The 18 reported IC-era encoders + the random-init floor, i.e. exactly the
# rows in fixed_panel_P3S2_{augsup,sslic}.json. All manifest-resolved, so they
# pair with an EVAL month and work under --months as well as off MONTHS.
MODEL_ORDER = [
    # LeJEPA pairings, each at its own holdout-2 lambda
    "pair_rrc_final", "pair_warp_final", "pair_noise_final",
    "pair_k2_final", "pair_k2ind_final",
    # supervised heads, one recipe
    "sup_return_w8", "sup_vol_w8", "sup_spread_w8", "sup_multi_w8",
    # IC-era re-optimized SSL baselines
    "dino_final", "byol_final", "cpc_final", "ijepa_final", "mae_final",
    "ts2vec_final", "cost_final", "tfc_final", "timemae_final",
    # The six-month-span wave (2026-09-15). These are the SAME METHODS as the
    # *_final rows above on the supervised specialists' budget, and they are
    # what this box can still score -- the one-month campaign's checkpoints are
    # gone from here. The two spans stay separate keys, never one cell; a
    # result file carries whichever of the two its run produced, and
    # fixed_panel_table only emits rows the file actually has.
    "pair_rrc_6mo", "pair_warp_6mo", "pair_noise_6mo",
    "pair_k2_6mo", "pair_k2ind_6mo",
    "dino_6mo", "byol_6mo", "cpc_6mo", "ijepa_6mo", "mae_6mo",
    "ts2vec_6mo", "cost_6mo", "tfc_6mo", "timemae_6mo",
    "random",
]

# RETIRED 2026-09-04, kept in MODEL_SPECS but out of every default. These are
# the pre-IC-migration glob-resolved entries (a different lambda regime, a
# different target generation, a different supervised recipe — see the block
# above the pair_* keys). They were MODEL_ORDER until now, which is why the
# default month list was still the 14-month cross-stock sweep. Their
# checkpoints are gone from this box anyway: of the five projects only
# cross-stock-ind-c8dd09 survives, in lab/models-archive. Pass them
# explicitly via --models/--series if a restore ever makes them resolvable.
LEGACY_GLOB_KEYS = [
    "lejepa", "k2", "k2ind",
    "warp", "noise", "volj", "pricej", "chdrop",
    "sup", "supvol", "supspread", "supmulti",
]


def month_dates_suffix(ym: str) -> str:
    y, m = map(int, ym.split("-"))
    return f"-{ym}-01-{ym}-{calendar.monthrange(y, m)[1]:02d}"


def resolve_run(project_glob: str, run_name: str):
    """Newest checkpoint dir whose train_meta.json run_name matches."""
    hits = []
    for proj in eg.CHECKPOINT_ROOT.glob(project_glob):
        for run_dir in proj.iterdir():
            meta_f = run_dir / "train_meta.json"
            if not meta_f.is_file():
                continue
            try:
                meta = json.loads(meta_f.read_text())
            except json.JSONDecodeError:
                continue
            if meta.get("run_name") == run_name:
                hits.append(run_dir)
    if not hits:
        return None
    return max(hits, key=lambda d: (d / "train_meta.json").stat().st_mtime)


def draw_day_sets(
    ev_month: str, machine, n_portfolios: int, n_stocks: int, seed: int,
) -> dict[str, dict[str, int]]:
    """{date -> {ticker -> ff49}} — a fresh P-industry / S-stock draw per
    trading day, restricted to tickers present in the mosaic that day."""
    ind = pd.read_parquet(INDUSTRY_MAP)
    ind = ind[ind.month == ev_month]
    tick2ff = dict(zip(ind.ticker, ind.ff49.astype(int)))

    y, m = ev_month.split("-")
    meta = ensure_ticker_date_sidecar(f"{machine.mosaic_dir}/{y}/{m}")
    by_day: dict[str, list[str]] = defaultdict(list)
    for t, d in zip(meta["tickers"], meta["dates"]):
        if t in tick2ff:
            by_day[d].append(t)

    rng = np.random.default_rng([seed, int(y), int(m)])
    out: dict[str, dict[str, int]] = {}
    for d in sorted(by_day):
        pools: dict[int, list[str]] = defaultdict(list)
        for t in by_day[d]:
            pools[tick2ff[t]].append(t)
        eligible = sorted(ff for ff, ts in pools.items() if len(ts) >= n_stocks)
        if len(eligible) < n_portfolios:
            print(f"  {d}: only {len(eligible)} eligible industries — skipped")
            continue
        chosen_ff = rng.choice(eligible, size=n_portfolios, replace=False)
        day_map: dict[str, int] = {}
        for ff in chosen_ff:
            for t in rng.choice(sorted(pools[int(ff)]), size=n_stocks, replace=False):
                day_map[str(t)] = int(ff)
        out[d] = day_map
    return out


def nn_same_industry_day(X: np.ndarray, inds: np.ndarray) -> float:
    """Fraction of the day's points whose nearest OTHER point shares its
    industry. One view per stock, so 'excluding own stock' is automatic."""
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = d2.argmin(1)
    return float((inds[nn] == inds).mean())


def nn_same_industry_pooled(
    X: np.ndarray, tickers: np.ndarray, inds: np.ndarray,
) -> tuple[float, float]:
    """(rate, chance) over the month's pooled points: fraction whose
    nearest neighbor among OTHER STOCKS' points shares its industry.
    Chance = mean same-industry share of each point's non-own-stock pool
    (identical across models for the same draws)."""
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    same_stock = tickers[:, None] == tickers[None, :]
    d2[same_stock] = np.inf
    nn = d2.argmin(1)
    rate = float((inds[nn] == inds).mean())
    pool = ~same_stock
    same_ind = inds[:, None] == inds[None, :]
    chance = float(((same_ind & pool).sum(1) / pool.sum(1)).mean())
    return rate, chance


_filter_rows = eg._filter_rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_portfolios", type=int, default=3)
    p.add_argument("--n_stocks", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--months", default=None, help="comma list, default all 14")
    p.add_argument("--models", default=",".join(MODEL_ORDER))
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    P, S = args.n_portfolios, args.n_stocks
    chance = (S - 1) / (P * S - 1)
    months = args.months.split(",") if args.months else MONTHS
    models = args.models.split(",")

    eg.apply_variant("mixed")  # module init only
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    machine = LocalMachineConfig()
    schedule = MarketSchedule(machine.holiday_csv)

    # day_rates[model][train_month] = [per-day NN rates] (secondary metric)
    day_rates: dict[str, dict[str, list[float]]] = defaultdict(dict)
    # pooled[model][train_month] = (rate, chance) (primary metric)
    pooled: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)

    for ym in months:
        ev_month, ev_start, ev_end = eg.eval_window_t_plus_n(ym, 1)
        dates_sfx = month_dates_suffix(ym)

        runs = {}
        for key in models:
            spec = MODEL_SPECS[key]
            run_dir = resolve_run(
                spec["project_glob"].format(dates=dates_sfx), spec["run_name"],
            )
            if run_dir is None:
                print(f"{ym}: no checkpoint for {key} — skipped")
                continue
            runs[key] = run_dir
        if not runs:
            continue

        cache = eg.CACHE_DIR / (
            f"indnn__{ev_month}__seed{args.seed}__P{P}S{S}.pkl"
        )
        if cache.exists():
            with open(cache, "rb") as f:
                blob = pickle.load(f)
            day_sets, batches = blob["day_sets"], blob["batches"]
            print(f"{ym}: window cache hit ({cache.name})")
        else:
            day_sets = draw_day_sets(ev_month, machine, P, S, args.seed)
            all_tickers = sorted({t for dm in day_sets.values() for t in dm})
            batches, _counts = eg.collect_curated_windows(
                ev_start, ev_end, target_tickers=all_tickers,
                samples_per_ticker=25, machine=machine, schedule=schedule,
                seed=42,
            )
            batches = _filter_rows(
                batches,
                lambda t, d: str(t) in day_sets.get(str(d), {}),
            )
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump({"day_sets": day_sets, "batches": batches}, f)

        n_expected = P * S
        for key, run_dir in runs.items():
            backbone = eg._load_backbone(
                run_dir, run_dir, run_dir.parent.name,
                pool=eg.LATENT_POOL,
            ).to(device).eval()
            try:
                res = eg.forward_cached(
                    backbone, batches, device, cap=10 ** 9,
                )
            finally:
                backbone.to("cpu")
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            tk = np.asarray([str(t) for t in res["tickers"]], dtype=object)
            dt = np.asarray([str(d) for d in res["dates"]], dtype=object)
            rates = []
            keep = np.zeros(len(tk), dtype=bool)
            for d in sorted(day_sets):
                sel = dt == d
                # Windows can be missing for a drawn stock (thin day);
                # score only complete days so chance stays exact.
                if sel.sum() != n_expected:
                    continue
                keep |= sel
                inds = np.asarray([day_sets[d][t] for t in tk[sel]])
                rates.append(nn_same_industry_day(res["X"][sel], inds))
            day_rates[key][ym] = rates

            p_inds = np.asarray(
                [day_sets[d][t] for t, d in zip(tk[keep], dt[keep])]
            )
            p_rate, p_chance = nn_same_industry_pooled(
                res["X"][keep], tk[keep], p_inds,
            )
            pooled[key][ym] = (p_rate, p_chance)
            print(
                f"{ym} [{key}]: pooled NN-same-industry={p_rate:.1%} "
                f"(chance {p_chance:.1%}, n={int(keep.sum())}) | "
                f"within-day={np.mean(rates):.1%} (chance {chance:.0%}, "
                f"{len(rates)} days)"
            )

    # ---- Aggregate + report (primary = month-pooled) ----
    print()
    print(f"P={P} portfolios x S={S} stocks; within-day chance={chance:.1%}")
    print(f"{'model':<22} {'months':>6} {'pooled':>7} {'chance':>7} "
          f"{'ratio':>6} {'t':>6} {'| within-day':>12} {'t':>6}")
    summary = {}
    for key in models:
        if not pooled.get(key):
            continue
        p_rates = np.array([r for r, _ in pooled[key].values()])
        p_chances = np.array([c for _, c in pooled[key].values()])
        excess = p_rates - p_chances
        t_pool = (
            excess.mean() / (excess.std(ddof=1) / np.sqrt(len(excess)))
            if len(excess) > 1 else float("nan")
        )
        month_means = np.array([np.mean(r) for r in day_rates[key].values()])
        t_day = (
            (month_means.mean() - chance)
            / (month_means.std(ddof=1) / np.sqrt(len(month_means)))
            if len(month_means) > 1 else float("nan")
        )
        label = MODEL_SPECS[key]["label"]
        print(
            f"{label:<22} {len(p_rates):>6} {p_rates.mean():>6.1%} "
            f"{p_chances.mean():>6.1%} {p_rates.mean() / p_chances.mean():>5.2f}x "
            f"{t_pool:>6.2f} {month_means.mean():>11.1%} {t_day:>6.2f}"
        )
        summary[key] = {
            "label": label,
            "pooled_rate": float(p_rates.mean()),
            "pooled_chance": float(p_chances.mean()),
            "pooled_ratio": float(p_rates.mean() / p_chances.mean()),
            "pooled_t": float(t_pool),
            "pooled_by_month": {
                m: {"rate": float(r), "chance": float(c)}
                for m, (r, c) in pooled[key].items()
            },
            "day_month_means": {
                m: float(np.mean(r)) for m, r in day_rates[key].items()
            },
            "day_t": float(t_day),
        }

    out_stem = f"industry_nn_sweep_P{P}S{S}"
    with open(eg.OUT_DIR / f"{out_stem}.json", "w") as f:
        json.dump(
            {"P": P, "S": S, "day_chance": chance, "seed": args.seed,
             "models": summary},
            f, indent=1,
        )
    print(f"wrote {eg.OUT_DIR / f'{out_stem}.json'}")

    # ---- Figure: per-month pooled rate/chance ratio per model ----
    keys = [k for k in models if pooled.get(k)]
    fig, ax = plt.subplots(figsize=(eg.WIDTH_FULL * 0.55, 2.6))
    for i, key in enumerate(keys):
        ratios = np.array([r / c for r, c in pooled[key].values()])
        jx = np.random.default_rng(1).normal(i, 0.05, len(ratios))
        ax.scatter(jx, ratios, s=14, alpha=0.55, color="#4878a8", edgecolor="none")
        ax.errorbar(
            i, ratios.mean(),
            yerr=1.96 * ratios.std(ddof=1) / np.sqrt(len(ratios)),
            fmt="o", color="black", ms=5, capsize=3, zorder=3,
        )
    ax.axhline(1.0, color="crimson", lw=1, ls="--", label="chance")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(
        [MODEL_SPECS[k]["label"] for k in keys], fontsize=7, rotation=15,
    )
    ax.set_ylabel("pooled NN same-industry / chance", fontsize=8)
    ax.set_title(
        f"{P} industries x {S} stocks per eval day - month-pooled NN",
        fontsize=9,
    )
    ax.legend(fontsize=7, frameon=False)
    written = save_figure(fig, eg.OUT_DIR / out_stem)
    plt.close(fig)
    for pth in written:
        print(f"Saved {pth}")


if __name__ == "__main__":
    main()
