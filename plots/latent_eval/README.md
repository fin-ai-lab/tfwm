# latent_eval — the latent-structure evaluation suite

What does a frozen encoder's embedding carry beyond its probe IC? One
directory, one pipeline, three task families, every model through the same
stages. Consolidated 2026-08-26 from `plots/embedding_geometry` +
`plots/factor_structure` (both deleted; history at commit 6450600 and
earlier).

## The three task families

1. **Fixed-panel latent structure (T1–T4)** — `fixed_panel/`. Panels of
   P=3 random FF49 industries × S=2 full-presence member firms (6 stocks),
   held fixed across every trading day of an eval month, one full-day view
   per (firm, day). T1 pooled-NN = partner's same-day view; T2 own
   day-centroid nearest (focal firm excluded from every centroid); T3 own
   firm-centroid nearest (focal day excluded — kills the same-day
   co-movement shortcut); T4 firm centroid's NN = partner's centroid.
   Each reported as rate/chance ratio + mean-rank percentile. Producer:
   `fixed_panel_metrics.py`; renderer: `fixed_panel_table.py` -- one level
   up, since it also renders the two factor columns (markdown by default;
   `--latex` writes the paper table, four task columns T1-T4 plus F1/F2 by
   method, families grouped and the floor in its own block).

   BOTH READOUTS ARE IN EVERY TASK CELL, stacked -- the ratio above, the
   mean-rank percentile below -- because THE TWO DO NOT AGREE: the top-1
   rate saturates once a model is good and says nothing about how it fails,
   the mean rank does, which is why task_rank_corr ranks on both and
   averages. They were two separate .tex files until 2026-09-15, which made
   a disagreement something a reader had to find by flipping between
   tables. Each half carries its own bold-two and its own floor dagger.

   The tags default to the reported roster, so a bare run is the table:

       uv run python plots/core/fixed_panel_table.py
       uv run python plots/core/fixed_panel_table.py --latex \
           --out plots/core/fixed_panel_table.tex

   (`--top1 rate` and `--rank rank` swap either half for its un-normalized
   form; the percentile is the one comparable ACROSS columns, since the
   candidate pool is ~103 on T1 and 6 on T3. `--with-t` adds each half's
   own t. The .tex is not written by `run_eval.sh`: it runs one TAG at a
   time and the table is a merge of several.)

   THE .tex CARRIES NO CAPTION and no column descriptions -- bare `T1..T4`,
   `F1`, `F2` -- because the paper's caption is hand-written. Below-floor
   cells are struck through rather than daggered, which needs
   `\usepackage[normalem]{ulem}`; the emitted file says so on line 1. The
   per-method n is markdown-only: read it before writing the caption, since
   an arm whose wave is still filling is pooled over fewer months.

   The illustrated definitions: `metric_diagram.py` →
   `fixed_panel_metric_diagram*.png` -- CURRENTLY ARCHIVED, since those
   panels are drawn from real embeddings and the archived ones were read at
   the wrong token; re-run `metric_diagram.py --clean` off the new panel.

   THE REPORTED TABLES ARE BEING REBUILT (2026-09-14). See "The readout
   archive" below before quoting any latent number.

2. **Pelger latent factors** — `factors/`. Statistical factors estimated
   per month from the panel itself (Pelger 2018, *Large-dimensional factor
   modeling based on high-frequency observations*): PCA of the realized
   quadratic-correlation matrix of 5-min log mid-quote returns, perturbed
   eigenvalue-ratio factor count, elementwise continuous/jump split (a=3,
   TOD × bipower local vol). `pelger.py` is the estimator library (tested
   in `tests/test_pelger.py`). Two analyses over month-mean full-day
   embeddings (ridge, 5-fold cross-firm CV grouped by gvkey; OOS Pearson
   r, t across months):
   * `decode_loadings.py` — predict a stock's loadings (total 1–4, R²,
     cont 1–2, jump 1–2) from its month-mean embedding. (The 9-stat
     liquidity-control variants — control/emb+ctl/resid — were retired
     2026-08-26 as ad hoc; pre-retirement JSONs still carry them.)
   * `subspace_alignment.py` — canonical correlations between the top-10
     embedding PCs and the K̂-dim loading space; ρ̄ = Σρ²/K̂ = the share of
     the loading-space variation the embedding subspace captures,
     calibrated against 200 row permutations.
   The illustrated definitions: `factor_diagram.py` →
   `factor_diagram*.png` (`--clean` = the paper variant, titles only);
   `factors/caption.md` is its paper caption and panel-by-panel prose.
   (A third analysis, `temporal_prediction.py` — predicting half-2
   loadings from half-1 features beyond the trailing estimate — was
   retired 2026-08-26; it was null everywhere. Recoverable from git.)

3. **Day-view uniformity** — `day_view/`. Reserved; see its README.

## Do the tasks agree?

Moved to [`plots/task_corr/`](../task_corr/README.md) on 2026-09-26.

## The readout archive (2026-09-14)

EVERY LATENT RESULT BEFORE THIS DATE WAS READ AT THE WRONG TOKEN and is in
`_archive/pre_meanpool_readout/` (its README has the detail). The short
version: the loaders passed no `pool`, so each checkpoint was read at the
readout it TRAINED with -- supervised at `pool="last"`, i.e. one patch of the
day, beside mean-pooled LeJEPA/SSL -- and the random-init floor was built at
`"cls"`, a third token again.

`7ee9cdf` (2026-09-08) was supposed to fix this and DID NOT REACH THE REPORTED
ROSTER. It pinned `LATENT_POOL` at every `_load_backbone` call site, but every
key in `MODEL_ORDER` is manifest-resolved and a manifest row is loaded through
`load_encoder`, which had no `pool` parameter at all; `load_supervised` (the
`--sup-families` depth sweep) was the same. Both take one now, the two latent
call sites pass `LATENT_POOL`, and the guard in `tests/test_default_recipe.py`
scans BOTH loader names instead of only `_load_backbone`. Anything produced
between 2026-09-08 and 2026-09-14 is still pre-fix for a manifest series.

`plots/latent_eval/run_meanpool_reeval.sh` is the re-run: the three supervised
span specialists (the 581eb2 wave `all_metrics_supervised_ic.png` draws) and
the frozen TSFMs kronos / timesfm3 / chronos2 at every layer, on the 31 eval
months all of them share, tagged `_meanpool`. The TSFMs were never mis-read --
`PretrainedTSFM` pools `time_pool="mean"` internally and every captured layer
goes through `_pool_time` -- but their FLOOR was, so they are rebuilt with it.

The 18-method campaign is NOT re-runnable here: `ssl-ic-final-222d99-*` and the
`pair_*` / `sup_*_w8` projects are gone from this box and the caches hold only
the optimization months, so those arms need retraining, not re-scoring.

## The pipeline (single, consistent)

`run_eval.sh` is the canonical driver — idempotent stages, identical for
every model family, run **month by month** rather than stage by stage:

```
per month (JOBS of them at once, STAGGER seconds apart):
  1. fixed_panel/fixed_panel_metrics.py     T1-T4 on the fixed panels
                                            -> a per-month SHARD
  2. factors/build_fullday_embs.py          full-day embeddings, ONE builder:
                                            --families (TSFM, all layers/pass)
                                            --sup-families (supervised depths)
                                            --series (any trained registry key)
                                            --randvit-seeds (the floor)
  3. factors/build_return_panel.py          5-min mid panels    (model-free,
     factors/estimate_factors.py            Pelger factors       cached)
once, over the whole panel:
     fixed_panel/merge_month_shards.py      pool the shards
  4. factors/{decode_loadings,subspace_alignment}.py
  5. factors/factor_summary.py + fixed_panel/fixed_panel_table.py
```

MONTH-MAJOR BECAUSE STAGES 1-3 ALL READ THE SAME MONTH of raw data out of
mosaic — the panel windows, the full-day grid, the 5-min mid panel. Walking
the month list once per stage read every month three times, each read cold. It
also changes what a crash leaves: N complete months instead of one complete
stage. Stages 4-5 stay at the end because they are cross-month by construction
(fit per month, pool a t across months) and read embeddings from
`ff_fullday_cache`, so there is nothing to co-locate.

PARALLEL BECAUSE ONE MONTH CANNOT SATURATE THE GPU. A month alternates
between collecting (CPU/IO, GPU measured at 0%) and forwarding (GPU at 93%).
`JOBS` months in flight fill the gaps — but starting them together runs their
phases in lockstep, which measured 48% mean GPU on four workers, so `STAGGER`
offsets the first pass and they drift apart on their own afterwards.

Each month writes its own stage-1 shard, so parallel months never
read-modify-write one json; `merge_month_shards.py` pools them with
`fixed_panel_metrics._repool`, the same arithmetic the in-file merge uses (`t`
is re-derived from the union of the per-month series, never averaged). A
month whose shard already exists is SKIPPED, so a restart costs only the
months that did not finish; `REDO=1` forces it.

Parameters via env (`MODELS`, `SERIES`, `MONTHS`, `TAG` — see the header of
`run_eval.sh`). `plots/tsfm_layers/run_latent.sh` is this same pipeline
instantiated for the frozen-TSFM layer sweep on the optimization months; its
reducers (`layer_sweep.py`, `latent_sweep.py`, `policy.py`) stay in
`plots/tsfm_layers`.

The model registry is `fixed_panel/industry_nn_sweep.py` (`MODEL_SPECS`,
`MODEL_ORDER`, `MONTHS`, `resolve_run`); the shared eval library
(variants, window collection, cached forwards, the t-SNE figure CLI) is
`fixed_panel/panel_lib.py` (né `embedding_geometry.py`). All checkpoint
loading goes through `market_jepa/eval/checkpoints.py` — no parallel loader
copies, and `panel_lib.load_series_encoder` is the single (key, eval month)
-> encoder resolver every stage calls.

Bare defaults, since the pre-IC entries were retired on 2026-09-04:
`MONTHS` is the 32 canonical sweep months (`style.load_sweep_months`) and
`MODEL_ORDER` is the 18 reported IC-era encoders plus the random-init
floor, all manifest-resolved. The retired glob-resolved entries are still
in `MODEL_SPECS`, listed in `LEGACY_GLOB_KEYS` and out of every default —
they point at pre-IC-migration projects whose checkpoints are no longer on
this box. Their rows come from two manifests (the repo one and a
machine-local `_scratch/latent_eval/manifest_*.json`), which `run_eval.sh`
now joins into `MJ_NOCLAMP_MANIFEST`; that split is why the reported panel
was originally produced as the two tagged runs `_augsup` and `_sslic`.
`fixed_panel_table.py --tags _augsup _sslic` merges them into one table.

## What is current vs pending (2026-08-26)

**Current — the frozen-TSFM results (kept; the TSFMs cannot ingest the new
info tokens, so their eval stands):**
* `fixed_panel/fixed_panel_P3S2_tsfmlayers.json` — T1–T4, 61 layer-points,
  5 optimization months.
* `factors/*_tsfmlayers.{json,out}` — the three analyses, same 5 months.
* `factors/tsfm32/` — the same three, 32 reported months (merged shards).
* Everything in `plots/tsfm_layers/` (IC sweep, latent reductions, policy,
  channel-pool ablation).

**Pending — every trained-from-scratch model (retraining with info tokens /
new recipes; old results deleted 2026-08-26, recoverable from git):**
* Re-run `run_eval.sh` once the retrained checkpoints land in the registry.
* New random-init floors to match the new architecture (`randvit` seeds in
  stage 2, `random` in stage 1).
* The supervised layer sweep (`--sup-families`) once the per-month
  specialists are retrained ([[tsfm_layer_sweep_ic]] "still to redo").
* Panel-window caches (`fixed_panel/cache/`) were deleted with the old
  results — stage 1 recollects them from mosaic on first run.
* The (retired) supervised recency-bias arm motivated `day_view/` — design it before
  the re-eval wave if it should ride the same panels.

## External data

* `lab/market-jepa-checkpoints/factor_structure_cache` —
  `panel_<ym>.npz` + `factors_<ym>.npz`. **Model-independent** — raw
  mid-quote panels and their factors — so it survives every retrain.
  ~45 s/month (panel) + ~1 min/month (factors) to extend. IT HOLDS ONLY
  THE 5 OPTIMIZATION MONTHS as of 2026-09-14; this entry claimed "32
  reported + 5 optimization" long after the reported months went, so stage
  3 is a ~1 h rebuild on any reported-panel run, not the no-op it reads
  like.
* `lab/market-jepa-checkpoints/ff_fullday_cache` — one
  `emb_<series>.npz` per (month, series) + `grid_meta.npz` per month. ALSO
  ONLY THE 5 OPTIMIZATION MONTHS as of 2026-09-14 — the reported months'
  TSFM embeddings are gone with the trained ones, so a reported-panel run
  rebuilds every arm.
  ITS KEY DOES NOT RECORD THE READOUT. `emb_<series>.npz` names the series
  and nothing else, and the builder SKIPS any series whose file exists
  unless `--overwrite`, so a post-fix rerun over a month that was built
  before the fix silently serves pre-fix embeddings. (This is the bug
  929f582 fixed for the IC cache by putting the readout in the key; the
  same fix has not been made here.) The `randvit_s*` entries on the
  optimization months are symlinks into models-archive, i.e. the OLD
  cls-pooled floors — delete them before re-running those months.
  `_experiment/` belongs to the Fama-French experiment — do not touch.
* Both paths are env-overridable (`FACTOR_STRUCTURE_CACHE`,
  `FF_FULLDAY_CACHE`) because compute nodes do not mount lab/; the
  cluster drivers are our cluster's `slurm_tsfm_latent` and `run_tsfm_latent`
  launchers (not included).

## Protocol notes that survive every rerun

* Consumers of the return panel drop the 09:30→09:35 increment — one-sided
  opening books produce garbage first marks (e.g. $1000 placeholder asks) —
  leaving the paper's 77 increments/day from 9:35. Any analysis sampling
  instantaneous quotes near the open needs the same guard.
* Loadings are functions of returns: within-month decoding shares data with
  its own targets' estimation window — the encoder never trained on the
  month, but its features and the loadings are computed FROM the same
  returns. The margin over the random-ViT floor is the evidential
  standard; never quote a decode or ρ̄ number without the floor beside it.
* Factor k's identity is stable only up to rotation across months;
  cross-month means over "factor k" follow the eigenvalue ordering.
* Jump K̂ is unstable across ε (the paper reports the same); jump-loading
  targets use the top-2 jump factors regardless.
* The T2/T3 centroid exclusions are load-bearing: without them the tasks
  reduce to same-day co-movement / trivial self-matching.
* The `random` / `randvit_s*` floor is not optional — several tasks put an
  untrained ViT well above chance, so ratio-over-chance alone does not say
  whether anything was learned.
