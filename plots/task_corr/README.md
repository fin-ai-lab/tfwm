# task_corr — do the tasks agree?  (`task_rank_corr.py`)

The six latent tasks in `plots/latent_eval/` and the three forward-IC probes in
`plots/metrics` are nine scorings of the same encoders, each read on its own.
`task_rank_corr.py` ranks the 18 METHODS 1–18 per task and correlates the
RANKINGS — one lower-triangle 9×9 Spearman matrix,
`task_rank_corr.{png,pdf,json}`. A method is not an encoder: every arm is
retrained per eval month, so one cell of the rank table summarizes 26–32
separately trained encoders (32 for the probe and organization columns, 26
for the factor columns), each scored on the month after it trained. n = 18
for every correlation because 18 is the number of things being ranked; the
months are what make each of those 18 numbers stable rather than one draw.

ONE figure, one configuration, no variant flags — the choices are made in the
script and documented in its header. The roster is the 18 TRAINED methods
(the random-init floor is evidential for the underlying tables and lives in
RESULTS.md, but it is not a method being compared); T1–T4 average the
top-1-rate rank with the mean-rank rank and re-rank; the factor columns are
one number each — mean decode r over loadings 1–4, and ρ̄. READOUT: every
method is read through the ridge probe EXCEPT the four supervised arms, which
are read through their own trained head on the target(s) they were trained
for (the multihead on all three). A head is the model's actual output, and
the multihead's — an untrained-head artifact for 31 of 32 months — became
usable when the full-history re-score landed 2026-08-31; `ICRun.head` enforces
`style.HEAD_SCHEMA` so an un-re-scored trunk surfaces as missing rather than
as noise. The asymmetry is real (a trained head for four arms, a linear probe
for fourteen) and belongs in any caption; it moves one matrix cell by ≥ 0.05
and hands the multihead first place on forecasting, ahead of MAE. Factor-model R² and
the permutation z are deliberately unused, for reasons the header gives. It
recomputes the factor means on the months `_augsup` and `_sslic` SHARE (all
32 since 2026-08-31 — the intersection is kept as a guard, not because it
still cuts anything), and takes the IC columns on the 32-month fair panel;
absolute IC, since on a shared panel that orders models identically to ΔIC.
All nine columns are now on the same 32 months.

Tasks are labelled with the PANEL TITLES of the two task diagrams
(`fixed_panel/metric_diagram.py --clean`, `factors/factor_diagram.py
--clean`) rather than T1–T4, so a talk can show the diagram and then this
matrix without renaming anything mid-deck.

The answer is that they do not agree: mean |ρ| off-diagonal is 0.36, and only
9 of 36 pairs clear p < 0.05 (|ρ| ≥ 0.47 at n = 18; those are bold on the
figure). What holds is *within* families — the three prediction tasks
(ρ = 0.75–0.89), Partner Matched View ↔ Own-Day Centroid (0.90), Decoded
Loadings ↔ Subspace Alignment (0.69) — while the prediction block and the
Partner-Matched/Own-Day block run NEGATIVE (Spread Change ↔ Partner Matched
View = −0.51, the only significant negative pair). Subspace Alignment is the
one latent task that tracks the prediction tasks (0.35–0.61); Own-Firm
Centroid tracks nothing positively at all, and its strongest relation is
−0.46 against Decoded Loadings.

`task_rank_corr_split.png` is the same result as one scatter: each method's
average rank on the three forecasting tasks against its average rank on the
other six. ρ = **+0.02** — the two axes are unrelated, which is the matrix's
finding in a form a slide can carry. Best three at forecasting: Multihead,
MAE, BYOL. Best three at structure: I-JEPA, BYOL, CoST — BYOL is the only
method in both lists. Drawn deliberately bare (no y = x line,
no median crosshairs, no quadrant captions, no title): with eighteen labelled
points every one of those competed with the content. Point names and family
colours come from `plots/style.SERIES_STYLES`, so a method reads the same
here as on every other paper figure; labels are placed by scored candidate
slots, nearest-slot-first, because a push-apart relaxation oscillates.

`task_rank_corr_split_org.png` is the SAME scatter with the y axis narrowed to
the four organization tasks — for a poster whose latent section shows only
those panels, so the scatter cannot promise a factor axis the poster never
draws. **The finding changes character and the caption must change with it:**
ρ = **−0.31** (p = 0.21, n = 18), not +0.02. Subspace Alignment is the one
latent task that tracks forecasting positively (+0.35 to +0.61), so dropping
the two factor columns removes the only positive contributor and what is left
leans negative. The defensible sentence is "no significant relationship — if
anything the ordering runs backwards", NOT "uncorrelated". It also raises
Partner Firm Centroid's weight from 1/6 to 1/4 of the axis, and that task sits
at the random-reranking floor (see the caveats below), so the y coordinate is
noisier than the six-task version. Quote the six-task figure unless the
surrounding panels force otherwise.

TWO CAVEATS FROM A PER-MONTH REBUILD (scratch, not checked in). Rebuilding
all nine columns per eval month gives 32 independent matrices over the 18
methods: mean |ρ| within-month is **0.219**, against 0.349 on month-averaged
scores. That gap is attenuation — one month's rank is a noisy measurement, so
the aggregate is the better estimate, and the disagreement finding is
CONSERVATIVE as reported. Second, decomposing each task's rank variance into
between-method and month-to-month noise puts **Partner Firm Centroid at the
random-reranking floor** (0.03, against a permutation null of 0.032, 95th
pct 0.052) — that task does not separate the methods, so its whole row is
aggregated noise and should not be quoted. Return is weak too (0.31);
Volatility Change (0.88), Own-Firm Centroid (0.88) and Spread Change (0.85)
are the reproducible ones.

## Slide builds — `task_rank_corr_build.py`

Progressive-reveal frames of both figures for beamer `\only`: `task_rank_corr_build{1..4}` (the matrix) and `task_rank_corr_split_org_build{1..4}` (the organization scatter). Reads `task_rank_corr.json` only, so run `task_rank_corr.py` first.

Moved here from `plots/core/` and `plots/latent_eval/` on 2026-09-26. The pre-mean-pool renders are still in `plots/latent_eval/_archive/pre_meanpool_readout/`.
