# Finance baselines

Classical forecasting models scored on the **synchronized anchor panel** by
**rank IC**, so a textbook econometric model and a learned encoder land on one
axis, row for row.

```
prebuild_tables.py   build every panel table a pass needs, in parallel
panel_tables.py      the tables: encoder-panel rows, predictors read off the view
models.py            AR(p) / ARMA(p,q) / HAR-RV / GARCH(1,1) / Ridge ARDL / mean reversion
view_models.py       Ridge + GBM regressors on the whole 2048x9 view tensor
calculate_panel_ic.py  fit on the pool, score on M+1 -> plots/metrics/*.json
baseline_table.py    the appendix table -> baseline_table.tex
```

The reported artifact is `plots/metrics/finance_panel_ic_6mo.json` and the
reported output is `baseline_table.tex`, the appendix table a reader is sent
to for context on how good the supervised numbers are in absolute terms.

## Running a pass

```bash
uv run plots/finance_baselines/prebuild_tables.py --months-from-k2ind \
    --workers 8
uv run plots/finance_baselines/calculate_panel_ic.py --months-from-k2ind \
    --view-learners
```

Tables cache under
`lab/market-jepa-checkpoints/finance_baselines/panel_tables/`. The view
tensors the Ridge/GBM learners read do NOT cache — 37 GB per fit month — so
`--view-learners` pays a second decode pass over every month.

## The fit span, and why the reported pass is six months

`--fit-span 1` — fit on M, score on M+1 — is what `finance_panel_ic.json`
holds, and it is NOT what the learned arms on those axes get. A supervised
head trains on the six months ending at M (`build_probe_breadth_manifest.py`),
and the probes in `plots/core/probe_fit_table.py` are fitted on that same
six-month pool at 36 anchors/day, ~2.1M rows. A classical model fitted on one
month's 358k is being compared on fit-set size as much as on model class,
which is the one thing this panel exists to hold fixed.

`--fit-span 6` pools the six months ending at M and fits on all of them. That
is legitimate here because every model in `models.py` is a ROW-WISE fit — an
ARMA's MA terms are reconstructed per row by Hannan-Rissanen, never carried
along a time index — so pooling months is exactly pooling their rows. Cells
stay unique across the pool (`{date}@{anchor}`), so GARCH's rank-IC selection
still ranks within one cross-section.

The reported six-month artifact is `plots/metrics/finance_panel_ic_6mo.json`,
written at the 15-minute horizon only:

```bash
uv run plots/finance_baselines/prebuild_tables.py --months <32 fit months> \
    --fit-span 6 --workers 10
uv run plots/finance_baselines/calculate_panel_ic.py --months <...> \
    --fit-span 6 --horizons 900 \
    --json plots/metrics/finance_panel_ic_6mo.json
```

Three things about that pass:

**It is a different file, not an append.** One file holds one protocol;
`--append` refuses a span that disagrees with what is already there. Keeping
the one-month artifact is what makes the 1-vs-6 gap measurable at all.

**Every record says what it got** — `fit_months`, `n_fit_months`,
`n_fit_rows`. The mosaic starts 2008-01, so eval month 2008-03 has a
two-month pool and says so rather than being averaged in as a six.

**The cost is in the tables, then in the stream.** 32 fit months need 127
distinct 36-anchor tables at `--fit-span 6` (the pools overlap), ~385 MB and
~5–10 min of decode each; `prebuild_tables.py` deduplicates them. The
classical fits themselves are minutes. `--view-learners`/`--full-view` then
re-stream all six months per eval month — ~45 min of decode, ~17 min of
float64 syrk at the full width, and a GBM sample that scales with the pool
(`--hgb-fit-rows`). Budget ~1.5 h and ~50 GB of RSS per eval month, and size
the concurrency to the memory rather than to the core count.

## Plotting

There is no figure script here. The series are declared in
`plots/metrics/metrics.py` (`SERIES_DEFS`, keyed by `json_ic_model`) and drawn
by the standard IC figure, which is what puts them on the same axes as the
encoders. Pass `--out-name`: without it the render overwrites the standard
`all_metrics_ic` figure.

```bash
uv run plots/metrics/metrics.py --absolute \
    --out-name all_metrics_finance_ic \
    --series cross_stock_k2ind supervised_specialist supervised_head \
             randinit_vit mean_reversion har_rv garch11 ar_p arma_pq ardl \
             ridge_full hgb_full
```

`supervised_head` is the gray star at 15 min: the same supervised specialist
read out through its own trained head instead of a fresh ridge probe. A head
is trained at one horizon, so it is one point per panel rather than a line. It
carries NO legend key (`no_legend` in `SERIES_DEFS`) — the caption names it,
so it does not spend a legend slot.
Rebuild its artifact with
`uv run plots/metrics/build_supervised_ic.py --readout head`.

## The appendix table

```bash
uv run plots/finance_baselines/baseline_table.py --latex \
    --out plots/finance_baselines/baseline_table.tex --one-month-delta
```

`baseline_table.py` renders the six-month artifact beside the supervised arms
and the untrained floor, in `plots/core/probe_fit_table.tex`'s style (a head
IC in parentheses; bold = the best two per target **on each side**, so the two
strongest supervised arms and the two strongest baselines are both marked and
the gap between the pairs is the comparison).

**Nothing is struck here, unlike `tab:probe_fit`.** That table strikes cells
below the floor because it is asking whether a pretraining method learned
anything at all; this one is read for absolute context, so the floor is simply
the last row and the reader compares across it. Striking would also assert a
binary on gaps the numbers do not support -- Ridge ARDL clears the floor on return
by +0.0002.

**The supervised rows and the floor are imported, not recomputed.** They come
from `plots/core/probe_fit_table`'s own loaders at its alpha, so a cell here
and the same cell in `tab:probe_fit` cannot disagree. Every row is then
restricted to `comparable_months` -- the months the floor was read on -- so
the classical rows do not average over 32 months against the floor's 31.

`--one-month-delta` prints what the six-month refit bought against
`finance_panel_ic.json` on the same months. It is not cosmetic on return:
Ridge ARDL goes +0.0112 -> +0.0177, which is the difference between sitting below
the untrained floor and clearing it.

## Two things a reader will ask

**Why are the predictors read off the view and not the order book?**
Because the encoder is fed the view. `iter_panel` hands out a normalized
(2048, 9) tensor — aggregated to 6–11 s per token, then `normalize_numpy`
standardizes the price group by *each view's own* mean and std. Reading a
baseline off the raw 1 Hz book would give it absolute price and spread levels
the encoder provably cannot see: on the book `-spread(t)` reaches rank IC ~0.9,
on the view ~0.6, and the gap is the normalization rather than the model. See
the header of `panel_tables.py`.

**Where did the naive floor go?**
`NaiveZero` and `TimeOfDay` are constant within a cross-section, so every row
ties and their rank IC has no value — under AUC they scored exactly 0.500 and
were the floor, but here the floor is the zero line. `PriorDaySpread` read
yesterday's spread at the same clock time, which is not inside a 2048-token
window. `models.py` documents all three.
