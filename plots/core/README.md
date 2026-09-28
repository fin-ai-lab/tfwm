# Core results

The three artifacts the paper's claims rest on, and the code that draws them.

| artifact | generator |
|---|---|
| `probe_fit_table.tex` | `probe_fit_table.py` |
| `fixed_panel_table.tex` | `fixed_panel_table.py` |
| `probe_fit_breadth.png` / `probe_fit_breadth_floor.png` | `probe_fit_breadth.py` |

Rebuild all three:

```bash
plots/core/regen.sh
```

Do not run the generators by hand unless you have read `regen.sh` first. Two of
them take a flag that is easy to forget and that **fails quietly** when omitted
(`fixed_panel_table.py` emits no caption at all without `--caption`;
`probe_fit_breadth.py` must be run twice, once with `--with-floor`). Those flags
live in `regen.sh` so they are not retyped from memory.

Never pipe a generator to `head`. SIGPIPE kills the writer mid-write while the
shell still reports exit 0. That has already produced one stale table here.

## Why the two tables agree

Both read their row labels, family grouping and `\citep` keys from
`task_rank_corr.ROSTER` (in `plots/task_corr/`). That is the only reason a rename lands in both tables
from one edit -- as the DINO/BYOL rename on 2026-09-17 did. Renaming a method
anywhere else will desynchronize them.

`ROSTER` carries one fact the labels no longer do: **DINO and BYOL pin
`dataset_overrides` so the positive pair is two time warps of a single window** --
the same view LeJEPA's "Time Warping" row uses, not an image-style crop. The
`(Time Warping)` suffix that used to say so was removed on request. The prose
has to carry it now.

## Where the inputs live -- and why they are not here

`paths.py` names every input. The files themselves stay beside the code that
produces them, on purpose: each is written in place and several are read by
other consumers under a `<dir>/<stem><tag>.json` convention. Moving one here
would leave its producer writing to the old path and this folder holding a
stale copy that still *exists*, so nothing would raise. For the fixed-panel
JSONs the producer includes `scripts/pythia/slurm_tsfm_latent.sh`, which
hardcodes the `plots/latent_eval` path and runs on pythia from a checkout that
may sit at a different commit.

**`/data/lab/probe_breadth/` is outside the repo and is pinned by nothing in
it.** Both probe artifacts read it directly, so neither the figure nor the
table is reproducible from a clean checkout alone.

`plots/style.py` also stays put: ~15 importers outside `plots/`, including
`tests/` and `scripts/`. It is repo infrastructure, not core presentation.

## What is NOT junk

`plots/metrics/` looks retired and mostly is not. The horizon axis was removed
*from the scripts*, not the scripts from the repo, so what is left is live:

- `metrics.py` exports `SUP_SPAN_NO_SPAN`, `SERIES_DEFS` and `load_ic_metrics`,
  used by `run_6mo_eval.sh`, `run_meanpool_reeval.sh`,
  `build_multihead_manifest.py`, `scripts/eval/build_probe_breadth_manifest.py`
  and `plots/task_corr/task_rank_corr._plot_ic()` (a **function-local** import, so
  breaking its path does not fail until that code path runs).
- `lejepa_augs.py` is the numeric source for `plots/augmentations/build_artifact.py`.
- `mechanical_baseline.py` is the reference `panel_tables.py:290` asserts against.
- `all_metrics_*.png`, `tsfm_ic.py` are referenced from three live READMEs.

Checked 2026-09-17: nothing in `plots/metrics/` was archivable.
