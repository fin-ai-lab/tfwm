# Sweep files

A sweep file names **the method and what is being swept, and nothing else.**

Everything a run needs beyond that — learning rate, batch size, epochs, loss
function, the dataset's train/eval targets, the encoder's readout and position
encoding, live-eval — is a **default in `market_jepa/schemas.py`**, and the
sweep file must not restate it.

## Why

The obvious argument runs the other way, and this repo believed it until
2026-09-08. It says: a sweep is a scientific record, so pin every knob, because
a default that moves mid-flight silently changes the arms. That really did
happen — the bins × penalty sweep lost a set of arms when
`soft_label_temperature` moved 0.0 → 0.5 and the later arms of already-running
jobs picked up the new value with nothing in the run name to say so.

But pinning does not prevent drift. It makes drift **silent and undetectable**:

- `schemas.py` moves to the new recipe.
- The sweep file still says the old one, so it keeps training the old recipe.
- Nothing errors. Nothing warns. The run names are unchanged.
- The figures now mix two recipes and the only evidence is inside each
  checkpoint's `train_meta.json`.

That is strictly worse than the failure it was meant to prevent, because a
moving default at least moves *everything* in one direction. `variance_decomp.sh`
is the worked example: it pinned `blr 1e-4 @ bs128`, the retired binned recipe,
and would have measured the seed spread of a model the paper no longer reports.
`full_data_supervised.sh` pinned the same values and, being the file that
*produced* the reported numbers, happened to stay correct — by luck, not design.

So the rule inverts: **one place defines the recipe, and it is `schemas.py`.**
A sweep that reads a default tracks the reported model by construction.

## What that costs, stated plainly

A sweep's meaning now moves when `schemas.py` moves. That is intended — the
sweep measures *the reported model, whatever it currently is* — but it means a
defaults change mid-wave splits the sample. Two mitigations, both cheap:

1. Every run records its fully resolved config in `train_meta.json`. Check
   those before pooling runs across a wave that straddles a `schemas.py` commit.
2. Commit `schemas.py` before launching. The wandb project is stamped with
   `git rev-parse --short=6 HEAD`, so an uncommitted recipe change makes two
   different recipes share one project name.

## The trap this rule creates

Stripping a line from a sweep file is only safe if the default **actually is**
what you stripped. Verify it, don't assume it.

This bit us immediately. `training.live_eval=false` was stripped from
`full_data_supervised.sh` on the belief it had become a default. It had not:
the change had been made to `LiveEvalConfig.enabled`, a dataclass **nothing
reads**, while the switch `pretrain.py:234` actually consults is
`TrainingConfig.live_eval`, still `True`. The net effect was to silently turn
live eval back on — costing dataloader cores on every future run, with no error.

Verify a default the way the trainer resolves it, not by reading the field you
think is the right one:

```bash
uv run python train.py mode=supervised mode.task=return_900 \
    backbone=transformer --cfg job --resolve | less
```

Fields that resolve to `null` there are not unset — `blr`, batch size, epochs
and the encoder's `pool`/`pos_embed` are filled at runtime from the mode's
`training_overrides` / `backbone_overrides` under
`EXPLICIT PIN > MODE OVERRIDE > fallback` (`pretrain.py:448`, `:742`). Read
those dataclasses to see what a stripped sweep will inherit.

## What still belongs in a sweep file

- The mode and the task: `mode=supervised mode.task=spread_change_900`.
- The backbone config group: `backbone=transformer`.
- **The swept axis itself.** If the sweep varies λ, λ is named here.
- Values that are genuinely not the default and are the point of the arm —
  e.g. LeJEPA's `cross_stock` K=2 / FF49-industry pairing and `lamb=0.01`.
- wandb project / group / run name.

## What does not

- `optimizer.blr`, `training.per_device_train_batch_size`,
  `training.num_epochs`, `training.max_train_steps`
- `mode.loss_fn`
- `dataset.xs_target`, `dataset.xs_eval_target`, `dataset.n_pairs_per_obs`,
  `dataset.risk_factor_tickers`
- `training.live_eval`
- the encoder's `pool` / `pos_embed` (`last` / `sinusoidal` for supervised)

## A run name may still quote a default

`supervised_specialists.sh` (renamed from `full_data_supervised.sh`) builds
`wandb.run_name=${YM}_pairwise_blr${BLR}` from a local `BLR` that is **never
passed to training** and that now **resolves from `schemas.py`**
(`SupervisedModeConfig.training_overrides.blr`) instead of being a literal —
the literal `1e-5` it used to carry outlived the recipe, and a stale LR in a
run name is worse than no LR at all, because the name is what gets read years
later and believed. The name records which recipe produced the run; the recipe
comes from `schemas.py`. Set `BLR=...` only to label a deliberate one-off.
