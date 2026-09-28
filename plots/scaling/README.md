# Supervised scaling

Rank IC against training FLOPs for the supervised specialist at three ViT
widths, on the 31 reported months. Replaces the `plots/scaling_study/`
figures on `origin/scaling-study`, which measured retired targets with a
retired loss on a pre/post-2022 split; nothing from that tree is reused.

| artifact | generator |
|---|---|
| `supervised_scaling.png` (head readout) | `supervised_scaling.py` |
| `metrics/supervised_default_model.json` (the star) | `scripts/eval/default_model_flops.py` |

## Pipeline

1. **LR per scale, holdout set 2** — `scripts/sweeps/supervised_scaling_lr.sh`.
   Small keeps the recipe's 2e-4; tiny and base get a 4-point grid on the five
   set-2 months. Read with `scripts/eval/summarize_supervised_scaling_lr.py`,
   record the winner in `VIT_SCALE_BLR` (`market_jepa/schemas.py`). The ladder
   sweep refuses to run a scale whose entry is still `None`.
2. **The targets** — `scripts/sweeps/supervised_scaling.sh`. One run per
   (month, task, scale): the specialist recipe on a stable schedule (pinned
   128-step warmup, then the peak rate to the end) with a **branch cooldown**
   at every half-decade FLOPs target: the run leaves the stable trajectory,
   decays linearly to 0.1x over the last 10% of that target's steps, saves,
   and is put back (`checkpoint.anneal_steps`, `market_jepa/training/pretrain.py`).
   Targets are 1e16 to 3.33e17 (tiny), 1e18 (small), 3.33e18 (base), as step
   counts per scale (`market_jepa/eval/flops.py:steps_for_flops`); a target
   whose branch point is inside the warmup is dropped for that scale. Every
   checkpoint is scored in-job, head only (`POST_TRAIN_PROBE=0`; the sweep
   refuses to run otherwise).
3. **Collect** — `scripts/eval/collect_supervised_scaling.py` →
   `plots/metrics/supervised_scaling.json`, one row per checkpoint with its
   FLOPs counted from the run's own weights (`market_jepa/eval/flops.py`).
4. **The reported model's compute** — `scripts/eval/default_model_flops.py`
   → `plots/metrics/supervised_default_model.json`. Twelve passes over a
   six-month span is not a FLOPs target, so it is measured per month from
   runs of that exact recipe and averaged in log. Re-run it only when the
   recipe or the reported months change; the figure just reads the file.
5. **Draw** — `uv run plots/scaling/supervised_scaling.py`.

## What the axis is

Training FLOPs = 3 × (forward FLOPs per view) × views seen, with the forward
counted analytically over every matrix product in the backbone and head
(2 per multiply-add; norms, activations, biases and the optimizer excluded).
`tests/test_supervised_scaling.py` holds the count to torch's own
`FlopCounterMode` plus the attention products the counter does not see.
The shape is read from `backbone.pt`, not the config, because the config
records `n_info_channels: 0` on every run.

## Reading it

- The y axis is the trained head's rank IC, the model's own forecast, not
  the ridge probe every other figure reports. No probe is scored in this
  sweep; `--readout probe` exists only for a tree that carries one.

- A solid curve is the mean over months at the targets every month reached;
  the band is 1 s.e. across months. A target is one x for a whole scale
  because a step is the same compute in every month.
- **Every dot is an annealed model** at its own compute, from one run: the
  cooldown branches off the stable run and the run resumes from where it
  left. There is no separate endpoint; the run ends on its last target.
- **The star is the reported model**: the paper's supervised specialist is
  this same ViT-Small at 12 passes over its six months, which costs
  5.07e17 FLOPs on average (3.5-8.0e17 across months, 3,482 steps mean).
  The star is drawn ON the Small curve at that compute -- it marks where the
  paper spends, not what it scored, and it is left off if the curve has not
  reached that far. What it scored: head IC +0.027 return, +0.090 vol
  change, +0.250 spread change over its 31 months, in the same json.
- The runs are the target's length, not the recipe's 12 passes. Past
  3.33e17 a run is longer than the recipe and the six-month pool does not
  grow, so the high end measures more passes over the same data.
- The larger scales start later on the axis: base's 1e16 and 3.33e16 would
  branch inside the warmup and are not run.
- The first wave (2026-09-19, Small only) was the older ladder: raw
  stable-phase rungs under WSD with only the root annealed. The collector
  and figure still read such a tree (`endpoint` rows, hollow marker), but it
  is not pooled with the branch wave (different commit).
