# Towards Financial World Modeling (TFWM)

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)](https://openreview.net/forum?id=OEokq7iASc)
[![arXiv](https://img.shields.io/badge/arXiv-2610.09048-b31b1b.svg)](https://arxiv.org/abs/2610.09048)
[![Encoders](https://img.shields.io/badge/%F0%9F%A4%97%20Encoders-TFWM-ffd21e.svg)](https://huggingface.co/collections/fin-ai-lab/tfwm-pre-trained-encoders-6ab871e942535b9c6041698d)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Market--1T-ffd21e.svg)](https://huggingface.co/fin-ai-lab)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Code for *Towards Financial World Modeling*: 18 encoders pretrained on 1 Hz US
equity market data, compared by how well a linear probe on their frozen
features ranks the cross-section of stocks on three 15-minute-ahead targets:
return, change in realised volatility, and change in quoted spread.

- **Data:** A sample of [Market-1T](https://huggingface.co/fin-ai-lab) is on Hugging Face
  Hub, July 2019 to December 2020 (230,025 ticker-days).
  - Code to assemble the full dataset: [https://github.com/fin-ai-lab/market-1t](https://github.com/fin-ai-lab/market-1t)
  - [Massive](https://massive.com) is working on hosting an open and complete version of `Market-1T`
- **Pretrained encoders:** [TFWM Pre-Trained Encoders](https://huggingface.co/collections/fin-ai-lab/tfwm-pre-trained-encoders-6ab871e942535b9c6041698d),
  one repo per method.

The 18 methods are 4 supervised arms (three single-target specialists and a
multihead), 5 LeJEPA positive pairings (same-stock crops, time warp, Gaussian
noise, cross-stock, cross-stock within an industry) and 9 self-supervised
baselines (DINO, BYOL, CPC, I-JEPA, MAE, TS2Vec, CoST, TF-C, TimeMAE). All 18
share one transformer backbone (12 layers, width 384, ~22M parameters) and one
schedule, 12 passes over a six-month span, so they differ mainly in the
training objective.

## Install

```bash
git clone --recursive https://github.com/fin-ai-lab/tfwm
cd tfwm
uv sync
```

[uv](https://docs.astral.sh/uv/) is the only supported installer. `--recursive`
fetches [stable-finance](https://github.com/galilai-group/stable-finance), the
data pipeline, which `uv sync` installs from the submodule. Training needs a
CUDA 12.8 GPU. The paper's tables and figures, and the test suite, run on CPU.

## Reproduce the paper's tables from the released results

```bash
plots/core/regen.sh
uv run pytest
```

`regen.sh` rebuilds `plots/core/probe_fit_table.tex`,
`plots/core/fixed_panel_table.tex` and `plots/core/probe_fit_breadth.png`
from the result files checked into this repo. The per-checkpoint probe
results are in `plots/core/probe_breadth_snapshot.json.gz`, and the head,
latent and RankMe results are in `plots/metrics/` and `plots/latent_eval/`.
`plots/core/README.md` explains how the tables are assembled.

## Replicate a released encoder

Two scripts in `examples/` do the whole loop on one GPU. Each one downloads
the data, trains the reported recipe for one evaluation month, scores the new
checkpoint and the released one with the paper's scorer on the same machine,
and checks the two agree:

```bash
hf auth login                                  # downloads wait out Hub rate limits
uv run examples/supervised_spread.py           # supervised spread change
uv run examples/lejepa_time_warp.py            # LeJEPA, time-warp pairing
uv run examples/lejepa_time_warp.py --smoke    # 50 steps: checks the pipeline in minutes
```

Supervised spread change is the tightest check. Its seed-to-seed standard
deviation is about 1% of its IC (`plots/variance_decomp/`), so a 5% band is
roughly four standard deviations wide. The LeJEPA script gates on volatility
change and spread change and reports return without gating on it, because
return's IC (~0.01) is small enough that a 5% band sits inside run-to-run
noise.

## Train an encoder

Every default in `market_jepa/schemas.py` is the reported recipe, so a run
names only its method and its dates. The default machine config,
`machine=market1t`, downloads the months a run needs into `market1t/` before
training starts. It verifies that every file arrived, decompresses the day
store, and builds the cross-sectional target tables the released encoders
trained with. Months already on disk are not downloaded again. Log in first
(`hf auth login`). The Hub allows 1,000 API requests per 5 minutes per
account, and one evaluation month is about 2,500 files, so the first download
pauses for rate limits and resumes on its own.

To reproduce the 2020-01 evaluation month (trained on 2019-07 to 2019-12):

```bash
DATES="dataset.train_date_start=2019-07-01 dataset.train_date_end=2019-12-31 \
       dataset.eval_train_date_start=2019-07-01 dataset.eval_train_date_end=2019-12-31 \
       dataset.eval_date_start=2020-01-01 dataset.eval_date_end=2020-01-31"

# LeJEPA, time-warp pairing
uv run train.py mode=lejepa mode.lamb=0.001 dataset.augmentations.0.name=time_warp $DATES

# A self-supervised baseline: the mode is the whole arm
uv run train.py mode=ts2vec $DATES

# Supervised specialist; supervised arms train from the day store
uv run train.py mode=supervised mode.task=spread_change_900 dataset.backend=days $DATES

# Supervised multihead
uv run train.py mode=multi_supervised \
    mode.tasks=[return_900,volatility_change_900,spread_change_900] dataset.backend=days $DATES
```

`machine=hub` streams from the Hub instead of downloading. It then needs
`dataset.xs_anchor_stats_dir` pointed at target tables, which it does not
build.

Each arm's exact overrides are in `scripts/sweeps/`. The LeJEPA pairings and
the SSL baselines are in `ssl_lejepa_all.sh`, the specialists in
`supervised_specialists.sh`, and the multihead in `supervised_multihead.sh`.
Runs log to Weights & Biases. Set `WANDB_MODE=offline` or pass
`wandb.entity=<yours>`.

Of the five evaluation months with released encoders, 2020-01, 2020-08,
2020-09 and 2020-12 can be retrained from the released data. The 2019-09
encoders trained on 2019-03 to 2019-08, which starts before the released
window. The paper's tables average over 32 evaluation months from 2008 to
2023. The other 27 are not publicly released.

## Score an encoder

The probe fits a ridge per target on the encoder's six training months
(36 anchors a day) and scores cross-sectional rank IC on the evaluation month
(8 anchors a day). Fetch the data and target tables for those seven months
(the same download training does), and one released encoder:

```bash
uv run python -m market_jepa.market1t 2019-07 2020-01 --layouts dense
uv run python -c "from huggingface_hub import snapshot_download as s; \
    s('fin-ai-lab/tfwm-lejepa-time-warp', allow_patterns=['2020-01/*'], local_dir='encoders/warp')"
```

Embed and fit:

```bash
ARGS="--ckpt encoders/warp/2020-01 --pool last --mosaic-dir market1t/1Hz_mosaic_mnth \
      --xs-anchor-stats-dir market1t/xs_anchor_stats_fwdvwap60 --out-dir cache/warp"
for m in 2019-07 2019-08 2019-09 2019-10 2019-11 2019-12; do
    uv run scripts/eval/probe_fit_size.py embed $ARGS --month $m --anchors-per-day 36
done
uv run scripts/eval/probe_fit_size.py embed $ARGS --month 2020-01 --anchors-per-day 8
uv run scripts/eval/probe_fit_size.py reduce --out-dir cache \
    --eval-month-glob 2020-01 --alphas 10 --json results.json
```

Each `embed` is single-process. The examples split the work across processes
with `embed-many`, which is much faster.

`--pool last` is the forecasting readout the paper reports, and the latent
analyses use `--pool mean`. The largest-n row of `results.json` is the number
in the paper's table.

## Layout

| Path | Contents |
|---|---|
| `market_jepa/` | Models, objectives, training loop, evaluation |
| `stable-finance/` | Data pipeline: sharding, dense grids, day store, targets (submodule) |
| `examples/` | End-to-end replication of a released encoder, with a parity check |
| `train.py` | Hydra entry point; `market_jepa/schemas.py` defines every option |
| `scripts/sweeps/` | The exact arms behind every reported method |
| `scripts/eval/` | Probe scoring, RankMe, FLOPs, checkpoint audits |
| `scripts/generic/` | SLURM launchers for a single-node GPU cluster, kept for reference |
| `plots/core/` | The paper's main tables and figure |
| `plots/` | Every other figure, next to the script that draws it |
| `data/` | Market holiday calendars and the monthly ticker-to-FF49 map |

The analysis scripts under `plots/` and `scripts/eval/` default to
`machine=local`, which reads the mosaic and checkpoints from `lab/` and the
Polygon snapshots from `polygon/` under the repo root. Point those at your
storage with symlinks, or override the paths with the `MJ_*` environment
variables in `market_jepa/schemas.py`.

## Data notice

`data/industry_map.parquet` gives each ticker's Fama-French 49 industry for
every month from 2007-01 to 2024-12 (columns `month`, `ticker`, `ff49`). The
assignments were derived from SIC codes in Compustat; neither the SIC codes nor
any other Compustat field is included. The market data is released on the Hub
under the terms on its dataset cards.

## Citation

If you use this code, the encoders or Market-1T in your research, please cite our paper:

```bibtex
@inproceedings{merchant2026towards,
  title={Towards Financial World Modeling},
  author={Merchant, Humzah and Guthrie, Alec and Mahns, Simon and Balestriero, Randall and Levy, Bradford},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026},
  url={https://arxiv.org/abs/2610.09048}
}
```

## License

Code: MIT (see `LICENSE`). Data and weights: see their Hugging Face cards.
