# Towards Financial World Modeling (TFWM)

Code for *Towards Financial World Modeling*: 18 encoders pretrained on 1 Hz US
equity market data, compared by how well a linear probe on their frozen
features ranks the cross-section of stocks on three 15-minute-ahead targets:
return, change in realised volatility, and change in quoted spread.

- **Data:** [Market-1T](https://huggingface.co/fin-ai-lab) on the Hugging Face
  Hub, July 2019 to December 2020 (230,025 ticker-days).
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
`plots/core/fixed_panel_table.tex` and `plots/core/probe_fit_breadth{,_floor}.png`
from the result files checked into this repo. The per-checkpoint probe
results are in `plots/core/probe_breadth_snapshot.json.gz`, and the head,
latent and RankMe results are in `plots/metrics/` and `plots/latent_eval/`.
`plots/core/README.md` explains how the tables are assembled.

## Train an encoder

The default machine config, `machine=hub`, reads Market-1T straight from the
Hub and downloads only the months a run needs, into `$HF_HOME`. Log in
first (`hf auth login`), because anonymous downloads are rate-limited. Every
default in `market_jepa/schemas.py` is the reported recipe, so a run names
only its method and dates. To reproduce the 2020-01 evaluation month (trained
on 2019-07 to 2019-12):

```bash
DATES="dataset.train_date_start=2019-07-01 dataset.train_date_end=2019-12-31 \
       dataset.eval_train_date_start=2019-07-01 dataset.eval_train_date_end=2019-12-31 \
       dataset.eval_date_start=2020-01-01 dataset.eval_date_end=2020-01-31"

# LeJEPA, time-warp pairing
uv run train.py mode=lejepa mode.lamb=0.001 dataset.augmentations.0.name=time_warp $DATES

# A self-supervised baseline: the mode is the whole arm
uv run train.py mode=ts2vec $DATES

# Supervised specialist; supervised arms train from the day store
uv run train.py mode=supervised mode.task=return_900 dataset.backend=days $DATES

# Supervised multihead
uv run train.py mode=multi_supervised \
    mode.tasks=[return_900,volatility_change_900,spread_change_900] dataset.backend=days $DATES
```

Each arm's exact overrides are in `scripts/sweeps/`. The LeJEPA pairings and
the SSL baselines are in `ssl_lejepa_all.sh`, the specialists in
`supervised_specialists.sh`, and the multihead in `supervised_multihead.sh`.
Runs log to Weights & Biases. Set `WANDB_MODE=offline` or pass
`wandb.entity=<yours>`.

Of the five evaluation months with released encoders, 2020-01, 2020-08,
2020-09 and 2020-12 can be retrained from the released data. The 2019-09
encoders trained on 2019-03 to 2019-08, which starts before the released
window. The paper's tables average over 31 evaluation months from 2008 to
2023. The other 26 are not publicly released.

## Score an encoder

The probe fits a ridge per target on the encoder's six training months
(36 anchors a day) and scores cross-sectional rank IC on the evaluation month
(8 anchors a day). Scoring reads local copies of the data. Download them with
a Hugging Face login (`hf auth login`, or set `HF_TOKEN`), because anonymous
requests are rate-limited well below what a few months of shards need.
Check that the download is complete, since a rate-limited
`snapshot_download` can return without error with files missing.

```python
from huggingface_hub import snapshot_download

months = ["2019/07", "2019/08", "2019/09", "2019/10", "2019/11", "2019/12", "2020/01"]
for repo, sub in [("Market-1T-1Hz-2019H2-2020-dense", "1Hz_mosaic_mnth"),
                  ("Market-1T-1Hz-2019H2-2020-sparse", "1Hz_mosaic_mnth_sparse")]:
    snapshot_download(f"fin-ai-lab/{repo}", repo_type="dataset", local_dir="market1t",
                      allow_patterns=[f"{sub}/{m}/*" for m in months])
snapshot_download("fin-ai-lab/tfwm-lejepa-time-warp", allow_patterns=["2020-01/*"],
                  local_dir="encoders/lejepa-time-warp")
```

Build the cross-sectional target tables, embed, and fit:

```bash
uv run sf-build-targets --mosaic-dir market1t/1Hz_mosaic_mnth_sparse \
    --out-dir targets --start 2019-07 --end 2020-01 --holiday-csv data/market_holidays.csv

CKPT=encoders/lejepa-time-warp/2020-01
ARGS="--ckpt $CKPT --pool last --mosaic-dir market1t/1Hz_mosaic_mnth \
      --xs-anchor-stats-dir targets --out-dir cache/warp"
for m in 2019-07 2019-08 2019-09 2019-10 2019-11 2019-12; do
    uv run scripts/eval/probe_fit_size.py embed $ARGS --month $m --anchors-per-day 36
done
uv run scripts/eval/probe_fit_size.py embed $ARGS --month 2020-01 --anchors-per-day 8
uv run scripts/eval/probe_fit_size.py reduce --out-dir cache \
    --eval-month-glob 2020-01 --alphas 10 --json results.json
```

`--pool last` is the forecasting readout the paper reports, and the latent
analyses use `--pool mean`. The largest-n row of `results.json` is the number
in the paper's table.

## Layout

| Path | Contents |
|---|---|
| `market_jepa/` | Models, objectives, training loop, evaluation |
| `stable-finance/` | Data pipeline: sharding, dense grids, day store, targets (submodule) |
| `train.py` | Hydra entry point; `market_jepa/schemas.py` defines every option |
| `scripts/sweeps/` | The exact arms behind every reported method |
| `scripts/eval/` | Probe scoring, RankMe, FLOPs, checkpoint audits |
| `scripts/pythia/`, `scripts/generic/` | Our SLURM launchers, kept for reference; paths are site-specific |
| `plots/core/` | The paper's main tables and figure |
| `plots/` | Every other figure, next to the script that draws it |
| `data/` | Market holiday calendars and the Fama-French 49 industry map |

## Data notice

`data/industry_map.parquet` and `data/char_table_2023-01.parquet` assign
tickers to Fama-French 49 industries. The assignments were derived from SIC
codes in Compustat, and the Compustat records themselves are not included.
The market data is released on the Hub under the terms on its dataset cards.

## Citation

```bibtex
@misc{tfwm2026,
  title  = {Towards Financial World Modeling},
  author = {TODO},
  year   = {2026},
}
```

## License

Code: MIT (see `LICENSE`). Data and weights: see their Hugging Face cards.
