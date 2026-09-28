# Frozen-TSFM latent layer sweep

Chronos-2, Kronos and TimesFM 3.0 at every hidden state, on the six latent
tasks (T1–T4 fixed panel, LOAD decode, SUBSP align), over the reported 31
eval months.

```bash
uv run python plots/tsfm_layers/latent_sweep.py --metric multiple  # top-1 / chance
uv run python plots/tsfm_layers/latent_sweep.py --metric pctile    # mean rank percentile
uv run python plots/tsfm_layers/latent_sweep.py --metric rankme    # layer-wise RankMe, one panel
```

RankMe for every layer comes from `scripts/eval/rankme_fullday.py --tsfm --json
plots/tsfm_layers/rankme_tsfm.json`; its last-layer values also feed the
appendix table `plots/latent_eval/rankme/rankme_table.tex`
(`plots/latent_eval/rankme/rankme_table.py`).

| file | what |
|---|---|
| `latent_sweep.py` | the figure; writes `latent_sweep_<metric>.{png,pdf,json}` |
| `optimization_picks.json` | per-(family, task) layer chosen on the 5-month optimization set (`scripts/experiments/holdout_months.py`), re-made 2026-09-25 from `_cmean_opt` by `latent_sweep.py --make-picks` — the stars |
| `merge_latent.py` | reassembles per-month pythia jsons (`scripts/pythia/slurm_tsfm_latent.sh`) |

**Data.** The `_cmean` wave (2026-09-25, `plots/latent_eval/run_cmean_eval.sh`):
`plots/latent_eval/fixed_panel/fixed_panel_P3S2_cmean.json` and
`plots/latent_eval/factors/{decode_loadings,subspace_alignment}_cmean.json`
(series `tsfm_<fam>_cmean_l<L>`). Each TSFM is read at the latent protocol's
mean over valid patches, with the nine per-channel states **averaged** into one
d_model vector -- the prediction evals' readout. The concatenated `_meanpool`
series are retired. Each panel has two
reference lines, both read through `plots/core/fixed_panel_table.py`'s own
loaders: black dashed is the untrained Random ViT, gray dashed the best trained
method in that column of the latent table (named in the panel). Styling is
`plots/style.py` (`SERIES_STYLES` colors, full frames, no titles).

**Stars are optimization-set picks, never this panel's argmax.** They were picked on the
percentile readout, so on the multiple figure a star can sit off its curve's
peak.

**Not the predictive readout.** The prediction tables read the TSFMs at the
last layer, last token, channels averaged, through the probe-breadth sweep
(`scripts/eval/build_probe_breadth_manifest.py --tsfm`). This folder's old IC
layer sweep (`layer_sweep*`, `policy*`, `channel_pool*`) was deleted
2026-09-23 as superseded.
