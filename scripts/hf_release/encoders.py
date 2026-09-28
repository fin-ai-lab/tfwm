"""Stage, card, upload and verify the TFWM encoder repos on the Hugging Face Hub.

One repo per method (``fin-ai-lab/tfwm-<slug>``), one folder per evaluation
month. The checkpoints come from the probe-breadth manifest, never a glob:
ts2vec and timemae were resubmitted under different hashes, and only the
manifest knows which run a month means.

    uv run scripts/hf_release/encoders.py stage  --months 2019-09 2020-01 ...
    uv run scripts/hf_release/encoders.py cards
    HF_TOKEN=... uv run scripts/hf_release/encoders.py upload
    HF_TOKEN=... uv run scripts/hf_release/encoders.py verify

``stage`` hardlinks into ``--out`` (same filesystem as the checkpoints, so it
costs no space) and is additive: staging more months later adds folders beside
the existing ones. ``cards`` rewrites every README from what is staged, so the
month table always matches the folders. Run ``upload`` alone -- in parallel
with another upload it trips the Hub's 2500-requests-per-5-minutes quota.

See docs/hf_release.md for what is released and why.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ORG = "fin-ai-lab"
MANIFEST = Path("/data/lab/probe_breadth/manifest_all.index.json")
STAGE = Path("/data/lab/market1t-hf-release/v2/encoders")
COLLECTION = f"{ORG}/tfwm-pre-trained-encoders-6ab871e942535b9c6041698d"
DENSE = f"{ORG}/Market-1T-1Hz-2019H2-2020-dense"
DAYSTORE = f"{ORG}/Market-1T-1Hz-2019H2-2020-daystore"
# The months Market-1T may publish. A checkpoint whose training span leaves
# this window gets a "cannot be retrained from the released data" note.
RELEASED = ("2019-07", "2020-12")

# (series_key, slug, label, family, description, files, base LR, batch)
_SUP = ("Encoder trained end-to-end with an MLP head to rank the cross-section of stocks on {} "
        "over the next 900 s (pairwise ranking loss). Each training cell is one (day, 5-minute "
        "anchor) with 16 stocks drawn from the same cross-section; the target is computed from "
        "data strictly after the anchor.")
_LEJ = ("LeJEPA (joint-embedding predictive architecture with the SIGReg isotropic-Gaussian "
        "regulariser) with two global and six local views per sample. ")
_CFG = "`config.json` + `model.pt`"
METHODS = [
    ("sup_return_w8", "supervised-return", "Supervised (return)", "Supervised",
     _SUP.format("forward VWAP return"), "`backbone.pt` + `head.pt`", 2e-4, 256),
    ("sup_vol_w8", "supervised-vol", "Supervised (vol)", "Supervised",
     _SUP.format("the change in realised volatility"), "`backbone.pt` + `head.pt`", 2e-4, 256),
    ("sup_spread_w8", "supervised-spread", "Supervised (spread)", "Supervised",
     _SUP.format("the change in quoted spread"), "`backbone.pt` + `head.pt`", 2e-4, 256),
    ("sup_multi_w8", "supervised-multihead", "Supervised (multihead)", "Supervised",
     "One shared encoder with three heads (return, volatility change, spread change, all at "
     "900 s), trained jointly with gradient-norm balancing. Otherwise identical to the "
     "single-task supervised encoders.", "`backbone.pt` + `heads.pt`", 2e-4, 256),
    ("pair_rrc_6mo", "lejepa-same-stock", "LeJEPA Same Stock, Diff. View", "LeJEPA",
     _LEJ + "Views are random resized crops of the same stock-day.", _CFG, 6e-5, 256),
    ("pair_warp_6mo", "lejepa-time-warp", "LeJEPA Time Warping", "LeJEPA",
     _LEJ + "Views are two independent smooth time warps of one window.", _CFG, 6e-5, 256),
    ("pair_noise_6mo", "lejepa-gaussian-noise", "LeJEPA Gaussian Noising", "LeJEPA",
     _LEJ + "Views are Gaussian-noised copies of one window.", _CFG, 6e-5, 256),
    ("pair_k2_6mo", "lejepa-cross-stock", "LeJEPA Cross Stock", "LeJEPA",
     _LEJ + "Views come from two different stocks at the same time of the same day.", _CFG, 6e-5, 256),
    ("pair_k2ind_6mo", "lejepa-cross-stock-industry", "LeJEPA C-S, Same Industry", "LeJEPA",
     _LEJ + "Views come from two different stocks in the same Fama-French 49 industry, at the "
     "same time of the same day. The industry map is derived from licensed data and is not "
     "released.", _CFG, 6e-5, 256),
    ("dino_6mo", "dino", "DINO", "SSL",
     "DINO self-distillation (Caron et al., 2021). The positive pair is two time warps of a "
     "single window, the same view as LeJEPA Time Warping.",
     _CFG + " (`student`, `ema`, `center`)", 5e-4, 128),
    ("byol_6mo", "byol", "BYOL", "SSL",
     "BYOL (Grill et al., 2020). The positive pair is two time warps of a single window, the "
     "same view as LeJEPA Time Warping.", _CFG, 5e-4, 128),
    ("cpc_6mo", "cpc", "CPC", "SSL",
     "Contrastive Predictive Coding (van den Oord et al., 2018): a GRU summarises past patches "
     "and predicts future patch embeddings against negatives.", _CFG, 1.5e-4, 2048),
    ("ijepa_6mo", "ijepa", "I-JEPA", "SSL",
     "I-JEPA (Assran et al., 2023): a predictor regresses EMA-target embeddings of masked patch "
     "blocks from a context block (smooth-L1).",
     "`model.pt` (`backbone`, `ema`, `predictor`) — no `config.json`; rebuild from "
     "`train_meta.json`", 1e-4, 2048),
    ("mae_6mo", "mae", "MAE", "SSL",
     "Masked autoencoder (He et al., 2022), 75% of patches masked.", _CFG, 5e-4, 2048),
    ("ts2vec_6mo", "ts2vec", "TS2Vec", "SSL",
     "TS2Vec (Yue et al., 2022): hierarchical contrastive learning over timestamps and "
     "instances.", _CFG + " (`backbone`, `swa_backbone`)", 1e-3, 128),
    ("cost_6mo", "cost", "CoST", "SSL",
     "CoST (Woo et al., 2022): contrastive learning of disentangled seasonal and trend "
     "representations.", _CFG, 1e-3, 128),
    ("tfc_6mo", "tfc", "TF-C", "SSL",
     "TF-C (Zhang et al., 2022): time-frequency consistency between a time-domain and a "
     "frequency-domain encoder.", _CFG + " (`backbone`, `freq_backbone`, projectors)", 3e-4, 128),
    ("timemae_6mo", "timemae", "TimeMAE", "SSL",
     "TimeMAE (Cheng et al.): masked-patch reconstruction and representation alignment, 60% of "
     "patches masked.", _CFG, 1e-3, 128),
]
BY_KEY = {m[0]: m for m in METHODS}


def stage(months: list[str], out: Path) -> None:
    rows = [r for r in json.loads(MANIFEST.read_text())
            if r["series_key"] in BY_KEY and r["eval_month"] in months]
    found = {(r["series_key"], r["eval_month"]) for r in rows}
    missing = [(k, m) for k in BY_KEY for m in months if (k, m) not in found]
    if missing:
        raise SystemExit(f"not in the manifest: {missing}")
    for r in rows:
        dst = out / f"tfwm-{BY_KEY[r['series_key']][1]}" / r["eval_month"]
        dst.mkdir(parents=True, exist_ok=True)
        for f in os.listdir(r["ckpt_dir"]):
            if not (dst / f).exists():
                os.link(Path(r["ckpt_dir"]) / f, dst / f)
    print(f"staged {len(rows)} checkpoints under {out}")


def _span(month_dir: Path) -> tuple[str, str]:
    ds = json.loads((month_dir / "train_meta.json").read_text())["config"]["dataset"]
    return ds["train_date_start"][:7], ds["train_date_end"][:7]


def model_card(method, repo_dir: Path) -> str:
    key, slug, label, fam, desc, files, blr, batch = method
    sup = fam == "Supervised"
    months = sorted(p.name for p in repo_dir.iterdir() if p.is_dir() and p.name[:2] in ("19", "20"))
    rows = []
    for m in months:
        a, b = _span(repo_dir / m)
        note = ("Training span is **outside the released data** (Market-1T covers "
                f"{RELEASED[0]} → {RELEASED[1]}); this encoder cannot be retrained from it."
                if a < RELEASED[0] or b > RELEASED[1] else "")
        rows.append(f"| `{m}/` | {a} → {b} | {m} | {note} |")
    repo_data, sub = (DAYSTORE, "1Hz_daystore") if sup else (DENSE, "1Hz_mosaic_mnth")
    kind = ("day-major: one trading day per record, every ticker plus precomputed targets, so each "
            "training cell is a same-day cross-section" if sup else
            "one ticker-day per record on the filled 1 Hz grid, shuffled within each month")
    override = "dataset.backend=days machine.daystore_dir" if sup else "machine.mosaic_dir"
    weights = "backbone.pt" if sup else "model.pt"
    readout_note = ("not applicable: supervised checkpoints have no `config.json`" if sup
                    else "`mean` for every self-supervised encoder")
    head_note = ("\nThe heads are the trained forecasting heads the paper reports; `xs_ic.json` is "
                 "the head's cross-sectional IC on the evaluation month." if sup else "")
    table = "\n".join(["| Folder | Trained on (6 months) | Evaluated on | Note |",
                       "|---|---|---|---|", *rows])
    return f"""---
license: other
license_name: derived-market-data
library_name: pytorch
pipeline_tag: feature-extraction
tags:
  - finance
  - time-series
  - market-data
  - {"supervised" if sup else "self-supervised"}
  - tfwm
datasets:
  - {repo_data}
---

# TFWM encoder — {label}

> ⚠️ **Pre-release.** These weights and the code that loads them are a work in
> progress. Contents and layout may change without notice. The training code
> (`market_jepa`, `stable_finance`) is not public yet.

**{fam}.** {desc}

One of 18 encoders compared in *Towards Financial World Modeling* (TFWM). All 18
share the same backbone and are trained for 12 passes over the same six-month
spans, so they differ mainly in the training objective. See the
[TFWM Pre-Trained Encoders](https://huggingface.co/collections/{COLLECTION}) collection for the others.

## Checkpoints

One checkpoint per evaluation month. Each was trained on the six months
immediately before it and never saw the evaluation month.

{table}

Each folder holds {files}, plus `train_meta.json` (the full resolved training
config, the training span and the view-normalisation settings).{head_note}

## Architecture and training

| | |
|---|---|
| Backbone | Transformer, 12 layers, width 384, 6 heads, MLP 1536, patch 8, sinusoidal positions (~22M parameters) |
| Input | 1 Hz regular-session US equity data: 9 market channels (`bid_price, vwap_all, high, low, ask_price, bid_size, ask_size, volume, n`) + 11 view-information channels computed at load time (per-view normalisation statistics and window geometry) = 20 channels |
| Training data | [`{repo_data}` → `{sub}/`](https://huggingface.co/datasets/{repo_data}/tree/main/{sub}) ({kind}). Train from it with `{override}=hf://datasets/{repo_data}/{sub}` |
| Schedule | 12 passes over the 6-month span, base LR {blr:g}, weight decay 0.05, {"effective " if sup else ""}batch {batch} |
| Pooling in `config`/training | `{"last" if sup else "mean"}` |

## Readout

The paper reads every encoder two ways:

- **Forecasting probes:** the embedding of the **last** patch (`pool="last"`), i.e. the state at the decision time.
- **Latent analyses:** the **mean** over patches (`pool="mean"`).

Loading a checkpoint through a mode class's `from_pretrained` uses the pool stored
in `config.json` ({readout_note}) and ignores any pool
you pass in a separate config. To get the last-patch readout, set `.pool = "last"` on
every sub-backbone after loading (`backbone`, and also `swa_backbone` for TS2Vec and
`freq_backbone` for TF-C).

## Usage

Download one month:

```python
from huggingface_hub import snapshot_download

path = snapshot_download("{ORG}/tfwm-{slug}", allow_patterns=["{months[-1]}/*"])
ckpt = f"{{path}}/{months[-1]}"
```

With the project code (release forthcoming):

```python
from market_jepa.eval.checkpoints import load_encoder

encoder = load_encoder(ckpt, pool="last")   # or pool="mean"
```

Without it, the files are plain PyTorch state dicts:

```python
import torch

state = torch.load(f"{{ckpt}}/{weights}", map_location="cpu", weights_only=True)
```
"""


def cards(out: Path) -> None:
    for method in METHODS:
        repo_dir = out / f"tfwm-{method[1]}"
        if repo_dir.is_dir():
            (repo_dir / "README.md").write_text(model_card(method, repo_dir))
    print("cards written")


def _retry(fn, *a, **k):
    from huggingface_hub.errors import HfHubHTTPError

    for _ in range(12):
        try:
            return fn(*a, **k)
        except HfHubHTTPError as e:
            if "429" not in str(e):
                raise
            print("rate limited (429); sleeping 310 s", flush=True)
            time.sleep(310)
    raise RuntimeError("still rate limited after an hour")


def upload(out: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    for method in METHODS:
        repo, folder = f"{ORG}/tfwm-{method[1]}", out / f"tfwm-{method[1]}"
        if not folder.is_dir():
            continue
        _retry(api.create_repo, repo, repo_type="model", private=False, exist_ok=True)
        # Resumable, and skips files already on the Hub.
        api.upload_large_folder(repo_id=repo, repo_type="model", folder_path=folder,
                                num_workers=8, print_report=False)
        _retry(api.add_collection_item, COLLECTION, item_id=repo, item_type="model", exists_ok=True)
        print("uploaded", repo, flush=True)


def verify(out: Path) -> None:
    from huggingface_hub import HfApi

    api, bad = HfApi(), 0
    for method in METHODS:
        repo, folder = f"{ORG}/tfwm-{method[1]}", out / f"tfwm-{method[1]}"
        if not folder.is_dir():
            continue
        local = {p.relative_to(folder).as_posix(): p.stat().st_size for p in folder.rglob("*")
                 if p.is_file() and ".cache" not in p.parts}
        remote = {f.path: f.size for f in _retry(api.list_repo_tree, repo, recursive=True)
                  if getattr(f, "size", None) is not None}
        wrong = [k for k in local if remote.get(k) != local[k]]
        bad += len(wrong)
        print(f"{repo:50s} {len(local):4d} files, {len(wrong)} missing or wrong size {wrong[:3]}")
    raise SystemExit(1 if bad else 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["stage", "cards", "upload", "verify"])
    p.add_argument("--months", nargs="+", help="eval months (YYYY-MM) to stage")
    p.add_argument("--out", type=Path, default=STAGE)
    args = p.parse_args()
    if args.command == "stage":
        if not args.months:
            p.error("stage needs --months")
        stage(args.months, args.out)
    else:
        {"cards": cards, "upload": upload, "verify": verify}[args.command](args.out)


if __name__ == "__main__":
    main()
