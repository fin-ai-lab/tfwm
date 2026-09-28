"""Stage, card, upload and verify the three Market-1T dataset repos.

Three layouts of the same 230,025 ticker-days, one repo each:

  sparse    1Hz_mosaic_mnth_sparse/  ticker-day, observed seconds only (MDS)
  dense     1Hz_mosaic_mnth/         ticker-day on the filled 1 Hz grid (MDS)
  daystore  1Hz_daystore/            one trading day per record + targets

    uv run scripts/hf_release/datasets.py stage
    uv run scripts/hf_release/datasets.py cards
    HF_TOKEN=... uv run scripts/hf_release/datasets.py upload daystore
    HF_TOKEN=... uv run scripts/hf_release/datasets.py verify

Only compressed bytes ship: MDS shards as ``.zstd``, the day store's features as
``features.npy.zst``. The local sparse ``index.json`` files lost their zstd
declaration in July, so staging rewrites them (every shard has a byte-identical
``.zstd``). Staging hardlinks, so it costs no disk. Upload one repo at a time --
see docs/hf_release.md.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ORG = "fin-ai-lab"
SRC = Path("/data/lab/market-jepa-mosaic")
STAGE = Path("/data/lab/market1t-hf-release/v2")
ENCODERS = f"https://huggingface.co/collections/{ORG}/tfwm-pre-trained-encoders-6ab871e942535b9c6041698d"
MONTHS = [f"2019/{m:02d}" for m in range(7, 13)] + [f"2020/{m:02d}" for m in range(1, 13)]
REPOS = {"sparse": f"{ORG}/Market-1T-1Hz-2019H2-2020-sparse",
         "dense": f"{ORG}/Market-1T-1Hz-2019H2-2020-dense",
         "daystore": f"{ORG}/Market-1T-1Hz-2019H2-2020-daystore"}
DAY_FILES = {"meta.json", "first_row.npy", "count.npy", "features.npy.zst",
             "targets_raw.npy", "targets_zscore.npy", "targets_uniform.npy", "targets_rank.npy"}


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.link(src, dst)


def stage(out: Path) -> None:
    for m in MONTHS:
        s, d = SRC / "1Hz_mosaic_mnth_sparse" / m, out / "sparse" / "1Hz_mosaic_mnth_sparse" / m
        idx = json.loads((s / "index.json").read_text())
        for sh in idx["shards"]:
            z = s / (sh["raw_data"]["basename"] + ".zstd")
            _link(z, d / z.name)
            sh["compression"] = "zstd"
            sh["zip_data"] = {"basename": z.name, "bytes": z.stat().st_size, "hashes": {}}
        (d / "index.json").write_text(json.dumps(idx))

        s, d = SRC / "1Hz_mosaic_mnth" / m, out / "dense" / "1Hz_mosaic_mnth" / m
        idx = json.loads((s / "index.json").read_text())
        for sh in idx["shards"]:
            assert sh["compression"] == "zstd" and sh["zip_data"], (m, sh["raw_data"])
            _link(s / sh["zip_data"]["basename"], d / sh["zip_data"]["basename"])
        _link(s / "index.json", d / "index.json")

        s, d = SRC / "1Hz_daystore" / m, out / "daystore" / "1Hz_daystore" / m
        for day in (e["date"] for e in json.loads((s / "index.json").read_text())["days"]):
            missing = DAY_FILES - {p.name for p in (s / day).iterdir()}
            assert not missing, (day, missing)
            for f in DAY_FILES:
                _link(s / day / f, d / day / f)
        _link(s / "index.json", d / "index.json")
    print("staged", ", ".join(REPOS), "under", out)


_HEAD = """---
license: other
license_name: derived-market-data
task_categories:
  - time-series-forecasting
tags:
  - finance
  - market-data
  - time-series
  - self-supervised
  - jepa
size_categories:
  - 100K<n<1M
viewer: false
---

# Market-1T — 1 Hz, July 2019 → December 2020 ({layout})

> ⚠️ **Pre-release.** This dataset and its accompanying code are a work in
> progress and not yet final. Contents, schema and splits may change without
> notice. The code (`stable_finance`, `market_jepa`) is not public yet.

**230,025 ticker-days** of 1 Hz US equity market data over **376 trading days**
(2019-07-01 → 2020-12-31). The window deliberately straddles the February–March
2020 COVID crash and the recovery through 2020, so it contains a genuine regime
break rather than a single stationary market.

## Three layouts, three repositories

| Repository | One record | Size | Use it for |
|---|---|---|---|
| [`{sparse}`](https://huggingface.co/datasets/{sparse}) | one ticker-day, only the seconds that carry data, **07:00–20:00 ET** (pre-market and after-hours included) | ~27 GB | the raw source: your own fill policy, and the only copy of pre-market / after-hours data |
| [`{dense}`](https://huggingface.co/datasets/{dense}) | one ticker-day on the filled 1 Hz grid, **regular session only (09:30–16:00 ET)** | ~23 GB | self-supervised training and probing (the SSL / LeJEPA encoders trained on this) |
| [`{daystore}`](https://huggingface.co/datasets/{daystore}) | one trading day: every ticker × channel × second, plus precomputed targets, **regular session only (09:30–16:00 ET)** | ~25 GB | cross-sectional / supervised training (the supervised encoders trained on this) |

All three carry the same nine channels:
`bid_price, vwap_all, high, low, ask_price, bid_size, ask_size, volume, n`.
The encoders' 11 extra view-information channels are computed at load time and are not stored.

"""

_TAIL = """
## With the project code

`stable_finance` accepts a Hub path anywhere it takes a local root, and it downloads
only the months a run asks for:

```bash
uv run train.py ... machine.mosaic_dir=hf://datasets/{dense}/1Hz_mosaic_mnth
uv run train.py ... dataset.backend=days machine.daystore_dir=hf://datasets/{daystore}/1Hz_daystore
```

## Related

- [TFWM Pre-Trained Encoders]({encoders}): the encoders trained on this data
"""


def _card(layout: str, body: str) -> str:
    fmt = dict(layout=layout, encoders=ENCODERS, **REPOS)
    return _HEAD.format(**fmt) + body + _TAIL.format(**fmt)


SPARSE_BODY = """## This repository: sparse

[MosaicML Streaming](https://github.com/mosaicml/streaming) (MDS) shards under
`1Hz_mosaic_mnth_sparse/YYYY/MM/`, zstd-compressed, shuffled within each month.
Each record covers **07:00–20:00 ET**: pre-market, the regular session and after-hours.
Only the seconds that carry data are stored. The two dense repositories keep only
09:30–16:00, so **the pre-market and after-hours data exist only here**. That is also where
sparse storage pays off most. Outside the regular session, observed seconds are few and far
apart (in a sample of August 2020 records, the 6.5 extended hours held only ~5% of a record's
rows), and a filled grid there would be almost all padding. Inside the regular session the data
is nearly dense, and the dense layouts compress better (see the size column above).
Some NaNs are retained. Nothing is pre-filled, so the fill policy is yours.

| Field | Type | Meaning |
|---|---|---|
| `ticker` | `str` | Symbol |
| `date` | `str` | `YYYY-MM-DD` |
| `ts_interval` | `int32 (n_steps,)` | Unix time (s) of each row |
| `features` | `float32 (n_steps, 9)` | One row per observed second |

```python
from huggingface_hub import snapshot_download
from streaming import StreamingDataset

root = snapshot_download("{sparse}", repo_type="dataset",
                         allow_patterns=["1Hz_mosaic_mnth_sparse/2020/08/*"])
ds = StreamingDataset(local=f"{{root}}/1Hz_mosaic_mnth_sparse/2020/08", shuffle=True, batch_size=64)
rec = ds[0]            # rec["features"].shape == (n_steps, 9)
```

The dense repository was built from these shards by the project's densifier
(`sf-densify`), which fills the regular session and drops extended hours.
"""

DENSE_BODY = """## This repository: dense ticker-day

**Regular trading hours only: 09:30–16:00 ET, 23,400 one-second rows per ticker-day.**
Pre-market and after-hours seconds are dropped. They exist only in the
[sparse repository](https://huggingface.co/datasets/{sparse}), which covers 07:00–20:00 ET.
This is the window every TFWM encoder was trained and evaluated on.
That is also where the sparse layout pays off most. Outside the regular session,
observed seconds are few and far apart: in a sample of August 2020 records, the 6.5
extended hours held only ~5% of a record's rows. Filled onto a grid, those hours would be
almost all padding.

MDS shards under `1Hz_mosaic_mnth/YYYY/MM/`, zstd-compressed, shuffled within each
month. 2,921 shards.

| Field | Type | Meaning |
|---|---|---|
| `ticker` | `str` | Symbol |
| `date` | `str` | `YYYY-MM-DD` |
| `grid_start` | `int32` | Unix time (s) of row 0 |
| `features` | `float32 (23400, 9)` | The regular session, 09:30–16:00 ET, one row per second |

Fill policy: prices and VWAP are forward-filled across empty seconds, and sizes, volume and trade counts
are zero-filled. A ticker that starts trading late has a trimmed grid (`grid_start` later than the open).

```python
from huggingface_hub import snapshot_download
from streaming import StreamingDataset

root = snapshot_download("{dense}", repo_type="dataset",
                         allow_patterns=["1Hz_mosaic_mnth/2020/08/*"])
ds = StreamingDataset(local=f"{{root}}/1Hz_mosaic_mnth/2020/08", shuffle=True, batch_size=64)
rec = ds[0]            # rec["features"].shape == (23400, 9)
```
"""

DAYSTORE_BODY = """## This repository: day store

**Regular trading hours only: 09:30–16:00 ET, 23,400 one-second rows per ticker.**
Pre-market and after-hours seconds are dropped. They exist only in the
[sparse repository](https://huggingface.co/datasets/{sparse}), which covers 07:00–20:00 ET.
Every target is measured inside this window too.
That is also where the sparse layout pays off most. Outside the regular session,
observed seconds are few and far apart: in a sample of August 2020 records, the 6.5
extended hours held only ~5% of a record's rows. Filled onto a grid, those hours would be
almost all padding.

One directory per trading day, `1Hz_daystore/YYYY/MM/YYYY-MM-DD/`, holding every ticker
that traded that day on one grid. This makes it cheap to draw a same-day
cross-section of stocks, which is what the supervised encoders train on. Each month has an
`index.json` listing its days.

| File | Shape | Meaning |
|---|---|---|
| `features.npy.zst` | `(N_tickers, 9, 23400)` float32, channel-major | zstd copy of `features.npy`; NaN before a ticker's first quote |
| `first_row.npy` | `(N_tickers,)` | First populated row per ticker |
| `targets_{{raw,zscore,uniform,rank}}.npy` | `(78, N_tickers, 3, 6)` float32 | Forward targets at each 5-minute anchor: `return, volatility_change, spread_change` × horizons `300, 600, 900, 1800, 3600, 7200` s, cross-sectionally transformed four ways |
| `count.npy` | `(78, 3, 6)` | Finite names per cross-section |
| `meta.json` | | Tickers, anchors, column names, sizes |

Targets use only data after each anchor. Returns are measured between 60 s forward VWAPs.
The project code decompresses `features.npy.zst` on arrival. By hand:

```python
import json, numpy as np, zstandard
from huggingface_hub import snapshot_download

root = snapshot_download("{daystore}", repo_type="dataset",
                         allow_patterns=["1Hz_daystore/2020/08/*"])
day = f"{{root}}/1Hz_daystore/2020/08/2020-08-03"
with open(f"{{day}}/features.npy.zst", "rb") as f, open(f"{{day}}/features.npy", "wb") as g:
    zstandard.ZstdDecompressor().copy_stream(f, g)
x = np.load(f"{{day}}/features.npy", mmap_mode="r")         # (N_tickers, 9, 23400)
y = np.load(f"{{day}}/targets_uniform.npy", mmap_mode="r")  # (78, N_tickers, 3, 6)
```
"""


def cards(out: Path) -> None:
    for name, layout, body in (("sparse", "sparse", SPARSE_BODY), ("dense", "dense ticker-day", DENSE_BODY),
                               ("daystore", "day store", DAYSTORE_BODY)):
        (out / name / "README.md").write_text(_card(layout, body.format(**REPOS)))
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


def upload(out: Path, names: list[str]) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    for name in names:
        _retry(api.create_repo, REPOS[name], repo_type="dataset", private=False, exist_ok=True)
        api.upload_large_folder(repo_id=REPOS[name], repo_type="dataset", folder_path=out / name,
                                num_workers=16, print_report=False)
        print("uploaded", REPOS[name], flush=True)


def verify(out: Path, names: list[str]) -> None:
    from huggingface_hub import HfApi

    api, bad = HfApi(), 0
    for name in names:
        folder = out / name
        local = {p.relative_to(folder).as_posix(): p.stat().st_size for p in folder.rglob("*")
                 if p.is_file() and ".cache" not in p.parts}
        remote = {f.path: f.size for f in _retry(api.list_repo_tree, REPOS[name], repo_type="dataset",
                                                 recursive=True) if getattr(f, "size", None) is not None}
        wrong = [k for k in local if remote.get(k) != local[k]]
        extra = [k for k in remote if k not in local and k != ".gitattributes"]
        bad += len(wrong) + len(extra)
        print(f"{REPOS[name]:50s} {len(local):5d} files, {len(wrong)} missing/wrong size, "
              f"{len(extra)} on the Hub but not staged {extra[:3]}")
    raise SystemExit(1 if bad else 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["stage", "cards", "upload", "verify"])
    p.add_argument("repos", nargs="*", help=f"any of {', '.join(REPOS)} (default: all)")
    p.add_argument("--out", type=Path, default=STAGE)
    args = p.parse_args()
    names = args.repos or list(REPOS)
    if set(names) - set(REPOS):
        p.error(f"unknown repo(s) {sorted(set(names) - set(REPOS))}; choose from {list(REPOS)}")
    if args.command == "stage":
        stage(args.out)
    elif args.command == "cards":
        cards(args.out)
    elif args.command == "upload":
        upload(args.out, names)
    else:
        verify(args.out, names)


if __name__ == "__main__":
    main()
