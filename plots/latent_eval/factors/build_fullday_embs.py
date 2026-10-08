"""Full-day embeddings into ff_fullday_cache — ONE builder for every family.

The latent-factor analyses (decode_loadings / subspace_alignment)
read one ``emb_<series>.npz`` per encoder from
ff_fullday_cache. Four arms, one grid, one forward convention:

  --families        frozen TSFMs, every layer from one pass
                    (``PretrainedTSFM.compute_features_multi``)
  --sup-families    supervised specialists, every depth from one pass;
                    eval month pairs with the encoder trained the month before
  --series          any trained-encoder key from industry_nn_sweep.MODEL_SPECS
                    (lejepa, k2ind, dino, ts2vec_cb028b, ...), resolved and
                    loaded exactly as fixed_panel_metrics resolves it — glob
                    keys by training month (= eval month - 1), manifest keys
                    by eval month, both through the canonical loaders in
                    market_jepa.eval.checkpoints
  --randvit-seeds   the untrained-ViT floor

This replaced the delta-AUC-era pair build_series_embs.py /
build_variant_embs.py (deleted 2026-08-26): those imported the purged
mass_eval_world_model modules and each carried its own loader copy.

Also BUILDS the month's grid when it does not exist yet. Where a
grid_meta.npz already exists it is verified against a fresh deterministic
rebuild rather than trusted — an embedding written against a different grid
would silently misalign with every target.

Usage:
    uv run python plots/latent_eval/factors/build_fullday_embs.py \
        --months 2009-02 2011-04 2014-06 2020-06 2022-03
    uv run python plots/latent_eval/factors/build_fullday_embs.py \
        --months 2017-01 --families --randvit-seeds --series lejepa k2ind
    # then the analyses, restricted to the same months and series:
    uv run python plots/latent_eval/factors/decode_loadings.py \
        --months ... --series $(...) --tag _tsfmlayers
"""
import os
import argparse
import calendar
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "plots"))
sys.path.insert(0, str(REPO / "plots" / "latent_eval" / "fixed_panel"))

import factor_grid as bc  # noqa: E402
from stable_finance.dataset import MarketSchedule, timeline_bounds_est  # noqa: E402
from market_jepa.modeling import TransformerConfig, create_backbone  # noqa: E402
from market_jepa.eval.checkpoints import load_supervised, prev_month  # noqa: E402
from market_jepa.modeling.modes.pretrained_tsfm import (  # noqa: E402
    _FAMILIES, PretrainedTSFM, resolve_family)
from market_jepa.schemas import (  # noqa: E402
    LocalMachineConfig, TransformerBackboneConfig, TransformerInnerConfig,
    machine_from_env,
)

# Overridable so the sweep can run one month per SLURM job on a node
# that does not mount lab/; unset, the path is exactly what it
# always was, so local runs are unchanged. Added 2026-08-20.
CACHE = Path(os.environ.get(
    "FF_FULLDAY_CACHE",
    "lab/market-jepa-checkpoints/ff_fullday_cache"))
SEQ_LEN = 2048
# tsfm_batch_size stays 64 (chronos2's group attention is dense across the
# chunk); this is the OUTER batch of full-day views.
BS = 64


def fullday_params(date_str: str, schedule) -> tuple[int, int]:
    """Deterministic full-session view, end-aligned at the close."""
    ts_open, ts_close = timeline_bounds_est(date_str, schedule=schedule)
    n_day = ts_close - ts_open
    agg = max(1, n_day // SEQ_LEN)
    window = agg * SEQ_LEN
    return n_day - window, window


bc._day_crop_params = fullday_params


def make_random_vit(seed: int) -> torch.nn.Module:
    """The floor, built exactly as ff_random_build.py built the reported one.

    Every layer number below is only interpretable against an untrained
    encoder on the same grid — on this data the untrained floor is not zero.
    The trunk is the sweep-default ViT, which is the same floor the other
    encoder families in this cache are quoted against.
    """
    import random as _random

    torch.manual_seed(seed)
    np.random.seed(seed)
    _random.seed(seed)
    bb, inner = TransformerBackboneConfig(), TransformerInnerConfig()
    return create_backbone(
        backbone_type="transformer", n_features=9, d_embedding=bb.d_embedding,
        # MEAN, matching how every trained encoder in this cache is read
        # (panel_lib.LATENT_POOL). A floor read at a different token is not a
        # floor for these numbers. Explicit because the schema default is now
        # None -- pool is resolved per mode -- so `bb.pool` silently means
        # "cls" here rather than "whatever the models use".
        pool="mean",
        config=TransformerConfig(
            hidden_size=inner.hidden_size,
            num_hidden_layers=inner.num_hidden_layers,
            num_attention_heads=inner.num_attention_heads,
            intermediate_size=inner.intermediate_size,
            patch_size=inner.patch_size,
            layer_norm_eps=inner.layer_norm_eps,
            drop_path_rate=inner.drop_path_rate,
        ),
    )


def load_series_encoder(skey: str, ev_month: str, device):
    """One loader for every trained-encoder key in the registry.

    Thin alias: the implementation moved to panel_lib on 2026-09-04 so the
    fixed_panel scripts can share it (they resolved ``spec["project_glob"]``
    by hand, which KeyErrors on the manifest keys that are now the default
    model set). Kept as a name here because --series callers import it.
    """
    import panel_lib as eg
    return eg.load_series_encoder(skey, ev_month, device)


def _narrow(x, enc):
    """Trim the grid window to the channel count ``enc`` was built for.

    The information-token channels are trailing constants, so an info-less
    encoder reads the identical data channels by dropping the tail.
    """
    want = getattr(enc, "n_features", None)
    if want is None or x.shape[1] == want:
        return x
    if x.shape[1] < want:
        raise ValueError(
            f"grid window has {x.shape[1]} channels, encoder wants {want}")
    return x[:, :want]


def forward_plain(enc, views_f16, device, bs=512):
    outs = []
    with torch.no_grad():
        for i in range(0, len(views_f16), bs):
            x = torch.from_numpy(views_f16[i:i + bs].astype(np.float32)).to(device)
            x = _narrow(x, enc)
            lens = torch.full((x.shape[0],), SEQ_LEN, dtype=torch.long,
                              device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                emb = enc(x, lens)
            outs.append(emb.float().cpu().numpy().astype(np.float16))
    return np.concatenate(outs, axis=0)


def forward_multi(model, views_f16, device, layers):
    outs = {L: [] for L in layers}
    with torch.no_grad():
        for i in range(0, len(views_f16), BS):
            x = torch.from_numpy(views_f16[i:i + BS].astype(np.float32)).to(device)
            # The grid now carries the information-token channels; the
            # supervised-layer families here were built without them, so narrow
            # first or they see 11 columns of constants as price data.
            #
            # THIS IS A NO-OP FOR A FROZEN TSFM, which exposes no n_features --
            # they narrow themselves, by `self.channels`, inside
            # _compute_features_all. Do not "fix" that by giving PretrainedTSFM
            # an n_features: the two narrowings would both apply and the second
            # would index a nine-wide view by channel numbers meant for the
            # full grid.
            x = _narrow(x, model)
            lens = torch.full((x.shape[0],), SEQ_LEN, dtype=torch.long,
                              device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                by = model.compute_features_multi(x, lens, layers)
            for L in layers:
                outs[L].append(by[L].float().cpu().numpy().astype(np.float16))
    return {L: np.concatenate(v, axis=0) for L, v in outs.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--months", nargs="+", required=True, help="EVAL months")
    ap.add_argument("--families", nargs="*", default=sorted(_FAMILIES))
    # The supervised specialists. Depth 0 is skipped: under CLS pooling it is
    # the class token before any attention, a learned constant identical for
    # every input, so its embedding column has zero variance.
    ap.add_argument("--sup-families", nargs="*", default=[],
                    choices=["sup_return", "sup_vol", "sup_spread"])
    ap.add_argument("--series", nargs="*", default=[],
                    help="trained-encoder keys from industry_nn_sweep."
                         "MODEL_SPECS (e.g. lejepa k2ind dino ts2vec_cb028b)")
    ap.add_argument("--randvit-seeds", nargs="*", type=int,
                    default=list(range(5)),
                    help="untrained-ViT floor seeds to build alongside; the "
                         "layer numbers are not interpretable without it")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    machine = machine_from_env(LocalMachineConfig)
    schedule = MarketSchedule(machine.holiday_csv)

    for ym in args.months:
        d = CACHE / ym
        d.mkdir(parents=True, exist_ok=True)
        todo = {}
        for fam in args.families:
            # "<fam>_cmean" is the channel-averaged readout (resolve_family).
            layers = list(range(_FAMILIES[resolve_family(fam)[0]]["n_layers"] + 1))
            missing = [L for L in layers
                       if args.overwrite
                       or not (d / f"emb_tsfm_{fam}_l{L}.npz").exists()]
            if missing:
                todo[fam] = layers          # all layers ride one forward anyway
        sup_todo = {}
        for fam in args.sup_families:
            layers = list(range(1, 13))
            if any(args.overwrite or not (d / f"emb_{fam}_l{L}.npz").exists()
                   for L in layers):
                sup_todo[fam] = layers
        series_todo = [s for s in args.series
                       if args.overwrite or not (d / f"emb_{s}.npz").exists()]
        rv_todo = [sd for sd in args.randvit_seeds
                   if args.overwrite or not (d / f"emb_randvit_s{sd}.npz").exists()]
        if not todo and not rv_todo and not sup_todo and not series_todo:
            print(f"[{ym}] complete — skip", flush=True)
            continue

        t0 = time.time()
        y, m = int(ym[:4]), int(ym[5:7])
        # Grid built WITH the information-token channels so it can feed both
        # kinds of encoder: the token's channels are trailing constants, and
        # _narrow() below hands each encoder exactly the width it was built
        # for. Without this every LeJEPA pairing and every supervised head
        # dies here on
        #     mat1 and mat2 shapes cannot be multiplied (Nx9 and 11x384)
        # since those were all trained with the token and the ssl_ic models,
        # which were not, are the only reason this ever worked.
        ds = bc._make_dataset(f"{ym}-01",
                              f"{ym}-{calendar.monthrange(y, m)[1]:02d}",
                              machine, schedule, info=True)
        g = bc.build_grid(ds, np.arange(ds.num_samples), schedule,
                          sync_daily=True)
        del ds
        if g is None:
            raise RuntimeError(f"{ym}: no grid points built")
        gm_path = d / "grid_meta.npz"
        if gm_path.exists():
            gm = np.load(gm_path, allow_pickle=True)
            assert np.array_equal(g["ticker"], gm["ticker"]) and \
                np.array_equal(g["date"], gm["date"]), \
                f"{ym}: rebuilt grid does not match the stored grid_meta"
            print(f"[{ym}] grid {len(g['tod'])} pts rebuilt+verified "
                  f"({time.time() - t0:.0f}s)", flush=True)
        else:
            np.savez_compressed(gm_path, **{k: g[k] for k in bc.META_KEYS})
            print(f"[{ym}] grid {len(g['tod'])} pts built from {g['n_obs']} obs "
                  f"-> {gm_path.name} ({time.time() - t0:.0f}s)", flush=True)

        views = g["views"]
        for fam, layers in todo.items():
            t0 = time.time()
            base, channel_pool = resolve_family(fam)
            model = PretrainedTSFM(backbone=None, model=base,
                                   channels=list(range(9)),
                                   channel_pool=channel_pool)
            model = model.to(device).eval()
            try:
                X_by = forward_multi(model, views, device, layers)
            finally:
                model.to("cpu")
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            for L in layers:
                np.savez(d / f"emb_tsfm_{fam}_l{L}.npz", X_eval=X_by[L])
            print(f"    emb tsfm_{fam} layers {layers[0]}..{layers[-1]}  "
                  f"{X_by[layers[0]].shape} ({time.time() - t0:.0f}s)",
                  flush=True)
            del X_by

        for fam, layers in sup_todo.items():
            t0 = time.time()
            # The encoder for an EVAL month is the one trained the month
            # before -- the same pairing the IC sweep uses. Read at the latent
            # suite's pool, not the "last" these trained with: panel_lib is
            # imported here because `eg` is local to load_series_encoder above.
            import panel_lib as _pl
            enc = load_supervised(fam, prev_month(ym), device,
                                  pool=_pl.LATENT_POOL)
            try:
                X_by = forward_multi(enc, views, device, layers)
            finally:
                enc.to("cpu")
                del enc
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            for L in layers:
                np.savez(d / f"emb_{fam}_l{L}.npz", X_eval=X_by[L])
            print(f"    emb {fam} layers {layers[0]}..{layers[-1]}  "
                  f"{X_by[layers[0]].shape} ({time.time() - t0:.0f}s)",
                  flush=True)
            del X_by

        for skey in series_todo:
            t0 = time.time()
            enc = load_series_encoder(skey, ym, device)
            if enc is None:
                print(f"    emb {skey}: no checkpoint for {ym} — skip",
                      flush=True)
                continue
            try:
                X = forward_plain(enc, views, device)
            finally:
                enc.to("cpu")
                del enc
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            np.savez(d / f"emb_{skey}.npz", X_eval=X)
            print(f"    emb {skey:<20} {X.shape} "
                  f"({time.time() - t0:.0f}s)", flush=True)

        for sd in rv_todo:
            t0 = time.time()
            enc = make_random_vit(sd).to(device).eval()
            try:
                X = forward_plain(enc, views, device)
            finally:
                enc.to("cpu")
                del enc
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            np.savez(d / f"emb_randvit_s{sd}.npz", X_eval=X)
            print(f"    emb randvit_s{sd:<17} {X.shape} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        del g, views
    return 0


if __name__ == "__main__":
    sys.exit(main())
