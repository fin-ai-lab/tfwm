"""Draw the paper's shared random-init floor encoders and save them as checkpoints.

THE FLOOR IS A CHECKPOINT, NOT A SEED. Every delta-IC number in the paper is
quoted against the IC an UNTRAINED encoder of the same architecture reaches
with the same probe on the same panel. Consumers load these dirs rather than
re-drawing from a seed, so that one figure's subtrahend IS another's:

    plots/forward_eval_v2/calculate_random_init_baselines.py   RANDINIT_ROOT
    plots/latent_eval/fixed_panel/panel_lib.py                 (read at MEAN)
    scripts/pythia/specific/run_score_ckpts.sh --tag randinit-fwd3
        -> /data/lab/score_results/, which plots/metrics reads via
           style.load_randinit_ic

ARCHITECTURE IS COPIED FROM A REAL TRAINED CHECKPOINT, not written out here.
A floor only floors models it shares an ``architecture_signature`` with, and
that signature moved on 2026-09-08 when the supervised default readout became
pool=last + pos_embed=rope:

    transformer-384-cls -9-384-12-6-1536-8-learned-own   <- the archived floor
    transformer-384-last-9-384-12-6-1536-8-rope   -own   <- what we train now

The old encoders survive at /data/lab/models-archive/randinit_bb and are NOT
comparable to anything current. Templating off a live checkpoint is what stops
this from happening again silently: if the recipe moves, rerunning this picks
the move up, and --check fails loudly when the drawn signature stops matching
the reference checkpoint.

ONE SET OF WEIGHTS SERVES BOTH POOLS. Pooling is applied after the last block,
so no weight depends on it (see eval/checkpoints.load_backbone). train_meta
records pool=last, the predictive convention; the latent suite passes
pool="mean" at load time and gets the same encoder read at a different token.
Writing two weight sets would create two floors that could drift apart.

Usage:
    uv run scripts/eval/make_randinit_backbones.py --check
    uv run scripts/eval/make_randinit_backbones.py --write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for xs_ic_eval

from market_jepa.backbone_config import backbone_block  # noqa: E402
from market_jepa.eval.checkpoints import (  # noqa: E402
    architecture_signature, build_untrained_encoder, load_backbone,
)
# THE PANEL DECIDES THE INPUT WIDTH, so the floor has to ask the same accessor
# the scorer asks. a22cd03 threaded this through xs_ic_eval's floor loop and
# did NOT reach here -- the other producer of the same floor -- so --write kept
# drawing a 9-channel encoder for a panel that carries 9 + 11.
from xs_ic_eval import _info_channel_width, panel_kwargs_for  # noqa: E402

OUT_ROOT = Path("/data/lab/randinit_bb")
SEEDS = (42, 43, 44)
# Any month's specialist: they all share the reported encoder. Read for its
# backbone block only, so WHICH month -- and which task -- does not matter.
#
# RESOLVED, NOT PINNED. This was a single hardcoded project directory and it
# died the moment that generation was deleted, which is exactly the failure a
# pin invites: the script does not care which checkpoint it reads, only that
# the backbone block is the current one, so naming one specific run bought
# nothing and cost the whole script when the name went away.
CKPT_ROOT = Path("/data/lab/market-jepa-checkpoints")
# ANY supervised project, most specific first. The script needs one thing --
# a train_meta.json whose backbone block is the CURRENT recipe -- and every
# supervised run has one. Naming only the full-month specialists made this
# script depend on that particular campaign existing on disk, which is how it
# broke the moment those checkpoints were deleted.
REFERENCE_GLOBS = (
    "supervised-full-month-*",
    "supervised-*",
)


def reference_backbone_cfg() -> tuple[dict, Path, str]:
    """The reference block, the file it came from, and that run's signature.

    READ VIA backbone_block, NOT cfg["backbone"]. The top-level block on a
    supervised run is not absent -- it carries pool=None / pos_embed=None,
    which the backbone resolves to cls/learned. Templating the floor off it
    produced a cls/learned encoder whose architecture_signature matches NO
    real checkpoint, so every floor built here was the wrong architecture for
    everything it was meant to floor. That is the precise bug this script
    exists to repair, and it had it too.
    """
    metas: list[Path] = []
    for pat in REFERENCE_GLOBS:
        for proj in sorted(CKPT_ROOT.glob(pat)):
            metas.extend(sorted(proj.glob("*/train_meta.json")))
        if metas:
            break
    if not metas:
        raise SystemExit(
            "no supervised specialist checkpoint to template the architecture "
            f"off: none of {REFERENCE_GLOBS} under {CKPT_ROOT} holds a "
            "train_meta.json. Run a specialist first."
        )
    ref_cfg = json.loads(metas[0].read_text())["config"]
    bb = backbone_block(ref_cfg) or {}
    if not bb:
        raise SystemExit(f"{metas[0]} declares no backbone in mode or top level")
    return dict(bb), metas[0], architecture_signature(ref_cfg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--write", action="store_true", help="write the checkpoints")
    ap.add_argument("--check", action="store_true",
                    help="report what WOULD be written and verify the signature")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args()
    if not (args.write or args.check):
        ap.error("pass --check or --write")

    bb, ref_path, ref_sig = reference_backbone_cfg()
    # THE DATASET BLOCK TRAVELS WITH THE FLOOR, and it has to, twice over. The
    # info flags live there, so without it (a) panel_kwargs_for reports 0 info
    # channels and the encoder is drawn 9 wide against models that are 9 + 11,
    # and (b) this cfg is what gets WRITTEN to the floor's train_meta.json, so
    # the scorer would later build the floor a 9-channel panel while every
    # model it floors is scored on 20. architecture_signature keys on the
    # backbone, so carrying it does not move the signature -- asserted below.
    ref_cfg = json.loads(ref_path.read_text())["config"]
    cfg = {"backbone": bb, "mode": {}, "n_features": 9}
    if ref_cfg.get("dataset"):
        cfg["dataset"] = ref_cfg["dataset"]
    sig = architecture_signature(cfg)
    # THE FLOOR MUST SIGN AS THE CHECKPOINTS IT FLOORS. Comparing the floor's
    # signature to its own config is vacuous -- it is the same dict. The
    # comparison that means something is against the REFERENCE RUN.
    if sig != ref_sig:
        raise SystemExit(
            f"floor signature {sig} != reference run {ref_sig}\n"
            f"({ref_path}) -- the floor would be matched to nothing."
        )
    print(f"reference : {ref_path.parent.parent.name}/{ref_path.parent.name}")
    print(f"  pool={bb.get('pool')}  pos_embed={(bb.get('config') or {}).get('pos_embed')}")
    print(f"  signature: {sig}")

    out_root = Path(args.out_root)
    dev = torch.device("cpu")
    # 9 real + 8 norm-stat + 3 window = 20 under the default recipe. Derived
    # from the panel flags, never a constant.
    n_info = _info_channel_width(panel_kwargs_for(cfg))
    print(f"  info channels: {n_info} (input width {9 + n_info})")
    for seed in SEEDS:
        model = build_untrained_encoder(cfg, dev, seed=seed,
                                        n_info_channels=n_info)
        backbone = model.backbone
        n_par = sum(p.numel() for p in backbone.parameters())
        d = out_root / f"randinit_s{seed}"
        print(f"  seed {seed}: {n_par:,} params -> {d}")
        if not args.write:
            continue
        d.mkdir(parents=True, exist_ok=True)
        torch.save(backbone.state_dict(), d / "backbone.pt")
        (d / "train_meta.json").write_text(json.dumps({
            "config": cfg, "run_name": f"randinit_s{seed}",
        }))

    if args.write:
        # THE SIGNATURE CANNOT SEE THE INPUT WIDTH, so checking it is not
        # enough -- that is exactly how a 9-channel floor passed this block
        # while the models it floors carry 9 + 11. The load-bearing check is
        # whether a REAL trained state dict loads into the floor with the same
        # keys and shapes, which is how a22cd03 verified the other producer.
        ref_sd = torch.load(ref_path.parent / "backbone.pt",
                            map_location="cpu", weights_only=True)
        ref_sd = ref_sd.get("model", ref_sd) if isinstance(ref_sd, dict) else ref_sd
        print("\nround-trip check:")
        for seed in SEEDS:
            d = out_root / f"randinit_s{seed}"
            got = architecture_signature(
                json.loads((d / "train_meta.json").read_text())["config"])
            b_last = load_backbone(d, d, "")
            b_mean = load_backbone(d, d, "", pool="mean")
            floor_sd = b_last.state_dict()
            missing = sorted(set(ref_sd) - set(floor_sd))
            extra = sorted(set(floor_sd) - set(ref_sd))
            shape_diff = sorted(
                k for k in set(ref_sd) & set(floor_sd)
                if tuple(ref_sd[k].shape) != tuple(floor_sd[k].shape))
            loads = not (missing or extra or shape_diff)
            ok = (got == sig == ref_sig and loads
                  and b_last.pool == bb.get("pool") and b_mean.pool == "mean")
            print(f"  seed {seed}: sig{'==' if got == sig else '!='}reference  "
                  f"pool(default)={b_last.pool}  pool(override)={b_mean.pool}  "
                  f"state_dict{'==' if loads else '!='}reference  "
                  f"{'OK' if ok else 'MISMATCH'}")
            if not loads:
                for label, ks in (("only in reference", missing),
                                  ("only in floor", extra)):
                    if ks:
                        print(f"      {label}: {ks[:6]}"
                              f"{' ...' if len(ks) > 6 else ''}")
                for k in shape_diff[:6]:
                    print(f"      shape {k}: reference "
                          f"{tuple(ref_sd[k].shape)} vs floor "
                          f"{tuple(floor_sd[k].shape)}")
            if not ok:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
