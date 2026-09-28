"""Manifest for the probe-fit-size sweep: four arms, the 31-month panel.

Replicates plots/core/probe_fit_breadth.png for the models the latent
table reports, rather than for the deep-vs-broad question the original asked.
The curve per arm is the ridge probe's IC against the number of rows it was fit
on; the reference line is the supervised head on the same eval panel, which is
already in each checkpoint's xs_ic.json and costs no forward.

THE FIT POOL IS THE HEAD'S OWN TRAINING SPAN -- the six months the checkpoint
trained on, at 36 anchors/day (~358k rows a month, so ~2.1M in the pool). That
is the point of the comparison: the head saw those six months, and until the
probe is fit on the same rows their gap confounds "a head beats a probe" with
"a head saw 25x the data".

Rows are ``<ckpt_dir>\t<fit_month>\t<eval_month>``, the TSV the pythia scoring
path already consumes, SIX rows per (arm, eval month) -- one per fit month.
The embed phase needs no change for that; only the reduce differs, and
probe_fit_size.py reduce already pools every non-eval month it finds in a
checkpoint's cache dir and walks the nested-prefix ladder over it.

A (ckpt, eval month) group's six fit months MUST be reduced together, so the
submitter chunks by EVAL MONTH and never splits a group across jobs.

Run::

    uv run python scripts/eval/build_probe_breadth_manifest.py \\
        /data/lab/probe_breadth/manifest.tsv

    # the frozen TSFMs on their own (manifest_tsfm.tsv + .index.json)
    uv run python scripts/eval/build_probe_breadth_manifest.py \\
        /data/lab/probe_breadth/manifest_tsfm.tsv --tsfm
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "plots")
sys.path.insert(0, "plots/metrics")
from style import load_sweep_months  # noqa: E402
from metrics import SUP_SPAN_NO_SPAN  # noqa: E402
from stable_finance.dataset import next_month  # noqa: E402

SCRATCH = Path("/data/lab/market-jepa-checkpoints/_scratch/latent_eval")
# arm label -> (manifest file, series_key). The three supervised specialists
# are one per task, so each panel of the figure compares ITS specialist with
# the one self-supervised arm; the warp arm is probed on all three.
SUP = SCRATCH / "manifest_supspan.json"
SIX = SCRATCH / "manifest_6mo.json"
MULTI = SCRATCH / "manifest_multi.json"
# Every method the latent table reports, so one staging of a month serves all
# of them. The three supervised specialists are one per task; everything else
# is probed on all three. The random-init floor is NOT here: its checkpoints
# under /data/lab/randinit_bb carry a `sinusoidal` position embedding against
# the `rope` the current recipe trains, and a floor only floors models it
# shares an architecture_signature with -- it needs a rebuild
# (make_randinit_backbones.py --write) before it can join this panel.
ARMS = [
    ("sup_return", SUP, "sup_return_w8"),
    ("sup_vol", SUP, "sup_vol_w8"),
    ("sup_spread", SUP, "sup_spread_w8"),
    ("sup_multi", MULTI, "sup_multi_w8"),
    ("lejepa_rrc", SIX, "pair_rrc_6mo"),
    ("lejepa_warp", SIX, "pair_warp_6mo"),
    ("lejepa_noise", SIX, "pair_noise_6mo"),
    ("lejepa_k2", SIX, "pair_k2_6mo"),
    ("lejepa_k2ind", SIX, "pair_k2ind_6mo"),
    ("ssl_dino", SIX, "dino_6mo"),
    ("ssl_byol", SIX, "byol_6mo"),
    ("ssl_cpc", SIX, "cpc_6mo"),
    ("ssl_ijepa", SIX, "ijepa_6mo"),
    ("ssl_mae", SIX, "mae_6mo"),
    ("ssl_ts2vec", SIX, "ts2vec_6mo"),
    ("ssl_cost", SIX, "cost_6mo"),
    ("ssl_tfc", SIX, "tfc_6mo"),
    ("ssl_timemae", SIX, "timemae_6mo"),
]
SPAN = 6

# THE FROZEN TSFMs, on exactly the protocol above: the same eval months, the
# same six fit months per eval month, the same readout token (last), scored
# by the same embed / reduce. What is fixed per arm is the model's own
# readout: the LAST hidden state, and the nine per-channel states MEANED into
# one d_model vector (Kronos reads one multivariate bar per step, so it has no
# channel axis and its channel_pool is inert). They withhold the eleven
# information-token channels, the asymmetry decided on 2026-09-11.
#
# ONE CHECKPOINT DIR PER EVAL MONTH, though the weights never change. The
# reduce pools EVERY non-eval month in a checkpoint's cache dir, and every
# trained arm has a checkpoint per eval month, so nothing ever had to say which
# fit months belong to which eval month. One shared TSFM dir would break that
# the moment two eval months rode in one job -- a 7- or 12-month pool reduced
# as if it were the six-month one. The floor's randinit_bb/per_eval_month is
# the same answer to the same problem. Each dir holds only a config.json
# (PretrainedTSFM.save_pretrained); load_model dispatches on its "class".
TSFM_ROOT = Path("/data/lab/probe_breadth/tsfm_ckpts/per_eval_month")
TSFM_ARMS = [
    # (arm, family, series_key, PretrainedTSFM kwargs)
    ("tsfm_chronos2", "chronos2", "tsfm_chronos2",
     {"channel_pool": "mean"}),
    ("tsfm_kronos", "kronos", "tsfm_kronos", {}),
    ("tsfm_timesfm3", "timesfm3", "tsfm_timesfm3",
     {"channel_pool": "mean"}),
]


def tsfm_ckpt(arm: str, family: str, kwargs: dict, ev: str) -> Path:
    """The config-only checkpoint dir for (arm, eval month), written if absent.

    Deterministic, so re-writing is a no-op; an existing dir whose config
    disagrees with what this would write is REFUSED rather than overwritten,
    because a results row is keyed by the dir name and a silent change of
    readout under the same name is exactly what the readout stamps exist to
    prevent.
    """
    from market_jepa.modeling.modes.pretrained_tsfm import PretrainedTSFM
    m = PretrainedTSFM(backbone=None, model=family, channels=list(range(9)),
                       layer=-1, time_pool="last", **kwargs)
    d = TSFM_ROOT / f"{arm}_{ev}"
    tmp = TSFM_ROOT / f".{arm}_{ev}.new"
    m.save_pretrained(str(tmp))
    want = json.loads((tmp / "config.json").read_text())
    if d.is_dir():
        have = json.loads((d / "config.json").read_text())
        (tmp / "config.json").unlink(); tmp.rmdir()
        if have != want:
            raise SystemExit(f"{d}: config differs from the protocol's; "
                             f"remove it deliberately if that is intended")
        return d
    tmp.rename(d)
    return d


def prev_month(ym: str) -> str:
    y, m = map(int, ym.split("-"))
    return f"{y - 1:04d}-12" if m == 1 else f"{y:04d}-{m - 1:02d}"


def _fit_months(ev: str) -> list[str]:
    # The span ENDS on the month before the eval month, which is what both
    # manifests record -- checked rather than assumed, because a pool off by
    # one month is a silent leak into the eval month.
    fits, m = [], prev_month(ev)
    for _ in range(SPAN):
        fits.append(m)
        m = prev_month(m)
    return sorted(fits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?",
                    default="/data/lab/probe_breadth/manifest.tsv")
    ap.add_argument("--tsfm", action="store_true",
                    help="write the frozen-TSFM arms INSTEAD of the trained "
                         "ones, to their own manifest")
    a = ap.parse_args()
    out = Path(a.out)
    want_eval = sorted(next_month(m) for m in
                       set(load_sweep_months()) - SUP_SPAN_NO_SPAN)

    rows, index, missing = [], [], []
    if a.tsfm:
        TSFM_ROOT.mkdir(parents=True, exist_ok=True)
        for arm, fam, skey, kw in TSFM_ARMS:
            for ev in want_eval:
                d = tsfm_ckpt(arm, fam, kw, ev)
                fits = _fit_months(ev)
                rows += [f"{d}\t{f}\t{ev}" for f in fits]
                index.append({"arm": arm, "series_key": skey,
                              "eval_month": ev, "ckpt_dir": str(d),
                              "run_id": d.name, "fit_months": fits})
    arms = TSFM_ARMS if a.tsfm else ARMS
    for arm, mf, skey in ([] if a.tsfm else ARMS):
        blob = json.loads(mf.read_text())
        rows_in = blob["ckpts"] if isinstance(blob, dict) else blob
        by_ev = {r["eval_month"]: r for r in rows_in
                 if r.get("series_key") == skey}
        for ev in want_eval:
            r = by_ev.get(ev)
            if r is None:
                missing.append(f"{arm} {ev}")
                continue
            d = Path(r["ckpt_dir"])
            if not d.is_dir():
                missing.append(f"{arm} {ev}: {d} is gone")
                continue
            fits = _fit_months(ev)
            for f in fits:
                rows.append(f"{d}\t{f}\t{ev}")
            index.append({"arm": arm, "series_key": skey, "eval_month": ev,
                          "ckpt_dir": str(d), "run_id": d.name,
                          "fit_months": fits})

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n")
    # The submitter and the plot both need (run_id -> arm, eval month); the TSV
    # cannot carry it, and re-deriving it from a path is how two readers end up
    # disagreeing about which arm a cache dir belongs to.
    idx = out.with_suffix(".index.json")
    idx.write_text(json.dumps(index, indent=1))
    print(f"wrote {out}  ({len(rows)} rows)")
    print(f"wrote {idx}  ({len(index)} (arm, eval month) groups)")
    months = {r.split("\t")[1] for r in rows} | {r.split("\t")[2] for r in rows}
    print(f"  {len(want_eval)} eval months x {len(arms)} arms x {SPAN} fit "
          f"months; {len(months)} distinct months to stage")
    for arm, *_ in arms:
        n = sum(1 for e in index if e["arm"] == arm)
        print(f"  {arm:12s} {n}/{len(want_eval)} eval months")
    for m in missing:
        print(f"  MISSING {m}")


if __name__ == "__main__":
    main()
