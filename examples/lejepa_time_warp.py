"""Train LeJEPA with the time-warp pairing and check it against the released encoder.

    uv run examples/lejepa_time_warp.py            # full recipe, one GPU
    uv run examples/lejepa_time_warp.py --smoke    # pipeline check, minutes

Trains on the six months before --eval-month (12 passes, the reported recipe),
then scores both this run and the released `tfwm-lejepa-time-warp` checkpoint
for that month with the paper's forecasting probe, and passes if they agree
within --tolerance on volatility change and spread change.

Return is reported but not gated. Its IC is ~0.01, so a few percent of it is
far inside the run-to-run spread, and a 5% check on it would fail at random.

VOLATILITY CHANGE IS THE GATE THAT DISCRIMINATES. The probe reads spread change
well off almost any encoder of this architecture: a 50-step --smoke model
already matches the released encoder on it (0.121 vs 0.118 for 2020-01) while
missing volatility change by 13%. So a spread pass alone says little; a
volatility pass says the training worked.

Measured 2026-09-28 on a clean H100 from the public repo, 2020-01:
volatility change -0.9%, spread change +3.5% against the released encoder.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _replicate as R  # noqa: E402

SLUG = "lejepa-time-warp"
# plots/core/probe_breadth_snapshot.json.gz, series pair_warp_6mo, largest n.
PAPER = {"2020-01": {"return_900": 0.0133, "volatility_change_900": 0.0694,
                     "spread_change_900": 0.1205}}


def main():
    a = R.parse_args(__doc__)
    m, work = R.Month(a.eval_month), Path(a.work)
    R.check_login()
    data = work / "market1t"
    mosaic = R.download(data, R.DENSE, m.all)
    targets = R.build_targets(R.download(data, R.SPARSE, m.all),
                              work / "xs_anchor_stats_fwdvwap60", m.all)
    rel = R.released(work, SLUG, m.eval)

    ckpts = {"released": rel}
    if not a.skip_train:
        ours = R.train(work, SLUG, [
            "mode=lejepa", "mode.lamb=0.001", "dataset.augmentations.0.name=time_warp",
            f"machine.mosaic_dir={mosaic}", f"dataset.xs_anchor_stats_dir={targets}",
            *m.overrides()], a.smoke)
        diff = R.config_diff(ours / "train_meta.json", rel / "train_meta.json")
        print("\nconfig vs released:", *(diff or ["identical"]), sep="\n  ")
        ckpts["ours"] = ours

    s = R.score_probe(work, ckpts, m, mosaic, targets, a.workers, a.smoke)
    if a.skip_train:
        for t, (ic, se) in s["released"].items():
            print(f"released {t}: {ic:.4f} (se {se:.4f}), paper "
                  f"{PAPER.get(m.eval, {}).get(t, float('nan')):.4f}")
        return
    ok = R.compare(f"LeJEPA time warp, eval {m.eval}", s["ours"], s["released"],
                   ["volatility_change_900", "spread_change_900"],
                   PAPER.get(m.eval, {}), a.tolerance, work / f"{SLUG}_{m.eval}.json")
    sys.exit(0 if ok or a.smoke else 1)


if __name__ == "__main__":
    main()
