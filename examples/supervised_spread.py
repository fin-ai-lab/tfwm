"""Train the supervised spread-change model and check it against the released one.

    uv run examples/supervised_spread.py            # full recipe, one GPU
    uv run examples/supervised_spread.py --smoke    # pipeline check, minutes

Trains the spread-change specialist from the day store on the six months
before --eval-month (12 passes, the reported recipe), then scores both this
run's head and the released `tfwm-supervised-spread` head for that month with
the paper's scorer, and passes if they agree within --tolerance.

Spread change is the parity check because it has the least run-to-run noise:
across 10 seeds x 9 months (plots/variance_decomp, the 6-month recipe) the
seed standard deviation of its head IC is 0.0026, about 1% of its ~0.2 level,
so a 5% band is roughly four standard deviations wide.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _replicate as R  # noqa: E402

SLUG = "supervised-spread"
TASK = "spread_change_900"
# plots/metrics/probe_head_ic.json, series sup_spread_w8.
PAPER = {"2020-01": {TASK: 0.2038}}


def main():
    a = R.parse_args(__doc__)
    m, work = R.Month(a.eval_month), Path(a.work)
    R.check_login()
    data = work / "market1t"
    mosaic = R.download(data, R.DENSE, m.all)
    daystore = R.decompress_daystore(R.download(data, R.DAYSTORE, m.all), m.all)
    targets = R.build_targets(R.download(data, R.SPARSE, m.all),
                              work / "xs_anchor_stats_fwdvwap60", m.all)
    rel = R.released(work, SLUG, m.eval)

    r_ic = R.score_head(rel, m, mosaic, targets, TASK)
    if a.skip_train:
        print(f"released {TASK}: {r_ic[0]:.4f} (se {r_ic[1]:.4f}), paper "
              f"{PAPER.get(m.eval, {}).get(TASK, float('nan')):.4f}")
        return
    ours = R.train(work, SLUG, [
        "mode=supervised", f"mode.task={TASK}", "dataset.backend=days",
        f"machine.mosaic_dir={mosaic}", f"machine.daystore_dir={daystore}",
        f"dataset.xs_anchor_stats_dir={targets}", *m.overrides()], a.smoke)
    diff = R.config_diff(ours / "train_meta.json", rel / "train_meta.json")
    print("\nconfig vs released:", *(diff or ["identical"]), sep="\n  ")
    o_ic = R.score_head(ours, m, mosaic, targets, TASK)

    ok = R.compare(f"Supervised spread change, eval {m.eval}", {TASK: o_ic},
                   {TASK: r_ic}, [TASK], PAPER.get(m.eval, {}), a.tolerance,
                   work / f"{SLUG}_{m.eval}.json")
    sys.exit(0 if ok or a.smoke else 1)


if __name__ == "__main__":
    main()
