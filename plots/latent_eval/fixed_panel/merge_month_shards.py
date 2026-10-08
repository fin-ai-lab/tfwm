"""Pool the per-month fixed-panel shards into one panel json.

run_eval.sh is month-major and runs JOBS months at once, so each month writes
its OWN ``fixed_panel_P{P}S{S}{tag}_m{YYYY-MM}.json`` rather than all of them
read-modify-writing one file. This pools those shards into
``fixed_panel_P{P}S{S}{tag}.json``, which is what every reader expects.

THE POOLING IS NOT AN AVERAGE OF SUMMARIES. ``t`` is (rate - chance) over
MONTH-MEANS and ``rank_t`` is (pctile - 0.5) over them, so both are recomputed
from the union of the per-month series; averaging the shards' own ``t`` values
would be wrong and would not even be close (each shard's is NaN anyway -- one
month has no spread to test). The arithmetic is ``fixed_panel_metrics._repool``,
the same function the in-file merge uses, so there is one implementation and
one place for it to be wrong.

A model missing from some months is pooled over the months it has, so one
family failing in one month degrades that cell rather than the file.

Run::

    uv run python plots/latent_eval/fixed_panel/merge_month_shards.py --tag _meanpool
    uv run python .../merge_month_shards.py --tag _meanpool --keep-shards
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fixed_panel_metrics import _repool  # noqa: E402

METRICS = ("metric1", "metric2", "metric3", "metric4")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="", help="the run's TAG, e.g. _meanpool")
    p.add_argument("--p", type=int, default=3)
    p.add_argument("--s", type=int, default=2)
    p.add_argument("--dir", type=Path, default=HERE,
                   help="where the shards live (default: beside this script)")
    p.add_argument("--keep-shards", action="store_true",
                   help="leave the per-month files in place (default: delete "
                        "once the pooled file is written)")
    args = p.parse_args()

    d = args.dir
    stem = f"fixed_panel_P{args.p}S{args.s}{args.tag}"
    pat = re.compile(rf"^{re.escape(stem)}_m(\d{{4}}-\d{{2}})\.json$")
    shards = {}
    for f in sorted(d.glob(f"{stem}_m*.json")):
        m = pat.match(f.name)
        if m:
            shards[m.group(1)] = json.loads(f.read_text())
    if not shards:
        raise SystemExit(f"no shards matching {stem}_m<YYYY-MM>.json in {d}")

    months = sorted(shards)
    print(f"pooling {len(months)} months: {months[0]}..{months[-1]}")

    models: dict = {}
    for ym in months:
        for key, ent in shards[ym]["models"].items():
            for metric in METRICS:
                cell = ent.get(metric)
                if not cell or "month_rates" not in cell:
                    continue
                prior = models.get(key, {}).get(metric)
                # EVERY cell goes through _repool, the first month included:
                # copying the first shard through would keep a NaN month that
                # _repool exists to drop, and the arm's n_months would count a
                # month it never had a checkpoint for.
                models.setdefault(key, {})[metric] = _repool(prior or {}, cell)

    # Every shard shares the panel geometry; take it from the first and check
    # the rest agree rather than silently pooling two different panels.
    first = shards[months[0]]
    for ym in months[1:]:
        for f in ("P", "S", "n_panels"):
            if shards[ym][f] != first[f]:
                raise SystemExit(
                    f"{ym} has {f}={shards[ym][f]}, {months[0]} has "
                    f"{first[f]} -- these are different panels, not shards")

    out = d / f"{stem}.json"
    out.write_text(json.dumps({
        "P": first["P"], "S": first["S"], "n_panels": first["n_panels"],
        "metric_names": first["metric_names"], "months": months,
        "models": models,
    }, indent=1))

    n_full = sum(1 for e in models.values()
                 for c in e.values() if c.get("n_months") == len(months))
    n_cells = sum(len(e) for e in models.values())
    print(f"wrote {out}  ({len(models)} models, {n_full}/{n_cells} cells on "
          f"all {len(months)} months)")
    for key, ent in sorted(models.items()):
        short = {c.get("n_months", 1) for c in ent.values()}
        if short and max(short) < len(months):
            print(f"  {key}: only {sorted(short)} month(s)")

    if not args.keep_shards:
        for f in d.glob(f"{stem}_m*.json"):
            if pat.match(f.name):
                f.unlink()
        print(f"removed {len(shards)} shard files")


if __name__ == "__main__":
    main()
