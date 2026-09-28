"""Submit probe-breadth evals for exactly the (arm, eval month) groups that
have no result yet.

WHY NOT JUST RE-RUN THE WAVE. A full panel is 31 eval months x 19 arms and
several GPU-days; almost all of it is already on disk. The unit of missing
work is the GROUP -- one (checkpoint, eval month) with its six fit months --
and a group is missing when NO result json anywhere carries that
checkpoint's run_id for that eval month.

KEYING ON RUN ID IS WHAT MAKES A RERUN ARM WORK. When an arm is retrained the
new wave has NEW run ids, so every one of its groups reads as missing and gets
re-evaluated, while the superseded wave's results sit in the results dir doing
nothing (probe_fit_table resolves run_id -> series through the manifest index,
and a run id no longer in the index simply stops resolving). Nothing has to be
deleted for the table to move to the new weights.

    uv run python scripts/eval/submit_missing_breadth.py            # dry run
    uv run python scripts/eval/submit_missing_breadth.py --submit
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/data/lab/probe_breadth")
RESULTS = ROOT / "results"
MONTH = re.compile(r"(\d{4}-\d{2})(?=\.json$)")


def scored_pairs(alpha: float = 10.0) -> set[tuple[str, str]]:
    """{(run_id, eval_month)} that already have a reduce result."""
    out = set()
    for f in sorted(RESULTS.glob("*.json")):
        m = MONTH.search(f.name)
        if not m:
            continue
        try:
            rows = json.loads(f.read_text())
        except json.JSONDecodeError:            # still being written
            continue
        for r in rows:
            if abs(float(r.get("alpha", -1)) - alpha) > 1e-9:
                continue
            out.add((r["ckpt"], m.group(1)))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=str(ROOT / "manifest_all.tsv"))
    p.add_argument("--tag", default="pbgap")
    p.add_argument("--batch", type=int, default=6,
                   help="eval months submitted per stagger step. bll01's sshd "
                        "runs MaxStartups 10:30:100 and each job fans out one "
                        "rsync per staged month, so a big wave starting at "
                        "once loses months to dropped connections -- that is "
                        "what killed pbfl2-part02/03.")
    p.add_argument("--stagger", type=int, default=7,
                   help="minutes between batches, via sbatch --begin")
    p.add_argument("--partition", default="standard_hopper")
    p.add_argument("--exclude", default="pgpu012,pgpu015")
    p.add_argument("--submit", action="store_true")
    a = p.parse_args()

    mf = Path(a.manifest)
    idx = json.loads(mf.with_suffix(".index.json").read_text())
    done = scored_pairs()

    todo = [e for e in idx if (e["run_id"], e["eval_month"]) not in done]
    if not todo:
        print("nothing missing: every group in the manifest has a result")
        return 0

    by_arm = defaultdict(int)
    for e in todo:
        by_arm[e["series_key"]] += 1
    print(f"{len(todo)} group(s) with no result, over "
          f"{len({e['eval_month'] for e in todo})} eval month(s):")
    for k, n in sorted(by_arm.items(), key=lambda kv: -kv[1]):
        print(f"   {k:18s} {n:>3d}")

    # The submitter chunks by eval month and never splits a group, so the
    # sub-manifest is every row of the TSV whose (ckpt, eval) is in todo.
    want = {(e["ckpt_dir"], e["eval_month"]) for e in todo}
    rows = [ln for ln in mf.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
            and (ln.split("\t")[0], ln.split("\t")[2]) in want]
    months = sorted({ln.split("\t")[2] for ln in rows})
    print(f"\n{len(rows)} manifest row(s), {len(months)} eval month(s)")
    bad = defaultdict(int)
    for ln in rows:
        bad[(ln.split("\t")[0], ln.split("\t")[2])] += 1
    partial = {k: v for k, v in bad.items() if v != 6}
    if partial:
        print(f"REFUSING: {len(partial)} group(s) do not have 6 fit rows")
        for (c, e), v in list(partial.items())[:5]:
            print(f"   {Path(c).name} {e}: {v}")
        return 1

    # ONE RUNNER CALL PER EVAL MONTH, tagged with that month.
    #
    # The runner already chunks by eval month and never splits a group, so a
    # multi-month call produced jobs named <tag>-partNN -- and NN is a
    # position inside that call's sorted month list, which nothing downstream
    # records. `squeue` then shows pbgap03-part02 with no way to tell which
    # month is stuck without reconstructing the chunker. Calling it per month
    # makes the job name carry the month, which is the only identifier that
    # matters when one of thirty-one is slow.
    #
    # The stagger is unchanged: months are still released in groups of
    # --batch, --stagger minutes apart, via sbatch --begin.
    print(f"{len(months)} month job(s), released {a.batch} at a time, "
          f"{a.stagger} min apart\n")
    tmp = Path("/tmp") / f"{a.tag}-batches"
    tmp.mkdir(parents=True, exist_ok=True)
    runner = "./scripts/pythia/specific/run_probe_breadth.sh"
    import os
    for i, mo in enumerate(months):
        sub = tmp / f"{a.tag}-{mo}.tsv"
        sub.write_text("\n".join(ln for ln in rows
                                 if ln.split("\t")[2] == mo) + "\n")
        wave = i // a.batch
        begin = f"now+{wave * a.stagger}minutes" if wave else "now"
        env = {"PARTITION": a.partition,
               "SBATCH_EXTRA": f"--exclude={a.exclude} --begin={begin}"}
        cmd = [runner, "--manifest", str(sub), "--tag", f"{a.tag}-{mo}"]
        ngrp = sum(1 for ln in rows if ln.split("\t")[2] == mo) // 6
        print(f"  {mo}  {ngrp:>2d} group(s)  wave {wave}  begin {begin}")
        if a.submit:
            r = subprocess.run(cmd, env={**os.environ, **env},
                               capture_output=True, text=True)
            tail = [l for l in r.stdout.splitlines() if "job" in l]
            print("      " + ("\n      ".join(tail[-2:]) if tail
                              else r.stderr.strip()[-200:]))
            if r.returncode != 0:
                print(f"      SUBMIT FAILED rc={r.returncode}")
                return 1
    if not a.submit:
        print("\nDRY RUN -- pass --submit to send them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
