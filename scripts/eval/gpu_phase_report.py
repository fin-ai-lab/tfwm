"""Where did a SLURM allocation's wall-clock and its GPU actually go?

A job holds its GPU from allocation to exit, and on a scoring sweep most of
what happens in between is not a forward: syncing a venv, rsyncing mosaic
months, rsyncing checkpoints, reducing on CPU. All of that is GPU time bought
and left idle, and it does not show up in "the embed took N minutes".

Joins the two files slurm_probe_breadth.sh writes -- ``phases.tsv``
(unix_ts, clock, phase) and ``gpu.csv`` (nvidia-smi --query-gpu -l) -- into one
row per phase: wall-clock, share of the allocation, and the GPU utilization
observed DURING it. The last column is the one to act on: a phase with a long
wall and a low mean is what to attack next, and a phase that is already at 90%+
is done being optimized.

Run on the node at the end of a job, or locally over a pushed pair::

    uv run python scripts/eval/gpu_phase_report.py --phases phases.tsv --gpu gpu.csv
    uv run python scripts/eval/gpu_phase_report.py --dir /data/lab/probe_breadth/results/pb3-part00
"""
from __future__ import annotations

import argparse
import statistics
from datetime import datetime
from pathlib import Path


def _parse_gpu(path: Path) -> list[tuple[float, float, float]]:
    """(unix_ts, gpu_util_pct, mem_MiB) per sample.

    nvidia-smi stamps a local-time string, not an epoch, so it is parsed back
    against the same clock the phase file used. A row that does not parse is
    dropped rather than guessed at -- the sampler is killed mid-write at exit,
    so the last line is routinely a fragment.
    """
    out = []
    for ln in path.read_text().splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 4:
            continue
        try:
            ts = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            out.append((ts, float(parts[1].rstrip(" %")),
                        float(parts[3].rstrip(" MiB"))))
        except (ValueError, IndexError):
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, help="directory holding both files")
    ap.add_argument("--phases", type=Path)
    ap.add_argument("--gpu", type=Path)
    a = ap.parse_args()
    ph_f = a.phases or (a.dir / "phases.tsv")
    gpu_f = a.gpu or (a.dir / "gpu.csv")
    if not ph_f.is_file():
        raise SystemExit(f"no phase file: {ph_f}")

    phases = []
    for ln in ph_f.read_text().splitlines():
        if not ln.strip():
            continue
        ts, clock, name = ln.split("\t")
        phases.append((float(ts), clock, name))
    if not phases:
        raise SystemExit(f"{ph_f} is empty")
    samples = _parse_gpu(gpu_f) if gpu_f.is_file() else []

    end = max(phases[-1][0], samples[-1][0] if samples else phases[-1][0])
    total = end - phases[0][0]
    print(f"allocation {total / 60:.1f} min, {len(samples)} GPU samples")
    print(f"{'phase':<16}{'start':>9}{'min':>8}{'% wall':>8}"
          f"{'GPU mean':>10}{'GPU p90':>9}{'idle%':>7}")
    busy_s = 0.0
    for i, (ts, clock, name) in enumerate(phases):
        stop = phases[i + 1][0] if i + 1 < len(phases) else end
        dur = stop - ts
        us = [u for t, u, _ in samples if ts <= t < stop]
        if us:
            mean, p90 = statistics.mean(us), sorted(us)[int(0.9 * (len(us) - 1))]
            idle = 100.0 * sum(1 for u in us if u < 5) / len(us)
            cells = f"{mean:>9.1f}%{p90:>8.0f}%{idle:>6.0f}%"
            busy_s += dur * mean / 100.0
        else:
            cells = f"{'--':>10}{'--':>9}{'--':>7}"
        print(f"{name:<16}{clock:>9}{dur / 60:>8.1f}{100 * dur / total:>7.0f}%"
              + cells)
    if samples:
        # THE NUMBER THAT MATTERS FOR THE NEXT WAVE: GPU-seconds actually used
        # over GPU-seconds held. A job at 30% is one where two thirds of what
        # the queue charged for was staging and reducing.
        print(f"\nGPU-seconds used / held: {busy_s / total:.0%}  "
              f"({busy_s / 60:.1f} of {total / 60:.1f} min)")


if __name__ == "__main__":
    main()
