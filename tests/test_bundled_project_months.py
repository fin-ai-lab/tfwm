"""A bundled project spans several months, and each run must name its own.

iter_ic_runs took the training month from the PROJECT NAME suffix. That holds
for one-job-per-month sweeps (run_all_months_sweep.sh writes
"...-2008-02-01-2008-02-29") and fails silently for anything submitted with
--months-per-job N > 1: the 203-month multihead campaign lands in projects
ending "-2008-01-2008-04", which _MONTH_SUFFIX_RE does not match at all, so
every one of its 51 projects was skipped and the series read as ZERO RUNS.

Zero is the dangerous answer here. It is indistinguishable from "the sweep has
not run yet", which is exactly how the previous generation of this figure came
back empty.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plots.style import iter_ic_runs  # noqa: E402


def _write(root: Path, project: str, run_id: str, run_name: str):
    d = root / project / run_id
    d.mkdir(parents=True)
    (d / "train_meta.json").write_text(json.dumps({
        "run_name": run_name,
        "xs_anchor_stats": "xs_anchor_stats_fwdvwap60",
    }))
    (d / "xs_ic.json").write_text(json.dumps({"xs_ic/return_900": 0.01}))


def test_a_bundled_project_yields_one_run_per_month(tmp_path):
    """Four months in one project, each run carrying its own."""
    proj = "supervised-full-month-multihead-3e5087-2008-01-2008-04"
    for i, ym in enumerate(("2008-01", "2008-02", "2008-03", "2008-04")):
        _write(tmp_path, proj, f"run{i}", f"{ym}_multi_pairwise_s42")
    runs = iter_ic_runs("supervised-full-month-multihead-*", tmp_path,
                        xs_stats="xs_anchor_stats_fwdvwap60")
    assert len(runs) == 4, f"bundled project yielded {len(runs)} runs"
    assert {r.train_month for r in runs} == {
        "2008-01", "2008-02", "2008-03", "2008-04"}


def test_a_bundled_project_respects_the_month_filter(tmp_path):
    """The filter must apply per RUN, since the project names no single month."""
    proj = "supervised-full-month-multihead-3e5087-2008-01-2008-04"
    for i, ym in enumerate(("2008-01", "2008-02", "2008-03", "2008-04")):
        _write(tmp_path, proj, f"run{i}", f"{ym}_multi_pairwise_s42")
    runs = iter_ic_runs("supervised-full-month-multihead-*", tmp_path,
                        xs_stats="xs_anchor_stats_fwdvwap60",
                        train_months={"2008-02", "2008-04"})
    assert {r.train_month for r in runs} == {"2008-02", "2008-04"}


def test_a_single_month_project_still_uses_its_suffix(tmp_path):
    """Unchanged for every one-job-per-month sweep in the tree."""
    proj = "supervised-full-month-return-3e5087-2008-02-01-2008-02-29"
    _write(tmp_path, proj, "r0", "2008-02_pairwise_blr1e-5")
    runs = iter_ic_runs("supervised-full-month-return-*", tmp_path,
                        xs_stats="xs_anchor_stats_fwdvwap60")
    assert len(runs) == 1 and runs[0].train_month == "2008-02"


def test_a_single_month_project_filter_is_unchanged(tmp_path):
    proj = "supervised-full-month-return-3e5087-2008-02-01-2008-02-29"
    _write(tmp_path, proj, "r0", "2008-02_pairwise_blr1e-5")
    assert iter_ic_runs("supervised-full-month-return-*", tmp_path,
                        xs_stats="xs_anchor_stats_fwdvwap60",
                        train_months={"2009-01"}) == []


def test_a_run_naming_no_month_in_a_bundle_is_dropped(tmp_path):
    """Better to drop one run than to label it with a sibling's month."""
    proj = "someseries-3e5087-2008-01-2008-04"
    _write(tmp_path, proj, "r0", "no_month_here")
    assert iter_ic_runs("someseries-*", tmp_path,
                        xs_stats="xs_anchor_stats_fwdvwap60") == []
