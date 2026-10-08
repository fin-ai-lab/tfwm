"""stage_data.sh:reclaim_scratch_if_low -- frees a node's scratch only when it
is short, and never a month a running job of ours on the node touched.

The routine is bash; the test drives it in a shell where ``squeue`` and ``df``
are shell functions, so the age rule is exercised against a fabricated node.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "scripts/generic/lib/stage_data.sh"


def _tree(tmp: Path):
    d = tmp / "data"
    for kind, months in (("1Hz_daystore", ("2008-02", "2015-06")),
                         ("1Hz_mosaic_mnth", ("2008-07", "2020-11")),
                         ("panel_cache/keyA", ("2008-07", "2020-11")),
                         ("panel_cache/keyB", ("2008-07",))):
        for ym in months:
            y, m = ym.split("-")
            sub = d / kind / (ym if "panel" in kind else f"{y}/{m}")
            sub.mkdir(parents=True)
            (sub / "index.json").write_text("{}")
    mk = d / ".markers"
    mk.mkdir()
    old = time.time() - 7200          # touched two hours ago: before the job
    for name in ("daystore-2015-06", "mosaic-2020-11", "panel-2020-11"):
        f = mk / f"{name}.v7.done"
        f.write_text("")
        os.utime(f, (old, old))
    for name in ("daystore-2008-02", "mosaic-2008-07", "panel-2008-07"):
        (mk / f"{name}.v7.done").write_text("")      # touched now: live
    # An unmarked partial copy from a crashed job, old.
    part = d / "1Hz_daystore/2011/12"
    part.mkdir(parents=True)
    os.utime(part, (old, old))
    (tmp / "market-jepa-venv-111").mkdir()
    (tmp / "market-jepa-venv-222").mkdir()
    return d


def _run(tmp: Path, data: Path, free_gb: int, min_free: int) -> subprocess.CompletedProcess:
    start = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 1800))
    script = f"""
    squeue() {{ case " $* " in *" -t R "*) echo "{start}" ;; *) echo 111 ;; esac; }}
    df() {{ printf 'Avail\\n{free_gb}G\\n'; }}
    export -f squeue df
    DATA_DIR="{data}"; LOCK_DIR="{data}/.locks"; MARKER_DIR="{data}/.markers"
    SCRATCH_BASE="{tmp}"; REPO_DIR="{ROOT}"
    DATA_HOST=x; DATA_HOST_DATA=x; DATA_HOST_METADATA=x
    source "{LIB}"
    SCRATCH_MIN_FREE_GB={min_free} reclaim_scratch_if_low
    """
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_nothing_is_touched_while_there_is_room(tmp_path):
    d = _tree(tmp_path)
    r = _run(tmp_path, d, free_gb=900, min_free=400)
    assert r.returncode == 0, r.stderr
    assert "nothing reclaimed" in r.stdout
    assert (d / "1Hz_daystore/2015/06").exists()
    assert (tmp_path / "market-jepa-venv-222").exists()


def test_only_months_older_than_the_oldest_running_job_go(tmp_path):
    d = _tree(tmp_path)
    r = _run(tmp_path, d, free_gb=100, min_free=400)
    assert r.returncode == 0, r.stderr
    # stale: untouched since before the running job started
    assert not (d / "1Hz_daystore/2015/06").exists()
    assert not (d / "1Hz_mosaic_mnth/2020/11").exists()
    assert not (d / "panel_cache/keyA/2020-11").exists()
    assert not (d / ".markers/daystore-2015-06.v7.done").exists()
    # live: touched after the job started, in every key variant
    assert (d / "1Hz_daystore/2008/02/index.json").exists()
    assert (d / "1Hz_mosaic_mnth/2008/07/index.json").exists()
    assert (d / "panel_cache/keyA/2008-07").exists()
    assert (d / "panel_cache/keyB/2008-07").exists()
    assert (d / ".markers/mosaic-2008-07.v7.done").exists()
    # an old unmarked partial copy goes; a dead job's venv goes, a live one stays
    assert not (d / "1Hz_daystore/2011/12").exists()
    assert not (tmp_path / "market-jepa-venv-222").exists()
    assert (tmp_path / "market-jepa-venv-111").exists()
    assert "reclaimed 4 month(s)" in r.stdout, r.stdout


def test_the_bundle_reclaims_before_it_stages():
    body = (ROOT / "scripts/generic/lib/train_bundle_body.sh").read_text()
    i = body.index("reclaim_scratch_if_low || true")
    assert i < body.index('if ! stage_all_daystore "${TRAIN_START}"')
