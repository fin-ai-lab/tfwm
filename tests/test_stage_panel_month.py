"""stage_data.sh:stage_panel_month -- a month staged under ONE cache key is
not a month staged.

A panel month lives under one directory per cache key (eval, probe-fit, the
risk-factor variants). On 2026-09-20 pgpu014 held 2008-10's probe-fit panel
complete and its eval panel as an empty directory left by a job that died on
ENOSPC; the stager counted the one .ready, said "already staged", and
require_panel_cache failed the job on a panel that sat built on bll01. The
rsync is incremental, so the fix is to run it every time and let the marker
decide only the message. These tests drive the function with rsync faked as a
shell function copying from a fabricated bll01.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "scripts/pythia/lib/stage_data.sh"
YM = "2008-10"


def _panel(root: Path, key: str, ym: str = YM, ready: bool = True) -> Path:
    d = root / key / ym
    d.mkdir(parents=True, exist_ok=True)
    (d / "panel.parquet").write_bytes(b"x" * 64)
    if ready:
        (d / ".ready").write_text("")
    return d


def _run(tmp: Path, rsync_ok: bool = True, marked: bool = False) -> subprocess.CompletedProcess:
    data = tmp / "data"
    (data / "panel_cache").mkdir(parents=True, exist_ok=True)
    mk = data / ".markers"
    mk.mkdir(exist_ok=True)
    if marked:
        (mk / f"panel-{YM}.v7.done").write_text("")
    src = tmp / "bll01"
    fail = "" if rsync_ok else "return 23;"
    script = f"""
    rsync() {{ {fail}
        local s="${{@: -2:1}}" d="${{@: -1}}"; s="${{s#*:}}"
        for k in "$s"*/; do kn=$(basename "$k")
            [ -d "$k/{YM}" ] || continue
            mkdir -p "$d/$kn"; cp -r "$k/{YM}" "$d/$kn/"
        done; }}
    export -f rsync
    DATA_DIR="{data}"; LOCK_DIR="{data}/.locks"; MARKER_DIR="{data}/.markers"
    REPO_DIR="{ROOT}"; BLL01_HOST=x; BLL01_DATA=x; BLL01_METADATA=x
    BLL01_PANEL_DIR="{src}"
    source "{LIB}"
    stage_panel_month {YM}
    """
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_a_missing_key_is_pulled_past_a_complete_one(tmp_path):
    src = tmp_path / "bll01"
    _panel(src, "evalkey")
    _panel(src, "probekey")
    node = tmp_path / "data/panel_cache"
    _panel(node, "probekey")                       # complete, with .ready
    (node / "evalkey" / YM).mkdir(parents=True)    # the ENOSPC leftover
    r = _run(tmp_path, marked=True)
    assert r.returncode == 0, r.stderr
    assert (node / "evalkey" / YM / ".ready").exists(), r.stdout
    assert "staged (2 panel(s), 1 were here)" in r.stdout, r.stdout


def test_a_complete_month_is_refreshed_not_reported_as_new(tmp_path):
    src = tmp_path / "bll01"
    _panel(src, "evalkey")
    node = tmp_path / "data/panel_cache"
    _panel(node, "evalkey")
    r = _run(tmp_path, marked=True)
    assert r.returncode == 0, r.stderr
    assert "already staged (1 panel(s)), refreshed" in r.stdout, r.stdout
    assert (tmp_path / "data/.markers" / f"panel-{YM}.v7.done").exists()


def test_a_failed_rsync_keeps_what_is_here(tmp_path):
    _panel(tmp_path / "bll01", "evalkey")
    node = tmp_path / "data/panel_cache"
    _panel(node, "probekey")
    r = _run(tmp_path, rsync_ok=False, marked=True)
    assert r.returncode == 0, r.stderr
    assert "keeping the 1 staged panel(s)" in r.stdout, r.stdout
    assert (node / "probekey" / YM / ".ready").exists()


def test_nothing_upstream_is_not_an_error(tmp_path):
    (tmp_path / "bll01").mkdir()
    r = _run(tmp_path)
    assert r.returncode == 0, r.stderr
    assert "none built upstream" in r.stdout, r.stdout
