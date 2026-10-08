"""A train-only wave must stage no panels and be gated on none.

THE BUG THIS PINS. `panel_months_needed` and `require_panel_cache` honoured
POST_TRAIN_PROBE but not POST_TRAIN_IC_EVAL, so a wave launched with
POST_TRAIN_IC_EVAL=0 -- which scores nothing at all -- still staged the
probe-fit panel (~22 GB/job; 31 jobs on 2026-09-14) and still `exit 1`'d the
job when it was absent. Panels are SCORING inputs; a run that does not score
must neither move them nor be blocked on them.

POST_TRAIN_PROBE=0 remains a real knob: it drops the probe-fit panel while
KEEPING the eval panel, which is what head-only scoring wants.
"""

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LIB = REPO / "scripts" / "generic" / "lib" / "stage_data.sh"

BASE = {
    "TRAIN_START": "2011-10-01", "TRAIN_END": "2012-03-31",
    "EVAL_START": "2012-04-01", "EVAL_END": "2012-04-30",
}


def _run(snippet: str, **env) -> str:
    """Source the library with a clean env and run one snippet."""
    e = {"PATH": "/usr/bin:/bin", **BASE, **{k: str(v) for k, v in env.items()}}
    r = subprocess.run(
        ["bash", "-c", f'source "{LIB}" >/dev/null 2>&1; {snippet}'],
        capture_output=True, text=True, env=e, cwd=REPO,
    )
    return r.stdout.strip()


def _months(**env) -> list[str]:
    return _run("panel_months_needed", **env).split()


def test_train_only_stages_no_panel_at_all():
    assert _months(POST_TRAIN_IC_EVAL=0) == []
    # and not because POST_TRAIN_PROBE happened to be off too
    assert _months(POST_TRAIN_IC_EVAL=0, POST_TRAIN_PROBE=1) == []


def test_scoring_run_still_stages_both_roles():
    """The default path is unchanged: probe-fit month + eval month."""
    assert _months() == ["2012-03", "2012-04"]
    assert _months(POST_TRAIN_IC_EVAL=1, POST_TRAIN_PROBE=1) == ["2012-03", "2012-04"]


def test_head_only_still_drops_the_probe_panel_but_keeps_eval():
    assert _months(POST_TRAIN_PROBE=0) == ["2012-04"]


def test_require_panel_cache_never_blocks_a_train_only_wave():
    """It exit 1's the whole job, so this is the expensive half of the bug."""
    out = _run(
        'BUNDLE_SWEEPS=(sweeps/ssl_lejepa_all.sh); '
        'MJ_PANEL_CACHE=/nonexistent DATA_DIR=/nonexistent '
        'require_panel_cache 2012-03 2012-04 && echo PASS || echo BLOCKED',
        POST_TRAIN_IC_EVAL=0,
    )
    assert out.endswith("PASS"), out


def test_require_panel_cache_still_blocks_a_scoring_run_with_no_cache():
    """The guard must keep working where it matters — otherwise this test
    would pass on a function that simply returns 0 always."""
    out = _run(
        'BUNDLE_SWEEPS=(sweeps/ssl_lejepa_all.sh); '
        'MJ_PANEL_CACHE=/nonexistent DATA_DIR=/nonexistent '
        'require_panel_cache 2012-03 2012-04 && echo PASS || echo BLOCKED',
        POST_TRAIN_IC_EVAL=1,
    )
    assert out.endswith("BLOCKED"), out
