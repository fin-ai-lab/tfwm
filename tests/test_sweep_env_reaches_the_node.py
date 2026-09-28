"""A sweep knob that does not travel is a SILENT duplicate run.

run_all_months_sweep.sh does not forward the caller's environment. It builds an
EXPLICIT allowlist in the ssh command and exports those names in the remote
shell; `sbatch --export=ALL` then propagates that shell, so a variable missing
from the list simply never arrives and the sweep file's own default applies.

That is not a crash. On 2026-09-18 a wave launched with FT_BLR=1e-6 trained 31
months at the default 1e-5 -- an exact duplicate of the wave it was meant to be
compared against, discovered only by reading `Learning rate:` out of a job log
twenty minutes in.

So: every knob an ssl_finetune sweep reads from the environment must either be
on the launcher's allowlist or be set by the bundle on the node.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/pythia/run_all_months_sweep.sh"
BUNDLE = ROOT / "scripts/pythia/lib/train_bundle_body.sh"
SWEEPS = sorted((ROOT / "scripts/sweeps/ssl_finetune").glob("*.sh")) + sorted(
    (ROOT / "scripts/sweeps").glob("*scaling*.sh"))

# Read with a default by the sweep, but supplied on the NODE rather than by the
# caller: the staging paths the sweep library defines for itself.
NODE_LOCAL = {"SSL_BASE_DIR", "SSL_HEAD_DIR"}


def _allowlisted() -> set[str]:
    """Names the launcher exports into the remote shell before sbatch."""
    text = LAUNCHER.read_text()
    line = next(l for l in text.splitlines() if "&& sbatch" in l)
    return set(re.findall(r"([A-Z_][A-Z_0-9]*)='", line))


def _bundle_sets() -> set[str]:
    return set(re.findall(r"^\s*(?:export\s+)?([A-Z_][A-Z_0-9]*)=",
                          BUNDLE.read_text(), flags=re.M))


def _knobs_read(path: Path) -> set[str]:
    return set(re.findall(r"\$\{([A-Z_][A-Z_0-9]*):-", path.read_text()))


def test_every_env_read_sweep_knob_reaches_the_node():
    reachable = _allowlisted() | _bundle_sets() | NODE_LOCAL
    missing = {}
    for f in SWEEPS:
        gone = _knobs_read(f) - reachable
        if gone:
            missing[f.name] = sorted(gone)
    assert not missing, (
        f"these sweep knobs are read from the environment but never travel to "
        f"the compute node, so setting them silently changes nothing: {missing}. "
        f"Add them to the export list in {LAUNCHER.name}.")


def test_ft_blr_specifically_is_forwarded():
    """The one that cost a wave."""
    assert "FT_BLR" in _allowlisted()


def test_the_allowlist_is_found_at_all():
    """If the sbatch line is reformatted, the parser above must not go quiet."""
    got = _allowlisted()
    assert {"TRAIN_START", "EVAL_START", "COMMIT_HASH"} <= got, got
