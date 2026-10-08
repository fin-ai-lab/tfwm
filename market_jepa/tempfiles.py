"""Machine-safe temporary-directory defaults for market-jepa workloads."""

from __future__ import annotations

import getpass
import os
import tempfile
from pathlib import Path


LAB_TMP_ROOT = Path("lab/tmp")


def configure_tempdir() -> Path:
    """Use lab storage when no temporary directory was explicitly selected.

    Python's :mod:`tempfile` caches its first answer, so reset that cache after
    installing the default. Explicit ``TMPDIR`` and ``MARKET_JEPA_TMPDIR``
    settings always win, which keeps cluster-local scratch injectable.
    """
    configured = os.environ.get("MARKET_JEPA_TMPDIR") or os.environ.get("TMPDIR")
    if configured:
        destination = Path(configured).expanduser()
    elif LAB_TMP_ROOT.is_dir() and os.access(LAB_TMP_ROOT, os.W_OK):
        destination = LAB_TMP_ROOT / getpass.getuser()
    else:
        return Path(tempfile.gettempdir())

    destination.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(destination)
    tempfile.tempdir = None
    return destination
