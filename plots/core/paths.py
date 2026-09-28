"""Every input the three core results read, named exactly once.

This module exists because the inputs do NOT live here. They are written in
place, next to the code that produces them, and several of them are read by
other consumers under a ``<dir>/<stem><tag>.json`` convention. Moving a file
into this folder would leave its producer writing to the old path and this
folder holding a stale copy that still EXISTS -- so nothing would raise and
the tables would quietly go out of date. The paths move here; the bytes stay.

Who writes what:

  probe_head_ic.json        plots/metrics/build_probe_head_ic.py
  rankme_6mo.json           scripts/eval/rankme.py --json <path>
  fixed_panel_P3S2*.json    plots/latent_eval/fixed_panel/fixed_panel_metrics.py
                            (and merge_month_shards.py; also written by
                            scripts/pythia/slurm_tsfm_latent.sh ON PYTHIA,
                            which hardcodes the plots/latent_eval path)
  decode_loadings*.json     plots/latent_eval/factors/decode_loadings.py
  subspace_alignment*.json  plots/latent_eval/factors/subspace_alignment.py
  probe_breadth_snapshot    plots/core/snapshot_probe_breadth.py

Changing any path below means changing its producer in the same commit, and
for the fixed-panel JSONs that includes a script that runs on pythia from a
checkout which may sit at a different commit.
"""
import gzip
import json
import os
import sys
from pathlib import Path

CORE = Path(__file__).resolve().parent
ROOT = CORE.parents[1]

# --- OUTSIDE THE REPO, with an in-repo snapshot. ---
# The probe-breadth sweep's per-(arm, month) results. The live directory holds
# the embeddings too (2.2 TB), so it cannot ship; SNAPSHOT is the ~1 MB of
# probe rows and manifest indexes the two probe artifacts actually read, frozen
# by snapshot_probe_breadth.py. THE LIVE DIRECTORY WINS WHENEVER IT EXISTS, so
# on the lab machines a stale snapshot is never read in place of fresh results;
# every load says on stderr which source it used. TFWM_PROBE_BREADTH points at
# a different live directory.
PROBE_BREADTH = Path(os.environ.get("TFWM_PROBE_BREADTH", "/data/lab/probe_breadth"))
RESULTS = PROBE_BREADTH / "results"
# The frozen TSFMs ride the same sweep (build_probe_breadth_manifest.py
# --tsfm), so their results sit in RESULTS beside everyone else's and only
# need their own index to be named.
INDEXES = [PROBE_BREADTH / "manifest_all.index.json",
           PROBE_BREADTH / "manifest_floor.index.json",
           PROBE_BREADTH / "manifest_tsfm.index.json"]
SNAPSHOT = CORE / "probe_breadth_snapshot.json.gz"


def _snapshot():
    if not SNAPSHOT.exists():
        raise SystemExit(f"neither {PROBE_BREADTH} nor {SNAPSHOT} exists")
    return json.loads(gzip.decompress(SNAPSHOT.read_bytes()))


def index_rows():
    """Every manifest-index row, from the live directory or the snapshot."""
    if PROBE_BREADTH.exists():
        return [r for p in INDEXES if p.exists()
                for r in json.loads(p.read_text())]
    print(f"probe-breadth indexes: snapshot {SNAPSHOT.name}", file=sys.stderr)
    return _snapshot()["indexes"]


def result_files(results=None):
    """[(file name, rows)] sorted by name, from `results` or the snapshot.

    A file that does not parse is one the sweep is still writing and is
    skipped, as the readers always did.
    """
    results = Path(results or RESULTS)
    if results.exists():
        out = []
        for f in sorted(results.glob("*.json")):
            try:
                out.append((f.name, json.loads(f.read_text())))
            except json.JSONDecodeError:
                continue
        return out
    print(f"probe-breadth results: snapshot {SNAPSHOT.name}", file=sys.stderr)
    return sorted(_snapshot()["results"].items())

# --- in-repo, beside their producers ---
METRICS = ROOT / "plots/metrics"
LATENT = ROOT / "plots/latent_eval"
PANELS = LATENT / "fixed_panel"
FACTORS = LATENT / "factors"

# PER-ARM head ICs, read off the sweep's own checkpoints. supervised_head_ic.json
# is deliberately NOT used: it records no arm, so the multihead would read the
# specialists' numbers.
HEAD_JSON = METRICS / "probe_head_ic.json"
RANKME_JSON = METRICS / "rankme_6mo.json"

# OPTIONAL and ABSENT as of 2026-09-17. probe_fit_breadth.load_finetune()
# returns early when this is missing, so the finetuned overlay silently draws
# nothing rather than raising. If that curve is meant to appear, this file is
# why it does not.
FT_JSON = METRICS / "ssl_finetune_breadth.json"
