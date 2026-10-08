"""Freeze the probe-breadth rows the core artifacts read into the repo.

probe_fit_table.py and probe_fit_breadth.py read the probe sweep's result
jsons and manifest indexes from PROBE_BREADTH, which also holds the sweep's
embeddings and so cannot ship. This writes just the rows they read to
paths.SNAPSHOT, which paths.index_rows() / result_files() fall back to when
the live directory is absent -- i.e. on a clone of the public repo.

Index rows keep only the fields the readers key on; ckpt_dir is a lab path and
is dropped. Re-run after the sweep changes, then plots/core/regen.sh, and
commit the snapshot with the tables it produced.

    uv run python plots/core/snapshot_probe_breadth.py
"""
import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import PROBE_BREADTH, RESULTS, SNAPSHOT, index_rows, result_files  # noqa: E402

INDEX_KEYS = ("arm", "series_key", "eval_month", "run_id", "fit_months")


def main():
    if not PROBE_BREADTH.exists():
        raise SystemExit(f"{PROBE_BREADTH} is not mounted; nothing to snapshot")
    indexes = [{k: r[k] for k in INDEX_KEYS if k in r} for r in index_rows()]
    results = dict(result_files(RESULTS))
    blob = json.dumps({"indexes": indexes, "results": results},
                      separators=(",", ":"), sort_keys=True).encode()
    # mtime=0 so an unchanged sweep rewrites identical bytes.
    SNAPSHOT.write_bytes(gzip.compress(blob, mtime=0))
    print(f"{SNAPSHOT}: {len(indexes)} index rows, {len(results)} result files, "
          f"{sum(map(len, results.values()))} probe rows, "
          f"{SNAPSHOT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
