"""Materialize an MDS month directory in place (idempotent).

Decompress any shard present only as .zstd, size-verify every raw shard
against index.json, rewrite the index to declare no compression, and delete
the compressed copies. After this, mosaicml streaming reads raw .mds files
directly and never runs its lazy per-shard decompression — which is unsafe
when several independent SLURM jobs share one staged dataset directory
(concurrent .mds.tmp -> .mds renames race and die with FileNotFoundError).

Runs under stage_data.sh's per-month flock via the job venv (zstandard is a
mosaicml-streaming dependency).

Usage: python materialize_mds.py <month_dir>
"""

from __future__ import annotations

import json
import os
import sys

import zstandard


def main() -> int:
    # --keep-zips: for the CANONICAL mosaic on the data host the compressed copies
    # stay (staging elsewhere pulls them); only the index stops declaring them.
    keep_zips = "--keep-zips" in sys.argv
    if keep_zips:
        sys.argv.remove("--keep-zips")
    target = sys.argv[1]
    index_path = os.path.join(target, "index.json")
    with open(index_path) as fh:
        idx = json.load(fh)

    zips: list[str] = []
    bad: list[str] = []
    changed = False
    for s in idx["shards"]:
        raw_name = s["raw_data"]["basename"]
        raw = os.path.join(target, raw_name)
        want = int(s["raw_data"]["bytes"])
        zip_info = s.get("zip_data")
        zip_path = (os.path.join(target, zip_info["basename"])
                    if zip_info else None)

        # A RAW SHARD OF THE WRONG SIZE IS STALE, NOT CORRUPT, and is
        # re-decompressed rather than failed on. The staging rsync ships
        # .mds.zstd only (see stage_data.sh), so a raw shard left on the node
        # by an EARLIER dataset is never overwritten by the transfer -- which
        # is exactly what happened when the mosaic went dense: every node
        # holding materialized sparse shards checked them against the new
        # index and reported 120 size mismatches per month. Trusting the
        # compressed copy is right in general: it is the file the source
        # actually sent, and the index describes it.
        stale = (os.path.exists(raw)
                 and os.path.getsize(raw) != want
                 and zip_path and os.path.exists(zip_path))
        if stale:
            os.remove(raw)
        if not os.path.exists(raw):
            if not zip_path or not os.path.exists(zip_path):
                bad.append(f"{raw_name}: raw missing and no compressed copy")
                continue
            tmp = f"{raw}.tmp.{os.getpid()}"
            try:
                with open(zip_path, "rb") as src, open(tmp, "wb") as dst:
                    zstandard.ZstdDecompressor().copy_stream(src, dst)
                os.replace(tmp, raw)
            except Exception as e:  # noqa: BLE001 - report and fail the month
                if os.path.exists(tmp):
                    os.remove(tmp)
                bad.append(f"{raw_name}: decompress failed: {e}")
                continue

        have = os.path.getsize(raw)
        if have != want:
            # Still wrong AFTER a fresh decompression, so the compressed copy
            # and the index disagree -- a genuinely broken shard.
            bad.append(f"{raw_name}: size {have} != index {want}"
                       + (" (after re-decompressing)" if stale else ""))
            continue

        if zip_path:
            zips.append(zip_path)
        if s.get("compression") is not None or s.get("zip_data") is not None:
            s["compression"] = None
            s["zip_data"] = None
            changed = True

    if bad:
        for b in bad:
            print(f"MATERIALIZE ERROR {b}", file=sys.stderr)
        return 1

    if changed:
        tmp = index_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(idx, fh)
        os.replace(tmp, index_path)
    if not keep_zips:
        for z in zips:
            if os.path.exists(z):
                os.remove(z)
    print(f"materialized {target}: {len(idx['shards'])} shards"
          f"{' (index rewritten)' if changed else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
