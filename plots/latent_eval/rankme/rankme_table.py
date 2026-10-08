"""Appendix table: RankMe and embedding width for every row of the latent table.

RankMe left the latent table (plots/core/fixed_panel_table.tex) on 2026-09-23.
It is bounded by the embedding width, and with the frozen TSFMs in that table
(then 6912 / 832 / 11520 wide, channels concatenated, against everyone else's
384) one column with daggers was no longer a column. Channel-averaged they are
768 / 832 / 1280 -- still not 384. This table prints the width beside it; the TSFMs'
layer-wise RankMe is its own figure (plots/tsfm_layers/latent_sweep.py
--metric rankme).

SAME ROWS, SAME READOUT. Rows, labels, families and citations come from
fixed_panel_table's own loaders, so this table cannot list an arm the latent
table does not. RankMe is scripts/eval/rankme_fullday.py over the latent
suite's full-day embeddings -- mean-pooled for every arm, floor included --
in plots/core/rankme_latent.json; the three TSFM rows (last layer, channels
averaged) come from plots/tsfm_layers/rankme_tsfm.json, the same file the
layer-wise figure reads.

    uv run python plots/latent_eval/rankme/rankme_table.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (REPO, REPO / "plots", REPO / "plots/core", REPO / "plots/task_corr",
           REPO / "plots/latent_eval/fixed_panel"):
    sys.path.insert(0, str(_p))

import fixed_panel_table as fpt  # noqa: E402

OUT = HERE / "rankme_table.tex"
CITE = "garrido2023rankmeassessingdownstreamperformance"
LABEL = "tab:rankme_latent"
FIG_REF = "fig:tsfm_rankme"
CAPTION = (
    r"RankMe~\citep{" + CITE + r"} of each frozen embedding on the latent "
    r"suite's full-day panels of Table~\ref{tab:latent_results}, read at the "
    r"same mean pooling, mean $\pm$ SE over the eval months, beside the "
    r"embedding width that bounds it. RankMe counts how many directions the "
    r"embedding spends its variance on; it is not a score -- the supervised "
    r"models, the strongest forecasters, have the lowest -- and a width far "
    r"from 384 puts a row off the rest of the column's scale. The frozen TSFMs "
    r"are read at their last layer with the nine per-channel states "
    r"averaged, the prediction evals' readout; "
    r"Figure~\ref{" + FIG_REF + r"} sweeps every layer.")


def render(rows, rankme, *, tsfm=(), caption=CAPTION, label=LABEL,
           size="footnotesize"):
    """One row per latent-table row: width and RankMe, under its families."""
    from style import render_latex_table
    from task_rank_corr import latex_size_label

    roster = fpt.latex_roster()
    cites, fam_cites = fpt.latex_cites()

    def cells(k):
        got = rankme.get(k)
        if not got:
            return ["--", "--"]
        mu, se, _n, ws = got
        w = ", ".join(str(x) for x in sorted(ws))
        return [f"${w}$", f"${mu:.1f} \\pm {se:.1f}$"]

    body = []

    def group(name, cite=""):
        body.append([r"\addlinespace \multicolumn{3}{l}{\emph{"
                     + fpt._tex_escape(name) + r"}" + fpt._citep(cite) + r"}"])

    by_fam: dict[str, list] = {}
    for k, reg_lbl in rows:
        if k == fpt.FLOOR:
            continue
        lbl, fam = roster.get(k, (reg_lbl, "Other"))
        by_fam.setdefault(fam, []).append((k, fpt.trim_family(lbl, fam)))
    order = fpt.FAMILY_ORDER + [f for f in by_fam if f not in fpt.FAMILY_ORDER]
    for fam in order:
        if fam not in by_fam:
            continue
        group(fam, fam_cites.get(fam, ""))
        for k, lbl in by_fam[fam]:
            body.append([latex_size_label(fpt._tex_escape(lbl), lbl)
                         + fpt._citep(cites.get(k, ""))] + cells(k))
    if tsfm:
        group(fpt.TSFM_BLOCK + " (last layer, channels averaged)")
        for k, lbl in tsfm:
            body.append([fpt._tex_escape(lbl)] + cells(k))
    if fpt.FLOOR in rankme:
        group("Untrained floor")
        body.append([fpt._tex_escape(fpt.MODEL_SPECS[fpt.FLOOR]["label"])]
                    + cells(fpt.FLOOR))
    return render_latex_table(
        body, header_rows=[["", "Width", "RankMe"]], col_spec="lcc",
        caption=caption, label=label, caption_above=True, small=bool(size),
        size=size or None)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()

    geo, tags = fpt.load_geo(fpt.DEFAULT_TAGS, required=False)
    rows = fpt.select_rows(geo, tags)
    rankme = fpt.load_rankme()
    if not rankme:
        raise SystemExit(f"no {fpt.RANKME_JSON} -- scripts/eval/"
                         f"rankme_fullday.py --json writes it")
    tsfm = [(k, l) for k, l in fpt.TSFM_ROWS if k in geo]
    if fpt.TSFM_RANKME.is_file():
        # Only the three reported layers: the file also holds 44 others.
        trk = fpt.load_rankme(fpt.TSFM_RANKME)
        rankme.update({k: trk[k] for k, _ in fpt.TSFM_ROWS if k in trk})
    missing = [k for k, _ in tsfm if k not in rankme]
    if missing:
        print(f"  no RankMe for {missing}; those rows print --",
              file=sys.stderr)
    Path(a.out).write_text(render(rows, rankme, tsfm=tsfm) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
