"""Summary table + figure for the representational factor-structure result.

Reads decode_loadings<tag>.json and subspace_alignment<tag>.json (default
tag _14mo — the common 14 post-2015 months where the LeJEPA variants
exist) and emits:

  * a markdown table (stdout): decode r for total loadings 1-4 and
    r2_k, plus subspace rhobar. Bold = top two per column; strikethrough
    = below the 5-seed random-ViT floor. The floor row is the seed mean.
  * factor_summary.png/.pdf — two-panel horizontal dot plot: mean
    decode r over loadings 1-4 (left) and rhobar (right), with the random
    floor as a line + seed band.

The liquidity-control variants (control/emb+ctl/resid) were retired
2026-08-26; --variant emb is the reported number. Pre-retirement JSONs
still carry the other variants and can be rendered with --variant resid.

Usage:  uv run python plots/latent_eval/factors/factor_summary.py [--tag _14mo]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from style import SERIES_STYLES, apply_style, save_figure  # noqa: E402

RAND = [f"randvit_s{i}" for i in range(5)]
ROWS = [
    ("lejepa", "lejepa"), ("k2", "cross_stock_k2"),
    ("k2ind", "cross_stock_k2ind"), ("warp", "aug_warp"),
    ("noise", "aug_noise"), ("volj", "aug_volj"),
    ("pricej", "aug_pricej"), ("chdrop", "aug_chdrop"),
    ("supervised_return", None), ("supervised_vol", None),
    ("supervised_spread", None), ("multihead", None),
    ("dino", "dino"), ("byol", "byol"), ("cpc", "cpc"),
    ("mae", "mae"), ("ijepa", "ijepa"),
    ("ts2vec_cb028b", "ts2vec"), ("cost_4c05e0", "cost"),
    ("tfc_4c05e0", "tfc"), ("timemae_4c05e0", "timemae"),
    ("tsfm_chronos2_l5", "chronos2"), ("tsfm_chronos2_l7", "chronos2"),
    ("tsfm_chronos2_l11", "chronos2"), ("tsfm_timesfm_l12", "timesfm"),
    ("tsfm_timesfm_l20", "timesfm"), ("tsfm_sundial_l5", "sundial"),
    ("tsfm_kronos_l2", "kronos"), ("tsfm_kronos_l5", "kronos"),
    ("tsfm_kronos_l9", "kronos"),
]
LABELS = {
    "lejepa": "LeJEPA", "k2": "Cross-stock K=2", "k2ind": "K=2 same-ind.",
    "warp": "Time warp", "noise": "Gauss. noise", "volj": "Volume noise",
    "pricej": "Price jitter", "chdrop": "Channel drop",
    "supervised_return": "Supervised (ret)", "supervised_vol":
    "Supervised (vol)", "supervised_spread": "Supervised (spread)",
    "multihead": "Supervised (multi)", "dino": "DINO", "byol": "BYOL",
    "cpc": "CPC", "mae": "MAE", "ijepa": "I-JEPA",
    "ts2vec_cb028b": "TS2Vec", "cost_4c05e0": "CoST", "tfc_4c05e0": "TF-C",
    "timemae_4c05e0": "TimeMAE", "tsfm_chronos2_l5": "Chronos-2 L5",
    "tsfm_chronos2_l7": "Chronos-2 L7", "tsfm_chronos2_l11": "Chronos-2 L11",
    "tsfm_timesfm_l12": "TimesFM L12", "tsfm_timesfm_l20": "TimesFM L20",
    "tsfm_sundial_l5": "Sundial L5", "tsfm_kronos_l2": "Kronos L2",
    "tsfm_kronos_l5": "Kronos L5", "tsfm_kronos_l9": "Kronos L9",
}
LOAD_COLS = ["tot1", "tot2", "tot3", "tot4", "r2_k"]


def color_of(key, style_key):
    if style_key and style_key in SERIES_STYLES:
        return SERIES_STYLES[style_key]["color"]
    return "tab:gray" if key.startswith(("supervised", "multihead")) \
        else "tab:gray"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_14mo")
    ap.add_argument("--variant", default="emb",
                    choices=["emb", "resid", "emb+ctl"],
                    help="decode feature set shown in the table (resid / "
                         "emb+ctl exist only in pre-2026-08-26 JSONs)")
    args = ap.parse_args()

    dec = json.load(open(HERE / f"decode_loadings{args.tag}.json"))
    sub = json.load(open(HERE / f"subspace_alignment{args.tag}.json"))
    n_mo = len(dec["months"])
    global ROWS
    ROWS = [(k, sk) for k, sk in ROWS
            if f"{k}|{args.variant}" in dec["results"] and k in sub["results"]]

    def dec_mean(series, target):
        return float(np.mean(dec["results"][f"{series}|{args.variant}"][target]))

    def sub_mean(series):
        return float(np.mean(sub["results"][series]["rhobar"]))

    floor = {t: np.mean([dec_mean(s, t) for s in RAND]) for t in LOAD_COLS}
    floor["rhobar"] = np.mean([sub_mean(s) for s in RAND])
    vals = {k: {**{t: dec_mean(k, t) for t in LOAD_COLS},
                "rhobar": sub_mean(k)} for k, _ in ROWS}

    cols = LOAD_COLS + ["rhobar"]
    top2 = {c: set(sorted(vals, key=lambda k: -vals[k][c])[:2]) for c in cols}

    hdr = ["model"] + cols
    print(f"{args.variant} decode r (n={n_mo} months) + subspace rhobar; "
          "bold top-2, struck < random-ViT floor")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for k, _ in ROWS:
        cells = []
        for c in cols:
            txt = f"{vals[k][c]:+.3f}" if c != "rhobar" else f"{vals[k][c]:.3f}"
            if vals[k][c] < floor[c]:
                txt = f"~~{txt}~~"
            if k in top2[c]:
                txt = f"**{txt}**"
            cells.append(txt)
        print(f"| {LABELS[k]} | " + " | ".join(cells) + " |")
    print("| Random ViT (5 seeds) | "
          + " | ".join(f"{floor[c]:+.3f}" if c != "rhobar"
                       else f"{floor[c]:.3f}" for c in cols) + " |")

    # ── figure ─────────────────────────────────────────────────────────────
    apply_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 1.6 + 0.22 * len(ROWS)), sharey=True)
    order = list(reversed([k for k, _ in ROWS]))
    ypos = np.arange(len(order))
    load_avg = {k: np.mean([vals[k][t] for t in LOAD_COLS[:4]]) for k in vals}
    rand_load = [np.mean([dec_mean(s, t) for t in LOAD_COLS[:4]])
                 for s in RAND]
    rand_rho = [sub_mean(s) for s in RAND]
    panels = [
        (axes[0], {k: load_avg[k] for k in vals}, rand_load,
         "Loading decode\n(OOS $r$, factors 1–4)"),
        (axes[1], {k: vals[k]["rhobar"] for k in vals}, rand_rho,
         r"Subspace alignment ($\bar\rho$)"),
    ]
    style_of = dict(ROWS)
    for ax, v, rand_vals, title in panels:
        ax.axvspan(min(rand_vals), max(rand_vals), color="0.85", zorder=0)
        ax.axvline(np.mean(rand_vals), color="0.4", lw=1.0, ls="--",
                   zorder=1, label="Random ViT (5 seeds)")
        for i, k in enumerate(order):
            ax.plot(v[k], ypos[i], "o", ms=6,
                    color=color_of(k, style_of[k]), zorder=3)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([LABELS[k] for k in order], fontsize=8)
    axes[1].legend(loc="lower right", fontsize=7, frameon=False)
    fig.suptitle(
        f"Latent factor structure in the embeddings ({n_mo} months)",
        fontsize=10)
    fig.tight_layout()
    save_figure(fig, HERE / f"factor_summary{args.tag}")
    print(f"\nwrote {HERE / f'factor_summary{args.tag}'}.png/.pdf")


if __name__ == "__main__":
    main()
