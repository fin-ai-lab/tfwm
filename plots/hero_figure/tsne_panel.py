"""Hero figure: one fixed panel, t-SNE'd under two LeJEPA pairings.

The panel is one of the ten the latent table scores (plots/core/
fixed_panel_table.tex, T1-T4): 3 FF49 industries x 2 full-presence firms,
one full-day view per (firm, trading day) of one eval month, so ~22 days x 6
firms = ~132 points. The SAME points are embedded by the two arms of the
table's LeJEPA block that sit at opposite ends of it:

  * Same Stock, Diff. View   (pair_rrc_6mo)   -- best on T3 by far: views
    cluster by FIRM.
  * C-S, Same Industry       (pair_k2ind_6mo) -- best on T1/T2: views
    cluster by DAY, partners side by side.

Each arm is read exactly as the table reads it: the six-month manifest row
for the eval month, panel_lib.LATENT_POOL, the information-token panel
(fixedRi cache). t-SNE is fit per arm (perplexity 30, seed 42, as
metric_diagram.py) -- the two maps share points, not axes.

Colors: one hue per industry, dark/light = the two firms in it
(metric_diagram.SECTOR_SHADES). Days are drawn two ways, one file each:

  * ``labels`` -- the day of month printed in every marker.
  * ``shapes`` -- no numbers: each day its own marker shape (up to
    len(DAY_MARKERS) days), firm colour; grey centroid lines and a grey
    centroid in the day's shape on top (``lines=False`` drops the lines;
    ``industry=True`` adds each industry's centroid as a star in its colour).
  * ``centroid`` -- day numbers, plus a grey line from each view to its
    day's 2-D centroid (grey diamond), drawn OVER the dots; no partner lines.
  * ``partner`` -- day numbers, plus a coloured line from each view to its
    industry partner's view on the same day (the T1 target) and a dashed
    grey line to that day's 2-D centroid (grey diamond). Short lines =
    partners, and days, co-locate.
  * ``spokes`` -- a thin line from each view to its day's 2-D centroid,
    with the day of month at the centroid. Tight short spokes = the day is
    a cluster; long crossing spokes = days are scattered.

PANEL CHOICE. ``--panel -1`` (default) picks the panel where the two arms
differ most in the direction the table says they differ (k2ind better on
T1+T2 percentile, rrc better on T3), and prints every panel's T1-T4 so the
pick is visible. That is a hand-picked illustration, not a typical panel:
say so in the caption, or pin ``--panel`` to something unremarkable.

Embeddings are cached under panel_lib.CACHE_DIR (one GPU forward per arm and
month); a re-plot is CPU-only.

Run (from the repo root):
    uv run python plots/hero_figure/tsne_panel.py --month 2023-01
    uv run python plots/hero_figure/tsne_panel.py --month 2023-01 --panel 4
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXED_PANEL = ROOT / "plots" / "latent_eval" / "fixed_panel"
sys.path.insert(0, str(FIXED_PANEL))

# The six-month wave has no repo-manifest row; its manifest sits in scratch
# (run_6mo_eval.sh names it the same way). Set before the first lookup.
MANIFEST_6MO = Path(
    "lab/market-jepa-checkpoints/_scratch/latent_eval/manifest_6mo.json")
os.environ.setdefault(
    "MJ_NOCLAMP_MANIFEST",
    f"{ROOT / 'plots/metrics/noclamp_manifest.json'}{os.pathsep}{MANIFEST_6MO}")

import panel_lib as eg  # noqa: E402
from fixed_panel_metrics import (  # noqa: E402
    _load_manifest_encoder, _manifest_row, load_panel_batches, panel_metrics,
)
from metric_diagram import SECTOR_SHADES  # noqa: E402
from style import save_figure  # noqa: E402

P, S = 3, 2
ARMS = [("pair_rrc_6mo", "Same Stock, Diff. View"),
        ("pair_k2ind_6mo", "Cross Stock, Same Industry")]
SICCODES = ROOT / "scripts" / "data" / "ff_raw" / "Siccodes49.txt"
INK = "#333333"
CENTROID_GREY = "#7a7a7a"
# One marker per day for the "shapes" mode, in date order.
DAY_MARKERS = ["o", "s", "^", "D", "v"]


def ff49_names() -> dict[int, str]:
    """{code: long name} off the header lines of Siccodes49.txt."""
    names = {}
    for line in SICCODES.read_text(errors="replace").splitlines():
        m = re.match(r"^\s*(\d+)\s+([A-Za-z]\S*)\s+(.+?)\s*$", line)
        if m:
            names[int(m.group(1))] = m.group(3)
    return names


def embed_month(ev_month, key, device):
    """[(X, tickers, dates)] per panel batch for one arm, cached."""
    cache = eg.CACHE_DIR / f"hero_tsne__{ev_month}__{key}.npz"
    batch_sets = _panel_batches(ev_month)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return [(z[f"X{b}"], z[f"t{b}"], z[f"d{b}"])
                for b in range(len(batch_sets))]
    import torch
    row = _manifest_row(key, ev_month)
    if row is None:
        raise SystemExit(f"no manifest row for {key} @ {ev_month}; "
                         f"MJ_NOCLAMP_MANIFEST={os.environ['MJ_NOCLAMP_MANIFEST']}")
    print(f"  {key}: {row['ckpt_dir']}")
    backbone = _load_manifest_encoder(row).to(device).eval()
    out = []
    try:
        for _, batches in batch_sets:
            fwd = eg.forward_cached(backbone, batches, device, cap=10**9)
            out.append((
                np.asarray(fwd["X"], dtype=np.float32),
                np.asarray([str(t) for t in fwd["tickers"]], dtype=object),
                np.asarray([str(d) for d in fwd["dates"]], dtype=object),
            ))
    finally:
        backbone.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    np.savez(cache, **{f"{c}{b}": v for b, trip in enumerate(out)
                       for c, v in zip("Xtd", trip)})
    return out


_BATCHES: dict[str, list] = {}


def _panel_batches(ev_month):
    """The month's info-token panels, exactly as fixed_panel_metrics builds
    them (the fixedRi cache; collected on a miss)."""
    if ev_month not in _BATCHES:
        from market_jepa.schemas import LocalMachineConfig
        from stable_finance.dataset import MarketSchedule
        machine = LocalMachineConfig()
        _, ev_start, ev_end = eg.eval_window_t_plus_n(ev_month, 0)
        _BATCHES[ev_month] = load_panel_batches(
            ev_month, ev_start, ev_end, P, S, machine,
            MarketSchedule(machine.holiday_csv), info=True)
    return _BATCHES[ev_month]


def panel_points(emb, bi, panel):
    """(X, tks, dts) for one panel: days where all P*S firms have a view."""
    X, tk, dt = emb[bi]
    in_panel = np.isin(tk, sorted(panel))
    keep = np.zeros(len(tk), dtype=bool)
    for d in sorted(set(dt)):
        sel = in_panel & (dt == d)
        if sel.sum() == P * S:
            keep |= sel
    order = np.lexsort((tk[keep], dt[keep]))
    return X[keep][order], tk[keep][order], dt[keep][order]


def firm_colors(panel):
    """{firm: color}, {industry: hue index}: industries in code order."""
    by_ind = {}
    for t, ff in sorted(panel.items()):
        by_ind.setdefault(ff, []).append(t)
    colors, hue = {}, {}
    for h, ff in enumerate(sorted(by_ind)):
        dark, light = SECTOR_SHADES[h % len(SECTOR_SHADES)]
        a, b = sorted(by_ind[ff])
        colors[a], colors[b] = dark, light
        hue[ff] = h
    return colors, hue


def day_numbers(dts):
    """Day of month, as printed: '2023-01-03' -> '3'."""
    return np.asarray([str(int(str(d)[-2:])) for d in dts], dtype=object)


def partner_pctile(X, tks, dts, days, panel):
    """Mean rank pctile of each view's partner same-day view among all
    other-firm views on ``days`` -- T1, restricted to those days."""
    m = np.isin(dts, days)
    X, tks, dts = X[m], tks[m], dts[m]
    D = ((X[:, None] - X[None]) ** 2).sum(-1)
    out = []
    for i in range(len(X)):
        cand = np.flatnonzero(tks != tks[i])
        tgt = [j for j in cand if dts[j] == dts[i]
               and panel[tks[j]] == panel[tks[i]]][0]
        out.append((D[i, cand] < D[i, tgt]).sum() / (len(cand) - 1))
    return float(np.mean(out))


def draw(ax, Z, tks, dts, colors, mode, panel=None, big=False, lines=True,
         industry=False):
    ms, fs = (110, 5.6) if big else (46, 3.6)
    dnum = day_numbers(dts)
    if mode == "shapes":
        # Day by SHAPE instead of a printed number (a handful of days only):
        # each view is its day's marker in its firm's colour; the grey lines
        # to each day's 2-D centroid and the centroid itself, grey, in the
        # same shape, go ON TOP.
        days = sorted(set(dts))
        if len(days) > len(DAY_MARKERS):
            raise ValueError(f"{len(days)} days, {len(DAY_MARKERS)} shapes")
        for d, mk in zip(days, DAY_MARKERS):
            m = dts == d
            ax.scatter(Z[m, 0], Z[m, 1], s=ms * 0.75, marker=mk,
                       c=[colors[t] for t in tks[m]], edgecolors="white",
                       linewidths=0.3, zorder=3)
        for d, mk in zip(days, DAY_MARKERS):
            m = dts == d
            c = Z[m].mean(0)
            for z in Z[m] if lines else ():
                ax.plot([z[0], c[0]], [z[1], c[1]], color=CENTROID_GREY,
                        lw=0.8 if big else 0.55, zorder=5)
            ax.scatter(*c, s=ms * 0.55, marker=mk, facecolor=CENTROID_GREY,
                       edgecolor="black", linewidth=0.6, zorder=6)
        if industry:
            # Each industry's 2-D centroid over all its views, a star in the
            # industry's dark shade.
            for ff in sorted(set(panel.values())):
                firms = sorted(t for t, f in panel.items() if f == ff)
                c = Z[np.isin(tks, firms)].mean(0)
                ax.scatter(*c, s=ms * 1.6, marker="*",
                           facecolor=colors[firms[0]], edgecolor="white",
                           linewidth=0.5, zorder=7)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        return
    if mode == "centroid":
        # The day structure alone: firm dots (with day numbers) first, then
        # each view's line to its day's 2-D centroid and the centroid's grey
        # diamond ON TOP of them, so the grey is what the eye follows.
        _dots(ax, Z, tks, dnum, colors, ms, fs)
        for d in sorted(set(dts)):
            m = dts == d
            c = Z[m].mean(0)
            for z in Z[m]:
                ax.plot([z[0], c[0]], [z[1], c[1]], color="#7a7a7a",
                        lw=0.8 if big else 0.55, zorder=5)
            ax.scatter(*c, s=26 if big else 14, marker="D",
                       facecolor="#7a7a7a", edgecolor="white",
                       linewidth=0.4, zorder=6)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        return
    if mode == "partner":
        # Grey spokes: each view to its day's 2-D centroid (the T2 picture),
        # with a small grey diamond at the centroid. Under the partner lines.
        for d in sorted(set(dts)):
            m = dts == d
            c = Z[m].mean(0)
            for z in Z[m]:
                ax.plot([z[0], c[0]], [z[1], c[1]], color="#9a9a9a",
                        lw=0.6 if big else 0.4, ls=(0, (2, 1.5)), zorder=0.5)
            ax.scatter(*c, s=22 if big else 10, marker="D",
                       facecolor="#9a9a9a", edgecolor="white",
                       linewidth=0.4, zorder=2)
        # One segment per (industry pair, day): a view to its partner's
        # same-day view -- the T1 target. Drawn in the industry's dark shade.
        for i in range(len(Z)):
            for j in range(i + 1, len(Z)):
                if (dts[i] == dts[j] and tks[i] != tks[j]
                        and panel[tks[i]] == panel[tks[j]]):
                    dark = colors[min(tks[i], tks[j])]
                    ax.plot([Z[i, 0], Z[j, 0]], [Z[i, 1], Z[j, 1]],
                            color=dark, lw=0.9 if big else 0.5, alpha=0.55,
                            zorder=1)
        mode = "labels"
    if mode == "spokes":
        for d in sorted(set(dts)):
            m = dts == d
            c = Z[m].mean(0)
            for z in Z[m]:
                ax.plot([z[0], c[0]], [z[1], c[1]], color="#b0b0b0",
                        lw=0.45, zorder=1)
        for f in sorted(set(tks)):
            m = tks == f
            ax.scatter(Z[m, 0], Z[m, 1], s=16, facecolor=colors[f],
                       edgecolor="white", linewidth=0.3, zorder=3)
        for d in sorted(set(dts)):
            c = Z[dts == d].mean(0)
            ax.text(c[0], c[1], dnum[dts == d][0], fontsize=4.6, color=INK,
                    ha="center", va="center", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              edgecolor="#b0b0b0", linewidth=0.35))
    else:
        _dots(ax, Z, tks, dnum, colors, ms, fs)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")


def _dots(ax, Z, tks, dnum, colors, ms, fs):
    """One marker per view in its firm's colour, day of month inside."""
    light = {c for pair in SECTOR_SHADES for c in pair[1:]}
    for i in range(len(Z)):
        col = colors[tks[i]]
        ax.scatter(*Z[i], s=ms, facecolor=col, edgecolor="white",
                   linewidth=0.3, zorder=3)
        ax.text(*Z[i], dnum[i], fontsize=fs, ha="center", va="center",
                color=INK if col in light else "white", zorder=4)


def score_panels(ev_month, embs):
    """[(batch, panel_idx_in_batch, panel, {key: {metric: pctile}})]."""
    rows = []
    for bi, (panels, _) in enumerate(_panel_batches(ev_month)):
        for pi, panel in enumerate(panels):
            sc = {}
            for key, _ in ARMS:
                X, tks, dts = panel_points(embs[key], bi, panel)
                m = panel_metrics(X, tks, dts, panel, S, P * S, n_perm=0)
                sc[key] = {k: (v[0] / v[1], v[3]) for k, v in m.items()}
            rows.append((bi, pi, panel, sc))
    return rows


def scan(device, out):
    """Score every (month, panel) of the reported table under both arms.

    Ranks panels by the Cross Stock, Same Industry arm's T1 and T2: the mean
    of their mean-rank percentiles (lower = cleaner), with the top-1 ratios
    alongside. Writes every row to ``out`` so a re-rank is free.
    """
    import json
    months = json.loads((FIXED_PANEL / "fixed_panel_P3S2_6mo.json")
                        .read_text())["months"]
    rows = []
    for ym in months:
        embs = {key: embed_month(ym, key, device) for key, _ in ARMS}
        for n, (_, _, panel, sc) in enumerate(score_panels(ym, embs)):
            rows.append({"month": ym, "panel": n, "firms": panel,
                         "scores": {key: {f"T{k}": list(v)
                                          for k, v in sc[key].items()}
                                    for key in sc}})
        print(f"{ym} scored", flush=True)
    out.write_text(json.dumps(rows, indent=1))
    k2 = ARMS[1][0]
    key = lambda r: (r["scores"][k2]["T1"][1] + r["scores"][k2]["T2"][1]) / 2
    rows.sort(key=key)
    print(f"\n{len(rows)} panels; top 15 by {k2} mean(T1, T2) pctile")
    print(f"{'month':<8}{'#':>2}  {'T1':>13}{'T2':>13}{'T3':>13}{'T4':>13}"
          f"   firms")
    for r in rows[:15]:
        sc = r["scores"][k2]
        print(f"{r['month']:<8}{r['panel']:>2}  " + "".join(
            f"{sc[t][0]:>6.2f}x/{sc[t][1]:>5.1%}" for t in ("T1", "T2", "T3", "T4"))
            + "   " + " ".join(sorted(r["firms"], key=r["firms"].get)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--month", default="2023-01", help="EVAL month")
    ap.add_argument("--panel", type=int, default=-1,
                    help="0-9 over both panel batches; -1 = largest contrast")
    ap.add_argument("--days", choices=["labels", "spokes", "partner", "both"],
                    default="both")
    ap.add_argument("--n-days", type=int, default=0,
                    help="keep only N CONSECUTIVE trading days (0 = all); "
                         "t-SNE is fit on those days alone")
    ap.add_argument("--window", type=int, default=-1,
                    help="which N-day block (0 = trading days 1..N, 1 = N+1..2N, "
                         "...); "
                         "-1 = the one where the k2ind partner is nearest")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--scan", action="store_true",
                    help="score every month x panel and rank; no figure")
    args = ap.parse_args()

    eg.apply_variant("mixed")
    import torch
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    if args.scan:
        return scan(device, HERE / "panel_scan.json")
    embs = {key: embed_month(args.month, key, device) for key, _ in ARMS}
    rows = score_panels(args.month, embs)

    rrc, k2 = ARMS[0][0], ARMS[1][0]
    print(f"\n{args.month}: per-panel ratio-over-chance / mean-rank pctile")
    print(f"{'#':>2}  {'arm':<15}" + "".join(f"{f'T{k}':>15}" for k in (1, 2, 3, 4)))
    contrast = []
    for n, (_, _, panel, sc) in enumerate(rows):
        for key in (rrc, k2):
            print(f"{n:>2}  {key:<15}" + "".join(
                f"{sc[key][k][0]:>8.2f}x/{sc[key][k][1]:>5.1%}"
                for k in (1, 2, 3, 4)))
        # pctile: lower is better. k2ind should win T1+T2, rrc should win T3.
        contrast.append(
            (sc[rrc][1][1] - sc[k2][1][1]) + (sc[rrc][2][1] - sc[k2][2][1])
            + (sc[k2][3][1] - sc[rrc][3][1]))
    pick = args.panel if args.panel >= 0 else int(np.argmax(contrast))
    bi, _, panel, _ = rows[pick]
    names = ff49_names()
    print(f"\npanel {pick} ({'pinned' if args.panel >= 0 else 'largest contrast'},"
          f" contrast {contrast[pick]:.3f}):")
    for t, ff in sorted(panel.items(), key=lambda x: (x[1], x[0])):
        print(f"  {t:<6} FF49 {ff:>2} {names.get(ff, '?')}")

    colors, hue = firm_colors(panel)
    full = {key: panel_points(embs[key], bi, panel) for key, _ in ARMS}
    days = sorted(set(full[rrc][2]))
    keep_days = days
    if args.n_days:
        # Non-overlapping blocks: trading days 1-5, 6-10, ... of the month.
        wins = [days[i:i + args.n_days]
                for i in range(0, len(days) - args.n_days + 1, args.n_days)]
        print(f"\n{args.n_days}-day windows: partner's same-day view, mean "
              "rank pctile among the window's other-firm views")
        wsc = []
        for w, wd in enumerate(wins):
            row = {key: partner_pctile(*full[key], wd, panel)
                   for key, _ in ARMS}
            wsc.append(row[k2])
            print(f"  {w:>2} {wd[0]}..{wd[-1]}  "
                  + "  ".join(f"{key} {v:5.1%}" for key, v in row.items()))
        win = args.window if args.window >= 0 else int(np.argmin(wsc))
        keep_days = wins[win]
        print(f"window {win} ({'pinned' if args.window >= 0 else 'best k2ind'}): "
              f"{keep_days[0]}..{keep_days[-1]}")
    pts = {}
    for key, _ in ARMS:
        X, tks, dts = full[key]
        m = np.isin(dts, keep_days)
        X, tks, dts = X[m], tks[m], dts[m]
        pts[key] = (eg.run_tsne(X, perplexity=args.perplexity, seed=args.seed),
                    tks, dts)
    n_days = len(keep_days)
    print(f"  {len(pts[rrc][1])} views = {n_days} days x {P * S} firms")

    modes = ["labels", "spokes"] if args.days == "both" else [args.days]
    for mode in modes:
        fig, axes = plt.subplots(1, 2, figsize=(eg.WIDTH_FULL, 3.35))
        fig.subplots_adjust(wspace=0.04, left=0.01, right=0.99,
                            top=0.93, bottom=0.17)
        for ax, (key, title) in zip(axes, ARMS):
            Z, tks, dts = pts[key]
            draw(ax, Z, tks, dts, colors, mode, panel,
                 big=len(Z) <= 60)
            ax.set_title(title, fontsize=9, pad=4)
            for s in ax.spines.values():
                s.set_color("#cccccc")
        handles = []
        for ff in sorted(hue):
            for t in sorted(t for t, f in panel.items() if f == ff):
                handles.append(plt.Line2D(
                    [], [], ls="", marker="o", ms=5, mfc=colors[t],
                    mec="white", mew=0.3, label=f"{t} ({names.get(ff, ff)})"))
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.5,
                   frameon=False, handletextpad=0.2, columnspacing=1.2,
                   bbox_to_anchor=(0.5, 0.0))
        day_note = {
            "labels": "number = day of month",
            "spokes": "spokes join each view to its day's centroid "
                      "(number = day of month)",
            "partner": "coloured: industry partner's same-day view; grey: "
                       "that day's centroid (number = day of month)",
        }[mode]
        span = (f"{n_days} trading days" if not args.n_days else
                f"{keep_days[0][5:]} to {keep_days[-1][5:]}")
        fig.text(0.5, 0.115, f"{args.month}, {span}; {day_note}",
                 ha="center", fontsize=6.2, color="#666666")
        wtag = f"_d{args.n_days}w{win}" if args.n_days else ""
        stem = HERE / f"tsne_panel_{args.month}_p{pick}{wtag}_{mode}"
        for pth in save_figure(fig, stem):
            print(f"Saved {pth}")
        plt.close(fig)


if __name__ == "__main__":
    main()
