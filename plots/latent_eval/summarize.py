# /// script
# requires-python = ">=3.10"
# ///
"""One readable summary of every latent-eval result set on disk.

The pipeline writes three JSONs per run, in two directories, keyed by a
``--tag``/``--out-suffix`` that differs per campaign:

    fixed_panel/fixed_panel_P3S2<tag>.json     organization tasks T1-T4
    factors/decode_loadings<tag>.json          factor-loading decode (R^2)
    factors/subspace_alignment<tag>.json       subspace alignment (rho, z)

Reading them by hand means three files, three shapes and no coverage
information. This walks whatever exists, prints a table per analysis, and
writes RESULTS.md beside it. It is READ-ONLY: no recomputation, so it is safe
to run against a campaign that is still in flight -- a missing file just shows
as a missing section, which is the point when you want to see how far a run
has actually got.

Run:
    uv run plots/latent_eval/summarize.py
    uv run plots/latent_eval/summarize.py --tag _sslic
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
FP, FS = HERE / "fixed_panel", HERE / "factors"
# Tags skipped by default. _timing/_infotest are scratch; _tsfmlayers is a
# different campaign (61 frozen-TSFM layer series) that would bury everything
# else in a shared listing -- ask for it explicitly with --tag _tsfmlayers.
SKIP_TAGS = {"_timing", "_infotest", "_debug", "_tsfmlayers"}

LABEL = {
    "byol_final": "BYOL", "cost_final": "CoST", "cpc_final": "CPC",
    "dino_final": "DINO", "ijepa_final": "I-JEPA", "mae_final": "MAE",
    "tfc_final": "TF-C", "timemae_final": "TimeMAE", "ts2vec_final": "TS2Vec",
    "pair_rrc_final": "LeJEPA crops", "pair_warp_final": "LeJEPA +time warp",
    "pair_k2_final": "LeJEPA cross-stock K=2",
    "pair_k2ind_final": "LeJEPA cross-stock same-ind",
    "sup_return_w8": "Supervised (return)", "sup_vol_w8": "Supervised (vol)",
    "sup_spread_w8": "Supervised (spread)", "sup_multi_w8": "Supervised (multi)",
    "random": "Random ViT", "random_vit_pooled": "Random ViT (5-seed mean)",
}
def lab(k: str) -> str:
    k = k.split("|")[0]
    return LABEL.get(k, k.replace("randvit_s", "Random ViT s"))


def discover() -> list[str]:
    tags = set()
    for p in glob.glob(str(FP / "fixed_panel_P3S2*.json")):
        m = re.match(r"fixed_panel_P3S2(.*)\.json", Path(p).name)
        if m: tags.add(m.group(1))
    for pat, rx in ((FS / "decode_loadings*.json", r"decode_loadings(.*)\.json"),
                    (FS / "subspace_alignment*.json", r"subspace_alignment(.*)\.json")):
        for p in glob.glob(str(pat)):
            m = re.match(rx, Path(p).name)
            if m: tags.add(m.group(1))
    return sorted(t for t in tags if t and t not in SKIP_TAGS)


RANDVIT_RE = re.compile(r"^randvit_s\d+$")


def pool_randvit(results: dict) -> dict:
    """Collapse ``randvit_s0..s4`` into ONE ``Random ViT`` entry.

    The five seeds are replicates of the same untrained architecture, not five
    models: listing them separately pads the table and, worse, makes the floor
    look like a spread of competitors that trained models can rank "between".
    Averaged PER MONTH across whichever seeds have that month, so a seed that
    is missing a month cannot shift the floor's level -- then the caller means
    over months as it does for every other row.
    """
    pooled, out = {}, {}
    for key, res in results.items():
        base = key.split("|")[0]
        if not RANDVIT_RE.match(base):
            out[key] = res
            continue
        suffix = key[len(base):]          # keeps the "|emb" variant tag
        pooled.setdefault(suffix, []).append(res)
    for suffix, group in pooled.items():
        merged = {}
        for field in {k for g in group for k in g}:
            cols = [g[field] for g in group if isinstance(g.get(field), list)]
            if not cols:
                continue
            n = max(len(c) for c in cols)
            avg = []
            for i in range(n):
                vals = [c[i] for c in cols
                        if i < len(c) and isinstance(c[i], (int, float))]
                avg.append(sum(vals) / len(vals) if vals else float("nan"))
            merged[field] = avg
        out[f"random_vit_pooled{suffix}"] = merged
    return out


def _load(p: Path):
    try: return json.loads(p.read_text())
    except Exception: return None


FAMILY = {
    "pair_rrc_final": "LeJEPA", "pair_warp_final": "LeJEPA",
    "pair_noise_final": "LeJEPA", "pair_k2_final": "LeJEPA",
    "pair_k2ind_final": "LeJEPA",
    "sup_return_w8": "Supervised", "sup_vol_w8": "Supervised",
    "sup_spread_w8": "Supervised", "sup_multi_w8": "Supervised",
    "byol_final": "SSL", "cost_final": "SSL", "cpc_final": "SSL",
    "dino_final": "SSL", "ijepa_final": "SSL", "mae_final": "SSL",
    "tfc_final": "SSL", "timemae_final": "SSL", "ts2vec_final": "SSL",
    "random": "Floor", "random_vit_pooled": "Floor",
}
def fam_of(k: str) -> str:
    return FAMILY.get(k.split("|")[0], "other")


def _merged_org(tags: list[str], out: list[str]) -> None:
    """T1-T4 with every family in ONE ranked table per task.

    The tags are separate runs, so this only merges when they are actually
    comparable: same panel geometry (P, S, n_panels) and the same chance rate
    per task. Chance here is a property of the PANEL, not the model, so two
    tags disagreeing on it means they scored different month sets -- averaging
    them into one ranking would make part of the gap a difference in panels.
    That is checked and refused loudly rather than silently merged.
    """
    loaded = {}
    for t in tags:
        j = _load(FP / f"fixed_panel_P3S2{t}.json")
        if j is not None:
            loaded[t] = j
    if not loaded:
        out.append("_(no organization-task results yet)_\n"); return
    geoms = {t: (j.get("P"), j.get("S"), j.get("n_panels")) for t, j in loaded.items()}
    if len(set(geoms.values())) > 1:
        out.append(f"**NOT MERGED** — panel geometry differs across runs: "
                   f"{geoms}. Showing them separately instead.\n")
        for t in tags:
            out.append(f"#### `{t}`\n"); organization(t, out)
        return
    names = next(iter(loaded.values())).get("metric_names", {})
    P, S, npan = next(iter(geoms.values()))
    out.append(f"Panels: P={P} S={S} n_panels={npan}. Rate = top-1 hit rate; "
               f"rank = mean rank of the target among the candidates (lower is "
               f"better); pctile = that rank normalized by pool size.\n")
    for mi in ("1", "2", "3", "4"):
        key = f"metric{mi}"
        rows, chances, seen = [], set(), set()
        for t, j in loaded.items():
            for k, v in j["models"].items():
                d = v.get(key) or {}
                if d.get("rate") is None or d["rate"] != d["rate"]:
                    continue
                base = k.split("|")[0]
                if base in seen:        # e.g. `random`, present in both runs
                    continue
                seen.add(base)
                chances.add(round(d.get("chance") or 0, 4))
                rows.append((d["rate"], d.get("mean_rank"), d.get("mean_pctile"),
                             lab(k), fam_of(k), d.get("chance")))
        if not rows:
            continue
        # CHANCE IS NOT ALWAYS A CONSTANT. T1/T3/T4 have a fixed 1/pool
        # chance, but T2's is estimated per model by day-label permutation, so
        # it legitimately varies row to row. An earlier version required one
        # shared chance and therefore refused to render T2 at all, reporting
        # eighteen "disagreeing" values that were simply eighteen models'
        # permutation baselines. Panel GEOMETRY is already checked above --
        # that is the real comparability guard -- so here just carry each
        # model's own chance into its row.
        rows.sort(reverse=True)
        spread = max(chances) - min(chances) if chances else 0.0
        note = ("" if spread < 0.002 else
                "  \n_Chance is per-model here (day-label permutation), so the "
                "`vs chance` column is each model against its own baseline._")
        out.append(f"\n**T{mi} — {names.get(mi, key)}**  "
                   f"(chance {min(chances):.1%}"
                   + (f"–{max(chances):.1%}" if spread >= 0.002 else "")
                   + f"){note}\n")
        out.append("| # | model | family | rate | vs chance | mean rank | rank % |")
        out.append("|---|---|---|---|---|---|---|")
        for i, (rate, mr, mp, name, f, ch) in enumerate(rows, 1):
            ok = lambda x: isinstance(x, (int, float)) and x == x
            ratio = f"{rate / ch:.2f}x" if ch else "--"
            rk = f"{mr:.2f}" if ok(mr) else "--"
            pc = f"{mp:.1%}" if ok(mp) else "--"
            out.append(f"| {i} | {name} | {f} | {rate:.1%} | {ratio} | "
                       f"{rk} | {pc} |")
        out.append("")


def _merged_factor(tags: list[str], out: list[str], kind: str) -> None:
    """Decode or alignment, all families in one ranked table."""
    rows, seen, months = [], set(), set()
    for t in tags:
        fn = (f"decode_loadings{t}.json" if kind == "decode"
              else f"subspace_alignment{t}.json")
        j = _load(FS / fn)
        if j is None:
            continue
        months.add(len(j.get("months", [])))
        for key, res in pool_randvit(j["results"]).items():
            base = key.split("|")[0]
            if base in seen:
                continue
            seen.add(base)
            if kind == "decode":
                v4 = []
                for tt in ("tot1", "tot2", "tot3", "tot4"):
                    v = [x for x in (res.get(tt) or []) if isinstance(x, (int, float))]
                    if v:
                        v4.append(st.fmean(v))
                r2 = [x for x in (res.get("r2_k") or []) if isinstance(x, (int, float))]
                rows.append((st.fmean(v4) if v4 else float("nan"),
                             st.fmean(r2) if r2 else float("nan"),
                             lab(key), fam_of(key)))
            else:
                rho = [x for x in (res.get("rhobar") or []) if isinstance(x, (int, float))]
                z = [x for x in (res.get("zperm") or []) if isinstance(x, (int, float))]
                if not rho:
                    continue
                rows.append((st.fmean(rho), st.fmean(z) if z else float("nan"),
                             lab(key), fam_of(key)))
    if not rows:
        out.append(f"_(no {kind} results yet)_\n"); return
    if len(months) > 1:
        out.append(f"NOTE: month counts differ across runs {sorted(months)} — "
                   f"rows are NOT over the same panel.\n")
    rows.sort(reverse=True)
    if kind == "decode":
        out.append("| # | model | family | top-4 factors (mean r) | factor-model R^2 (r) |")
        out.append("|---|---|---|---|---|")
        for i, (a, b, name, f) in enumerate(rows, 1):
            out.append(f"| {i} | {name} | {f} | **{a:.3f}** | "
                       + (f"{b:.3f} |" if b == b else "-- |"))
    else:
        out.append("| # | model | family | rho (fraction captured) | z (perm) |")
        out.append("|---|---|---|---|---|")
        for i, (a, b, name, f) in enumerate(rows, 1):
            out.append(f"| {i} | {name} | {f} | **{a:.3f}** | {b:.1f} |")
    out.append("")


def organization(tag: str, out: list[str]) -> None:
    """T1-T4, both readouts: top-1 rate AND mean rank.

    They answer different questions. The RATE is "how often is the right
    answer first", which saturates once a model is good and says nothing about
    how it fails. The MEAN RANK is where the target sits on average among the
    candidates, so a model that is rarely first but always second reads very
    differently from one that is sometimes first and otherwise last. pctile
    normalizes rank by the candidate count (0% = always first), which is the
    only way T1-T4 compare to each other -- they have different pool sizes.
    """
    j = _load(FP / f"fixed_panel_P3S2{tag}.json")
    if j is None:
        out.append(f"_(no fixed_panel_P3S2{tag}.json -- organization tasks not "
                   f"written)_\n")
        return
    names = j.get("metric_names", {})
    out.append(f"Panels: P={j.get('P')} S={j.get('S')} "
               f"n_panels={j.get('n_panels')}. "
               f"Rate = top-1 hit rate; rank = mean rank of the target among "
               f"the candidates (lower is better), pctile = that rank "
               f"normalized by pool size.\n")
    for mi in ("1", "2", "3", "4"):
        key = f"metric{mi}"
        rows = []
        for k, v in j["models"].items():
            d = v.get(key) or {}
            if d.get("rate") is None:
                continue
            rows.append((d["rate"], d.get("chance"), d.get("mean_rank"),
                         d.get("mean_pctile"), lab(k)))
        if not rows:
            continue
        # NaN sorts unpredictably and lands mid-table, which reads as a real
        # placing. A NaN here means the series had no scorable month at all
        # (its checkpoints did not exist when the stage ran), so push those to
        # the bottom and render them as "--" rather than a number.
        rows.sort(key=lambda r: (r[0] == r[0], r[0] if r[0] == r[0] else 0),
                  reverse=True)
        ch = rows[0][1]
        out.append(f"\n**T{mi} — {names.get(mi, key)}**  "
                   f"(chance rate {ch:.1%})\n")
        out.append("| # | model | rate | vs chance | mean rank | rank pctile |")
        out.append("|---|---|---|---|---|---|")
        for i, (rate, chance, mr, mp, name) in enumerate(rows, 1):
            if rate != rate:
                out.append(f"| — | {name} | -- | -- | -- | -- |  _(no scored "
                           f"month; checkpoints absent when this ran)_")
                continue
            ratio = rate / chance if chance else float("nan")
            ok = lambda x: isinstance(x, (int, float)) and x == x
            mrs = f"{mr:.2f}" if ok(mr) else "--"
            mps = f"{mp:.1%}" if ok(mp) else "--"
            out.append(f"| {i} | {name} | {rate:.1%} | {ratio:.2f}x | "
                       f"{mrs} | {mps} |")
        out.append("")


def decode(tag: str, out: list[str]) -> None:
    """Factor-loading decode, condensed to the two numbers worth reading.

    WHAT IS MEASURED: can a ridge read a stock's factor loadings out of its
    frozen embedding? Per (ticker, month), targets come from
    estimate_factors.py in standardized units; features are the month-mean
    full-day embedding. 5-fold CROSS-FIRM CV (folds grouped by gvkey so dual
    listings never straddle a fold), Ridge(alpha=100), score = OUT-OF-SAMPLE
    PEARSON r, averaged over months. NOT an R^2 -- an r of 0.5 is 25% of
    variance.

    Two columns, per the ask: ``top-4 factors`` is the mean over the loadings
    on total factors 1-4, and ``factor-model R^2`` is the r2_k target (the
    stock's own top-K_hat factor-model fit) -- the overall-loading readout.
    The per-factor tot1..tot4 and the continuous/jump split stay in the JSON.

    THE REFERENCE IS THE RANDOM-INIT FLOOR, not zero: an untrained ViT already
    decodes some loading structure on this data, so a row only means something
    against the Random ViT rows in the same table.
    """
    j = _load(FS / f"decode_loadings{tag}.json")
    if j is None:
        out.append(f"_(no decode_loadings{tag}.json yet)_\n")
        return
    months = j.get("months", [])
    TOPN = ["tot1", "tot2", "tot3", "tot4"]
    out.append(f"Out-of-sample Pearson r, 5-fold cross-firm CV, mean over "
               f"{len(months)} months. Higher is better; compare against the "
               f"Random ViT rows.\n")
    rows = []
    for key, res in pool_randvit(j["results"]).items():
        vals = []
        for t in TOPN:
            v = [x for x in (res.get(t) or []) if isinstance(x, (int, float))]
            if v:
                vals.append(st.fmean(v))
        r2 = [x for x in (res.get("r2_k") or []) if isinstance(x, (int, float))]
        top4 = st.fmean(vals) if vals else float("nan")
        rows.append((top4, st.fmean(r2) if r2 else float("nan"), lab(key)))
    rows.sort(reverse=True)
    out.append("| # | model | top-4 factors (mean r) | factor-model R^2 (r) |")
    out.append("|---|---|---|---|")
    for i, (top4, r2, name) in enumerate(rows, 1):
        f4 = f"{top4:.3f}" if top4 == top4 else "--"
        fr = f"{r2:.3f}" if r2 == r2 else "--"
        out.append(f"| {i} | {name} | **{f4}** | {fr} |")
    out.append("")


def alignment(tag: str, out: list[str]) -> None:
    """Subspace alignment, one number.

    WHAT IS MEASURED: how much of the factor-loading space the embedding
    actually spans. Per month, canonical correlations between the top-J PCs of
    the ticker-centroid embeddings and the estimated total loadings Lambda
    (both demeaned); ``rho`` = sum(rho^2)/K_hat in [0, 1], i.e. THE FRACTION OF
    THE LOADING SPACE CAPTURED. Calibration is 200 row permutations of Lambda,
    reported as a z-score -- so z says the overlap is not chance, and rho says
    how big it is. Read rho; z only guards it.
    """
    j = _load(FS / f"subspace_alignment{tag}.json")
    if j is None:
        out.append(f"_(no subspace_alignment{tag}.json yet)_\n")
        return
    months = j.get("months", [])
    out.append(f"Fraction of the loading space captured by the embedding "
               f"subspace (J={j.get('J')}), mean over {len(months)} months. "
               f"z = permutation z-score.\n")
    rows = []
    for key, res in pool_randvit(j["results"]).items():
        rho = [x for x in (res.get("rhobar") or []) if isinstance(x, (int, float))]
        z = [x for x in (res.get("zperm") or []) if isinstance(x, (int, float))]
        if not rho:
            continue
        rows.append((st.fmean(rho), st.fmean(z) if z else float("nan"),
                     lab(key), len(rho)))
    rows.sort(reverse=True)
    out.append("| # | model | rho (fraction captured) | z (perm) | n months |")
    out.append("|---|---|---|---|---|")
    for i, (r, z, name, n) in enumerate(rows, 1):
        out.append(f"| {i} | {name} | **{r:.3f}** | {z:.1f} | {n} |")
    out.append("")


ROSTER = {
    "LeJEPA": ["pair_rrc_final", "pair_warp_final", "pair_noise_final",
               "pair_k2_final", "pair_k2ind_final"],
    "Supervised": ["sup_return_w8", "sup_vol_w8", "sup_spread_w8",
                   "sup_multi_w8"],
    "SSL": ["byol_final", "cost_final", "cpc_final", "dino_final",
            "ijepa_final", "mae_final", "tfc_final", "timemae_final",
            "ts2vec_final"],
}


def pending_report(tags: list[str], out: list[str]) -> None:
    """What is absent, and what is present-but-unusable.

    Two different states, and conflating them wastes time: a series with NO
    entry has not been run, while one that is present with a NaN rate HAS run
    and produced nothing usable -- almost always because it was missing at
    least one month, since fixed_panel_metrics aggregates with np.mean and a
    single empty month NaNs that model's whole row.
    """
    org, dec, ali = {}, set(), set()
    for t in tags:
        j = _load(FP / f"fixed_panel_P3S2{t}.json")
        if j:
            for k, v in j["models"].items():
                r = (v.get("metric1") or {}).get("rate")
                org[k.split("|")[0]] = (r is not None and r == r)
        for fn, acc in ((f"decode_loadings{t}.json", dec),
                        (f"subspace_alignment{t}.json", ali)):
            jj = _load(FS / fn)
            if jj:
                acc.update(k.split("|")[0] for k in jj["results"])
    lines = []
    for fam, keys in ROSTER.items():
        for k in keys:
            miss = []
            if k not in org:
                miss.append("organization (absent)")
            elif not org[k]:
                miss.append("organization (ran, all-NaN — missing month/s)")
            if k not in dec:
                miss.append("decode")
            if k not in ali:
                miss.append("alignment")
            if miss:
                lines.append(f"| {lab(k)} | {fam} | {', '.join(miss)} |")
    out.append("## What is still missing\n")
    if not lines:
        out.append("Nothing — every series in the roster has all three "
                   "analyses.\n")
        return
    out.append("These will populate as their runs land; a row here means the "
               "result does not exist yet, not that it is zero.\n")
    out.append("| model | family | missing |")
    out.append("|---|---|---|")
    out.extend(lines)
    out.append("")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default=None, help="only this tag (e.g. _sslic)")
    p.add_argument("--per-tag", action="store_true",
                   help="one section per campaign instead of one merged table")
    p.add_argument("--out", default=str(HERE / "RESULTS.md"))
    a = p.parse_args()
    tags = [a.tag] if a.tag else discover()
    if not tags:
        print("no result sets found")
        return 1
    out: list[str] = ["# Latent-eval results", "",
                      "Generated by `plots/latent_eval/summarize.py` "
                      "(read-only; re-run any time).", ""]
    if a.per_tag or a.tag:
        for tag in tags:
            out.append(f"## `{tag}`\n")
            out.append("### Organization tasks (T1-T4)\n"); organization(tag, out)
            out.append("### Factor-loading decode\n"); decode(tag, out)
            out.append("### Subspace alignment\n"); alignment(tag, out)
    else:
        out.append(f"Campaigns merged: {', '.join('`' + t + '`' for t in tags)}. "
                   f"`--per-tag` splits them.\n")
        out.append("## Organization tasks (T1-T4)\n")
        _merged_org(tags, out)
        out.append("## Factor-loading decode\n")
        out.append("Out-of-sample Pearson r, 5-fold cross-firm CV, mean over "
                   "months. Compare against the Random ViT floor.\n")
        _merged_factor(tags, out, "decode")
        out.append("## Subspace alignment\n")
        out.append("Fraction of the factor-loading space captured by the "
                   "embedding subspace (J=10). z = permutation z-score.\n")
        _merged_factor(tags, out, "alignment")
        pending_report(tags, out)
    text = "\n".join(out)
    Path(a.out).write_text(text)
    print(text)
    print(f"\n[wrote {a.out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
