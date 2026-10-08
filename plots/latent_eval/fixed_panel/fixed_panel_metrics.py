"""Fixed monthly panel metrics — the industry/day/partner table.

Per eval month (train+1 of each of the 32 canonical sweep months): N panels of
``--n_portfolios`` random FF49 industries x ``--n_stocks`` random
FULL-PRESENCE member firms, each panel held fixed across every trading
day of the month, one global view per (firm, day). Panels are drawn in
batches (seed_batch, R) so extra replicates can be added later without
recollecting earlier ones; windows cache per batch under cache/.

Metrics per panel (rates vs chance, t across month-means):
  1. a point's NN among other firms' views (month-pooled) is the SAME-DAY
     view of its same-industry partner (chance ~ 1/pool)
  2. the closest of the ~21 day-centroids is the point's own day; the
     focal FIRM's views are excluded from EVERY centroid, so all
     candidates are unbiased 5-view centroids of the other firms;
     chance 1/n_days exactly -- the panel is full-presence, so excluding the
focal firm leaves every day-centroid the same size and the candidates
exchangeable (a day-label permutation null is BIASED here; see the
metric-2 block)
  3. the closest of the P*S firm-centroids is the point's own firm; the
     focal DAY's views are excluded from EVERY centroid (the mirror of
     2), so all candidates are unbiased ~20-day centroids and no
     same-day co-movement shortcut survives (chance 1/(P*S) = 16.7%)
  4. the firm's month centroid (~21 daily views averaged) is nearest to
     its partner firm's centroid among the other P*S-1 firm centroids
     (chance (S-1)/(P*S-1) = 20% at 3x2)

Writes fixed_panel_P{P}S{S}.json and prints the per-metric tables.

Run:
    uv run plots/latent_eval/fixed_panel/fixed_panel_metrics.py
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch

import panel_lib as eg
from stable_finance.dataset import MarketSchedule
from market_jepa.modeling import TransformerConfig, create_backbone
from market_jepa.schemas import (
    LocalMachineConfig,
    machine_from_env,
    TransformerBackboneConfig,
    TransformerInnerConfig,
)
from market_jepa.eval.checkpoints import load_encoder
from market_jepa.training.streaming_dataset import ensure_ticker_date_sidecar

from industry_nn_sweep import (
    INDUSTRY_MAP, MODEL_ORDER, MODEL_SPECS, MONTHS, month_dates_suffix,
    resolve_run,
)

# TS-SSL specs carry "manifest_series": ckpts resolve from the noclamp
# manifest by (series_key, eval month) and load mode-aware — a TS-SSL model's
# representation is its own encode(), not its backbone's pooled output.
_NOCLAMP_ROWS: dict | None = None


def _manifest_row(series_key: str, ev_month: str):
    global _NOCLAMP_ROWS
    if _NOCLAMP_ROWS is None:
        import json as _json
        import os as _os
        # MJ_NOCLAMP_MANIFEST lets a cluster job point manifest resolution at
        # a job-local file whose ckpt_dir paths are node-local — the final32
        # bundle writes one for the checkpoints it just trained. Unset, the
        # repo manifest is read exactly as before.
        #
        # SEVERAL files may be named, os.pathsep-separated, and are merged in
        # order (later wins on a repeated (series, month)). The reported
        # 18-encoder panel needs this: the pair_*/sup_*_w8 rows and the
        # ssl *_final rows live in two different manifests, which is why it
        # had to be produced as two runs under two tags. run_eval.sh builds
        # the list; a single path behaves exactly as before.
        man_paths = _os.environ.get(
            "MJ_NOCLAMP_MANIFEST",
            str(eg._REPO_ROOT / "plots" / "metrics" / "noclamp_manifest.json"),
        ).split(_os.pathsep)
        _NOCLAMP_ROWS = {}
        for man_path in [p for p in man_paths if p]:
            man = _json.loads(open(man_path).read())
            _NOCLAMP_ROWS.update({(r["series_key"], r["eval_month"]): r
                                  for r in man["ckpts"]})
    return _NOCLAMP_ROWS.get((series_key, ev_month))


def _load_manifest_encoder(row) -> torch.nn.Module:
    """The manifest row's encoder, READ AT THE LATENT SUITE'S POOL.

    Every key in MODEL_ORDER is manifest-resolved, so this is the loader the
    whole reported roster goes through -- it is not a side path. Without the
    pin the supervised arms come back at pool="last" (their training readout)
    beside mean-pooled SSL, which is the confound panel_lib.LATENT_POOL exists
    to remove.
    """
    return load_encoder(row["ckpt_dir"], row["project"], pool=eg.LATENT_POOL)


# ── Frozen TSFMs: every layer from one forward per panel batch ────────────────
#
# A TSFM has no checkpoint — (family, layer) determines it — so it never goes
# through the manifest. And its layers must not be swept one model at a time:
# the panel is the same rows for every layer, and compute_features_multi
# captures all of them in one pass, so the 61-layer sweep costs four forwards
# per batch set instead of 61. The bank holds ONE family's forward at a time,
# keyed by (batch set, family), which is why --models should list a family's
# layers consecutively (they are, coming from industry_nn_sweep.TSFM_KEYS).
_TSFM_BANK: dict = {"key": None, "fwd": None}


_SUP_BANK: dict = {"key": None, "fwd": None}


def _sup_layer_forward(spec, batches, device, batch_key, ev_month):
    """Every depth of one supervised trunk from ONE pass over the panel.

    The encoder for an EVAL month is the one trained on the month BEFORE it,
    which is the same pairing the IC sweep and post_train_ic_eval use.
    """
    from market_jepa.eval.checkpoints import load_supervised, prev_month

    family = spec["sup_family"]
    key = (batch_key, family)
    if _SUP_BANK["key"] != key:
        _SUP_BANK["key"], _SUP_BANK["fwd"] = None, None   # free first
        model = load_supervised(family, prev_month(ev_month), device,
                                pool=eg.LATENT_POOL)
        layers = list(range(1, model.n_layers + 1))
        try:
            _SUP_BANK["fwd"] = eg.forward_cached_multi(
                model, batches, device, 10**9, layers)
        finally:
            model.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        _SUP_BANK["key"] = key
    bank = _SUP_BANK["fwd"]
    return {"X": bank["X"][spec["sup_layer"]], "tickers": bank["tickers"],
            "dates": bank["dates"], "return_900": bank["return_900"]}


def _tsfm_layer_forward(spec, batches, device, batch_key):
    from market_jepa.modeling.modes.pretrained_tsfm import (
        _FAMILIES, PretrainedTSFM, resolve_family)

    family = spec["tsfm_family"]
    key = (batch_key, family)
    if _TSFM_BANK["key"] != key:
        _TSFM_BANK["key"], _TSFM_BANK["fwd"] = None, None   # free first
        base, channel_pool = resolve_family(family)
        model = PretrainedTSFM(backbone=None, model=base,
                               channels=list(range(9)),
                               channel_pool=channel_pool).to(device).eval()
        layers = list(range(_FAMILIES[base]["n_layers"] + 1))
        try:
            _TSFM_BANK["fwd"] = eg.forward_cached_multi(
                model, batches, device, 10**9, layers)
        finally:
            model.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        _TSFM_BANK["key"] = key
    bank = _TSFM_BANK["fwd"]
    return {"X": bank["X"][spec["tsfm_layer"]], "tickers": bank["tickers"],
            "dates": bank["dates"], "return_900": bank["return_900"]}


def _make_random_vit() -> torch.nn.Module:
    """Random-init ViT — the same recipe as the dAUC baseline models
    (mass_eval_world_model_noclamp._make_transformer)."""
    bb = TransformerBackboneConfig()
    inner = TransformerInnerConfig()
    return create_backbone(
        backbone_type="transformer",
        n_features=9,
        d_embedding=bb.d_embedding,
        # MEAN, matching how every trained encoder in this cache is read
        # (panel_lib.LATENT_POOL). A floor read at a different token is not a
        # floor for these numbers. Explicit because the schema default is now
        # None -- pool is resolved per mode -- so `bb.pool` silently means
        # "cls" here rather than "whatever the models use".
        pool="mean",
        config=TransformerConfig(
            hidden_size=inner.hidden_size,
            num_hidden_layers=inner.num_hidden_layers,
            num_attention_heads=inner.num_attention_heads,
            intermediate_size=inner.intermediate_size,
            patch_size=inner.patch_size,
            layer_norm_eps=inner.layer_norm_eps,
            drop_path_rate=inner.drop_path_rate,
        ),
    )

# (seed_batch, R, cache_prefix): batch 0 = the original 3 panels/month,
# batch 1 = the 7-panel extension. Add (2, R, "fixedR3") etc. to grow N.
BATCHES = [(0, 3, "fixedR"), (1, 7, "fixedR2")]

METRIC_NAMES = {
    1: "pooled NN = partner's same-day view",
    2: "own day-centroid closest (focal firm excluded from all centroids)",
    3: "own firm-centroid closest (focal day excluded from all centroids)",
    4: "firm centroid NN = partner (of other P*S-1)",
}
ALL_METRICS = tuple(METRIC_NAMES)


def pick_panels(ev_month, rng, r, P, S, machine):
    """r panels of {ticker -> ff49}; industries/firms disjoint within a
    batch (across batches independence is enough)."""
    ind = pd.read_parquet(INDUSTRY_MAP)
    ind = ind[ind.month == ev_month]
    tick2ff = dict(zip(ind.ticker, ind.ff49.astype(int)))
    y, m = ev_month.split("-")
    meta = ensure_ticker_date_sidecar(f"{machine.mosaic_dir}/{y}/{m}")
    n_days = len(set(meta["dates"]))
    counts = Counter(meta["tickers"])
    full = {t for t, c in counts.items() if c == n_days}
    pools = defaultdict(list)
    for t in sorted(full):
        if t in tick2ff:
            pools[tick2ff[t]].append(t)
    eligible = sorted(ff for ff, ts in pools.items() if len(ts) >= S)
    chosen = rng.choice(eligible, size=P * r, replace=False)
    panels = []
    for i in range(r):
        panel = {}
        for ff in chosen[i * P:(i + 1) * P]:
            for t in rng.choice(pools[int(ff)], size=S, replace=False):
                panel[str(t)] = int(ff)
        panels.append(panel)
    return panels


def day_centroid_hits(X, firms_arr, dts, labels=None):
    """Fraction of points whose nearest day-centroid is their own day;
    own-firm views excluded from every centroid (=> own-day is LOO)."""
    labels = dts if labels is None else labels
    days = np.unique(labels)
    sums = np.stack([X[labels == d].sum(0) for d in days])
    cnts = np.array([(labels == d).sum() for d in days], float)
    pos = {d: i for i, d in enumerate(days)}
    hits = 0
    for p in range(len(X)):
        own = firms_arr == firms_arr[p]
        adj_s, adj_c = sums.copy(), cnts.copy()
        for d in np.unique(labels[own]):
            i = pos[d]
            m = own & (labels == d)
            adj_s[i] -= X[m].sum(0)
            adj_c[i] -= m.sum()
        valid = adj_c > 0
        cent = adj_s[valid] / adj_c[valid, None]
        d2 = ((cent - X[p]) ** 2).sum(1)
        hits += int(np.flatnonzero(valid)[np.argmin(d2)] == pos[labels[p]])
    return hits / len(X)


def firm_centroid_scores(X, firms_arr, dts):
    """Nearest firm-centroid = own firm, with ALL of the focal point's DAY
    dropped from EVERY centroid (the mirror of day_centroid_hits).

    Dropping the whole day, not just the focal view, keeps the own-firm
    centroid leave-one-out AND removes the same-day co-movement shortcut
    the other firms' centroids would otherwise get. All P*S candidates
    are then ~20-day centroids over the same days, so chance is 1/(P*S).
    Same task as firm_centroid_id.py, which runs it on the cached
    full-day embeddings instead of these sampled windows.

    Returns (hit rate, ranks of the own-firm centroid, n candidates).
    """
    firms = np.unique(firms_arr)
    sums = np.stack([X[firms_arr == f].sum(0) for f in firms])
    cnts = np.array([(firms_arr == f).sum() for f in firms], float)
    pos = {f: i for i, f in enumerate(firms)}
    hits, ranks, ns = 0, [], []
    for p in range(len(X)):
        day = dts == dts[p]
        adj_s, adj_c = sums.copy(), cnts.copy()
        for f in np.unique(firms_arr[day]):
            i = pos[f]
            m = day & (firms_arr == f)
            adj_s[i] -= X[m].sum(0)
            adj_c[i] -= m.sum()
        valid = adj_c > 0
        cent = adj_s[valid] / adj_c[valid, None]
        d2 = ((cent - X[p]) ** 2).sum(1)
        own = np.flatnonzero(np.flatnonzero(valid) == pos[firms_arr[p]])
        if len(own) != 1:
            continue
        hits += int(np.argmin(d2) == own[0])
        ranks.append(1 + int((d2 < d2[own[0]]).sum()))
        ns.append(int(valid.sum()))
    return hits / len(X), ranks, ns


def panel_metrics(X, tks, dts, ind_of_firm, S, PS, n_perm,
                  metrics=ALL_METRICS):
    """{metric: (rate, chance, mean_rank, mean_pctile, n_candidates)} for
    one panel's kept points. rate/chance are the top-1 numbers; mean_rank
    is the graded version (rank of the target among the candidates,
    1 = nearest, chance mean = (n+1)/2) with mean_pctile = (rank-1)/(n-1)
    (chance 0.5) for pool-size-independent aggregation. `metrics` selects
    which of the four to compute (the rest are simply absent from out)."""
    inds = np.asarray([ind_of_firm[t] for t in tks])
    same_firm = tks[:, None] == tks[None, :]
    pool = ~same_firm
    out = {}

    if 1 in metrics:
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        d2[same_firm] = np.inf
        nn = d2.argmin(1)
        target = (dts[:, None] == dts[None, :]) & (inds[:, None] == inds[None, :]) & pool
        ranks, ns = [], []
        for i in range(len(X)):
            tgt = np.flatnonzero(target[i])
            if len(tgt) != 1:
                continue
            cand = pool[i]
            ranks.append(1 + int((d2[i, cand] < d2[i, tgt[0]]).sum()))
            ns.append(int(cand.sum()))
        out[1] = (
            float(target[np.arange(len(X)), nn].mean()),
            float((target.sum(1) / pool.sum(1)).mean()),
            float(np.mean(ranks)),
            float(np.mean([(r - 1) / (n - 1) for r, n in zip(ranks, ns)])),
            float(np.mean(ns)),
        )

    if 2 in metrics:
        # CHANCE IS ANALYTIC, 1/n_days -- the permutation null was REMOVED
        # 2026-08-30 because it is biased for this panel, not merely redundant.
        #
        # pick_panels draws FULL-PRESENCE firms: every member trades every day,
        # so a panel is exactly P*S views per day for all n_days. Under the
        # true labels, excluding the focal firm removes exactly ONE view from
        # EVERY day-centroid, so all candidates are size-(P*S-1) centroids over
        # the same other firms -- perfectly exchangeable, chance exactly
        # 1/n_days, by the same argument the T3 block gives for 1/(P*S).
        #
        # Shuffling the day labels breaks that: the focal firm's n_days views
        # scatter multinomially, so some groups lose two views and some lose
        # none, and the candidates stop being exchangeable. The null therefore
        # measured an asymmetry the real configuration does not have, modulated
        # by each model's geometry -- which is why it came out model-dependent
        # (4.1%-5.6% against an analytic 4.8%) while correlating with nothing
        # (r = -0.06 against T3's firm-identity rate).
        #
        # Dropping it also removes n_perm (100) extra O(N^2) passes per panel.
        rate2 = day_centroid_hits(X, tks, dts)
        # rank of the own-day centroid (own-firm views excluded everywhere)
        days = np.unique(dts)
        sums = np.stack([X[dts == d].sum(0) for d in days])
        cnts = np.array([(dts == d).sum() for d in days], float)
        pos = {d: i for i, d in enumerate(days)}
        ranks, ns = [], []
        for i in range(len(X)):
            own = tks == tks[i]
            adj_s, adj_c = sums.copy(), cnts.copy()
            for d in np.unique(dts[own]):
                k = pos[d]
                m = own & (dts == d)
                adj_s[k] -= X[m].sum(0)
                adj_c[k] -= m.sum()
            valid = adj_c > 0
            cent = adj_s[valid] / adj_c[valid, None]
            dd = ((cent - X[i]) ** 2).sum(1)
            own_idx = np.flatnonzero(np.flatnonzero(valid) == pos[dts[i]])
            if len(own_idx) != 1:
                continue
            ranks.append(1 + int((dd < dd[own_idx[0]]).sum()))
            ns.append(int(valid.sum()))
        out[2] = (
            rate2, float(np.mean([1.0 / n for n in ns])) if ns else float("nan"),
            float(np.mean(ranks)),
            float(np.mean([(r - 1) / (n - 1) for r, n in zip(ranks, ns)])),
            float(np.mean(ns)),
        )

    if 3 in metrics:
        # Own-firm centroid, focal day dropped everywhere: every candidate
        # is the same set of other days, so chance is exactly 1/(P*S) and
        # no permutation null is needed.
        rate3, ranks, ns = firm_centroid_scores(X, tks, dts)
        out[3] = (
            rate3, 1.0 / PS,
            float(np.mean(ranks)),
            float(np.mean([(r - 1) / (n - 1) for r, n in zip(ranks, ns)])),
            float(np.mean(ns)),
        )

    if 4 in metrics:
        firms = sorted(set(tks))
        F = np.stack([X[tks == f].mean(0) for f in firms])
        Df = ((F[:, None, :] - F[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(Df, np.inf)
        nnf = Df.argmin(1)
        hits4, ranks = [], []
        for i, f in enumerate(firms):
            hits4.append(ind_of_firm[firms[nnf[i]]] == ind_of_firm[f])
            partner = [
                j for j, g in enumerate(firms)
                if g != f and ind_of_firm[g] == ind_of_firm[f]
            ]
            if len(partner) == 1:
                ranks.append(1 + int(
                    (Df[i, [j for j in range(len(firms)) if j != i and j != partner[0]]]
                     < Df[i, partner[0]]).sum()
                ))
        out[4] = (
            float(np.mean(hits4)), (S - 1) / (PS - 1),
            float(np.mean(ranks)),
            float(np.mean([(r - 1) / (PS - 2) for r in ranks])),
            float(PS - 1),
        )
    return out


def load_panel_batches(ev_month, ev_start, ev_end, P, S, machine, schedule,
                       info=False):
    """[(panels, batches)] per configured batch, collecting on first use.

    ``info`` collects the windows WITH the information-token channels, for
    encoders trained with the token (every LeJEPA pairing, every supervised
    head). It gets its own cache namespace (``<prefix>i__``) so the info-less
    panels collected before 2026-08-29 stay valid and are not silently served
    to a model that needs the wider input -- the cache key carries no channel
    count, so sharing one namespace would do exactly that.

    THE TWO PANELS ARE THE SAME WINDOWS. ``pick_panels`` draws from a
    deterministic rng seeded on (batch, year, month) and the collection seed is
    fixed, so info only adds trailing constant columns; panel_lib._match_width
    then lets either kind of encoder read either panel.
    """
    out = []
    for seed_batch, r, prefix in BATCHES:
        prefix = f"{prefix}i" if info else prefix
        cache = eg.CACHE_DIR / f"{prefix}__{ev_month}__seed0__P{P}S{S}R{r}.pkl"
        if cache.exists():
            with open(cache, "rb") as f:
                blob = pickle.load(f)
            out.append((blob["panels"], blob["batches"]))
            continue
        y, m = ev_month.split("-")
        rng = np.random.default_rng([seed_batch, int(y), int(m)])
        panels = pick_panels(ev_month, rng, r, P, S, machine)
        union = sorted({t for p_ in panels for t in p_})
        batches, _ = eg.collect_curated_windows(
            ev_start, ev_end, target_tickers=union,
            samples_per_ticker=31, machine=machine, schedule=schedule, seed=42,
            info=info,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump({"panels": panels, "batches": batches}, f)
        out.append((panels, batches))
    return out


def _repool(old_cell: dict, new_cell: dict) -> dict:
    """One T-metric cell re-pooled over the union of two runs' months.

    The four per-month series are unioned (the NEW run wins a month both
    carry, so re-running a month refreshes rather than double-counts) and
    every aggregate is recomputed from the union. Averaging the two cells'
    summary numbers -- or averaging their t values -- would be wrong and
    would not even be close.
    """
    series = ("month_rates", "month_chances", "month_ranks", "month_pctiles")
    merged = {f: {**old_cell.get(f, {}), **new_cell.get(f, {})} for f in series}
    months = sorted(merged["month_rates"])
    # A month that is not complete across all four series cannot be pooled.
    months = [m for m in months if all(m in merged[f] for f in series)]
    # A SKIPPED MODEL STILL WRITES ITS MONTH. When no manifest row resolves for
    # (model, eval month) -- the six-month wave fills arm by arm, so an arm can
    # be short a month for days -- stage 1 records that month as NaN rather
    # than leaving it out, and a plain mean over the union would turn ONE
    # missing checkpoint into an all-NaN row for the arm. A NaN month is not an
    # observation: drop it, so an arm pools over the months it actually has and
    # n_months counts those.
    months = [m for m in months
              if not any(np.isnan(float(merged[f][m])) for f in series)]
    if not months:
        return {"rate": float("nan"), "chance": float("nan"),
                "t": float("nan"), "mean_rank": float("nan"),
                "mean_pctile": float("nan"), "rank_t": float("nan"),
                "n_candidates": new_cell["n_candidates"],
                "month_rates": {}, "month_ranks": {},
                "month_chances": {}, "month_pctiles": {}, "n_months": 0}
    rates = np.array([merged["month_rates"][m] for m in months], dtype=float)
    chances = np.array([merged["month_chances"][m] for m in months], dtype=float)
    ranks = np.array([merged["month_ranks"][m] for m in months], dtype=float)
    pcts = np.array([merged["month_pctiles"][m] for m in months], dtype=float)

    def _t(ex):
        # One month has no spread to test; NaN is what a single-month run
        # already writes, so the merged file says the same thing it would.
        if len(ex) < 2:
            return float("nan")
        return float(ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex))))

    return {
        "rate": float(rates.mean()),
        "chance": float(chances.mean()),
        "t": _t(rates - chances),
        "mean_rank": float(ranks.mean()),
        "mean_pctile": float(pcts.mean()),
        "rank_t": _t(pcts - 0.5),
        # NOT fixed geometry, and NOT the new cell's. T1 and T2 pool a whole
        # panel-month, so the candidate count is a MEAN over the months pooled
        # here -- copying the incoming cell's left an arm topped up on 8 of its
        # 31 months claiming the 8-month mean, which then read as two arms
        # scored on different panels and tripped fixed_panel_table.py's guard.
        # Recomputed from the union like every other aggregate; for the
        # fixed-pool metrics (T3, T4) this returns their constant unchanged.
        "n_candidates": float((1.0 / chances).mean()),
        "month_rates": {m: merged["month_rates"][m] for m in months},
        "month_ranks": {m: merged["month_ranks"][m] for m in months},
        "month_chances": {m: merged["month_chances"][m] for m in months},
        "month_pctiles": {m: merged["month_pctiles"][m] for m in months},
        "n_months": len(months),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_portfolios", type=int, default=3)
    p.add_argument("--n_stocks", type=int, default=2)
    p.add_argument("--n_perm", type=int, default=100)
    p.add_argument("--models", default=",".join(MODEL_ORDER))
    p.add_argument("--metrics", default=",".join(str(k) for k in ALL_METRICS),
                   help="which metrics to compute; the JSON is merged "
                        "per-metric, so a subset backfills one task")
    p.add_argument("--months", nargs="*", default=None,
                   help="EVAL months to run (default: train+1 of each entry "
                        "in industry_nn_sweep.MONTHS). Pass the optimization "
                        "set's eval months to tune a layer off-panel.")
    p.add_argument("--out-suffix", default="",
                   help="appended to fixed_panel_P{P}S{S}<suffix>.json, so a "
                        "run on a different month panel does not merge into "
                        "the reported one")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    P, S = args.n_portfolios, args.n_stocks
    models = args.models.split(",")
    metrics = tuple(int(k) for k in args.metrics.split(","))
    eg.apply_variant("mixed")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    machine = machine_from_env(LocalMachineConfig)
    schedule = MarketSchedule(machine.holiday_csv)

    # res[model][metric] = per month: list over panels of (rate, chance)
    res = {m: {k: [] for k in metrics} for m in models}
    # MONTHS are TRAINING months and the panel is built on train+1. --months
    # names the EVAL months directly, because a frozen TSFM has no training
    # month for the convention to key off.
    if args.months:
        month_panel = [(None, ev) for ev in args.months]
    else:
        month_panel = [(ym, eg.eval_window_t_plus_n(ym, 1)[0]) for ym in MONTHS]
    label_months = [ev for _, ev in month_panel]
    for ym, ev_month in month_panel:
        # n=0 on the eval month itself is the same window n=1 on its
        # training month produces — pure month math either way.
        _, ev_start, ev_end = eg.eval_window_t_plus_n(
            ev_month if ym is None else ym, 0 if ym is None else 1)
        # Two panel kinds, both lazy: most runs need only one. Keyed by
        # whether the encoder carries an information token.
        _panel_cache: dict[bool, list] = {}

        def panels_for(info: bool):
            if info not in _panel_cache:
                _panel_cache[info] = load_panel_batches(
                    ev_month, ev_start, ev_end, P, S, machine, schedule,
                    info=info,
                )
            return _panel_cache[info]

        batch_sets = panels_for(False)
        sfx = month_dates_suffix(ym) if ym is not None else None
        month_vals = {k: {j: [] for j in metrics} for k in models}
        for bi, (panels, batches) in enumerate(batch_sets):
            for key in models:
                spec = MODEL_SPECS[key]
                if key == "random":
                    # The dAUC-baseline model: random-init ViT, same recipe
                    # as the baseline pipeline's seed models. Fresh init per
                    # month (seed = YYYYMM) — like the trained rows, each
                    # month gets its own model, not one frozen encoder.
                    torch.manual_seed(int(ev_month.replace("-", "")))
                    backbone = _make_random_vit().to(device).eval()
                elif "sup_family" in spec:
                    fwd = _sup_layer_forward(spec, batches, device,
                                             (ev_month, bi), ev_month)
                    backbone = None
                elif "tsfm_family" in spec:
                    fwd = _tsfm_layer_forward(spec, batches, device,
                                              (ev_month, bi))
                    backbone = None
                elif "manifest_series" in spec:
                    row = _manifest_row(spec["manifest_series"], ev_month)
                    if row is None:
                        # Loud: a missing manifest row used to drop a model
                        # from the table with no trace, and the reported
                        # encoders live across two manifests, so half the
                        # default model set can vanish this way.
                        print(f"  {ev_month}: no manifest row for {key} "
                              f"({spec['manifest_series']}) — skipped; check "
                              "MJ_NOCLAMP_MANIFEST")
                        continue
                    backbone = _load_manifest_encoder(row).to(device).eval()
                else:
                    # Only a glob that actually interpolates {dates} needs a
                    # training month. The fd_* entries pin ONE checkpoint
                    # trained on 2018-2022, so their glob is date-free and
                    # resolves under --months like a manifest key — this guard
                    # used to reject them anyway and blocked the whole
                    # fullday-readout eval.
                    pglob = spec["project_glob"]
                    if sfx is None and "{dates}" in pglob:
                        raise ValueError(
                            f"{key} resolves a checkpoint by TRAINING month, "
                            "which --months does not supply; run it off "
                            "industry_nn_sweep.MONTHS instead")
                    run_dir = resolve_run(
                        pglob.format(dates=sfx or ""), spec["run_name"],
                    )
                    if run_dir is None:
                        continue
                    backbone = eg._load_backbone(
                        run_dir, run_dir, run_dir.parent.name,
                        pool=eg.LATENT_POOL,
                    ).to(device).eval()
                if backbone is not None:
                    # An information-token encoder needs the wider window; the
                    # panels are the same draws either way (see
                    # load_panel_batches), so this changes the INPUT WIDTH and
                    # nothing about which firms/days are scored.
                    # DECIDED ON THE INPUT WIDTH, not on n_info_channels.
                    # A backbone rebuilt from a state dict gets n_features off
                    # the weights (20 = 9 data + 11 info) but NOT the
                    # n_info_channels kwarg, which stays 0 -- so the old guard
                    # read False for exactly the info-bearing checkpoints it
                    # exists to widen, handed them the 9-channel panel and died
                    # in _match_width. n_features is the number both halves
                    # agree on.
                    use_batches = batches
                    if (getattr(backbone, "n_features", 0) or 0) > eg.N_FEATURES:
                        use_batches = panels_for(True)[bi][1]
                    try:
                        fwd = eg.forward_cached(backbone, use_batches, device,
                                                cap=10**9)
                    finally:
                        backbone.to("cpu")
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                tk = np.asarray([str(t) for t in fwd["tickers"]], dtype=object)
                dt = np.asarray([str(d) for d in fwd["dates"]], dtype=object)
                for panel in panels:
                    in_panel = np.isin(tk, sorted(panel))
                    keep = np.zeros(len(tk), dtype=bool)
                    for d in sorted(set(dt)):
                        sel = in_panel & (dt == d)
                        if sel.sum() == P * S:
                            keep |= sel
                    m_out = panel_metrics(
                        fwd["X"][keep], tk[keep], dt[keep], panel,
                        S, P * S, args.n_perm, metrics,
                    )
                    for k, v in m_out.items():
                        month_vals[key][k].append(v)
        for key in models:
            for k in metrics:
                res[key][k].append(month_vals[key][k])
        print(f"{ev_month} done", flush=True)

    n_panels = len(res[models[0]][metrics[0]][0])
    print(f"\n{n_panels} panels/month; t across {len(label_months)} month-means")
    summary = {}
    for metric in metrics:
        n_cand = np.mean([
            np.mean([v[4] for v in ml]) for ml in res[models[0]][metric]
        ])
        print(f"\n=== {metric}. {METRIC_NAMES[metric]} ===")
        print(f"(top-1 | mean rank of target, {n_cand:.0f} candidates, "
              f"chance rank {(n_cand + 1) / 2:.1f})")
        print(f"{'model':<22} {'rate':>7} {'chance':>7} {'ratio':>6} {'t':>6}"
              f" {'| rank':>7} {'pctile':>7} {'t':>6}")
        for key in models:
            month_arr = np.array([
                [np.mean([v[i] for v in ml]) for i in range(4)]
                for ml in res[key][metric]
            ])  # (months, 4): rate, chance, rank, pctile
            rates, chances = month_arr[:, 0], month_arr[:, 1]
            ranks, pcts = month_arr[:, 2], month_arr[:, 3]
            ex = rates - chances
            t = ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))
            ex_r = pcts - 0.5
            t_r = ex_r.mean() / (ex_r.std(ddof=1) / np.sqrt(len(ex_r)))
            print(f"{MODEL_SPECS[key]['label']:<22} {rates.mean():>6.1%} "
                  f"{chances.mean():>6.1%} {rates.mean() / chances.mean():>5.2f}x "
                  f"{t:>6.2f} {ranks.mean():>7.2f} {pcts.mean():>6.1%} {t_r:>6.2f}")
            summary.setdefault(key, {})[f"metric{metric}"] = {
                "rate": float(rates.mean()),
                "chance": float(chances.mean()),
                "t": float(t),
                "mean_rank": float(ranks.mean()),
                "mean_pctile": float(pcts.mean()),
                "rank_t": float(t_r),
                "n_candidates": float(n_cand),
                # The four per-month series, not just two. `t` is taken over
                # (rate - chance) across month-means and `rank_t` over
                # (pctile - 0.5), so a merge of per-month runs can only
                # reproduce this file's own arithmetic if chances and pctiles
                # survive alongside rates and ranks. Added 2026-08-20 so the
                # 32-month panel can be run one month per job and merged.
                "month_rates": {m: float(r) for m, r in zip(label_months, rates)},
                "month_ranks": {m: float(r) for m, r in zip(label_months, ranks)},
                "month_chances": {m: float(c) for m, c in zip(label_months, chances)},
                "month_pctiles": {m: float(q) for m, q in zip(label_months, pcts)},
            }

    out_json = eg.OUT_DIR / f"fixed_panel_P{P}S{S}{args.out_suffix}.json"
    # Merge over an existing file so a --models/--metrics subset run
    # refreshes only those entries instead of clobbering the rest of the
    # panel (per-metric, so backfilling one task keeps the others).
    #
    # THE MERGE RE-POOLS ACROSS MONTHS rather than replacing the cell. A
    # month-major driver calls this once per month, and a plain dict update
    # would leave the file describing the LAST month alone while still
    # carrying panel-wide field names -- silently, since every field is
    # present and well-formed. The per-month series exist for exactly this
    # (they were added 2026-08-20 "so the 32-month panel can be run one month
    # per job and merged"); this is that merge, done in place, with the same
    # arithmetic as plots/tsfm_layers/merge_latent.merge_fixed_panel: t is
    # (rate - chance) over MONTH-MEANS and rank_t is (pctile - 0.5) over them,
    # so both are recomputed from the union and never averaged.
    #
    # A month present on both sides takes the NEW run's value, so re-running
    # one month refreshes it instead of double-counting it.
    if out_json.exists():
        prior = json.load(open(out_json)).get("models", {})
        for key, ent in summary.items():
            for metric, cell in ent.items():
                old_cell = prior.get(key, {}).get(metric)
                if old_cell and "month_rates" in old_cell:
                    cell = _repool(old_cell, cell)
                prior.setdefault(key, {})[metric] = cell
        summary = prior
    with open(out_json, "w") as f:
        json.dump(
            {"P": P, "S": S, "n_panels": n_panels,
             "metric_names": {str(k): v for k, v in METRIC_NAMES.items()},
             "models": summary},
            f, indent=1,
        )
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
