"""Per-month fixed-panel runs must POOL, not replace.

The latent driver is month-major -- one month's data is read once and every
stage consumes it -- so fixed_panel_metrics is called once per month and its
in-file merge is what turns 31 single-month runs into the panel. Before this
it was a dict update: the file ended up describing the LAST month alone while
still carrying panel-wide field names, with every field present and
well-formed, so nothing downstream could notice.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "plots" / "latent_eval" / "fixed_panel"))
sys.path.insert(0, str(REPO / "plots" / "tsfm_layers"))


def _cell(month, rate, chance, rank, pct, n_cand=103.0):
    return {
        "rate": rate, "chance": chance, "t": float("nan"),
        "mean_rank": rank, "mean_pctile": pct, "rank_t": float("nan"),
        "n_candidates": n_cand,
        "month_rates": {month: rate}, "month_chances": {month: chance},
        "month_ranks": {month: rank}, "month_pctiles": {month: pct},
    }


MONTHS = {
    "2008-08": (0.033, 0.0097, 40.2, 0.383),
    "2008-09": (0.058, 0.0095, 35.9, 0.336),
    "2008-10": (0.053, 0.0087, 37.8, 0.323),
}


def _accumulate():
    from fixed_panel_metrics import _repool
    cells = [_cell(m, *v) for m, v in MONTHS.items()]
    out = cells[0]
    for nxt in cells[1:]:
        out = _repool(out, nxt)
    return out


def test_pooling_matches_the_shard_merger_exactly():
    """The same arithmetic as merge_latent.merge_fixed_panel, not an average."""
    from merge_latent import merge_fixed_panel

    shards = {
        m: {"P": 3, "S": 2, "n_panels": 10, "metric_names": {},
            "models": {"x": {"metric1": _cell(m, *v)}}}
        for m, v in MONTHS.items()
    }
    ref = merge_fixed_panel(shards)["models"]["x"]["metric1"]
    got = _accumulate()
    for field in ("rate", "chance", "t", "mean_rank", "mean_pctile", "rank_t"):
        assert got[field] == pytest.approx(ref[field], rel=1e-12), field
    assert got["n_months"] == len(MONTHS) == ref["n_months"]


def test_t_is_recomputed_from_the_union_not_averaged():
    got = _accumulate()
    rates = np.array([v[0] for v in MONTHS.values()])
    chances = np.array([v[1] for v in MONTHS.values()])
    ex = rates - chances
    assert got["t"] == pytest.approx(
        ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex))), rel=1e-12)
    # Each input cell carries t = NaN (one month has no spread), so anything
    # that combined the cells' own t values could not produce a real number.
    assert np.isfinite(got["t"])


def test_every_month_survives_the_merge():
    got = _accumulate()
    for field in ("month_rates", "month_chances", "month_ranks",
                  "month_pctiles"):
        assert sorted(got[field]) == sorted(MONTHS), field


def test_rerunning_a_month_refreshes_it_rather_than_double_counting():
    from fixed_panel_metrics import _repool
    first = _cell("2008-08", 0.033, 0.0097, 40.2, 0.383)
    second = _cell("2008-09", 0.058, 0.0095, 35.9, 0.336)
    pooled = _repool(first, second)
    redone = _repool(pooled, _cell("2008-08", 0.040, 0.0097, 39.0, 0.370))
    assert redone["n_months"] == 2
    assert redone["month_rates"]["2008-08"] == pytest.approx(0.040)
    assert redone["rate"] == pytest.approx((0.040 + 0.058) / 2)


def test_a_single_month_still_reports_nan_t():
    """Merging nothing new must not invent a spread the panel does not have."""
    from fixed_panel_metrics import _repool
    one = _cell("2008-08", 0.033, 0.0097, 40.2, 0.383)
    assert np.isnan(_repool(one, one)["t"])


def _shard(tmp, month, rate, chance, rank, pct, models=("x",)):
    import json
    payload = {
        "P": 3, "S": 2, "n_panels": 10, "metric_names": {"1": "t1"},
        "models": {m: {"metric1": _cell(month, rate, chance, rank, pct)}
                   for m in models},
    }
    (tmp / f"fixed_panel_P3S2_tst_m{month}.json").write_text(json.dumps(payload))


def test_merge_month_shards_pools_and_cleans_up(tmp_path):
    """The driver's merge step reproduces the serial run's numbers."""
    import json
    import subprocess

    for m, v in MONTHS.items():
        _shard(tmp_path, m, *v)
    r = subprocess.run(
        [sys.executable,
         str(REPO / "plots" / "latent_eval" / "fixed_panel" / "merge_month_shards.py"),
         "--tag", "_tst", "--dir", str(tmp_path)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    out = json.loads((tmp_path / "fixed_panel_P3S2_tst.json").read_text())
    assert out["months"] == sorted(MONTHS)
    cell = out["models"]["x"]["metric1"]
    assert cell["n_months"] == len(MONTHS)
    ref = _accumulate()
    for field in ("rate", "chance", "t", "mean_rank", "mean_pctile", "rank_t"):
        assert cell[field] == pytest.approx(ref[field], rel=1e-12), field
    for field in ("month_rates", "month_chances", "month_ranks",
                  "month_pctiles"):
        assert cell[field] == pytest.approx(ref[field], rel=1e-12), field
    # Shards are consumed, so a later run cannot silently re-pool stale months.
    assert not list(tmp_path.glob("fixed_panel_P3S2_tst_m*.json"))


def test_merge_pools_a_model_over_only_the_months_it_has(tmp_path):
    """One family failing in one month degrades that cell, not the file."""
    import json
    import subprocess

    ms = list(MONTHS)
    _shard(tmp_path, ms[0], *MONTHS[ms[0]], models=("x", "y"))
    _shard(tmp_path, ms[1], *MONTHS[ms[1]], models=("x",))
    _shard(tmp_path, ms[2], *MONTHS[ms[2]], models=("x", "y"))
    r = subprocess.run(
        [sys.executable,
         str(REPO / "plots" / "latent_eval" / "fixed_panel" / "merge_month_shards.py"),
         "--tag", "_tst", "--dir", str(tmp_path)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads((tmp_path / "fixed_panel_P3S2_tst.json").read_text())
    assert out["models"]["x"]["metric1"]["n_months"] == 3
    assert out["models"]["y"]["metric1"]["n_months"] == 2
    assert sorted(out["models"]["y"]["metric1"]["month_rates"]) == [ms[0], ms[2]]
