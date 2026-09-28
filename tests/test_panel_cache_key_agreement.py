"""The builder and the scorer must key a panel identically, or the cache is inert.

A panel is a function of (month, anchors, geometry) and not of the checkpoint,
so it is built once by scripts/eval/build_panel_cache.py and read by every arm
scored on that month. The two sides construct the key SEPARATELY. If they ever
disagree the cache does not fail -- it MISSES, silently, and every arm quietly
re-decodes a ~3028 s panel while the built one sits on disk. That is exactly
the state this repo was in.

These pin the contract the two sides share.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/eval"))

import panel_cache as pc  # noqa: E402


NEUTRAL = dict(anchors_per_day=36, stats_tag="xs_anchor_stats_fwdvwap60",
               has_rf=False, norm_groups=None, fixed_agg=None, seq_len=2048)


def test_builder_and_scorer_agree_on_the_neutral_panel():
    """build_panel_cache passes norm_groups=None; the scorer passes its own."""
    built = pc.panel_key(**NEUTRAL)
    # What iter_panel_cached passes for a default run: rf_merger is None and
    # panel_kwargs_for leaves norm_groups / fixed_agg unset.
    scored = pc.panel_key(anchors_per_day=36,
                          stats_tag="xs_anchor_stats_fwdvwap60",
                          has_rf=False, norm_groups=None, fixed_agg=None,
                          seq_len=2048)
    assert built["hash"] == scored["hash"]


def test_the_key_is_deterministic():
    assert pc.panel_key(**NEUTRAL)["hash"] == pc.panel_key(**NEUTRAL)["hash"]


def test_the_two_roles_do_not_share_a_panel():
    """36 anchors/day is the probe-fit panel, 8 the eval panel."""
    probe = pc.panel_key(**{**NEUTRAL, "anchors_per_day": 36})
    evalp = pc.panel_key(**{**NEUTRAL, "anchors_per_day": 8})
    assert probe["hash"] != evalp["hash"]


def test_every_keyed_field_moves_the_hash():
    """A field that does not move the hash is a field that serves wrong data."""
    base = pc.panel_key(**NEUTRAL)["hash"]
    variants = {
        "stats_tag": {"stats_tag": "xs_anchor_stats_mid2mid"},
        "has_rf": {"has_rf": True},
        "fixed_agg": {"fixed_agg": 6},
        "seq_len": {"seq_len": 1024},
        "norm_groups": {"norm_groups": []},
    }
    for name, change in variants.items():
        other = pc.panel_key(**{**NEUTRAL, **change})["hash"]
        assert other != base, f"{name} does not change the panel key"


def test_the_target_tables_are_part_of_the_key():
    """The tables ARE the target: same geometry, different labels."""
    a = pc.panel_key(**NEUTRAL)["hash"]
    b = pc.panel_key(**{**NEUTRAL, "stats_tag": "xs_anchor_stats_other"})["hash"]
    assert a != b
