"""k2ind is the standard model: who inherits it, and who must not.

The dataset's default positive pair is now cross_stock K=2 restricted to the
focal's FF49 industry. That default reaches any mode that does not pin itself
away from it, which makes "which modes pin" a correctness property rather than
a style question: a single-view mode handed a partner ticker would draw K
stocks and discard K-1, and the eval path handed one would embed a DIFFERENT
STOCK than the row it labels.

Both schemes sample their window with a random resized crop. What this default
selects is only what counts as a POSITIVE PAIR.

DINO AND BYOL LEFT THE DEFAULT ON 2026-09-14. They are still joint-embedding
modes, so the old rule here -- every uses_multi_view mode inherits k2ind --
put them on an industry-matched partner ticker. They now pin time_warp: two
warps of ONE window, a same-stock pair. The point is comparability, since the
lejepa-6mo-warp arm draws its pair exactly this way, so DINO / BYOL / LeJEPA
over time_warp differ only in the objective. LeJEPA keeps inheriting k2ind
because its own arms name their pairing on the command line.

NEITHER SCHEME GIVES THESE TWO LOCAL CROPS, which is worth saying plainly
because both docstrings advertise global+local multi-crop: cross_stock reads
cross_stock_local_views (0) and the corruption family emits no n_local_views
key at all, so the local branch of both losses is inert either way. Only
random_resized_crop would feed it.
"""

import inspect

import pytest

import market_jepa.schemas as S
from market_jepa.schemas import DatasetConfig, STANDARD_INDUSTRY_TABLE

# The three joint-embedding methods — the ones with uses_multi_view=True.
# All form a positive pair; they differ in WHICH pair, so they split two ways.
MULTI_VIEW_MODES = ["LeJEPAModeConfig", "DINOModeConfig", "BYOLModeConfig"]

# Inherits the k2ind default: a partner ticker in the focal's FF49 industry.
PAIRING_INHERITED_MODES = ["LeJEPAModeConfig"]

# Pins a same-stock pair instead (see the module docstring).
WARP_PAIRED_MODES = ["DINOModeConfig", "BYOLModeConfig"]


def _mode_configs():
    for name in dir(S):
        if not name.endswith("ModeConfig"):
            continue
        cls = getattr(S, name)
        if inspect.isclass(cls):
            yield name, cls


def test_training_default_is_k2ind():
    aug = DatasetConfig().augmentations["0"]
    assert aug.name == "cross_stock"
    assert aug.n_stocks == 2
    assert aug.industry_table == STANDARD_INDUSTRY_TABLE
    # k2ind carries no local views; cross_stock reads its own knob, never
    # n_local_views.
    assert aug.cross_stock_local_views == 0


def test_eval_never_pairs():
    """The probe embeds one stock per row. Inheriting cross_stock here would
    silently score a partner ticker's window against the focal's label."""
    ev = DatasetConfig().eval_augmentations["0"]
    assert ev.name == "random_resized_crop"
    assert ev.n_local_views == 0


@pytest.mark.parametrize("name,cls", list(_mode_configs()))
def test_single_view_modes_pin_away_from_the_pairing(name, cls):
    overrides = getattr(cls(), "dataset_overrides", None)
    pinned = getattr(overrides, "name", None) if overrides else None

    if name in PAIRING_INHERITED_MODES:
        assert pinned is None, (
            f"{name} is a joint-embedding mode and must INHERIT the standard "
            f"k2ind pairing, but it pins name={pinned!r}."
        )
    elif name in WARP_PAIRED_MODES:
        assert pinned == "time_warp", (
            f"{name} takes a SAME-STOCK pair -- two warps of one window, the "
            f"pairing the lejepa-6mo-warp arm uses -- so it must pin "
            f"dataset_overrides.name='time_warp'. It has {pinned!r}; None "
            f"would put it back on an industry-matched partner ticker."
        )
    elif name in CELL_TRAINING_MODES:
        # Not a positive pair: a labelled CELL. The specialist ranks K stocks
        # at one anchor (compute_group_loss), which is the quantity the
        # reported IC measures; one crop per row could only pair rows that
        # collided into a cell by chance (~74 pairs per 256 rows, 2026-09-11).
        assert pinned == "cross_stock" and (overrides.n_stocks or 0) >= 2, (
            f"{name} trains on cross-sectional cells and must pin "
            f"name='cross_stock' with n_stocks>=2; has {pinned!r}, "
            f"n_stocks={getattr(overrides, 'n_stocks', None)}")
        assert overrides.industry_table is None if hasattr(overrides, "industry_table") else True
    else:
        assert pinned == "random_resized_crop", (
            f"{name} is single-view and never forms a positive pair, so it "
            f"must pin dataset_overrides.name='random_resized_crop'. It has "
            f"{pinned!r}, so it would inherit the cross-stock default and draw "
            f"partner tickers it then discards."
        )


def test_multi_view_set_matches_the_models():
    """The pin list above is derived from uses_multi_view; keep them in sync."""
    from market_jepa.modeling.modes.byol import BYOL
    from market_jepa.modeling.modes.dino import DINO
    from market_jepa.modeling.modes.lejepa import LeJEPA
    from market_jepa.modeling.modes.mae import MAE
    from market_jepa.modeling.modes.supervised import SupervisedModel

    assert LeJEPA.uses_multi_view and DINO.uses_multi_view and BYOL.uses_multi_view
    assert not MAE.uses_multi_view and not SupervisedModel.uses_multi_view
    # The two pairing lists partition the joint-embedding set: a mode added to
    # one and not the other would fall through to the single-view branch and
    # be asked to pin random_resized_crop.
    assert sorted(PAIRING_INHERITED_MODES + WARP_PAIRED_MODES) == sorted(MULTI_VIEW_MODES)
    assert not set(PAIRING_INHERITED_MODES) & set(WARP_PAIRED_MODES)


# The supervised specialist trains on CELLS (cross_stock K=16, no industry
# restriction) since 2026-09-11 -- see SupervisedModeConfig.dataset_overrides.
# Still single-view for the eval/probe side, which pretrain.py pins to one
# rrc global regardless.
CELL_TRAINING_MODES = ["SupervisedModeConfig", "MultiTaskSupervisedModeConfig"]

# Every single-view mode that TRAINS. Pinning name alone stops the pairing but
# not the view counts, so these would still build 2 globals + 6 locals per
# sample and throw seven of the eight away -- pure dataloader cost on a path
# that is already the bottleneck.
SINGLE_VIEW_TRAINING_MODES = [
    "MAEModeConfig", "IJEPAModeConfig", "CPCModeConfig", "TS2VecModeConfig",
    "CoSTModeConfig", "TFCModeConfig", "TimeMAEModeConfig",
    "SupervisedModeConfig", "MultiTaskSupervisedModeConfig",
]

# These two are single-view but never train -- a frozen TSFM is embedded and a
# finance baseline is fit -- so their loader cost is a one-off rather than a
# per-step tax, and they leave the counts inherited.
SINGLE_VIEW_NON_TRAINING_MODES = [
    "PretrainedTSFMModeConfig", "FinanceBaselineModeConfig",
]


@pytest.mark.parametrize("name", SINGLE_VIEW_TRAINING_MODES)
def test_single_view_training_modes_pin_one_global_and_no_locals(name):
    """MAE and IJEPA were the last two not pinning these; the other five SSL
    modes and both supervised arms always did.

    train_meta is where this is checked after the fact, NOT the saved
    cfg.dataset: dataset_overrides are applied at runtime, so a supervised
    checkpoint records cross_stock/2/6 while having trained one rrc view.
    """
    o = getattr(S, name)().dataset_overrides
    if name in CELL_TRAINING_MODES:
        assert o.name == "cross_stock"
    else:
        assert o.name == "random_resized_crop"
    assert o.n_global_views == 1, f"{name} does not pin n_global_views=1"
    assert o.n_local_views == 0, f"{name} does not pin n_local_views=0"


@pytest.mark.parametrize("name", SINGLE_VIEW_NON_TRAINING_MODES)
def test_the_non_training_modes_leave_view_counts_inherited(name):
    """Nothing is gained by pinning a loader that runs once."""
    o = getattr(S, name)().dataset_overrides
    assert o.name == "random_resized_crop"
    assert o.n_global_views is None, f"{name} unexpectedly pins n_global_views"
    assert o.n_local_views is None, f"{name} unexpectedly pins n_local_views"


def test_the_multi_view_modes_never_pin_view_counts():
    """LeJEPA/DINO/BYOL need the 2 globals + 6 locals they inherit; pinning
    1/0 here would silently delete the multi-crop objective."""
    for name in MULTI_VIEW_MODES:
        o = getattr(S, name, None)
        o = getattr(o(), "dataset_overrides", None) if o else None
        assert o is None or (o.n_global_views is None and o.n_local_views is None), (
            f"{name} pins view counts and would lose its local crops")
