"""A frozen TSFM must read the same nine data channels on any grid width.

The full-day grid is built WITH the information-token channels (9 data + 11
trailing constants) so one grid can feed both the info-bearing supervised
encoders and the frozen TSFMs. Every family narrows to ``self.channels``
inside ``_compute_features_all`` -- except kronos, which returned before that
line and so was handed the whole 20-wide view. It reads vwap/high/low/volume
by FEATURE_COLUMNS position and aggregates under a 9-column schema, so it died
with "features do not match the supplied schema" the first time a run put
supervised arms and TSFMs on one grid.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from market_jepa.modeling.modes.pretrained_tsfm import PretrainedTSFM  # noqa: E402
from market_jepa.training.streaming_dataset import FEATURE_COLUMNS  # noqa: E402

N_DATA = len(FEATURE_COLUMNS)
N_INFO = 11


def _view(batch, width, seq=512, seed=0):
    """A plausible 1 Hz window: data channels vary, info channels are constant."""
    rng = np.random.default_rng(seed)
    x = np.zeros((batch, width, seq), dtype=np.float32)
    x[:, :N_DATA] = rng.uniform(1.0, 2.0, (batch, N_DATA, seq)).astype(np.float32)
    # volume / n are counts; keep them non-negative and integral-ish
    x[:, FEATURE_COLUMNS.index("volume")] = rng.integers(0, 50, (batch, seq))
    if width > N_DATA:
        x[:, N_DATA:] = 0.25   # the trailing info constants
    return torch.from_numpy(x)


def test_kronos_refuses_a_channel_subset():
    """The positional bar reader cannot honour one, and says so at build time."""
    with pytest.raises(ValueError, match="FEATURE_COLUMNS layout"):
        PretrainedTSFM(backbone=None, model="kronos", channels=[0, 1, 2])


@pytest.mark.parametrize("family", ["kronos"])
def test_info_grid_gives_the_same_features_as_a_bare_panel(family):
    """Widening the grid with constant info columns must not move the features."""
    model = PretrainedTSFM(backbone=None, model=family,
                           channels=list(range(N_DATA))).eval()
    bare = _view(2, N_DATA)
    wide = _view(2, N_DATA + N_INFO)
    assert torch.equal(bare, wide[:, :N_DATA])      # same data, wider frame

    lengths = torch.full((2,), bare.shape[-1], dtype=torch.long)
    layers = [0, 1]
    a = model.compute_features_multi(bare, lengths, layers)
    b = model.compute_features_multi(wide, lengths, layers)
    for L in layers:
        assert torch.allclose(a[L], b[L], atol=1e-5), (
            f"layer {L}: the info columns changed the features")


def test_the_narrowing_happens_before_the_kronos_branch():
    """Source-level guard: the branch must not receive the raw view.

    The failure this pins is silent on a 9-wide panel and only appears on a
    grid built for info-bearing encoders, so a unit test that never builds one
    would not catch a regression here.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(
        inspect.getsource(PretrainedTSFM._compute_features_all)))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "attr", "") == "_compute_features_kronos"]
    assert len(calls) == 1, "expected exactly one kronos dispatch"
    view_arg = calls[0].args[0]
    assert isinstance(view_arg, ast.Subscript), (
        "kronos is handed the un-narrowed view again -- it must be sliced to "
        "self.channels before _kronos_bars reads columns positionally")
    assert "channels" in ast.dump(view_arg), (
        "kronos must select self.channels, not some other slice")
