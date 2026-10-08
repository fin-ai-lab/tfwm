"""Three numbers about the window, in the info token -- not three channels.

The patch embedding sees the nine real feature columns and nothing else. The
window descriptors occupy one final-row metadata payload that the transformer
projects into one positionless token.
"""
import numpy as np
import pytest
import torch

from market_jepa.modeling.backbones import create_backbone
from market_jepa.modeling.backbones.transformer import TransformerConfig
from market_jepa.augmentations import (
    N_WINDOW_INFO, SESSION_SECONDS, encode_view_metadata,
)
from market_jepa.training.utils import append_view_info
from stable_finance.dataset import ViewMetadata


def window_info(tod_start_sec: float, agg: float, n_tokens: int) -> np.ndarray:
    """The three window descriptors, via the encoder the dataset actually uses.

    These tests used to call a second implementation living in training.utils;
    it was deleted once nothing on the training path called it. Building the
    metadata here keeps the assertions pointed at production arithmetic.
    """
    return encode_view_metadata(
        ViewMetadata(
            start_seconds=float(tod_start_sec),
            end_seconds=float(tod_start_sec + n_tokens * agg),
            aggregation_seconds=float(agg),
            normalization_means=np.zeros(0),
            normalization_scales=np.zeros(0),
        ),
        include_normalization=False,
        include_window=True,
    )


def test_the_three_numbers_are_start_end_and_log_resolution():
    v = window_info(tod_start_sec=11700, agg=8, n_tokens=2048)
    assert len(v) == N_WINDOW_INFO == 3
    assert v[0] == pytest.approx(11700 / SESSION_SECONDS)
    assert v[1] == pytest.approx((11700 + 2048 * 8) / SESSION_SECONDS)
    assert v[2] == pytest.approx(np.log(8))


def test_resolution_is_logged_so_a_2x_span_is_a_constant_offset():
    a = window_info(0, 6, 2048)[2]
    b = window_info(0, 12, 2048)[2]
    assert b - a == pytest.approx(np.log(2))


def test_metadata_is_not_repeated_along_the_series():
    view = append_view_info(np.ones((16, 2)), np.array([3.0, 4.0]))
    assert np.count_nonzero(view[:-1, 2:]) == 0
    np.testing.assert_array_equal(view[-1, 2:], [3.0, 4.0])


def test_nothing_reaches_the_patch_embedding():
    """9 real columns + 8 norm stats + 3 window descriptors = 20 in, 9 patched."""
    bb = create_backbone(
        backbone_type="transformer", n_features=20, d_embedding=384,
        config=TransformerConfig(hidden_size=384, num_hidden_layers=2,
                                 num_attention_heads=6, patch_size=8),
        max_seq_len=2048, pool="cls", n_info_channels=8 + N_WINDOW_INFO)
    assert bb.patch_embed.proj.weight.shape[1] == 9
    assert bb.info_proj.weight.shape[1] == 11


def test_the_model_reads_the_resolution():
    bb = create_backbone(
        backbone_type="transformer", n_features=20, d_embedding=384,
        config=TransformerConfig(hidden_size=384, num_hidden_layers=2,
                                 num_attention_heads=6, patch_size=8),
        max_seq_len=2048, pool="cls", n_info_channels=8 + N_WINDOW_INFO).eval()

    def view_at(agg):
        v = np.zeros((2048, 17), dtype=np.float64)
        v = append_view_info(v, window_info(11700, agg, 2048))
        return torch.from_numpy(v.T[None]).float().repeat(2, 1, 1)

    L = torch.full((2,), 2048, dtype=torch.long)
    with torch.no_grad():
        a = bb(view_at(6), lengths=L)
        b = bb(view_at(11), lengths=L)
    a = a["embeddings"] if isinstance(a, dict) else a
    b = b["embeddings"] if isinstance(b, dict) else b
    # Same series, different declared resolution -> different embedding.
    # Without this, anything the model learns as a function of PATCH distance
    # is a 2x wall-clock smear it cannot see.
    #
    # THE THRESHOLD DROPPED AN ORDER OF MAGNITUDE WHEN THE RECENCY PRIOR WAS
    # RETIRED, and that is a fact about the model, not slack in the test. The
    # info token was EXEMPT from the prior's distance penalty while all 256
    # patches paid it, so it held a large share of the readout's attention.
    # Under global attention it holds ~1/P of it, and the sensitivity measured
    # here falls as 1/P almost exactly: 3.1e-3 at 32 patches, 1.4e-3 at 64,
    # 7.4e-4 at 128, 4.1e-4 at 256. An untrained backbone can only show that
    # the path EXISTS; how much a trained one uses it is a separate question,
    # and one the retirement has made materially harder for the model.
    assert (a - b).abs().max() > 1e-4


def test_information_token_reads_each_samples_last_valid_timestep():
    bb = create_backbone(
        backbone_type="transformer", n_features=20, d_embedding=384,
        config=TransformerConfig(hidden_size=384, num_hidden_layers=1,
                                 num_attention_heads=6, patch_size=8),
        max_seq_len=32, pool="cls", n_info_channels=11,
    ).eval()
    x = torch.zeros(2, 20, 32)
    expected = torch.stack([torch.arange(11), torch.arange(11) + 20]).float()
    x[0, -11:, 15] = expected[0]
    x[1, -11:, 23] = expected[1]
    x[:, -11:, -1] = -999  # right-padding must not become metadata
    seen = []
    hook = bb.info_proj.register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach()))
    try:
        with torch.no_grad():
            bb(x, lengths=torch.tensor([16, 24]))
    finally:
        hook.remove()
    torch.testing.assert_close(seen[0], expected)


def test_a_view_builder_that_forgets_them_raises_rather_than_zero_filling():
    """Eight view builders across two files call _normalize_view. One that
    omitted the arguments would emit a view the backbone strips anyway, so the
    info token would be fed zeros and nothing would fail."""
    from market_jepa.training.streaming_dataset import StreamingMarketDataset

    obj = StreamingMarketDataset.__new__(StreamingMarketDataset)
    obj.info_norm_stats = False
    obj.info_window = True
    obj._norm_groups = []
    with pytest.raises(ValueError, match="tod_start_sec"):
        obj._normalize_view(np.zeros((16, 9)))


def test_the_panel_accessor_carries_it_so_no_caller_can_forget():
    import sys
    sys.path.insert(0, "scripts/eval")
    from xs_ic_eval import panel_kwargs_for

    assert "info_window" in panel_kwargs_for(None)
    assert panel_kwargs_for({"dataset": {"info_window": True}})["info_window"] is True


def test_a_checkpoint_stamped_with_the_old_key_still_scores_the_way_it_trained():
    """``time_info`` was the key until 2026-08-25. A checkpoint is a historical
    record, so the reader accepts both spellings -- and defaults to False even
    though the config default is now True, because save_train_meta writes these
    only when they are ON and absence therefore still means off."""
    import sys
    sys.path.insert(0, "scripts/eval")
    from xs_ic_eval import panel_kwargs_for

    assert panel_kwargs_for({"dataset": {"time_info": True}})["info_window"] is True
    assert panel_kwargs_for({"dataset": {}})["info_window"] is False
    assert panel_kwargs_for(
        {"dataset": {"norm_stats_channels": True}})["info_norm_stats"] is True
    assert panel_kwargs_for({"dataset": {}})["info_norm_stats"] is False
