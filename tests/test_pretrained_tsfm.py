"""Tests for the frozen pretrained-TSFM mode.

Two layers, cheapest first:

  * **Mechanics** (no model download) — config validation, tail alignment,
    timestamp phase features, embedding-dim bookkeeping, save/load, and the
    harness interface (``wants_metadata``, hidden TSFM registry).
  * **Model-backed** — tiny batches through each real checkpoint: shapes,
    finiteness, per-sample length handling, hook capture vs Sundial's native
    ``output_hidden_states`` (validates the layer indexing for all hook-based
    adapters), the TimesFM 3.0 multivariate feed (layout against a per-sample
    decode, variate attention actually mixing channels, and the horizon
    shortcut), the Chronos-2 timestamp covariates, and Kronos hook capture vs
    a manual block-by-block forward. These skip cleanly when the checkpoint
    cannot be loaded (no network and no HF cache).

All CPU, all synthetic — no real data, no W&B.
"""

from __future__ import annotations

import math
import tempfile

import pytest
import torch

from market_jepa.modeling.modes import PretrainedTSFM


class _DummyBackbone(torch.nn.Module):
    n_features = 9

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)


def _model(**kwargs) -> PretrainedTSFM:
    kwargs.setdefault("backbone", _DummyBackbone())
    kwargs.setdefault("model", "sundial")
    return PretrainedTSFM(**kwargs)


def _window(B: int = 3, C: int = 9, T: int = 256, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, C, T, generator=g).cumsum(dim=-1) * 0.01 + 1.0


def _load_or_skip(model: PretrainedTSFM) -> None:
    try:
        model._tsfm(torch.device("cpu"))
    except Exception as e:  # no network + no cache
        pytest.skip(f"cannot load {model.model_id}: {e}")


# ---------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_unknown_family_rejected(self):
        with pytest.raises(ValueError, match="Unknown TSFM family"):
            _model(model="prophet")

    def test_layer_bounds(self):
        with pytest.raises(ValueError, match="layer must be in"):
            _model(model="sundial", layer=13)
        with pytest.raises(ValueError, match="layer must be in"):
            _model(model="timesfm", layer=-22)

    def test_negative_layer_resolves_from_end(self):
        assert _model(model="sundial", layer=-1).layer == 12
        assert _model(model="timesfm", layer=-1).layer == 20
        assert _model(model="chronos2", layer=-13).layer == 0

    def test_timestamps_only_for_chronos2(self):
        with pytest.raises(ValueError, match="only supported for chronos2"):
            _model(model="timesfm", use_timestamps=True)
        with pytest.raises(ValueError, match="only supported for chronos2"):
            _model(model="sundial", use_timestamps=True)
        # timesfm3 HAS a covariate channel that could carry them; the arm just
        # is not implemented, and the guard must still refuse rather than
        # silently drop the request.
        with pytest.raises(ValueError, match="only supported for chronos2"):
            _model(model="timesfm3", use_timestamps=True)
        assert _model(model="chronos2", use_timestamps=True).use_timestamps

    def test_reg_pool_only_for_chronos2(self):
        with pytest.raises(ValueError, match="only meaningful for chronos2"):
            _model(model="sundial", time_pool="reg")

    def test_invalid_pools_rejected(self):
        with pytest.raises(ValueError, match="channel_pool"):
            _model(channel_pool="max")
        with pytest.raises(ValueError, match="time_pool"):
            _model(time_pool="cls")

    def test_empty_channels_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            _model(channels=[])

    def test_d_embedding_bookkeeping(self):
        assert _model(model="sundial").d_embedding == 9 * 768
        assert _model(model="timesfm").d_embedding == 9 * 1280
        # timesfm3 reads the nine channels as one multivariate group but the
        # READOUT still concatenates the nine per-variate states, so it lands
        # on TimesFM 2.5's width and the two are directly comparable.
        assert _model(model="timesfm3").d_embedding == 9 * 1280
        assert _model(model="chronos2", channel_pool="mean").d_embedding == 768
        assert _model(model="sundial", channels=[0, 4]).d_embedding == 2 * 768
        # kronos: one multivariate OHLCVA row per sample, never concat.
        assert _model(model="kronos").d_embedding == 832

    def test_kronos_config(self):
        m = _model(model="kronos")
        assert (m.bar_agg, m.max_context, m.n_layers) == (4, 512, 12)
        assert m.tokenizer_id == "NeoQuasar/Kronos-Tokenizer-base"
        assert _model(model="kronos", layer=-1).layer == 12
        assert _model(model="kronos", bar_agg=1).bar_agg == 1
        with pytest.raises(ValueError, match="only meaningful for kronos"):
            _model(model="sundial", bar_agg=4)
        with pytest.raises(ValueError, match="only meaningful for kronos"):
            _model(model="sundial", tokenizer_id="NeoQuasar/Kronos-Tokenizer-2k")
        with pytest.raises(ValueError, match="full 9-channel"):
            _model(model="kronos", channels=[1, 2, 3])
        with pytest.raises(ValueError, match="only supported for chronos2"):
            _model(model="kronos", use_timestamps=True)

    def test_max_context_capped_at_family_limit(self):
        assert _model(model="sundial", max_context=100_000).max_context == 2880
        assert _model(model="chronos2", max_context=1024).max_context == 1024


class TestHarnessInterface:
    def test_wants_metadata(self):
        # collect_probe_data gates on this attribute; without it the
        # timestamp covariates would never see tod_secs/agg_factors.
        assert PretrainedTSFM.wants_metadata is True

    def test_not_multi_view(self):
        assert PretrainedTSFM.uses_multi_view is False

    def test_tsfm_hidden_from_module_registry(self):
        m = _model()
        # Nothing loaded yet, and even the holder must not be a submodule:
        # state_dict/optimizer must only ever see the dummy backbone.
        assert all("tsfm" not in k for k in m.state_dict())
        assert sum(p.numel() for p in m.parameters()) == sum(
            p.numel() for p in m.backbone.parameters()
        )

    def test_training_step_is_noop(self):
        assert _model().training_step({"buckets": []}, torch.device("cpu")) is None

    def test_describe_parameters(self):
        counts, summary = _model().describe_parameters()
        assert "total" in counts
        assert "sundial" in summary

    def test_save_load_roundtrip(self):
        m = _model(
            model="chronos2", layer=5, use_timestamps=True,
            channels=[0, 4], channel_pool="mean", time_pool="reg",
        )
        with tempfile.TemporaryDirectory() as d:
            m.save_pretrained(d)
            m2 = PretrainedTSFM.from_pretrained(d)
        assert m2.family == "chronos2"
        assert m2.layer == 5
        assert m2.use_timestamps is True
        assert m2.channels == [0, 4]
        assert (m2.channel_pool, m2.time_pool) == ("mean", "reg")
        assert m2.d_embedding == m.d_embedding

    def test_kronos_save_load_roundtrip(self):
        m = _model(model="kronos", layer=7, bar_agg=2)
        with tempfile.TemporaryDirectory() as d:
            m.save_pretrained(d)
            m2 = PretrainedTSFM.from_pretrained(d)
        assert (m2.family, m2.layer, m2.bar_agg) == ("kronos", 7, 2)
        assert m2.tokenizer_id == m.tokenizer_id
        assert m2.d_embedding == 832


class TestTailAlign:
    def test_full_rows_pass_through(self):
        m = _model(model="sundial")  # patch 16
        s = torch.arange(64, dtype=torch.float32).repeat(2, 1)
        out, pad = m._tail_align(s, torch.tensor([64, 64]), pad_value=0.0)
        assert out.shape == (2, 64)
        assert not pad.any()
        assert torch.equal(out, s)

    def test_short_row_front_padded(self):
        m = _model(model="sundial")
        s = torch.arange(64, dtype=torch.float32).repeat(2, 1)
        out, pad = m._tail_align(s, torch.tensor([64, 10]), pad_value=-7.0)
        # Row 1: last 10 valid values right-aligned, front filled with pad.
        assert pad[1, :-10].all() and not pad[1, -10:].any()
        assert torch.equal(out[1, -10:], s[1, :10])
        assert (out[1, :-10] == -7.0).all()
        assert not pad[0].any()

    def test_output_is_patch_multiple(self):
        m = _model(model="timesfm")  # patch 32
        s = torch.randn(1, 100)
        out, pad = m._tail_align(s, torch.tensor([100]), pad_value=0.0)
        assert out.shape[1] % 32 == 0
        assert out.shape[1] == 128
        assert pad[0, :28].all() and not pad[0, 28:].any()

    def test_long_rows_keep_most_recent(self):
        m = _model(model="sundial", max_context=32)
        s = torch.arange(64, dtype=torch.float32)[None, :]
        out, pad = m._tail_align(s, torch.tensor([64]), pad_value=0.0)
        assert out.shape == (1, 32)
        assert not pad.any()
        assert torch.equal(out[0], s[0, 32:])


class TestTimestampRows:
    def test_phase_features_encode_wall_clock(self):
        m = _model(model="chronos2", use_timestamps=True)
        meta = {
            "tod_secs": torch.tensor([0, 21600]),  # 6h apart
            "agg_factors": torch.tensor([1, 1]),
        }
        rows = m._timestamp_rows(meta, torch.tensor([8, 8]), 8, torch.device("cpu"))
        assert rows.shape == (2, 2, 8)
        # t=0: phase 0 -> sin 0, cos 1.
        assert rows[0, 0, 0].abs() < 1e-6
        assert (rows[0, 1, 0] - 1.0).abs() < 1e-6
        # 6h later: phase pi/2 -> sin 1, cos 0.
        assert (rows[1, 0, 0] - 1.0).abs() < 1e-5
        assert rows[1, 1, 0].abs() < 1e-5

    def test_agg_factor_sets_token_spacing(self):
        m = _model(model="chronos2", use_timestamps=True)
        meta = {
            "tod_secs": torch.tensor([0]),
            "agg_factors": torch.tensor([60]),
        }
        rows = m._timestamp_rows(meta, torch.tensor([4]), 4, torch.device("cpu"))
        expected = math.sin(2 * math.pi * 60.0 / 86400.0)
        assert (rows[0, 0, 1] - expected).abs() < 1e-6

    def test_missing_metadata_falls_back(self):
        m = _model(model="chronos2", use_timestamps=True)
        rows = m._timestamp_rows(None, torch.tensor([4]), 4, torch.device("cpu"))
        assert rows.shape == (1, 2, 4)
        assert torch.isfinite(rows).all()


class TestKronosBars:
    """Bar construction: the dataset aggregator's semantics must survive the
    trip through ``_kronos_bars``. Kronos's per-column z-score is monotone, so
    orderings — not levels — are the invariant to check."""

    def _view(self, B=2, T=64, seed=3):
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(B, 9, T, generator=g).cumsum(dim=-1) * 0.01
        v[:, 2] = v[:, 1] + torch.rand(B, T, generator=g) * 0.05  # high > vwap
        v[:, 3] = v[:, 1] - torch.rand(B, T, generator=g) * 0.05  # low  < vwap
        return v

    def test_shapes_padding_and_clip(self):
        m = _model(model="kronos")
        v = self._view(B=3, T=64)
        # Garbage beyond each row's valid length must be ignored.
        v[1, :, 40:] = 1e9
        bars, valid = m._kronos_bars(v, torch.tensor([64, 40, 64]))
        assert bars.shape == (3, 16, 6) and valid.shape == (3, 16)
        assert valid[0].all() and valid[2].all()
        assert (~valid[1, :6]).all() and valid[1, 6:].all()  # ceil(40/4)=10 bars
        assert bars.abs().max() <= 5.0 + 1e-6
        assert (bars[1, :6] == 0).all()  # front padding is neutral zeros

    def test_column_semantics_match_dataset_aggregator(self):
        m = _model(model="kronos")
        v = self._view(B=1, T=64)
        bars, _ = m._kronos_bars(v, torch.tensor([64]))
        f = v[0].T.numpy()  # (T, 9)
        buckets = f.reshape(16, 4, 9)
        # z-score is monotone per column -> orderings must match exactly.
        def order(a):
            return a.argsort().tolist()
        assert order(bars[0, :, 3].numpy()) == order(buckets[:, -1, 1])  # close = last vwap
        assert order(bars[0, :, 1].numpy()) == order(buckets[:, :, 2].max(axis=1))
        assert order(bars[0, :, 2].numpy()) == order(buckets[:, :, 3].min(axis=1))
        assert order(bars[0, :, 4].numpy()) == order(buckets[:, :, 7].sum(axis=1))
        # open = previous close (first bar: its first fine vwap).
        assert order(bars[0, 1:, 0].numpy()) == order(buckets[:-1, -1, 1])

    def test_tiny_length_collapses_to_one_bar(self):
        m = _model(model="kronos")
        v = self._view(B=1, T=64)
        bars, valid = m._kronos_bars(v, torch.tensor([3]))
        assert bars.shape == (1, 1, 6)
        assert valid.all()
        assert torch.isfinite(bars).all()


# ---------------------------------------------------------------------------
# Model-backed (skip when the checkpoint cannot be loaded)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family", ["timesfm", "timesfm3", "sundial", "chronos2", "kronos"])
class TestEncodeWithRealCheckpoint:
    def test_encode_shapes_and_lengths(self, family):
        m = _model(model=family, layer=1, tsfm_batch_size=32)
        _load_or_skip(m)
        x = _window(B=3, T=256)
        lengths = torch.tensor([256, 200, 97])
        out = m.encode([x], [lengths])
        e = out["embeddings"]
        assert e.shape == (3, 1, m.d_embedding)
        assert torch.isfinite(e).all()
        # Distinct windows must not collapse to one point.
        assert e[:, 0, :].std(dim=0).mean() > 1e-4

    def test_padding_does_not_leak(self, family):
        """A sample's embedding must not depend on other rows' padding."""
        m = _model(model=family, layer=1, tsfm_batch_size=32)
        _load_or_skip(m)
        x = _window(B=2, T=256)
        lengths = torch.tensor([256, 128])
        both = m.encode([x], [lengths])["embeddings"][0, 0]
        alone = m.encode([x[:1]], [lengths[:1]])["embeddings"][0, 0]
        assert torch.allclose(both, alone, atol=1e-4), (
            (both - alone).abs().max().item()
        )


class TestSundialLayerSemantics:
    def test_hook_capture_matches_output_hidden_states(self):
        """Validates the layer indexing shared by all hook-based adapters:
        0 = embeddings, k = block k output, last layer includes final norm."""
        for layer in [0, 5, 12]:
            m = _model(model="sundial", layer=layer, channels=[0])
            _load_or_skip(m)
            x = _window(B=2, C=9, T=128)
            feats = m._compute_features(x, None, None)

            inner = m._tsfm(torch.device("cpu")).model
            s = x[:, 0, :]
            mean = s.mean(dim=1, keepdim=True)
            std = s.std(dim=1, keepdim=True, unbiased=False)
            std = torch.where(std > 1e-2, std, torch.ones_like(std))
            with torch.no_grad():
                ref = inner(
                    input_ids=(s - mean) / std,
                    use_cache=False,
                    output_hidden_states=True,
                ).hidden_states[layer].mean(dim=1)
            assert torch.allclose(feats, ref, atol=1e-5), layer


class TestKronosLayerSemantics:
    def test_hook_capture_matches_manual_forward(self):
        """0 = embedding output, k = block k, last layer gets the final
        RMSNorm — checked block-by-block against the vendored model."""
        m = _model(model="kronos", layer=0)
        _load_or_skip(m)
        x = _window(B=2, T=64)
        lengths = torch.full((2,), 64, dtype=torch.long)
        multi = m.compute_features_multi(x, lengths, layers=[0, 5, 12])

        core = m._tsfm(torch.device("cpu"))
        tok = m._tsfm_holder["tok"]
        bars, _valid = m._kronos_bars(x, lengths)
        with torch.no_grad():
            s1, s2 = tok.encode(bars, half=True)
            h = core.embedding([s1, s2])
            ref = {0: h.mean(dim=1)}
            for i, blk in enumerate(core.transformer, start=1):
                h = blk(h)
                if i == 5:
                    ref[5] = h.mean(dim=1)
            ref[12] = core.norm(h).mean(dim=1)
        for layer, expect in ref.items():
            assert torch.allclose(multi[layer], expect, atol=1e-6), layer


class TestTimesFM3Semantics:
    """The multivariate feed: layout, layer indexing, and the horizon shortcut.

    TimesFM 3.0 is the one family where a sample's nine channels enter as the
    nine VARIATES of a single ``decode()`` call, so the readout flattens a
    ``(b, v, n, d)`` state back into rows. Getting that flatten wrong would
    transpose channels against samples and still produce plausible-looking
    finite embeddings, which is exactly the kind of bug a shape assertion
    misses.
    """

    def _ref(self, m, x, layer):
        """One sample at a time, straight through the package's own decode."""
        model = m._tsfm(torch.device("cpu"))
        caps = {}
        if layer == 0:
            h = model.transformer_stack.layers[0].register_forward_pre_hook(
                lambda _m, a: caps.__setitem__("s", a[0].detach()))
        else:
            h = model.transformer_stack.layers[layer - 1].register_forward_hook(
                lambda _m, _a, o: caps.__setitem__("s", o[0].detach()))
        try:
            with torch.no_grad():
                model.decode(
                    target=x[None],
                    horizon=1,
                    mask=torch.zeros(1, x.shape[-1], dtype=torch.bool),
                )
        finally:
            h.remove()
        P = x.shape[-1] // 32
        return caps["s"][0, :, :P, :].mean(dim=1).reshape(-1)

    def test_hook_capture_matches_manual_decode(self):
        """0 = pre-transformer resblock output, k = block k, and the variate
        axis flattens into rows in sample-major order."""
        m = _model(model="timesfm3", tsfm_batch_size=18)
        _load_or_skip(m)
        x = _window(B=2, C=9, T=128)
        feats = m.compute_features_multi(x, None, layers=[0, 7, 20])
        for layer, got in feats.items():
            for b in range(2):
                ref = self._ref(m, x[b], layer)
                assert torch.allclose(got[b], ref, atol=1e-4), (
                    f"layer {layer} sample {b}: "
                    f"{(got[b] - ref).abs().max().item()}"
                )

    @staticmethod
    def _ch0_moved(family, x, x2):
        """How far channel 0's slice of the embedding moves between two views."""
        m = _model(model=family, layer=6, tsfm_batch_size=18)
        _load_or_skip(m)
        d = m.d_model
        a = m._compute_features(x, None, None)[0, :d]
        b = m._compute_features(x2, None, None)[0, :d]
        return (a - b).abs().max().item()

    def test_variate_attention_mixes_channels(self):
        """The point of the family: a change in ONE channel must move the
        slice of the embedding that belongs to a DIFFERENT channel.

        TimesFM 2.5 reads the nine channels as nine independent univariate
        series, so the same perturbation leaves every other channel's slice
        bit-identical there. This is the test that would fail if the model
        were quietly being fed one variate at a time.

        The perturbation REVERSES channel 3 rather than shifting or scaling
        it — see the next test for why that distinction is the whole story.
        """
        x = _window(B=1, C=9, T=128)
        x2 = x.clone()
        x2[0, 3] = x2[0, 3].flip(-1)

        assert self._ch0_moved("timesfm", x, x2) == 0.0
        assert self._ch0_moved("timesfm3", x, x2) > 1e-2

    def test_cross_channel_amplitude_is_invisible(self):
        """What the multivariate feed does NOT carry, pinned deliberately.

        Each variate gets its own RevIN (and its own linear detrend), both
        fitted per series, so a channel reaches the transformer stripped of
        its level and its scale. Tripling channel 3 — a threefold change in
        that channel's amplitude relative to every other channel — moves
        channel 0's slice by ~1e-5 on a ~3e2 scale, i.e. by rounding. What
        crosses between variates is SHAPE, never relative magnitude.

        That matters for reading this family's results: a good deal of the
        cross-channel structure in a market window is relative magnitude
        (this stock's volume against its spread), and none of it survives
        the normalization. A null result here is not evidence that
        cross-channel structure is absent from the window.
        """
        x = _window(B=1, C=9, T=128)
        x2 = x.clone()
        x2[0, 3] = x2[0, 3] * 3.0
        assert self._ch0_moved("timesfm3", x, x2) < 1e-3

    def test_context_states_ignore_the_horizon(self):
        """Why ``_states_timesfm3`` can decode at horizon=1.

        Sequence attention is causal and variate attention acts within one
        patch position, so horizon patches cannot reach the context states.
        If a future release changed either, this pins the assumption rather
        than letting the readout drift into reading a longer forecast.
        """
        m = _model(model="timesfm3")
        _load_or_skip(m)
        model = m._tsfm(torch.device("cpu"))
        x = _window(B=1, C=9, T=128)
        mask = torch.zeros(1, 128, dtype=torch.bool)

        def states(horizon):
            caps = {}
            h = model.transformer_stack.layers[-1].register_forward_hook(
                lambda _m, _a, o: caps.__setitem__("s", o[0].detach()))
            try:
                with torch.no_grad():
                    model.decode(target=x, horizon=horizon, mask=mask)
            finally:
                h.remove()
            return caps["s"][:, :, : 128 // 32, :]

        assert torch.equal(states(1), states(64))


class TestChronos2Timestamps:
    def test_timestamps_change_embeddings(self):
        m_ts = _model(model="chronos2", layer=3, use_timestamps=True)
        _load_or_skip(m_ts)
        m_no = _model(model="chronos2", layer=3)
        x = _window(B=2, T=256)
        meta = {
            "tod_secs": torch.tensor([0, 12000]),
            "agg_factors": torch.tensor([1, 4]),
        }
        e_no = m_no.encode([x], metadata=meta)["embeddings"]
        e_ts = m_ts.encode([x], metadata=meta)["embeddings"]
        assert (e_no - e_ts).abs().mean() > 1e-5

    def test_tod_shift_changes_embeddings_only_with_timestamps(self):
        m_ts = _model(model="chronos2", layer=3, use_timestamps=True)
        _load_or_skip(m_ts)
        m_no = _model(model="chronos2", layer=3)
        x = _window(B=2, T=256)
        meta_a = {"tod_secs": torch.tensor([0, 0]), "agg_factors": torch.tensor([1, 1])}
        meta_b = {"tod_secs": torch.tensor([14400, 14400]), "agg_factors": torch.tensor([1, 1])}
        with_a = m_ts.encode([x], metadata=meta_a)["embeddings"]
        with_b = m_ts.encode([x], metadata=meta_b)["embeddings"]
        no_a = m_no.encode([x], metadata=meta_a)["embeddings"]
        no_b = m_no.encode([x], metadata=meta_b)["embeddings"]
        assert (with_a - with_b).abs().mean() > 1e-5
        assert torch.allclose(no_a, no_b)
