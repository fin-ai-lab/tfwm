"""Frozen pretrained time-series foundation models as a probe-scored mode.

The "did the learned representation beat an off-the-shelf TSFM?" reference
point. Public forecasting foundation models, used purely as frozen
feature extractors:

  ==========  ==================================  ========  ===========
  family      default checkpoint                  d_model   layers
  ==========  ==================================  ========  ===========
  timesfm     google/timesfm-2.5-200m-pytorch     1280      20 (causal)
  timesfm3    google/timesfm-3.0-pytorch          1280      20 (causal)
  sundial     thuml/sundial-base-128m             768       12 (causal)
  chronos2    amazon/chronos-2                    768       12 (bidir)
  kronos      NeoQuasar/Kronos-base               832       12 (causal)
  ==========  ==================================  ========  ===========

None of these can natively predict the probe targets, so — mirroring
:class:`FinanceBaseline` — nothing is trained (``num_epochs=0``) and
:meth:`encode` emits pooled hidden states that the existing probe-eval
pipeline scores at ``probe/ridge_ic_*``, the identical metric used for
every learned representation.

The sweep knob is ``layer``: which hidden state to pull. ``layer=0`` is the
patch-embedding output, ``layer=k`` the output of transformer block ``k``,
and the last layer additionally gets the model's final norm where the
architecture has one (Sundial, Chronos-2), matching each model's own readout.
Negative values index from the end (``-1`` = last).

Input mapping — each selected channel of the ``(B, C, T)`` view is fed as a
univariate series in the model's native format:

  - **timesfm**: series front-padded to a multiple of 32, patched, and
    normalized with the exact running-stats RevIN of the model's own
    ``decode()`` prefill; per-patch states pooled causally.
  - **timesfm3**: like Chronos-2 and unlike TimesFM 2.5, natively
    multivariate — the nine channels of one sample are the nine *variates*
    of a single ``decode()`` call, and every block's variate attention mixes
    them at each patch position (``d = 9 x d_model`` is still the readout,
    the same width as TimesFM 2.5, so the two are directly comparable).
    Going through the model's own ``decode()`` means the context reaches it
    exactly as at inference — linear detrending, causal running-stats RevIN,
    front-pad masks — at the smallest horizon the model accepts: sequence
    attention is causal and variate attention acts within one patch
    position, so the context states do not depend on the horizon at all
    (verified: max |diff| is exactly 0.0 between horizon 1 and 64).
    WHAT CROSSES BETWEEN THE VARIATES IS SHAPE, NOT MAGNITUDE: each variate
    carries its own RevIN and its own linear detrend, both fitted on that
    series alone, so a channel arrives stripped of its level and its scale.
    Tripling one channel's amplitude moves another channel's slice of the
    embedding by ~1e-5 on a ~3e2 scale; reversing it in time moves it by
    ~4. So relative magnitude across channels — this window's volume
    against its spread — never reaches the model, and a null result for
    this family is not evidence that the window lacks cross-channel
    structure (``tests/test_pretrained_tsfm.py`` pins both directions).
  - **sundial**: series grouped by valid length (its patch embedding cannot
    mask right-padding), normalized like the model's ``revin=True`` inference
    path (mean/std with the 1e-2 std floor).
  - **chronos2**: channels of one sample share a ``group_id``, so its group
    attention sees the window as a true multivariate task; NaN front-padding
    drives its native missing-data masks, and its internal instance norm
    handles scaling.
  - **kronos**: the exception to per-channel univariate rows — Kronos
    natively consumes multivariate OHLCVA K-line bars through a discrete
    tokenizer (``NeoQuasar/Kronos-Tokenizer-base``, BSQ), so
    ``d_embedding = d_model``. The 2048-token view is re-aggregated
    ``bar_agg``-fold (default 4x, filling its 512-bar context with the FULL
    window at coarser bars — Kronos trained on >= 1-min K-lines) using the
    dataset's own per-column aggregation (``_aggregate_numpy_jittered``:
    high=max, low=min, volume=sum). Coarse close is the LAST fine vwap of
    each bucket (candlestick close) because the aggregator's volume-weighted
    vwap is invalid on the view's z-scored log1p volumes; open is the
    previous close. The dataset's price normalization is one shared affine
    per window and Kronos's own preprocessing (per-column z-score + clip 5,
    replicated from KronosPredictor) is affine-invariant, so prices reach
    the model bit-for-bit as they would from raw levels; volume rides in
    log1p space (the dataset transform is not invertible per-sample).
    Upstream attention can't take a padding mask (attn_mask + is_causal
    collide in SDPA), so rare front-pad bars ride through as z=0 "flat"
    bars and validity is honored at pooling.

Timestamps (``use_timestamps``) — Chronos-2 only. Its high-level dataframe
API accepts a timestamp column but uses it purely for indexing; the model
never sees wall-clock time. The one native channel for injecting time is a
covariate series in the group, and its per-series instance norm erases affine
structure (a raw timestamp ramp normalizes to the same shape regardless of
date, time of day, or sampling rate — and a constant column to zero). So
``use_timestamps=True`` appends sin/cos time-of-day phase series built from
the sample's ``tod_secs``/``agg_factors`` metadata, which survive instance
norm. TimesFM 2.5 and Sundial expose no input channel that reaches the
encoder; TimesFM 3.0 does (``decode`` takes past-future covariates) but has no
such arm built. Requesting timestamps for any of them is an error rather than
a silent no-op.

The frozen TSFM is deliberately kept OUT of the nn.Module registry (held in a
plain dict): its weights never enter ``state_dict()``, the optimizer, or
``train()``/``eval()`` propagation — it is loaded lazily on first use and
pinned in eval mode. The ``backbone`` argument is required by the training
harness but unused, exactly as in :class:`FinanceBaseline`.
"""

from __future__ import annotations

import json
import logging
import math
import os

import numpy as np
import torch

from .base import TrainingModel

logger = logging.getLogger(__name__)

_FAMILIES: dict[str, dict] = {
    "timesfm": {
        "model_id": "google/timesfm-2.5-200m-pytorch",
        "d_model": 1280,
        "n_layers": 20,
        "patch": 32,
        "max_context": 16384,
        "causal": True,
    },
    "timesfm3": {
        "model_id": "google/timesfm-3.0-pytorch",
        "d_model": 1280,
        "n_layers": 20,
        "patch": 32,
        # timesfm3._MAX_CONTEXT_LENGTH, rounded up to a patch boundary.
        "max_context": 15360,
        "causal": True,
    },
    "sundial": {
        "model_id": "thuml/sundial-base-128m",
        "d_model": 768,
        "n_layers": 12,
        "patch": 16,
        "max_context": 2880,
        "causal": True,
    },
    "chronos2": {
        "model_id": "amazon/chronos-2",
        "d_model": 768,
        "n_layers": 12,
        "patch": 16,
        "max_context": 8192,
        "causal": False,
    },
    "kronos": {
        "model_id": "NeoQuasar/Kronos-base",
        "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base",
        "d_model": 832,
        "n_layers": 12,
        "patch": 1,  # one bar = one token
        "max_context": 512,
        "causal": True,
        "bar_agg": 4,
    },
}

# KronosPredictor.predict clip default: z-scored bars clamped to +/- this.
_KRONOS_CLIP = 5.0

_TIME_POOLS = ("mean", "last", "reg")
_CHANNEL_POOLS = ("concat", "mean")

# A "<family>_cmean" name is that family with its per-channel states AVERAGED
# (channel_pool="mean") instead of concatenated -- the readout the prediction
# evals use. It is a separate series name, not a flag, so the two readouts can
# never share a cache file or a result key.
CMEAN_SUFFIX = "_cmean"


def resolve_family(name: str) -> tuple[str, str]:
    """``(family, channel_pool)`` for a series family name (see CMEAN_SUFFIX)."""
    if name.endswith(CMEAN_SUFFIX):
        return name[: -len(CMEAN_SUFFIX)], "mean"
    return name, "concat"

_SECONDS_PER_DAY = 86400.0


class _StopForward(Exception):
    """Raised inside a capture hook to skip the layers above the target."""


class PretrainedTSFM(TrainingModel):
    """Frozen TSFM hidden states exposed as a probe-scored embedding bank.

    Args:
        backbone: Unused; stored for harness compatibility (may be ``None``
            when reconstructed via :meth:`from_pretrained`).
        model: TSFM family — ``"timesfm"``, ``"sundial"``, or ``"chronos2"``.
        model_id: HF repo override. Non-default checkpoints of a different
            width/depth must also override ``d_model``/``n_layers``.
        layer: Hidden state to extract (see module docstring). Negative
            indexes from the end; ``-1`` (default) = last layer.
        channels: Feature-channel indices to feed. ``None`` = all.
        channel_pool: ``"concat"`` (default) keeps per-channel embeddings
            side by side; ``"mean"`` averages them.
        time_pool: ``"mean"`` (default) over valid patches, ``"last"`` for
            the final context patch, ``"reg"`` for Chronos-2's [REG] token.
        use_timestamps: Chronos-2 only — append time-of-day sin/cos phase
            covariates to each sample's group (see module docstring).
        max_context: Cap on context steps (most recent kept). ``None`` =
            the family's trained context limit.
        tsfm_batch_size: Series rows per internal forward chunk. Do not
            raise it for chronos2: its group attention is dense across every
            row in the chunk, so larger chunks are *slower* (and quadratic
            in memory); 64 is near the A40 optimum for all three families.
        d_model: Width override for non-default checkpoints.
        n_layers: Depth override for non-default checkpoints.
    """

    mode_label: str = "TSFM"
    uses_multi_view: bool = False
    # Opt into the metadata channel in collect_probe_data: timestamp
    # covariates need each sample's tod_secs / agg_factors.
    wants_metadata: bool = True

    def __init__(
        self,
        backbone,
        model: str,
        model_id: str | None = None,
        layer: int = -1,
        channels: list[int] | None = None,
        channel_pool: str = "concat",
        time_pool: str = "mean",
        use_timestamps: bool = False,
        max_context: int | None = None,
        tsfm_batch_size: int = 64,
        d_model: int | None = None,
        n_layers: int | None = None,
        bar_agg: int | None = None,
        tokenizer_id: str | None = None,
    ):
        super().__init__()
        if model not in _FAMILIES:
            raise ValueError(
                f"Unknown TSFM family {model!r}; valid: {list(_FAMILIES)}"
            )
        fam = _FAMILIES[model]
        self.family = model
        self.model_id = model_id or fam["model_id"]
        self.d_model = d_model or fam["d_model"]
        self.n_layers = n_layers or fam["n_layers"]
        self.patch = fam["patch"]
        self.causal = fam["causal"]
        self.max_context = min(max_context or fam["max_context"], fam["max_context"])
        if model != "kronos" and (bar_agg is not None or tokenizer_id is not None):
            raise ValueError(
                "bar_agg / tokenizer_id are only meaningful for kronos "
                "(the one family with a bar re-aggregation and a separate "
                f"tokenizer checkpoint), got them for {model!r}"
            )
        self.bar_agg = int(bar_agg) if bar_agg is not None else fam.get("bar_agg", 1)
        if self.bar_agg < 1:
            raise ValueError(f"bar_agg must be >= 1, got {self.bar_agg}")
        self.tokenizer_id = tokenizer_id or fam.get("tokenizer_id")

        # Hidden states are indexed 0..n_layers (0 = patch embedding output).
        n_states = self.n_layers + 1
        if not (-n_states <= layer <= self.n_layers):
            raise ValueError(
                f"layer must be in [-{n_states}, {self.n_layers}] for {model}, got {layer}"
            )
        self.layer = layer % n_states if layer < 0 else layer
        # Eval-time override (not config): when set, every forward captures
        # states at ALL these depths in one pass — see compute_features_multi.
        self.capture_layers: tuple[int, ...] | None = None

        if channel_pool not in _CHANNEL_POOLS:
            raise ValueError(f"channel_pool must be one of {_CHANNEL_POOLS}, got {channel_pool!r}")
        if time_pool not in _TIME_POOLS:
            raise ValueError(f"time_pool must be one of {_TIME_POOLS}, got {time_pool!r}")
        if time_pool == "reg" and model != "chronos2":
            raise ValueError("time_pool='reg' is only meaningful for chronos2 ([REG] token)")
        if use_timestamps and model != "chronos2":
            raise ValueError(
                "use_timestamps is only supported for chronos2 (covariate group "
                "attention); timesfm and sundial have no input channel for it, "
                "kronos's native temporal embeddings need weekday/day/month "
                "the batch metadata does not carry, and timesfm3 — which does "
                "have a past-future covariate channel that could carry them — "
                "has no such arm implemented"
            )
        self.channel_pool = channel_pool
        self.time_pool = time_pool
        self.use_timestamps = use_timestamps
        self.tsfm_batch_size = tsfm_batch_size

        self.backbone = backbone
        n_features = getattr(backbone, "n_features", None)
        if channels is None and n_features is None:
            raise ValueError(
                "channels must be given explicitly when the backbone does not "
                "expose n_features"
            )
        self.channels = list(channels) if channels is not None else list(range(n_features))
        if not self.channels:
            raise ValueError("channels must select at least one feature channel")
        if model == "kronos" and self.channels != list(range(9)):
            # The bar aggregator's per-column semantics are positional over
            # the full FEATURE_COLUMNS layout — a subset cannot be honored.
            raise ValueError(
                "kronos consumes the full 9-channel FEATURE_COLUMNS layout "
                f"(channels=[0..8]); got {self.channels}"
            )

        n_ch = len(self.channels)
        if model == "kronos":
            # One multivariate OHLCVA row per sample — no per-channel concat.
            self.d_embedding = self.d_model
        else:
            self.d_embedding = self.d_model * (n_ch if channel_pool == "concat" else 1)

        # Holds the frozen TSFM without registering it as a submodule.
        self._tsfm_holder: dict[str, object] = {}
        self._warned_no_metadata = False

    @property
    def mode_str(self) -> str:
        return f"pretrained_tsfm: {self.family} layer {self.layer}"

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _tsfm(self, device: torch.device):
        m = self._tsfm_holder.get("m")
        if m is None:
            logger.info("Loading frozen TSFM %s (%s)...", self.family, self.model_id)
            if self.family == "timesfm":
                import timesfm

                wrapper = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                    self.model_id, torch_compile=False
                )
                m = wrapper.model
            elif self.family == "timesfm3":
                # The 3.0 weights ship as their own top-level package; the
                # `timesfm` one above still holds the 2.5 API, so both
                # families load from a single `timesfm>=3` install.
                from timesfm3 import TimesFM3Torch

                m = TimesFM3Torch.from_pretrained(self.model_id)
            elif self.family == "sundial":
                from transformers import AutoModelForCausalLM

                m = AutoModelForCausalLM.from_pretrained(
                    self.model_id, trust_remote_code=True
                )
            elif self.family == "kronos":
                from .kronos_vendor import Kronos, KronosTokenizer

                m = Kronos.from_pretrained(self.model_id)
                tok = KronosTokenizer.from_pretrained(self.tokenizer_id)
                tok = tok.float().eval()
                for p in tok.parameters():
                    p.requires_grad_(False)
                self._tsfm_holder["tok"] = tok
            else:  # chronos2
                from chronos import Chronos2Pipeline

                m = Chronos2Pipeline.from_pretrained(self.model_id, device_map="cpu").model
            m = m.float().eval()
            for p in m.parameters():
                p.requires_grad_(False)
            self._tsfm_holder["m"] = m
        if next(m.parameters()).device != device:
            m.to(device)
            if self.family == "timesfm":
                # Its module caches a device attribute used by decode(); keep
                # it coherent even though we only call forward().
                m.device = device
            if self.family == "kronos":
                self._tsfm_holder["tok"].to(device)
        return m

    # ------------------------------------------------------------------
    # Row preparation
    # ------------------------------------------------------------------

    def _tail_align(
        self, series: torch.Tensor, lengths: torch.Tensor, pad_value: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move each row's valid prefix to the row's tail, front-padded.

        Args:
            series: ``(N, T)`` rows, valid data left-aligned (right-padded).
            lengths: ``(N,)`` valid lengths.
            pad_value: Fill for the front padding.

        Returns:
            (out, pad) where out is ``(N, C)`` with each row's last
            ``min(length, C)`` valid values right-aligned, ``C`` a multiple of
            the family patch size capped at ``max_context``, and pad is a
            ``(N, C)`` bool mask (True = padding).
        """
        N, T = series.shape
        device = series.device
        lengths = lengths.clamp(min=1, max=T)
        max_len = int(lengths.max().item())
        C = min(self.max_context, math.ceil(max_len / self.patch) * self.patch)

        # Column j of the output maps to input index length - C + j; negative
        # index = front padding.
        j = torch.arange(C, device=device)[None, :]
        src = lengths[:, None] - C + j
        pad = src < 0
        gathered = series.gather(1, src.clamp(min=0))
        out = torch.where(pad, torch.full_like(gathered, pad_value), gathered)
        return out, pad

    def _timestamp_rows(
        self,
        metadata: dict | None,
        lengths: torch.Tensor,
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Per-sample time-of-day sin/cos series, ``(B, 2, T)`` left-aligned.

        ``tod_secs`` is the second-of-session at view start and
        ``agg_factors`` the seconds per token, so token ``j`` of sample ``b``
        happens at ``tod_secs[b] + j * agg[b]`` — converted to a 24h phase.
        """
        B = lengths.shape[0]
        tod = metadata.get("tod_secs") if metadata else None
        agg = metadata.get("agg_factors") if metadata else None
        if tod is None or agg is None:
            if not self._warned_no_metadata:
                logger.warning(
                    "use_timestamps=True but the batch carries no tod_secs/"
                    "agg_factors metadata; falling back to a 1 Hz grid from 0"
                )
                self._warned_no_metadata = True
            tod = torch.zeros(B)
            agg = torch.ones(B)
        tod = tod.to(device).float()[:, None]
        agg = agg.to(device).float().clamp(min=1.0)[:, None]
        seconds = tod + torch.arange(T, device=device).float()[None, :] * agg
        phase = seconds * (2.0 * math.pi / _SECONDS_PER_DAY)
        return torch.stack([phase.sin(), phase.cos()], dim=1)

    # ------------------------------------------------------------------
    # Per-family hidden-state extraction
    # ------------------------------------------------------------------

    def _layers_wanted(self) -> tuple[int, ...]:
        return tuple(self.capture_layers) if self.capture_layers else (self.layer,)

    def _hook_specs(self, blocks, last_unpack, mid_unpack=None):
        """{layer: (module, unpack)} for every wanted layer.

        ``None`` module means "capture the input of blocks[0]" (layer 0 =
        embeddings); ``last_unpack`` mirrors whatever output_hidden_states
        would apply after the last block (final norm where one exists).
        ``mid_unpack`` extracts the states from an intermediate block's
        output — default assumes HF-style tuples; kronos blocks return
        plain tensors.
        """
        if mid_unpack is None:
            mid_unpack = lambda o: o[0]  # noqa: E731
        specs: dict[int, tuple] = {}
        for L in self._layers_wanted():
            if L == 0:
                specs[L] = (None, None)
            elif L < self.n_layers:
                specs[L] = (blocks[L - 1], mid_unpack)
            else:
                specs[L] = (blocks[-1], last_unpack)
        return specs

    def _capture(self, blocks, run, specs):
        """Run ``run()`` once, capturing states at every depth in ``specs``
        and early-exiting above the deepest requested layer.

        Blocks execute in order, so shallower hooks have always fired by the
        time the deepest one raises the stop.

        Args:
            blocks: Full block list.
            run: Zero-arg closure running the forward pass.
            specs: ``{layer: (module_or_None, unpack)}`` from _hook_specs.

        Returns:
            ``{layer: hidden states tensor}`` for every requested layer.
        """
        mods = [id(m) for m, _ in specs.values() if m is not None]
        if len(mods) != len(set(mods)):
            raise ValueError(
                f"capture layers {sorted(specs)} collide on one module "
                f"(n_layers={self.n_layers}); request distinct depths"
            )
        captured: dict[int, torch.Tensor] = {}
        deepest = max(specs)
        handles = []

        def grab(L, value):
            captured[L] = value
            if L == deepest:
                raise _StopForward

        for L, (mod, unpack) in specs.items():
            if mod is None:
                def pre_hook(_m, args, kwargs=None, L=L):
                    grab(L, args[0])
                handles.append(blocks[0].register_forward_pre_hook(pre_hook))
            else:
                def hook(_m, _args, output, L=L, unpack=unpack):
                    grab(L, unpack(output))
                handles.append(mod.register_forward_hook(hook))

        try:
            run()
        except _StopForward:
            pass
        finally:
            for h in handles:
                h.remove()
        missing = set(specs) - set(captured)
        if missing:
            raise RuntimeError(
                f"{self.family} forward finished without reaching capture "
                f"hooks for layers {sorted(missing)}"
            )
        return captured

    def _states_timesfm(self, rows: torch.Tensor, pad: torch.Tensor):
        """(N, C) tail-aligned rows -> per-patch states + patch validity."""
        from timesfm.torch import util as tfm_util

        core = self._tsfm(rows.device)
        N, C = rows.shape
        P = C // self.patch
        patched = rows.reshape(N, P, self.patch)
        masks = pad.reshape(N, P, self.patch)

        # Exact replica of the model's decode() prefill normalization:
        # causal running-stats RevIN, masked positions zeroed.
        n = torch.zeros(N, device=rows.device)
        mu = torch.zeros(N, device=rows.device)
        sigma = torch.zeros(N, device=rows.device)
        patch_mu, patch_sigma = [], []
        for i in range(P):
            (n, mu, sigma), _ = tfm_util.update_running_stats(
                n, mu, sigma, patched[:, i], masks[:, i]
            )
            patch_mu.append(mu)
            patch_sigma.append(sigma)
        context_mu = torch.stack(patch_mu, dim=1)
        context_sigma = torch.stack(patch_sigma, dim=1)
        normed = tfm_util.revin(patched, context_mu, context_sigma, reverse=False)
        normed = torch.where(masks, torch.zeros_like(normed), normed)
        normed = torch.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)

        # Layer 0 = tokenizer output, captured as the input of block 0.
        # TimesFM has no final norm, so the last layer unpacks like the rest.
        blocks = core.stacked_xf
        by_layer = self._capture(
            blocks,
            run=lambda: core(normed, masks),
            specs=self._hook_specs(blocks, last_unpack=lambda o: o[0]),
        )
        # A patch enters attention iff the last step is unmasked — mirror it.
        patch_valid = ~masks[..., -1]
        return by_layer, patch_valid, None

    def _states_timesfm3(
        self, rows: torch.Tensor, pad: torch.Tensor, n_samples: int
    ):
        """A sample's channel rows as the VARIATES of one decode call.

        Args:
            rows: ``(N, C)`` tail-aligned rows, ``N = n_samples * n_ch``,
                ordered sample-major.
            pad: ``(N, C)`` bool, True = front padding.
            n_samples: How many samples the rows belong to.

        Returns:
            ``({layer: (N, P, d_model)}, (N, P) patch validity)`` — the
            variate axis is flattened back into the row axis in the same
            sample-major order the caller passed in.
        """
        model = self._tsfm(rows.device)
        n_ch = len(self.channels)
        N, C = rows.shape
        pad3 = pad.view(n_samples, n_ch, C)
        if bool((pad3 != pad3[:, :1, :]).any()):
            raise RuntimeError(
                "timesfm3 feeds a sample's channels as one multivariate group, "
                "which takes ONE (b, context) mask — but the channel rows of a "
                "sample disagree on where the padding is"
            )

        blocks = model.transformer_stack.layers
        by_layer = self._capture(
            blocks,
            # horizon=1 is the smallest decode() accepts. The horizon patches
            # sit after the context in the sequence, sequence attention is
            # causal and variate attention acts within one patch position, so
            # the context states are horizon-independent (max |diff| 0.0
            # between horizon 1 and 64) and this is the cheapest faithful pass.
            run=lambda: model.decode(
                target=rows.view(n_samples, n_ch, C),
                horizon=1,
                mask=pad3[:, 0, :],
            ),
            # No final norm after the stack (the output head is a bare
            # Linear), so the last layer unpacks like every other one.
            specs=self._hook_specs(blocks, last_unpack=lambda o: o[0]),
        )

        P = C // self.patch
        # (b, v, n, d) -> (b*v, P, d): the leading two axes flatten to exactly
        # the sample-major row order, and the trailing horizon patches drop.
        states = {
            L: s[:, :, :P, :].reshape(N, P, self.d_model)
            for L, s in by_layer.items()
        }
        # A patch enters attention iff its last step is unmasked, as for 2.5.
        patch_valid = ~pad.view(N, P, self.patch)[..., -1]
        return states, patch_valid

    def _states_sundial(self, rows: torch.Tensor, pad: torch.Tensor):
        """Group rows by valid length; replicate the revin=True inference path."""
        outer = self._tsfm(rows.device)
        inner = outer.model
        N, C = rows.shape
        valid = (~pad).sum(dim=1)

        d = self.d_model
        pooled_parts: dict[int, list[torch.Tensor]] = {}
        order: list[torch.Tensor] = []
        for length in valid.unique().tolist():
            idx = (valid == length).nonzero(as_tuple=True)[0]
            x = rows[idx, C - int(length):]
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True, unbiased=False)
            std = torch.where(std > 1e-2, std, torch.ones_like(std))
            x = (x - mean) / std
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            blocks = inner.layers
            by_layer = self._capture(
                blocks,
                run=lambda x=x: inner(input_ids=x, use_cache=False),
                # Last layer gets the final norm, matching output_hidden_states.
                specs=self._hook_specs(
                    blocks, last_unpack=lambda o: inner.norm(o[0]),
                ),
            )
            for L, states in by_layer.items():
                pooled_parts.setdefault(L, []).append(
                    self._pool_time(states, None, None)
                )
            order.append(idx)

        idx_all = torch.cat(order)
        out: dict[int, torch.Tensor] = {}
        for L, parts in pooled_parts.items():
            o = torch.empty(N, d, device=rows.device, dtype=parts[0].dtype)
            o[idx_all] = torch.cat(parts, dim=0)
            out[L] = o
        return out  # {layer: (N, d)}, already pooled — see _encode_rows

    def _states_chronos2(
        self, rows: torch.Tensor, pad: torch.Tensor, group_ids: torch.Tensor
    ):
        """NaN-padded rows through Chronos2Model.encode with group attention."""
        model = self._tsfm(rows.device)
        ctx = torch.where(pad, torch.full_like(rows, float("nan")), rows)

        blocks = model.encoder.block

        def run():
            model.encode(context=ctx, group_ids=group_ids, num_output_patches=1)

        by_layer = self._capture(
            blocks,
            run=run,
            # encoder output applies final_layer_norm — mirror it on the last
            # block's states rather than hooking (dropout is inert in eval).
            specs=self._hook_specs(
                blocks,
                last_unpack=lambda o: model.encoder.final_layer_norm(o[0]),
            ),
        )

        P = ctx.shape[1] // self.patch
        patch_valid = ~pad.reshape(pad.shape[0], P, self.patch).all(dim=-1)
        # Sequence layout: [context patches (P), REG, future patch].
        states_by = {L: s[:, :P, :] for L, s in by_layer.items()}
        reg_by = {L: s[:, P, :] for L, s in by_layer.items()}
        return states_by, patch_valid, reg_by

    def _kronos_bars(
        self, view: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, C, T) normalized view -> z-scored OHLCVA bars for the tokenizer.

        Coarsening reuses the dataset's own per-column aggregation
        (``_aggregate_numpy_jittered``: high=max, low=min, volume=sum,
        quotes/sizes=last) at ``bar_agg`` so the bar semantics have a single
        source of truth. Coarse close deviates deliberately: the aggregator's
        volume-weighted vwap is invalid on the view's z-scored log1p volumes
        (weight sums can be ~0 or negative), so close = the LAST fine vwap of
        each bucket — proper candlestick close, and it commutes with the
        dataset's shared price affine. Open = previous close (first bar: its
        bucket's first fine vwap); amount = volume * mean(OHLC), mirroring
        KronosPredictor's own fallback. Normalization replicates
        ``KronosPredictor.predict``: per-sample per-column z-score + clip 5.

        Returns:
            (bars, bar_valid): ``(B, P, 6)`` float32 front-padded bars and a
            ``(B, P)`` validity mask (False = front padding).
        """
        from market_jepa.augmentations import _aggregate_numpy_jittered
        from market_jepa.training.streaming_dataset import FEATURE_COLUMNS

        i_vwap = FEATURE_COLUMNS.index("vwap_all")
        i_high = FEATURE_COLUMNS.index("high")
        i_low = FEATURE_COLUMNS.index("low")
        i_vol = FEATURE_COLUMNS.index("volume")

        A = self.bar_agg
        B = view.shape[0]
        arr = view.permute(0, 2, 1).cpu().numpy().astype(np.float64)  # (B, T, C)
        lens = lengths.cpu().numpy().astype(int)

        per_sample: list[np.ndarray] = []
        for b in range(B):
            n = max(int(lens[b]), 1)
            f = arr[b, :n]
            agg = _aggregate_numpy_jittered(f, A)
            if agg is None:  # < 2 buckets: collapse the prefix into one bar
                agg = f[-1:, :].copy()
                agg[0, i_high] = f[:, i_high].max()
                agg[0, i_low] = f[:, i_low].min()
                agg[0, i_vol] = f[:, i_vol].sum()
            P = agg.shape[0]
            last_idx = np.minimum((np.arange(P) + 1) * A - 1, n - 1)
            close = f[last_idx, i_vwap]
            opn = np.empty_like(close)
            opn[1:] = close[:-1]
            opn[0] = f[0, i_vwap]
            high = agg[:, i_high]
            low = agg[:, i_low]
            vol = agg[:, i_vol]
            amount = vol * (opn + high + low + close) / 4.0
            bars = np.stack([opn, high, low, close, vol, amount], axis=-1)
            if P > self.max_context:
                bars = bars[-self.max_context :]
            mu = bars.mean(axis=0)
            sd = bars.std(axis=0)
            bars = np.clip((bars - mu) / (sd + 1e-5), -_KRONOS_CLIP, _KRONOS_CLIP)
            per_sample.append(bars)

        p_max = max(b.shape[0] for b in per_sample)
        out = np.zeros((B, p_max, 6), dtype=np.float32)
        valid = np.zeros((B, p_max), dtype=bool)
        for b, bars in enumerate(per_sample):
            P = bars.shape[0]
            out[b, p_max - P :] = bars  # front-padded: most recent at the tail
            valid[b, p_max - P :] = True
        return torch.from_numpy(out), torch.from_numpy(valid)

    def _states_kronos(self, bars: torch.Tensor) -> dict[int, torch.Tensor]:
        """(N, P, 6) z-scored bars -> {layer: (N, P, d)} hidden states."""
        model = self._tsfm(bars.device)
        tok = self._tsfm_holder["tok"]
        s1_ids, s2_ids = tok.encode(bars, half=True)

        blocks = model.transformer
        return self._capture(
            blocks,
            # No padding mask: upstream attention passes attn_mask together
            # with is_causal=True, which SDPA rejects — front-pad bars ride
            # through as z=0 "flat" bars; pooling honors validity instead.
            run=lambda: model.decode_s1(s1_ids, s2_ids, stamp=None),
            specs=self._hook_specs(
                blocks,
                # Layer n gets the final RMSNorm, matching decode_s1's output.
                last_unpack=lambda o: model.norm(o),
                mid_unpack=lambda o: o,  # kronos blocks return plain tensors
            ),
        )

    def _compute_features_kronos(
        self, view: torch.Tensor, lengths: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """One ``(B, C, T)`` view -> ``{layer: (B, d_model)}`` via OHLCVA bars."""
        device = view.device
        bars, bar_valid = self._kronos_bars(view, lengths)
        B = bars.shape[0]

        chunks: dict[int, list[torch.Tensor]] = {}
        step = self.tsfm_batch_size
        with torch.no_grad():
            for s in range(0, B, step):
                e = min(B, s + step)
                x = bars[s:e].to(device)
                v = bar_valid[s:e].to(device)
                by_layer = self._states_kronos(x)
                for L, states in by_layer.items():
                    chunks.setdefault(L, []).append(self._pool_time(states, v, None))

        return {
            L: torch.nan_to_num(
                torch.cat(parts, dim=0).float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            for L, parts in chunks.items()
        }

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------

    def _pool_time(
        self,
        states: torch.Tensor,
        patch_valid: torch.Tensor | None,
        reg: torch.Tensor | None,
    ) -> torch.Tensor:
        """(N, P, d) -> (N, d) according to time_pool."""
        if self.time_pool == "reg":
            if reg is None:
                raise RuntimeError("reg pooling requested but no [REG] state captured")
            return reg
        if self.time_pool == "last":
            return states[:, -1, :]
        if patch_valid is None:
            return states.mean(dim=1)
        w = patch_valid.to(states.dtype)[..., None]
        return (states * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)

    def _encode_rows(
        self,
        rows: torch.Tensor,
        row_lengths: torch.Tensor,
        ts_rows: torch.Tensor | None,
        n_samples: int,
    ) -> dict[int, torch.Tensor]:
        """Univariate rows -> pooled per-row embeddings ``{layer: (N, d_model)}``
        for every wanted layer (one entry when capture_layers is unset).

        ``rows`` is ``(N, T)`` left-aligned with ``N = n_samples * n_ch``,
        ordered sample-major. ``ts_rows`` is ``(n_samples, 2, T)`` timestamp
        covariates (chronos2 only).
        """
        if self.family == "sundial":
            out, pad = self._tail_align(rows, row_lengths, pad_value=0.0)
            return self._states_sundial(out, pad)

        if self.family == "timesfm":
            out, pad = self._tail_align(rows, row_lengths, pad_value=0.0)
            by_layer, patch_valid, _reg = self._states_timesfm(out, pad)
            return {
                L: self._pool_time(s, patch_valid, None)
                for L, s in by_layer.items()
            }

        if self.family == "timesfm3":
            out, pad = self._tail_align(rows, row_lengths, pad_value=0.0)
            by_layer, patch_valid = self._states_timesfm3(out, pad, n_samples)
            return {
                L: self._pool_time(s, patch_valid, None)
                for L, s in by_layer.items()
            }

        # chronos2: build the group — channel rows plus optional timestamp
        # covariate rows sharing each sample's group id.
        n_ch = len(self.channels)
        out, pad = self._tail_align(rows, row_lengths, pad_value=0.0)
        gids = torch.arange(n_samples, device=rows.device).repeat_interleave(n_ch)
        if ts_rows is not None:
            T = rows.shape[1]
            flat_ts = ts_rows.reshape(n_samples * 2, T)
            ts_lengths = row_lengths.reshape(n_samples, n_ch)[:, 0].repeat_interleave(2)
            ts_out, ts_pad = self._tail_align(flat_ts, ts_lengths, pad_value=0.0)
            out = torch.cat([out, ts_out], dim=0)
            pad = torch.cat([pad, ts_pad], dim=0)
            gids = torch.cat(
                [gids, torch.arange(n_samples, device=rows.device).repeat_interleave(2)]
            )
        states_by, patch_valid, reg_by = self._states_chronos2(out, pad, gids)
        n_rows = n_samples * n_ch
        return {
            L: self._pool_time(
                states_by[L][:n_rows],
                patch_valid[:n_rows],
                reg_by[L][:n_rows] if reg_by is not None else None,
            )
            for L in states_by
        }

    def _compute_features_all(
        self,
        view: torch.Tensor,
        lengths: torch.Tensor | None,
        metadata: dict | None,
    ) -> dict[int, torch.Tensor]:
        """One ``(B, C, T)`` view -> ``{layer: (B, d_embedding)}`` for every
        wanted layer, from a single forward pass per chunk."""
        B, _C, T = view.shape
        device = view.device
        n_ch = len(self.channels)
        if lengths is None:
            lengths = torch.full((B,), T, device=device, dtype=torch.long)
        else:
            lengths = lengths.to(device).long().clamp(min=1, max=T)

        if self.family == "kronos":
            # SELECT THE CHANNELS HERE TOO. Every other family narrows on the
            # next line; kronos returned before it and so was handed whatever
            # width the caller had. That is fine on a bare 9-channel panel and
            # wrong on an information-token grid (9 data + 11 trailing
            # constants), where _kronos_bars indexes columns by their
            # FEATURE_COLUMNS position and aggregate() rejects the width
            # outright -- "features do not match the supplied schema". It only
            # surfaced when a run first mixed info-bearing encoders with the
            # frozen TSFMs on ONE grid.
            #
            # self.channels is [0..8] for kronos -- the constructor refuses
            # anything else, since _kronos_bars reads vwap_all/high/low/volume
            # by FEATURE_COLUMNS position -- so this is exactly "take the nine
            # data channels and drop any trailing info columns".
            return self._compute_features_kronos(
                view[:, self.channels, :], lengths)

        x = view[:, self.channels, :].float()  # (B, n_ch, T)
        ts_rows = (
            self._timestamp_rows(metadata, lengths, T, device)
            if (self.use_timestamps and self.family == "chronos2")
            else None
        )

        chunks: dict[int, list[torch.Tensor]] = {}
        step = max(1, self.tsfm_batch_size // n_ch)
        with torch.no_grad():
            for s in range(0, B, step):
                e = min(B, s + step)
                rows = x[s:e].reshape((e - s) * n_ch, T)
                row_lengths = lengths[s:e].repeat_interleave(n_ch)
                pooled_by = self._encode_rows(
                    rows,
                    row_lengths,
                    ts_rows[s:e] if ts_rows is not None else None,
                    n_samples=e - s,
                )
                for L, pooled in pooled_by.items():
                    chunks.setdefault(L, []).append(
                        pooled.reshape(e - s, n_ch, self.d_model)
                    )

        out: dict[int, torch.Tensor] = {}
        for L, parts in chunks.items():
            emb = torch.cat(parts, dim=0)
            if self.channel_pool == "mean":
                emb = emb.mean(dim=1)
            else:
                emb = emb.reshape(B, n_ch * self.d_model)
            # The probe's LogisticRegression rejects NaN/inf; scrub so one
            # degenerate window can never poison the whole probe fit.
            out[L] = torch.nan_to_num(
                emb.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
        return out

    def _compute_features(
        self,
        view: torch.Tensor,
        lengths: torch.Tensor | None,
        metadata: dict | None,
    ) -> torch.Tensor:
        """One ``(B, C, T)`` view -> ``(B, d_embedding)`` pooled TSFM states."""
        return self._compute_features_all(view, lengths, metadata)[self.layer]

    @torch.no_grad()
    def compute_features_multi(
        self,
        view: torch.Tensor,
        lengths: torch.Tensor | None,
        layers: list[int],
        metadata: dict | None = None,
    ) -> dict[int, torch.Tensor]:
        """Features at several depths from ONE forward pass per chunk.

        A single run to ``max(layers)`` with passive capture hooks at the
        shallower depths — the single-pass equivalent of ``len(layers)``
        per-layer instances (each of which would early-exit at its own
        depth, re-paying tokenization and every shared block).
        """
        bad = [L for L in layers if not (0 <= L <= self.n_layers)]
        if bad:
            raise ValueError(f"layers {bad} outside 0..{self.n_layers}")
        prev = self.capture_layers
        self.capture_layers = tuple(sorted(set(layers)))
        try:
            return self._compute_features_all(view, lengths, metadata)
        finally:
            self.capture_layers = prev

    # ------------------------------------------------------------------
    # TrainingModel interface
    # ------------------------------------------------------------------

    def encode(self, x, lengths=None, metadata=None) -> dict[str, torch.Tensor]:
        """Return ``{"embeddings": (B, n_views, d_embedding)}``.

        Mirrors :meth:`TrainingModel.encode`'s input handling so the probe's
        ``collect_probe_data`` (which passes a list of views) works unchanged.
        """
        self._validate_instance_attrs()
        if isinstance(x, torch.Tensor) and x.dim() == 4:
            views = [x[:, i, :, :] for i in range(x.shape[1])]
            view_lengths = (
                [lengths] * x.shape[1] if lengths is not None else [None] * x.shape[1]
            )
        elif isinstance(x, list):
            views = x
            view_lengths = lengths if lengths is not None else [None] * len(views)
        else:
            views = [x]
            view_lengths = [lengths]

        feats = [
            self._compute_features(v, vl, metadata)
            for v, vl in zip(views, view_lengths)
        ]
        return {"embeddings": torch.stack(feats, dim=1)}

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the ``(B, d)`` feature bank for a single view (interface req.)."""
        return self._compute_features(x, lengths, None)

    def training_step(self, batch, device, grad_accum_steps: int = 1):
        # Nothing to train. Returning None makes the harness advance the step
        # counter without an optimizer step (see pretrain.py training loop).
        return None

    @torch.no_grad()
    def eval_step(self, eval_batches, device) -> dict[str, float]:
        """Cheap sanity diagnostic — the real numbers come from the probe.

        Reports the cross-sample std of the embedding dims (averaged), so a
        collapsed or constant bank is visible at a glance.
        """
        self.eval()
        collected: list[torch.Tensor] = []
        for batch in eval_batches:
            for bucket in batch["buckets"]:
                if bucket.get("n_global_views", 0) < 1:
                    continue
                v = bucket["views"][0].to(device)
                l = bucket["lengths"][0].to(device)
                collected.append(self._compute_features(v, l, bucket).cpu())
        if not collected:
            return {"eval/tsfm_feature_std": float("nan")}
        feats = torch.cat(collected, dim=0)
        return {
            "eval/tsfm_feature_std": float(feats.std(dim=0).mean()),
            "eval/tsfm_n": float(feats.shape[0]),
        }

    def post_training_step(self, completed_steps, max_train_steps):
        return {}

    def describe_parameters(self):
        from market_jepa.training.utils import count_parameters

        param_counts = count_parameters(self)
        summary = (
            f"Pretrained TSFM (frozen, no trainable objective):\n"
            f"  Family: {self.family} ({self.model_id})\n"
            f"  Layer: {self.layer} of 0..{self.n_layers} "
            f"(0 = patch embedding)\n"
            f"  Channels: {self.channels} -> {self.channel_pool} | "
            f"time_pool: {self.time_pool} | timestamps: {self.use_timestamps}\n"
            f"  Max context: {self.max_context} steps | d_embedding: {self.d_embedding}\n"
            f"  Scored by the probe -> probe/ridge_ic_<target>_<horizon>"
        )
        return param_counts, summary

    def default_run_name(self, backbone_type, cfg):
        parts = [
            "mode=tsfm",
            f"model={self.family}",
            f"layer={self.layer}",
            f"tp={self.time_pool}",
        ]
        if self.use_timestamps:
            parts.append("ts=1")
        return "__".join(parts)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_pretrained(self, path: str) -> None:
        """Persist the extraction config; the TSFM weights stay on the Hub."""
        os.makedirs(path, exist_ok=True)
        config = {
            "class": "PretrainedTSFM",
            "model": self.family,
            "model_id": self.model_id,
            "layer": self.layer,
            "channels": self.channels,
            "channel_pool": self.channel_pool,
            "time_pool": self.time_pool,
            "use_timestamps": self.use_timestamps,
            "max_context": self.max_context,
            "tsfm_batch_size": self.tsfm_batch_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
        }
        if self.family == "kronos":
            config["bar_agg"] = self.bar_agg
            config["tokenizer_id"] = self.tokenizer_id
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "PretrainedTSFM":
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        config.pop("class", None)
        config.update(kwargs)
        return cls(backbone=config.pop("backbone", None), **config)
