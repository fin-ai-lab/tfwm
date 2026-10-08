"""Event-conditioned predictor: a pretrained task head plus a text-embedding branch.

The market pathway is the head trained by the supervised / SSL-head-finetune
runs, reused verbatim. The event text enters through a small bottleneck MLP
whose output is summed into the head's hidden activation, before the norm::

    logits = mlp[1:]( mlp[0](cls) + g(emb) )

``mlp[0]`` is frozen — it holds 98.5% of the head's parameters and was fitted on
a full month of market windows, far more data than the event set. Only the norm,
the output layer, and ``g`` train.

When ``g``'s output layer is zero-initialised the module is bit-identical to the
pretrained head at step 0, so the market-only baseline is exactly the point every
event-conditioned run starts from.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EventEmbeddingBranch(nn.Module):
    """``emb_dim -> bottleneck -> hidden``, optionally zero-init at the output.

    The bottleneck is what makes this trainable on a few hundred events: a dense
    ``emb_dim -> hidden`` map would be ~2.4M parameters at emb_dim=3072, against
    order-1e2 training rows.
    """

    def __init__(
        self,
        emb_dim: int,
        hidden: int,
        bottleneck: int = 16,
        *,
        zero_init: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, bottleneck)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(bottleneck, hidden)

        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        if zero_init:
            # Start as an exact no-op so the module reduces to the pretrained head.
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        else:
            nn.init.trunc_normal_(self.fc2.weight, std=0.02)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(emb)))


class PositionWeightedEmbeddingBranch(nn.Module):
    """Learn one softmax weight per chunk POSITION, then the same bottleneck.

    The document is always split into ``n_chunks`` chunks in reading order, so a
    single vector of ``n_chunks`` learnable logits — softmaxed to a distribution
    shared across every event — lets the model up-weight, e.g., the opening and
    closing of a filing and discount the boilerplate middle. That is only
    ``n_chunks`` parameters and is content-agnostic (it cannot pick a specific
    event's material chunk, only learn where signal tends to sit by position).

    Logits start at zero, so the pool is exactly the plain mean at step 0 and
    training departs from the mean-pool baseline only if position carries signal.
    ``fc2`` zero-init additionally keeps the whole branch a no-op at step 0, so
    warm-start identity with the pretrained head holds.
    """

    def __init__(
        self,
        emb_dim: int,
        hidden: int,
        bottleneck: int = 16,
        n_chunks: int = 64,
        *,
        zero_init: bool = True,
    ):
        super().__init__()
        self.chunk_logits = nn.Parameter(torch.zeros(n_chunks))  # uniform == mean
        self.fc1 = nn.Linear(emb_dim, bottleneck)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(bottleneck, hidden)

        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        if zero_init:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        else:
            nn.init.trunc_normal_(self.fc2.weight, std=0.02)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, chunks: torch.Tensor) -> torch.Tensor:
        # chunks: (batch, n_chunks, emb_dim)
        w = torch.softmax(self.chunk_logits, dim=0)          # (C,)
        pooled = torch.einsum("c,bcd->bd", w, chunks)        # (B, emb_dim)
        return self.fc2(self.act(self.fc1(pooled)))


class PerDimPositionEmbeddingBranch(nn.Module):
    """Per-(position, dimension) softmax pool, then the same bottleneck.

    Generalises the single 64-way position weighting: instead of one
    distribution over chunk positions shared by all dimensions, each of the
    ``emb_dim`` dimensions gets its OWN softmax over the ``n_chunks`` positions
    (an ``n_chunks x emb_dim`` logit matrix, softmax down the position axis). So
    a dimension can source from the opening of the document while another sources
    from the close. That is ``n_chunks * emb_dim`` parameters — far more than the
    64 of the shared version, hence more expressive but more prone to overfit.

    Logits start at zero → uniform per dimension → exactly the plain mean at
    step 0; ``fc2`` zero-init keeps the whole branch a no-op, preserving warm
    start.
    """

    def __init__(
        self,
        emb_dim: int,
        hidden: int,
        bottleneck: int = 16,
        n_chunks: int = 64,
        *,
        zero_init: bool = True,
    ):
        super().__init__()
        self.chunk_logits = nn.Parameter(torch.zeros(n_chunks, emb_dim))  # (C, D)
        self.fc1 = nn.Linear(emb_dim, bottleneck)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(bottleneck, hidden)

        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        if zero_init:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        else:
            nn.init.trunc_normal_(self.fc2.weight, std=0.02)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, chunks: torch.Tensor) -> torch.Tensor:
        # chunks: (batch, n_chunks, emb_dim)
        w = torch.softmax(self.chunk_logits, dim=0)          # (C, D), per-dim over positions
        pooled = torch.einsum("cd,bcd->bd", w, chunks)       # (B, emb_dim)
        return self.fc2(self.act(self.fc1(pooled)))


class EventPredictor(nn.Module):
    """Pretrained head + event-embedding branch, with per-arm input switching.

    ``use_market`` / ``use_text`` select the ablation arm by changing what feeds
    the hidden junction, rather than by zeroing an input — a masked-out channel
    would still consume parameters and leave a bias path.

    Args:
        head: Pretrained ``RegressionHead`` (scalar z-score output). Its
            ``.mlp`` is a ``torchvision.ops.MLP``: ``[Linear, Norm, ReLU,
            Dropout, Linear, Dropout]``.
        emb_dim: Event text embedding dimension.
        use_market: Feed the encoder CLS token through ``mlp[0]``.
        use_text: Feed the event embedding through ``g``.
        bottleneck: Width of ``g``'s bottleneck.
        freeze_first_layer: Freeze ``mlp[0]``. Only meaningful with ``use_market``.
        n_chunks: Chunk count the position pools expect (data property of the
            chunk cache; 64 for the OpenAI store, 32 for the Qwen store).
    """

    def __init__(
        self,
        head: nn.Module,
        emb_dim: int,
        *,
        use_market: bool = True,
        use_text: bool = True,
        bottleneck: int = 16,
        freeze_first_layer: bool = True,
        text_pool: str = "mean",
        n_chunks: int = 64,
        aux_dim: int = 0,
        aux_bottleneck: int = 2,
    ):
        super().__init__()
        if not use_market and not use_text:
            raise ValueError("at least one of use_market / use_text must be True")
        if text_pool not in ("mean", "position", "position_perdim"):
            raise ValueError(
                f"text_pool must be 'mean', 'position', or 'position_perdim'; got {text_pool}")

        mlp = getattr(head, "mlp", None)
        if mlp is None or len(mlp) < 5:
            raise TypeError(
                "head must expose a torchvision-style .mlp Sequential "
                f"(Linear, Norm, ReLU, Dropout, Linear, ...); got {type(head).__name__}"
            )
        if not isinstance(mlp[0], nn.Linear):
            raise TypeError(f"expected mlp[0] to be nn.Linear, got {type(mlp[0]).__name__}")

        self.use_market = use_market
        self.use_text = use_text

        self.first = mlp[0]           # Linear(d_embedding -> hidden)
        self.rest = nn.Sequential(*list(mlp)[1:])
        hidden = self.first.out_features

        if use_market and freeze_first_layer:
            self.first.requires_grad_(False)

        self.text_pool = text_pool
        if use_text:
            # Zero-init is only safe when the market term keeps the pre-norm
            # activation non-zero. Without it the sum would be exactly zero,
            # where RMSNorm's gradient is scaled by rsqrt(eps) ~ 1e3.
            if text_pool == "position":
                self.g = PositionWeightedEmbeddingBranch(
                    emb_dim, hidden, bottleneck, n_chunks, zero_init=use_market,
                )
            elif text_pool == "position_perdim":
                self.g = PerDimPositionEmbeddingBranch(
                    emb_dim, hidden, bottleneck, n_chunks, zero_init=use_market,
                )
            else:
                self.g = EventEmbeddingBranch(
                    emb_dim, hidden, bottleneck, zero_init=use_market,
                )
        else:
            self.g = None

        # A SECOND ACTION, ADDED AT THE SAME JUNCTION. The earnings surprise is
        # a handful of numbers and the transcript is 4096 dimensions; putting
        # them through one projection lets the 4 be swamped by the 4096 before
        # the bottleneck ever sees them. Two branches summed keeps each one's
        # path to the hidden state its own, and matches how the text is already
        # combined with the market term -- added, never concatenated.
        #
        # Zero-init on the output for the same reason as `g`: at step 0 the
        # model reproduces the market-only solution exactly, so every parameter
        # this branch moves is spent on the action rather than on relearning
        # the baseline.
        self.aux_dim = aux_dim
        self.g_aux = (
            EventEmbeddingBranch(aux_dim, hidden, aux_bottleneck, zero_init=True)
            if aux_dim else None
        )

    @property
    def hidden_dim(self) -> int:
        return self.first.out_features

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        cls: torch.Tensor | None = None,
        emb: torch.Tensor | None = None,
        aux: torch.Tensor | None = None,
        act_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
            cls: ``(batch, d_embedding)`` encoder CLS token. Required when
                ``use_market``.
            emb: ``(batch, emb_dim)`` pooled event embedding. Required when
                ``use_text``.
            aux: ``(batch, aux_dim)`` second action block — the earnings
                surprise. Required when the model was built with ``aux_dim``.
            act_mask: ``(batch, 1)`` 0/1. Where 0, BOTH action branches
                contribute exactly nothing and the output is the market-only
                prediction. This is what lets a model be fine-tuned on event
                rows alone and still be scored on the whole panel: without it
                the branch's response to the null action is a learned constant
                that was never trained, applied to the ~98% of rows with no
                event, and a common shift to that many names moves their ranks
                against the few that do have one.

        Returns:
            ``(batch, 1)`` z-score predictions.
        """
        parts = []
        if self.use_market:
            if cls is None:
                raise ValueError("use_market=True but cls is None")
            parts.append(self.first(cls))
        if self.use_text:
            if emb is None:
                raise ValueError("use_text=True but emb is None")
            g = self.g(emb)
            parts.append(g if act_mask is None else g * act_mask)
        if self.g_aux is not None:
            if aux is None:
                raise ValueError("aux_dim was set but aux is None")
            ga = self.g_aux(aux)
            parts.append(ga if act_mask is None else ga * act_mask)

        h = parts[0]
        for extra in parts[1:]:
            h = h + extra
        return self.rest(h)


def pool_chunk_embeddings(
    chunks: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean-pool ``(batch, n_chunks, emb_dim)`` over valid chunks.

    ``mask`` is ``True`` for valid chunks. Rows with no valid chunk pool to
    zero; the caller is responsible for excluding or flagging them, since a
    zero embedding is not a meaningful "no event" token.
    """
    if chunks.dim() != 3:
        raise ValueError(f"chunks must be (batch, n_chunks, emb_dim); got {tuple(chunks.shape)}")
    if mask is None:
        return chunks.mean(dim=1)
    if mask.shape != chunks.shape[:2]:
        raise ValueError(
            f"mask must be (batch, n_chunks) = {tuple(chunks.shape[:2])}; got {tuple(mask.shape)}"
        )
    w = mask.to(chunks.dtype).unsqueeze(-1)
    return (chunks * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)


class LatentWorldModel(nn.Module):
    """I-JEPA's predictor, with DAYS in place of patches.

    ``IJEPAPredictor`` takes the encoder's tokens for the visible patches, adds
    position embeddings, appends a mask token at the target position, runs a
    narrow transformer, and projects back to encoder width. This is the same
    module with one substitution: a token is a whole trading day's frozen CLS
    embedding, positions are lags, and the target is tomorrow.

        [ z_{t-K+1} ... z_t | q_1 ... q_H ]  ->  blocks  ->  z_hat_{t+1..t+H}

    MULTIPLE STEPS ARE MULTIPLE QUERY TOKENS, not a rollout. I-JEPA predicts
    every masked position in ONE forward from ``target_indices``, and the same
    shape works here: one query per horizon, each with its own position, all
    attending to the same context. A rollout would feed z_hat back in and
    compound its own error; this way the h-day-ahead prediction is trained
    against the real z_{t+h} and horizons stay comparable. It also matters for
    the event study, where an earnings reaction does not finish in one
    session.

    EVERY SLOT CARRIES AN ACTION, not just the query. A window of K days spans
    K transitions, and each one either had a call or did not; a model that only
    sees the action on the step it is predicting cannot know this stock
    reported three days ago, which is the state post-earnings drift depends on.
    ``emb`` is therefore (B, K+H, ...) — one action per slot, the null action
    where no call landed — and the same branch scores all of them. Passing a
    (B, ...) tensor keeps the older query-only behaviour.

    THE ACTION IS A ZERO-INIT RESIDUAL, not an extra token. An
    extra token would change the attention pattern the moment it exists, so a
    text arm could not start bit-identical to its market-only warm start, and
    the with-vs-without comparison would confound the branch with the sequence
    length. Added to the query through the same ``emb_dim -> bottleneck ->
    width`` branch ``EventPredictor`` uses, with a zero output layer, the model
    at step 0 IS the market-only model.

    Context slots are FIXED lags, so the position embedding is a plain
    (K+1, D) table rather than I-JEPA's gather: every row of a batch has the
    same layout, and a ticker missing a session is handled by the padding
    mask instead of by re-indexing.

    Args:
        d_latent: Width of the frozen encoder's embedding (the token width).
        pred_emb_dim: The predictor's internal width. Narrower than the
            encoder, as in I-JEPA — the transition is a smaller problem than
            the representation.
        depth, num_heads: Transformer size.
        context_days: How many past sessions are visible, most recent last.
        horizon: How many sessions ahead to predict. 1 is the plain
            next-day transition.
        residual: Predict the CHANGE in embedding and add today's back, rather
            than predicting tomorrow's outright. Roughly 60% of the next-day
            latent is already predictable from today's, and almost all of that
            is persistent structure — which firm this is, what regime it
            trades in. Predicting the level makes the network spend its
            capacity re-deriving that, and the transition it actually has to
            model is a small residue on top. With this on, the identity map is
            FREE: the projection starts near zero, so the model begins at
            "tomorrow looks like today" and every parameter it moves is spent
            on the change.
        use_text / emb_dim / bottleneck / text_pool / n_chunks: the action
            branch, matching EventPredictor's arguments.
    """

    def __init__(
        self,
        d_latent: int,
        *,
        pred_emb_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        context_days: int = 5,
        horizon: int = 1,
        residual: bool = True,
        drop_path_rate: float = 0.0,
        use_text: bool = False,
        emb_dim: int = 4096,
        bottleneck: int = 16,
        text_pool: str = "mean",
        n_chunks: int = 32,
    ):
        super().__init__()
        from market_jepa.modeling.backbones.transformer import ViTBlock

        self.context_days = context_days
        self.horizon = horizon
        self.residual = residual
        self.use_text = use_text

        self.predictor_embed = nn.Linear(d_latent, pred_emb_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_emb_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # K context slots + one query per horizon. ONE mask token shared
        # across horizons, as in I-JEPA: what separates t+1 from t+5 is the
        # position embedding, which is the thing the model should be reading.
        self.predictor_pos_embed = nn.Parameter(
            torch.randn(1, context_days + horizon, pred_emb_dim) * 0.02
        )

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.predictor_blocks = nn.ModuleList([
            ViTBlock(hidden_size=pred_emb_dim, num_attention_heads=num_heads,
                     intermediate_size=pred_emb_dim * 4, drop_path_rate=dpr[i])
            for i in range(depth)
        ])
        self.predictor_norm = nn.LayerNorm(pred_emb_dim)
        self.predictor_proj = nn.Linear(pred_emb_dim, d_latent)

        for m in (self.predictor_embed, self.predictor_proj):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        if residual:
            # Start AT the identity, not near it: with a zero output layer the
            # model's first prediction is exactly today's latent, which is a
            # far better starting point than a random one and makes the whole
            # of training a search over changes.
            nn.init.zeros_(self.predictor_proj.weight)

        self.g = None
        if use_text:
            if text_pool == "position":
                self.g = PositionWeightedEmbeddingBranch(
                    emb_dim, pred_emb_dim, bottleneck, n_chunks, zero_init=True)
            elif text_pool == "position_perdim":
                self.g = PerDimPositionEmbeddingBranch(
                    emb_dim, pred_emb_dim, bottleneck, n_chunks, zero_init=True)
            else:
                self.g = EventEmbeddingBranch(
                    emb_dim, pred_emb_dim, bottleneck, zero_init=True)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor | None = None,
        emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
            ctx: (B, K, d_latent) past day latents, most recent LAST.
            ctx_pad: (B, K) True where that lag is missing for this ticker.
            emb: the action, when ``use_text``.

        Returns:
            (B, horizon, d_latent), or (B, d_latent) when horizon == 1.
        """
        B, K, _ = ctx.shape
        if K != self.context_days:
            raise ValueError(
                f"expected {self.context_days} context days, got {K}")
        H = self.horizon

        x = self.predictor_embed(ctx) + self.predictor_pos_embed[:, :K]
        q = (self.mask_token + self.predictor_pos_embed[:, K:]).expand(B, -1, -1)
        x = torch.cat([x, q], dim=1)
        if self.use_text:
            if emb is None:
                raise ValueError("use_text=True but emb is None")
            if emb.shape[0] == B and emb.dim() >= 2 and emb.shape[1] == K + H:
                # One action per slot. Flatten slots into the batch so the
                # branch is applied identically everywhere — it is ONE branch,
                # shared, exactly as one encoder is shared across the frames.
                flat = emb.reshape(B * (K + H), *emb.shape[2:])
                x = x + self.g(flat).reshape(B, K + H, -1)
            else:
                # Query-only. The SAME action on every horizon: one call is one
                # event, and what changes with h is how far its effect has
                # propagated, which the position embeddings carry.
                x[:, K:] = x[:, K:] + self.g(emb).unsqueeze(1)
        kpm = None
        if ctx_pad is not None:
            kpm = torch.cat(
                [ctx_pad,
                 torch.zeros(B, H, dtype=torch.bool, device=x.device)], dim=1)
        for block in self.predictor_blocks:
            x = block(x, key_padding_mask=kpm)
        out = self.predictor_proj(self.predictor_norm(x)[:, K:])
        if self.residual:
            out = out + ctx[:, -1:]          # today, broadcast over horizons
        return out[:, 0] if H == 1 else out
