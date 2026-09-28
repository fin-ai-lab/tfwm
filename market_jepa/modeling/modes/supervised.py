"""Supervised model: backbone + task-specific prediction head.

Every task is a scalar regression onto the configured cross-sectional target
(empirical-uniform rank by default; see ``market_jepa.eval.tasks``), scored by Spearman
rank IC.
"""


from market_jepa.backbone_config import backbone_block
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import TrainingModel
from market_jepa.eval.tasks import TASK_REGISTRY

# Losses available to the supervised heads, all SCALAR: a RegressionHead
# against the configured target.
#
#   The two POINTWISE ones (mse, smooth_l1) are both minimized at the
#   conditional mean, which for a cross-sectionally standardized target is ~0
#   everywhere -- so a constant prediction is very nearly optimal and runs
#   collapse onto it. See _neg_corr_loss for the measurements.
#
#   The two RANKING ones (corr, pairwise) cannot be minimized by a constant
#   and exist to fix that. PAIRWISE IS THE ONE IN USE: it is the LTR objective
#   every supervised arm trains under, and the default below.
#
# THE BINNED FAMILY IS RETIRED. cross_entropy, expected_bin_mse and
# expected_bin_mae ran a k-way ClassificationHead over equal-count quantile
# bins, which brought with it n_bins, a soft-label temperature and its anneal,
# and an auxiliary expected-bin penalty -- four knobs and a calibration pass
# over the training dataloader, all in service of an objective nothing trains
# under any more. Removed 2026-09-07. The reported number was never the head
# anyway: it is a ridge probe on the frozen embeddings, scored by rank IC.
LOSS_FNS = ("mse", "smooth_l1", "corr", "pairwise")

# Ranking losses compare samples inside one cross-section; everything else is
# pointwise and ignores the cell structure.
_RANKING_LOSS_FNS = ("corr", "pairwise")


def _check_loss_fn(name: str) -> str:
    if name not in LOSS_FNS:
        raise ValueError(f"loss_fn must be one of {LOSS_FNS}, got {name!r}")
    return name


def _regression_loss(
    pred: torch.Tensor, target: torch.Tensor, loss_fn: str, beta: float,
) -> torch.Tensor:
    """Scalar regression loss over the already-NaN-filtered rows."""
    if loss_fn == "corr":
        return _neg_corr_loss(pred, target)
    if loss_fn == "pairwise":
        return _pairwise_rank_loss(pred, target)
    if loss_fn == "mse":
        return F.mse_loss(pred, target)
    # beta is in z units: residuals under beta are squared, above it linear.
    return F.smooth_l1_loss(pred, target, beta=beta)


def _neg_corr_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - Pearson(pred, target) over the batch. Optimum cannot be a constant.

    WHY THIS EXISTS. Both pointwise losses are minimized at the conditional
    mean of the target, and the cross-sectional rank target is zero-mean in
    EVERY cell by construction — so predicting the constant 0 is very nearly
    optimal for them. Measured against the analytic constant-0 loss, the best
    achievable training loss is ~78% of trivial for spread_change, ~97% for
    volatility_change, and ~100% for return. That is the entire gradient
    budget available to distinguish a useful model from a useless one, and it
    is why runs collapse onto an identical degenerate solution (grad_norm_avg
    ~0.3 against ~5.9 for healthy runs) and why return never escaped it at all.

    Correlation inverts that incentive: a constant prediction has zero
    variance and therefore zero correlation, scoring 1.0. That is the MIDPOINT
    of the range, not the worst value (anti-correlation is 2.0) — but it is no
    longer NEARLY OPTIMAL, which is the property that mattered. A full unit of
    improvement is on the table instead of the 0-22% the pointwise losses
    offered, and no particular constant is better than any other, so there is
    nothing for the model to settle into.

    Pooling across the batch rather than within cells is deliberate and
    correct HERE: the target is already standardized within its own cell (z or
    empirical rank), so between-cell level differences are already removed and
    a pooled correlation estimates the mean within-cell correlation. That is
    the same property the rank target exists to provide, and it is what lets
    this work without a cell-grouped sampler.

    Scale-free by construction, so the head's output scale is unconstrained —
    which is fine because rank IC only reads ordering.

    NUMERICS. The gradient of a correlation scales as 1/sd(pred), so it blows
    up exactly where this loss is trying to push hardest: measured, |grad| runs
    1.6e-2 at sd=1 and 1.6e6 at sd=1e-8, and is NaN at a perfectly constant
    prediction (0/0). A relative floor on the prediction's norm caps that. The
    floor is a FRACTION OF THE TARGET'S norm rather than an absolute epsilon,
    because the head's output scale is unconstrained under a scale-free loss
    and an absolute epsilon would mean different things at different scales.
    """
    p = pred - pred.mean()
    t = target - target.mean()
    t_norm = t.norm()
    # sd(pred) below _CORR_SD_FLOOR x sd(target) is treated as the degenerate
    # case: loss stays ~1 (the worst value) and the gradient stays bounded.
    p_norm = p.norm().clamp_min(_CORR_SD_FLOOR * t_norm.clamp_min(1e-8))
    return 1.0 - (p * t).sum() / (p_norm * t_norm).clamp_min(1e-8)


# 1e-3 of the target's scale. Small enough that a genuinely learning model
# never touches the floor, large enough to bound the gradient near collapse.
_CORR_SD_FLOOR = 1e-3


def _within_cell_rank_loss(pred, y, valid, cells, loss_fn):
    """Rank only among samples sharing a (date, anchor). Returns (loss, n_pairs).

    MODULE LEVEL BECAUSE TWO CLASSES NEED IT. It began as a method on
    SupervisedModel, and MultiTaskSupervisedModel -- a different class in this
    same file, with its own loss plumbing -- therefore never used it: it
    reached _pairwise_rank_loss directly, which takes no cells and pairs
    across the whole batch. So the multihead kept training the flat
    cross-cell surrogate for the entire life of the fix that removed it from
    the specialists, and a 203-month campaign was trained that way before
    anyone noticed.

    Cross-cell pairs are EXCLUDED rather than downweighted: comparing a stock
    in one cross-section against a stock in another is a different question
    from the one the metric asks.
    """
    same = cells[:, None] == cells[None, :]
    both = valid[:, None] & valid[None, :]
    yj = torch.nan_to_num(y, nan=0.0)
    dt = yj[:, None] - yj[None, :]
    mask = same & both & (dt > 0)
    n_pairs = int(mask.sum().item())
    if n_pairs == 0:
        # Every sample landed in its own cell -- nothing is comparable. A
        # silent zero would look like a converged step, so the count says so.
        return torch.tensor(0.0, device=pred.device, requires_grad=True), 0

    dp = pred[:, None] - pred[None, :]
    if loss_fn == "pairwise":
        return F.softplus(-dp[mask]).mean(), n_pairs

    # corr: sign-weighted margin normalized by its own scale -- a correlation
    # restricted to within-cell pairs.
    num = (torch.sign(dt[mask]) * dp[mask]).mean()
    scale = dp[mask].abs().mean().clamp_min(_CORR_SD_FLOOR)
    return 1.0 - num / scale, n_pairs


def _rows_in_pairs(y: torch.Tensor, cells: torch.Tensor | None) -> int:
    """How many rows of ``y`` sit in at least one strictly-ordered same-cell pair.

    ``y`` is (B,) with ``cells`` (B,) int ids, or (B, K) with cells implied
    by the row. This is the count of rows that receive ANY gradient from a
    ranking loss: a row whose cell holds no other valid, differently-labelled
    member contributes nothing, however many rows the batch carries. The
    2026-09-11 audit found the single-crop recipe left more than half of
    every batch in that state; ``train/rows_with_grad_frac`` makes it visible.
    """
    valid = ~torch.isnan(y)
    yj = torch.nan_to_num(y, nan=0.0)
    if y.dim() == 2:
        dt = yj[:, :, None] - yj[:, None, :]
        mask = (valid[:, :, None] & valid[:, None, :]) & (dt != 0)
        return int((mask.any(dim=2)).sum().item())
    if cells is None:
        # Flat path: any other valid row with a different label is a partner.
        dt = yj[:, None] - yj[None, :]
        mask = (valid[:, None] & valid[None, :]) & (dt != 0)
        return int(mask.any(dim=1).sum().item())
    same = cells[:, None] == cells[None, :]
    dt = yj[:, None] - yj[None, :]
    mask = same & (valid[:, None] & valid[None, :]) & (dt != 0)
    return int(mask.any(dim=1).sum().item())


def _pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """RankNet: softplus over every ordered pair. Purely ordinal.

    For each pair with ``target_i > target_j`` the model should put
    ``pred_i > pred_j``; the penalty is ``softplus(-(pred_i - pred_j))``, which
    is 0 for a confident correct ordering, log 2 for a tie, and grows linearly
    for a confident wrong one.

    WHY THIS OVER THE CORRELATION LOSS. Both refuse the constant solution that
    the pointwise losses reward, but they behave differently exactly AT it. A
    correlation's gradient is 0/0 there and has to be rescued with a variance
    floor, which leaves it artificially large (measured |grad| ~44 against a
    natural scale of ~0.02). This loss is smooth and well-conditioned at the
    same point: every pair contributes softplus'(0) = 1/2, and the resulting
    gradient on each sample is proportional to (n_below - n_above), i.e. it
    points straight at that sample's rank. Measured, that gradient correlates
    0.968 with the target at the collapse point.

    It is also purely ORDINAL, matching a Spearman metric more closely than a
    Pearson correlation on values does: no single extreme prediction can
    dominate, which is the same robustness argument that motivated the rank
    target in the first place.

    Pairs are taken across the whole batch rather than within cells, on the
    same grounds as the correlation loss: the target is already standardized
    inside its own cell, so a cross-cell comparison of standardized targets is
    meaningful and no cell-grouped sampler is needed.

    NOT scale-free, unlike the correlation loss — inflating the output scale
    sharpens correct pairs and lowers the loss without changing the ordering.
    That is benign (softplus saturates, so it buys a bounded amount and then
    stops) but it does mean the training loss keeps drifting down after the
    ordering has stopped improving. Read IC, not loss, for this one.

    Cost is O(B^2): 65k pairs at batch 256, 262k at 512 — negligible next to
    the backbone.
    """
    dp = pred[:, None] - pred[None, :]
    dt = target[:, None] - target[None, :]
    # dt > 0 takes each ordered pair exactly once and drops ties, which are
    # common: the rank target quantizes through a 64-knot grid.
    mask = dt > 0
    if not mask.any():
        return pred.sum() * 0.0
    return F.softplus(-dp[mask]).mean()


def load_pretrained_backbone(backbone: torch.nn.Module, ckpt_dir: str) -> None:
    """Initialize ``backbone`` from an SSL checkpoint directory.

    Expects the LeJEPA ``save_pretrained`` layout (``model.pt`` holding
    the full SSL model state dict with ``backbone.``-prefixed keys).
    Strict loading catches any architecture mismatch between the
    checkpoint and the configured backbone.
    """
    import os

    weights_path = os.path.join(ckpt_dir, "model.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"init_backbone_from: no model.pt in {ckpt_dir}"
        )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    prefix = "backbone."
    backbone_state = {
        k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
    }
    if not backbone_state:
        raise ValueError(
            f"init_backbone_from: no '{prefix}*' keys in {weights_path}"
        )
    backbone.load_state_dict(backbone_state, strict=True)
    print(
        f"Loaded pretrained backbone from {ckpt_dir} "
        f"({len(backbone_state)} tensors)"
    )


class _ScaleSharedGradient(torch.autograd.Function):
    """Identity in the forward pass; scale only the downstream shared gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (scale,) = ctx.saved_tensors
        return grad_output * scale.to(dtype=grad_output.dtype), None


class SupervisedModel(TrainingModel):
    """Backbone + task-specific prediction head for supervised training.

    The task fixes WHAT is predicted (target type and horizon); ``loss_fn``
    fixes HOW, and is the paper's supervised ablation axis. Every loss is
    scalar and gets a RegressionHead onto the cross-sectional target;
    ``pairwise`` (LTR) is the default and the one every arm trains under.

    Whatever the head, the REPORTED number is a ridge probe on the frozen
    backbone embeddings scored by rank IC — the same probe, on the same
    synchronized panel, that the SSL arm gets. That symmetry is the point: the
    two arms differ in their training objective and in nothing else, so
    ``encode()`` stays compatible with collect_probe_data() and the head is
    never on the reporting path.

    Constructor signature matches SupervisedModeConfig for Hydra instantiation:
        instantiate(cfg.mode, backbone=backbone)
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        task: str = "return_900",
        loss_fn: str = "pairwise",
        smooth_l1_beta: float = 1.0,
        init_backbone_from: str | None = None,
        init_head_from: str | None = None,
        head_init_scale: str = "unit",
    ):
        super().__init__()
        self.backbone = backbone
        self.task_spec = TASK_REGISTRY[task]
        self.loss_fn = _check_loss_fn(loss_fn)
        self.smooth_l1_beta = float(smooth_l1_beta)
        from market_jepa.eval.heads import (  # lazy: circular import with eval
            SkipRegressionHead, create_prediction_head, load_ridge_init,
        )

        if init_head_from:
            # The finetune head: a ridge probe on a skip, MLP branch at zero.
            # See SkipRegressionHead for why it is not just a linear head.
            self.head = SkipRegressionHead(backbone.d_embedding)
        else:
            self.head = create_prediction_head(self.task_spec, backbone.d_embedding)
        self.d_embedding = backbone.d_embedding

        if init_backbone_from:
            load_pretrained_backbone(self.backbone, init_backbone_from)
        if init_head_from:
            if not init_backbone_from:
                raise ValueError(
                    "init_head_from without init_backbone_from would put a "
                    "probe fit on a PRETRAINED encoder's features onto a "
                    "random one, where those features do not exist — pass "
                    "init_backbone_from, or unset init_head_from."
                )
            weight, bias, gain = load_ridge_init(
                init_head_from, task=task, scale=head_init_scale)
            self.head.init_from_ridge(weight, bias, gain)
            print(
                f"Initialized head from ridge probe {init_head_from} "
                f"(task={task}, scale={head_init_scale})"
            )

        # Set by the training harness once the dataset's target column order is
        # known. A SCALAR loss needs nothing else: the dataset emits the
        # cross-sectional z-score directly, so the head trains on the same
        # quantity the IC eval scores.
        self.target_col_idx: int | None = None
        self.eval_target_col_idx: int | None = None

    uses_multi_view: bool = False

    @property
    def loss_label(self) -> str:
        """Compact description of the objective, for run names and logs."""
        return self.loss_fn

    @property
    def mode_label(self) -> str:
        return f"[{self.task_spec.name}/{self.loss_label}]"

    @property
    def mode_str(self) -> str:
        return f"supervised: {self.task_spec.name} ({self.loss_label})"

    def default_run_name(self, backbone_type, cfg):
        return "__".join(
            [
                f"task={self.task_spec.name}",
                f"loss={self.loss_label}",
                f"bb={backbone_type}",
                f"d_emb={backbone_block(cfg).d_embedding}",
                f"blr={(cfg.optimizer.blr or cfg.mode.training_overrides.blr):.1e}",
                f"bs={cfg.training.per_device_train_batch_size}",
                f"steps={cfg.training.max_train_steps}",
            ]
        )

    def describe_parameters(self):
        from market_jepa.training.utils import count_parameters

        param_counts = count_parameters(self, backbone=self.backbone, head=self.head)
        summary = (
            f"Supervised model parameters:\n"
            f"  Backbone: {param_counts['backbone']:,}\n"
            f"  Head: {param_counts['head']:,}\n"
            f"  Total: {param_counts['total']:,}"
        )
        return param_counts, summary

    def post_training_step(self, completed_steps, max_train_steps):
        """Nothing to schedule. Required by the TrainingModel contract.

        This walked the soft-label temperature until the binned family was
        retired on 2026-09-07; every remaining loss is scalar and has no
        schedule of its own.
        """
        del completed_steps, max_train_steps
        return {}

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.backbone(x, lengths)   # (batch, d_embedding)
        return self.head(emb)             # (batch,)

    def compute_loss(self, pred, targets, cells=None):
        """MSE against the cross-sectional z-score, skipping NaN samples.

        NaN carries the no-clamp rule: the dataset emits NaN for any (anchor,
        horizon) whose forward window would run past the close, so those rows
        never reach the loss.
        """
        if self.target_col_idx is None:
            raise RuntimeError(
                "SupervisedModel.target_col_idx is None — set it from the "
                "dataset's target column order before training."
            )

        y = targets[:, self.target_col_idx]
        valid = ~torch.isnan(y)
        n_valid = int(valid.sum().item())

        if n_valid == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True), 0

        if cells is not None and self.loss_fn in _RANKING_LOSS_FNS:
            return self._within_cell_loss(pred, y, valid, cells)

        return _regression_loss(
            pred[valid], y[valid], self.loss_fn, self.smooth_l1_beta,
        ), n_valid

    def _within_cell_loss(self, pred, y, valid, cells):
        """Rank only among samples that share a (date, anchor).

        No partner fetching: independently drawn samples already collide into
        shared cells because the anchor lattice is finite, and how often is a
        knob (``end_grid_sec``, a multiple of ANCHOR_STEP). Coarser grid ->
        fewer cells -> more stocks per cell -> more pairs, at the cost of
        seeing fewer distinct cross-sections per step. At the default 36
        anchors/day a batch of 256 spreads over ~720 monthly cells and almost
        never collides, which is why the flat loss ends up ranking across
        unrelated cross-sections instead.

        Cross-cell pairs are EXCLUDED rather than downweighted: comparing a
        stock in one cross-section against a stock in another is a different
        question from the one the metric asks, and mixing the two would put
        the surrogate back in through the side door.
        """
        return _within_cell_rank_loss(pred, y, valid, cells, self.loss_fn)

    def compute_group_loss(self, pred, targets):
        """Rank WITHIN each cell. ``pred`` (B, K), ``targets`` (B, K, n_targets).

        Each row is K stocks at ONE shared wall-clock anchor, so comparisons
        made inside a row are exactly the comparisons the reported metric
        makes. The flat path can only pair across cells: with ~720 cells in a
        training month and a batch of 256, ~99.9% of its pairs put a stock from
        one cross-section against a stock from another. Those are meaningful
        only because the target is standardized within its cell — a surrogate.
        These are the real thing.

        Pointwise losses fall through to the flat computation on purpose: it
        makes "same batches, different loss" a controlled comparison rather
        than a confound, since the grouped sampler changes the data too.
        """
        if self.target_col_idx is None:
            raise RuntimeError(
                "SupervisedModel.target_col_idx is None — set it from the "
                "dataset's target column order before training."
            )
        y = targets[:, :, self.target_col_idx]                      # (B, K)
        valid = ~torch.isnan(y)

        if self.loss_fn not in _RANKING_LOSS_FNS:
            flat_ok = valid.reshape(-1)
            n = int(flat_ok.sum().item())
            if n == 0:
                return torch.tensor(0.0, device=pred.device, requires_grad=True), 0
            return _regression_loss(
                pred.reshape(-1)[flat_ok], y.reshape(-1)[flat_ok],
                self.loss_fn, self.smooth_l1_beta,
            ), n

        if self.loss_fn == "pairwise":
            # A pair needs both sides present and a strict ordering.
            yj = torch.nan_to_num(y, nan=0.0)
            dt = yj[:, :, None] - yj[:, None, :]
            mask = (valid[:, :, None] & valid[:, None, :]) & (dt > 0)
            n_pairs = int(mask.sum().item())
            if n_pairs == 0:
                return torch.tensor(0.0, device=pred.device, requires_grad=True), 0
            dp = pred[:, :, None] - pred[:, None, :]
            return F.softplus(-dp[mask]).mean(), n_pairs

        # corr: a genuine Pearson INSIDE each cell, averaged over cells. Cells
        # with fewer than 2 valid stocks carry no ordering and are dropped
        # rather than contributing a degenerate 0.
        m = valid.to(pred.dtype)
        n_k = m.sum(1)
        keep = n_k >= 2
        if not bool(keep.any()):
            return torch.tensor(0.0, device=pred.device, requires_grad=True), 0
        m, n_k = m[keep], n_k[keep]
        p, t = pred[keep] * m, torch.nan_to_num(y[keep], nan=0.0) * m
        pc = (p - (p.sum(1, keepdim=True) / n_k[:, None])) * m
        tc = (t - (t.sum(1, keepdim=True) / n_k[:, None])) * m
        num = (pc * tc).sum(1)
        tss = (tc * tc).sum(1).clamp_min(1e-12)
        # The floor goes INSIDE the sqrt. Clamping the product afterwards
        # leaves sqrt() differentiating at 0, whose derivative is inf; the
        # unselected branch of the max then contributes 0 * inf = NaN even
        # though the VALUE is correct. Measured: grad was nan at pred=const.
        pss = torch.maximum((pc * pc).sum(1), (_CORR_SD_FLOOR ** 2) * tss)
        den = torch.sqrt(pss) * torch.sqrt(tss)
        return 1.0 - (num / den).mean(), int(keep.sum().item())

    def training_step(self, batch, device, grad_accum_steps=1):
        batch_loss = 0.0
        batch_n = 0
        rows_with_grad = 0
        n_cells = 0

        for bucket in batch["buckets"]:
            targets_t = bucket["targets"].to(device)

            # (B, K, n_targets) means the dataset handed us CELLS: K stocks
            # sharing one wall-clock anchor. Rank inside them — that is the
            # quantity the metric measures. (B, n_targets) is the ordinary
            # one-observation-per-row path.
            if targets_t.dim() == 3:
                k = targets_t.shape[1]
                if self.loss_fn in _RANKING_LOSS_FNS:
                    rows_with_grad += _rows_in_pairs(
                        targets_t[:, :, self.target_col_idx], None)
                    n_cells += int(targets_t.shape[0])
                # One forward over all K stocks of every cell; reshaping after
                # keeps the group structure the loss needs.
                x = torch.cat([bucket["views"][i].to(device) for i in range(k)])
                lengths = torch.cat(
                    [bucket["lengths"][i].to(device) for i in range(k)])
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = self(x, lengths)
                    # cat stacked view i as block i, so unflatten is (K, B).
                    pred = pred.reshape(k, -1).transpose(0, 1)   # (B, K)
                    loss, n_valid = self.compute_group_loss(pred, targets_t)
                if n_valid == 0 or not torch.isfinite(loss):
                    continue
                (loss / grad_accum_steps).backward()
                batch_loss += loss.item() * n_valid
                batch_n += n_valid
                continue

            x = bucket["views"][0].to(device)
            lengths = bucket["lengths"][0].to(device)
            # Present whenever the dataset emits cross-sectional targets; the
            # ranking losses use it to compare only within a cross-section.
            cells = bucket.get("xs_cell")
            if cells is not None:
                cells = cells.to(device)
            if self.loss_fn in _RANKING_LOSS_FNS:
                y_col = targets_t[:, self.target_col_idx]
                rows_with_grad += _rows_in_pairs(y_col, cells)
                n_cells += (int(torch.unique(cells).numel())
                            if cells is not None else 1)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = self(x, lengths)
                loss, n_valid = self.compute_loss(pred, targets_t, cells)

            if n_valid == 0 or not torch.isfinite(loss):
                continue

            (loss / grad_accum_steps).backward()
            batch_loss += loss.item() * n_valid
            batch_n += n_valid

        if batch_n == 0:
            return None

        # WHICH LOSS PATH RAN, made visible. batch_n is the count compute_loss
        # weighted by, and it means DIFFERENT THINGS per path: pairs for
        # _within_cell_loss, ROWS for the flat fallback. Logging it alongside
        # the row count is how a reader can tell which one a run took.
        #
        # This matters because they are not interchangeable. _within_cell_loss
        # ranks only within a (date, anchor) -- the quantity the reported IC
        # measures -- while the flat path ranks across the whole batch, where
        # ~99.9% of pairs compare stocks at DIFFERENT instants. Until the
        # xs_cell plumbing was fixed, cells never reached compute_loss and
        # every supervised head trained on the flat surrogate while the code
        # read as though it were ranking within cells.
        #
        # ROWS WITH GRADIENT is the third number, and the one that says
        # whether the batch is doing work: a row in no usable pair gets none.
        # On the single-crop recipe this was ~0.45 of the batch; on cells it
        # is ~1.0. CELLS PER STEP counts the distinct cross-sections ranked.
        n_rows = sum(int(b["targets"].reshape(-1, b["targets"].shape[-1]).shape[0])
                     for b in batch["buckets"] if "targets" in b)
        n_buckets = max(len(batch["buckets"]), 1)
        metrics = {
            "train/loss": batch_loss / batch_n,
            "train/loss_units_per_step": batch_n / n_buckets,
            "train/rows_per_step": n_rows / n_buckets,
        }
        if self.loss_fn in _RANKING_LOSS_FNS:
            metrics["train/rows_with_grad_frac"] = rows_with_grad / max(n_rows, 1)
            metrics["train/cells_per_step"] = n_cells / n_buckets
        return {"loss": batch_loss / batch_n, "metrics": metrics}

    @torch.no_grad()
    def eval_step(self, eval_batches, device):
        """Eval loss plus a pooled rank IC of the head's own prediction.

        This IC is POOLED, not per-cross-section: in-training eval batches are
        independent draws, not synchronized instants. It tracks training
        progress; the reported per-cell IC comes from the synchronized
        cross-section eval, which is a separate offline pass.

        A BINNED run is ranked by its expected bin, ``sum_c c * p_c``, which
        is what makes a k-way classifier scoreable against a continuous
        target at all. Its eval LOSS is skipped rather than reported: the eval
        dataset emits the z-score (so the probe stays comparable with the SSL
        arm) while the bin edges were fitted on raw returns, and cross-entropy
        against labels from the wrong scale is not a number worth logging.
        The pooled IC is unaffected, because the eval column is only ever
        RANKED against the prediction.
        """
        from market_jepa.eval.metrics import rank_ic

        col_idx = self.eval_target_col_idx if self.eval_target_col_idx is not None else self.target_col_idx
        if col_idx is None:
            raise RuntimeError(
                "SupervisedModel: neither eval_target_col_idx nor target_col_idx is set."
            )

        self.eval()
        total_loss = 0.0
        total_n = 0
        all_pred: list[np.ndarray] = []
        all_true: list[np.ndarray] = []

        # Temporarily swap target_col_idx for compute_loss
        saved_col_idx = self.target_col_idx
        self.target_col_idx = col_idx

        try:
            for batch in eval_batches:
                for bucket in batch["buckets"]:
                    x = bucket["views"][0].to(device)
                    lengths = bucket["lengths"][0].to(device)
                    targets_t = bucket["targets"].to(device)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred = self(x, lengths)
                        loss, n_valid = self.compute_loss(pred, targets_t)
                        total_loss += loss.item() * n_valid
                        total_n += n_valid

                    y = targets_t[:, col_idx]
                    valid = ~torch.isnan(y)
                    if not bool(valid.any()):
                        continue
                    all_pred.append(pred[valid].float().cpu().numpy())
                    all_true.append(y[valid].float().cpu().numpy())

            metrics: dict[str, float] = {
                "eval/supervised_loss": (
                    total_loss / total_n if total_n > 0 else float("nan")
                )
            }

            if all_pred:
                ic = rank_ic(np.concatenate(all_pred), np.concatenate(all_true))
                if np.isfinite(ic):
                    metrics["eval/supervised_ic_pooled"] = float(ic)

            return metrics
        finally:
            self.target_col_idx = saved_col_idx


class MultiTaskSupervisedModel(TrainingModel):
    """Backbone + multiple task-specific heads sharing one forward pass.

    Each task loss is NaN-masked and sample-averaged independently. Task
    gradients are normalized over the shared backbone parameters before being
    mixed according to ``task_weights``, so the configured weights have direct
    semantics over the gradient signal entering the backbone, independent of
    each loss's native gradient scale (an MSE return loss and a 5-class
    cross-entropy otherwise contribute wildly different magnitudes). Task
    heads receive their ordinary, unnormalized task gradients because they do
    not compete for parameters.

    Normalization divides each task's objective by an EMA of its backbone
    gradient norm, so it costs one extra backbone backward per task per step.

    Constructor signature matches MultiTaskSupervisedModeConfig for Hydra
    instantiation: ``instantiate(cfg.mode, backbone=backbone)``.
    """

    uses_multi_view: bool = False

    def __init__(
        self,
        backbone: torch.nn.Module,
        tasks: list[str],
        loss_fn: str = "mse",
        smooth_l1_beta: float = 1.0,
        task_weights: dict[str, float] | None = None,
        gradient_norm_ema_decay: float = 0.99,
        gradient_norm_min: float = 1.0e-4,
        gradient_norm_max_scale: float = 10.0,
    ):
        super().__init__()
        if not tasks:
            raise ValueError("MultiTaskSupervisedModel requires at least one task")
        if len(set(tasks)) != len(tasks):
            raise ValueError("MultiTaskSupervisedModel tasks must be unique")
        if not 0.0 <= gradient_norm_ema_decay < 1.0:
            raise ValueError("gradient_norm_ema_decay must be in [0, 1)")
        if gradient_norm_min <= 0.0:
            raise ValueError("gradient_norm_min must be positive")
        if gradient_norm_max_scale <= 0.0:
            raise ValueError("gradient_norm_max_scale must be positive")
        self.backbone = backbone
        self.loss_fn = _check_loss_fn(loss_fn)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.task_names: list[str] = list(tasks)
        self.task_specs: dict[str, "object"] = {
            t: TASK_REGISTRY[t] for t in self.task_names
        }
        from market_jepa.eval.heads import create_prediction_head  # lazy

        self.heads = nn.ModuleDict({
            t: create_prediction_head(
                self.task_specs[t], backbone.d_embedding,
            )
            for t in self.task_names
        })
        self.d_embedding = backbone.d_embedding

        configured_weights = dict(task_weights or {})
        unknown_weights = set(configured_weights) - set(self.task_names)
        if unknown_weights:
            raise ValueError(
                "task_weights contains tasks not present in tasks: "
                f"{sorted(unknown_weights)}"
            )
        resolved_weights = [
            float(configured_weights.get(t, 1.0)) for t in self.task_names
        ]
        if any(not np.isfinite(w) or w < 0.0 for w in resolved_weights):
            raise ValueError("task_weights values must be finite and non-negative")
        weight_sum = sum(resolved_weights)
        if weight_sum <= 0.0:
            raise ValueError("at least one task weight must be positive")
        self.task_weights: dict[str, float] = {
            t: w / weight_sum for t, w in zip(self.task_names, resolved_weights)
        }
        self.gradient_norm_ema_decay = float(gradient_norm_ema_decay)
        self.gradient_norm_min = float(gradient_norm_min)
        self.gradient_norm_max_scale = float(gradient_norm_max_scale)
        self.register_buffer(
            "_task_gradient_norm_ema",
            torch.zeros(len(self.task_names), dtype=torch.float32),
        )
        self.register_buffer(
            "_task_gradient_norm_initialized",
            torch.zeros(len(self.task_names), dtype=torch.bool),
        )


        # Filled in by the training harness from the dataset's column order.
        # No calibration step: the dataset emits z-scores directly.
        self.target_col_idx: dict[str, int] = {}
        self.eval_target_col_idx: dict[str, int] = {}

    @property
    def loss_label(self) -> str:
        """Compact description of the objective, for run names and logs."""
        return self.loss_fn

    @property
    def mode_label(self) -> str:
        return f"[multi:{'+'.join(self.task_names)}]"

    @property
    def mode_str(self) -> str:
        return (f"multi-supervised: {','.join(self.task_names)} "
                f"({self.loss_label})")

    def default_run_name(self, backbone_type, cfg):
        weight_label = "+".join(
            f"{self.task_weights[t]:.3g}" for t in self.task_names
        )
        return "__".join(
            [
                f"tasks={'+'.join(self.task_names)}",
                f"task_weights={weight_label}",
                f"bb={backbone_type}",
                f"d_emb={backbone_block(cfg).d_embedding}",
                f"blr={(cfg.optimizer.blr or cfg.mode.training_overrides.blr):.1e}",
                f"bs={cfg.training.per_device_train_batch_size}",
                f"steps={cfg.training.max_train_steps}",
            ]
        )

    def describe_parameters(self):
        from market_jepa.training.utils import count_parameters

        head_kwargs = {f"head_{t}": h for t, h in self.heads.items()}
        param_counts = count_parameters(self, backbone=self.backbone, **head_kwargs)
        head_lines = "\n".join(
            f"  Head[{t}]: {param_counts[f'head_{t}']:,}" for t in self.task_names
        )
        summary = (
            f"MultiTaskSupervised model parameters:\n"
            f"  Backbone: {param_counts['backbone']:,}\n"
            f"{head_lines}\n"
            f"  Total: {param_counts['total']:,}"
        )
        return param_counts, summary

    def post_training_step(self, completed_steps, max_train_steps):
        """Nothing to schedule. Required by the TrainingModel contract.

        This walked the soft-label temperature until the binned family was
        retired on 2026-09-07; every remaining loss is scalar and has no
        schedule of its own.
        """
        del completed_steps, max_train_steps
        return {}

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        emb = self.backbone(x, lengths)  # (batch, d_embedding)
        return {t: head(emb) for t, head in self.heads.items()}

    def _check_state(self):
        if not self.target_col_idx:
            raise RuntimeError(
                "MultiTaskSupervisedModel.target_col_idx is empty — set it "
                "from the dataset's target column order before training."
            )

    def _per_task_loss(
        self,
        pred: torch.Tensor,
        y: torch.Tensor,
        valid: torch.Tensor,
        cells: torch.Tensor | None,
        spec,
        task_name: str,
    ) -> tuple[torch.Tensor, int]:
        """One objective for every head, applied per task. Returns (loss, n).

        WITHIN-CELL WHEN THE CELLS ARE THERE, mirroring
        SupervisedModel.compute_loss. This method used to take pre-masked
        tensors and go straight to _regression_loss, so the multihead ranked
        across the whole batch -- the flat cross-cell surrogate -- for the
        entire life of the fix that removed it from the specialists. The
        arguments are unmasked now because a within-cell mask is built over
        the full batch: pre-masking would break the correspondence between
        rows and their cell ids.

        The z-scores are already on a common scale, so per-task loss choices
        would only confound the gradient-norm balancing below.

        n is PAIRS on the within-cell path and ROWS on the flat one, which is
        what the caller weights by -- the same convention SupervisedModel
        uses, and the reason train/loss_units_per_step is logged beside the
        row count rather than on its own.
        """
        del spec, task_name
        if cells is not None and self.loss_fn in _RANKING_LOSS_FNS:
            return _within_cell_rank_loss(pred, y, valid, cells, self.loss_fn)
        n_valid = int(valid.sum().item())
        return _regression_loss(
            pred[valid], y[valid], self.loss_fn, self.smooth_l1_beta,
        ), n_valid

    def _shared_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def _backbone_gradient_norm(
        self,
        loss: torch.Tensor,
        parameters: list[torch.nn.Parameter],
    ) -> torch.Tensor:
        """L2 norm of a task objective's gradient w.r.t. shared parameters.

        Measured on the backbone parameters themselves rather than on the
        embedding, so ``task_weights`` describes the actual contribution each
        task makes to the trunk update.
        """
        grads = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True,
        )
        norm_sq = torch.zeros((), device=loss.device, dtype=torch.float32)
        for grad in grads:
            if grad is not None:
                norm_sq = norm_sq + grad.detach().float().square().sum()
        return norm_sq.sqrt()

    def _head_output_gradient_norm(
        self,
        loss: torch.Tensor,
        outputs: list[torch.Tensor],
    ) -> torch.Tensor:
        """L2 norm at a head's outputs — diagnostic only, not normalized."""
        grads = torch.autograd.grad(
            loss, outputs, retain_graph=True, allow_unused=True,
        )
        norm_sq = torch.zeros((), device=loss.device, dtype=torch.float32)
        for grad in grads:
            if grad is not None:
                norm_sq = norm_sq + grad.detach().float().square().sum()
        return norm_sq.sqrt()

    def _update_gradient_norm_ema(
        self,
        task_idx: int,
        current_norm: torch.Tensor,
    ) -> torch.Tensor:
        current = current_norm.detach().to(
            device=self._task_gradient_norm_ema.device,
            dtype=self._task_gradient_norm_ema.dtype,
        )
        if not bool(self._task_gradient_norm_initialized[task_idx].item()):
            updated = current
            self._task_gradient_norm_initialized[task_idx] = True
        else:
            beta = self.gradient_norm_ema_decay
            updated = (
                self._task_gradient_norm_ema[task_idx] * beta
                + current * (1.0 - beta)
            )
        self._task_gradient_norm_ema[task_idx] = updated
        return updated

    def training_step(self, batch, device, grad_accum_steps=1):
        self._check_state()
        device = torch.device(device)
        autocast_kwargs = dict(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        )
        n_tasks = len(self.task_names)
        loss_sum: dict[str, torch.Tensor | None] = {t: None for t in self.task_names}
        n_by_task: dict[str, int] = {t: 0 for t in self.task_names}
        # ROWS alongside the loss units, so a reader can tell which path ran:
        # n_by_task counts PAIRS within cells and ROWS on the flat fallback,
        # and the two are indistinguishable from the loss value alone.
        rows_seen: dict[str, int] = {t: 0 for t in self.task_names}
        # Retain each bucket's shared representation and targets. The first
        # head pass measures raw task-gradient norms; a cheap second head pass
        # then installs task-specific gradient routing without recomputing the
        # backbone.
        bucket_records: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
        ] = []
        head_outputs: dict[str, list[torch.Tensor]] = {t: [] for t in self.task_names}

        for bucket in batch["buckets"]:
            targets_t = bucket["targets"].to(device)
            if targets_t.dim() == 3:
                # CELLS, the same (B, K, n_targets) shape SupervisedModel
                # ranks inside. This class keeps its row-wise loss plumbing,
                # so the cell is flattened to K rows per cell and the cell
                # index becomes the xs_cell id -- _per_task_loss then ranks
                # within it exactly as it would for rows that collided into
                # one (date, anchor) by chance, except that now every row has
                # K-1 partners.
                b, k = targets_t.shape[:2]
                x = torch.cat([bucket["views"][i].to(device) for i in range(k)])
                lengths = torch.cat(
                    [bucket["lengths"][i].to(device) for i in range(k)])
                # cat stacked view i as block i: row r of block i is cell r.
                cells = torch.arange(b, device=device).repeat(k)
                targets_t = targets_t.transpose(0, 1).reshape(b * k, -1)
            else:
                x = bucket["views"][0].to(device)
                lengths = bucket["lengths"][0].to(device)
                # THE CROSS-SECTION ID. collate_bucketed carries it since
                # 00fbb35; this class simply never read it.
                cells = bucket.get("xs_cell")
                if cells is not None:
                    cells = cells.to(device)

            with torch.autocast(**autocast_kwargs):
                emb = self.backbone(x, lengths)
                bucket_records.append((emb, targets_t, cells))
                for t, spec in self.task_specs.items():
                    pred = self.heads[t](emb)
                    head_outputs[t].append(pred)
                    col = self.target_col_idx[t]
                    y = targets_t[:, col]
                    valid = ~torch.isnan(y)
                    if not bool(valid.any()):
                        continue
                    rows_seen[t] += int(valid.sum().item())
                    loss_t, n_units = self._per_task_loss(
                        pred, y, valid, cells, spec, t)
                    if n_units == 0 or not torch.isfinite(loss_t):
                        continue
                    weighted = loss_t * n_units
                    previous = loss_sum[t]
                    loss_sum[t] = weighted if previous is None else previous + weighted
                    n_by_task[t] += n_units

        # Per-task sample-weighted mean loss over the whole batch.
        objectives: dict[str, torch.Tensor] = {
            t: loss_sum[t] / n_by_task[t]
            for t in self.task_names
            if loss_sum[t] is not None and n_by_task[t] > 0
        }
        if not objectives:
            return None

        metrics: dict[str, float] = {}
        shared_parameters = self._shared_parameters()

        if shared_parameters:
            scales: dict[str, torch.Tensor] = {}
            for task_idx, t in enumerate(self.task_names):
                objective = objectives.get(t)
                if objective is None:
                    continue
                head_output_norm = self._head_output_gradient_norm(
                    objective, head_outputs[t]
                )
                raw_norm = self._backbone_gradient_norm(objective, shared_parameters)
                ema_norm = self._update_gradient_norm_ema(task_idx, raw_norm)
                weight = self.task_weights[t]
                scale = (
                    torch.as_tensor(weight, device=device, dtype=torch.float32)
                    / ema_norm.clamp_min(self.gradient_norm_min).to(device)
                ).clamp_max(self.gradient_norm_max_scale).detach()
                scales[t] = scale
                metrics[f"train/task_head_output_gradient_norm_{t}"] = float(
                    head_output_norm.item()
                )
                metrics[f"train/task_gradient_norm_{t}"] = float(raw_norm.item())
                metrics[f"train/task_gradient_norm_ema_{t}"] = float(ema_norm.item())
                metrics[f"train/task_gradient_scale_{t}"] = float(scale.item())
                metrics[f"train/task_gradient_contribution_norm_{t}"] = float(
                    (raw_norm * scale).item()
                )
                metrics[f"train/task_weight_{t}"] = weight

            # Re-run only the lightweight heads. The custom identity leaves
            # head gradients untouched while scaling each task's gradient as it
            # crosses into the shared backbone.
            routed_sum: dict[str, torch.Tensor | None] = {
                t: None for t in self.task_names
            }
            for emb, targets_t, cells in bucket_records:
                with torch.autocast(**autocast_kwargs):
                    for t, spec in self.task_specs.items():
                        if t not in scales:
                            continue
                        col = self.target_col_idx[t]
                        y = targets_t[:, col]
                        valid = ~torch.isnan(y)
                        if not bool(valid.any()):
                            continue
                        routed_emb = _ScaleSharedGradient.apply(emb, scales[t])
                        # UNMASKED, like the first pass: the within-cell mask
                        # is built over the whole batch, so pre-masking here
                        # would misalign rows against their cell ids.
                        pred_v = self.heads[t](routed_emb)
                        loss_t, n_units = self._per_task_loss(
                            pred_v, y, valid, cells, spec, t)
                        if n_units == 0 or not torch.isfinite(loss_t):
                            continue
                        weighted = loss_t * n_units
                        previous = routed_sum[t]
                        routed_sum[t] = (
                            weighted if previous is None else previous + weighted
                        )

            backward_loss = None
            for t, summed in routed_sum.items():
                if summed is None:
                    continue
                term = summed / n_by_task[t]
                backward_loss = term if backward_loss is None else backward_loss + term
        else:
            # No trainable shared parameters, so there is nothing to balance
            # and the heads take their ordinary gradients. Unreachable from a
            # config since freeze_backbone was retired (2026-09-16); kept as a
            # guard for a caller that freezes the backbone by hand.
            backward_loss = None
            for objective in objectives.values():
                backward_loss = (
                    objective if backward_loss is None else backward_loss + objective
                )

        if backward_loss is None or not torch.isfinite(backward_loss):
            return None

        (backward_loss / grad_accum_steps).backward()

        per_task_loss = {t: float(o.item()) for t, o in objectives.items()}
        # Constant denominator: the logged curve stays comparable across steps
        # where a task happens to have no labels.
        primary = sum(per_task_loss.values()) / n_tasks
        metrics["train/loss"] = primary
        # WHICH LOSS PATH RAN, made visible -- the same pair the specialists
        # log. n_by_task counts PAIRS within cells and ROWS on the flat
        # fallback, so the two together say which one this step took. The
        # absence of these keys on the multihead is what hid a 203-month
        # campaign trained on the flat surrogate.
        n_buckets = max(len(batch["buckets"]), 1)
        metrics["train/loss_units_per_step"] = (
            sum(n_by_task.values()) / max(n_tasks, 1) / n_buckets)
        metrics["train/rows_per_step"] = (
            sum(rows_seen.values()) / max(n_tasks, 1) / n_buckets)
        for t, value in per_task_loss.items():
            metrics[f"train/loss_{t}"] = value

        return {
            "loss": primary,
            "metrics": metrics,
        }

    @torch.no_grad()
    def eval_step(self, eval_batches, device):
        """Per-task eval loss plus a pooled rank IC per head.

        Pooled, not per-cross-section — see ``SupervisedModel.eval_step``.
        """
        from market_jepa.eval.metrics import rank_ic

        self._check_state()
        col_idx_map = {
            t: self.eval_target_col_idx.get(t, self.target_col_idx[t])
            for t in self.task_names
        }

        self.eval()
        sum_loss: dict[str, float] = {t: 0.0 for t in self.task_names}
        sum_n: dict[str, int] = {t: 0 for t in self.task_names}
        all_pred: dict[str, list[np.ndarray]] = {t: [] for t in self.task_names}
        all_true: dict[str, list[np.ndarray]] = {t: [] for t in self.task_names}

        for batch in eval_batches:
            for bucket in batch["buckets"]:
                x = bucket["views"][0].to(device)
                lengths = bucket["lengths"][0].to(device)
                targets_t = bucket["targets"].to(device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    preds = self(x, lengths)

                for t, spec in self.task_specs.items():
                    col = col_idx_map[t]
                    y = targets_t[:, col]
                    valid = ~torch.isnan(y)
                    n_valid = int(valid.sum().item())
                    if n_valid == 0:
                        continue
                    pred_v = preds[t][valid]
                    y_v = y[valid]
                    # A BINNED head is ranked by its expected bin,
                    score_v = pred_v
                    # FLAT ON PURPOSE (cells=None). This is the monitoring
                    # eval loss, not the reported IC, and it is compared
                    # across checkpoints -- switching it to within-cell would
                    # silently change what the logged number means without
                    # improving anything the scorer reports.
                    loss_t, _ = self._per_task_loss(
                        pred_v, y_v, torch.ones_like(y_v, dtype=torch.bool),
                        None, spec, t)
                    sum_loss[t] += float(loss_t.item()) * n_valid
                    sum_n[t] += n_valid
                    all_pred[t].append(score_v.float().cpu().numpy())
                    all_true[t].append(y_v.float().cpu().numpy())

        metrics: dict[str, float] = {}
        per_task_avg: list[float] = []
        for t in self.task_names:
            if sum_n[t] > 0:
                avg = sum_loss[t] / sum_n[t]
                metrics[f"eval/supervised_loss_{t}"] = avg
                per_task_avg.append(avg)

            if all_pred[t]:
                ic = rank_ic(
                    np.concatenate(all_pred[t]), np.concatenate(all_true[t]),
                )
                if np.isfinite(ic):
                    metrics[f"eval/supervised_ic_pooled_{t}"] = float(ic)

        if per_task_avg:
            metrics["eval/supervised_loss"] = float(np.mean(per_task_avg))
        return metrics
