"""Tests for the probing eval pipeline: discretization, portfolio metrics,
RRC target computation, eval caching, and collect_probe_data.
"""

import numpy as np
import pytest
import torch

from market_jepa.augmentations import _random_resized_crop_numpy


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_dense_features(n_rows=23400, seed=42):
    """Create a dense (n_rows, 9) feature array mimicking post-preprocessing data."""
    rng = np.random.RandomState(seed)
    features = np.empty((n_rows, 9), dtype=np.float64)
    features[:, 0] = 100.0 + rng.randn(n_rows).cumsum() * 0.01  # bid_price
    features[:, 2] = features[:, 0] + rng.rand(n_rows) * 0.1     # high > bid
    features[:, 3] = features[:, 0] - rng.rand(n_rows) * 0.1     # low < bid
    features[:, 4] = features[:, 0] + 0.05 + rng.randn(n_rows) * 0.005  # ask
    features[:, 5] = rng.randint(100, 1000, n_rows).astype(float)  # bid_size
    features[:, 6] = rng.randint(100, 1000, n_rows).astype(float)  # ask_size
    features[:, 7] = rng.randint(1, 500, n_rows).astype(float)     # volume (>0)
    features[:, 8] = rng.randint(1, 50, n_rows).astype(float)      # n (>0)

    mid = (features[:, 0] + features[:, 4]) / 2
    features[:, 1] = mid + rng.randn(n_rows) * 0.005
    features[features[:, 8] == 0, 1] = np.nan

    return features


# -----------------------------------------------------------------------
# Metrics under test
# -----------------------------------------------------------------------

from market_jepa.eval.metrics import MIN_CELL_N, grouped_rank_ic, rank_ic


# =======================================================================
# Part 1: RRC target computation in streaming_dataset
# =======================================================================


class TestRRCTargetComputation:
    """Tests that the RRC block in streaming_dataset computes real (non-NaN) targets."""

    def test_targets_not_all_nan(self):
        """With targets configured, RRC crop + compute_pair_targets produces non-NaN."""
        from stable_finance.dataset import compute_pair_targets

        features = _make_dense_features(23400)
        rng = np.random.RandomState(42)

        # Simulate what streaming_dataset does: crop, get g0_start/g0_window
        v, g0_start, agg, g0_window = _random_resized_crop_numpy(
            features, (0.3, 0.7), 2048, rng,
        )
        assert v is not None

        t_idx = g0_start + g0_window - 1
        targets = compute_pair_targets(
            focal_features=features,
            t_idx=t_idx,
            horizons=[300, 600, 900],
            types=["return", "spread_change", "volatility_change"],
            rf_data=None,
            rf_price_mode=None,
            date_str=None,
            rf_t_idx=None,
        )

        assert len(targets) == 9
        assert not np.isnan(targets).all(), "All targets are NaN"

    def test_t_idx_within_bounds(self):
        """t_idx = g0_start + g0_window - 1 is a valid index into features."""
        features = _make_dense_features(23400)
        N = len(features)

        for seed in range(50):
            rng = np.random.RandomState(seed)
            v, s, a, w = _random_resized_crop_numpy(
                features, (0.3, 1.0), 2048, rng,
            )
            if v is not None:
                t_idx = s + w - 1
                assert 0 <= t_idx < N, f"t_idx={t_idx} out of bounds for N={N}"

    def test_target_return_sign(self):
        """If future mid-price is higher than current, return target should be positive."""
        from stable_finance.dataset import compute_pair_targets

        N = 5000
        features = np.zeros((N, 9), dtype=np.float64)
        # Prices rise linearly
        features[:, 0] = 100.0 + np.arange(N) * 0.001  # bid
        features[:, 4] = 100.05 + np.arange(N) * 0.001  # ask
        features[:, 1] = (features[:, 0] + features[:, 4]) / 2  # vwap
        features[:, 7] = 100.0  # volume
        features[:, 8] = 10.0   # n

        # Use a moderate scale so the crop ends well before N, leaving room
        # for forward horizons without clamping
        rng = np.random.RandomState(42)
        v, s, a, w = _random_resized_crop_numpy(
            features, (0.3, 0.5), 256, rng,
        )
        assert v is not None
        assert s + w + 900 <= N, "Crop too close to end for this test"

        t_idx = s + w - 1
        targets = compute_pair_targets(
            focal_features=features,
            t_idx=t_idx,
            horizons=[300, 600, 900],
            types=["return"],
            rf_data=None, rf_price_mode=None, date_str=None, rf_t_idx=None,
        )

        # Prices are rising, so all forward returns should be positive
        for i, h in enumerate([300, 600, 900]):
            assert targets[i] > 0, f"Return at horizon {h} should be positive, got {targets[i]}"

# =======================================================================
# Part 5: run_eval with cached batches
# =======================================================================


class TestRunEvalCached:
    """Test that run_eval works correctly with a list of pre-cached batch dicts."""

    def _make_model(self, n_features=10, d_embedding=32, proj_dim=16):
        from market_jepa.modeling import LeJEPA, create_backbone
        backbone = create_backbone("resnet", n_features=n_features, d_embedding=d_embedding, pool="mean")
        return LeJEPA(backbone, proj_dim=proj_dim, lamb=0.02)

    def _make_batch(self, batch_size=4, n_features=10, seq_len=64, n_views=2):
        views = [torch.randn(batch_size, n_features, seq_len) for _ in range(n_views)]
        lengths = [torch.full((batch_size,), seq_len, dtype=torch.long) for _ in range(n_views)]
        return {
            "buckets": [{
                "views": views,
                "lengths": lengths,
                "n_global_views": n_views,
            }],
        }

    def test_returns_expected_keys(self):
        from market_jepa.training.pretrain import run_jepa_eval

        model = self._make_model()
        batches = [self._make_batch() for _ in range(3)]

        result = run_jepa_eval(model, batches, torch.device("cpu"))
        assert "eval/jepa_loss" in result
        assert "eval/jepa_sigreg_loss" in result
        assert "eval/jepa_inv_loss" in result

    def test_returns_finite_values(self):
        from market_jepa.training.pretrain import run_jepa_eval

        model = self._make_model()
        batches = [self._make_batch() for _ in range(5)]

        result = run_jepa_eval(model, batches, torch.device("cpu"))
        for key, val in result.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_empty_batch_list(self):
        """Empty eval_batches returns NaN (np.mean of empty list)."""
        from market_jepa.training.pretrain import run_jepa_eval

        model = self._make_model()
        result = run_jepa_eval(model, [], torch.device("cpu"))
        # np.mean([]) returns nan
        assert np.isnan(result["eval/jepa_loss"])

    def test_model_in_eval_mode_during_run(self):
        """run_eval sets model to eval mode."""
        from market_jepa.training.pretrain import run_jepa_eval

        model = self._make_model()
        model.train()
        batches = [self._make_batch()]
        run_jepa_eval(model, batches, torch.device("cpu"))
        # After run_eval, model should still be in eval mode (caller restores)
        assert not model.training


# =======================================================================
# Part 6: collect_probe_data
# =======================================================================


class TestCollectProbeData:
    """Tests for collect_probe_data in pretrain.py."""

    def _make_model(self, n_features=10, d_embedding=32, proj_dim=16):
        from market_jepa.modeling import LeJEPA, create_backbone
        backbone = create_backbone("resnet", n_features=n_features, d_embedding=d_embedding, pool="mean")
        return LeJEPA(backbone, proj_dim=proj_dim, lamb=0.02)

    def _make_batch_with_targets(self, batch_size=4, n_features=10, seq_len=64,
                                  n_views=2, n_targets=9):
        views = [torch.randn(batch_size, n_features, seq_len) for _ in range(n_views)]
        lengths = [torch.full((batch_size,), seq_len, dtype=torch.long) for _ in range(n_views)]
        targets = torch.randn(batch_size, n_targets)
        return {
            "buckets": [{
                "views": views,
                "lengths": lengths,
                "n_global_views": n_views,
                "targets": targets,
            }],
        }

    def test_output_shapes(self):
        from market_jepa.training.pretrain import collect_probe_data

        model = self._make_model()
        n_targets = 9
        batches = [self._make_batch_with_targets(batch_size=8, n_targets=n_targets) for _ in range(3)]

        X, y = collect_probe_data(model, batches, torch.device("cpu"))
        assert X.ndim == 2
        assert y.ndim == 2
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] == 24  # 3 batches * 8 samples
        assert X.shape[1] == 32  # d_embedding
        assert y.shape[1] == n_targets

    def test_only_all_nan_target_rows_filtered(self):
        """Partial NaNs survive; the probe masks each column on its own.

        Under no-clamp, whether a row's target exists depends on the horizon —
        an all-columns filter would keep only the rows valid at the LONGEST
        horizon and fit every probe on that sliver.
        """
        from market_jepa.training.pretrain import collect_probe_data

        model = self._make_model()
        batch = self._make_batch_with_targets(batch_size=10, n_targets=3)
        batch["buckets"][0]["targets"][0, :] = float("nan")   # dead row
        batch["buckets"][0]["targets"][3, :] = float("nan")   # dead row
        batch["buckets"][0]["targets"][7, 1] = float("nan")   # one horizon only

        X, y = collect_probe_data(model, [batch], torch.device("cpu"))
        # 2 all-NaN rows drop; the partial one stays, NaN and all.
        assert X.shape[0] == 8
        assert y.shape[0] == 8
        assert np.isnan(y).sum() == 1

    def test_all_nan_targets(self):
        from market_jepa.training.pretrain import collect_probe_data

        model = self._make_model()
        batch = self._make_batch_with_targets(batch_size=4, n_targets=3)
        batch["buckets"][0]["targets"][:] = float("nan")

        X, y = collect_probe_data(model, [batch], torch.device("cpu"))
        assert X.shape[0] == 0
        assert y.shape[0] == 0

    def test_empty_batches(self):
        from market_jepa.training.pretrain import collect_probe_data

        model = self._make_model()
        X, y = collect_probe_data(model, [], torch.device("cpu"))
        assert X.shape[0] == 0
        assert y.shape[0] == 0

    def test_no_targets_key(self):
        """Batches without 'targets' in bucket produce no y data."""
        from market_jepa.training.pretrain import collect_probe_data

        model = self._make_model()
        batch = {
            "buckets": [{
                "views": [torch.randn(4, 10, 64), torch.randn(4, 10, 64)],
                "lengths": [torch.full((4,), 64, dtype=torch.long)] * 2,
                "n_global_views": 2,
            }],
        }

        X, y = collect_probe_data(model, [batch], torch.device("cpu"))
        # X collected but y empty → returns empty
        assert X.shape[0] == 0

    def test_model_left_in_eval_mode(self):
        """collect_probe_data sets model to eval mode."""
        from market_jepa.training.pretrain import collect_probe_data

        model = self._make_model()
        model.train()
        batch = self._make_batch_with_targets(batch_size=4, n_targets=3)
        collect_probe_data(model, [batch], torch.device("cpu"))
        assert not model.training

    def test_bucket_without_n_global_views_skipped(self):
        """Buckets without n_global_views are skipped."""
        from market_jepa.training.pretrain import collect_probe_data

        model = self._make_model()
        batch = {
            "buckets": [{
                "views": [torch.randn(4, 10, 64), torch.randn(4, 10, 64)],
                "lengths": [torch.full((4,), 64, dtype=torch.long)] * 2,
                "targets": torch.randn(4, 3),
                # No n_global_views key
            }],
        }

        X, y = collect_probe_data(model, [batch], torch.device("cpu"))
        assert X.shape[0] == 0


# =======================================================================
# Part 7: dispatch_probe_eval (unit-level, no actual subprocess spawn)
# =======================================================================


class TestDispatchProbeEval:
    """Test dispatch_probe_eval writes npz and spawns subprocess."""

    def _make_model(self, n_features=10, d_embedding=32, proj_dim=16):
        from market_jepa.modeling import LeJEPA, create_backbone
        backbone = create_backbone("resnet", n_features=n_features, d_embedding=d_embedding, pool="mean")
        return LeJEPA(backbone, proj_dim=proj_dim, lamb=0.02)

    def _make_batch_with_targets(self, batch_size=4, n_features=10, seq_len=64,
                                  n_views=2, n_targets=9):
        views = [torch.randn(batch_size, n_features, seq_len) for _ in range(n_views)]
        lengths = [torch.full((batch_size,), seq_len, dtype=torch.long) for _ in range(n_views)]
        targets = torch.randn(batch_size, n_targets)
        return {
            "buckets": [{
                "views": views,
                "lengths": lengths,
                "n_global_views": n_views,
                "targets": targets,
            }],
        }

    def test_npz_file_created(self, tmp_path, monkeypatch):
        """dispatch_probe_eval creates a .npz file with expected arrays."""
        from market_jepa.training.utils import dispatch_probe_eval

        monkeypatch.chdir(tmp_path)

        model = self._make_model()
        batch = self._make_batch_with_targets(batch_size=8, n_targets=3)
        batches = [batch]

        # Patch subprocess.Popen to not actually spawn
        import market_jepa.training.utils as utils_mod
        popen_calls = []
        class FakePopen:
            def __init__(self, *args, **kwargs):
                popen_calls.append((args, kwargs))
        monkeypatch.setattr(utils_mod.subprocess, "Popen", FakePopen)

        results = dispatch_probe_eval(
            model, batches, torch.device("cpu"),
            obs_seen=6400,
            wandb_run_id="test-run-id",
            wandb_project="test-project",
            wandb_entity="test-entity",
            target_names=["return_300", "return_600", "return_900"],
        )
        assert len(results) == 1
        proc, log_path = results[0]
        assert log_path.exists()

        # Check npz was created
        npz_files = list((tmp_path / "temp_probe_eval").glob("*.npz"))
        assert len(npz_files) == 1

        data = np.load(npz_files[0])
        assert "X" in data
        assert "y" in data
        assert "target_names" in data
        assert data["X"].shape[0] == data["y"].shape[0]
        assert data["y"].shape[1] == 3
        assert list(data["target_names"]) == ["return_300", "return_600", "return_900"]

        # Check Popen was called with wandb shared-mode args
        assert len(popen_calls) == 1
        cmd = popen_calls[0][0][0]
        assert "probe_eval_worker.py" in cmd[1]
        assert "--obs_seen" in cmd
        assert "6400" in cmd
        assert "--wandb_run_id" in cmd
        assert "test-run-id" in cmd
        # The k-way binning grid is gone: the worker fits one ridge probe per
        # target column and reports rank IC, so there is no --bins to pass.
        assert "--bins" not in cmd

    def test_model_restored_to_train(self, tmp_path, monkeypatch):
        """dispatch_probe_eval restores model to train mode."""
        from market_jepa.training.utils import dispatch_probe_eval
        import market_jepa.training.utils as utils_mod

        monkeypatch.chdir(tmp_path)

        class FakePopen:
            def __init__(self, *args, **kwargs):
                pass
        monkeypatch.setattr(utils_mod.subprocess, "Popen", FakePopen)

        model = self._make_model()
        model.train()
        batch = self._make_batch_with_targets(batch_size=8, n_targets=3)

        results = dispatch_probe_eval(
            model, [batch], torch.device("cpu"),
            obs_seen=6400,
            wandb_run_id="x",
            wandb_project="x",
            wandb_entity=None,
            target_names=["t1", "t2", "t3"],
        )
        assert len(results) >= 1

        assert model.training  # dispatch_probe_eval calls model.train()

    def test_skip_when_no_valid_data(self, tmp_path, monkeypatch):
        """If all targets are NaN, no npz file or subprocess is created."""
        from market_jepa.training.utils import dispatch_probe_eval
        import market_jepa.training.utils as utils_mod

        monkeypatch.chdir(tmp_path)
        popen_calls = []
        monkeypatch.setattr(utils_mod.subprocess, "Popen",
                            lambda *a, **k: popen_calls.append(1))

        model = self._make_model()
        batch = self._make_batch_with_targets(batch_size=8, n_targets=3)
        batch["buckets"][0]["targets"][:] = float("nan")

        results = dispatch_probe_eval(
            model, [batch], torch.device("cpu"),
            obs_seen=6400,
            wandb_run_id="x",
            wandb_project="x",
            wandb_entity=None,
            target_names=["t1", "t2", "t3"],
        )

        assert len(results) == 0
        assert len(popen_calls) == 0

    def test_pre_split_npz_created(self, tmp_path, monkeypatch):
        """With probe_train_batches, npz contains X_train/y_train/X_val/y_val."""
        from market_jepa.training.utils import dispatch_probe_eval
        import market_jepa.training.utils as utils_mod

        monkeypatch.chdir(tmp_path)

        popen_calls = []
        class FakePopen:
            def __init__(self, *args, **kwargs):
                popen_calls.append((args, kwargs))
        monkeypatch.setattr(utils_mod.subprocess, "Popen", FakePopen)

        model = self._make_model()
        train_batches = [self._make_batch_with_targets(batch_size=8, n_targets=3)]
        eval_batches = [self._make_batch_with_targets(batch_size=6, n_targets=3)]

        results = dispatch_probe_eval(
            model, eval_batches, torch.device("cpu"),
            obs_seen=6400,
            wandb_run_id="test-run-id",
            wandb_project="test-project",
            wandb_entity="test-entity",
            target_names=["return_300", "return_600", "return_900"],
            probe_train_batches=train_batches,
        )
        assert len(results) >= 1

        npz_files = list((tmp_path / "temp_probe_eval").glob("*.npz"))
        assert len(npz_files) == 1

        data = np.load(npz_files[0])
        assert "X_train" in data
        assert "y_train" in data
        assert "X_val" in data
        assert "y_val" in data
        assert "X" not in data  # Legacy keys should not be present
        assert data["X_train"].shape[0] == 8
        assert data["X_val"].shape[0] == 6
        assert data["y_train"].shape[1] == 3
        assert data["y_val"].shape[1] == 3

    def test_pre_split_skip_when_train_empty(self, tmp_path, monkeypatch):
        """With probe_train_batches where all targets are NaN, returns empty list."""
        from market_jepa.training.utils import dispatch_probe_eval
        import market_jepa.training.utils as utils_mod

        monkeypatch.chdir(tmp_path)
        popen_calls = []
        monkeypatch.setattr(utils_mod.subprocess, "Popen",
                            lambda *a, **k: popen_calls.append(1))

        model = self._make_model()
        train_batch = self._make_batch_with_targets(batch_size=8, n_targets=3)
        train_batch["buckets"][0]["targets"][:] = float("nan")
        eval_batches = [self._make_batch_with_targets(batch_size=6, n_targets=3)]

        results = dispatch_probe_eval(
            model, eval_batches, torch.device("cpu"),
            obs_seen=6400,
            wandb_run_id="x",
            wandb_project="x",
            wandb_entity=None,
            target_names=["t1", "t2", "t3"],
            probe_train_batches=[train_batch],
        )

        assert len(results) == 0
        assert len(popen_calls) == 0


# =======================================================================
# Part 10: Pre-split npz loading (worker logic)
# =======================================================================


class TestPreSplitNpzLoading:
    """Test that the worker correctly loads pre-split npz files."""

    def test_pre_split_npz_loaded(self, tmp_path):
        """Pre-split npz with X_train/X_val is loaded without temporal splitting."""
        rng = np.random.RandomState(42)
        n_train, n_val, d, n_targets = 50, 30, 16, 3
        X_train = rng.randn(n_train, d).astype(np.float32)
        y_train = rng.randn(n_train, n_targets).astype(np.float32)
        X_val = rng.randn(n_val, d).astype(np.float32)
        y_val = rng.randn(n_val, n_targets).astype(np.float32)

        npz_path = tmp_path / "test_presplit.npz"
        np.savez(npz_path, X_train=X_train, y_train=y_train,
                 X_val=X_val, y_val=y_val,
                 target_names=np.array(["r300", "r600", "r900"]))

        data = np.load(npz_path, allow_pickle=False)
        assert "X_train" in data
        assert "X" not in data

        X_tr = data["X_train"]
        y_tr = data["y_train"]
        X_va = data["X_val"]
        y_va = data["y_val"]

        assert X_tr.shape == (n_train, d)
        assert y_tr.shape == (n_train, n_targets)
        assert X_va.shape == (n_val, d)
        assert y_va.shape == (n_val, n_targets)

    def test_legacy_npz_loaded(self, tmp_path):
        """Legacy npz with X/y is loaded and split temporally."""
        rng = np.random.RandomState(42)
        n, d, n_targets = 80, 16, 3
        X = rng.randn(n, d).astype(np.float32)
        y = rng.randn(n, n_targets).astype(np.float32)

        npz_path = tmp_path / "test_legacy.npz"
        np.savez(npz_path, X=X, y=y,
                 target_names=np.array(["r300", "r600", "r900"]))

        data = np.load(npz_path, allow_pickle=False)
        assert "X" in data
        assert "X_train" not in data

        # Simulate temporal split
        temporal_train_frac = 0.5
        split_idx = int(n * temporal_train_frac)
        X_train = data["X"][:split_idx]
        X_val = data["X"][split_idx:]
        assert X_train.shape[0] == 40
        assert X_val.shape[0] == 40


# =======================================================================
# Part 8: Rank IC — the reported metric
# =======================================================================


class TestRankIC:
    """``rank_ic`` / ``grouped_rank_ic`` against scipy, plus the guard rails."""

    def test_matches_scipy_spearman(self):
        from scipy.stats import spearmanr

        rng = np.random.RandomState(0)
        x = rng.randn(400)
        y = 0.4 * x + rng.randn(400)
        assert rank_ic(x, y) == pytest.approx(spearmanr(x, y).statistic, rel=1e-10)

    def test_ties_use_midranks(self):
        """Heavy tie mass is the norm here: spread_change is mostly zeros and
        wide-quote names never tick, so mid-rank averaging is what keeps those
        from dragging the correlation."""
        from scipy.stats import spearmanr

        rng = np.random.RandomState(1)
        x = rng.randint(0, 3, size=300).astype(float)
        y = rng.randint(0, 4, size=300).astype(float)
        assert rank_ic(x, y) == pytest.approx(spearmanr(x, y).statistic, rel=1e-10)

    def test_perfect_and_inverted_signal(self):
        x = np.arange(100, dtype=float)
        assert rank_ic(x, x) == pytest.approx(1.0)
        assert rank_ic(x, -x) == pytest.approx(-1.0)

    def test_monotone_transform_is_invariant(self):
        """Rank IC must not care about the predictor's scale or offset — the
        SSL ridge probe and the supervised head are compared on the same
        footing only if this holds."""
        rng = np.random.RandomState(2)
        x = rng.randn(200)
        y = rng.randn(200)
        assert rank_ic(x, y) == pytest.approx(rank_ic(3.0 * x + 7.0, y))
        assert rank_ic(x, y) == pytest.approx(rank_ic(np.exp(x), y))

    def test_nan_rows_dropped(self):
        rng = np.random.RandomState(3)
        x = rng.randn(200)
        y = 0.5 * x + rng.randn(200)
        xn, yn = x.copy(), y.copy()
        xn[:20] = np.nan
        yn[180:] = np.nan
        expected = rank_ic(x[20:180], y[20:180])
        assert rank_ic(xn, yn) == pytest.approx(expected)

    def test_too_few_rows_is_nan(self):
        x = np.arange(MIN_CELL_N - 1, dtype=float)
        assert np.isnan(rank_ic(x, x))

    def test_constant_side_is_nan(self):
        x = np.ones(100)
        y = np.arange(100, dtype=float)
        assert np.isnan(rank_ic(x, y))

    def test_grouped_averages_per_cell(self):
        from scipy.stats import spearmanr

        rng = np.random.RandomState(4)
        n_cells, per_cell = 8, 60
        x = rng.randn(n_cells * per_cell)
        y = 0.3 * x + rng.randn(n_cells * per_cell)
        g = np.repeat(np.arange(n_cells), per_cell)

        mean_ic, se, n = grouped_rank_ic(x, y, g)
        by_hand = [
            spearmanr(x[g == i], y[g == i]).statistic for i in range(n_cells)
        ]
        assert n == n_cells
        assert mean_ic == pytest.approx(float(np.mean(by_hand)), rel=1e-10)
        assert se == pytest.approx(
            float(np.std(by_hand, ddof=1) / np.sqrt(n_cells)), rel=1e-10
        )

    def test_grouped_skips_undersized_cells(self):
        rng = np.random.RandomState(5)
        big = MIN_CELL_N * 3
        x = np.concatenate([rng.randn(big), rng.randn(MIN_CELL_N - 1)])
        y = np.concatenate([rng.randn(big), rng.randn(MIN_CELL_N - 1)])
        g = np.concatenate([np.zeros(big), np.ones(MIN_CELL_N - 1)])
        _, _, n = grouped_rank_ic(x, y, g)
        assert n == 1

    def test_grouped_unsorted_groups(self):
        """Group ids arrive interleaved (one row per stock per anchor), so the
        implementation must not assume they are contiguous."""
        rng = np.random.RandomState(6)
        n_cells, per_cell = 4, 50
        x = rng.randn(n_cells * per_cell)
        y = 0.3 * x + rng.randn(n_cells * per_cell)
        g = np.tile(np.arange(n_cells), per_cell)

        mean_ic, _, n = grouped_rank_ic(x, y, g)
        assert n == n_cells
        assert np.isfinite(mean_ic)

    def test_grouped_all_cells_too_small_is_nan(self):
        x = np.arange(10, dtype=float)
        mean_ic, se, n = grouped_rank_ic(x, x, np.zeros(10))
        assert n == 0
        assert np.isnan(mean_ic) and np.isnan(se)
