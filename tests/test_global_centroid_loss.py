"""Tests for global-only centroid loss in LeJEPA.compute_loss."""

import torch
import pytest

from market_jepa.modeling import LeJEPA, create_backbone


@pytest.fixture
def model():
    """Small JEPA model for testing."""
    backbone = create_backbone(backbone_type="resnet", n_features=9, d_embedding=64)
    return LeJEPA(backbone=backbone, proj_dim=32, lamb=0.02)


class TestGlobalCentroidLoss:
    def test_centroid_from_global_only(self, model):
        """n_global_views=2, shape (B, 8, D) → centroid from first 2."""
        B, n_views, D = 4, 8, 32
        proj = torch.randn(B, n_views, D)

        result = model.compute_loss(proj, n_global_views=2)

        # Verify centroid was computed from first 2 views
        mu_expected = proj[:, :2, :].mean(dim=1, keepdim=True)
        inv_expected = (mu_expected - proj).square().mean()
        torch.testing.assert_close(result["inv_loss"], inv_expected)

    def test_fallback_when_none(self, model):
        """n_global_views=None → centroid from all (existing behavior)."""
        B, n_views, D = 4, 8, 32
        proj = torch.randn(B, n_views, D)

        result_none = model.compute_loss(proj, n_global_views=None)
        result_default = model.compute_loss(proj)

        torch.testing.assert_close(result_none["inv_loss"], result_default["inv_loss"])

    def test_all_global_equivalent(self, model):
        """n_global_views=n_views → same as current behavior."""
        B, n_views, D = 4, 4, 32
        proj = torch.randn(B, n_views, D)

        result_all = model.compute_loss(proj, n_global_views=n_views)
        result_default = model.compute_loss(proj)

        torch.testing.assert_close(result_all["inv_loss"], result_default["inv_loss"])

    def test_gradient_flows(self, model):
        """loss.backward() succeeds with n_global_views."""
        B, n_views, D = 4, 8, 32
        proj = torch.randn(B, n_views, D, requires_grad=True)

        result = model.compute_loss(proj, n_global_views=2)
        result["lejepa_loss"].backward()

        assert proj.grad is not None
        assert not torch.all(proj.grad == 0)

    def test_sigreg_unaffected(self, model):
        """SIGReg loss identical regardless of n_global_views (same RNG seed)."""
        B, n_views, D = 4, 8, 32
        proj = torch.randn(B, n_views, D)

        # SIGReg draws a random projection matrix each call, so fix seed
        torch.manual_seed(0)
        result_global = model.compute_loss(proj, n_global_views=2)
        torch.manual_seed(0)
        result_all = model.compute_loss(proj, n_global_views=None)

        torch.testing.assert_close(
            result_global["sigreg_loss"], result_all["sigreg_loss"],
        )

    def test_output_keys(self, model):
        """Dict has raw and normalized loss keys."""
        proj = torch.randn(4, 8, 32)
        result = model.compute_loss(proj, n_global_views=2)
        assert set(result.keys()) == {
            "lejepa_loss", "sigreg_loss", "inv_loss",
            "sigreg_loss_normalized", "lejepa_loss_normalized",
        }
