"""
Tests for model components.

Run with: pytest tests/test_models.py -v
"""

import numpy as np
import pytest
import torch

from fusiongp.models.kernels import SpatioTemporalKernel, create_base_kernel
from fusiongp.models.likelihoods import MultiSourceLikelihood
from fusiongp.models.svgp import FusionSVGP


class TestSpatioTemporalKernel:
    """Tests for SpatioTemporalKernel class."""
    
    def test_kernel_creation(self):
        """Test basic kernel creation."""
        kernel = SpatioTemporalKernel()
        
        assert kernel is not None
        assert kernel.spatial_ard == True
    
    def test_kernel_forward(self):
        """Test kernel forward pass."""
        kernel = SpatioTemporalKernel()
        
        # Create random inputs: (N, 3) with [lat, lon, time]
        x = torch.randn(10, 3)
        
        K = kernel(x, x).to_dense()
        
        assert K.shape == (10, 10)
        # Covariance matrix should be symmetric
        assert torch.allclose(K, K.T, atol=1e-5)
        # Diagonal should be positive
        assert (K.diag() > 0).all()
    
    def test_kernel_types(self):
        """Test different kernel types."""
        for kernel_type in ['matern12', 'matern32', 'matern52', 'rbf']:
            kernel = SpatioTemporalKernel(
                spatial_kernel_type=kernel_type,
                temporal_kernel_type=kernel_type,
            )
            
            x = torch.randn(5, 3)
            K = kernel(x, x).to_dense()
            
            assert K.shape == (5, 5)
    
    def test_lengthscale_initialization(self):
        """Test custom lengthscale initialization."""
        kernel = SpatioTemporalKernel(
            initial_spatial_lengthscale=(0.05, 0.05),
            initial_temporal_lengthscale=0.2,
        )
        
        assert kernel.temporal_lengthscale.item() == pytest.approx(0.2, rel=1e-4)
    
    def test_hyperparameter_dict(self):
        """Test get_hyperparameters method."""
        kernel = SpatioTemporalKernel()
        params = kernel.get_hyperparameters()
        
        assert 'spatial_lengthscale' in params
        assert 'temporal_lengthscale' in params
        assert 'outputscale' in params


class TestMultiSourceLikelihood:
    """Tests for MultiSourceLikelihood class."""
    
    def test_likelihood_creation(self):
        """Test basic likelihood creation."""
        likelihood = MultiSourceLikelihood()
        
        assert likelihood is not None
        assert 'epa' in likelihood.sources
        assert 'low_cost' in likelihood.sources
        assert 'satellite' in likelihood.sources
    
    def test_noise_properties(self):
        """Test noise parameter properties."""
        likelihood = MultiSourceLikelihood(
            initial_noise={'epa': 1.0, 'low_cost': 5.0, 'satellite': 3.0}
        )
        
        noise_std = likelihood.noise_std
        
        assert noise_std['epa'].item() == pytest.approx(1.0, rel=0.1)
        assert noise_std['low_cost'].item() == pytest.approx(5.0, rel=0.1)
        assert noise_std['satellite'].item() == pytest.approx(3.0, rel=0.1)
    
    def test_calibration_parameters(self):
        """Test calibration parameters for low-cost sensors."""
        likelihood = MultiSourceLikelihood(
            initial_calibration={'slope': 1.2, 'intercept': 2.0}
        )
        
        assert likelihood.lc_slope.item() == pytest.approx(1.2, rel=0.1)
        assert likelihood.lc_intercept.item() == pytest.approx(2.0, rel=0.1)
    
    def test_transform_latent(self):
        """Test latent transformation for different sources."""
        likelihood = MultiSourceLikelihood()
        
        f = torch.tensor([10.0, 20.0, 30.0])
        
        # EPA and satellite should be identity
        f_epa = likelihood.transform_latent(f, 'epa')
        f_sat = likelihood.transform_latent(f, 'satellite')
        
        assert torch.allclose(f_epa, f)
        assert torch.allclose(f_sat, f)
        
        # Low-cost should apply calibration
        f_lc = likelihood.transform_latent(f, 'low_cost')
        expected = likelihood.lc_slope * f + likelihood.lc_intercept
        assert torch.allclose(f_lc, expected)
    
    def test_log_marginal(self):
        """Test log marginal likelihood computation."""
        from gpytorch.distributions import MultivariateNormal
        
        likelihood = MultiSourceLikelihood()
        
        # Create mock GP distribution
        n = 10
        mean = torch.randn(n)
        var = torch.rand(n) + 0.1
        
        # Create diagonal MultivariateNormal
        covar = torch.diag(var)
        f_dist = MultivariateNormal(mean, covar)
        
        # Create observations
        y = torch.randn(n, 3)  # 3 sources
        masks = torch.ones(n, 3, dtype=torch.bool)
        
        log_prob = likelihood.log_marginal(y, f_dist, masks)
        
        assert log_prob.dim() == 0  # Scalar
        assert not torch.isnan(log_prob)


class TestFusionSVGP:
    """Tests for FusionSVGP model class."""
    
    def test_model_creation(self):
        """Test basic model creation."""
        model = FusionSVGP(n_inducing=50)
        
        assert model is not None
        assert model.n_inducing == 50
    
    def test_inducing_point_initialization(self):
        """Test inducing point initialization."""
        model = FusionSVGP(n_inducing=20)
        
        # Create some training data
        x = torch.randn(100, 3)
        
        model.initialize_inducing_points(x, method='kmeans')
        
        inducing = model.get_inducing_points()
        assert inducing.shape == (20, 3)
    
    def test_forward_pass(self):
        """Test model forward pass."""
        model = FusionSVGP(n_inducing=20)
        
        x_train = torch.randn(100, 3)
        model.initialize_inducing_points(x_train)
        
        x_test = torch.randn(10, 3)
        
        model.eval()
        with torch.no_grad():
            dist = model(x_test)
        
        assert dist.mean.shape == (10,)
        assert dist.variance.shape == (10,)
    
    def test_elbo_computation(self):
        """Test ELBO computation."""
        model = FusionSVGP(n_inducing=20)
        
        x = torch.randn(50, 3)
        model.initialize_inducing_points(x)
        
        y = torch.randn(50, 3)  # 3 sources
        masks = torch.ones(50, 3, dtype=torch.bool)
        
        model.train()
        elbo = model.elbo(x, y, masks)
        
        assert elbo.dim() == 0  # Scalar
        assert not torch.isnan(elbo)
    
    def test_prediction(self):
        """Test model prediction."""
        model = FusionSVGP(n_inducing=20)
        
        x_train = torch.randn(100, 3)
        model.initialize_inducing_points(x_train)
        
        x_test = torch.randn(10, 3)
        
        mean, var = model.predict(x_test)
        
        assert mean.shape == (10,)
        assert var.shape == (10,)
        assert (var > 0).all()
    
    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        model = FusionSVGP(n_inducing=20)
        
        x = torch.randn(50, 3)
        model.initialize_inducing_points(x)
        
        y = torch.randn(50, 3)
        masks = torch.ones(50, 3, dtype=torch.bool)
        
        model.train()
        model.likelihood.train()
        
        elbo = model.elbo(x, y, masks)
        loss = -elbo
        loss.backward()
        
        # Check that some gradients are non-zero
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "No gradients computed"


class TestCreateBaseKernel:
    """Tests for create_base_kernel function."""
    
    def test_matern_kernels(self):
        """Test Matérn kernel creation."""
        for kernel_type in ['matern12', 'matern32', 'matern52']:
            kernel = create_base_kernel(kernel_type)
            assert kernel is not None
    
    def test_rbf_kernel(self):
        """Test RBF kernel creation."""
        kernel = create_base_kernel('rbf')
        assert kernel is not None
    
    def test_invalid_kernel(self):
        """Test error on invalid kernel type."""
        with pytest.raises(ValueError):
            create_base_kernel('invalid')
    
    def test_active_dims(self):
        """Test active_dims parameter."""
        kernel = create_base_kernel(
            'matern32',
            active_dims=torch.tensor([0, 1])
        )
        
        assert kernel is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
