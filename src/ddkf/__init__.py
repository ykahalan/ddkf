"""
DDKF - Dual Dynamic Kernel Filtering

PyTorch implementation with:
- Arbitrary number of kernels
- Learnable parameters (alpha, beta, gamma)
- Backpropagatable cubic interpolation
- No scipy dependency

Quick Start
-----------
>>> import torch
>>> from ddkf import DDKFLayer
>>> 
>>> # Hybrid kernel with interpolation (matches TwoDKF)
>>> layer = DDKFLayer(
...     kernel_names=['polynomial', 'gaussian'],
...     gamma=[0.5, 0.5],  # 50% poly, 50% gaussian
...     interp_factor=0.25,  # 4x upsampling
...     alpha=0.15,
...     beta=0.9
... )
>>> 
>>> signal = torch.randn(16, 1000)
>>> tfr = layer(signal)

Training (All Parameters Learnable)
------------------------------------
>>> # Create layer
>>> layer = DDKFLayer(
...     kernel_names=['polynomial', 'gaussian', 'polynomial'],
...     gamma=[0.5, 0.3, 0.2],  # Learnable!
...     alpha=0.15,  # Learnable!
...     beta=0.9,    # Learnable!
...     interp_factor=0.25  # Cubic interpolation
... )
>>> 
>>> # Training loop
>>> optimizer = torch.optim.Adam(layer.parameters())
>>> for signal, target in dataloader:
...     tfr = layer(signal)
...     loss = criterion(tfr, target)
...     loss.backward()  # Gradients flow through interpolation!
...     optimizer.step()

Features
--------
- **Cubic Interpolation**: Backpropagatable via torch.nn.functional.grid_sample
- **Multiple Kernels**: Combine any number of polynomial/gaussian kernels
- **Learnable Weights**: alpha, beta, and gamma all support backpropagation
- **No scipy**: Pure PyTorch implementation
"""

from .ddkf_pytorch import (
    DDKFLayer,
    DDKFFeatureExtractor,
    Kernels,
    cubic_interpolate_1d,
)

__version__ = "3.0.0"
__all__ = [
    'DDKFLayer',
    'DDKFFeatureExtractor', 
    'Kernels',
    'cubic_interpolate_1d',
]
