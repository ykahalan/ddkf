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
>>> # Hybrid kernel with interpolation (matches 2DKF)
>>> layer = DDKFLayer(
...     kernel_names=['polynomial', 'gaussian'],
...     gamma=[0.5, 0.5],  # 50% poly, 50% gaussian
...     interp_factor=0.25,  # 4x upsampling
...     alpha=0.12   # Beta threshold (paper default)
...     beta=0.9,   # Smart minimum threshold (paper default)
... )
>>> 
>>> signal = torch.randn(16, 1000)
>>> tfr = layer(signal)

Training (All Parameters Learnable)
------------------------------------
>>> # Create layer
>>> layer = DDKFLayer(
...     kernel_names=['polynomial', 'gaussian', 'polynomial'],
...     gamma=[0.5, 0.3, 0.2],     # Learnable!
...     alpha=0.12,          # Learnable!
...     beta=0.9,           # Learnable!
...     interp_factor=0.25         # Cubic interpolation
... )
>>> 
>>> # Training loop
>>> optimizer = torch.optim.Adam(layer.parameters())
>>> for signal, target in dataloader:
...     tfr = layer(signal)
...     loss = criterion(tfr, target)
...     loss.backward()  # Gradients flow through interpolation!
...     optimizer.step()

Parameter Names (Following Paper Convention)
---------------------------------------------
- **alpha** (default=0.12): Local threshold (alpha in paper equation 4)
  - Suppresses weak frequency components in final TFR
  - Applied AFTER smart minimum operation
  - Higher = more aggressive smoothing
  
- **beta** (default=0.9): Global threshold (beta in paper equation 5)
  - Controls which frequency components participate in smart minimum
  - Applied DURING smart minimum operation  
  - Higher = more selective (only strongest frequencies)

- **gamma**: Kernel mixing weights (learnable, auto-normalized)
"""

from .ddkf import (
    DDKFLayer,
    DDKFFeatureExtractor,
    Kernels,
    cubic_interpolate_1d,
)

__version__ = "4.0.0"  # Major version bump due to breaking changes
__all__ = [
    'DDKFLayer',
    'DDKFFeatureExtractor', 
    'Kernels',
    'cubic_interpolate_1d',
]
