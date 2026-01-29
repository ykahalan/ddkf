"""
DDKF - Dual Dynamic Kernel Filtering

PyTorch implementation with:
- Arbitrary number of kernels
- Learnable parameters (alpha, beta, gamma)
- Backpropagatable cubic interpolation
- No scipy dependency
- MATLAB-accurate behavior

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
...     alpha=0.12   # Beta threshold (MATLAB default)
...     beta=0.9,   # Smart minimum threshold (MATLAB default)
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

Parameter Names (CORRECTED to match MATLAB)
--------------------------------------------
- **alpha** (default=0.12): Beta threshold for final smoothing
  - Suppresses weak frequency components in final TFR
  - Higher = more aggressive smoothing
  
- **beta** (default=0.9): Smart minimum threshold for alpha masking
  - Controls which frequency components participate in smart minimum
  - Higher = more selective (only strongest frequencies)

- **gamma**: Kernel mixing weights (learnable, auto-normalized)

Features
--------
- **Cubic Interpolation**: Backpropagatable via torch.nn.functional.grid_sample
- **Multiple Kernels**: Combine any number of polynomial/gaussian kernels
- **Learnable Weights**: c_smart_min, c_smoothing, and gamma all support backpropagation
- **No scipy**: Pure PyTorch implementation

Algorithm Overview
------------------
1. **Interpolate** signal using cubic splines (4x upsampling by default)
2. **Sliding Window** with kernel application:
   - First pass: Apply kernel WITHIN window, zeros outside → FFT → M
   - Second pass: Apply kernel OUTSIDE window, zeros inside → FFT → M1
3. **Alpha Threshold**: Suppress result where result < alpha * max(result)
4. **Smart Minimum**: result = min(M, M1 * M * mask) where mask = (M > beta * max(M))
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
