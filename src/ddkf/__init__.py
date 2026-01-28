"""
DDKF - Dual Dynamic Kernel Filtering

PyTorch implementation with:
- Arbitrary number of kernels
- Learnable parameters (c_smart_min, c_smoothing, gamma)
- Backpropagatable cubic interpolation
- No scipy dependency
- MATLAB-accurate behavior

Quick Start
-----------
>>> import torch
>>> from ddkf import DDKFLayer
>>> 
>>> # Hybrid kernel with interpolation (matches MATLAB TwoDKF)
>>> layer = DDKFLayer(
...     kernel_names=['polynomial', 'gaussian'],
...     gamma=[0.5, 0.5],  # 50% poly, 50% gaussian
...     interp_factor=0.25,  # 4x upsampling
...     c_smart_min=0.9,   # Smart minimum threshold (MATLAB default)
...     c_smoothing=0.12   # Beta threshold (MATLAB default)
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
...     c_smart_min=0.9,           # Learnable!
...     c_smoothing=0.12,          # Learnable!
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
- **c_smart_min** (default=0.9): Smart minimum threshold for alpha masking
  - Controls which frequency components participate in smart minimum
  - MATLAB equivalent: c_smart_min
  - Higher = more selective (only strongest frequencies)

- **c_smoothing** (default=0.12): Beta threshold for final smoothing
  - Suppresses weak frequency components in final TFR
  - MATLAB equivalent: c_smoothing  
  - Higher = more aggressive smoothing

- **gamma**: Kernel mixing weights (learnable, auto-normalized)

Backward Compatibility
----------------------
If you have old code using `alpha` and `beta` parameter names:

>>> # OLD CODE (incorrect parameter names):
>>> # layer = DDKFLayer(alpha=0.15, beta=0.9)
>>> 
>>> # NEW CODE (correct parameter names):
>>> layer = DDKFLayer(c_smoothing=0.15, c_smart_min=0.9)
>>>
>>> # OR use the compatibility helper:
>>> from ddkf import create_ddkf_with_old_params
>>> layer = create_ddkf_with_old_params(alpha=0.15, beta=0.9)

Features
--------
- **Cubic Interpolation**: Backpropagatable via torch.nn.functional.grid_sample
- **Multiple Kernels**: Combine any number of polynomial/gaussian kernels
- **Learnable Weights**: c_smart_min, c_smoothing, and gamma all support backpropagation
- **No scipy**: Pure PyTorch implementation
- **MATLAB-accurate**: Window-by-window kernel application matches reference implementation

Algorithm Overview
------------------
1. **Interpolate** signal using cubic splines (4x upsampling by default)
2. **Sliding Window** with kernel application:
   - First pass: Apply kernel WITHIN window, zeros outside → FFT → M
   - Second pass: Apply kernel OUTSIDE window, zeros inside → FFT → M1
3. **Smart Minimum**: result = min(M, M1 * M * mask) where mask = (M > c_smart_min * max(M))
4. **Beta Threshold**: Suppress result where result < c_smoothing * max(result)

Important Implementation Notes
------------------------------
The corrected implementation applies kernels **window-by-window**, not globally:
- ✓ CORRECT: Extract window → Apply kernel → Pad with zeros → FFT
- ✗ WRONG: Apply kernel to entire signal → Extract window → FFT

This matches the MATLAB reference implementation exactly.
"""

from .ddkf import (
    DDKFLayer,
    DDKFFeatureExtractor,
    Kernels,
    cubic_interpolate_1d,
    create_ddkf_with_old_params,  # Backward compatibility
)

__version__ = "4.0.0"  # Major version bump due to breaking changes
__all__ = [
    'DDKFLayer',
    'DDKFFeatureExtractor', 
    'Kernels',
    'cubic_interpolate_1d',
    'create_ddkf_with_old_params',
]

# Deprecation warning for old imports
def __getattr__(name):
    """Provide helpful warnings for deprecated usage."""
    if name == 'alpha':
        import warnings
        warnings.warn(
            "Parameter 'alpha' has been renamed to 'c_smoothing' to match MATLAB. "
            "Use DDKFLayer(c_smoothing=...) instead of DDKFLayer(alpha=...)",
            DeprecationWarning,
            stacklevel=2
        )
        return 'c_smoothing'
    elif name == 'beta':
        import warnings
        warnings.warn(
            "Parameter 'beta' has been renamed to 'c_smart_min' to match MATLAB. "
            "Use DDKFLayer(c_smart_min=...) instead of DDKFLayer(beta=...)",
            DeprecationWarning,
            stacklevel=2
        )
        return 'c_smart_min'
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
