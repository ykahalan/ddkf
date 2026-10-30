"""
DDKF - Dual Dynamic Kernel Filtering

PyTorch implementation with:
- Arbitrary number of kernels (built-in, custom callables or learnable nn.Modules)
- Learnable parameters (alpha, beta, gamma, sigmoid temperature)
- Backpropagatable cubic interpolation
- Vectorised forward pass (batched FFT, no Python loop over windows)
- No scipy dependency

Quick Start
-----------
>>> import torch
>>> from ddkf import DDKFLayer
>>>
>>> layer = DDKFLayer(
...     kernel_names=['polynomial', 'gaussian'],
...     gamma=[0.5, 0.5],      # 50% poly, 50% gaussian
...     interp_factor=0.25,    # 4x upsampling
...     alpha=0.12,            # global threshold (paper default)
...     beta=0.9,              # smart minimum threshold (paper default)
... )
>>>
>>> signal = torch.randn(16, 1000)
>>> tfr = layer(signal)                 # (16, n_windows, n_freqs)
>>> recovered = layer.inverse_transform(tfr)

Training (all parameters learnable)
-----------------------------------
>>> layer = DDKFLayer(
...     kernel_names=['polynomial', 'gaussian'],
...     threshold_mode='soft',     # differentiable thresholds
...     learn_alpha=True,
...     learn_beta=True,
...     learn_gamma=True,
...     learn_sigmoid_temp=True,
... )
>>> optimizer = torch.optim.Adam(layer.parameters())
>>> for signal, target in dataloader:
...     tfr = layer(signal)
...     loss = criterion(tfr, target)
...     loss.backward()
...     optimizer.step()

Parameter names (paper convention)
----------------------------------
- alpha (default 0.12): global threshold (paper eq. 4), applied AFTER the smart minimum
- beta  (default 0.9):  smart-minimum threshold (paper eq. 5), applied DURING the smart minimum
- gamma: kernel mixing weights, normalised to sum to 1
"""

from .ddkf import (
    DDKFLayer,
    DDKFFeatureExtractor,
    NTFALayer,
    NTFAFeatureExtractor,
    Kernels,
    LearnableChebyshevKernel,
    cubic_interpolate_1d,
    soft_minimum,
)

__version__ = "4.2.0"
__all__ = [
    'DDKFLayer',
    'DDKFFeatureExtractor',
    'NTFALayer',
    'NTFAFeatureExtractor',
    'Kernels',
    'LearnableChebyshevKernel',
    'cubic_interpolate_1d',
    'soft_minimum',
]
