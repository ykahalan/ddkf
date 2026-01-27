"""
DDKF - Dual Dynamic Kernel Filtering

Minimal, clean implementation with:
- Arbitrary number of kernels
- Learnable parameters (alpha, beta, gamma)
- NumPy and PyTorch support
- No scipy dependency

Quick Start
-----------
>>> from ddkf import DDKF, denoise
>>> 
>>> # Simple denoising
>>> clean = denoise(noisy_signal, kernel="gaussian", alpha=0.15)
>>> 
>>> # Multiple kernels
>>> dkf = DDKF(
...     kernel=["polynomial", "gaussian"],
...     gamma=[0.7, 0.3],  # 70% poly, 30% gaussian
...     alpha=0.15
... )
>>> dkf.fit(signal)
>>> tfr = dkf.get_tfr()

PyTorch (Learnable Parameters)
-------------------------------
>>> import torch
>>> from ddkf import DDKFLayer
>>> 
>>> # Arbitrary number of kernels!
>>> layer = DDKFLayer(
...     kernel_names=['polynomial', 'gaussian', 'polynomial'],
...     gamma=[0.5, 0.3, 0.2],  # Learnable!
...     alpha=0.15,  # Learnable!
...     beta=0.9,    # Learnable!
... )
>>> 
>>> tfr = layer(torch.randn(16, 1000))
>>> loss = criterion(tfr, target)
>>> loss.backward()  # Gradients flow through all parameters!
"""

from .ddkf_final import (
    DDKF,
    DDKFLayer,
    DDKFFeatureExtractor,
    Kernels,
    denoise,
    HAS_TORCH,
)

__version__ = "3.0.0"
__all__ = ['DDKF', 'DDKFLayer', 'DDKFFeatureExtractor', 'Kernels', 'denoise', 'HAS_TORCH']
