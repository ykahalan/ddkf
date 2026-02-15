# ntfa-lk

Minimal package providing `NTFA-LK` for time-frequency decomposition with arbitrary kernels. Available to install with pip as package `ntfa-lk`.

## Quick install

```bash
# Basic install
python3 -m pip install ntfa-lk
```

## ⚠️ Important: Use the Polynomial Kernel

**The polynomial kernel performs exceptionally well in any combination and should be included in your kernel configuration.** We don't set it as the default to allow users flexibility in configuring the polynomial parameters (degree and offset) for their specific use case.

**Recommended configuration:**
```python
kernel_names=['polynomial', 'gaussian']
gamma=[0.5, 0.5]
```

## Examples

The `examples/` folder contains complete working examples:

- **run_example.py** - Basic NTFA-LK demonstration showing signal decomposition, time-frequency representation, and inverse transform with a noisy multi-frequency signal (3 Hz + 7 Hz + noise)

- **NTFAvsDWT.py** - Machine learning comparison demonstrating NTFA-LK vs DWT (Discrete Wavelet Transform) as feature extractors for time series classification:
  - Extracts 2D time-frequency features using NTFA-LK
  - Extracts 2D wavelet coefficients using DWT
  - Trains separate CNNs on each feature type
  - Compares classification accuracy on ECG200 dataset
  - Visualizes feature representations and training curves

To run the examples:
```bash
cd examples/
python3 run_example.py        # Basic signal processing demo
python3 NTFAvsDWT.py          # ML classification comparison (requires aeon, pywt)
```

## API Overview

### Main Class: `NTFA-LK`

```python
DDKF(
    kernel="gaussian",           # Single kernel or list of kernels
    gamma=None,                  # Kernel weights (auto-normalized to sum=1)
    alpha=0.15,            # Alpha threshold (smoothing coefficient)
    beta=0.9,             # Beta threshold (smart minimum)
    window_size=20,              # Sliding window size
    step_size=4,                 # Step between windows
    kernel_params=None           # Parameters for each kernel
)
```

**Methods:**
- `fit(signal)` - Process signal and compute TFR
- `get_tfr()` - Get time-frequency representation
- `inverse_transform(correction_factor=None)` - Reconstruct signal

### Convenience Function: `denoise`

```python
denoised = denoise(
    signal,
    kernel=["polynomial", "gaussian"],
    gamma=[0.6, 0.4],
    window_size=20,
    alpha=0.15,
    beta=0.9
)
```

### PyTorch Layer: `NTFALayer`

```python
NTFALayer(
    kernel_names=['polynomial', 'gaussian'],
    gamma=[0.5, 0.5],
    alpha=0.15,            # Alpha threshold
    beta=0.9,             # Beta threshold
    threshold_mode='hard', # 'hard' or 'soft' (differentiable)
    window_size=20,
    step_size=4,
    interp_factor=0.25,    # Cubic interpolation factor
    learn_alpha=False,     # Make alpha learnable
    learn_beta=False,      # Make beta learnable
    learn_sigmoid_temp=False,  # Make sigmoid temp learnable (soft mode only)
    sigmoid_temp=1.0       # Temperature for soft operations
)
```

**All parameters are learnable via backpropagation.**

**Forward pass:**
```python
tfr = layer(signal)  # Returns time-frequency representation
```

**Inverse transform:**
```python
recovered = layer.inverse_transform(tfr)  # Phase automatically stored
# Or provide phase explicitly:
recovered = layer.inverse_transform(tfr, tfr_phase)
```

## Parameter Descriptions

### alpha (default: 0.15)
Alpha threshold for final smoothing. Suppresses weak frequency components in the final time-frequency representation. Higher values result in more aggressive smoothing.

### beta (default: 0.9)
Beta threshold for the smart minimum operation. Controls which frequency components participate in the smart minimum calculation. Higher values (closer to 1.0) make the filter more selective, only including the strongest frequency components.

### gamma (default: equal weights)
Kernel mixing weights. Automatically normalized to sum to 1. For a hybrid kernel with two components, `gamma=[0.5, 0.5]` gives equal weight to each kernel.

### threshold_mode (default: 'hard')
Controls differentiability of thresholding operations:
- `'hard'`: Binary masking (faster, non-differentiable)
- `'soft'`: Smooth gradients via LogSumExp (fully differentiable, enables learning)

### sigmoid_temp (default: 1.0)
Temperature parameter for soft operations (only used when threshold_mode='soft'). Higher values approach hard thresholding. Can be made learnable with learn_sigmoid_temp=True.

## Available Kernels

- `"polynomial"` - Polynomial kernel: (x + offset)^degree
  - Default params: `degree=2, offset=1.3`
- `"gaussian"` - Gaussian kernel: exp(-0.5 * ((x - center) / sigma)^2)
  - Default params: `center=0.7, sigma=1.0`
- `"matern32"` - Matérn 3/2 kernel: (1 + √3(x + offset)) * exp(-√3(x + offset))
  - Default params: offset=1.7
- `"matern52"` - Matérn 5/2 kernel: (1 + √5(x + offset) + (5/3)(x + offset)²) * exp(-√5(x + offset))
  - Default params: offset=1.7
- `"rational"` - Rational quadratic kernel: (1 + scale * x)^(-power)
  - Default params: scale=1/3, power=3
- `"gamma_rational"` - Gamma rational quadratic kernel: (1 + scale * (x + offset)²)^(-power)
  - Default params: scale=1/3, offset=1.7, power=3
- **LearnableChebyshevKernel** - Learnable Chebyshev polynomial kernel with trainable coefficients
  - Example: `LearnableChebyshevKernel(degree=4, init_coeffs=[0, 1, 0, 0])`

**Custom kernels:** You can also pass your own callable functions.
```

## Key Features

- **Window-by-window processing**: Kernels applied correctly within each window
- **Arbitrary kernels**: Use 1, 2, 3, or more kernels
- **No scipy**: Pure NumPy/PyTorch implementation
- **Flexible**: Works for denoising, TFR, feature extraction

## Custom Kernels

You can provide your own kernel functions:
```python
import torch
from ntfa_lk import NTFALayer

# Define custom kernel
def my_custom_kernel(x, scale=2.0, power=3):
    """Custom kernel function."""
    return (x * scale) ** power

# Use with DDKF
layer = NTFALayer(
    kernel_names=[my_custom_kernel, 'gaussian'],  # Mix custom + builtin
    kernel_params=[
        {'scale': 1.5, 'power': 2},  # params for custom kernel
        {'center': 0.5, 'sigma': 0.8}  # params for gaussian
    ],
    gamma=[0.6, 0.4]
)

# Or use lambda functions
layer = NTFALayer(
    kernel_names=[
        lambda x, scale=1.0: torch.exp(-x * scale),
        'polynomial'
    ],
    kernel_params=[
        {'scale': 2.0},
        {'degree': 3, 'offset': 1.0}
    ]
)
```

## Reference

If you use this package or the underlying DDKF technique in your research or software, please cite the original work:

```bibtex
@article{bensegueni2025dual,
  title={Dual Dynamic Kernel Filtering: Accurate Time-Frequency Representation, Reconstruction, and Denoising},
  author={Bensegueni, Skander and Belhaouari, Samir Brahim and Kahalan, Yunis Carreon},
  journal={Digital Signal Processing},
  pages={105407},
  year={2025},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License.

## Authors

- Skander Bensegueni
- Yunis Kahalan

---

**v4.0.0** - Corrected algorithm, updated parameter names, backpropagatable interpolation
