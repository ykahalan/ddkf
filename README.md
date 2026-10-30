# ddkf

Minimal PyTorch package providing **DDKF (Dual Dynamic Kernel Filtering)** for time-frequency decomposition with arbitrary kernels and learnable parameters. Available on PyPI as `ddkf`.

The same algorithm is also published under the name NTFA-LK; `NTFALayer` and `NTFAFeatureExtractor` are provided as aliases of `DDKFLayer` and `DDKFFeatureExtractor`.

## Quick install

```bash
python3 -m pip install ddkf
```

For the examples and the test suite:

```bash
python3 -m pip install "ddkf[examples,dev]"
```

## Quick start

```python
import torch
from ddkf import DDKFLayer

layer = DDKFLayer(
    kernel_names=['polynomial', 'gaussian'],
    gamma=[0.5, 0.5],
    alpha=0.12,
    beta=0.9,
    window_size=20,
    step_size=4,
    interp_factor=0.25,
)

signal = torch.randn(16, 1000)          # (batch, length) or (length,)
tfr = layer(signal)                      # (batch, n_windows, n_freqs)
recovered = layer.inverse_transform(tfr) # (batch, n_windows), phase stored by forward()

# Or get the phase explicitly, e.g. to call inverse_transform later / elsewhere
tfr, phase = layer(signal, return_phase=True)
recovered = layer.inverse_transform(tfr, phase)
```

## ⚠️ Important: Use the Polynomial Kernel

**The polynomial kernel performs exceptionally well in any combination and should be included in your kernel configuration.** We don't set it as the only default to allow users flexibility in configuring the polynomial parameters (degree and offset) for their specific use case.

**Recommended configuration:**
```python
kernel_names=['polynomial', 'gaussian']
gamma=[0.5, 0.5]
```

## Examples

The `examples/` folder contains complete working examples:

- **run_example.py** - Basic DDKF demonstration showing signal decomposition and the time-frequency representation of a noisy multi-frequency signal (3 Hz + 7 Hz + noise)

- **NTFAvsDWT.py** - Machine learning comparison of DDKF vs DWT (Discrete Wavelet Transform) as feature extractors for time series classification:
  - Extracts 2D time-frequency features using DDKF
  - Extracts 2D wavelet coefficients using DWT
  - Trains separate CNNs on each feature type
  - Compares classification accuracy on the ECG200 dataset
  - Visualizes feature representations and training curves

To run the examples:
```bash
cd examples/
python3 run_example.py        # Basic signal processing demo
python3 NTFAvsDWT.py          # ML classification comparison (requires aeon, pywt, scikit-learn)
```

## API Overview

### Main class: `DDKFLayer`
```python
DDKFLayer(
    kernel_names=['polynomial', 'gaussian'],  # str, callable or nn.Module per kernel
    gamma=None,                # Kernel weights, normalised to sum to 1 (default: equal)
    alpha=0.12,                # Global threshold on the final TFR
    beta=0.9,                  # Smart-minimum threshold
    threshold_mode='hard',     # 'hard' or 'soft' (differentiable)
    window_size=20,            # Window length in interpolated samples
    step_size=4,               # Window hop in interpolated samples
    interp_factor=0.25,        # Cubic interpolation step (0.25 = 4x upsampling)
    kernel_params=None,        # list of dicts, one per kernel
    learn_alpha=False,         # Make alpha an nn.Parameter
    learn_beta=False,          # Make beta an nn.Parameter
    learn_gamma=False,         # Make the kernel weights an nn.Parameter
    learn_sigmoid_temp=False,  # Make the soft-mode temperature an nn.Parameter
    sigmoid_temp=1.0,          # Temperature for soft operations
)
```

**Forward pass:**
```python
tfr = layer(signal)                       # time-frequency representation
tfr, phase = layer(signal, return_phase=True)
```

**Inverse transform:**
```python
recovered = layer.inverse_transform(tfr)         # uses the phase stored by the last forward()
recovered = layer.inverse_transform(tfr, phase)  # or provide the phase explicitly
```

**Feature extractor:**
```python
from ddkf import DDKFFeatureExtractor
fe = DDKFFeatureExtractor(kernel_names=['polynomial', 'gaussian'], flatten=True)
features = fe(signal)   # (batch, n_windows * n_freqs)
```

### Training

With `threshold_mode='soft'` every operation is differentiable, so alpha, beta, gamma, the temperature and any learnable kernel can be trained end to end:

```python
layer = DDKFLayer(
    kernel_names=['polynomial', 'gaussian'],
    threshold_mode='soft',
    learn_alpha=True, learn_beta=True, learn_gamma=True, learn_sigmoid_temp=True,
)
optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
for signal, target in dataloader:
    loss = criterion(layer(signal), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

In `'hard'` mode gradients still flow to the input signal and to learnable kernels, but the thresholds themselves (alpha, beta) only receive gradients in `'soft'` mode.

## Parameter Descriptions

### alpha (default: 0.12)
Global threshold applied to the final time-frequency representation (paper eq. 4), as a fraction of the maximum TFR value. Suppresses weak components. Higher values give a sparser TFR.

### beta (default: 0.9)
Smart-minimum threshold (paper eq. 5), applied per window as a fraction of that window's spectral maximum. Controls which frequency components take part in the smart minimum. Higher values (closer to 1.0) make the filter more selective. `beta` also scales the output of `inverse_transform`.

### gamma (default: equal weights)
Kernel mixing weights. Automatically normalised to sum to 1, so `[0.5, 0.5]` and `[2, 2]` are equivalent. Set `learn_gamma=True` to train them.

### threshold_mode (default: 'hard')
- `'hard'`: binary masking (faster, thresholds are not differentiable)
- `'soft'`: smooth approximations via LogSumExp / sigmoid (fully differentiable)

### sigmoid_temp (default: 1.0)
Temperature for the soft operations (only used when `threshold_mode='soft'`). Higher values approach hard thresholding. Can be made learnable with `learn_sigmoid_temp=True`.

### interp_factor (default: 0.25)
Step of the cubic interpolation in units of the original sample spacing. `0.25` gives 4x as many points, `1.0` keeps the length, values above `1.0` down-sample. `window_size` and `step_size` are expressed in interpolated samples.

## Available Kernels

All kernels are applied point-wise to the (interpolated) signal values.

- `"polynomial"` - Polynomial kernel: (x + offset)^degree
  - Default params: `degree=2, offset=1.3`
- `"gaussian"` - Gaussian kernel: exp(-0.5 * ((x - center) / sigma)^2)
  - Default params: `center=0.7, sigma=1.0`
- `"matern32"` - Matérn 3/2 kernel: (1 + √3(x + offset)) * exp(-√3(x + offset))
  - Default params: `offset=1.7`
- `"matern52"` - Matérn 5/2 kernel: (1 + √5(x + offset) + (5/3)(x + offset)²) * exp(-√5(x + offset))
  - Default params: `offset=1.7`
- `"rational"` - Rational quadratic kernel: (1 + scale * x)^(-power)
  - Default params: `scale=1/3, power=3`
- `"gamma_rational"` - Gamma rational quadratic kernel: (1 + scale * (x + offset)²)^(-power)
  - Default params: `scale=1/3, offset=1.7, power=3`
- **`LearnableChebyshevKernel`** - Learnable Chebyshev polynomial kernel with trainable coefficients
  - Example: `LearnableChebyshevKernel(degree=4)` or `LearnableChebyshevKernel(init_coeffs=[0, 1, 0, 0])`

When `kernel_params` is omitted, each built-in kernel gets the defaults listed above (regardless of its position in `kernel_names`).

## Custom Kernels

You can provide your own point-wise kernel functions, or any `nn.Module` whose parameters you want to learn:

```python
import torch
from ddkf import DDKFLayer, LearnableChebyshevKernel

def my_custom_kernel(x, scale=2.0, power=3):
    return (x * scale) ** power

layer = DDKFLayer(
    kernel_names=[my_custom_kernel, 'gaussian'],
    kernel_params=[
        {'scale': 1.5, 'power': 2},      # params for the custom kernel
        {'center': 0.5, 'sigma': 0.8},   # params for the gaussian
    ],
    gamma=[0.6, 0.4],
)

# Lambdas work too
layer = DDKFLayer(
    kernel_names=[lambda x, scale=1.0: torch.exp(-x * scale), 'polynomial'],
    kernel_params=[{'scale': 2.0}, {'degree': 3, 'offset': 1.0}],
)

# Learnable kernel (an nn.Module); its parameters show up in layer.parameters()
layer = DDKFLayer(kernel_names=[LearnableChebyshevKernel(4), 'polynomial'])
```

## Key Features

- **Window-by-window processing**: kernels applied inside each window and outside it, exactly as in the reference algorithm
- **Vectorised**: one batched FFT per pass instead of a Python loop over windows (identical numbers, much faster)
- **Arbitrary kernels**: use 1, 2, 3 or more built-in, custom or learnable kernels
- **Learnable**: alpha, beta, gamma, temperature and kernel parameters can all be trained (soft mode)
- **No scipy**: pure PyTorch implementation, works with float32 and float64, CPU or GPU
- **Flexible**: works for denoising, TFR, feature extraction

## Development

```bash
git clone https://github.com/ykahalan/ddkf
cd ddkf
python3 -m pip install -e ".[dev]"
python3 -m pytest
```

The test suite checks the vectorised implementation against a loop-based re-implementation of the original algorithm in both threshold modes.

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

## Changelog

**v4.2.0**
- Fixed: the package could not be imported (`__init__` referenced a module that does not exist in this repository)
- Fixed: float64 inputs raised a dtype error in the interpolation and padding
- Fixed: `inverse_transform` on a 2D TFR with the stored phase raised an `IndexError`
- Fixed: gaussian kernel default `center` was 13.7 instead of the documented 0.7
- Fixed: default kernel parameters are now chosen per kernel name, not by position
- Fixed: `DDKFFeatureExtractor(flatten=True)` on a single (1D) signal
- Added: `learn_gamma`, `return_phase`, `soft_minimum` and `LearnableChebyshevKernel` exports, `NTFALayer` aliases
- Added: argument validation (`step_size`, `window_size`, `interp_factor`, `threshold_mode`, lengths of `gamma` / `kernel_params`)
- Changed: forward pass and inverse transform are vectorised (same results, no per-window Python loops)
- Added: test suite (`tests/`)

**v4.1.0** - Soft threshold mode, learnable sigmoid temperature, learnable Chebyshev kernel

**v4.0.0** - Corrected algorithm, updated parameter names, backpropagatable interpolation
