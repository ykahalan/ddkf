# ddkf

Minimal package providing `DDKF` for time-frequency decomposition with arbitrary kernels and learnable parameters. Available to install with pip as package `ddkf`.

**New in v4.0:**
-   **Corrected algorithm** - Window-by-window kernel application matching MATLAB reference
-   **Updated parameter names** - `c_smart_min` and `c_smoothing` (more descriptive)
-   **Arbitrary number of kernels** (not limited to 2)
-   **Learnable parameters** (c_smart_min, c_smoothing, gamma) via PyTorch
-   **Backpropagatable cubic interpolation** - Gradients flow through interpolation
-   **No scipy dependency** (pure NumPy/PyTorch)

**Breaking Changes from v3.x:**
- Parameter `alpha` renamed to `c_smoothing` (beta threshold)
- Parameter `beta` renamed to `c_smart_min` (alpha threshold for smart minimum)
- Algorithm corrected to apply kernels window-by-window (not globally)
- Backward compatibility helper available: `create_ddkf_with_old_params()`

## Quick install

```bash
# Basic install (NumPy only)
python3 -m pip install ddkf

# With PyTorch support (for learnable parameters)
python3 -m pip install ddkf[torch]

# With plotting
python3 -m pip install ddkf[all]
```

## Examples

The `examples/` folder contains complete working examples:

- **run_example.py** - Basic DDKF demonstration showing signal decomposition, time-frequency representation, and inverse transform with a noisy multi-frequency signal (3 Hz + 7 Hz + noise)

- **DDKFvsDWT.py** - Machine learning comparison demonstrating DDKF vs DWT (Discrete Wavelet Transform) as feature extractors for time series classification:
  - Extracts 2D time-frequency features using DDKF
  - Extracts 2D wavelet coefficients using DWT
  - Trains separate CNNs on each feature type
  - Compares classification accuracy on ECG200 dataset
  - Visualizes feature representations and training curves

To run the examples:
```bash
cd examples/
python3 run_example.py        # Basic signal processing demo
python3 DDKFvsDWT.py          # ML classification comparison (requires aeon, pywt)
```

## PyTorch Example (Learnable Parameters)

DDKF v4.0 supports learnable parameters via PyTorch. All parameters (c_smart_min, c_smoothing, gamma) can be optimized through backpropagation:

```python
import torch
from ddkf import DDKFLayer

# Create learnable DDKF layer
layer = DDKFLayer(
    kernel_names=['polynomial', 'gaussian', 'polynomial'],
    gamma=[0.5, 0.3, 0.2],        # Initial weights (learnable)
    c_smoothing=0.12,             # Initial beta threshold (learnable)
    c_smart_min=0.9,              # Initial alpha threshold (learnable)
    window_size=20,
    interp_factor=0.25            # Backpropagatable cubic interpolation
)

# Training loop
optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)

for epoch in range(100):
    # Forward pass
    signal_batch = torch.randn(16, 1000)  # Batch of signals
    tfr = layer(signal_batch)
    
    # Compute loss
    loss = your_loss_function(tfr, target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()  # Gradients flow through interpolation!
    optimizer.step()
    
    # Parameters are updated
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: c_smart_min={layer.c_smart_min.item():.4f}, "
              f"c_smoothing={layer.c_smoothing.item():.4f}")

# Check final learned parameters
print(f"Learned gamma weights: {layer.gamma.detach().numpy()}")
print(f"Learned c_smart_min: {layer.c_smart_min.item():.4f}")
print(f"Learned c_smoothing: {layer.c_smoothing.item():.4f}")
```

## API Overview

### Main Class: `DDKF`

```python
DDKF(
    kernel="gaussian",           # Single kernel or list of kernels
    gamma=None,                  # Kernel weights (auto-normalized to sum=1)
    c_smoothing=0.12,            # Beta threshold (smoothing coefficient)
    c_smart_min=0.9,             # Alpha threshold (smart minimum)
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
    c_smoothing=0.15,
    c_smart_min=0.9
)
```

### PyTorch Layer: `DDKFLayer`

```python
DDKFLayer(
    kernel_names=['polynomial', 'gaussian'],
    gamma=[0.5, 0.5],
    c_smoothing=0.12,            # Beta threshold
    c_smart_min=0.9,             # Alpha threshold
    window_size=20,
    step_size=4,
    interp_factor=0.25           # Cubic interpolation factor
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

### c_smart_min (default: 0.9)
Alpha threshold for the smart minimum operation. Controls which frequency components participate in the smart minimum calculation. Higher values (closer to 1.0) make the filter more selective, only including the strongest frequency components.

**MATLAB equivalent:** `c_smart_min`

### c_smoothing (default: 0.12)
Beta threshold for final smoothing. Suppresses weak frequency components in the final time-frequency representation. Higher values result in more aggressive smoothing.

**MATLAB equivalent:** `c_smoothing`

### gamma (default: equal weights)
Kernel mixing weights. Automatically normalized to sum to 1. For a hybrid kernel with two components, `gamma=[0.5, 0.5]` gives equal weight to each kernel.

## Available Kernels

- `"polynomial"` - Polynomial kernel: (x + offset)^degree
  - Default params: `degree=2, offset=1.3`
- `"gaussian"` - Gaussian kernel: exp(-0.5 * ((x - center) / sigma)^2)
  - Default params: `center=0.7, sigma=1.0`

**Custom kernels:** You can also pass your own callable functions.

**Hybrid kernel (recommended):** Combine polynomial and gaussian for best results:
```python
kernel_names=['polynomial', 'gaussian']
gamma=[0.5, 0.5]
```

## Migrating from v3.x

### Parameter Name Changes

| v3.x | v4.0 | Purpose |
|------|------|---------|
| `alpha` | `c_smoothing` | Beta threshold (smoothing) |
| `beta` | `c_smart_min` | Alpha threshold (smart minimum) |

### Quick Migration

**Option 1: Update parameter names (recommended)**
```python
# OLD (v3.x)
layer = DDKFLayer(alpha=0.15, beta=0.9)

# NEW (v4.0)
layer = DDKFLayer(c_smoothing=0.15, c_smart_min=0.9)
```

**Option 2: Use backward compatibility helper**
```python
from ddkf import create_ddkf_with_old_params

# Still uses old parameter names
layer = create_ddkf_with_old_params(alpha=0.15, beta=0.9)
```

### Algorithm Changes

v4.0 corrects the kernel application to match the MATLAB reference implementation:
- **v3.x:** Applied kernels to entire signal, then windowed (incorrect)
- **v4.0:** Applies kernels window-by-window (correct)

Results will differ from v3.x due to this correction. The new implementation matches the published MATLAB code exactly.

## Key Features

- **Window-by-window processing**: Kernels applied correctly within each window
- **Arbitrary kernels**: Use 1, 2, 3, or more kernels
- **Learnable parameters**: Optimize c_smart_min, c_smoothing, gamma via PyTorch
- **Backpropagatable interpolation**: Gradients flow through cubic interpolation
- **No scipy**: Pure NumPy/PyTorch implementation
- **MATLAB-accurate**: Matches reference implementation exactly
- **Flexible**: Works for denoising, TFR, feature extraction

## Advanced Usage

### Feature Extraction for ML

DDKF can extract time-frequency features for machine learning applications. For a complete working example comparing DDKF vs DWT features for time series classification, see `examples/DDKFvsDWT.py`.

```python
from ddkf import DDKFLayer
import torch

# Create feature extractor
ddkf = DDKFLayer(
    kernel_names=['polynomial', 'gaussian'],
    gamma=[0.5, 0.5],
    c_smoothing=0.15,
    c_smart_min=0.85
)

# Extract features from time series
time_series = torch.randn(100, 1000)  # 100 samples, 1000 time points
features = ddkf(time_series)  # Shape: (100, n_windows, n_freqs)

# Flatten for ML model
features_flat = features.view(features.size(0), -1)  # Shape: (100, n_windows*n_freqs)
```

### End-to-End Trainable Pipeline

```python
import torch.nn as nn

class DDKFClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.ddkf = DDKFLayer(
            kernel_names=['polynomial', 'gaussian'],
            c_smoothing=0.12,
            c_smart_min=0.9
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_windows * n_freqs, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        tfr = self.ddkf(x)  # Learnable TFR
        return self.classifier(tfr)

# Train entire pipeline end-to-end
model = DDKFClassifier(num_classes=10)
optimizer = torch.optim.Adam(model.parameters())
# ... training loop ...
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
