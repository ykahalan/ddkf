# ddkf

Minimal package providing `DDKF` for time-frequency decomposition with arbitrary kernels and learnable parameters. Available to install with pip as package `ddkf`.

**New in v2.0:**
-   **Arbitrary number of kernels** (not limited to 2)
-   **Learnable parameters** (alpha, beta, gamma) via PyTorch
-   **No scipy dependency** (pure NumPy/PyTorch)
-   **Same proven algorithm** (FFT-based dual-pass)

## Quick install

```bash
# Basic install (NumPy only)
python3 -m pip install ddkf

# With PyTorch support (for learnable parameters)
python3 -m pip install ddkf[torch]

# With plotting
python3 -m pip install ddkf[all]
```

## Full example (runnable)

The following example from the examples folder demonstrates DDKF with multiple kernels for time-frequency decomposition and denoising. Copy it into a file (e.g. `run_example.py`) or run interactively.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from ddkf import DDKFLayer

if __name__ == "__main__":
    print("=" * 70)
    print("DDKF Example: Noisy Signal Decomposition and Recovery")
    print("(PyTorch version with cubic interpolation)")
    print("=" * 70)
    
    # Initialize
    step_size = 4
    c_smoothing = 0.12
    c_smart_min = 0.9
    window_size = 20
    Fs = 100
    t = np.arange(0, 5, 1/Fs)
    
    # Generate signal
    print("Generating test signal (3 Hz + 7 Hz with Gaussian noise)...")
    signal = np.sin(2*np.pi*3*t) + 0.5*np.sin(2*np.pi*7*t)
    signal = signal + 0.2*np.random.randn(len(signal))
    
    # Convert to PyTorch
    signal_torch = torch.from_numpy(signal).float()
    
    # Create DDKF layer with hybrid kernel
    print("Applying DDKF with hybrid kernel...")
    layer = DDKFLayer(
        kernel_names=["polynomial", "gaussian"],
        gamma=[0.5, 0.5],
        window_size=window_size,
        step_size=step_size,
        c_smart_min=c_smart_min,
        c_smoothing=c_smoothing,
        interp_factor=0.25  # NEW: Cubic interpolation!
    )
    
    # Forward pass
    with torch.no_grad():
        tfr = layer(signal_torch)
    
    # Convert back to numpy for plotting
    tfr_np = tfr.numpy()
    
    print(f"Original signal shape: {signal.shape}")
    print(f"TFR shape: {tfr_np.shape}")
    
    # Create plots
    print("Generating plots...")
    plt.figure(figsize=(10, 6))
    plt.imshow(tfr_np, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title('DDKF Time-Frequency Representation (with cubic interpolation)')
    plt.xlabel('Frequency bins')
    plt.ylabel('Time windows')
    plt.tight_layout()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(signal, 'r-', alpha=0.6, linewidth=0.8, label='Original noisy signal')
    plt.title('Original Noisy Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(tfr_np.sum(axis=1), 'b-', linewidth=1.5, label='DDKF output')
    plt.title('DDKF Processing Result')
    plt.xlabel('Time windows')
    plt.ylabel('Magnitude')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("=" * 70)
    print("DDKF processing complete")
    print("=" * 70)
```

## PyTorch Example (Learnable Parameters)

DDKF v3.0 supports learnable parameters via PyTorch! All parameters (alpha, beta, gamma) can be optimized through backpropagation:

```python
import torch
from ddkf import DDKFLayer

# Create learnable DDKF layer
layer = DDKFLayer(
    kernel_names=['polynomial', 'gaussian', 'polynomial'],
    gamma=[0.5, 0.3, 0.2],  # Initial weights (will be learned!)
    alpha=0.15,              # Initial alpha (will be learned!)
    beta=0.9,                # Initial beta (will be learned!)
    window_size=20
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
    loss.backward()
    optimizer.step()
    
    # Parameters are updated!
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: alpha={layer.alpha.item():.4f}, "
              f"beta={layer.beta.item():.4f}")

# Check final learned parameters
print(f"Learned gamma weights: {layer.gamma.detach().numpy()}")
```

## API Overview

### Main Class: `DDKF`

```python
DDKF(
    kernel="gaussian",           # Single kernel or list of kernels
    gamma=None,                  # Kernel weights (auto-normalized to sum=1)
    alpha=0.15,                  # Threshold for strong frequencies
    beta=0.9,                    # Smoothing coefficient
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
    alpha=0.15
)
```

### PyTorch Layer: `DDKFLayer`

```python
DDKFLayer(
    kernel_names=['polynomial', 'gaussian'],
    gamma=[0.5, 0.5],
    alpha=0.15,
    beta=0.9,
    window_size=20,
    step_size=4
)
```

**All parameters are learnable via backpropagation!**

## Available Kernels

- `"polynomial"` - Polynomial kernel (x + offset)^degree
- `"gaussian"` - Gaussian kernel exp(-0.5 * ((x - center) / sigma)^2)

**Custom kernels:** You can also pass your own callable functions!

## Key Features

- **Arbitrary kernels**: Use 1, 2, 3, or more kernels
- **Learnable parameters**: Optimize alpha, beta, gamma via PyTorch
- **No scipy**: Pure NumPy/PyTorch implementation
- **Proven algorithm**: Same FFT-based dual-pass method
- **Flexible**: Works for denoising, TFR, feature extraction

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

**v3.0.0** - Arbitrary kernels, learnable parameters, no scipy dependency
