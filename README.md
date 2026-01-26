# ddkf

Minimal package providing `2DKF` for time-frequency decomposition and a
`denoise` convenience function. Available to install with pip as package `ddkf`. 
## Quick install

```
python3 -m pip install ddkf
```

## Full example (runnable)

The following example produces plots highlighting time frequency representation and denoising. Copy
it into a file (e.g. `run_example.py`) or run interactively.

```python
import numpy as np
import matplotlib.pyplot as plt
from ddkf import ddkf, denoise

if __name__ == "__main__":
    print("=" * 70)
    print("2DKF Example: Noisy Signal Decomposition and Recovery")
    print("=" * 70)
    
    # Initialize
    step_size = 4
    c_smoothing = 0.12
    c_smart_min = 0.9
    window_size = 20
    Fs = 100  # Sampling frequency
    t = np.arange(0, 5, 1/Fs)  # Time vector (5 seconds)
    
    # Generate signal: 3 Hz + 7 Hz mixed sine wave
    print("Generating test signal (3 Hz + 7 Hz with Gaussian noise)...")
    signal = np.sin(2*np.pi*3*t) + 0.5*np.sin(2*np.pi*7*t)
    signal = signal + 0.2*np.random.randn(len(signal))  # Add Gaussian noise
    
    # Create 2DKF instance with hybrid kernel
    print("Applying 2DKF with hybrid kernel...")
    dkf = ddkf(
        window_size=window_size,
        step_size=step_size,
        alpha=c_smart_min,      # c_smart_min in original
        beta=c_smoothing,       # c_smoothing in original
        kernel='hybrid',
        kernel_params={'gamma': 0.5}  # Equal weighting
    )
    
    # Fit and transform
    dkf.fit(signal)
    recovered = dkf.inverse_transform(correction_factor=c_smart_min)
    
    # Get time-frequency representation
    tfr = dkf.get_tfr()
    
    print(f"Original signal shape: {signal.shape}")
    print(f"Interpolated signal shape: {dkf.signal_values_.shape}")
    print(f"TFR shape: {tfr.shape}")
    print(f"Recovered signal shape: {recovered.shape}")
    
    # Create plots
    print("Generating plots...")
    
    # Figure 1: 2DKF Module Image
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.imshow(tfr, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title('2DKF Module Image (Time-Frequency Representation)')
    plt.xlabel('Frequency bins')
    plt.ylabel('Time windows')
    plt.tight_layout()
    
    # Figure 2: Comparison plot
    plt.figure(figsize=(12, 6))
    
    # Plot original noisy signal
    plt.subplot(2, 1, 1)
    plt.plot(signal, 'r-', alpha=0.6, linewidth=0.8, label='Original noisy signal')
    plt.title('Original Noisy Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot recovered signal (adjusted to match original length for comparison)
    plt.subplot(2, 1, 2)
    plt.plot(recovered, 'b-', linewidth=1.5, label='Recovered signal')
    plt.title('Recovered Signal (Denoised)')
    plt.xlabel('Sample (time windows)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    print("All plots generated successfully!")
    print("Closing plots will end the program...")
    plt.show()
    
    print("" + "=" * 70)
    print("2DKF processing complete")
    print("=" * 70)
```

## Reference

If you use this package or the underlying 2DKF technique in your research or software, please cite the original work:

```
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
