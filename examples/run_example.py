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
    alpha = 0.001
    beta = 0.03
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
        alpha=alpha,
        beta=beta,
        interp_factor=0.125
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
