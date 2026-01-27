"""DDKF - Dual Dynamic Kernel Filtering (PyTorch-only with cubic interpolation)

Learnable parameters (alpha, beta, gamma) with backpropagatable cubic interpolation.
"""
import torch
import torch.nn as nn
from typing import List, Optional


# =============================================================================
# Backpropagatable Cubic Interpolation
# =============================================================================

def cubic_interpolate_1d(signal: torch.Tensor, interp_factor: float = 0.25) -> torch.Tensor:
    """
    Cubic spline interpolation for 1D signals (backpropagatable).
    
    Uses Catmull-Rom spline which matches scipy's cubic interpolation behavior.
    
    Parameters
    ----------
    signal : torch.Tensor
        Shape (batch_size, length) or (length,)
    interp_factor : float
        Interpolation factor (0.25 means 4x points)
    
    Returns
    -------
    torch.Tensor
        Interpolated signal
    """
    if signal.dim() == 1:
        signal = signal.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    batch_size, n = signal.shape
    
    if n < 2:
        return signal.squeeze(0) if squeeze else signal
    
    # Calculate new length
    num_points = int((n - 1) / interp_factor) + 1
    
    # Create normalized coordinates [0, 1]
    t_normalized = torch.linspace(0, 1, num_points, device=signal.device)
    
    # Scale to signal indices [0, n-1]
    t_indices = t_normalized * (n - 1)
    
    # For grid_sample, we need to reshape and normalize to [-1, 1]
    # grid_sample expects (N, C, H, W) input and (N, H_out, W_out, 2) grid
    
    # Reshape signal: (batch, 1, 1, length)
    signal_4d = signal.unsqueeze(1).unsqueeze(2)
    
    # Create grid for sampling: normalize to [-1, 1]
    grid_x = (t_indices / (n - 1)) * 2 - 1
    grid_y = torch.zeros_like(grid_x)
    
    # Grid shape: (batch, 1, num_points, 2) where last dim is (x, y)
    grid = torch.stack([grid_x, grid_y], dim=-1)
    grid = grid.unsqueeze(0).expand(batch_size, 1, -1, 2)
    
    # Apply cubic interpolation
    interpolated = torch.nn.functional.grid_sample(
        signal_4d,
        grid,
        mode='bicubic',
        padding_mode='border',
        align_corners=True
    )
    
    # Reshape back: (batch, num_points)
    result = interpolated.squeeze(1).squeeze(1)
    
    return result.squeeze(0) if squeeze else result


# =============================================================================
# Kernels
# =============================================================================

class Kernels:
    """Kernel functions for PyTorch."""
    
    @staticmethod
    def polynomial(x: torch.Tensor, degree: int = 2, offset: float = 1.3) -> torch.Tensor:
        """Polynomial kernel: (x + offset)^degree"""
        return (x + offset) ** degree
    
    @staticmethod
    def gaussian(x: torch.Tensor, center: float = 0.7, sigma: float = 1.0) -> torch.Tensor:
        """Gaussian kernel: exp(-0.5 * ((x - center) / sigma)^2)"""
        return torch.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    @staticmethod
    def get(name: str):
        """Get kernel by name."""
        kernels = {
            'polynomial': Kernels.polynomial,
            'gaussian': Kernels.gaussian,
        }
        if name not in kernels:
            raise ValueError(f"Unknown kernel: {name}. Available: {list(kernels.keys())}")
        return kernels[name]


# =============================================================================
# PyTorch DDKF (Learnable with Interpolation)
# =============================================================================

class DDKFLayer(nn.Module):
    """Learnable DDKF layer for PyTorch with cubic interpolation.
    
    Matches TwoDKF behavior with backpropagatable operations.
    
    Parameters
    ----------
    kernel_names : list of str, optional
        Kernel names. Default: ['polynomial', 'gaussian'] (hybrid)
    alpha : float, default=0.15
        Initial alpha (learnable)
    beta : float, default=0.9
        Initial beta (learnable)
    gamma : list of float, optional
        Initial kernel weights. Default: [0.5, 0.5] for hybrid
    interp_factor : float, default=0.25
        Interpolation factor (0.25 = 4x upsampling)
    window_size : int, default=20
        Window size
    step_size : int, default=4
        Step size
    kernel_params : list of dict, optional
        Parameters for each kernel
    
    Examples
    --------
    >>> # Hybrid kernel (matches TwoDKF default)
    >>> layer = DDKFLayer(
    ...     kernel_names=['polynomial', 'gaussian'],
    ...     gamma=[0.5, 0.5],
    ...     interp_factor=0.25
    ... )
    
    >>> # Forward pass
    >>> signal = torch.randn(16, 1000)
    >>> tfr = layer(signal)
    
    >>> # Training
    >>> loss = criterion(tfr, target)
    >>> loss.backward()  # Gradients flow through interpolation!
    >>> optimizer.step()
    """
    
    def __init__(
        self,
        kernel_names: Optional[List[str]] = None,
        alpha: float = 0.15,
        beta: float = 0.9,
        gamma: Optional[List[float]] = None,
        interp_factor: float = 0.25,
        window_size: int = 20,
        step_size: int = 4,
        kernel_params: Optional[List[dict]] = None,
    ):
        super().__init__()
        
        # Default to hybrid kernel (polynomial + gaussian) like TwoDKF
        if kernel_names is None:
            kernel_names = ['polynomial', 'gaussian']
        
        self.kernel_names = kernel_names
        n_kernels = len(kernel_names)
        
        # Setup kernel parameters
        if kernel_params is None:
            # Default parameters matching TwoDKF
            self.kernel_params = [
                {'degree': 2, 'offset': 1.3},  # polynomial
                {'center': 0.7, 'sigma': 1.0}   # gaussian
            ] if n_kernels == 2 else [{} for _ in range(n_kernels)]
        else:
            if len(kernel_params) != n_kernels:
                raise ValueError("kernel_params length must match kernel_names")
            self.kernel_params = kernel_params
        
        self.interp_factor = interp_factor
        self.window_size = window_size
        self.step_size = step_size
        
        # Learnable parameters
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        
        # Default to equal weights (0.5, 0.5 for hybrid)
        if gamma is None:
            gamma = [1.0 / n_kernels] * n_kernels
        self._gamma = nn.Parameter(torch.tensor(gamma))
    
    @property
    def gamma(self):
        """Normalized gamma weights (always sum to 1)."""
        return torch.softmax(self._gamma, dim=0)
    
    def _interpolate_signal(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply cubic interpolation (backpropagatable)."""
        return cubic_interpolate_1d(signal, self.interp_factor)
    
    def _apply_kernels(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply kernel combination with learnable weights."""
        # Make non-negative
        if signal.min() < 0:
            signal = signal - signal.min()
        
        # Combine kernels with learnable weights
        result = torch.zeros_like(signal)
        gamma = self.gamma
        
        for i, (kname, params) in enumerate(zip(self.kernel_names, self.kernel_params)):
            kernel_fn = Kernels.get(kname)
            result += gamma[i] * kernel_fn(signal, **params)
        
        return result
    
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Process signal through DDKF with interpolation.
        
        Parameters
        ----------
        signal : torch.Tensor
            Shape (batch_size, length) or (length,)
        
        Returns
        -------
        torch.Tensor
            TFR of shape (batch_size, n_windows, n_freqs) or (n_windows, n_freqs)
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        batch_size = signal.shape[0]
        
        # Step 1: Cubic interpolation (backpropagatable!)
        interpolated_signal = self._interpolate_signal(signal)
        n = interpolated_signal.shape[1]
        
        # Step 2: Apply kernels
        kernel_signal = torch.stack([
            self._apply_kernels(interpolated_signal[b])
            for b in range(batch_size)
        ])
        
        if n < self.window_size:
            raise ValueError(f"Interpolated signal ({n}) shorter than window_size ({self.window_size})")
        
        # Compute number of windows
        n_windows = (n - self.window_size) // self.step_size + 1
        
        # First pass: kernel IN window
        M_list, Mphase_list = [], []
        for i in range(n_windows):
            start = i * self.step_size
            before = torch.zeros(batch_size, start, device=signal.device)
            window = kernel_signal[:, start:start + self.window_size]
            after = torch.zeros(batch_size, n - start - self.window_size, device=signal.device)
            sig_win = torch.cat([before, window, after], dim=1)
            
            L = torch.fft.fft(sig_win)
            M_list.append(torch.abs(L))
            Mphase_list.append(torch.angle(L))
        
        M = torch.stack(M_list, dim=1)  # (batch, windows, freqs)
        Mphase = torch.stack(Mphase_list, dim=1)
        
        # Second pass: kernel ZEROED in window
        M1_list, M1phase_list = [], []
        for i in range(n_windows):
            start = i * self.step_size
            before = kernel_signal[:, :start] if start > 0 else torch.zeros(batch_size, 0, device=signal.device)
            window_zeros = torch.zeros(batch_size, self.window_size, device=signal.device)
            after = kernel_signal[:, start + self.window_size:]
            sig_win = torch.cat([before, window_zeros, after], dim=1)
            
            L = torch.fft.fft(sig_win)
            M1_list.append(torch.abs(L))
            M1phase_list.append(torch.angle(L))
        
        M1 = torch.stack(M1_list, dim=1)  # (batch, windows, freqs)
        M1phase = torch.stack(M1phase_list, dim=1)
        
        # Smart minimum operation
        result = torch.zeros_like(M)
        result_phase = torch.zeros_like(Mphase)
        
        for i in range(n_windows):
            x = M[:, i, :]
            y = M1[:, i, :]
            
            # Alpha threshold (learnable!)
            x_max = x.max(dim=1, keepdim=True)[0]
            strong_mask = x > (x_max * self.alpha)
            
            combined = y * x * strong_mask.float()
            result[:, i, :] = torch.minimum(x, combined)
            
            # Phase selection
            use_x_phase = (result[:, i, :] == x)
            result_phase[:, i, :] = torch.where(use_x_phase, Mphase[:, i, :], M1phase[:, i, :])
        
        # Beta thresholding (learnable!)
        flat = result.view(batch_size, -1)
        threshold = self.beta * flat.max(dim=1, keepdim=True)[0]
        threshold = threshold.view(batch_size, 1, 1)
        result = result * (result > threshold).float()
        
        return result.squeeze(0) if squeeze else result
    
    def inverse_transform(self, tfr: torch.Tensor, tfr_phase: torch.Tensor, 
                         correction_factor: Optional[float] = None) -> torch.Tensor:
        """Reconstruct signal from TFR (backpropagatable)."""
        if tfr.dim() == 2:
            tfr = tfr.unsqueeze(0)
            tfr_phase = tfr_phase.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        batch_size, n_windows, _ = tfr.shape
        
        recovered = []
        for b in range(batch_size):
            windows_recovered = []
            for i in range(n_windows):
                complex_spec = tfr[b, i] * torch.exp(1j * tfr_phase[b, i])
                time_signal = torch.fft.ifft(complex_spec)
                windows_recovered.append(torch.sum(torch.abs(time_signal)))
            recovered.append(torch.stack(windows_recovered))
        
        result = torch.stack(recovered)
        
        if correction_factor is None:
            correction_factor = self.alpha.item()
        
        result = correction_factor * result
        
        return result.squeeze(0) if squeeze else result


class DDKFFeatureExtractor(nn.Module):
    """DDKF feature extractor for ML with interpolation."""
    
    def __init__(self, kernel_names=None, flatten=False, **kwargs):
        super().__init__()
        self.ddkf = DDKFLayer(kernel_names=kernel_names, **kwargs)
        self.flatten = flatten
    
    def forward(self, x):
        tfr = self.ddkf(x)
        return tfr.view(tfr.size(0), -1) if self.flatten else tfr
