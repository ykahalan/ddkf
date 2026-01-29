"""DDKF - Dual Dynamic Kernel Filtering (Corrected to match paper)

Fixed implementation matching the paper reference code exactly.
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
    def matern32(x: torch.Tensor, offset: float = 1.7) -> torch.Tensor:
        """Matern 3/2 kernel: (1 + sqrt(3)(x + offset)) * exp(-sqrt(3)(x + offset))"""
        sqrt3 = torch.sqrt(torch.tensor(3.0, device=x.device, dtype=x.dtype))
        shifted_x = x + offset
        return (1 + sqrt3 * shifted_x) * torch.exp(-sqrt3 * shifted_x)
    
    @staticmethod
    def matern52(x: torch.Tensor, offset: float = 1.7) -> torch.Tensor:
        """Matern 5/2 kernel: (1 + sqrt(5)(x + offset) + (5/3)(x + offset)²) * exp(-sqrt(5)(x + offset))"""
        sqrt5 = torch.sqrt(torch.tensor(5.0, device=x.device, dtype=x.dtype))
        shifted_x = x + offset
        return (1 + sqrt5 * shifted_x + (5/3) * (shifted_x ** 2)) * torch.exp(-sqrt5 * shifted_x)
    
    @staticmethod
    def rational(x: torch.Tensor, scale: float = 1/3, power: int = 3) -> torch.Tensor:
        """Rational quadratic kernel: (1 + scale * x)^(-power)"""
        return (1 + scale * x) ** (-power)
    
    @staticmethod
    def gamma_rational(x: torch.Tensor, scale: float = 1/3, offset: float = 1.7, power: int = 3) -> torch.Tensor:
        """Gamma rational quadratic kernel: (1 + scale * (x + offset)²)^(-power)"""
        return (1 + scale * ((x + offset) ** 2)) ** (-power)    
        
    @staticmethod
    def get(name):
        """Get kernel by name or return callable directly."""
        # If it's already callable, return it
        if callable(name):
            return name
        
        # Otherwise look up by string name
        kernels = {
            'polynomial': Kernels.polynomial,
            'gaussian': Kernels.gaussian,
            'matern32': Kernels.matern32,
            'matern52': Kernels.matern52,
            'rational': Kernels.rational,
            'gamma_rational': Kernels.gamma_rational,
        }
        
        if name not in kernels:
            raise ValueError(f"Unknown kernel: {name}. Available: {list(kernels.keys())}")
        return kernels[name]


# =============================================================================
# PyTorch DDKF (Corrected to match paper)
# =============================================================================

class DDKFLayer(nn.Module):
    """Learnable DDKF layer for PyTorch with cubic interpolation.
    
    CORRECTED to match paper reference implementation exactly.
    
    Key fixes:
    - Kernel applied ONLY within window (not globally)
    - Correct parameter naming (alpha, beta)
    - Proper phase handling
    - Correct inverse transform weighting
    
    Parameters
    ----------
    kernel_names : list of str or callable, optional
        Kernel names (str) or custom kernel functions (callable).
        Custom kernels should accept (x, **params) and return torch.Tensor.
        Default: ['polynomial', 'gaussian'] (hybrid)
        Example: [my_custom_kernel, 'gaussian'] or ['polynomial', lambda x, scale=1.0: x * scale]
    alpha : float, default=0.12
        Local thresholding parameter
    beta : float, default=0.9
        Smart minimum threshold 
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
    """
    
    def __init__(
        self,
        kernel_names: Optional[List[str]] = None,
        alpha: float = 0.12,
        beta: float = 0.9,
        gamma: Optional[List[float]] = None,
        interp_factor: float = 0.25,
        window_size: int = 20,
        step_size: int = 4,
        kernel_params: Optional[List[dict]] = None,
    ):
        super().__init__()
        
        # Default to hybrid kernel (polynomial + gaussian) like paper
        if kernel_names is None:
            kernel_names = ['polynomial', 'gaussian']
        
        self.kernel_names = kernel_names
        n_kernels = len(kernel_names)
        
        # Setup kernel parameters
        if kernel_params is None:
            # Default parameters matching paper
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
        
        # Learnable parameters (renamed to match paper)
        self.beta = nn.Parameter(torch.tensor(beta))
        self.alpha = nn.Parameter(torch.tensor(alpha))
        
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
        """Apply kernel combination with learnable weights.
        
        NOTE: In paper, this is applied PER WINDOW, not globally!
        """
        # Combine kernels with learnable weights
        result = torch.zeros_like(signal)
        gamma = self.gamma
        
        for i, (kname, params) in enumerate(zip(self.kernel_names, self.kernel_params)):
            kernel_fn = Kernels.get(kname)
            result += gamma[i] * kernel_fn(signal, **params)
        
        return result
    
    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Process signal through DDKF with interpolation.
        
        CORRECTED to match paper: kernels applied window-by-window!
        
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
        
        if n < self.window_size:
            raise ValueError(f"Interpolated signal ({n}) shorter than window_size ({self.window_size})")
        
        # Compute number of windows
        n_windows = (n - self.window_size) // self.step_size + 1
        
        # First pass: kernel IN window, zeros elsewhere (MATCHES paper!)
        M_list, Mphase_list = [], []
        for i in range(n_windows):
            start = i * self.step_size
            
            # Extract window data
            window_data = interpolated_signal[:, start:start + self.window_size]
            
            # Apply kernels ONLY to window data (CORRECTED!)
            values_in_window = self._apply_kernels(window_data)
            
            # Pad with zeros before and after
            before = torch.zeros(batch_size, start, device=signal.device)
            after = torch.zeros(batch_size, n - start - self.window_size, device=signal.device)
            sig_win = torch.cat([before, values_in_window, after], dim=1)
            
            # Compute FFT
            L = torch.fft.fft(sig_win)
            M_list.append(torch.abs(L))
            Mphase_list.append(torch.angle(L))
        
        M = torch.stack(M_list, dim=1)  # (batch, windows, freqs)
        Mphase = torch.stack(Mphase_list, dim=1)
        
        # Second pass: kernel ZEROED in window, kernel elsewhere (MATCHES paper!)
        M1_list, M1phase_list = [], []
        for i in range(n_windows):
            start = i * self.step_size
            
            # Apply kernels to before and after regions
            if start > 0:
                values_before = self._apply_kernels(interpolated_signal[:, :start])
            else:
                values_before = torch.zeros(batch_size, 0, device=signal.device)
            
            # Zeros in window
            values_in_window = torch.zeros(batch_size, self.window_size, device=signal.device)
            
            # After window
            if start + self.window_size < n:
                values_after = self._apply_kernels(interpolated_signal[:, start + self.window_size:])
            else:
                values_after = torch.zeros(batch_size, 0, device=signal.device)
            
            sig_win = torch.cat([values_before, values_in_window, values_after], dim=1)
            
            # Compute FFT
            L = torch.fft.fft(sig_win)
            M1_list.append(torch.abs(L))
            M1phase_list.append(torch.angle(L))
        
        M1 = torch.stack(M1_list, dim=1)  # (batch, windows, freqs)
        M1phase = torch.stack(M1phase_list, dim=1)
        
        # Smart minimum operation (CORRECTED: per-window threshold!)
        result = torch.zeros_like(M)
        result_phase = torch.zeros_like(Mphase)
        
        for i in range(n_windows):
            x = M[:, i, :]  # (batch, freqs)
            y = M1[:, i, :]
            
            # Threshold computed PER WINDOW (CORRECTED!)
            x_max = x.max(dim=1, keepdim=True)[0]  # (batch, 1)
            strong_mask = x > (x_max * self.beta)
            
            # Smart minimum: min(x, y*x*mask)
            combined = y * x * strong_mask.float()
            result[:, i, :] = torch.minimum(x, combined)
            
            # Phase selection: element-wise
            use_s1_phase = (M[:, i, :] < combined)
            result_phase[:, i, :] = torch.where(use_s1_phase, Mphase[:, i, :], M1phase[:, i, :])
        
        # Local thresholding (α in paper equation 4)
        flat = result.view(batch_size, -1)
        threshold = self.alpha * flat.max(dim=1, keepdim=True)[0]
        threshold = threshold.view(batch_size, 1, 1)
        result = result * (result > threshold).float()
        
        # Store phase for inverse transform
        self._last_phase = result_phase
        
        return result.squeeze(0) if squeeze else result
    
    def inverse_transform(self, tfr: torch.Tensor, 
                         tfr_phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reconstruct signal from TFR (backpropagatable).
        
        CORRECTED: Uses beta (0.9) as correction factor to match paper.
        """
        if tfr.dim() == 2:
            tfr = tfr.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        if tfr_phase is None:
            if not hasattr(self, '_last_phase'):
                raise ValueError("No phase information available. Run forward() first or provide tfr_phase.")
            tfr_phase = self._last_phase
            if squeeze:
                tfr_phase = tfr_phase.unsqueeze(0)
        elif tfr_phase.dim() == 2:
            tfr_phase = tfr_phase.unsqueeze(0)
        
        batch_size, n_windows, _ = tfr.shape
        
        # Reconstruct each window
        recovered = []
        for b in range(batch_size):
            windows_recovered = []
            for i in range(n_windows):
                # Complex spectrum
                complex_spec = tfr[b, i] * torch.exp(1j * tfr_phase[b, i])
                # IFFT
                time_signal = torch.fft.ifft(complex_spec)
                complex_sum = torch.sum(time_signal)  # Sum complex FIRST
                windows_recovered.append(torch.abs(complex_sum))  # Magnitude SECOND
            recovered.append(torch.stack(windows_recovered))
        
        result = torch.stack(recovered)
        
        # Apply correction factor: beta (0.9) to match paper
        result = self.beta * result
        
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
