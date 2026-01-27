"""DDKF - Dual Dynamic Kernel Filtering

Minimal implementation with:
- NumPy and PyTorch support
- Arbitrary number of kernels
- Learnable parameters (alpha, beta, gamma)
- No scipy dependency (works everywhere!)
"""
import numpy as np
from typing import List, Union, Optional, Callable

# Optional PyTorch support
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


# =============================================================================
# Kernels
# =============================================================================

class Kernels:
    """Kernel functions (work with both NumPy and PyTorch)."""
    
    @staticmethod
    def polynomial(x, degree=2, offset=1.3):
        """Polynomial kernel: (x + offset)^degree"""
        if HAS_TORCH and isinstance(x, torch.Tensor):
            return (x + offset) ** degree
        return (x + offset) ** degree
    
    @staticmethod
    def gaussian(x, center=0.7, sigma=1.0):
        """Gaussian kernel: exp(-0.5 * ((x - center) / sigma)^2)"""
        if HAS_TORCH and isinstance(x, torch.Tensor):
            return torch.exp(-0.5 * ((x - center) / sigma) ** 2)
        return np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    @staticmethod
    def get(name):
        """Get kernel by name."""
        kernels = {
            'polynomial': Kernels.polynomial,
            'gaussian': Kernels.gaussian,
        }
        if name not in kernels:
            raise ValueError(f"Unknown kernel: {name}. Available: {list(kernels.keys())}")
        return kernels[name]


# =============================================================================
# DDKF (NumPy)
# =============================================================================

class DDKF:
    """Dual Dynamic Kernel Filtering (NumPy implementation).
    
    Parameters
    ----------
    kernel : str, list of str, or callable
        Kernel(s) to use. Examples:
        - Single: "polynomial" or "gaussian"
        - Multiple: ["polynomial", "gaussian"]
        - Custom: your_kernel_function
    gamma : list of float, optional
        Kernel combination weights (must sum to 1).
        If None, uses equal weights.
    alpha : float, default=0.15
        Threshold for strong frequency detection
    beta : float, default=0.9
        Smoothing coefficient
    window_size : int, default=20
        Sliding window size
    step_size : int, default=4
        Step between windows
    kernel_params : dict or list of dict, optional
        Parameters for kernels
    
    Examples
    --------
    >>> # Single kernel
    >>> dkf = DDKF(kernel="gaussian", alpha=0.15, beta=0.9)
    >>> dkf.fit(signal)
    >>> tfr = dkf.get_tfr()
    
    >>> # Multiple kernels with custom weights
    >>> dkf = DDKF(
    ...     kernel=["polynomial", "gaussian"],
    ...     gamma=[0.7, 0.3],  # 70% poly, 30% gaussian
    ...     alpha=0.15
    ... )
    >>> dkf.fit(signal)
    
    >>> # Three kernels
    >>> dkf = DDKF(
    ...     kernel=["polynomial", "gaussian", "polynomial"],
    ...     gamma=[0.5, 0.3, 0.2],
    ...     kernel_params=[
    ...         {"degree": 2, "offset": 1.3},
    ...         {"center": 0.7, "sigma": 1.0},
    ...         {"degree": 3, "offset": 1.5}
    ...     ]
    ... )
    """
    
    def __init__(
        self,
        kernel: Union[str, List[str], Callable] = "gaussian",
        gamma: Optional[List[float]] = None,
        alpha: float = 0.15,
        beta: float = 0.9,
        window_size: int = 20,
        step_size: int = 4,
        kernel_params: Optional[Union[dict, List[dict]]] = None,
    ):
        # Setup kernels
        if callable(kernel):
            self.kernels = [kernel]
        elif isinstance(kernel, str):
            self.kernels = [Kernels.get(kernel)]
        else:
            self.kernels = [Kernels.get(k) if isinstance(k, str) else k for k in kernel]
        
        n_kernels = len(self.kernels)
        
        # Setup gamma weights
        if gamma is None:
            self.gamma = np.ones(n_kernels) / n_kernels
        else:
            self.gamma = np.array(gamma)
            if len(self.gamma) != n_kernels:
                raise ValueError(f"gamma length ({len(self.gamma)}) must match kernels ({n_kernels})")
            self.gamma = self.gamma / self.gamma.sum()  # Normalize
        
        # Setup kernel parameters
        if kernel_params is None:
            self.kernel_params = [{} for _ in range(n_kernels)]
        elif isinstance(kernel_params, dict):
            self.kernel_params = [kernel_params.copy() for _ in range(n_kernels)]
        else:
            if len(kernel_params) != n_kernels:
                raise ValueError(f"kernel_params length must match kernels ({n_kernels})")
            self.kernel_params = kernel_params
        
        self.alpha = alpha
        self.beta = beta
        self.window_size = window_size
        self.step_size = step_size
        
        # Results (set during fit)
        self.signal = None
        self.M_ = None
        self.Mphase_ = None
        self.M1_ = None
        self.M1phase_ = None
        self.tfr = None
        self.tfr_phase = None
    
    def _apply_kernels(self, signal):
        """Apply kernel combination with gamma weights."""
        s = np.asarray(signal).astype(float)
        
        # Make non-negative
        if s.min() < 0:
            s = s - s.min()
        
        # Combine kernels with weights
        result = np.zeros_like(s)
        for kernel_fn, weight, params in zip(self.kernels, self.gamma, self.kernel_params):
            result += weight * kernel_fn(s, **params)
        
        return result
    
    def fit(self, signal: np.ndarray):
        """Fit DDKF to signal.
        
        Implements the full DDKF algorithm:
        1. Apply kernel combination
        2. First pass: FFT with kernel in window
        3. Second pass: FFT with kernel zeroed in window
        4. Smart minimum operation
        5. Beta thresholding
        """
        self.signal = np.asarray(signal).flatten()
        kernel_signal = self._apply_kernels(self.signal)
        n = len(kernel_signal)
        
        if n < self.window_size:
            raise ValueError(f"Signal too short: {n} < {self.window_size}")
        
        # First pass: kernel IN window
        M_list, Mphase_list = [], []
        for i in range(0, n - self.window_size + 1, self.step_size):
            before = np.zeros(i)
            window = kernel_signal[i:i + self.window_size]
            after = np.zeros(n - i - self.window_size)
            sig_win = np.concatenate([before, window, after])
            
            L = np.fft.fft(sig_win)
            M_list.append(np.abs(L))
            Mphase_list.append(np.angle(L))
        
        self.M_ = np.array(M_list)
        self.Mphase_ = np.array(Mphase_list)
        
        # Second pass: kernel ZEROED in window
        M1_list, M1phase_list = [], []
        for i in range(0, n - self.window_size + 1, self.step_size):
            before = kernel_signal[:i] if i > 0 else np.array([])
            window_zeros = np.zeros(self.window_size)
            after = kernel_signal[i + self.window_size:]
            sig_win = np.concatenate([before, window_zeros, after])
            
            L = np.fft.fft(sig_win)
            M1_list.append(np.abs(L))
            M1phase_list.append(np.angle(L))
        
        self.M1_ = np.array(M1_list)
        self.M1phase_ = np.array(M1phase_list)
        
        # Smart minimum operation
        n_windows = self.M_.shape[0]
        result = np.zeros_like(self.M_)
        result_phase = np.zeros_like(self.Mphase_)
        
        for i in range(n_windows):
            x = self.M_[i]
            y = self.M1_[i]
            
            # Alpha threshold
            strong_mask = x > (x.max() * self.alpha)
            combined = y * x * strong_mask
            result[i] = np.minimum(x, combined)
            
            # Phase selection
            use_x_phase = (result[i] == x)
            result_phase[i] = np.where(use_x_phase, self.Mphase_[i], self.M1phase_[i])
        
        # Beta thresholding
        flat = result.flatten()
        threshold = self.beta * flat.max() if flat.size > 0 else 0.0
        flat = flat * (flat > threshold)
        self.tfr = flat.reshape(result.shape)
        self.tfr_phase = result_phase
        
        return self
    
    def get_tfr(self):
        """Get time-frequency representation."""
        if self.tfr is None:
            raise RuntimeError("Must call fit() first")
        return self.tfr
    
    def inverse_transform(self, correction_factor=None):
        """Reconstruct signal from TFR."""
        if self.tfr is None:
            raise RuntimeError("Must call fit() first")
        
        recovered = []
        for i in range(self.tfr.shape[0]):
            complex_spec = self.tfr[i] * np.exp(1j * self.tfr_phase[i])
            time_signal = np.fft.ifft(complex_spec)
            recovered.append(np.sum(np.abs(time_signal)))
        
        result = np.array(recovered)
        
        if correction_factor is None:
            correction_factor = self.alpha
        
        return correction_factor * result


def denoise(signal, **kwargs):
    """Denoise signal using DDKF."""
    dkf = DDKF(**kwargs)
    dkf.fit(signal)
    return dkf.inverse_transform()


# =============================================================================
# PyTorch DDKF (Learnable)
# =============================================================================

if HAS_TORCH:
    class DDKFLayer(nn.Module):
        """Learnable DDKF layer for PyTorch.
        
        All parameters (alpha, beta, gamma) are learnable via backpropagation!
        
        Parameters
        ----------
        kernel_names : list of str
            Kernel names (can repeat for multiple instances)
        alpha : float, default=0.15
            Initial alpha (learnable)
        beta : float, default=0.9
            Initial beta (learnable)
        gamma : list of float, optional
            Initial gamma weights (learnable)
        window_size : int, default=20
            Window size
        step_size : int, default=4
            Step size
        kernel_params : list of dict, optional
            Parameters for each kernel
        
        Examples
        --------
        >>> # Two kernels
        >>> layer = DDKFLayer(
        ...     kernel_names=['polynomial', 'gaussian'],
        ...     gamma=[0.5, 0.5],
        ...     alpha=0.15
        ... )
        
        >>> # Three kernels (can repeat!)
        >>> layer = DDKFLayer(
        ...     kernel_names=['polynomial', 'gaussian', 'polynomial'],
        ...     gamma=[0.4, 0.3, 0.3],
        ...     kernel_params=[
        ...         {'degree': 2, 'offset': 1.3},
        ...         {'center': 0.7, 'sigma': 1.0},
        ...         {'degree': 3, 'offset': 1.5}
        ...     ]
        ... )
        
        >>> # Forward pass
        >>> tfr = layer(torch.randn(16, 1000))
        
        >>> # Training
        >>> loss = criterion(tfr, target)
        >>> loss.backward()  # Gradients flow!
        >>> optimizer.step()
        """
        
        def __init__(
            self,
            kernel_names: List[str] = None,
            alpha: float = 0.15,
            beta: float = 0.9,
            gamma: Optional[List[float]] = None,
            window_size: int = 20,
            step_size: int = 4,
            kernel_params: Optional[List[dict]] = None,
        ):
            super().__init__()
            
            if kernel_names is None:
                kernel_names = ['gaussian', 'polynomial']
            
            self.kernel_names = kernel_names
            n_kernels = len(kernel_names)
            
            # Setup kernel parameters
            if kernel_params is None:
                self.kernel_params = [{} for _ in range(n_kernels)]
            else:
                if len(kernel_params) != n_kernels:
                    raise ValueError("kernel_params length must match kernel_names")
                self.kernel_params = kernel_params
            
            self.window_size = window_size
            self.step_size = step_size
            
            # Learnable parameters
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.beta = nn.Parameter(torch.tensor(beta))
            
            if gamma is None:
                gamma = [1.0 / n_kernels] * n_kernels
            self._gamma = nn.Parameter(torch.tensor(gamma))
        
        @property
        def gamma(self):
            """Normalized gamma weights (always sum to 1)."""
            return torch.softmax(self._gamma, dim=0)
        
        def _apply_kernels(self, signal):
            """Apply kernel combination."""
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
        
        def forward(self, signal):
            """Process signal through DDKF.
            
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
            
            batch_size, n = signal.shape
            
            # Apply kernels to each signal in batch
            kernel_signals = []
            for b in range(batch_size):
                k_sig = self._apply_kernels(signal[b])
                kernel_signals.append(k_sig)
            kernel_signal = torch.stack(kernel_signals)
            
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
    
    
    class DDKFFeatureExtractor(nn.Module):
        """DDKF feature extractor for ML."""
        
        def __init__(self, kernel_names=None, flatten=False, **kwargs):
            super().__init__()
            self.ddkf = DDKFLayer(kernel_names=kernel_names, **kwargs)
            self.flatten = flatten
        
        def forward(self, x):
            tfr = self.ddkf(x)
            return tfr.view(tfr.size(0), -1) if self.flatten else tfr

else:
    DDKFLayer = None
    DDKFFeatureExtractor = None
