"""DDKF - Dual Dynamic Kernel Filtering.

PyTorch implementation of DDKF with:

- an arbitrary number of (built-in, custom or learnable) kernels,
- learnable alpha / beta / gamma / sigmoid-temperature parameters,
- a backpropagatable cubic interpolation front end,
- a fully vectorised forward pass (one batched FFT per pass instead of a
  Python loop over windows),
- a ``'soft'`` threshold mode that makes every operation differentiable.

The algorithm itself is unchanged from the loop-based reference
implementation (window-by-window kernel application, per-window smart
minimum, global alpha threshold); the vectorised code produces the same
numbers, only faster.
"""
import torch
import torch.nn as nn
from typing import Callable, List, Optional, Sequence, Union

KernelSpec = Union[str, Callable[..., torch.Tensor], nn.Module]


# =============================================================================
# Backpropagatable Cubic Interpolation
# =============================================================================

def cubic_interpolate_1d(signal: torch.Tensor, interp_factor: float = 0.25) -> torch.Tensor:
    """Cubic interpolation for 1D signals (backpropagatable).

    Parameters
    ----------
    signal : torch.Tensor
        Shape ``(batch_size, length)`` or ``(length,)``. Any float dtype.
    interp_factor : float
        Sampling step in units of the original sample spacing. ``0.25``
        produces 4x as many points, ``1.0`` leaves the length unchanged and
        values above ``1.0`` down-sample.

    Returns
    -------
    torch.Tensor
        Interpolated signal with ``int((length - 1) / interp_factor) + 1``
        points, same dtype and device as the input.
    """
    if interp_factor <= 0:
        raise ValueError(f"interp_factor must be > 0, got {interp_factor}")

    if signal.dim() == 1:
        signal = signal.unsqueeze(0)
        squeeze = True
    elif signal.dim() == 2:
        squeeze = False
    else:
        raise ValueError(f"signal must be 1D or 2D, got shape {tuple(signal.shape)}")

    batch_size, n = signal.shape

    if n < 2:
        return signal.squeeze(0) if squeeze else signal

    num_points = int((n - 1) / interp_factor) + 1

    # grid_sample needs the grid in the same dtype as the input and expects
    # (N, C, H, W) input with an (N, H_out, W_out, 2) grid in [-1, 1].
    grid_x = torch.linspace(-1.0, 1.0, num_points, device=signal.device, dtype=signal.dtype)
    grid_y = torch.zeros_like(grid_x)
    grid = torch.stack([grid_x, grid_y], dim=-1)              # (num_points, 2)
    grid = grid.view(1, 1, num_points, 2).expand(batch_size, 1, num_points, 2)

    signal_4d = signal.unsqueeze(1).unsqueeze(2)              # (batch, 1, 1, n)

    interpolated = torch.nn.functional.grid_sample(
        signal_4d,
        grid,
        mode='bicubic',
        padding_mode='border',
        align_corners=True,
    )

    result = interpolated.reshape(batch_size, num_points)
    return result.squeeze(0) if squeeze else result


def soft_minimum(a: torch.Tensor, b: torch.Tensor, temperature=1.0) -> torch.Tensor:
    """Differentiable soft minimum using LogSumExp.

    ``-logsumexp([-a*t, -b*t]) / t``; approaches ``min(a, b)`` as ``t`` grows.
    Unlike :func:`torch.minimum`, gradients flow to both inputs.
    """
    stacked = torch.stack([-a * temperature, -b * temperature], dim=0)
    return -torch.logsumexp(stacked, dim=0) / temperature


class LearnableChebyshevKernel(nn.Module):
    """Learnable Chebyshev polynomial kernel with trainable coefficients.

    ``k(x) = sum_n c_n T_n(tanh(x))``. ``init_coeffs`` overrides ``degree``.
    """

    def __init__(self, degree: int = 4, init_coeffs: Optional[Sequence[float]] = None):
        super().__init__()
        if init_coeffs is not None:
            self.coeffs = nn.Parameter(torch.tensor(list(init_coeffs), dtype=torch.float32))
        else:
            if degree < 1:
                raise ValueError("degree must be >= 1")
            init = torch.zeros(degree)
            if degree > 1:
                init[1] = 1.0
            self.coeffs = nn.Parameter(init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cheb = torch.tanh(x)
        degree = len(self.coeffs)
        T_prev2 = torch.ones_like(x_cheb)
        out = self.coeffs[0] * T_prev2
        if degree > 1:
            T_prev1 = x_cheb
            out = out + self.coeffs[1] * T_prev1
            for n in range(2, degree):
                T_n = 2.0 * x_cheb * T_prev1 - T_prev2
                out = out + self.coeffs[n] * T_n
                T_prev2, T_prev1 = T_prev1, T_n
        return out


# =============================================================================
# Kernels
# =============================================================================

class Kernels:
    """Built-in point-wise kernel functions."""

    # Default parameters (paper values) used when the user passes no
    # ``kernel_params`` for a built-in kernel.
    DEFAULT_PARAMS = {
        'polynomial': {'degree': 2, 'offset': 1.3},
        'gaussian': {'center': 0.7, 'sigma': 1.0},
    }

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
        sqrt3 = 3.0 ** 0.5
        shifted_x = x + offset
        return (1 + sqrt3 * shifted_x) * torch.exp(-sqrt3 * shifted_x)

    @staticmethod
    def matern52(x: torch.Tensor, offset: float = 1.7) -> torch.Tensor:
        """Matern 5/2 kernel: (1 + sqrt(5)(x + offset) + (5/3)(x + offset)^2) * exp(-sqrt(5)(x + offset))"""
        sqrt5 = 5.0 ** 0.5
        shifted_x = x + offset
        return (1 + sqrt5 * shifted_x + (5 / 3) * (shifted_x ** 2)) * torch.exp(-sqrt5 * shifted_x)

    @staticmethod
    def rational(x: torch.Tensor, scale: float = 1 / 3, power: int = 3) -> torch.Tensor:
        """Rational quadratic kernel: (1 + scale * x)^(-power)"""
        return (1 + scale * x) ** (-power)

    @staticmethod
    def gamma_rational(x: torch.Tensor, scale: float = 1 / 3, offset: float = 1.7, power: int = 3) -> torch.Tensor:
        """Gamma rational quadratic kernel: (1 + scale * (x + offset)^2)^(-power)"""
        return (1 + scale * ((x + offset) ** 2)) ** (-power)

    @staticmethod
    def available() -> List[str]:
        return ['polynomial', 'gaussian', 'matern32', 'matern52', 'rational', 'gamma_rational']

    @staticmethod
    def get(name: KernelSpec) -> Callable[..., torch.Tensor]:
        """Get a kernel by name, or return a callable unchanged."""
        if callable(name):
            return name

        kernels = {
            'polynomial': Kernels.polynomial,
            'gaussian': Kernels.gaussian,
            'matern32': Kernels.matern32,
            'matern52': Kernels.matern52,
            'rational': Kernels.rational,
            'gamma_rational': Kernels.gamma_rational,
        }

        if name not in kernels:
            raise ValueError(f"Unknown kernel: {name!r}. Available: {list(kernels.keys())}")
        return kernels[name]

    @staticmethod
    def default_params(name: KernelSpec) -> dict:
        """Default parameters for a kernel spec (empty for custom kernels)."""
        if isinstance(name, str):
            return dict(Kernels.DEFAULT_PARAMS.get(name, {}))
        return {}


# =============================================================================
# PyTorch DDKF
# =============================================================================

class DDKFLayer(nn.Module):
    """Learnable DDKF layer for PyTorch with cubic interpolation.

    Parameters
    ----------
    kernel_names : list of str, callable or nn.Module, optional
        Kernel names (str), custom point-wise kernel functions (callable
        accepting ``(x, **params)``) or learnable kernels (``nn.Module``).
        Default: ``['polynomial', 'gaussian']`` (hybrid).
    alpha : float, default=0.12
        Global threshold applied to the final TFR (paper eq. 4).
    beta : float, default=0.9
        Smart-minimum threshold (paper eq. 5).
    threshold_mode : {'hard', 'soft'}, default='hard'
        ``'hard'`` uses binary masks; ``'soft'`` uses smooth approximations so
        that alpha, beta and the temperature receive gradients.
    gamma : list of float, optional
        Kernel mixing weights (normalised to sum to 1). Default: equal weights.
    interp_factor : float, default=0.25
        Interpolation step (0.25 = 4x up-sampling).
    window_size : int, default=20
        Window size in interpolated samples.
    step_size : int, default=4
        Window hop in interpolated samples.
    kernel_params : list of dict, optional
        Parameters for each kernel. Defaults to the paper values for
        ``'polynomial'`` and ``'gaussian'`` and ``{}`` otherwise.
    learn_alpha, learn_beta, learn_gamma, learn_sigmoid_temp : bool
        Register the corresponding quantity as an ``nn.Parameter``.
    sigmoid_temp : float, default=1.0
        Temperature for the soft operations (higher = closer to hard).
    """

    def __init__(
        self,
        kernel_names: Optional[Sequence[KernelSpec]] = None,
        alpha: float = 0.12,
        beta: float = 0.9,
        threshold_mode: str = 'hard',
        gamma: Optional[Sequence[float]] = None,
        interp_factor: float = 0.25,
        window_size: int = 20,
        step_size: int = 4,
        kernel_params: Optional[Sequence[dict]] = None,
        learn_alpha: bool = False,
        learn_beta: bool = False,
        learn_gamma: bool = False,
        learn_sigmoid_temp: bool = False,
        sigmoid_temp: float = 1.0,
    ):
        super().__init__()

        if kernel_names is None:
            kernel_names = ['polynomial', 'gaussian']
        kernel_names = list(kernel_names)
        if len(kernel_names) == 0:
            raise ValueError("kernel_names must contain at least one kernel")
        n_kernels = len(kernel_names)

        if threshold_mode not in ('hard', 'soft'):
            raise ValueError(f"threshold_mode must be 'hard' or 'soft', got {threshold_mode!r}")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if step_size < 1:
            raise ValueError(f"step_size must be >= 1, got {step_size}")
        if interp_factor <= 0:
            raise ValueError(f"interp_factor must be > 0, got {interp_factor}")

        # Learnable kernels (nn.Module instances) must be registered so that
        # their parameters are visible to optimisers and .to(device).
        self.learnable_kernels = nn.ModuleDict()
        for i, kname in enumerate(kernel_names):
            if isinstance(kname, nn.Module):
                self.learnable_kernels[f'learnable_{i}'] = kname
            elif not (isinstance(kname, str) or callable(kname)):
                raise TypeError(f"kernel_names[{i}] must be a str, callable or nn.Module, got {type(kname)}")
        self.kernel_names = kernel_names

        if kernel_params is None:
            self.kernel_params = [Kernels.default_params(k) for k in kernel_names]
        else:
            if len(kernel_params) != n_kernels:
                raise ValueError("kernel_params length must match kernel_names")
            self.kernel_params = [dict(p) if p is not None else {} for p in kernel_params]

        self.interp_factor = float(interp_factor)
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.threshold_mode = threshold_mode

        if learn_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha), dtype=torch.float32))
        else:
            self.alpha = float(alpha)

        if learn_beta:
            self.beta = nn.Parameter(torch.tensor(float(beta), dtype=torch.float32))
        else:
            self.beta = float(beta)

        if learn_sigmoid_temp:
            self.sigmoid_temp = nn.Parameter(torch.tensor(float(sigmoid_temp), dtype=torch.float32))
        else:
            self.sigmoid_temp = float(sigmoid_temp)

        if gamma is None:
            gamma = [1.0 / n_kernels] * n_kernels
        gamma = list(gamma)
        if len(gamma) != n_kernels:
            raise ValueError("gamma length must match kernel_names")
        gamma_t = torch.tensor(gamma, dtype=torch.float32)
        if learn_gamma:
            self.gamma = nn.Parameter(gamma_t)
        else:
            self.register_buffer('gamma', gamma_t, persistent=False)

        self._last_phase: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ utils

    def normalized_gamma(self) -> torch.Tensor:
        """Kernel weights normalised to sum to 1."""
        return self.gamma / self.gamma.sum()

    def _interpolate_signal(self, signal: torch.Tensor) -> torch.Tensor:
        return cubic_interpolate_1d(signal, self.interp_factor)

    def _apply_kernels(self, signal: torch.Tensor) -> torch.Tensor:
        """Weighted sum of all kernels applied point-wise to ``signal``."""
        gamma = self.normalized_gamma().to(signal.dtype)
        result = None
        for i, (kname, params) in enumerate(zip(self.kernel_names, self.kernel_params)):
            learnable_key = f'learnable_{i}'
            if learnable_key in self.learnable_kernels:
                kern_out = self.learnable_kernels[learnable_key](signal)
            else:
                kern_out = Kernels.get(kname)(signal, **params)
            term = gamma[i] * kern_out
            result = term if result is None else result + term
        return result

    def _window_mask(self, n: int, n_windows: int, device) -> torch.Tensor:
        """Boolean mask of shape (n_windows, n): True inside each window."""
        idx = torch.arange(n, device=device)
        starts = torch.arange(n_windows, device=device) * self.step_size
        return (idx.unsqueeze(0) >= starts.unsqueeze(1)) & (idx.unsqueeze(0) < starts.unsqueeze(1) + self.window_size)

    # ---------------------------------------------------------------- forward

    def forward(self, signal: torch.Tensor, return_phase: bool = False):
        """Compute the DDKF time-frequency representation.

        Parameters
        ----------
        signal : torch.Tensor
            Shape ``(batch_size, length)`` or ``(length,)``.
        return_phase : bool, default=False
            Also return the phase tensor needed by :meth:`inverse_transform`.
            The phase of the last call is always stored internally as well.

        Returns
        -------
        torch.Tensor or (torch.Tensor, torch.Tensor)
            TFR of shape ``(batch_size, n_windows, n_freqs)`` (or
            ``(n_windows, n_freqs)`` for a 1D input), optionally with the
            phase tensor of the same shape.
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
            squeeze = True
        elif signal.dim() == 2:
            squeeze = False
        else:
            raise ValueError(f"signal must be 1D or 2D, got shape {tuple(signal.shape)}")
        if not torch.is_floating_point(signal):
            signal = signal.to(torch.get_default_dtype())

        batch_size = signal.shape[0]

        # Step 1: cubic interpolation (backpropagatable)
        interpolated_signal = self._interpolate_signal(signal)
        n = interpolated_signal.shape[1]

        if n < self.window_size:
            raise ValueError(f"Interpolated signal ({n}) shorter than window_size ({self.window_size})")

        n_windows = (n - self.window_size) // self.step_size + 1

        # Kernels are point-wise, so applying them inside a window and zeros
        # elsewhere equals applying them once and masking. Compute once.
        kernelised = self._apply_kernels(interpolated_signal)          # (batch, n)
        in_window = self._window_mask(n, n_windows, signal.device)     # (windows, n)
        zero = kernelised.new_zeros(())

        # Pass 1: kernel IN the window, zeros elsewhere
        sig_in = torch.where(in_window.unsqueeze(0), kernelised.unsqueeze(1), zero)   # (batch, windows, n)
        spec_in = torch.fft.fft(sig_in, dim=-1)
        del sig_in
        M = spec_in.abs()
        Mphase = spec_in.angle()
        del spec_in

        # Pass 2: zeros IN the window, kernel elsewhere
        sig_out = torch.where(in_window.unsqueeze(0), zero, kernelised.unsqueeze(1))
        spec_out = torch.fft.fft(sig_out, dim=-1)
        del sig_out
        M1 = spec_out.abs()
        M1phase = spec_out.angle()
        del spec_out

        # Smart minimum with a per-window beta threshold
        temp = self.sigmoid_temp
        beta_val = self.beta
        x, y = M, M1

        if self.threshold_mode == 'soft':
            x_max = (x * temp).logsumexp(dim=-1, keepdim=True) / temp
            beta_threshold = x_max * beta_val
            strong_mask = torch.sigmoid(temp * (x - beta_threshold))
            # Direct multiplication by beta gives beta a strong gradient path.
            combined = beta_val * y * x * strong_mask
            result = soft_minimum(x, combined, temperature=temp)
        else:
            x_max = x.max(dim=-1, keepdim=True)[0]
            beta_threshold = x_max * beta_val
            strong_mask = (x > beta_threshold).to(x.dtype)
            combined = y * x * strong_mask
            result = torch.minimum(x, combined)

        # Phase: take the in-window phase where the in-window spectrum won
        result_phase = torch.where(x < combined, Mphase, M1phase)
        del M, M1, Mphase, M1phase, combined, strong_mask

        # Global alpha threshold (paper eq. 4)
        flat = result.reshape(batch_size, -1)
        if self.threshold_mode == 'soft':
            alpha_max = (flat * temp).logsumexp(dim=1, keepdim=True) / temp
        else:
            alpha_max = flat.max(dim=1, keepdim=True)[0]

        threshold = (self.alpha * alpha_max).view(batch_size, 1, 1)

        if self.threshold_mode == 'soft':
            result = torch.sign(result) * torch.clamp(result.abs() - threshold, min=0.0)
        else:
            result = result * (result > threshold).to(result.dtype)

        self._last_phase = result_phase.detach()

        if squeeze:
            result = result.squeeze(0)
            result_phase = result_phase.squeeze(0)
        if return_phase:
            return result, result_phase
        return result

    # ---------------------------------------------------------------- inverse

    def inverse_transform(self, tfr: torch.Tensor,
                          tfr_phase: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reconstruct a per-window signal from a TFR (backpropagatable).

        Each window's spectrum is rebuilt from magnitude and phase, inverse
        transformed, summed and scaled by ``beta``, giving one value per
        window: output shape ``(batch_size, n_windows)`` or ``(n_windows,)``.

        If ``tfr_phase`` is omitted the phase stored by the last
        :meth:`forward` call is used; it must match ``tfr`` in shape.
        """
        if tfr.dim() == 2:
            tfr = tfr.unsqueeze(0)
            squeeze = True
        elif tfr.dim() == 3:
            squeeze = False
        else:
            raise ValueError(f"tfr must be 2D or 3D, got shape {tuple(tfr.shape)}")

        if tfr_phase is None:
            if self._last_phase is None:
                raise ValueError("No phase information available. Run forward() first or provide tfr_phase.")
            tfr_phase = self._last_phase
        elif tfr_phase.dim() == 2:
            tfr_phase = tfr_phase.unsqueeze(0)

        if tfr_phase.shape != tfr.shape:
            raise ValueError(
                f"tfr_phase shape {tuple(tfr_phase.shape)} does not match tfr shape {tuple(tfr.shape)}"
            )
        tfr_phase = tfr_phase.to(device=tfr.device, dtype=tfr.dtype)

        complex_spec = tfr * torch.exp(1j * tfr_phase)
        time_signal = torch.fft.ifft(complex_spec, dim=-1)
        result = time_signal.sum(dim=-1).abs()          # (batch, windows)
        result = self.beta * result

        return result.squeeze(0) if squeeze else result


class DDKFFeatureExtractor(nn.Module):
    """DDKF feature extractor for ML pipelines, optionally flattened."""

    def __init__(self, kernel_names=None, flatten: bool = False, **kwargs):
        super().__init__()
        self.ddkf = DDKFLayer(kernel_names=kernel_names, **kwargs)
        self.flatten = flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tfr = self.ddkf(x)
        if not self.flatten:
            return tfr
        if tfr.dim() == 2:          # single (unbatched) signal
            return tfr.reshape(-1)
        return tfr.reshape(tfr.size(0), -1)


# Aliases: the same algorithm is also published under the NTFA-LK name.
NTFALayer = DDKFLayer
NTFAFeatureExtractor = DDKFFeatureExtractor
