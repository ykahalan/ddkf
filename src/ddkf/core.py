"""Compact implementation of Dual Dynamic Kernel Filtering (2DKF).

This single module contains both the kernel functions and the TwoDKF class to
keep the package as small as possible for journal submission.
"""
from typing import Callable, Optional
import numpy as np
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft


class KernelFunctions:
    @staticmethod
    def polynomial(signal, degree: int = 2, offset: float = 1.3):
        s = np.asarray(signal)
        return (s + offset) ** degree

    @staticmethod
    def gaussian(signal, center: float = 0.7, sigma: float = 1.0):
        s = np.asarray(signal)
        return np.exp(-0.5 * ((s - center) / sigma) ** 2)

    @staticmethod
    def hybrid(signal, gamma: float = 0.5, poly_offset: float = 1.3, poly_degree: int = 2,
               gauss_center: float = 0.7, gauss_sigma: float = 1.0):
        poly = KernelFunctions.polynomial(signal, degree=poly_degree, offset=poly_offset)
        gauss = KernelFunctions.gaussian(signal, center=gauss_center, sigma=gauss_sigma)
        return gamma * poly + (1 - gamma) * gauss


class TwoDKF:
    def __init__(self, window_size: int = 20, step_size: int = 4, alpha: float = 0.15,
                 beta: float = 0.9, interp_factor: float = 0.25, kernel: str | Callable = "hybrid",
                 kernel_params: dict | None = None):
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.interp_factor = float(interp_factor)
        self.kernel = kernel
        self.kernel_params = kernel_params or {}

        self.signal_values_ = None
        self.M_ = None
        self.Mphase_ = None
        self.M1_ = None
        self.M1phase_ = None
        self.result_ = None
        self.resultphase_ = None

    def _get_kernel_function(self):
        if callable(self.kernel):
            return self.kernel
        if self.kernel == "polynomial":
            return lambda s: KernelFunctions.polynomial(s, **self.kernel_params)
        if self.kernel == "gaussian":
            return lambda s: KernelFunctions.gaussian(s, **self.kernel_params)
        if self.kernel == "hybrid":
            return lambda s: KernelFunctions.hybrid(s, **self.kernel_params)
        raise ValueError(f"Unknown kernel: {self.kernel}")

    def _interpolate_signal(self, signal: np.ndarray) -> np.ndarray:
        sig = np.asarray(signal).flatten()
        if sig.size < 2:
            return sig.copy()
        if self.interp_factor <= 0:
            raise ValueError("interp_factor must be > 0")
        num_points = int((sig.size - 1) / self.interp_factor) + 1
        t_new = np.linspace(0, sig.size - 1, num=num_points)
        f = interp1d(np.arange(sig.size), sig, kind="cubic")
        return f(t_new)

    def _apply_kernel(self, signal: np.ndarray) -> np.ndarray:
        s = np.asarray(signal).astype(float)
        if np.min(s) < 0:
            s = s - np.min(s)
        kfn = self._get_kernel_function()
        return kfn(s)

    def fit(self, signal: np.ndarray) -> "TwoDKF":
        self.signal_values_ = self._interpolate_signal(signal)
        kernel_values = self._apply_kernel(self.signal_values_)
        n = kernel_values.size
        if n < self.window_size:
            raise ValueError("Interpolated signal shorter than window_size")

        M_list, Mphase_list = [], []
        for i in range(0, n - self.window_size + 1, self.step_size):
            before = np.zeros(i)
            window = kernel_values[i : i + self.window_size]
            after = np.zeros(n - i - self.window_size)
            sig_win = np.concatenate([before, window, after])
            L = fft(sig_win)
            M_list.append(np.abs(L))
            Mphase_list.append(np.angle(L))

        self.M_ = np.array(M_list)
        self.Mphase_ = np.array(Mphase_list)

        M1_list, M1phase_list = [], []
        for u in range(0, n - self.window_size + 1, self.step_size):
            before2 = kernel_values[:u] if u > 0 else np.array([])
            in_window2 = np.zeros(self.window_size)
            after2 = kernel_values[u + self.window_size :]
            sig_win2 = np.concatenate([before2, in_window2, after2])
            L1 = fft(sig_win2)
            M1_list.append(np.abs(L1))
            M1phase_list.append(np.angle(L1))

        self.M1_ = np.array(M1_list)
        self.M1phase_ = np.array(M1phase_list)

        n_windows, n_freqs = self.M_.shape
        result = np.zeros_like(self.M_)
        resultphase = np.zeros_like(self.Mphase_)

        for i in range(n_windows):
            x = self.M_[i]
            y = self.M1_[i]
            strong_mask = x > (np.max(x) * self.alpha)
            combined = y * x * strong_mask
            result[i] = np.minimum(x, combined)
            mask_keep_x = (result[i] == x)
            resultphase[i] = np.where(mask_keep_x, self.Mphase_[i], self.M1phase_[i])

        flat = result.flatten()
        thresh = self.beta * np.max(flat) if flat.size else 0.0
        flat = flat * (flat > thresh)
        self.result_ = flat.reshape(result.shape)
        self.resultphase_ = resultphase

        return self

    def get_tfr(self) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("Call fit() before get_tfr()")
        return self.result_

    def inverse_transform(self, correction_factor: Optional[float] = None) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("Must call fit() before inverse_transform()")
        recov_windows = []
        for i in range(self.result_.shape[0]):
            comp = self.result_[i] * np.exp(1j * self.resultphase_[i])
            sig_time = ifft(comp)
            recov_windows.append(np.sum(np.abs(sig_time)))
        recov = np.asarray(recov_windows)
        if correction_factor is None:
            correction_factor = self.alpha
        return correction_factor * recov


def denoise(signal: np.ndarray, **kwargs) -> np.ndarray:
    dkf = TwoDKF(**kwargs)
    dkf.fit(signal)
    return dkf.inverse_transform(kwargs.get("correction_factor"))
