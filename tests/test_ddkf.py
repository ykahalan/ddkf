"""Tests for the ddkf package.

The ``reference_*`` helpers below re-implement the original loop-based
algorithm (one FFT per window, Python loops) so that the vectorised
implementation can be checked against it numerically.
"""
import math
import pathlib

import pytest
import torch

import ddkf
from ddkf import (
    DDKFLayer,
    DDKFFeatureExtractor,
    Kernels,
    LearnableChebyshevKernel,
    NTFALayer,
    NTFAFeatureExtractor,
    cubic_interpolate_1d,
    soft_minimum,
)

torch.manual_seed(0)
ROOT = pathlib.Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------
# Loop-based reference implementation (original algorithm, kept for regression)
# ----------------------------------------------------------------------------

def reference_forward(layer: DDKFLayer, signal: torch.Tensor):
    """Original window-by-window implementation of DDKFLayer.forward."""
    if signal.dim() == 1:
        signal = signal.unsqueeze(0)
    batch_size = signal.shape[0]
    interp = cubic_interpolate_1d(signal, layer.interp_factor)
    n = interp.shape[1]
    ws, ss = layer.window_size, layer.step_size
    n_windows = (n - ws) // ss + 1

    M_list, Mphase_list, M1_list, M1phase_list = [], [], [], []
    for i in range(n_windows):
        start = i * ss
        values_in = layer._apply_kernels(interp[:, start:start + ws])
        before = torch.zeros(batch_size, start, dtype=interp.dtype)
        after = torch.zeros(batch_size, n - start - ws, dtype=interp.dtype)
        L = torch.fft.fft(torch.cat([before, values_in, after], dim=1))
        M_list.append(L.abs())
        Mphase_list.append(L.angle())

        vb = layer._apply_kernels(interp[:, :start]) if start > 0 else torch.zeros(batch_size, 0, dtype=interp.dtype)
        va = layer._apply_kernels(interp[:, start + ws:]) if start + ws < n else torch.zeros(batch_size, 0, dtype=interp.dtype)
        L1 = torch.fft.fft(torch.cat([vb, torch.zeros(batch_size, ws, dtype=interp.dtype), va], dim=1))
        M1_list.append(L1.abs())
        M1phase_list.append(L1.angle())

    M, Mphase = torch.stack(M_list, 1), torch.stack(Mphase_list, 1)
    M1, M1phase = torch.stack(M1_list, 1), torch.stack(M1phase_list, 1)

    result = torch.zeros_like(M)
    result_phase = torch.zeros_like(Mphase)
    temp, beta = layer.sigmoid_temp, layer.beta
    for i in range(n_windows):
        x, y = M[:, i, :], M1[:, i, :]
        if layer.threshold_mode == 'soft':
            x_max = (x * temp).logsumexp(dim=1, keepdim=True) / temp
            thr = x_max * beta
            mask = torch.sigmoid(temp * (x - thr))
            combined = beta * y * x * mask
            result[:, i, :] = soft_minimum(x, combined, temperature=temp)
        else:
            thr = x.max(dim=1, keepdim=True)[0] * beta
            mask = (x > thr).to(x.dtype)
            combined = y * x * mask
            result[:, i, :] = torch.minimum(x, combined)
        result_phase[:, i, :] = torch.where(M[:, i, :] < combined, Mphase[:, i, :], M1phase[:, i, :])

    flat = result.view(batch_size, -1)
    if layer.threshold_mode == 'soft':
        amax = (flat * temp).logsumexp(dim=1, keepdim=True) / temp
    else:
        amax = flat.max(dim=1, keepdim=True)[0]
    threshold = (layer.alpha * amax).view(batch_size, 1, 1)
    if layer.threshold_mode == 'soft':
        result = torch.sign(result) * torch.clamp(result.abs() - threshold, min=0.0)
    else:
        result = result * (result > threshold).to(result.dtype)
    return result, result_phase


def reference_inverse(layer: DDKFLayer, tfr: torch.Tensor, phase: torch.Tensor):
    """Original double-loop implementation of DDKFLayer.inverse_transform."""
    batch_size, n_windows, _ = tfr.shape
    out = []
    for b in range(batch_size):
        wins = []
        for i in range(n_windows):
            spec = tfr[b, i] * torch.exp(1j * phase[b, i])
            wins.append(torch.abs(torch.sum(torch.fft.ifft(spec))))
        out.append(torch.stack(wins))
    return layer.beta * torch.stack(out)


# ----------------------------------------------------------------------------
# Packaging
# ----------------------------------------------------------------------------

def test_public_api_and_aliases():
    assert NTFALayer is DDKFLayer
    assert NTFAFeatureExtractor is DDKFFeatureExtractor
    for name in ddkf.__all__:
        assert hasattr(ddkf, name), name


def test_version_matches_pyproject():
    tomllib = pytest.importorskip("tomllib")
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["name"] == "ddkf"
    assert data["project"]["version"] == ddkf.__version__


# ----------------------------------------------------------------------------
# Interpolation
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_interpolation_length_endpoints_dtype(dtype):
    x = torch.linspace(0, 4 * math.pi, 50, dtype=dtype)
    y = torch.sin(x)
    yi = cubic_interpolate_1d(y, 0.25)
    assert yi.dtype == dtype
    assert yi.shape == (int(49 / 0.25) + 1,)
    assert torch.allclose(yi[0], y[0], atol=1e-6)
    assert torch.allclose(yi[-1], y[-1], atol=1e-6)
    t = torch.linspace(0, 4 * math.pi, len(yi), dtype=dtype)
    assert (yi - torch.sin(t)).abs().max() < 0.05


def test_interpolation_identity_factor_one():
    y = torch.randn(40)
    assert torch.allclose(cubic_interpolate_1d(y, 1.0), y, atol=1e-5)


def test_interpolation_batched_and_grad():
    y = torch.randn(3, 30, requires_grad=True)
    out = cubic_interpolate_1d(y, 0.5)
    assert out.shape == (3, 59)
    out.sum().backward()
    assert y.grad is not None and y.grad.abs().sum() > 0


def test_interpolation_rejects_bad_factor():
    with pytest.raises(ValueError):
        cubic_interpolate_1d(torch.randn(10), 0.0)


# ----------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------

def test_gaussian_default_center_matches_docs():
    assert Kernels.gaussian(torch.tensor(0.7)).item() == pytest.approx(1.0)
    assert Kernels.DEFAULT_PARAMS['gaussian']['center'] == 0.7


def test_default_params_follow_kernel_names_not_position():
    layer = DDKFLayer(kernel_names=['gaussian', 'polynomial'])
    assert layer.kernel_params == [{'center': 0.7, 'sigma': 1.0}, {'degree': 2, 'offset': 1.3}]
    layer(torch.randn(100))  # must not raise

    layer3 = DDKFLayer(kernel_names=['polynomial', 'gaussian', 'matern32'])
    assert layer3.kernel_params[1] == {'center': 0.7, 'sigma': 1.0}
    assert layer3.kernel_params[2] == {}


@pytest.mark.parametrize("name", Kernels.available())
def test_every_builtin_kernel_runs(name):
    layer = DDKFLayer(kernel_names=[name])
    out = layer(torch.randn(120))
    assert torch.isfinite(out).all()


def test_unknown_kernel_raises():
    with pytest.raises(ValueError):
        Kernels.get('nope')


def test_custom_callable_kernel():
    layer = DDKFLayer(kernel_names=[lambda x, scale=1.0: torch.exp(-x * scale), 'polynomial'],
                      kernel_params=[{'scale': 2.0}, {'degree': 3, 'offset': 1.0}])
    out = layer(torch.randn(100))
    assert out.shape[1] == int(99 / 0.25) + 1


# ----------------------------------------------------------------------------
# Forward pass
# ----------------------------------------------------------------------------

def test_forward_shapes():
    layer = DDKFLayer()
    sig = torch.randn(200)
    n = int(199 / 0.25) + 1
    n_windows = (n - 20) // 4 + 1
    assert layer(sig).shape == (n_windows, n)
    assert layer(sig.unsqueeze(0).repeat(3, 1)).shape == (3, n_windows, n)


def test_forward_float64_and_integer_inputs():
    layer = DDKFLayer()
    out64 = layer(torch.randn(100, dtype=torch.float64))
    assert out64.dtype == torch.float64
    out_int = layer(torch.arange(100))
    assert torch.is_floating_point(out_int)


def test_forward_rejects_bad_configs():
    sig = torch.randn(100)
    with pytest.raises(ValueError):
        DDKFLayer(step_size=0)
    with pytest.raises(ValueError):
        DDKFLayer(window_size=0)
    with pytest.raises(ValueError):
        DDKFLayer(threshold_mode='medium')
    with pytest.raises(ValueError):
        DDKFLayer(kernel_params=[{}])          # 1 param set for 2 kernels
    with pytest.raises(ValueError):
        DDKFLayer(gamma=[1.0])                 # 1 weight for 2 kernels
    with pytest.raises(ValueError):
        DDKFLayer(window_size=10_000)(sig)     # longer than the signal
    with pytest.raises(ValueError):
        DDKFLayer()(torch.randn(2, 3, 50))


@pytest.mark.parametrize("mode", ['hard', 'soft'])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_vectorised_forward_matches_loop_reference(mode, dtype):
    layer = DDKFLayer(kernel_names=['polynomial', 'gaussian', 'matern52'], gamma=[0.5, 0.3, 0.2],
                      threshold_mode=mode, alpha=0.05, beta=0.8, window_size=16, step_size=3,
                      interp_factor=0.5)
    sig = torch.randn(2, 90, dtype=dtype)
    tfr, phase = layer(sig, return_phase=True)
    ref_tfr, ref_phase = reference_forward(layer, sig)
    tol = 1e-4 if dtype == torch.float32 else 1e-9
    assert torch.allclose(tfr, ref_tfr, rtol=tol, atol=tol)
    # phase only matters where the TFR is non-zero
    nz = tfr > 0
    assert torch.allclose(phase[nz], ref_phase[nz], rtol=tol, atol=tol)


def test_learnable_chebyshev_kernel_matches_reference_and_gets_grads():
    layer = DDKFLayer(kernel_names=[LearnableChebyshevKernel(4), 'gaussian'])
    sig = torch.randn(80)
    tfr = layer(sig)
    ref, _ = reference_forward(layer, sig)
    assert torch.allclose(tfr, ref.squeeze(0), rtol=1e-4, atol=1e-4)
    tfr.sum().backward()
    assert layer.learnable_kernels['learnable_0'].coeffs.grad.abs().sum() > 0


def test_hard_mode_gradient_reaches_input():
    layer = DDKFLayer()
    sig = torch.randn(100, requires_grad=True)
    layer(sig).sum().backward()
    assert sig.grad.abs().sum() > 0


def test_soft_mode_all_parameters_learnable():
    layer = DDKFLayer(threshold_mode='soft', learn_alpha=True, learn_beta=True,
                      learn_gamma=True, learn_sigmoid_temp=True)
    names = {n for n, _ in layer.named_parameters()}
    assert names == {'alpha', 'beta', 'gamma', 'sigmoid_temp'}
    layer(torch.randn(100)).sum().backward()
    for p in layer.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_gamma_is_normalised_and_scale_invariant():
    sig = torch.randn(100)
    a = DDKFLayer(gamma=[0.5, 0.5])(sig)
    b = DDKFLayer(gamma=[2.0, 2.0])(sig)
    assert torch.allclose(a, b)
    assert torch.allclose(DDKFLayer(gamma=[3.0, 1.0]).normalized_gamma(), torch.tensor([0.75, 0.25]))


def test_state_dict_round_trip_with_learnable_gamma():
    layer = DDKFLayer(learn_gamma=True, learn_alpha=True)
    with torch.no_grad():
        layer.gamma.copy_(torch.tensor([0.9, 0.1]))
    other = DDKFLayer(learn_gamma=True, learn_alpha=True)
    other.load_state_dict(layer.state_dict())
    assert torch.allclose(other.gamma, layer.gamma)


def test_move_to_dtype():
    layer = DDKFLayer(learn_gamma=True).double()
    out = layer(torch.randn(100, dtype=torch.float64))
    assert out.dtype == torch.float64


# ----------------------------------------------------------------------------
# Inverse transform
# ----------------------------------------------------------------------------

def test_inverse_transform_1d_with_stored_phase():
    layer = DDKFLayer()
    tfr = layer(torch.randn(100))
    rec = layer.inverse_transform(tfr)
    assert rec.shape == (tfr.shape[0],)
    assert torch.isfinite(rec).all()


def test_inverse_transform_batched_and_explicit_phase_match_reference():
    layer = DDKFLayer(beta=0.85)
    sig = torch.randn(3, 100)
    tfr, phase = layer(sig, return_phase=True)
    rec_stored = layer.inverse_transform(tfr)
    rec_explicit = layer.inverse_transform(tfr, phase)
    ref = reference_inverse(layer, tfr, phase)
    assert rec_stored.shape == (3, tfr.shape[1])
    assert torch.allclose(rec_stored, ref, rtol=1e-4, atol=1e-5)
    assert torch.allclose(rec_explicit, ref, rtol=1e-4, atol=1e-5)
    # 1D tfr with a 2D phase (the documented call style)
    rec_1d = layer.inverse_transform(tfr[0], phase[0])
    assert torch.allclose(rec_1d, ref[0], rtol=1e-4, atol=1e-5)


def test_inverse_transform_errors():
    layer = DDKFLayer()
    with pytest.raises(ValueError):
        layer.inverse_transform(torch.zeros(5, 10))        # no forward yet
    tfr = layer(torch.randn(100))
    with pytest.raises(ValueError):
        layer.inverse_transform(tfr[:3])                   # stored phase shape mismatch
    with pytest.raises(ValueError):
        layer.inverse_transform(tfr, tfr[:3])


def test_inverse_transform_is_differentiable():
    layer = DDKFLayer(threshold_mode='soft', learn_beta=True)
    tfr = layer(torch.randn(100))
    layer.inverse_transform(tfr).sum().backward()
    assert layer.beta.grad is not None


# ----------------------------------------------------------------------------
# Feature extractor
# ----------------------------------------------------------------------------

def test_feature_extractor_flatten():
    fe = DDKFFeatureExtractor(flatten=True)
    sig = torch.randn(100)
    n = int(99 / 0.25) + 1
    n_windows = (n - 20) // 4 + 1
    assert fe(sig).shape == (n_windows * n,)
    assert fe(sig.unsqueeze(0).repeat(2, 1)).shape == (2, n_windows * n)
    assert DDKFFeatureExtractor(flatten=False)(sig).shape == (n_windows, n)
