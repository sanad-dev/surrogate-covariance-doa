"""Sanity tests for the v2 fixes (notes/03_rerun_plan.md, CC9 and review R2-R4).

Run from the project folder:  python tests/test_v2_fixes.py   (pytest also works)
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_v2 import batched, source_errors  # noqa: E402
from itertools import permutations  # noqa: E402

from src.criterions import RMSPE, RMSPELoss, toeplitz_hermitian_projection  # noqa: E402
from src.data_handler import autocorrelation_matrix, create_observations, model_features  # noqa: E402
from src.methods import Esprit, RootMUSIC  # noqa: E402
from src.models import DeepAugmentedMUSIC, DeepCNN, SubspaceNet, SubspaceNetAttn, esprit, root_music  # noqa: E402
from src.signal_creation import Samples  # noqa: E402
from src.system_model import SystemModelParams  # noqa: E402
from src.training import TrainingParams  # noqa: E402
from src.utils import device, gram_diagonal_overload, set_unified_seed  # noqa: E402


def params(**kw):
    p = SystemModelParams()
    base = dict(N=8, M=3, T=200, snr=10, signal_type="NarrowBand",
                signal_nature="non-coherent", eta=0, bias=0, sv_noise_var=0)
    for k, v in {**base, **kw}.items():
        p.set_parameter(k, v)
    return p


def fitted_spacing(a, theta):
    """Element spacing [wavelengths] implied by a steering vector's phase slope."""
    slope = np.polyfit(np.arange(len(a)), np.unwrap(np.angle(a)), 1)[0]
    return -slope / (2 * np.pi * np.sin(theta))


# 1. SNR (M1): nominal SNR equals measured source power over unit noise power
def test_snr_is_true_snr():
    np.random.seed(0)
    for snr in (-10, 0, 10, 20):
        s = Samples(params(snr=snr, T=50000))
        s.set_doa(None)
        power_db = 10 * np.log10(np.mean(np.abs(s.signal_creation()) ** 2))
        assert abs(power_db - snr) < 0.1, (snr, power_db)


# 2. One array geometry per sample, shared by all its sources (M2, CC1)
def test_one_geometry_per_sample():
    np.random.seed(1)
    s = Samples(params(bias=0.05))
    s.set_doa([-50.0, 20.0, 40.0])
    spacings = []
    for _ in range(2):
        A = s.samples_creation()[2]
        d = [fitted_spacing(A[:, j], th) for j, th in enumerate(s.doa)]
        assert np.ptp(d) < 1e-9, d  # same spacing for every source of the sample
        spacings.append(d[0])
    assert abs(spacings[0] - spacings[1]) > 1e-6  # a new realization per sample


# 3. Estimators recover known DoAs from an ideal covariance (torch and numpy)
def test_estimators_on_ideal_covariance():
    s = Samples(params())
    doa = np.array([-40.0, 5.0, 33.0])
    s.set_doa(list(doa))
    A = np.array([s.steering_vec(t, nominal=True) for t in s.doa]).T
    R = A @ A.conj().T + 0.01 * np.eye(8)
    Rt = torch.tensor(R).unsqueeze(0)
    for est in (root_music(Rt, 3, 1)[0], esprit(Rt, 3, 1)):
        assert np.allclose(np.sort(np.rad2deg(est.numpy().real.ravel())), doa, atol=1e-6)
    for est in (RootMUSIC(s).narrowband(R, mode="covariance")[0], Esprit(s).narrowband(R, mode="covariance")[0]):
        assert np.allclose(np.sort(np.real(est)), doa, atol=1e-6)


# 4. Vectorized code equals the former per-sample loops (CC6)
def _old_gram(K, eps):
    return torch.stack([k.conj().T @ k + eps * torch.eye(k.shape[0]) for k in K])


def _old_esprit(Rz, M):
    out = []
    for R in Rz:
        w, V = torch.linalg.eig(R)
        Us = V[:, torch.argsort(torch.abs(w)).flip(0)][:, :M]
        phi = torch.linalg.pinv(Us[:-1]) @ Us[1:]
        out.append(-torch.arcsin(torch.angle(torch.linalg.eig(phi)[0]) / np.pi))
    return torch.stack(out)


def _old_root_music(Rz, M):
    out = []
    for R in Rz:
        w, V = torch.linalg.eig(R)
        Un = V[:, torch.argsort(torch.abs(w)).flip(0)][:, M:]
        F = Un @ Un.conj().T
        c = torch.stack([torch.diagonal(F, k).sum() for k in range(-len(R) + 1, len(R))])
        A = torch.diag(torch.ones(len(c) - 2, dtype=c.dtype), -1)
        A[0, :] = -c[1:] / c[0]
        r = torch.linalg.eigvals(A)
        r = r[sorted(range(len(r)), key=lambda k: abs(abs(r[k]) - 1))]
        out.append(torch.arcsin(torch.angle(r[torch.abs(r) < 1][:M]) / np.pi))
    return torch.stack(out)


def _old_toeplitz(Rz):
    out = []
    N = Rz.shape[-1]
    for R in Rz:
        P = torch.zeros_like(R)
        for k in range(N):
            if k == 0:
                P[range(N), range(N)] = R.diagonal().real.mean().to(R.dtype)
            else:
                c = torch.cat((R.diagonal(k), R.diagonal(-k).conj())).mean()
                i = torch.arange(N - k)
                P[i, i + k], P[i + k, i] = c, c.conj()
        out.append(P)
    return torch.stack(out)


def _old_autocorrelation(X, lag):
    R = torch.zeros(X.shape[0], X.shape[0], dtype=torch.complex128)
    n = X.shape[1] - lag
    for t in range(n):
        R += torch.outer(X[:, t] - X.mean(), X[:, t + lag].conj() - X.mean())
    R = R / n
    return torch.cat((R.real, R.imag), 0)


def _old_rmspe_loss(pred, doa):
    total = 0
    for p, t in zip(pred, doa):
        vals = [torch.linalg.norm(((torch.stack(q) - t) + np.pi / 2) % np.pi - np.pi / 2) / np.sqrt(len(t))
                for q in permutations(p)]
        total = total + torch.min(torch.stack([v for v in vals]))
    return total


def test_vectorized_equals_loops():
    torch.manual_seed(0)
    pred, doa = torch.randn(32, 3, dtype=torch.float64), torch.randn(32, 3, dtype=torch.float64)
    old = _old_rmspe_loss([torch.stack(list(p)) for p in pred], doa)
    assert torch.allclose(RMSPELoss()(pred, doa).cpu(), torch.as_tensor(old), atol=1e-9)
    K = torch.randn(16, 8, 8, dtype=torch.complex128)
    assert torch.allclose(gram_diagonal_overload(K, 1, 16).cpu(), _old_gram(K, 1), atol=1e-6)
    Rz = _old_gram(K, 1)
    assert torch.allclose(esprit(Rz, 3, 16), _old_esprit(Rz, 3), atol=1e-6)
    assert torch.allclose(torch.sort(root_music(Rz, 3, 16)[0])[0], torch.sort(_old_root_music(Rz, 3))[0], atol=1e-6)
    assert torch.allclose(toeplitz_hermitian_projection(Rz), _old_toeplitz(Rz), atol=1e-6)
    X = torch.randn(8, 50, dtype=torch.complex128)
    for lag in (0, 3, 7):
        assert torch.allclose(autocorrelation_matrix(X, lag), _old_autocorrelation(X, lag), atol=1e-6)


# 5. Baseline reshapes (CC4): the first layers receive correctly ordered tensors
class _Capture(torch.nn.Module):
    def forward(self, x):
        self.seen = x
        raise StopIteration


def test_baseline_input_layout():
    X = torch.randn(2, 8, 8, 3)
    cnn = DeepCNN(8, 361)
    cnn.conv1 = _Capture()
    try:
        cnn(X)
    except StopIteration:
        pass
    assert torch.equal(cnn.conv1.seen, X.permute(0, 3, 1, 2))
    assert DeepCNN(8, 361)(X).shape == (2, 361)  # full forward runs after the permute
    Xc = torch.randn(2, 8, 20, dtype=torch.complex64)
    da = DeepAugmentedMUSIC(8, 20, 3)
    da.BatchNorm = _Capture()
    try:
        da(Xc)
    except StopIteration:
        pass
    assert torch.equal(da.BatchNorm.seen, torch.cat((Xc.real, Xc.imag), 1).permute(0, 2, 1))


def test_damusic_noise_subspace_sorted():
    da = DeepAugmentedMUSIC(8, 20, 3)
    eig = torch.tensor([0.1, 5.0, 0.2, 4.0, 0.3, 3.0, 0.4, 0.5])  # signal = indices 1, 3, 5
    Rz = torch.diag(eig).to(torch.complex64).unsqueeze(0)
    noise = torch.eye(8, dtype=torch.complex64)[:, [0, 2, 4, 6, 7]]
    expected = da.spectrum_calculation(noise.unsqueeze(0))[0]
    assert torch.allclose(da.pre_MUSIC(Rz), expected, rtol=1e-4)


# 6. RMSPE: periodic wrap and permutation, hand-computed
def test_rmspe_wrap_and_permutation():
    expected = np.deg2rad(np.sqrt((2**2 + 5**2) / 2))  # pairs 10->12 and -80->-85
    assert np.isclose(RMSPE(np.array([10.0, -80.0]), np.array([-85.0, 12.0])), expected)
    assert np.isclose(np.sqrt(np.mean(source_errors([10.0, -80.0], [-85.0, 12.0]) ** 2)), expected)
    assert np.isclose(RMSPE(np.array([89.0]), np.array([-89.0])), np.deg2rad(2.0))  # wraps across +-90


# 7. Classical ESPRIT on a calibrated array at 20 dB, T=200, M=3 is accurate
def test_esprit_calibrated_high_snr():
    np.random.seed(2)
    s = Samples(params(snr=20))
    errs = []
    for _ in range(200):
        s.set_doa(None)
        X = s.samples_creation()[0]
        errs.append(RMSPE(Esprit(s).narrowband(X)[0], np.rad2deg(s.doa)))
    assert np.rad2deg(np.mean(errs)) < 1.0, np.rad2deg(np.mean(errs))


# 8. (R4) Per-source power offsets
def test_source_power_offsets():
    np.random.seed(3)
    for nature in ("non-coherent", "coherent"):
        s = Samples(params(T=50000, signal_nature=nature, source_power_db=[0, 0, -20]))
        s.set_doa(None)
        p = 10 * np.log10(np.mean(np.abs(s.signal_creation()) ** 2, axis=1))
        assert abs((p[2] - p[0]) - (-20)) < 0.2, (nature, p)


# 9. (R4) Mixed SNR draws stay in range and spread over it
def test_mixed_snr_range():
    np.random.seed(4)
    s = Samples(params(T=4, snr=(-10, 20)))
    s.set_doa(None)
    draws = []
    for _ in range(1000):
        s.signal_creation()
        draws.append(s.snr_draw)
    assert -10 <= min(draws) and max(draws) <= 20
    assert min(draws) < -8 and max(draws) > 18


# 10. (R3) Batched evaluation scores every sample of the batch
def test_batched_scores_every_sample():
    torch.manual_seed(0)
    model = SubspaceNet(8, 3, "esprit").to(device).eval()  # load_runs also moves models to device
    x = torch.randn(2, 8, 16, 8)
    doa, _ = batched(model, x)
    singles = torch.cat([batched(model, x[i : i + 1])[0] for i in range(2)])
    assert doa.shape == (2, 3)
    assert torch.allclose(doa, singles, atol=1e-5)
    assert not torch.allclose(doa[0], doa[1])


# R2: the train/validation split does not depend on the run seed
def test_split_independent_of_seed():
    data = [(torch.tensor([float(i)]), torch.tensor([0.0])) for i in range(100)]
    splits = []
    for seed in (0, 1):
        set_unified_seed(seed)
        tp = TrainingParams().set_batch_size(10).set_training_dataset(data)
        splits.append(sorted(float(x) for x, _ in tp.valid_dataset.dataset))
    assert splits[0] == splits[1]


# CC3: DeepCNN grid labels hit the nearest 0.5-degree bin
def test_deepcnn_grid_labels():
    doa = torch.deg2rad(torch.tensor([-90.0, 0.26, 45.1], dtype=torch.float64))
    y = model_features([(torch.zeros(8, 4, dtype=torch.complex64), doa)], "DeepCNN", grid_labels=True)[0][1]
    assert torch.nonzero(y).flatten().tolist() == [0, 181, 270]


# X10: per-sample SIR draw. Element power = source powers + unit noise, so against equal powers
# it grows by (2 + 10^(-SIR/10)) / 3 at high SNR: between 1 (0 dB) and 34 (-20 dB)
def test_sir_range_draw():
    power = lambda obs: np.array([float((X.abs() ** 2).mean()) for X, _ in obs])
    p = params(T=2000, snr=30)
    set_unified_seed(5)
    equal = power(create_observations(p, 60)[0]).mean()
    ratio = power(create_observations(p, 60, sir_range=(-20, 0))[0]) / equal
    assert 0.9 < ratio.min() < 2 and 15 < ratio.max() < 36, (ratio.min(), ratio.max())
    assert p.source_power_db is None  # the caller's params are restored


# X12: coherent sources give a rank-one signal part, so the second eigenvalue of the sample
# covariance stays at the noise level
def test_coherent_fraction_draw():
    def second_to_first(obs):
        ev = [torch.linalg.eigvalsh(X @ X.conj().T).flip(0) for X, _ in obs]
        return float(np.mean([e[1] / e[0] for e in ev]))
    p = params(snr=20)
    set_unified_seed(7)
    assert second_to_first(create_observations(p, 40, coherent_fraction=1.0)[0]) < 0.02
    assert second_to_first(create_observations(p, 40, coherent_fraction=0.0)[0]) > 0.1
    assert p.signal_nature == "non-coherent"  # the caller's params are restored


# X6: residual output of SubspaceNetAttn. With a zero head the correction is I (res_psd) or 0
# (res_herm), so the surrogate covariance is the lag-0 sample covariance (+ I)
def test_residual_output():
    set_unified_seed(6)
    obs, _ = create_observations(params(), 4)
    x = torch.stack([f for f, _ in model_features(obs, "SubspaceNetAttn", tau=8)])
    R0 = torch.complex(x[:, 0, :8], x[:, 0, 8:])
    R0 = (R0 + R0.conj().transpose(1, 2)) / 2
    for output, shift in (("res_psd", 1.0), ("res_herm", 0.0)):
        m = SubspaceNetAttn(tau=8, N=8, M=3, output=output).eval()
        torch.nn.init.zeros_(m.recon_head[-1].weight)
        torch.nn.init.zeros_(m.recon_head[-1].bias)
        with torch.no_grad():
            assert torch.allclose(m(x)[3], R0 + shift * torch.eye(8), atol=1e-5), output


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # report every test, do not stop at the first failure
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
