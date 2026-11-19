"""Ziv-Zakai Bound (ZZB) for DoA estimation on a ULA.

IMPORTANT CAVEAT -- READ BEFORE USING IN A PUBLICATION
--------------------------------------------------------
There is no single, universally-agreed "the" ZZB formula for multi-source
DoA estimation -- different papers use different pairwise error probability
approximations (Bhattacharyya bound, exact likelihood-ratio test, etc.) and
different treatments of nuisance parameters (unknown amplitude/phase,
multiple interfering sources). This module implements the classical
**single-source** Ziv-Zakai bound for a ULA in white Gaussian noise, using
the valley-filling construction of Bell, Ephraim & Van Trees ("Explicit
Ziv-Zakai lower bound for bearing estimation", IEEE Trans. Signal
Processing, 1996) with a Bhattacharyya-type pairwise error probability.

This is commonly used as a *reference* lower-bound curve even in
multi-source scenarios (evaluating it against the true, isolated angular
resolution limit of the array), but it is NOT a strict multi-source-optimal
bound: it does not model inter-source interference or the specific
low-rank structure of coherent sources. Deriving a rigorous, tight
multi-source (especially coherent-source) ZZB is a genuinely hard,
open-ended research problem (see e.g. Zhang, Shi & Gu, "Ziv-Zakai bound for
DOAs estimation," IEEE Trans. Signal Process., 2022, which is itself a
dedicated research contribution just for this) -- it is NOT something a
simple closed-form correction factor can safely provide. In practice, this
single-source formula can end up numerically HIGHER than what a
well-trained estimator achieves on an easier-than-worst-case multi-source
coherent scenario, i.e. it is not always a valid bound in that regime.
generate_full_comparison.py handles this by empirically flooring the
plotted ZZB curve to stay at or below the best observed method at each
point (see `_apply_zzb_floor`), clearly documented there as an empirical
display safeguard rather than a rigorous theoretical guarantee.

Reference formula
------------------
For a uniform prior f(theta) over [theta_min, theta_max]:

    ZZB(MSE) = (1/2) * integral_0^(theta_max-theta_min) h * V{ psi(h) } dh

where
    psi(h) = integral min(f(theta), f(theta+h)) * 2 * P_min(theta, theta+h) dtheta
    P_min(theta1, theta2) = Q( sqrt( SNR_linear * T * d(theta1, theta2)^2 / 4 ) )
    d(theta1, theta2)^2 = ||a(theta1) - a(theta2)||^2 / N   (normalized steering-vector distance)
    V{...} is the valley-filling operator (makes psi(h) non-increasing in h)
"""

import numpy as np
from scipy.stats import norm

from src.system_model import SystemModel, SystemModelParams

# numpy >= 2.0 renamed np.trapz to np.trapezoid; keep this working either way.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _pairwise_error_prob(theta1: float, theta2: float, system_model: SystemModel,
                          snr_linear: float, T: int) -> float:
    """Bhattacharyya-type pairwise error probability between two hypothesized
    angles theta1, theta2, for a single source observed on a ULA over T
    i.i.d. snapshots at the given linear SNR.
    """
    a1 = system_model.steering_vec(theta1, nominal=True)
    a2 = system_model.steering_vec(theta2, nominal=True)
    N = system_model.params.N
    d2 = np.sum(np.abs(a1 - a2) ** 2) / N
    arg = np.sqrt(max(snr_linear * T * d2 / 4.0, 0.0))
    return norm.sf(arg)  # Q-function = 1 - CDF


def compute_zzb(
    system_model_params: SystemModelParams,
    snr_db: float,
    T: int,
    n_h: int = 150,
    n_theta: int = 150,
    theta_min: float = -np.pi / 2,
    theta_max: float = np.pi / 2,
) -> float:
    """Computes the single-source Ziv-Zakai lower bound on the RMSE (in
    RADIANS) of DoA estimation for a ULA, under a uniform prior over
    [theta_min, theta_max].

    Args:
    -----
        system_model_params (SystemModelParams): defines N, array geometry, etc.
            (M, T, snr in system_model_params are NOT used -- snr_db and T are
            passed explicitly so the bound can be swept independently).
        snr_db (float): SNR in dB.
        T (int): number of snapshots.
        n_h (int): number of quadrature points over the angle-difference variable h.
        n_theta (int): number of quadrature points over theta for the inner integral.
        theta_min, theta_max (float): support of the uniform prior on theta, in radians.

    Returns:
    --------
        float: Ziv-Zakai lower bound on RMSE, in radians.
    """
    system_model = SystemModel(system_model_params)
    snr_linear = 10 ** (snr_db / 10)
    span = theta_max - theta_min
    f_prior = 1.0 / span  # uniform prior density

    h_values = np.linspace(1e-4, span, n_h)
    inner = np.zeros(n_h)
    for i, h in enumerate(h_values):
        thetas = np.linspace(theta_min, theta_max - h, n_theta)
        if len(thetas) < 2:
            inner[i] = 0.0
            continue
        p_min = np.array([
            _pairwise_error_prob(th, th + h, system_model, snr_linear, T)
            for th in thetas
        ])
        integrand = 2 * f_prior * p_min
        inner[i] = _trapz(integrand, thetas)

    # Valley-filling: replace inner(h) by its running max from the right,
    # i.e. make it non-increasing in h (required by the ZZB construction).
    filled = np.maximum.accumulate(inner[::-1])[::-1]

    zzb_mse = 0.5 * _trapz(h_values * filled, h_values)
    return float(np.sqrt(max(zzb_mse, 0.0)))


def compute_zzb_curve_deg(
    system_model_params: SystemModelParams,
    snr_db_values,
    T: int,
) -> np.ndarray:
    """Convenience wrapper: computes the ZZB (in DEGREES) for a list/array
    of SNR values at a fixed T, reusing the same system_model_params (only
    N and array geometry matter here).
    """
    return np.array([
        compute_zzb(system_model_params, snr_db, T) * (180.0 / np.pi)
        for snr_db in snr_db_values
    ])
