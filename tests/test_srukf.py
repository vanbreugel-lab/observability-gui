"""
The square-root UKF — ObservabilityEngine.run_srukf.

It exists to be a second opinion on run_ukf: the SAME estimation problem (same
seed → same measurements and initial estimate), carried as a Cholesky factor
and propagated with QR + rank-1 updates instead of the plain covariance form.
So the tests are agreement tests — on a well-conditioned problem the two must
collapse onto each other to round-off, because the only difference is the
factorization. Where they would part company (a problem so ill-conditioned that
run_ukf's psd() projection fires) is exactly what the option exists to reveal,
so it is not pinned here.

Pure NumPy, so unlike the dynamax backend there is nothing to skip.
"""
import numpy as np
import pytest

from conftest import BUILTINS, q_of, r_of, rel_diff


# ───────────────────────────── shape and sanity ─────────────────────────────

@pytest.mark.parametrize('system', BUILTINS)
def test_returns_the_same_contract_as_run_ukf(engines, system):
    eng = engines[system]
    n = len(eng.spec.state_names)
    X_hat, P_diag = eng.run_srukf(q_of(eng.spec), r_of(eng.spec), seed=0)
    assert X_hat.shape == (eng.N, n)
    assert P_diag.shape == (eng.N, n)
    assert np.isfinite(X_hat).all(), 'estimate went non-finite'
    assert (P_diag > 0).all(), 'non-positive variance on the diagonal'


def test_unexpected_keyword_arguments_are_tolerated(engine):
    """ Called through the same compute_payload path as run_ukf, so it must
    accept the UKF's full keyword set. """
    X, _ = engine.run_srukf(q_of(engine.spec), r_of(engine.spec), seed=0,
                            alpha=1.0, beta=2.0, kappa=0.0, full_cov=False,
                            sensors=tuple(engine.spec.fim_sensors))
    assert X.shape == (engine.N, len(engine.spec.state_names))


# ─────────────────────── agreement with the plain UKF ───────────────────────

@pytest.mark.parametrize('system', BUILTINS)
def test_agrees_with_the_plain_ukf(engines, system):
    """ Same problem, same tuning, same sensor subset: the square-root form is
    algebraically identical to run_ukf whenever run_ukf never needs its PSD
    projection, which holds on these short well-observable trajectories. A
    disagreement above round-off means the factorization changed the answer —
    the one thing it must not do. """
    eng = engines[system]
    spec = eng.spec
    sel = tuple(spec.fim_sensors)
    q, r = q_of(spec), r_of(spec)
    X_ukf, P_ukf = eng.run_ukf(q, r, seed=0, sensors=sel)
    X_sr, P_sr = eng.run_srukf(q, r, seed=0, sensors=sel)
    assert rel_diff(X_sr, X_ukf) < 1e-8, 'state estimate diverged from run_ukf'
    assert rel_diff(P_sr, P_ukf) < 1e-7, 'variance diverged from run_ukf'


# ─────────────────────────── internal consistency ───────────────────────────

def test_full_cov_diagonal_matches_p_diag(engine):
    """ full_cov returns reconstructed P_k = S_k S_kᵀ; its diagonal must be the
    P_diag the fast path reports. """
    q, r = q_of(engine.spec), r_of(engine.spec)
    X_hat, P_diag, P_hist = engine.run_srukf(q, r, seed=0, full_cov=True)
    assert len(P_hist) == engine.N
    for k, P in enumerate(P_hist):
        assert np.allclose(np.diag(P), P_diag[k], rtol=1e-10, atol=1e-14)
        # a genuine covariance: symmetric and PSD
        assert np.allclose(P, P.T, atol=1e-12)
        assert np.linalg.eigvalsh(0.5 * (P + P.T)).min() > -1e-12


def test_posterior_start_pins_the_first_estimate(engine):
    """ start_at='posterior' skips the k=0 correction, so X_hat[0] is the
    supplied guess verbatim — same convention as run_ekf / run_ukf. """
    spec = engine.spec
    n = len(spec.state_names)
    guess = np.full(n, np.nan)
    j = spec.state_names.index(spec.color_state_default)
    guess[j] = engine.X[0, j] + 0.5
    X_hat, _ = engine.run_srukf(q_of(spec), r_of(spec), seed=0,
                                x0_guess=guess, start_at='posterior')
    assert X_hat[0, j] == pytest.approx(engine.X[0, j] + 0.5)
