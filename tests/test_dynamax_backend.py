"""
The dynamax (JAX) filter backend — engine/dynamax_filters.py.

The backend exists to be a second opinion, so the tests are mostly agreement
tests: the same estimation problem run two ways has to come out the same. Where
the two implementations legitimately differ (the EKF's linearization point) the
difference is pinned by *removing* it — point the NumPy filter at dynamax's
convention and the answers must collapse onto each other.

The whole module skips cleanly when dynamax is not installed, because that is a
supported configuration: the app falls back to the NumPy filters and says so.
"""
import numpy as np
import pytest

import core
from conftest import BUILTINS, q_of, r_of, rel_diff, silent

pytestmark = pytest.mark.skipif(not core.DYNAMAX_AVAILABLE,
                                reason='dynamax is not installed')


@pytest.fixture(scope='module')
def dmx():
    import dynamax_filters
    assert dynamax_filters.AVAILABLE, dynamax_filters.IMPORT_ERROR
    return dynamax_filters


# ───────────────────────────── shape and sanity ─────────────────────────────

@pytest.mark.parametrize('system', BUILTINS)
@pytest.mark.parametrize('which', ['ekf', 'ukf'])
def test_returns_the_same_contract_as_the_numpy_runners(dmx, engines, system,
                                                        which):
    eng = engines[system]
    spec = eng.spec
    n = len(spec.state_names)
    run = dmx.run_ekf if which == 'ekf' else dmx.run_ukf
    X_hat, P_diag = run(eng, q_of(spec), r_of(spec), seed=0)
    assert X_hat.shape == (eng.N, n)
    assert P_diag.shape == (eng.N, n)
    assert np.isfinite(X_hat).all(), 'estimate went non-finite'
    assert (P_diag > 0).all(), 'non-positive variance on the diagonal'


def test_unexpected_keyword_arguments_are_tolerated(dmx, engine):
    """ The two backends are called through the same code path in
    compute_payload, so the dynamax runners have to accept (and ignore) the
    keywords only the NumPy ones understand. """
    spec = engine.spec
    X, _ = dmx.run_ekf(engine, q_of(spec), r_of(spec), seed=0,
                       full_cov=True, x0_scale=2.0, f_jac_at='post',
                       fd_eps_scaled=True)
    assert X.shape == (engine.N, len(spec.state_names))


def test_float64_is_enabled(dmx):
    """ JAX defaults to float32. A single-precision filter next to a
    double-precision one makes every comparison meaningless. """
    import jax.numpy as jnp
    assert jnp.zeros(1).dtype == np.float64


# ─────────────────────── agreement with the NumPy filters ───────────────────

def test_the_two_ekfs_agree_once_the_linearization_point_matches(dmx, engines):
    """ The sharp test. dynamax linearizes the prediction about the FILTERED
    mean; this repo's default is the PROPAGATED state. That is the only
    remaining difference between the two implementations, so pointing the NumPy
    EKF at dynamax's convention must collapse the difference to round-off — on
    the 18-state model with arctan2 measurements, not just on the easy one.

    If this loosens, the two have genuinely diverged. """
    for name in ('fly', 'fly7', 'drone', 'alt2d'):
        eng = engines[name]
        spec = eng.spec
        sel = tuple(spec.fim_sensors)
        X_dmx, _ = dmx.run_ekf(eng, q_of(spec), r_of(spec), seed=0, sensors=sel)
        X_eng, _ = eng.run_ekf(q_of(spec), r_of(spec), seed=0, sensors=sel,
                               f_jac_at='pre')
        assert rel_diff(X_dmx, X_eng) < 1e-6, name


def test_the_two_ukfs_agree(dmx, engines):
    """ No linearization point to reconcile here, so the UKFs should be close
    outright — the residue is the covariance-update form and the sigma-point
    square root. """
    for name in BUILTINS:
        eng = engines[name]
        spec = eng.spec
        kw = dict(alpha=1.0, beta=2.0, kappa=0.0)
        sel = tuple(spec.fim_sensors)
        X_dmx, _ = dmx.run_ukf(eng, q_of(spec), r_of(spec), seed=0,
                               sensors=sel, **kw)
        X_eng, _ = eng.run_ukf(q_of(spec), r_of(spec), seed=0, sensors=sel,
                               **kw)
        assert rel_diff(X_dmx, X_eng) < 5e-3, name


def test_both_backends_reach_the_same_accuracy(dmx, engines):
    """ The claim the UI makes: whichever implementation you pick, the estimate
    you are judging against the bound is the same estimate. """
    for name in BUILTINS:
        eng = engines[name]
        spec = eng.spec
        sel = tuple(spec.fim_sensors)
        for run_d, run_e, kw in (
                (dmx.run_ekf, eng.run_ekf, {}),
                (dmx.run_ukf, eng.run_ukf, dict(alpha=1.0, beta=2.0,
                                                kappa=0.0))):
            X_d, _ = run_d(eng, q_of(spec), r_of(spec), seed=0, sensors=sel,
                           **kw)
            X_e, _ = run_e(q_of(spec), r_of(spec), seed=0, sensors=sel, **kw)
            err_d = np.abs(X_d - eng.X).mean()
            err_e = np.abs(X_e - eng.X).mean()
            assert err_d == pytest.approx(err_e, rel=0.05), (name, err_d, err_e)


# ────────────────────────────── angular wrapping ────────────────────────────

def test_wrap_to_produces_exactly_the_wrapped_residual(dmx):
    """ The property the whole scheme rests on, tested directly rather than
    through a filter: after the shift, dynamax's plain y − h(x̂) IS the residual
    wrapped into (−π, π]. Non-angular channels must come through untouched. """
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    p = 6
    mask = np.zeros(p)
    mask[[0, 2, 5]] = 1.0                       # channels 0, 2, 5 are angular
    for _ in range(200):
        y = rng.uniform(-np.pi, np.pi, p)
        # raw predictions deliberately many turns away from the measurement
        raw = y + rng.integers(-4, 5, p) * 2 * np.pi + rng.normal(0, 0.7, p)
        shifted = np.asarray(dmx._wrap_to(jnp.asarray(y), jnp.asarray(raw),
                                          jnp.asarray(mask)), dtype=float)
        resid = y - shifted
        ang = mask.astype(bool)
        # angular residuals land in (-π, π] and equal the wrapped difference
        want = (y - raw + np.pi) % (2 * np.pi) - np.pi
        assert np.allclose(resid[ang], want[ang], atol=1e-9)
        assert np.all(np.abs(resid[ang]) <= np.pi + 1e-9)
        # everything else is exactly the unshifted prediction
        assert np.allclose(shifted[~ang], raw[~ang], atol=1e-12)


@pytest.mark.slow
def test_wrapping_moves_the_ukf_toward_the_numpy_filter(dmx):
    """ On a trajectory that actually crosses the ±π branch — the drone's own
    default, which turns through it — folding the wrap into h has to bring
    dynamax's UKF closer to the engine's circular-mean one, and closer to the
    truth.

    Deliberately NOT asserted in general: the engine brackets its sigma-point
    measurements around their own circular mean while this scheme brackets them
    around y_k. Both are legitimate, they differ once the cloud is wide, and on
    a barely-converged filter (a short trajectory swinging the heading through
    several radians) the residual difference goes either way by well under a
    percent. The claim is about branch crossings, not about being universally
    nearer. """
    spec = core.get_spec('drone')
    eng = core.ObservabilityEngine(spec)
    with silent():
        eng.simulate_mpc(spec.default_setpoint(float(spec.wind_default),
                                               float(spec.zeta_default)))
    kw = dict(alpha=1.0, beta=2.0, kappa=0.0, sensors=tuple(spec.fim_sensors))
    ref, _ = eng.run_ukf(q_of(spec), r_of(spec), seed=0, **kw)
    wrapped, _ = dmx.run_ukf(eng, q_of(spec), r_of(spec), seed=0,
                             wrap_angles=True, **kw)
    plain, _ = dmx.run_ukf(eng, q_of(spec), r_of(spec), seed=0,
                           wrap_angles=False, **kw)
    d_wrap, d_plain = rel_diff(wrapped, ref), rel_diff(plain, ref)
    assert d_wrap < d_plain / 2, (d_wrap, d_plain)
    err = lambda X: np.abs(X - eng.X).mean()
    assert err(wrapped) <= err(plain), (err(wrapped), err(plain))


def test_wrapping_is_a_no_op_without_angular_channels(dmx, engine):
    """ alt2d has none, so the flag must not perturb it at all — that is what
    makes it the reference system for exact agreement. """
    spec = engine.spec
    a, _ = dmx.run_ekf(engine, q_of(spec), r_of(spec), seed=0, wrap_angles=True)
    b, _ = dmx.run_ekf(engine, q_of(spec), r_of(spec), seed=0, wrap_angles=False)
    assert np.array_equal(a, b)


def test_wrapping_does_not_change_the_measurement_jacobian(dmx, engines):
    """ The wrap adds 2π·round(...), whose derivative is zero — so C must be
    the unshifted model's. Checked directly, because if jnp.round ever acquired
    a non-zero derivative the EKF would silently linearize the wrong thing. """
    import jax
    import jax.numpy as jnp
    eng = engines['fly']
    spec = eng.spec
    sel = tuple(spec.fim_sensors)
    P = dmx._problem(eng, q_of(spec), r_of(spec), 0, 0.0, None, None, None,
                     sel, 'uncorrelated', 'auto', True)
    assert P['wrapped'], 'the fly model should have wrapped channels'
    x = jnp.asarray(eng.X[0])
    uy = jnp.asarray(P['U'][0])
    n_u = len(spec.input_names)
    C_wrapped = np.asarray(jax.jacfwd(P['h'])(x, uy), dtype=float)
    plain = dmx._problem(eng, q_of(spec), r_of(spec), 0, 0.0, None, None, None,
                         sel, 'uncorrelated', 'auto', False)
    C_plain = np.asarray(jax.jacfwd(plain['h'])(x, jnp.asarray(plain['U'][0])),
                         dtype=float)
    assert C_wrapped.shape == C_plain.shape
    assert np.allclose(C_wrapped, C_plain, atol=1e-12)


# ───────────────────── the two ways of reaching the model ───────────────────

@pytest.mark.parametrize('system', BUILTINS)
def test_builtin_models_take_the_fast_jax_path(dmx, engines, system):
    """ All four are plain NumPy, so the jax.numpy rebuild should succeed. A
    silent fall back to pure_callback would still be correct but much slower —
    worth knowing about. """
    eng = engines[system]
    _f, _h, _sidx, mode = dmx._model_fns(eng, tuple(eng.spec.fim_sensors))
    assert mode == 'jax', f'{system} fell back to {mode}'


@pytest.mark.parametrize('which', ['ekf', 'ukf'])
def test_the_callback_path_agrees_with_the_jax_path(dmx, engine, which):
    """ The callback path is what an uploaded model that cannot be traced (one
    using scipy, or control flow on values) gets. It has to produce the same
    filter, or a custom system would quietly be filtered differently from a
    built-in one. """
    spec = engine.spec
    run = dmx.run_ekf if which == 'ekf' else dmx.run_ukf
    a, _ = run(engine, q_of(spec), r_of(spec), seed=0, mode='jax')
    b, _ = run(engine, q_of(spec), r_of(spec), seed=0, mode='callback')
    assert rel_diff(a, b) < 1e-6


def test_an_untraceable_model_falls_back_instead_of_failing(dmx, engine):
    """ mode='auto' must notice that the jax.numpy rebuild is unusable and take
    the callback path — the whole point of verifying the rebuild against the
    NumPy step. Simulated here with an f that JAX cannot trace. """
    import types
    spec = engine.spec
    bad = types.SimpleNamespace(**vars(spec))

    def untraceable(X, U):
        # a Python conditional on a traced value: fine in NumPy, a
        # ConcretizationTypeError under jit
        if float(X[0]) > 0:
            return list(np.asarray(spec.f(X, U), dtype=float))
        return list(np.asarray(spec.f(X, U), dtype=float))

    bad.f = untraceable
    stub = types.SimpleNamespace(spec=bad, X=engine.X, U=engine.U,
                                 N=engine.N, _rk4_sim=engine._rk4_sim)
    _f, _h, _sidx, mode = dmx._model_fns(stub, None, 'auto')
    assert mode == 'callback'
    # and mode='jax' must refuse loudly rather than silently produce nonsense
    with pytest.raises(Exception):
        dmx._model_fns(stub, None, 'jax')


# ─────────────────────────────── Q conventions ──────────────────────────────

def test_vanloan_q_is_accepted(dmx, engine):
    spec = engine.spec
    qd = {s: 1e-4 for s in spec.state_names}
    for run in (dmx.run_ekf, dmx.run_ukf):
        X, P = run(engine, qd, r_of(spec), seed=0, q_noise='vanloan')
        assert np.isfinite(X).all() and (P > 0).all()


def test_limitations_are_reported_accurately(dmx):
    """ These strings go into the status line, so they must not claim a caveat
    that no longer applies (angular wrapping) or hide one that does (the UKF's
    constant Q). """
    fly, alt2d = core.get_spec('fly'), core.get_spec('alt2d')
    assert dmx.limitations(alt2d) == []
    assert dmx.limitations(fly) == [], 'wrapping is on by default now'
    assert any('wrap' in s for s in dmx.limitations(fly, wrap_angles=False))
    assert any('constant Q' in s for s in dmx.limitations(alt2d, 'vanloan'))


# ───────────────────────── the app-level integration ────────────────────────

def test_selecting_the_backend_changes_which_filter_runs(app, engine):
    """ Through compute_payload, the way the page reaches it. """
    spec = engine.spec
    base = dict(w_raw=4, eps=1e-4, lam=1e-6, r_diag=r_of(spec),
                q_diag=q_of(spec), q_noise='uncorrelated',
                sensors=tuple(spec.measurement_names),
                states=tuple(spec.state_names), do_stoch=False, bases=(),
                do_ann=False)
    est = dict(EKF=core._realization(spec, 0, 'none', 0, 0, 0, 0.0, None),
               UKF=core._realization(spec, 0, 'none', 0, 0, 0, 0.0, None),
               ukf_alpha=1.0, ukf_beta=2.0, ukf_kappa=0.0)
    eng_payload = core.compute_payload(engine, ('update',),
                                       dict(base, backend='engine'), est)
    dmx_payload = core.compute_payload(engine, ('update',),
                                       dict(base, backend='dynamax'), est)
    assert eng_payload['backend'] == 'engine'
    assert dmx_payload['backend'] == 'dynamax'
    assert 'via dynamax' in dmx_payload['note']
    # same problem, so the two must land on the same estimate
    assert rel_diff(dmx_payload['results']['EKF'][0],
                    eng_payload['results']['EKF'][0]) < 1e-2


def test_the_backend_is_part_of_the_filter_cache_key(engine):
    """ The two implementations are allowed to differ, so serving one's cached
    result for the other would hide exactly what the option exists to show. """
    spec = engine.spec
    engine._filt_cache.clear()
    base = dict(w_raw=4, eps=1e-4, lam=1e-6, r_diag=r_of(spec),
                q_diag=q_of(spec), q_noise='uncorrelated',
                sensors=tuple(spec.measurement_names),
                states=tuple(spec.state_names), do_stoch=False, bases=(),
                do_ann=False)
    est = dict(EKF=core._realization(spec, 0, 'none', 0, 0, 0, 0.0, None),
               UKF=core._realization(spec, 0, 'none', 0, 0, 0, 0.0, None),
               ukf_alpha=1.0, ukf_beta=2.0, ukf_kappa=0.0)
    core.compute_payload(engine, ('update',), dict(base, backend='engine'), est)
    n_after_engine = len(engine._filt_cache)
    core.compute_payload(engine, ('update',), dict(base, backend='dynamax'), est)
    assert len(engine._filt_cache) > n_after_engine, \
        'the dynamax run was served from the engine backend cache'
