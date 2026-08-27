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
def test_state_wrapping_keeps_the_ukf_in_the_engine_branch(dmx):
    """ On a trajectory that crosses the ±π branch — the drone's own default,
    which turns through it — dynamax lets an angular STATE drift into whatever
    2π branch it started in, while the engine keeps it in (−π, π]. `_unpack`
    canonicalizes dynamax's angular-state columns with the same wrap, so the
    two land in the same branch and agree; without it they would sit an integer
    number of turns apart and the linear comparison would blow up. """
    spec = core.get_spec('drone')
    eng = core.ObservabilityEngine(spec)
    with silent():
        eng.simulate_mpc(spec.default_setpoint(float(spec.wind_default),
                                               float(spec.zeta_default)))
    kw = dict(alpha=1.0, beta=2.0, kappa=0.0, sensors=tuple(spec.fim_sensors))
    ref, _ = eng.run_ukf(q_of(spec), r_of(spec), seed=0, **kw)
    got, _ = dmx.run_ukf(eng, q_of(spec), r_of(spec), seed=0, **kw)
    ang = dmx._angular_state_idx(spec)
    assert ang, 'the drone should have angular states'
    assert np.all(np.abs(got[:, ang]) <= np.pi + 1e-9), 'not canonicalized'
    assert rel_diff(got, ref) < 5e-3, rel_diff(got, ref)


def test_no_angular_channels_means_tight_agreement(dmx, engine):
    """ alt2d has neither angular measurements nor angular states, so nothing is
    wrapped on either side — which is what makes it the reference system for
    tight agreement with the engine. """
    spec = engine.spec
    assert not getattr(spec, 'angle_states', ()) and not spec.angle_measurements
    assert dmx._angular_state_idx(spec) == []
    a, _ = dmx.run_ekf(engine, q_of(spec), r_of(spec), seed=0)
    b, _ = engine.run_ekf(q_of(spec), r_of(spec), seed=0, f_jac_at='pre')
    assert rel_diff(a, b) < 1e-6


def test_wrapping_does_not_change_the_measurement_jacobian(dmx, engines):
    """ Wrapping angular measurements folds a 2π·round(...) shift into h, whose
    derivative is zero — so ∂h/∂x must equal the unshifted model's. Checked
    directly, because if jnp.round ever acquired a non-zero derivative the EKF
    would silently linearize the wrong thing. The unshifted h is `_model_fns`'s
    own sensor-restricted output; the shifted one is what `_problem` builds. """
    import jax
    import jax.numpy as jnp
    eng = engines['fly']
    spec = eng.spec
    sel = tuple(spec.fim_sensors)
    x = jnp.asarray(eng.X[0])
    _f, h_core, _sidx, _mode = dmx._model_fns(eng, sel)      # unshifted h
    C_plain = np.asarray(jax.jacfwd(h_core)(x, jnp.asarray(eng.U[0])),
                         dtype=float)
    P = dmx._problem(eng, q_of(spec), r_of(spec), 0, 0.0, None, None, None,
                     sel, 'uncorrelated', 'auto')            # wrapping always on
    assert P['wrapped'], 'the fly model should have wrapped channels'
    C_wrapped = np.asarray(jax.jacfwd(P['h'])(x, jnp.asarray(P['U'][0])),
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
                                 N=engine.N, _rk4_sim=engine._rk4_sim,
                                 _sensor_subset=engine._sensor_subset)
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
    """ These strings go into the status line, so they must name a caveat that
    applies and omit one that does not. """
    fly, alt2d = core.get_spec('fly'), core.get_spec('alt2d')
    # alt2d has no angular states and, with uncorrelated Q, nothing to flag
    assert dmx.limitations(alt2d) == []
    # fly has angular states, which dynamax canonicalizes only after filtering
    assert any('angular state' in s for s in dmx.limitations(fly))
    # the UKF's constant-Q restriction applies under the Van Loan stack
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


def test_switching_the_backend_actually_reruns_the_filters(app, monkeypatch):
    """ "They look the same — is it even recomputing?"

    The two implementations agree to within a percent or two on the plotted
    states, so the curves overlap and the eye cannot tell them apart. That makes
    it worth proving mechanically that the switch does real work: dynamax's own
    runner has to be invoked, and its result has to land in a separate cache
    slot from the NumPy one. """
    import dynamax_filters as dmx_mod
    from test_compute import args

    calls = []
    real = dmx_mod.run_ekf
    monkeypatch.setattr(dmx_mod, 'run_ekf',
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    with silent():
        first = app._compute(*args(app, 'fly7', est_backend='engine'),
                             rebuild=True)
        sess = first[-1]
        assert not calls, 'the NumPy path called into dynamax'
        second = app._compute(*args(app, 'fly7', sess=sess,
                                    est_backend='dynamax'), rebuild=False)
    assert calls, 'selecting dynamax never called its EKF'
    assert 'via dynamax' in second[-2], second[-2]
    slots = {(k[0], k[1]) for k in sess['engine']._filt_cache}
    assert ('EKF', 'engine') in slots and ('EKF', 'dynamax') in slots, slots
    # switching back must serve the earlier NumPy result from cache rather than
    # recomputing it — that is the cache working, not the switch failing
    n_before = len(calls)
    with silent():
        app._compute(*args(app, 'fly7', sess=sess, est_backend='engine'),
                     rebuild=False)
    assert len(calls) == n_before


def test_the_backend_control_is_wired_to_recompute(app):
    """ A control that exists but is not in the recompute list looks exactly
    like a backend that is not being recalculated: you click it and nothing
    happens until the next Simulate. Check it is in the `.input` loop. """
    import ast
    import gradio as gr
    from test_compute import APP_PY

    events = [getattr(e, 'event_name', str(e)) for e in gr.Radio.EVENTS]
    assert 'input' in events, f'gr.Radio has no .input event: {events}'
    wired = set()
    for node in ast.walk(ast.parse(open(APP_PY).read())):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple):
            names = {e.id for e in node.iter.elts if isinstance(e, ast.Name)}
            if 'est_backend' in names:
                wired |= names
    assert 'est_backend' in wired, 'est_backend is not in any recompute loop'
    assert 'est_split' in wired


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
