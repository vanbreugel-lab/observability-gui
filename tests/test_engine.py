"""
ObservabilityEngine: the estimation problem, the sensor subset, the filters and
the observability numerics.

The properties pinned here are the ones the app's *claims* rest on. "Both
filters see identical data when the seeds match" is a sentence in the UI; if
`_estimation_problem` stops being reproducible it becomes a lie and every
head-to-head plot silently compares two different problems. Likewise the ±2σ
band means nothing if P is not positive, and the min-EV curve means nothing if
it does not respond to λ and R the way the equations say.
"""
import numpy as np
import pandas as pd
import pytest

import core
from conftest import BUILTINS, N_STEPS, build_engine, q_of, r_of, rel_diff


# ──────────────────────────── the solved trajectory ─────────────────────────

def test_trajectory_shapes_and_time(engines):
    for name, eng in engines.items():
        n, m = len(eng.spec.state_names), len(eng.spec.input_names)
        assert eng.N == N_STEPS, (name, eng.N)
        assert eng.X.shape == (eng.N, n), (name, eng.X.shape)
        assert eng.U.shape == (eng.N, m), (name, eng.U.shape)
        assert np.isfinite(eng.X).all() and np.isfinite(eng.U).all()
        t = np.asarray(eng.t, dtype=float).ravel()
        assert len(t) == eng.N
        assert np.all(np.diff(t) > 0), name
        assert np.allclose(np.diff(t), eng.spec.dt), name


def test_resolving_a_trajectory_invalidates_every_cache(fresh_engine):
    """ The caches are keyed on `_version`; if a re-solve did not clear them the
    app would keep plotting the previous trajectory's Gramians. """
    eng = fresh_engine
    spec = eng.spec
    qd, rd = q_of(spec), r_of(spec)
    eng.empirical_ev(4, 1e-4, rd, 1e-6)
    eng.run_ekf(qd, rd, seed=0)
    v0 = eng._version
    assert eng._ev_emp_cache or eng._eom_cache
    eng.set_trajectory(eng.t, eng.X, eng.U)
    assert eng._version != v0
    assert not eng._ev_emp_cache and not eng._eom_cache and not eng._filt_cache


# ───────────────────────── one estimation realization ───────────────────────

def test_same_seed_gives_the_identical_problem(engine):
    """ This is what makes an EKF-vs-UKF comparison meaningful, and what the
    filter cache assumes. """
    rd = r_of(engine.spec)
    a = engine._estimation_problem(rd, 7)
    b = engine._estimation_problem(rd, 7)
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_different_seeds_give_different_data(engine):
    rd = r_of(engine.spec)
    Y0 = engine._estimation_problem(rd, 0)[0]
    Y1 = engine._estimation_problem(rd, 1)[0]
    assert not np.allclose(Y0, Y1)


def test_measurement_noise_scales_with_R(engine):
    """ R is a variance, so the sample spread must scale like sqrt(R). A factor
    slip here silently mis-tunes every filter against the bound. """
    spec = engine.spec
    Y_true = np.array([np.asarray(spec.h(engine.X[k], engine.U[k]), dtype=float)
                       for k in range(engine.N)])
    s_small = (engine._estimation_problem(r_of(spec, 1e-6), 0)[0] - Y_true).std()
    s_big = (engine._estimation_problem(r_of(spec, 1e-2), 0)[0] - Y_true).std()
    assert s_big > s_small
    assert s_big / max(s_small, 1e-30) == pytest.approx(100.0, rel=0.5)


def test_x0_guess_is_used_verbatim_and_widens_that_states_p0(engine):
    """ A pinned guess must be exactly what was typed — and the prior must admit
    an error that big, or the filter starts out overconfident and the ±2σ band
    lies. """
    spec = engine.spec
    n = len(spec.state_names)
    rd = r_of(spec)
    _, xh_base, P_base, _, _ = engine._estimation_problem(rd, 0)
    guess = np.full(n, np.nan)
    far = engine.X[0][0] + 25.0
    guess[0] = far
    _, xh, P, _, _ = engine._estimation_problem(rd, 0, x0_guess=guess)
    assert xh[0] == far
    assert np.allclose(xh[1:], xh_base[1:])         # untouched states unchanged
    assert P[0, 0] > P_base[0, 0]
    assert np.allclose(np.diag(P)[1:], np.diag(P_base)[1:])


def test_p0_diag_pins_only_the_rows_it_names(engine):
    spec = engine.spec
    n = len(spec.state_names)
    rd = r_of(spec)
    _, _, P_base, _, _ = engine._estimation_problem(rd, 0)
    v = np.full(n, np.nan)
    v[0] = 42.0
    _, _, P, _, _ = engine._estimation_problem(rd, 0, p0_diag=v)
    assert P[0, 0] == pytest.approx(42.0)
    assert np.allclose(np.diag(P)[1:], np.diag(P_base)[1:])
    # a scalar broadcasts over every state (the Streamlit single-box path)
    _, _, P_all, _, _ = engine._estimation_problem(rd, 0, p0_diag=3.0)
    assert np.allclose(np.diag(P_all), 3.0)


def test_p0_and_x0_length_mismatches_are_rejected_loudly(engine):
    """ Better a ValueError naming the mismatch than a silently broadcast or
    truncated vector. """
    rd = r_of(engine.spec)
    with pytest.raises(ValueError, match='expected'):
        engine._estimation_problem(rd, 0, p0_diag=np.ones(2))
    with pytest.raises(ValueError, match='expected'):
        engine._estimation_problem(rd, 0, x0_guess=np.ones(2))


def test_inputs_are_untouched_without_a_disturbance(engine):
    """ Zero noise and no bias must leave U bit-identical: the "clean" case has
    to be genuinely clean, since the truth trajectory flew those inputs. """
    _, _, _, _, U = engine._estimation_problem(r_of(engine.spec), 0)
    assert np.array_equal(U, engine.U)


def test_input_bias_is_confined_to_its_channel_and_window(engine):
    spec = engine.spec
    if len(spec.input_names) < 2 or engine.N < 6:
        pytest.skip('needs at least two input channels')
    t = np.asarray(engine.t, dtype=float).ravel()
    t0, t1 = float(t[2]), float(t[5])
    _, _, _, _, U = engine._estimation_problem(
        r_of(spec), 0, u_bias=(1, 0.75, t0, t1))
    d = U - engine.U
    assert np.allclose(d[:, 0], 0.0)                    # other channels clean
    inside = (t >= t0) & (t < t1)
    assert np.allclose(d[inside, 1], 0.75)
    assert np.allclose(d[~inside, 1], 0.0)              # [t0, t1) is half-open


def test_input_noise_is_zero_mean_and_scales(engine):
    spec = engine.spec
    _, _, _, _, U = engine._estimation_problem(r_of(spec), 0, u_noise_var=1e-2)
    d = U - engine.U
    assert not np.allclose(d, 0.0)
    assert abs(d.mean()) < 0.5 * 0.1                    # loose: N is small


# ──────────────────────────── the sensor subset ─────────────────────────────

def test_unknown_sensor_is_a_clear_error(engine):
    spec = engine.spec
    Y, _, _, R, _ = engine._estimation_problem(r_of(spec), 0)
    with pytest.raises(ValueError, match='unknown sensors'):
        engine._sensor_subset(['not_a_sensor'], Y, R)


def test_empty_selection_means_every_channel(engine):
    spec = engine.spec
    Y, _, _, R, _ = engine._estimation_problem(r_of(spec), 0)
    for sel in (None, [], ()):
        _h, Ys, Rs, _ang = engine._sensor_subset(sel, Y, R)
        assert Ys.shape == Y.shape and Rs.shape == R.shape


def test_subset_follows_spec_order_not_the_callers(engines):
    """ The same set given in a different order must produce the same problem,
    or the filter cache would key two identical runs differently. """
    eng = engines['drone']
    spec = eng.spec
    Y, _, _, R, _ = eng._estimation_problem(r_of(spec), 0)
    a = eng._sensor_subset(['psi', 'g'], Y, R)
    b = eng._sensor_subset(['g', 'psi'], Y, R)
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
    assert a[3] == b[3]


def test_angular_indices_are_remapped_into_the_subset(engines):
    """ Angular channels are wrapped by index. Indexing them against the FULL
    measurement vector while a subset is in play wraps the wrong columns — a
    silent corruption, not a crash. """
    eng = engines['fly']
    spec = eng.spec
    names = list(spec.measurement_names)
    Y, _, _, R, _ = eng._estimation_problem(r_of(spec), 0)
    for sel in (['psi', 'gamma'], ['a', 'g'], ['phi', 'a'], names):
        h_sel, Ys, Rs, ang = eng._sensor_subset(sel, Y, R)
        kept = [m for m in names if m in set(sel)]
        assert Ys.shape[1] == len(kept)
        assert ang == [i for i, m in enumerate(kept)
                       if m in spec.angle_measurements]
        # h_sel must return exactly those channels, in that order
        got = np.asarray(h_sel(eng.X[0], eng.U[0]), dtype=float)
        full = np.asarray(spec.h(eng.X[0], eng.U[0]), dtype=float)
        assert np.allclose(got, [full[names.index(m)] for m in kept])


# ──────────────────────────────── the filters ───────────────────────────────

@pytest.mark.parametrize('system', BUILTINS)
@pytest.mark.parametrize('which', ['ekf', 'ukf'])
def test_filters_return_a_usable_estimate(engines, system, which):
    eng = engines[system]
    spec = eng.spec
    n = len(spec.state_names)
    run = eng.run_ekf if which == 'ekf' else eng.run_ukf
    X_hat, P_diag = run(q_of(spec), r_of(spec), seed=0)
    assert X_hat.shape == (eng.N, n)
    assert P_diag.shape == (eng.N, n)
    assert np.isfinite(X_hat).all(), 'estimate went non-finite'
    assert np.isfinite(P_diag).all(), 'covariance went non-finite'
    # a variance must stay positive: the plotted band is 2*sqrt(P)
    assert (P_diag > 0).all(), 'non-positive variance on the diagonal'


@pytest.mark.parametrize('which', ['ekf', 'ukf'])
def test_filters_are_deterministic(engine, which):
    spec = engine.spec
    run = engine.run_ekf if which == 'ekf' else engine.run_ukf
    a = run(q_of(spec), r_of(spec), seed=0)
    engine._filt_cache.clear()
    b = run(q_of(spec), r_of(spec), seed=0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


@pytest.mark.parametrize('which', ['ekf', 'ukf'])
def test_filters_beat_the_initial_error(engine, which):
    """ Not an accuracy claim — just that the filter is wired the right way
    round. An estimator whose error at the end is worse than the guess it
    started from is not filtering, and every sign error I can think of shows up
    here. """
    spec = engine.spec
    run = engine.run_ekf if which == 'ekf' else engine.run_ukf
    X_hat, _ = run(q_of(spec), r_of(spec), seed=0)
    obs = spec.state_names.index('z')           # alt2d: z is observable while accelerating
    start = abs(X_hat[0, obs] - engine.X[0, obs])
    late = np.abs(X_hat[engine.N // 2:, obs] - engine.X[engine.N // 2:, obs]).mean()
    assert late <= start + 1e-9, (start, late)


def test_ekf_full_cov_is_symmetric_and_psd(engine):
    """ full_cov feeds the NEES consistency checks, so P must be a real
    covariance, not just its diagonal. The Joseph form exists to keep it so. """
    spec = engine.spec
    X_hat, P_diag, P_hist = engine.run_ekf(q_of(spec), r_of(spec), seed=0,
                                           full_cov=True)
    assert len(P_hist) == engine.N
    for k, P in enumerate(P_hist):
        assert np.allclose(P, P.T, atol=1e-10), f'P[{k}] not symmetric'
        assert np.min(np.linalg.eigvalsh(P)) > -1e-9, f'P[{k}] not PSD'
        assert np.allclose(np.diag(P), P_diag[k])


def test_filters_can_be_driven_by_a_sensor_subset(engines):
    """ The estimators must see exactly the channels the observability analysis
    sees, or the bound and the estimate answer different questions. """
    eng = engines['drone']
    spec = eng.spec
    X_all, _ = eng.run_ekf(q_of(spec), r_of(spec), seed=0)
    X_one, _ = eng.run_ekf(q_of(spec), r_of(spec), seed=0, sensors=['psi'])
    assert np.isfinite(X_one).all()
    assert not np.allclose(X_all, X_one), 'the sensor selection had no effect'


def test_ukf_sigma_point_tuning_has_an_effect(engine):
    spec = engine.spec
    base, _ = engine.run_ukf(q_of(spec), r_of(spec), seed=0, alpha=1.0)
    wide, _ = engine.run_ukf(q_of(spec), r_of(spec), seed=0, alpha=0.5)
    assert np.isfinite(wide).all()
    assert not np.array_equal(base, wide)


def test_vanloan_q_is_accepted_by_both_filters(engine):
    spec = engine.spec
    qd, rd = {s: 1e-4 for s in spec.state_names}, r_of(spec)
    for run in (engine.run_ekf, engine.run_ukf):
        X_hat, P_diag = run(qd, rd, seed=0, q_noise='vanloan')
        assert np.isfinite(X_hat).all() and (P_diag > 0).all()


def test_bigger_q_widens_the_covariance(engine):
    """ Q is model distrust: more of it must leave the filter less certain.
    Backwards here would invert every band in the States plot. """
    spec = engine.spec
    rd = r_of(spec)
    _, P_small = engine.run_ekf({s: 1e-8 for s in spec.state_names}, rd, seed=0)
    _, P_big = engine.run_ekf({s: 1e-1 for s in spec.state_names}, rd, seed=0)
    assert P_big.mean() > P_small.mean()


# ─────────────────────────── observability numerics ─────────────────────────

def _ev(engine, w=4, eps=1e-4, r=None, lam=1e-6, **kw):
    """ empirical_ev returns (frame, was_cached) — the frame's selected-state
    columns as a float array, which is what every monotonicity check below
    compares. """
    spec = engine.spec
    frame, _cached = engine.empirical_ev(w, eps, r or r_of(spec), lam, **kw)
    cols = list(kw.get('states') or spec.state_names)
    return frame[cols].to_numpy(dtype=float)


def test_empirical_ev_frame_shape(engine):
    spec = engine.spec
    ev, cached = engine.empirical_ev(4, 1e-4, r_of(spec), 1e-6)
    assert isinstance(ev, pd.DataFrame)
    assert cached in (True, False)
    assert 'time_initial' in ev.columns
    for s in spec.state_names:
        assert s in ev.columns
    vals = ev[list(spec.state_names)].to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    assert finite.size and (finite > 0).all(), 'a min error variance was ≤ 0'


def test_min_ev_decreases_with_lambda(engine):
    """ diag((F + λI)^-1) is decreasing in λ. This is the sanity check on the
    regularizer: it also means an unobservable direction reports ~1/λ rather
    than ∞. """
    small = _ev(engine, lam=1e-8)
    big = _ev(engine, lam=1e-4)
    ok = np.isfinite(small) & np.isfinite(big)
    assert ok.any()
    assert (big[ok] <= small[ok] * (1 + 1e-9)).all()


def _informative(lam, *frames):
    """ Mask of entries that are information-limited rather than
    regularizer-limited.

    An unobservable (or barely observable) direction reports ≈1/λ *by
    construction* — that is what λ is for — so those entries carry no
    information about R or the sensor set, and comparing them across two runs
    compares eigenvalue-truncation round-off. On alt2d with λ=1e-8, x sits at
    exactly 1e8 and z, v_x within an order of it; only v_z is actually being
    bounded by the data. Keep the entries at most 1% of 1/λ. """
    ok = np.ones_like(frames[0], dtype=bool)
    for f in frames:
        ok &= np.isfinite(f) & (f < 0.01 / lam)
    return ok


def test_min_ev_scales_with_measurement_noise(engine):
    """ F = OᵀR⁻¹O, so the bound is linear in R: 100× the noise variance must
    give 100× the minimum error variance on the states the data actually
    constrains. """
    spec = engine.spec
    lam = 1e-8
    quiet = _ev(engine, r=r_of(spec, 1e-3), lam=lam)
    noisy = _ev(engine, r=r_of(spec, 1e-1), lam=lam)
    ok = _informative(lam, quiet, noisy)
    assert ok.any(), 'no information-limited entries to compare'
    assert (noisy[ok] >= quiet[ok] * (1 - 1e-9)).all()
    assert np.allclose(noisy[ok] / quiet[ok], 100.0, rtol=1e-6)


def test_more_sensors_never_hurt(engines):
    """ Adding a channel adds information, so the bound can only tighten. """
    eng = engines['drone']
    lam, states = 1e-8, ['z', 'v_x']
    # r_x = v_x/z and ground speed together already pin both states (r_x alone
    # only sees the ratio, and leaves them at 1/λ), so this compares two
    # information-limited bounds rather than two regularized ones.
    few = _ev(eng, lam=lam, sensors=['r_x', 'g'], states=states)
    many = _ev(eng, lam=lam, sensors=list(eng.spec.measurement_names),
               states=states)
    ok = _informative(lam, few, many)
    assert ok.any(), 'no information-limited entries to compare'
    assert (many[ok] <= few[ok] * (1 + 1e-6)).all()
    # and the extra channels must actually buy something somewhere
    assert (many[ok] < few[ok] * (1 - 1e-6)).any()


def test_fisher_inverse_diagonal_is_the_min_ev(engine):
    """ The |F⁻¹| map and the min-EV heatmap are drawn from different calls and
    documented as showing the same numbers on the diagonal. Pin that, or the two
    plots can drift apart. """
    spec = engine.spec
    w, k, eps, lam = 4, 3, 1e-4, 1e-6
    rd = r_of(spec)
    states = list(spec.state_names)
    ev, _ = engine.empirical_ev(w, eps, rd, lam, states=states)
    F_inv, kk = engine.empirical_window_fisher_inv(w, eps, rd, lam, k,
                                                  states=states)
    row = ev.iloc[kk][states].to_numpy(dtype=float)
    diag = np.diag(np.asarray(F_inv, dtype=float))
    assert np.allclose(diag, row, rtol=1e-6, atol=1e-12), (diag, row)


def test_observability_matrix_shape(engine):
    spec = engine.spec
    w, sensors = 4, list(spec.measurement_names)
    states = list(spec.state_names)
    O, k = engine.empirical_window_O(w, 1e-4, 2, sensors=sensors, states=states)
    assert O.shape == (w * len(sensors), len(states)), O.shape
    assert np.isfinite(np.asarray(O, dtype=float)).all()


def test_window_index_is_clamped_into_range(engine):
    """ The window sliders can outrun a shortened trajectory (switch to a
    coarser dt, or draw a shorter path, and the index stays where it was), so
    the engine must clamp rather than index past the end. """
    w = 4
    for k in (-5, 0, engine.N * 10):
        O, kk = engine.empirical_window_O(w, 1e-4, k)
        assert 0 <= kk <= engine.N - w, (k, kk)
        assert np.isfinite(np.asarray(O, dtype=float)).all()
        F, kf = engine.empirical_window_fisher_inv(w, 1e-4, r_of(engine.spec),
                                                   1e-6, k)
        assert 0 <= kf <= engine.N - w, (k, kf)
        assert np.isfinite(np.asarray(F, dtype=float)).all()


@pytest.mark.parametrize('basis', ['observability', 'constructability'])
def test_stochastic_gramian_runs_for_both_bases(engine, basis):
    spec = engine.spec
    ev = engine.linearized_ev(4, {s: 1e-4 for s in spec.state_names},
                              r_of(spec), 1e-6, basis=basis)
    assert 'time_initial' in ev.columns
    vals = ev[list(spec.state_names)].to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    assert finite.size and (finite > 0).all()


def test_both_bases_share_the_time_axis(engine):
    """ The two curves are plotted against one x axis; different window centers
    would misalign them silently. """
    spec = engine.spec
    qd = {s: 1e-4 for s in spec.state_names}
    a = engine.linearized_ev(4, qd, r_of(spec), 1e-6, basis='observability')
    b = engine.linearized_ev(4, qd, r_of(spec), 1e-6, basis='constructability')
    assert np.allclose(a['time_initial'].to_numpy(dtype=float),
                       b['time_initial'].to_numpy(dtype=float))


def test_caches_are_hit_on_a_repeat_call(engine):
    """ Every control nudge recomputes; without the caches the app is unusable
    on anything but a toy trajectory. """
    spec = engine.spec
    rd = r_of(spec)
    engine._ev_emp_cache.clear()
    first, cached_first = engine.empirical_ev(4, 1e-4, rd, 1e-6)
    n_keys = len(engine._ev_emp_cache)
    second, cached_second = engine.empirical_ev(4, 1e-4, rd, 1e-6)
    # the second return value IS the cache-hit flag, so this is exact
    assert cached_first is False and cached_second is True
    assert len(engine._ev_emp_cache) == n_keys, 'a repeat call added a cache key'
    assert np.allclose(first[list(spec.state_names)].to_numpy(dtype=float),
                       second[list(spec.state_names)].to_numpy(dtype=float),
                       equal_nan=True)


def test_a_different_selection_is_not_served_from_cache(engines):
    """ The sensor/state selection is part of the problem; serving a cached
    result across a change would show the previous selection's numbers. Needs a
    multi-sensor system — on alt2d "all channels" and "the first channel" are
    the same request. """
    eng = engines['drone']
    spec = eng.spec
    rd = r_of(spec)
    eng._ev_emp_cache.clear()
    eng.empirical_ev(4, 1e-4, rd, 1e-6, sensors=list(spec.measurement_names))
    for change in (dict(sensors=[spec.measurement_names[0]]),
                   dict(states=[spec.state_names[0]]),
                   dict(w=5), dict(lam=1e-5), dict(eps=1e-3),
                   dict(r=r_of(spec, 0.2))):
        _, cached = eng.empirical_ev(change.get('w', 4),
                                     change.get('eps', 1e-4),
                                     change.get('r', rd),
                                     change.get('lam', 1e-6),
                                     sensors=change.get('sensors'),
                                     states=change.get('states'))
        assert cached is False, f'{change} was served from cache'
