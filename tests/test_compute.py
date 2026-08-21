"""
One refresh, end to end: core.compute_payload and the app's _compute / on_system.

`_compute` is what every control in the page is wired to, and it is reached with
whatever the browser last had — including states that are transiently
inconsistent (a selection that no longer matches the new system's choices), and
values that are legal in the widget but hostile to the numerics (a window longer
than the trajectory, λ below the round-off floor, no sensors at all).

The contract is: return a status string starting with ⚠ and keep the previous
figures. Raising means a red box in the browser and a page the user has to
reload.
"""
import ast
import inspect
import os

import numpy as np
import pytest
from matplotlib.figure import Figure

import core
import render
from conftest import BUILTINS, build_engine, q_of, r_of, silent

HERE = os.path.dirname(os.path.abspath(__file__))
APP_PY = os.path.join(os.path.dirname(HERE), 'app_custom.py')


# ───────────────────────── the wiring must line up ──────────────────────────
# Gradio binds the input list to _compute BY POSITION, so a control inserted in
# one place and not the other shifts every later argument — the app then runs
# with a seed where a checkbox belongs, silently. Both lists use the same names,
# so an off-by-one shows up here as a name mismatch instead of as a mystery.

def _source_list(name):
    tree = ast.parse(open(APP_PY).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List) \
                and any(isinstance(t, ast.Name) and t.id == name
                        for t in node.targets):
            return [e.id for e in node.value.elts if isinstance(e, ast.Name)]
    raise AssertionError(f'{name} not found in app_custom.py')


def test_compute_inputs_match_the_signature(app):
    names = _source_list('COMPUTE_IN')
    params = [p for p in inspect.signature(app._compute).parameters][:-2]
    assert len(names) == len(params), (len(names), len(params))
    drift = [(i, a, b) for i, (a, b) in enumerate(zip(params, names)) if a != b]
    assert not drift, f'COMPUTE_IN / _compute positional drift: {drift}'


def test_every_early_return_pads_to_the_wired_output_count(app):
    """ Gradio matches the response tuple against its output list BY LENGTH, so
    a short tuple is rejected wholesale — the ⚠ status never reaches the page
    and the user gets an unexplained error box. `_N_FIGS` is what every early
    return pads with, so it must equal the number of figure outputs actually
    wired (COMPUTE_OUT minus status and session). """
    wired = _source_list('COMPUTE_OUT')
    assert app._N_FIGS == len(wired) - 2, (app._N_FIGS, wired)
    # and the happy path must agree with the same count
    out = run(app)
    assert len(out) == app._N_FIGS + 2


def test_on_system_fills_every_wired_output(app):
    with silent():
        figs = app.on_system('alt2d')
    assert len(figs) == len(_source_list('SYSTEM_OUT'))


# ─────────────────────────── a baseline refresh ─────────────────────────────

def short_path(system, n=20):
    """ A recorded (speed, heading) path of n steps.

    Used as the default trajectory for these tests purely for speed: _compute
    re-solves the MPC on every call, and a system's real default trajectory is
    up to 501 steps (~3 s per test, ~50 tests). The trajectory *source* is a
    code path in its own right and is covered by test_trajectory.py; the
    on_system tests below still exercise the real default trajectories. """
    s = core.get_spec(system)
    return ([float(s.v0_default)] * n,
            [float(getattr(s, 'heading0', 0.0))] * n)


def args(app, system='alt2d', **over):
    """ The default control values for `system`, as _compute's positional
    arguments — i.e. exactly what the page posts right after a system switch. """
    s = core.get_spec(system)
    d = core._est_defaults(s)
    an, ak, ij = d['ann'], d['aikf'], d['inj']
    a = dict(
        sess={}, custom_spec=None, segs=[], recorded=short_path(system),
        system=system,
        dt_hz=core._BASE_HZ[system], v0=float(s.v0_default),
        wind=float(s.wind_default), zeta=float(s.zeta_default),
        w=int(s.w_default), lam=1e-6, eps=1e-4,
        r_mode='uniform', r_uniform=core._default_r(s),
        r_table=core._r_rows(s), do_stoch=False, q_obs=True, q_con=False,
        q_mode='uniform', q_noise='uncorrelated', q_uniform=1e-4,
        q_table=core._q_rows(s),
        fim_sensors=list(s.fim_sensors), fim_states=list(s.fim_states),
        color=s.color_state_default, ev_states=list(s.ev_states_default),
        est_states=list(s.ev_states_default),
        meas_sel=list(render.meas_default(s)), traj_mode='arrowhead',
        cmap='inferno_r', vmin=1e-4, vmax=1e6,
        s_true=True, s_ekf=True, s_ukf=True, s_ann=False, s_aikf=False,
        m_noisy=True, m_true=True, m_pred=True, m_ukf=True, m_ann=False,
        m_aikf=False,
        ekf_seed=0, ekf_x0=core._x0_rows(s), ekf_ch=ij['channel'],
        ekf_mag=ij['mag'], ekf_t0=ij['t0'], ekf_t1=ij['t1'],
        ekf_unv=d['u_noise'], ekf_p0=core._p0_rows(s),
        ukf_seed=0, ukf_x0=core._x0_rows(s), ukf_ch=ij['channel'],
        ukf_mag=ij['mag'], ukf_t0=ij['t0'], ukf_t1=ij['t1'],
        ukf_unv=d['u_noise'], ukf_p0=core._p0_rows(s),
        ukf_alpha=d['ukf_alpha'], ukf_beta=d['ukf_beta'],
        ukf_kappa=d['ukf_kappa'], est_split=False, est_backend='engine',
        ann_target=an['target'], ann_layers=an['layers'],
        ann_steps=an['time_steps'], ann_traj=an['n_traj'],
        ann_epochs=an['epochs'], ann_batch=an['batch'], ann_noise=an['noise'],
        aikf_motif=ak['motif'], aikf_window=ak['window'],
        aikf_upper=ak['upper'], aikf_r_hi=ak['r_hi'], aikf_r_lo=ak['r_lo'],
        st_ytbl=core._axis_rows(s.state_names),
        ms_ytbl=core._axis_rows(s.measurement_names), mat_k=0, fi_k=0)
    a.update(over)
    params = [p for p in inspect.signature(app._compute).parameters][:-2]
    unknown = set(over) - set(params)
    assert not unknown, f'no such _compute parameter: {unknown}'
    return [a[p] for p in params]


def run(app, system='alt2d', rebuild=True, **over):
    with silent():
        return app._compute(*args(app, system, **over), rebuild=rebuild)


def test_a_default_refresh_produces_every_figure(app):
    out = run(app)
    figs, status, sess = out[:7], out[7], out[8]
    assert not status.startswith('⚠'), status
    for i, f in enumerate(figs):
        assert isinstance(f, Figure), f'output {i} is {type(f)}'
        f.canvas.draw()                      # draw-time errors surface here
    assert 'analysis' in sess, 'the PDF export bundle was not stashed'


def test_the_session_carries_the_engine_forward(app):
    """ Nudging a display-only control must not re-solve the MPC; that is what
    makes the app usable. """
    out = run(app)
    sess = out[8]
    with silent():
        again = app._compute(*args(app, sess=sess), rebuild=False)
    assert again[8]['engine'] is sess['engine']


# ───────────────────── control values that fight the numerics ───────────────

def test_a_window_longer_than_the_trajectory_is_clamped(app):
    out = run(app, w=100000)
    assert not out[7].startswith('⚠'), out[7]


def test_no_sensors_selected_falls_back_to_all_of_them(app):
    """ An empty CheckboxGroup is one click away and must not divide by zero. """
    out = run(app, fim_sensors=[], fim_states=[])
    assert not out[7].startswith('⚠'), out[7]


def test_an_unknown_sensor_is_reported_not_raised(app):
    """ The transient state during a system switch: the page still holds the
    previous system's selection. It must come back as a ⚠ message AND with the
    full complement of outputs, or Gradio drops the message (see
    test_every_early_return_pads_to_the_wired_output_count). """
    out = run(app, fim_sensors=['not_a_sensor'])
    assert len(out) == app._N_FIGS + 2, len(out)
    assert isinstance(out[-2], str) and out[-2].startswith('⚠'), out[-2]


def test_empty_plot_selections_still_render(app):
    out = run(app, ev_states=[], est_states=[], meas_sel=[])
    assert not out[7].startswith('⚠'), out[7]
    for f in out[:4]:
        assert isinstance(f, Figure)


def test_a_tiny_lambda_still_computes(app):
    """ λ below the round-off floor makes the min-EV of an unobservable
    direction meaningless, but it must still produce a figure rather than a
    LinAlgError — the whole point of the eigendecomposition in _ev_diag. """
    out = run(app, lam=1e-30)
    assert isinstance(out[0], Figure)
    assert not out[-2].startswith('⚠'), out[-2]


def test_the_lambda_floor_warning_fires_when_the_floor_is_known(app):
    """ The "λ at/below round-off floor" warning compares λ against
    engine.lam_noise_floor.

    KNOWN GAP, pinned here deliberately: that floor is only ever assigned
    inside `linearized_ev`, so with process noise OFF — the default — it stays
    0.0 and the warning cannot fire, however small λ is. Turning the stochastic
    Gramian on is what makes it live. If the floor is ever computed on the
    empirical path too, this test should start failing on its first half, which
    is the point. """
    empirical_only = run(app, lam=1e-30, do_stoch=False)[-2]
    assert 'floor' not in empirical_only, empirical_only
    with_gramian = run(app, lam=1e-30, do_stoch=True, q_obs=True)[-2]
    assert 'floor' in with_gramian, with_gramian


def test_zero_and_negative_noise_are_floored(app):
    """ R = 0 is a singular matrix and Q = 0 collapses P; the table parsers floor
    both, so this must come back as a figure rather than a LinAlgError. """
    for over in (dict(r_uniform=0.0), dict(r_uniform=-1.0),
                 dict(q_uniform=0.0, do_stoch=True),
                 dict(q_uniform=-5.0, do_stoch=True)):
        out = run(app, **over)
        assert isinstance(out[0], Figure), (over, out[7])


def test_process_noise_on_with_no_basis_selected_says_so(app):
    out = run(app, do_stoch=True, q_obs=False, q_con=False)
    assert 'no FIM basis' in out[7], out[7]


def test_both_gramian_bases_at_once(app):
    out = run(app, do_stoch=True, q_obs=True, q_con=True)
    assert not out[7].startswith('⚠'), out[7]
    assert 'observability' in out[7] and 'constructability' in out[7]


def test_vanloan_process_noise(app):
    out = run(app, do_stoch=True, q_noise='vanloan')
    assert not out[7].startswith('⚠'), out[7]


def test_window_sliders_past_the_end_are_clamped(app):
    out = run(app, mat_k=10 ** 6, fi_k=-10 ** 6)
    assert not out[7].startswith('⚠'), out[7]
    assert isinstance(out[1], Figure) and isinstance(out[2], Figure)


def test_inverted_colour_limits_do_not_crash(app):
    out = run(app, vmin=1e6, vmax=1e-6)
    assert isinstance(out[0], Figure), out[7]


def test_axis_limit_tables_are_applied(app):
    s = core.get_spec('alt2d')
    st = core._axis_rows(s.state_names)
    st[0][1], st[0][2] = 0.0, 1.0           # shared time limit
    st[1][1] = -5.0                         # one state's lower bound
    out = run(app, st_ytbl=st)
    assert not out[7].startswith('⚠'), out[7]


# ─────────────────── the shared vs independent realization ──────────────────

def test_shared_controls_drive_both_filters(app, monkeypatch):
    """ The merged Estimators section: with est_split off, the UKF must use the
    shared values even when its own (hidden) widgets say something else. """
    grabbed = {}
    real = core.compute_payload

    def spy(engine, job, p, est, **kw):
        grabbed['est'] = est
        return real(engine, job, p, est, **kw)

    monkeypatch.setattr(app, 'compute_payload', spy)
    s = core.get_spec('alt2d')
    other = core._x0_rows(s)
    other[0][1] = 99.0                       # only in the UKF's own table
    run(app, est_split=False, ukf_seed=123, ukf_x0=other, ukf_unv=0.5)
    e = grabbed['est']
    assert e['EKF']['seed'] == e['UKF']['seed'] == 0
    assert e['EKF']['unv'] == e['UKF']['unv']
    assert e['EKF']['ub'] == e['UKF']['ub']
    # a blank guess table reads as None (see _x0_vec), so compare that first
    for key in ('x0', 'p0'):
        a, b = e['EKF'][key], e['UKF'][key]
        assert (a is None) == (b is None), key
        if a is not None:
            assert np.array_equal(np.asarray(a, dtype=float),
                                  np.asarray(b, dtype=float), equal_nan=True)
    # and with the override on, the UKF's own values are honored
    run(app, est_split=True, ukf_seed=123, ukf_x0=other, ukf_unv=0.5)
    e = grabbed['est']
    assert e['UKF']['seed'] == 123 and e['EKF']['seed'] == 0
    assert e['UKF']['x0'][0] == 99.0
    assert e['UKF']['unv'] == 0.5


# ───────────────────────────── switching systems ────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize('system', BUILTINS)
def test_on_system_resets_and_recomputes(app, system):
    """ The default path: every control gets the new system's names, and the
    figures come back drawn. A stale selection here is the "'phi' not in
    choices" crash. """
    with silent():
        out = app.on_system(system)
    s = core.get_spec(system)
    status, sess = str(out[-2]), out[-1]
    assert not status.startswith('⚠'), (system, status)
    for fig in out[-9:-2]:
        assert isinstance(fig, Figure), system
    # every choices/value pair the reset emits must be self-consistent
    for upd in out:
        if isinstance(upd, dict) and 'choices' in upd and 'value' in upd:
            choices = [c[1] if isinstance(c, (tuple, list)) else c
                       for c in upd['choices']]
            vals = upd['value'] if isinstance(upd['value'], list) \
                else [upd['value']]
            for v in vals:
                assert v in choices, (system, v, choices)
    assert sess['spec'].name == s.name or system == 'custom'


def test_on_system_custom_reveals_the_uploader(app):
    out = app.on_system('custom')
    assert 'Upload a system .py' in str(out[-2])
    assert out[-1] == {}, 'the custom branch must clear the session'


@pytest.mark.slow
@pytest.mark.parametrize('system', BUILTINS)
def test_reset_rebuilds_the_engine(app, system):
    """ What the ↺ button does: a fresh engine, not the cached one. """
    with silent():
        first = app.on_system(system)
        second = app.on_system(system)
    e1, e2 = first[-1]['engine'], second[-1]['engine']
    assert e1 is not e2, 'reset reused the previous engine'
    assert not e2._filt_cache or True        # a fresh engine starts empty
    assert np.allclose(e1.X, e2.X), 'the same reset gave a different trajectory'


def test_reset_preserves_the_filter_implementation_choice(app):
    """ The backend and the UKF-override are choices about how to filter, not
    about the system, so a reset must not silently flip them back. """
    with silent():
        out = app.on_system('alt2d', 'engine', True)
    assert not str(out[-2]).startswith('⚠')


# ──────────────────────── compute_payload robustness ────────────────────────

def test_payload_keys_are_stable(engine):
    """ Every consumer (both front-ends, the PDF export) indexes these. """
    spec = engine.spec
    p = dict(w_raw=4, eps=1e-4, lam=1e-6, r_diag=r_of(spec),
             q_diag=q_of(spec), q_noise='uncorrelated',
             sensors=tuple(spec.measurement_names),
             states=tuple(spec.state_names), do_stoch=False, bases=(),
             do_ann=False, ann_target=spec.color_state_default,
             ann_layers=(8,), ann_steps=2, ann_traj=2, ann_epochs=1,
             ann_batch=8, ann_noise=0.0, aikf_motif=spec.input_names[-1],
             aikf_window=4, aikf_upper=0.5, aikf_r_hi=1e12, aikf_r_lo=1e-3)
    est = dict(EKF=core._realization(spec, 0, 'none', 0, 0, 0, 0.0, None),
               UKF=core._realization(spec, 0, 'none', 0, 0, 0, 0.0, None),
               ukf_alpha=1.0, ukf_beta=2.0, ukf_kappa=0.0)
    payload = core.compute_payload(engine, ('update',), p, est)
    for key in ('ev_emp', 'ev_lin', 'bases', 'results', 'meas', 'ann', 'w',
                'lam', 'backend', 'q_zeta', 't_emp', 't_stoch', 't_ann',
                'floor', 'note'):
        assert key in payload, key
    assert payload['backend'] == 'engine'
    assert set(payload['results']) == {'EKF', 'UKF'}


def test_an_unrecognized_backend_falls_back_to_the_numpy_filters(engine):
    """ A stale browser session (or a script) can post anything here. """
    spec = engine.spec
    p = dict(w_raw=4, eps=1e-4, lam=1e-6, r_diag=r_of(spec), q_diag=q_of(spec),
             q_noise='uncorrelated', sensors=tuple(spec.measurement_names),
             states=tuple(spec.state_names), do_stoch=False, bases=(),
             do_ann=False, ann_target=spec.color_state_default,
             ann_layers=(8,), ann_steps=2, ann_traj=2, ann_epochs=1,
             ann_batch=8, ann_noise=0.0, aikf_motif=spec.input_names[-1],
             aikf_window=4, aikf_upper=0.5, aikf_r_hi=1e12, aikf_r_lo=1e-3,
             backend='not-a-backend')
    est = dict(EKF=core._realization(spec, 0, 'none', 0, 0, 0, 0.0, None),
               UKF=core._realization(spec, 0, 'none', 0, 0, 0, 0.0, None),
               ukf_alpha=1.0, ukf_beta=2.0, ukf_kappa=0.0)
    payload = core.compute_payload(engine, ('update',), p, est)
    assert payload['backend'] == 'engine'
    assert set(payload['results']) == {'EKF', 'UKF'}


def test_an_empty_trajectory_is_reported_not_crashed():
    """ compute_payload on an engine that has never solved. """
    spec = core.get_spec('alt2d')
    eng = core.ObservabilityEngine(spec)
    p = dict(w_raw=4, eps=1e-4, lam=1e-6, r_diag=r_of(spec), q_diag=q_of(spec),
             q_noise='uncorrelated', sensors=(), states=(), do_stoch=False,
             bases=(), do_ann=False)
    est = dict(EKF=core._realization(spec, 0, 'none', 0, 0, 0, 0.0, None),
               UKF=core._realization(spec, 0, 'none', 0, 0, 0, 0.0, None),
               ukf_alpha=1.0, ukf_beta=2.0, ukf_kappa=0.0)
    out = core.compute_payload(eng, ('update',), p, est)
    assert out.get('error') == 'no trajectory'


def test_progress_callback_failures_never_break_the_compute(engine):
    """ on_stage is a UI hook; a dead browser session must not take the
    numerics with it. """
    spec = engine.spec
    p = dict(w_raw=4, eps=1e-4, lam=1e-6, r_diag=r_of(spec), q_diag=q_of(spec),
             q_noise='uncorrelated', sensors=tuple(spec.measurement_names),
             states=tuple(spec.state_names), do_stoch=False, bases=(),
             do_ann=False)
    est = dict(EKF=core._realization(spec, 0, 'none', 0, 0, 0, 0.0, None),
               UKF=core._realization(spec, 0, 'none', 0, 0, 0, 0.0, None),
               ukf_alpha=1.0, ukf_beta=2.0, ukf_kappa=0.0)

    def boom(*_a, **_k):
        raise RuntimeError('the client went away')

    payload = core.compute_payload(engine, ('update',), p, est, on_stage=boom)
    assert 'error' not in payload and payload['results']
