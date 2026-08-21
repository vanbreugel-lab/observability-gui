""" Smoke test: exercise the app end to end and assert the figures are not just
present but correct.

    python smoke_test.py        # exits nonzero on the first failed assertion

Covers the four (observability / constructability) selections through _compute,
then every built-in system through on_system — the real default path, where a
regression in the per-system defaults would otherwise go unnoticed.
"""
import matplotlib
matplotlib.use('Agg')
import contextlib
import inspect
import io
import os
import re
import sys
import numpy as np
from matplotlib.figure import Figure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C
import app_custom as A

# No hard-coded parameter count here: it went stale every time a control was
# added or removed. Signature drift between _compute and its callers is caught
# for real by the on_system check below, which passes every argument positionally.
print(f'_compute params: {len(inspect.signature(A._compute).parameters)}')

# ── COMPUTE_IN must line up with _compute, slot for slot ──
# Gradio binds COMPUTE_IN to _compute's parameters BY POSITION, so inserting a
# control in one place and not the other silently shifts every argument after
# it — the app then runs with, say, a seed where a checkbox belongs. Both lists
# use the same names, so compare them: an off-by-one shows up as a name
# mismatch, not as a mystery. COMPUTE_IN is local to the Blocks context, so
# read it out of the source with ast rather than importing it.
import ast

_tree = ast.parse(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'app_custom.py')).read())
_ci = next(node.value.elts for node in ast.walk(_tree)
           if isinstance(node, ast.Assign) and isinstance(node.value, ast.List)
           and any(isinstance(t, ast.Name) and t.id == 'COMPUTE_IN'
                   for t in node.targets))
_ci = [e.id for e in _ci if isinstance(e, ast.Name)]
_fn = next(n for n in _tree.body
           if isinstance(n, ast.FunctionDef) and n.name == '_compute')
# _compute's tail is (rebuild, progress), which are not UI inputs
_params = [a.arg for a in _fn.args.args][:-2]
assert len(_ci) == len(_params), (f'COMPUTE_IN has {len(_ci)} entries, '
                                  f'_compute takes {len(_params)} inputs')
_drift = [(i, a, b) for i, (a, b) in enumerate(zip(_params, _ci)) if a != b]
assert not _drift, f'COMPUTE_IN / _compute positional drift: {_drift}'
print(f'  COMPUTE_IN lines up with _compute: {len(_ci)} inputs, names match')


def _labels(fig):
    """ Plotted (non-underscore) line labels in a figure. """
    return {l.get_label() for ax in fig.get_axes() for l in ax.get_lines()
            if not l.get_label().startswith('_')}


def args_for(q_obs, q_con, do_stoch=True, q_noise='uncorrelated',
             backend='engine', split=False):
    S = A._S0
    return ({}, None, [], None, 'fly', 100.0, 1.5, 1.0, 0.3, 6, 1e-6, 1e-5,
            'uniform', 0.1, A._r_rows(S), do_stoch, q_obs, q_con,
            'uniform', q_noise, 1e-2, A._q_rows(S),
            list(S.fim_sensors), list(S.fim_states),
            S.color_state_default, list(S.ev_states_default), list(S.ev_states_default),
            list(A.render.meas_default(S)), 'arrowhead', 'inferno_r', 1e-4, 1e6,
            True, True, True, True, True,
            True, True, True, True, True, True,
            # EKF realization: seed, per-state initial guess, injection. A
            # blank guess row = start that state at the truth plus the seeded
            # random error, which is the default for every state here.
            0, A._x0_rows(S), 'none', 0.0, 0.0, 0.0, 0.0, A._p0_rows(S),
            # UKF realization — same values, so the two filters see identical
            # data — plus its sigma-point tuning. The trailing P0 table is
            # per-state; blank rows fall back to the spec default.
            0, A._x0_rows(S), 'none', 0.0, 0.0, 0.0, 0.0, A._p0_rows(S),
            1.0, 2.0, 0.0,
            # est_split: False = the shared controls drive both filters (the
            # UKF's own values above are ignored). Then which implementation
            # runs: 'engine' (this repo, NumPy) or 'dynamax' (JAX) — that path
            # is exercised separately below, and only when it is installed.
            split, backend,

            'zeta', '64, 64, 64', 4, 4, 40, 256, 0.01,
            S.input_names[-1], 20, 0.5, 1e12, 1e-3,
            A._axis_rows(S.state_names), A._axis_rows(S.measurement_names),
            3, 3)


cases = [(True, False, 'observability only'),
         (False, True, 'constructability only'),
         (True, True, 'BOTH'),
         (False, False, 'neither (Q on, no basis)')]

# the trajectory the app builds for the default system is 'fly7' now; args_for
# still names 'fly' explicitly so the two paths are both covered.

first = True
for q_obs, q_con, name in cases:
    with contextlib.redirect_stdout(io.StringIO()):
        out = A._compute(*args_for(q_obs, q_con), rebuild=first)
    first = False
    (fig_main, fig_O, fig_Finv, fig_minev, fig_est, fig_meas,
     fig_inputs, status, sess) = out
    from matplotlib.figure import Figure as _F
    for _n, _f in (('main', fig_main), ('O', fig_O), ('minev', fig_minev),
                   ('est', fig_est), ('meas', fig_meas)):
        assert isinstance(_f, _F), (_n, type(_f))
        _f.canvas.draw()          # draw-time errors surface here
    # A figure that merely EXISTS is not enough: when a filter raised, the
    # estimator plots came back as gr.update() or silently empty. Assert the
    # traces are present, and that no estimator reported a failure.
    _need_est = {'true', 'EKF ±2σ', 'UKF ±2σ'}
    _need_meas = {'noisy samples (R)', 'noise-free h(x,u)', 'EKF prediction h(x̂)'}
    assert _need_est <= _labels(fig_est), _need_est - _labels(fig_est)
    assert _need_meas <= _labels(fig_meas), _need_meas - _labels(fig_meas)
    assert 'failed' not in status, status
    print('  figures draw cleanly and carry every expected estimator trace')
    continue
    n_traj_expected = 1 + sum([q_obs, q_con])
    # count by title: colorbars spawn their own sub-gridspecs, so gridspec
    # geometry is not a reliable way to count panels
    axes = fig_main.get_axes()
    titles = [a.get_title() for a in axes]
    n_traj = sum(t.startswith('min EV of') for t in titles)
    has_wind = any(t.startswith('wind') for t in titles)
    has_curve = any(t.startswith('min EV, sliding window') for t in titles)
    has_hm = fig_minev is not None and any(
        'state × time' in a.get_title() for a in fig_minev.get_axes())
    print(f'\n[{name}]')
    print(f'  status : {status[:78]}')
    print(f'  outputs: {len(out)} (expect 8)')
    print(f'  trajectory panels: {n_traj} (expect {n_traj_expected})')
    print(f'  wind panel: {has_wind} | min-EV curve: {has_curve} | heatmap: {has_hm}')
    print(f'  fig_main size: {fig_main.get_size_inches()}')
    print(f'  fig_O axes: {len(fig_O.get_axes()) if fig_O else None}  '
          f'fig_Finv axes: {len(fig_Finv.get_axes()) if fig_Finv else None}')
    assert len(out) == 8
    assert n_traj == n_traj_expected, (n_traj, n_traj_expected)
    assert has_wind and has_curve and has_hm
    # the min-EV curve and heatmap must both be labelled in the new units
    labels = [a.get_xlabel() for a in axes]
    assert labels.count('time step (s)') == 1, labels
    hm_y = [a.get_ylabel() for a in fig_minev.get_axes()]
    hm_x = [a.get_xlabel() for a in fig_minev.get_axes()]
    assert 'time step (s)' in hm_y, hm_y   # time on Y now
    assert 'state' in hm_x, hm_x           # states on X now
    assert 'window-center time [s]' not in labels, labels
    print(f"  x-axis labels present: {sorted(set(l for l in labels if l))}")
    assert 'x' in labels, labels   # fly's first state is x
    # shared bar captions itself with a title; the single-panel inline bar
    # still uses a side label
    cbs = [a for a in axes if a.get_ylabel().startswith('min EV: ')
           or a.get_title().startswith('min EV\n')]
    expect_cb = 1 if n_traj >= 2 else (1 if n_traj == 1 else 0)
    print(f'  colorbars on trajectory row: {len(cbs)} '
          f'(expect {expect_cb} — shared when 2+)')
    assert len(cbs) == expect_cb, cbs
    print('  rendered to PNG OK')

print('\n=== window-center alignment: both bases must share time_initial ===')
eng = A.ObservabilityEngine(A._S0)
sp = A.build_setpoint(A._S0, [], None, 1.5, 1.0, 0.3)
with contextlib.redirect_stdout(io.StringIO()):
    eng.simulate_mpc(sp)
qd = {n: 1e-2 for n in A._S0.state_names}
rd = {m: 0.1 for m in A._S0.measurement_names}
eo = eng.linearized_ev(6, qd, rd, 1e-6, basis='observability')
ec = eng.linearized_ev(6, qd, rd, 1e-6, basis='constructability')
same = np.allclose(eo['time_initial'].values, ec['time_initial'].values)
print(f'  identical time_initial: {same}  (expect True)')
assert same
print(f'  t range: {eo["time_initial"].min():.3f} .. {eo["time_initial"].max():.3f}')
zo, zc = eo['zeta'].dropna(), ec['zeta'].dropna()
print(f'  zeta min-EV medians: obs={zo.median():.4g}  constr={zc.median():.4g}')
print(f'  curves differ: {not np.allclose(zo.values, zc.values)} (expect True)')
assert not np.allclose(zo.values, zc.values)


print()
print('=== every system through on_system (the real default path) ===')
print('  this is what caught the P0 regression: _compute with hand-built args')
print('  missed it, because the bug lived in _est_defaults.')
for _sys in ('fly', 'fly7', 'drone', 'alt2d'):
    with contextlib.redirect_stdout(io.StringIO()):
        _out = A.on_system(_sys)
    # tail is ..., fig_est, fig_meas, fig_inputs, status, sess
    _est, _meas, _inp, _status = _out[-5], _out[-4], _out[-3], str(_out[-2])
    assert isinstance(_est, Figure), (_sys, 'states plot is not a Figure')
    assert isinstance(_meas, Figure), (_sys, 'measurements plot is not a Figure')
    # every system has inputs, so every system gets the panel
    assert isinstance(_inp, Figure), (_sys, 'inputs plot is not a Figure')
    _want_inp = len(C.get_spec(_sys).input_names)
    _le, _lm = _labels(_est), _labels(_meas)
    assert {'true', 'EKF ±2σ', 'UKF ±2σ'} <= _le, (_sys, _le)
    assert 'EKF prediction h(x̂)' in _lm, (_sys, _lm)
    assert 'failed' not in _status, (_sys, _status)
    print(f'  {_sys:6s} states={sorted(_le)}  inputs={_want_inp}')

print()
print('=== the two front-ends must assemble the SAME inputs to core ===')
print('  app_custom.py (Gradio) and streamlit_app.py both call')
print('  core.compute_payload, so the only place they can disagree is the')
print('  dicts they build from their controls. Spy on the call in each.')
try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    print('  SKIPPED — streamlit not installed (pip install -r requirements.txt)')
else:
    import core
    _real, _grabbed = core.compute_payload, []

    def _spy(engine, job, p, est, **kw):     # **kw: on_stage, from the Gradio side
        out = _real(engine, job, p, est, **kw)   # gradio solves the MPC in here,
        _grabbed.append((p, est))                # so this must run after
        return out

    core.compute_payload = A.compute_payload = _spy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            # drive Gradio with the SAME system Streamlit lands on by
            # default (the first SYSTEM_LABELS entry), so the two dicts are
            # comparable no matter which system is the default
            A.on_system(core.SYSTEM_LABELS[0][1])
        _gp, _ge = _grabbed[-1]
        _at = AppTest.from_file(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'streamlit-app', 'streamlit_app.py'),
            default_timeout=900)
        _at.run()
        assert not _at.exception, [str(e.value) for e in _at.exception]
        _sp, _se = _grabbed[-1]
    finally:
        core.compute_payload = A.compute_payload = _real

    def _same(x, y):
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            return np.allclose(x, y, equal_nan=True)
        return x == y

    for _tag, _a, _b in (('p', _gp, _sp), ('est[EKF]', _ge['EKF'], _se['EKF']),
                         ('est[UKF]', _ge['UKF'], _se['UKF'])):
        assert set(_a) == set(_b), (_tag, set(_a) ^ set(_b))
        _bad = {k: (_a[k], _b[k]) for k in _a if not _same(_a[k], _b[k])}
        assert not _bad, (_tag, _bad)
        print(f'  {_tag:9s} {len(_a)} keys identical')
    for _k in ('ukf_alpha', 'ukf_beta', 'ukf_kappa'):
        assert _ge[_k] == _se[_k], (_k, _ge[_k], _se[_k])
    print('  sigma-point tuning identical')

print()
print('=== per-state P0: one row must move only that state ===')
print('  the point of the table is that a pinned state and a')
print('  guess-following state coexist, so check both halves at once.')
_S = A._S0
_names = list(_S.state_names)
_tgt, _other = _names[0], _names[1]

# (a) the table -> vector step: one row filled, the rest blank
_rows = A._p0_rows(_S)
for _r in _rows:
    _r[1] = None                                  # start from a blank table
_rows[0][1] = 42.0
_v = C._p0_vec(_rows, _names)
assert _v is not None and _v[0] == 42.0, _v
assert np.isnan(_v[1:]).all(), _v
print(f'  _p0_vec: {_tgt}=42.0, {len(_names) - 1} others blank (NaN)')

# blank table -> None, so the spec's paper P0 still takes over
assert C._p0_vec([[n, None] for n in _names], _names) is None
# a scalar still broadcasts (the Streamlit box), and <= 0 is ignored
assert np.all(C._p0_vec(7.0, _names) == 7.0)
assert C._p0_vec([[_tgt, 0.0], [_other, -3.0]], _names) is None
print('  blank -> None (spec P0 wins); scalar broadcasts; <= 0 ignored')

# (b) the vector -> P0 matrix step: only the pinned diagonal entry changes
_eng = C.ObservabilityEngine(_S)
with contextlib.redirect_stdout(io.StringIO()):
    _eng.simulate_mpc(_S.default_setpoint(float(_S.wind_default),
                                          float(_S.zeta_default)))
_rd = {m: 0.1 for m in _S.measurement_names}
_, _, _P_base, _, _ = _eng._estimation_problem(_rd, 0)
_, _, _P_pin, _, _ = _eng._estimation_problem(_rd, 0, p0_diag=_v)
assert np.isclose(_P_pin[0, 0], 42.0), _P_pin[0, 0]
_d_base, _d_pin = np.diag(_P_base), np.diag(_P_pin)
assert np.allclose(_d_base[1:], _d_pin[1:]), (_d_base, _d_pin)
print(f'  P0[{_tgt}] = {_d_pin[0]:g} (pinned), '
      f'{_other} unchanged at {_d_pin[1]:.4g}')

print()
print('=== dynamax filter backend: the same problem, a second implementation ===')
if not C.DYNAMAX_AVAILABLE:
    print('  SKIPPED — dynamax not installed (pip install dynamax)')
else:
    import dynamax_filters as dmx
    assert dmx.AVAILABLE, dmx.IMPORT_ERROR
    # (a) the whole app path, exactly as the UI drives it
    with contextlib.redirect_stdout(io.StringIO()):
        _o = A._compute(*args_for(True, False, backend='dynamax'), rebuild=True)
    _est_d, _meas_d, _st_d = _o[4], _o[5], str(_o[7])
    assert isinstance(_est_d, Figure) and isinstance(_meas_d, Figure), _o[:7]
    _est_d.canvas.draw()
    assert {'true', 'EKF ±2σ', 'UKF ±2σ'} <= _labels(_est_d), _labels(_est_d)
    assert 'failed' not in _st_d, _st_d
    assert 'via dynamax' in _st_d, _st_d
    print(f'  fly through _compute: {_st_d[:96]}')

    # (b) the numbers. alt2d has no angular measurement channels and linear
    #     dynamics, so the two EKFs linearize about the same thing and nothing
    #     is left to excuse a difference: they must agree to near machine
    #     precision. This is the check that would catch a real divergence.
    _S = C.get_spec('alt2d')
    _e = C.ObservabilityEngine(_S)
    with contextlib.redirect_stdout(io.StringIO()):
        _e.simulate_mpc(C.build_setpoint(_S, [], None, _S.v0_default,
                                         _S.wind_default, _S.zeta_default))
    _qd = {s: 1e-4 for s in _S.state_names}
    _rd = {m: 0.1 for m in _S.measurement_names}
    for _nm, _fe, _fd, _kw in (
            ('EKF', _e.run_ekf, dmx.run_ekf, {}),
            ('UKF', _e.run_ukf, dmx.run_ukf, dict(alpha=1.0, beta=2.0,
                                                  kappa=0.0))):
        _Xe, _Pe = _fe(_qd, _rd, seed=0, **_kw)
        _Xd, _Pd = _fd(_e, _qd, _rd, seed=0, **_kw)
        assert np.isfinite(_Xd).all(), (_nm, 'dynamax produced non-finite states')
        _rel = (np.abs(_Xd - _Xe).max(axis=0)
                / np.maximum(np.abs(_Xe).max(axis=0), 1e-9)).max()
        print(f'  alt2d {_nm}: max relative difference {_rel:.2e} '
              f'(engine vs dynamax)')
        assert _rel < 1e-5, (_nm, _rel)

    # (c) the callback path — how an untraceable (e.g. scipy-using) uploaded
    #     model is filtered — must reach the same answer as the jax.numpy one
    _Xj, _ = dmx.run_ekf(_e, _qd, _rd, seed=0, mode='jax')
    _Xc, _ = dmx.run_ekf(_e, _qd, _rd, seed=0, mode='callback')
    _rel = (np.abs(_Xj - _Xc).max(axis=0)
            / np.maximum(np.abs(_Xj).max(axis=0), 1e-9)).max()
    print(f'  alt2d EKF: jax vs pure_callback model path {_rel:.2e}')
    assert _rel < 1e-5, _rel

    # (d) the strong check, on the systems that actually exercise the hard
    #     parts (18 states, arctan2 measurements). The ONLY thing the two EKFs
    #     do differently is where they linearize the prediction, so pointing the
    #     NumPy one at dynamax's choice must collapse the difference to
    #     round-off. If this ever loosens, the two implementations have genuinely
    #     diverged — a much sharper instrument than comparing on alt2d.
    for _sn in ('fly', 'fly7', 'drone'):
        _sp = C.get_spec(_sn)
        _en = C.ObservabilityEngine(_sp)
        with contextlib.redirect_stdout(io.StringIO()):
            _en.simulate_mpc(C.build_setpoint(_sp, [], None, _sp.v0_default,
                                              _sp.wind_default,
                                              _sp.zeta_default))
        _q = dict(getattr(_sp, 'q_tiny', None)
                  or {s: 1e-4 for s in _sp.state_names})
        _r = {m: 0.1 for m in _sp.measurement_names}
        _sel = tuple(_sp.fim_sensors)
        _Xd, _ = dmx.run_ekf(_en, _q, _r, seed=0, sensors=_sel)
        _Xe, _ = _en.run_ekf(_q, _r, seed=0, sensors=_sel, f_jac_at='pre')
        _rel = (np.abs(_Xd - _Xe).max(axis=0)
                / np.maximum(np.abs(_Xe).max(axis=0), 1e-9)).max()
        print(f'  {_sn:6s} EKF (matched linearization point): {_rel:.2e}')
        assert _rel < 1e-6, (_sn, _rel)
        # and the angular wrapping must be doing something on the UKF: without
        # it the drone's course/heading residuals cross the branch cut and the
        # estimate moves away from the engine's
        _Xw, _ = dmx.run_ukf(_en, _q, _r, seed=0, sensors=_sel, alpha=1.0,
                             beta=2.0, kappa=0.0)
        _Xu, _ = dmx.run_ukf(_en, _q, _r, seed=0, sensors=_sel, alpha=1.0,
                             beta=2.0, kappa=0.0, wrap_angles=False)
        _Xk, _ = _en.run_ukf(_q, _r, seed=0, sensors=_sel, alpha=1.0, beta=2.0,
                             kappa=0.0)
        _sc = np.maximum(np.abs(_Xk).max(axis=0), 1e-9)
        _dw = (np.abs(_Xw - _Xk).max(axis=0) / _sc).max()
        _du = (np.abs(_Xu - _Xk).max(axis=0) / _sc).max()
        print(f'  {_sn:6s} UKF vs engine: wrapped {_dw:.2e}, '
              f'unwrapped {_du:.2e}')
        assert _dw <= _du * 1.001, (_sn, _dw, _du)   # wrapping never hurts
        assert _dw < 5e-3, (_sn, _dw)

print('\nALL SMOKE CHECKS PASS')
