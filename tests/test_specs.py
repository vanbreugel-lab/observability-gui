"""
Every built-in system's spec must be internally consistent.

A spec is a pile of names and defaults that the UI reads to build dropdowns,
tables and plots. Nothing checks it at import time, so a single typo — a sensor
listed in `fim_sensors` that `h` doesn't produce, a `color_state_default` that
isn't a state — surfaces much later as a Gradio "value not in choices" error or
an IndexError deep in a Gramian. These tests are that missing check, and they
also guard the contract an *uploaded* custom system has to satisfy.
"""
import numpy as np
import pytest

import core
import render
from conftest import BUILTINS


@pytest.fixture(params=BUILTINS)
def spec(request):
    return core.get_spec(request.param)


def test_names_are_unique_non_empty_strings(spec):
    for attr in ('state_names', 'input_names', 'measurement_names'):
        names = list(getattr(spec, attr))
        assert names, f'{attr} is empty'
        assert all(isinstance(n, str) and n.strip() for n in names), names
        assert len(set(names)) == len(names), f'duplicate names in {attr}: {names}'


def test_f_and_h_return_the_declared_dimensions(spec):
    """ The single most consequential contract in the repo: every Jacobian,
    Gramian and filter sizes itself from these name lists. """
    n, m, p = (len(spec.state_names), len(spec.input_names),
               len(spec.measurement_names))
    x = np.full(n, 0.5)                     # 0.5, not 0: h often divides by z
    u = np.full(m, 0.1)
    fx = np.asarray(spec.f(x, u), dtype=float).ravel()
    hx = np.asarray(spec.h(x, u), dtype=float).ravel()
    assert fx.shape == (n,), f'f returned {fx.shape}, expected {(n,)}'
    assert hx.shape == (p,), f'h returned {hx.shape}, expected {(p,)}'
    assert np.isfinite(fx).all() and np.isfinite(hx).all()


def test_f_and_h_are_pure(spec):
    """ Called twice with the same input they must give the same answer, and
    must not modify the caller's array — the empirical Gramian perturbs one
    shared state vector thousands of times. """
    n, m = len(spec.state_names), len(spec.input_names)
    x, u = np.full(n, 0.5), np.full(m, 0.1)
    x_keep, u_keep = x.copy(), u.copy()
    f1 = np.asarray(spec.f(x, u), dtype=float)
    h1 = np.asarray(spec.h(x, u), dtype=float)
    assert np.array_equal(x, x_keep) and np.array_equal(u, u_keep)
    assert np.allclose(f1, np.asarray(spec.f(x, u), dtype=float))
    assert np.allclose(h1, np.asarray(spec.h(x, u), dtype=float))


def test_declared_subsets_actually_exist(spec):
    sn, mn, inn = (list(spec.state_names), list(spec.measurement_names),
                   list(spec.input_names))
    assert set(spec.angle_states) <= set(sn), spec.angle_states
    assert set(spec.angle_measurements) <= set(mn), spec.angle_measurements
    assert set(spec.fim_sensors) <= set(mn), spec.fim_sensors
    assert set(spec.fim_states) <= set(sn), spec.fim_states
    assert set(spec.ev_states_default) <= set(sn), spec.ev_states_default
    assert spec.color_state_default in sn
    assert set(render.meas_default(spec)) <= set(mn)
    assert set(render.inputs_default(spec)) <= set(inn)
    if getattr(spec, 'r_default', None):
        assert set(spec.r_default) == set(mn)
    for attr in ('q_tiny', 'q_realistic', 'q_paper'):
        d = getattr(spec, attr, None)
        if d:
            assert set(d) <= set(sn), (attr, set(d) - set(sn))
    rp = getattr(spec, 'r_paper', None)
    if rp:
        assert set(rp) <= set(mn)


def test_selection_defaults_are_non_empty(spec):
    """ An empty default selection lands the user on a blank plot with no hint
    that they have to pick something. """
    assert len(spec.fim_sensors) >= 1
    assert len(spec.fim_states) >= 1
    assert len(spec.ev_states_default) >= 1
    assert len(render.meas_default(spec)) >= 1


def test_numeric_defaults_are_sane(spec):
    assert spec.dt > 0
    assert int(spec.w_default) >= 1
    assert spec.eps_default > 0
    assert spec.seg_duration > 0
    assert spec.rec_xlim[0] < spec.rec_xlim[1]
    assert spec.rec_ylim[0] < spec.rec_ylim[1]
    assert np.isfinite([spec.v0_default, spec.wind_default,
                        spec.zeta_default]).all()


def test_u_bias_default_names_a_real_channel(spec):
    ub = getattr(spec, 'u_bias_default', None)
    if ub:
        ch, mag, t0, t1 = ub
        assert ch in spec.input_names
        assert t1 > t0 and mag != 0


@pytest.mark.parametrize('system', BUILTINS)
def test_dt_override_is_honored(system):
    """ The sampling-rate control rebuilds the spec at a new dt. Everything
    else must come along unchanged, or changing Hz would quietly change the
    model. """
    base = core.get_spec(system)
    fine = core.get_spec(system, dt=base.dt / 2)
    assert np.isclose(fine.dt, base.dt / 2)
    assert list(fine.state_names) == list(base.state_names)
    assert list(fine.measurement_names) == list(base.measurement_names)
    assert list(fine.input_names) == list(base.input_names)


@pytest.mark.parametrize('system', BUILTINS)
def test_base_hz_matches_dt(system):
    """ The Hz dropdown's per-system default has to be a value the dropdown
    actually offers, or Gradio rejects it on system switch. """
    assert core._BASE_HZ[system] == pytest.approx(round(1.0 / core.get_spec(system).dt))
    assert core._BASE_HZ[system] in [v for _label, v in core.DT_HZ]


def test_system_labels_resolve():
    """ Every entry in the dropdown must either build a spec or be the special
    'custom' placeholder. """
    values = [v for _label, v in core.SYSTEM_LABELS]
    assert 'custom' in values
    assert len(set(values)) == len(values)
    for v in values:
        if v != 'custom':
            assert core.get_spec(v) is not None


def test_default_setpoint_builds_a_consistent_dict(spec):
    """ default_setpoint feeds the MPC's time-varying parameters; a missing
    state or a ragged array is a do-mpc error with no useful message. """
    sp = spec.default_setpoint(float(spec.wind_default), float(spec.zeta_default))
    assert set(sp) == set(spec.state_names), set(sp) ^ set(spec.state_names)
    lengths = {len(np.atleast_1d(v)) for v in sp.values()}
    assert len(lengths) == 1, f'ragged set-point arrays: {lengths}'
    assert all(np.isfinite(np.asarray(v, dtype=float)).all()
               for v in sp.values())


def test_lam_and_eps_choice_lists_are_plain_floats():
    """ Gradio cannot match a numpy scalar against its own choice list, which
    used to raise "not in the list of choices" on every system switch. """
    for vals in (core.LAM_VALS, core.EPS_VALS):
        assert vals and all(type(v) is float for v in vals)
        assert all(v > 0 for v in vals)
