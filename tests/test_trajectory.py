"""
Turning a user's intent into an MPC set-point: motif segments, a drawn stroke,
an uploaded (x, y) path, or nothing at all.

`build_setpoint` has a precedence order, and it is the one place where "I drew a
path but it used the default trajectory" bugs live. `canvas_to_recorded` takes
raw pointer events, which is to say it takes anything: a single click, a
double-click that lands twice on the same pixel, a 4000-point scribble.
"""
import numpy as np
import pytest

import core
from conftest import BUILTINS, silent


@pytest.fixture(params=BUILTINS)
def spec(request):
    return core.get_spec(request.param)


# ───────────────────────────── precedence order ─────────────────────────────

def test_nothing_selected_gives_the_systems_default_trajectory(spec):
    got = core.build_setpoint(spec, [], None, spec.v0_default,
                             spec.wind_default, spec.zeta_default)
    want = spec.default_setpoint(float(spec.wind_default),
                                 float(spec.zeta_default))
    assert set(got) == set(want)
    for k in want:
        assert np.allclose(np.asarray(got[k], dtype=float),
                           np.asarray(want[k], dtype=float))


def test_a_recorded_path_sets_the_length(spec):
    n = 17
    rec = ([spec.v0_default] * n, list(np.linspace(0, 1, n)))
    sp = core.build_setpoint(spec, [], rec, spec.v0_default,
                             spec.wind_default, spec.zeta_default)
    assert set(sp) == set(spec.state_names)
    assert all(len(np.atleast_1d(v)) == n for v in sp.values())


def test_motif_segments_are_used_when_present(spec):
    segs = [dict(motif='straight', duration=float(spec.seg_duration),
                 angle=0.0, speed=float(spec.v0_default), freq=0.0, amp=0.0)]
    sp = core.build_setpoint(spec, segs, None, spec.v0_default,
                             spec.wind_default, spec.zeta_default)
    n = len(np.atleast_1d(sp[spec.state_names[0]]))
    assert n == pytest.approx(round(spec.seg_duration / spec.dt), abs=2)


def test_segments_win_over_a_recorded_path(spec):
    """ Documented precedence: adding motifs after drawing uses the motifs.
    (The app also clears one when the other is set; this pins the order the
    numerics use, independently of the UI doing that.) """
    segs = [dict(motif='straight', duration=float(spec.seg_duration),
                 angle=0.0, speed=float(spec.v0_default), freq=0.0, amp=0.0)]
    rec = ([spec.v0_default] * 999, [0.0] * 999)
    sp = core.build_setpoint(spec, segs, rec, spec.v0_default,
                             spec.wind_default, spec.zeta_default)
    assert len(np.atleast_1d(sp[spec.state_names[0]])) != 999


def test_wind_and_zeta_reach_the_setpoint():
    """ fly/fly7 carry ambient wind as states, so the two boxes must land in
    the set-point rather than being silently dropped. """
    spec = core.get_spec('fly7')
    sp = core.build_setpoint(spec, [], None, spec.v0_default, 1.25, 0.5)
    assert np.allclose(np.asarray(sp['w'], dtype=float), 1.25)
    assert np.allclose(np.asarray(sp['zeta'], dtype=float), 0.5)


# ───────────────────────── the drawing canvas → path ────────────────────────

W, H = 600, 360


def test_degenerate_strokes_are_refused_with_a_message(spec):
    """ A click that never became a stroke must produce a message, not a
    traceback and not a one-point "path" the MPC would choke on. """
    for pts in ([], [[10, 10]], [[10, 10], [10, 10]],
                [[10, 10], [10 + 1e-12, 10]]):
        rec, note = core.canvas_to_recorded(spec, pts, W, H, spec.v0_default, 0)
        assert rec is None
        assert isinstance(note, str) and note


def test_a_two_point_stroke_becomes_a_usable_path(spec):
    rec, note = core.canvas_to_recorded(spec, [[0, 0], [W, H]], W, H,
                                        spec.v0_default, 0)
    speed, heading = rec
    assert len(speed) == len(heading) >= 8
    assert np.isfinite(speed).all() and np.isfinite(heading).all()
    assert all(s > 0 for s in speed)
    assert 'Simulate' in note


def test_canvas_y_axis_is_flipped(spec):
    """ Canvas pixels count y DOWNWARD; data coordinates count up. A stroke
    drawn left-to-right and downward must therefore head into negative y — get
    this wrong and every drawn path is mirrored. """
    rec, _ = core.canvas_to_recorded(spec, [[0, 0], [W, H]], W, H,
                                     spec.v0_default, 0)
    heading = np.asarray(rec[1])
    assert -np.pi / 2 < heading[0] < 0.0, heading[0]
    rec_up, _ = core.canvas_to_recorded(spec, [[0, H], [W, 0]], W, H,
                                        spec.v0_default, 0)
    assert 0.0 < np.asarray(rec_up[1])[0] < np.pi / 2


def test_duration_fixes_the_step_count(spec):
    """ dur > 0 must decide N (and so the MPC solve time) regardless of how long
    the stroke is — that is the control's whole purpose. """
    short = [[0, 0], [50, 50]]
    long_ = [[0, 0], [W, H]]
    dur = 40 * spec.dt
    n_short = len(core.canvas_to_recorded(spec, short, W, H, 1.0, dur)[0][0])
    n_long = len(core.canvas_to_recorded(spec, long_, W, H, 1.0, dur)[0][0])
    assert n_short == n_long == np.clip(round(dur / spec.dt), 8, 800)


def test_step_count_is_clamped_both_ways(spec):
    """ Unbounded N is a hang: N is what the MPC iterates over. """
    huge = core.canvas_to_recorded(spec, [[0, 0], [W, H]], W, H, 1.0,
                                   1e6 * spec.dt)[0]
    assert len(huge[0]) == 800
    tiny = core.canvas_to_recorded(spec, [[0, 0], [W, H]], W, H, 1.0,
                                   1e-9)[0]
    assert len(tiny[0]) == 8
    # v0 = 0 would divide by zero without the floor in canvas_to_recorded
    zero_v = core.canvas_to_recorded(spec, [[0, 0], [W, H]], W, H, 0.0, 0)[0]
    assert 8 <= len(zero_v[0]) <= 800


def test_missing_canvas_size_does_not_divide_by_zero(spec):
    """ The JS bridge can hand back w/h = 0 before the canvas has laid out. """
    rec, note = core.canvas_to_recorded(spec, [[0, 0], [10, 10]], 0, 0,
                                        spec.v0_default, 0)
    assert rec is None or np.isfinite(rec[0]).all()


def test_a_long_scribble_is_resampled_not_truncated(spec):
    """ A freehand stroke arrives as hundreds of points; the path must be
    resampled to N steps, and the traversal must still cover the whole
    stroke. """
    t = np.linspace(0, 1, 400)
    pts = np.column_stack([t * W, H / 2 + 80 * np.sin(6 * np.pi * t)])
    rec, _ = core.canvas_to_recorded(spec, pts.tolist(), W, H,
                                     spec.v0_default, 0)
    speed, heading = np.asarray(rec[0]), np.asarray(rec[1])
    assert len(speed) >= 8 and np.isfinite(heading).all()
    # a weaving stroke must show the heading turning both ways
    assert heading.max() > heading.min()


# ─────────────────────────── it has to actually solve ───────────────────────

def test_a_drawn_path_survives_the_mpc():
    """ End to end on the cheapest system: a stroke the user drew has to come
    back as a solved trajectory of the promised length, not a solver failure. """
    spec = core.get_spec('alt2d')
    rec, _ = core.canvas_to_recorded(spec, [[0, 200], [W, 120]], W, H,
                                     spec.v0_default, 20 * spec.dt)
    sp = core.build_setpoint(spec, [], rec, spec.v0_default,
                             spec.wind_default, spec.zeta_default)
    eng = core.ObservabilityEngine(spec)
    with silent():
        eng.simulate_mpc(sp)
    assert eng.N == len(rec[0])
    assert np.isfinite(eng.X).all() and np.isfinite(eng.U).all()
