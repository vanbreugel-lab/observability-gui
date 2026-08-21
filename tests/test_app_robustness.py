"""
What happens when the user does something unintended.

Every function here is a Gradio callback. A callback that raises shows the user
a red box with no explanation and leaves the page in whatever state it was in,
so "raises a clear exception" is NOT acceptable behaviour at this layer — the
requirement is a returned status string the user can act on. These tests are
therefore mostly of the form "give it something wrong, assert it comes back with
a message instead of an exception".

The one exception is `load_system_py`, which is allowed (and expected) to raise:
its caller `on_upload_system` is what has to turn that into a message, and both
halves are tested.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

import core
import render
from conftest import BUILTINS, silent


# ────────────────────────── uploading a system .py ──────────────────────────

VALID = '''
import numpy as np
state_names = ['p', 'v']
input_names = ['a']
measurement_names = ['p']
dt = 0.1
def f(X, U):
    p, v = X
    a, = U
    return [v, a]
def h(X, U):
    p, v = X
    return [p]
'''

WITH_MPC = VALID + '''
position_states = ['p']
input_bounds = {'a': (-5.0, 5.0)}
'''


def write(tmp_path, text, name='sys.py'):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_a_valid_upload_becomes_a_usable_spec(app, tmp_path):
    spec = app.load_system_py(write(tmp_path, VALID))
    assert list(spec.state_names) == ['p', 'v']
    assert list(spec.measurement_names) == ['p']
    assert list(spec.input_names) == ['a']
    assert spec.dt == 0.1
    assert spec.color_state_default in spec.state_names
    assert np.allclose(np.asarray(spec.f([1.0, 2.0], [3.0]), dtype=float),
                       [2.0, 3.0])


def test_upload_without_input_names_gets_a_placeholder(app, tmp_path):
    """ input_names is optional in the documented contract, but the engine and
    every plot index inputs by name, so something has to be there. """
    text = VALID.replace("input_names = ['a']", '')
    spec = app.load_system_py(write(tmp_path, text))
    assert len(spec.input_names) >= 1


def test_position_states_builds_the_tracking_mpc(app, tmp_path):
    """ Declaring position_states is what makes "upload an (x, y) path" work;
    without it the app must say so rather than fail inside do-mpc. """
    plain = app.load_system_py(write(tmp_path, VALID, 'a.py'))
    assert not plain.position_states
    assert not hasattr(plain, 'make_setpoint')
    with_mpc = app.load_system_py(write(tmp_path, WITH_MPC, 'b.py'))
    assert with_mpc.position_states == ['p']
    for attr in ('make_setpoint', 'default_setpoint', 'build_mpc'):
        assert hasattr(with_mpc, attr), attr
    sp = with_mpc.make_setpoint({'p': np.linspace(0, 1, 12)}, 12)
    assert set(sp) == set(with_mpc.state_names)
    assert all(len(np.atleast_1d(v)) == 12 for v in sp.values())


@pytest.mark.parametrize('missing', ['f', 'h', 'state_names',
                                     'measurement_names', 'dt'])
def test_a_missing_required_attribute_names_itself(app, tmp_path, missing):
    """ The error has to say WHICH attribute is missing — that message is the
    only documentation a user gets at 2am. """
    text = '\n'.join(l for l in VALID.splitlines()
                     if not l.startswith(f'{missing} =')
                     and not l.startswith(f'def {missing}('))
    if missing in ('f', 'h'):        # drop the function body too
        text = VALID.replace(f'def {missing}(X, U):', 'def _unused(X, U):')
    with pytest.raises(ValueError, match=missing):
        app.load_system_py(write(tmp_path, text, f'{missing}.py'))


def test_a_broken_upload_is_reported_not_raised(app, tmp_path):
    """ on_upload_system is the callback, so it must convert anything
    load_system_py throws into a status string. """
    for text, name in ((VALID.replace('def f(X, U):', 'def f(X, U)'), 'syn.py'),
                       ('raise RuntimeError("boom")', 'boom.py'),
                       ('', 'empty.py')):
        out = app.on_upload_system(write(tmp_path, text, name))
        assert out[0] is None, name              # no spec was adopted
        assert 'error' in str(out[-1]).lower(), (name, out[-1])


def test_no_file_selected_is_not_an_error(app):
    out = app.on_upload_system(None)
    assert out[0] is None and 'no file' in str(out[-1]).lower()


def test_a_valid_upload_repopulates_every_control(app, tmp_path):
    """ The uploaded system's names must reach the sensor/state pickers and the
    R/Q tables, or the next Simulate applies the previous system's names. """
    out = app.on_upload_system(write(tmp_path, WITH_MPC))
    spec = out[0]
    assert spec is not None
    assert 'loaded custom system' in str(out[-1])
    # the two dataframe updates carry one row per measurement / per state
    r_rows = out[8]['value']
    q_rows = out[9]['value']
    assert [r[0] for r in r_rows] == list(spec.measurement_names)
    assert [r[0] for r in q_rows] == list(spec.state_names)


def test_a_custom_system_solves_and_filters(app, tmp_path):
    """ End to end for the upload path: spec -> MPC -> Gramian -> filters. This
    is the flow a new user follows with their own model, and nothing in it is
    covered by the built-in systems. """
    spec = app.load_system_py(write(tmp_path, WITH_MPC))
    eng = core.ObservabilityEngine(spec)
    with silent():
        eng.simulate_mpc(spec.make_setpoint({'p': np.linspace(0, 1, 16)}, 16))
    assert eng.N == 16 and np.isfinite(eng.X).all()
    qd = {s: 1e-4 for s in spec.state_names}
    rd = {m: 0.1 for m in spec.measurement_names}
    ev, _ = eng.empirical_ev(4, 1e-4, rd, 1e-6)
    assert np.isfinite(ev[list(spec.state_names)].to_numpy(dtype=float)).any()
    for run in (eng.run_ekf, eng.run_ukf):
        X_hat, P_diag = run(qd, rd, seed=0)
        assert np.isfinite(X_hat).all() and (P_diag > 0).all()


# ─────────────────────── uploading a trajectory data file ───────────────────

def make_csv(tmp_path, name='traj.csv', n=30, ids=None):
    rows = []
    for oid in (ids or [None]):
        t = np.linspace(0, 1, n)
        d = {'frame': np.arange(n), 'x': t, 'y': 0.5 * t ** 2}
        if oid is not None:
            d['obj_id'] = [oid] * n
        rows.append(pd.DataFrame(d))
    path = tmp_path / name
    pd.concat(rows).to_csv(path, index=False)
    return str(path)


def test_read_table_rejects_an_unknown_extension(app, tmp_path):
    p = tmp_path / 'data.txt'
    p.write_text('x,y\n1,2\n')
    with pytest.raises(ValueError, match='unsupported file type'):
        app._read_table(str(p))


def test_read_table_reads_csv(app, tmp_path):
    df = app._read_table(make_csv(tmp_path))
    assert list(df.columns)[:3] == ['frame', 'x', 'y']


def test_physical_upload_happy_path(app, tmp_path):
    traj, dd, status = app.on_upload_physical(make_csv(tmp_path), 'x', 'y',
                                             '', 'alt2d', None)
    assert traj and '✓' in status
    (x, y, speed, heading), = traj.values()
    assert len(x) == len(y) == len(speed) == len(heading) == 30
    assert np.isfinite(speed).all() and np.isfinite(heading).all()
    assert dd['choices'] == list(traj)


def test_physical_upload_groups_by_id(app, tmp_path):
    path = make_csv(tmp_path, ids=['a', 'b'])
    traj, dd, status = app.on_upload_physical(path, 'x', 'y', 'obj_id',
                                             'alt2d', None)
    assert set(traj) == {'a', 'b'}
    assert dd['value'] in traj


def test_a_missing_column_is_reported_with_the_columns_it_has(app, tmp_path):
    traj, dd, status = app.on_upload_physical(make_csv(tmp_path), 'lon', 'lat',
                                             '', 'alt2d', None)
    assert traj is None
    assert 'load error' in status and 'lon' in status
    assert 'x' in status, 'the message should show what columns ARE there'


def test_physical_upload_edge_cases_do_not_raise(app, tmp_path):
    # no file
    assert app.on_upload_physical(None, 'x', 'y', '', 'alt2d', None)[0] is None
    # a nonexistent path
    out = app.on_upload_physical(str(tmp_path / 'nope.csv'), 'x', 'y', '',
                                 'alt2d', None)
    assert out[0] is None and 'error' in out[2].lower()
    # an empty file
    p = tmp_path / 'empty.csv'
    p.write_text('')
    assert app.on_upload_physical(str(p), 'x', 'y', '', 'alt2d', None)[0] is None
    # non-numeric coordinates
    p2 = tmp_path / 'text.csv'
    pd.DataFrame({'x': ['a', 'b'], 'y': ['c', 'd']}).to_csv(p2, index=False)
    out = app.on_upload_physical(str(p2), 'x', 'y', '', 'alt2d', None)
    assert out[0] is None and 'error' in out[2].lower()


def test_select_physical_needs_a_selection(app):
    rec, segs, disp, status = app.on_select_physical(None, None, None)
    assert rec is None and 'select' in status.lower()
    rec, *_ = app.on_select_physical({'a': ([0], [0], [0], [0])}, 'b', None)
    assert rec is None


def test_select_physical_maps_onto_a_custom_systems_position_states(app,
                                                                   tmp_path):
    """ A custom system needs ≥2 position_states for an (x, y) track; with one,
    the user must be told, not handed a broken set-point. """
    one_pos = app.load_system_py(write(tmp_path, WITH_MPC))
    traj = {'t': ([0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0])}
    rec, _segs, _disp, status = app.on_select_physical(traj, 't', one_pos)
    assert rec is None and 'position_states' in status


def test_select_physical_for_a_builtin_returns_speed_and_heading(app):
    traj = {'t': ([0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0])}
    rec, segs, _disp, status = app.on_select_physical(traj, 't', None)
    assert isinstance(rec, tuple) and len(rec) == 2
    assert segs == [], 'selecting a measured path must clear motif segments'


# ───────────────────────────── the drawing bridge ───────────────────────────

@pytest.mark.parametrize('payload', ['', 'not json', '{', 'null', '[]',
                                     '{"pts": []}', '{"pts": [[1,2]]}',
                                     '{"pts": "nonsense"}',
                                     '{"pts": [[0,0],[1,1]], "w": 0, "h": 0}'])
def test_malformed_canvas_payloads_are_survivable(app, payload):
    """ The canvas payload comes from JavaScript through a hidden textbox — the
    one input a user could also edit or corrupt by accident. """
    rec, note, segs, disp = app.apply_drawing(payload, 'alt2d', 1.0, 0.0)
    assert rec is None or (isinstance(rec, tuple) and len(rec) == 2)
    assert isinstance(note, str) and note
    assert segs == []


def test_a_real_canvas_payload_produces_a_path(app):
    payload = json.dumps({'pts': [[0, 300], [200, 200], [400, 120]],
                          'w': 600, 'h': 360})
    rec, note, segs, _ = app.apply_drawing(payload, 'alt2d', 1.0, 0.0)
    assert rec is not None and len(rec[0]) >= 8
    assert 'Simulate' in note


def test_clear_recorded(app):
    rec, note = app.clear_recorded()
    assert rec is None and 'Cleared' in note


# ────────────────────────── the motif segment builder ───────────────────────

def test_segment_list_edits(app):
    segs, text = app.add_seg([], 'turn', 0.1, -1.57, 1.0, 0.0, 0.0)
    assert len(segs) == 1 and 'turn' in text
    segs2, _ = app.add_seg(segs, 'straight', 0.2, 0.0, 1.0, 0.0, 0.0)
    assert len(segs2) == 2
    undone, _ = app.undo_seg(segs2)
    assert len(undone) == 1
    cleared, text = app.clear_seg()
    assert cleared == [] and 'No segments' in text
    # undo on an empty list must be a no-op, not an IndexError
    assert app.undo_seg([])[0] == []
    assert app.undo_seg(None)[0] == []


def test_seg_text_handles_every_motif(app):
    for motif in app.MOTIFS:
        segs, text = app.add_seg([], motif, 0.1, 0.5, 1.0, 1.0, 0.5)
        assert isinstance(text, str) and text
        assert core.build_setpoint(core.get_spec('fly7'), segs, None,
                                   1.0, 0.0, 0.0)


def test_on_motif_reveals_only_the_relevant_shape_controls(app):
    """ Each motif uses a different parameter; showing all of them is how a user
    ends up setting a weave amplitude on a straight line. """
    for motif in app.MOTIFS:
        angle, speed, freq, amp = app.on_motif(motif)
        assert speed['visible'] is (motif == 'speed')
        assert freq['visible'] is (motif == 'weave')
        assert amp['visible'] is (motif == 'weave')
