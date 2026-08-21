"""
The table/entry parsers in core.py.

These are the functions standing between a person typing in a browser and the
numerics, so this is where a naive user's input actually lands: blank rows, a
stray letter, a negative variance, a name that no longer exists after switching
systems. Every one of those has to end up as a defined, documented value rather
than an exception out of a Gradio callback — a traceback in a callback shows the
user a red box and no explanation.
"""
import numpy as np
import pandas as pd
import pytest

import core


NAMES = ['a', 'b', 'c']


# ───────────────────────────── initial guess (x₀) ────────────────────────────

def test_x0_vec_none_and_blank_mean_no_pin():
    """ No table, or a table with nothing filled in, must read as "use the
    seeded random draw for every state" — which is None, not zeros. Zeros would
    silently pin every state to 0. """
    assert core._x0_vec(None, NAMES) is None
    assert core._x0_vec([], NAMES) is None
    assert core._x0_vec([[n, None] for n in NAMES], NAMES) is None
    assert core._x0_vec([[n, ''] for n in NAMES], NAMES) is None


def test_x0_vec_pins_only_filled_rows():
    v = core._x0_vec([['a', 1.5], ['b', None], ['c', '']], NAMES)
    assert v[0] == 1.5
    assert np.isnan(v[1:]).all()


def test_x0_vec_keeps_negative_and_zero_values():
    """ An initial guess is a state value, so a negative or zero guess is
    ordinary and must pass through verbatim (unlike P₀, below). """
    v = core._x0_vec([['a', -2.5], ['b', 0.0]], NAMES)
    assert v[0] == -2.5 and v[1] == 0.0


def test_x0_vec_ignores_unknown_names_and_garbage():
    """ Stale rows survive a system switch in the browser; a typo survives
    anything. Neither may raise. """
    v = core._x0_vec([['nope', 9.0], ['a', 'abc'], ['b', 2.0], [], [None]],
                     NAMES)
    assert np.isnan(v[0])          # 'abc' unparseable -> left blank
    assert v[1] == 2.0
    assert np.isnan(v[2])


def test_x0_vec_accepts_a_dataframe():
    """ Gradio hands back a DataFrame in some configurations and a list of rows
    in others; both front-ends must behave identically. """
    df = pd.DataFrame([['a', 1.0], ['b', None], ['c', None]])
    v = core._x0_vec(df, NAMES)
    assert v[0] == 1.0 and np.isnan(v[1:]).all()


# ──────────────────────────────── P₀ diagonal ───────────────────────────────

def test_p0_vec_rejects_non_positive_variances():
    """ A variance ≤ 0 is not a value the filter can use — it makes P₀
    singular. Those entries must read as blank (fall back to the guess-implied
    spread), never as the number typed. """
    assert core._p0_vec([['a', 0.0], ['b', -3.0]], NAMES) is None
    v = core._p0_vec([['a', 4.0], ['b', -1.0]], NAMES)
    assert v[0] == 4.0 and np.isnan(v[1])


def test_p0_vec_scalar_broadcasts():
    """ The Streamlit front-end passes one number for every state. """
    assert np.all(core._p0_vec(7.0, NAMES) == 7.0)
    assert np.all(core._p0_vec('7', NAMES) == 7.0)
    assert core._p0_vec(0.0, NAMES) is None
    assert core._p0_vec(-1.0, NAMES) is None
    assert core._p0_vec('abc', NAMES) is None
    assert core._p0_vec(float('nan'), NAMES) is None


def test_p0_vec_blank_and_garbage():
    assert core._p0_vec(None, NAMES) is None
    assert core._p0_vec([[n, None] for n in NAMES], NAMES) is None
    v = core._p0_vec([['a', 'x'], ['b', 1.0]], NAMES)
    assert np.isnan(v[0]) and v[1] == 1.0


def test_p0_vec_dataframe():
    df = pd.DataFrame([['a', 2.0], ['b', None], ['c', None]])
    v = core._p0_vec(df, NAMES)
    assert v[0] == 2.0 and np.isnan(v[1:]).all()


# ─────────────────────────────── axis limits ────────────────────────────────

@pytest.mark.parametrize('lo,hi,want', [
    (None, None, None),
    ('', '', None),
    (1, None, (1.0, None)),
    (None, 2, (None, 2.0)),
    ('1.5', '2.5', (1.5, 2.5)),
    ('abc', 3, (None, 3.0)),        # one bad side must not kill the other
    (0, 0, (0.0, 0.0)),             # a real (if useless) limit, not "blank"
])
def test_lim(lo, hi, want):
    assert core._lim(lo, hi) == want


def test_ylim_dict_skips_blank_and_nameless_rows():
    d = core._ylim_dict([['a', 0, 1], ['b', None, None], [None, 1, 2],
                         ['', 1, 2], ['c', 5]])
    assert d == {'a': (0.0, 1.0), 'c': (5.0, None)}


def test_ylim_dict_none_and_dataframe():
    assert core._ylim_dict(None) == {}
    df = pd.DataFrame([['a', 0.0, 1.0]])
    assert core._ylim_dict(df) == {'a': (0.0, 1.0)}


def test_axis_lims_splits_the_shared_time_row():
    """ The time row is the shared x limit; every other row is that channel's
    y limit. A blank table must mean "autoscale everything". """
    rows = core._axis_rows(NAMES)
    assert core._axis_lims(rows) == (None, {})
    rows[0][1], rows[0][2] = 0.0, 5.0          # the time row
    rows[2][1] = -1.0                          # channel 'b', lower only
    xlim, ylims = core._axis_lims(rows)
    assert xlim == (0.0, 5.0)
    assert ylims == {'b': (-1.0, None)}
    assert core.TIME_ROW not in ylims          # must not leak into the y dict


# ─────────────────────────── ANN layer spec parsing ─────────────────────────

@pytest.mark.parametrize('text,want', [
    ('64, 64, 64', (64, 64, 64)),
    ('32;16', (32, 16)),
    ('8', (8,)),
    ('64.0, 12.9', (64, 12)),        # floats truncate rather than raise
    ('', (64, 64, 64)),              # empty -> repo default
    ('abc', (64, 64, 64)),
    (None, (64, 64, 64)),
    ('0', (64, 64, 64)),             # a zero-width layer is not a layer
    ('-5, 32', (32,)),
])
def test_parse_layers(text, want):
    assert core._parse_layers(text) == want


# ────────────────────────────── noise dicts ─────────────────────────────────

def test_uniform_dict_covers_every_name_and_floors_at_zero():
    d = core._uniform_dict(0.5, NAMES)
    assert set(d) == set(NAMES) and all(v == 0.5 for v in d.values())
    # zero/negative variance would make R or Q singular; floor instead
    assert all(v == 1e-30 for v in core._uniform_dict(0.0, NAMES).values())
    assert all(v == 1e-30 for v in core._uniform_dict(-9.0, NAMES).values())


def test_table_dict_fills_gaps_from_the_default():
    d = core._table_dict([['a', 1.0]], NAMES, 0.25)
    assert d == {'a': 1.0, 'b': 0.25, 'c': 0.25}


def test_table_dict_survives_bad_rows_and_unknown_names():
    d = core._table_dict([['a', 'x'], ['zz', 1.0], [], ['b'], ['c', 2.0]],
                         NAMES, 0.5)
    assert d == {'a': 0.5, 'b': 0.5, 'c': 2.0}


def test_table_dict_floors_non_positive_entries():
    d = core._table_dict([['a', 0.0], ['b', -1.0]], NAMES, 0.5)
    assert d['a'] == 1e-30 and d['b'] == 1e-30


def test_table_dict_none_and_dataframe():
    assert core._table_dict(None, NAMES, 0.5) == {n: 0.5 for n in NAMES}
    df = pd.DataFrame([['a', 3.0]])
    assert core._table_dict(df, NAMES, 0.5)['a'] == 3.0


# ─────────────────────── per-system default table shapes ────────────────────

@pytest.mark.parametrize('system', ['fly', 'fly7', 'drone', 'alt2d'])
def test_default_tables_match_the_systems_names(system):
    """ Every default table must be keyed by the CURRENT system's names: a row
    count or ordering mismatch here is what makes a switched-system table apply
    a value to the wrong state. """
    s = core.get_spec(system)
    sn, mn = list(s.state_names), list(s.measurement_names)
    for rows, names in ((core._q_rows(s), sn),
                        (core._q_rows_default(s), sn),
                        (core._x0_rows(s), sn),
                        (core._p0_rows(s), sn),
                        (core._r_rows(s), mn),
                        (core._r_rows_default(s), mn)):
        assert [r[0] for r in rows] == names
    # axis tables carry one extra shared time row, first
    assert [r[0] for r in core._axis_rows(sn)] == [core.TIME_ROW] + sn


@pytest.mark.parametrize('system', ['fly', 'fly7', 'drone', 'alt2d'])
def test_default_noise_values_are_usable(system):
    """ Defaults have to be strictly positive: a zero on the diagonal of R or Q
    is a singular matrix, and the user never touched it. """
    s = core.get_spec(system)
    assert core._default_r(s) > 0
    for rows in (core._q_rows_default(s), core._r_rows_default(s)):
        assert all(r[1] is None or float(r[1]) > 0 for r in rows)
    d = core._est_defaults(s)
    assert d['ann']['target'] in s.state_names
    assert d['aikf']['motif'] in s.input_names
    assert d['inj']['channel'] == 'none' or d['inj']['channel'] in s.input_names
    assert all(isinstance(d[k], float)
               for k in ('ukf_alpha', 'ukf_beta', 'ukf_kappa'))


# ───────────────────────────── one realization ──────────────────────────────

def test_realization_builds_a_bias_only_from_a_complete_spec():
    """ A half-filled disturbance box (channel picked, magnitude still 0) must
    mean "no disturbance", not a zero-magnitude bias that silently changes the
    cache key and the report. """
    s = core.get_spec('alt2d')
    ch = s.input_names[0]
    assert core._realization(s, 0, ch, 1.0, 0.0, 5.0, 0.0, None)['ub'] \
        == (0, 1.0, 0.0, 5.0)
    for args in (('none', 1.0, 0.0, 5.0),      # no channel
                 (ch, 0.0, 0.0, 5.0),          # no magnitude
                 (ch, 1.0, 5.0, 5.0),          # empty window
                 (ch, 1.0, 5.0, 1.0),          # inverted window
                 ('not_a_channel', 1.0, 0.0, 5.0)):
        assert core._realization(s, 0, *args, 0.0, None)['ub'] is None


def test_realization_normalizes_seed_and_noise():
    s = core.get_spec('alt2d')
    r = core._realization(s, None, 'none', 0, 0, 0, None, None)
    assert r['seed'] == 0 and r['unv'] == 0.0
    r = core._realization(s, 3.0, 'none', 0, 0, 0, 1e-3, None)
    assert r['seed'] == 3 and isinstance(r['seed'], int)
    assert r['unv'] == 1e-3


def test_realization_falls_back_to_the_specs_published_p0():
    """ A blank P₀ table must not mean "no prior": alt2d's published P₀ = 10 is
    what reproduces the paper figure, so it has to survive an untouched
    table. """
    s = core.get_spec('alt2d')
    blank = [[n, None] for n in s.state_names]
    r = core._realization(s, 0, 'none', 0, 0, 0, 0.0, None, p0=blank)
    if getattr(s, 'p0_paper', None):
        assert r['p0'] is not None and np.all(r['p0'] > 0)
    # an explicit entry always wins over the published value
    rows = [[n, None] for n in s.state_names]
    rows[0][1] = 42.0
    r = core._realization(s, 0, 'none', 0, 0, 0, 0.0, None, p0=rows)
    assert r['p0'][0] == 42.0
