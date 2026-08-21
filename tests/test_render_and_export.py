"""
Drawing the figures, and exporting them.

A figure that is *constructed* is not a figure that *draws*: matplotlib defers
most of its work to render time, so a bad limit, a missing label or an empty
collection only raises when the canvas is actually painted. Every test here
therefore calls `canvas.draw()`, which is what Gradio does when it serializes
the plot — and what the PDF export does on all seven at once.
"""
import os

import numpy as np
import pytest
from matplotlib.figure import Figure

import core
import render
from conftest import BUILTINS, q_of, r_of, silent
from test_compute import args, run


# ───────────────────────────── figure assembly ──────────────────────────────

@pytest.mark.parametrize('system', BUILTINS)
def test_every_system_draws_all_seven_figures(app, system):
    out = run(app, system=system, do_stoch=True, q_obs=True, q_con=True)
    assert not out[-2].startswith('⚠'), out[-2]
    for i, fig in enumerate(out[:app._N_FIGS]):
        assert isinstance(fig, Figure), (system, i, type(fig))
        fig.canvas.draw()


@pytest.mark.parametrize('over', [
    dict(ev_states=[], est_states=[], meas_sel=[]),          # nothing selected
    dict(fim_states=[], fim_sensors=[]),                     # nothing analyzed
    dict(traj_mode='line'),                                  # no arrowheads
    dict(s_true=False, s_ekf=False, s_ukf=False),            # every trace off
    dict(m_noisy=False, m_true=False, m_pred=False, m_ukf=False),
    dict(cmap='viridis', vmin=1.0, vmax=1.0),                # degenerate range
    dict(do_stoch=True, q_con=True, q_obs=False),            # one basis only
], ids=['no-selection', 'no-analysis', 'line-mode', 'no-state-traces',
        'no-meas-traces', 'flat-colour-range', 'constructability-only'])
def test_display_options_still_draw(app, over):
    out = run(app, **over)
    assert not out[-2].startswith('⚠'), (over, out[-2])
    for fig in out[:app._N_FIGS]:
        if isinstance(fig, Figure):
            fig.canvas.draw()


def test_selecting_more_states_than_fit_is_truncated_not_crashed(app):
    """ The states/measurements panels cap at 8 rows; selecting everything on
    the 18-state fly model must clip rather than build an unreadable (or
    zero-height) figure. """
    s = core.get_spec('fly')
    out = run(app, system='fly', est_states=list(s.state_names),
              ev_states=list(s.state_names),
              meas_sel=list(s.measurement_names))
    assert not out[-2].startswith('⚠'), out[-2]
    fig_est = out[4]
    fig_est.canvas.draw()
    assert fig_est.get_size_inches()[1] > 0


def test_a_stale_selection_is_dropped_not_plotted(app):
    """ A selection made for one system outlives the switch to the next, so a
    name that no longer exists can reach the plotters. Each one turns into a
    names.index(...) a few lines later, and the figures are assembled OUTSIDE
    _compute's try/except — so this used to raise ValueError straight out of the
    callback. It must fall back to the defaults and draw. """
    out = run(app, est_states=['not_a_state'], meas_sel=['not_a_sensor'],
              ev_states=['not_a_state'], color='not_a_state')
    assert len(out) == app._N_FIGS + 2
    assert not out[-2].startswith('⚠'), out[-2]
    for i, fig in enumerate(out[:app._N_FIGS]):
        assert isinstance(fig, Figure), (i, type(fig))
        fig.canvas.draw()


def test_render_helpers_drop_unknown_names(app):
    s = core.get_spec('fly7')
    assert render._known(['v_para', 'nope'], s.state_names, ['x']) == ['v_para']
    assert render._known(['nope'], s.state_names, ['x']) == ['x']
    assert render._known([], s.state_names, ['x']) == ['x']
    assert render._known(None, s.state_names, ['x']) == ['x']
    # order follows the caller's selection, not the spec
    assert render._known(['phi', 'x'], s.state_names, ['y']) == ['phi', 'x']


def test_render_helpers_agree_with_the_spec(app):
    for name in BUILTINS:
        s = core.get_spec(name)
        assert set(render.meas_default(s)) <= set(s.measurement_names)
        assert set(render.inputs_default(s)) <= set(s.input_names)


# ─────────────────────────────── the PDF export ─────────────────────────────

def test_export_writes_a_real_pdf(app, tmp_path):
    """ The download button reads the bundle _compute stashed, so the report is
    the analysis on screen rather than a fresh run. """
    out = run(app)
    sess = out[-1]
    assert 'analysis' in sess
    bundle = sess['analysis']
    assert len(bundle['figs']) == app._N_FIGS
    assert all(isinstance(f, Figure) for _title, f in bundle['figs'])
    path = app.download_analysis(sess)
    assert path and os.path.exists(path), path
    assert os.path.getsize(path) > 5000, 'suspiciously small PDF'
    with open(path, 'rb') as fh:
        assert fh.read(5) == b'%PDF-', 'not a PDF'


def test_export_summary_records_what_was_run(app):
    """ The summary page is what makes a run reproducible, so the fields that
    identify it have to be there. """
    out = run(app, system='fly7', do_stoch=True)
    meta = out[-1]['analysis']['meta']
    sections = dict((name, dict(rows)) for name, rows in meta)
    assert 'run' in sections and 'system' in sections
    assert 'observability' in sections and 'estimators' in sections
    assert sections['system']['name'] == 'fly7'
    assert 'implementation' in sections['estimators']
    assert 'realization' in sections['estimators']
    obs = sections['observability']
    for key in ('window w', 'lambda (regularizer)', 'epsilon (finite diff)',
                'sensors used', 'Q diagonal', 'R diagonal'):
        assert key in obs, key


def test_export_reports_the_shared_realization(app):
    shared = dict((n, dict(r)) for n, r in
                  run(app, est_split=False)[-1]['analysis']['meta'])
    split = dict((n, dict(r)) for n, r in
                 run(app, est_split=True, ukf_seed=5)[-1]['analysis']['meta'])
    assert 'shared' in shared['estimators']['realization']
    assert 'independent' in split['estimators']['realization']


def test_export_before_any_run_returns_nothing(app):
    """ Pressing download first must warn, not raise or write a broken file. """
    import gradio as gr
    try:
        assert app.download_analysis({}) is None
        assert app.download_analysis(None) is None
    except gr.Error:
        pass            # a Gradio warning outside a request context is fine


def test_analysis_pdf_handles_a_missing_figure(tmp_path):
    """ The matrix figures come back as None when they could not be built; the
    report must skip them instead of failing the whole export. """
    fig = Figure(figsize=(4, 3))
    fig.add_subplot(111).plot([0, 1], [0, 1])
    path = str(tmp_path / 'out.pdf')
    render.analysis_pdf(path, [('kept', fig), ('missing', None)],
                        [('run', [('generated', 'now')])], title='t')
    assert os.path.exists(path) and os.path.getsize(path) > 1000
