"""
Observability explorer — Streamlit edition.

A Streamlit front-end over the same compute core as the Gradio app
(``app_custom.py``). Everything numerical lives in ``core.py``: the set-point
builder, ``compute_payload``, the figure assembly and the per-system defaults.
This file is only controls and layout, so the two front-ends cannot disagree
about the science.

Run locally:

    pip install -r requirements.txt
    streamlit run streamlit_app.py

Deploy free on Streamlit Community Cloud: push the repo to GitHub, point a new
app at this file, and it stays reachable after you close your laptop (it sleeps
when idle and wakes on the next visit).

Not ported from the Gradio app, deliberately: uploading a custom system .py. It
exec()s the uploaded file, which is fine on your own machine but is remote code
execution on a public deployment. Use ``app_custom.py`` locally for that.

The draw-a-trajectory canvas needs the optional streamlit-drawable-canvas
package; without it that one control hides itself and everything else works.
"""
import contextlib
import io

import numpy as np
import pandas as pd
import streamlit as st

import core
from core import ObservabilityEngine
import render

try:
    # NB: never pass background_image= to st_canvas. That path calls
    # streamlit.elements.image.image_to_url, which Streamlit removed — the
    # canvas raises AttributeError. A plain canvas is all a path needs.
    from streamlit_drawable_canvas import st_canvas
except ImportError:                       # optional — the canvas just hides
    st_canvas = None

st.set_page_config(page_title='Observability Explorer', page_icon='📈',
                   layout='wide')

# the uploader entry is Gradio-only (see the module docstring)
SYSTEM_CHOICES = [(lab, key) for lab, key in core.SYSTEM_LABELS
                  if key != 'custom']
MOTIFS = ['straight', 'turn', 'circle', 'speed', 'weave']


# ───────────────────────────── cached compute ────────────────────────────────

@st.cache_resource(show_spinner=False, max_entries=3)
def build_engine(system, dt_hz, segs_key, v0, wind, zeta, recorded_key=None):
    """ An engine with its MPC trajectory already solved.

    Cached on exactly the inputs that change the trajectory, which is what the
    Gradio app expresses as its rebuild/update split: touching a display or
    noise control reuses this engine (and its internal result caches) instead of
    re-solving the MPC. Streamlit needs hashable arguments, hence segs_key.
    """
    spec = core.get_spec(system, 1.0 / float(dt_hz))
    engine = ObservabilityEngine(spec)
    segs = [dict(s) for s in segs_key]
    recorded = (list(recorded_key[0]), list(recorded_key[1])) \
        if recorded_key else None
    sp = core.build_setpoint(spec, segs, recorded, v0, wind, zeta)
    with contextlib.redirect_stdout(io.StringIO()):      # hush IPOPT
        engine.simulate_mpc(sp)
    return engine


def _segs_key(segs):
    """ The segment list as something st.cache_resource can hash. """
    return tuple(tuple(sorted(s.items())) for s in segs)


def _canvas_points(json_data):
    """ fabric.js canvas objects → an (n, 2) array of canvas pixel points.

    Freehand strokes arrive as SVG-ish path commands (['M', x, y],
    ['Q', cx, cy, x, y], …) — the last two numbers of each command are its
    endpoint, which is enough to trace the stroke. Clicked waypoints arrive as
    circles positioned by left/top, so their centre is offset by the radius. """
    pts = []
    for obj in (json_data or {}).get('objects', []):
        kind = obj.get('type')
        if kind == 'path':
            for cmd in obj.get('path', []):
                nums = [v for v in cmd[1:] if isinstance(v, (int, float))]
                if len(nums) >= 2:
                    pts.append((nums[-2], nums[-1]))
        elif kind == 'circle':                        # point / waypoint mode
            r = float(obj.get('radius', 0.0))
            pts.append((float(obj.get('left', 0.0)) + r,
                        float(obj.get('top', 0.0)) + r))
        elif kind == 'line':
            pts.append((float(obj.get('x1', 0.0)), float(obj.get('y1', 0.0))))
            pts.append((float(obj.get('x2', 0.0)), float(obj.get('y2', 0.0))))
    return np.asarray(pts, dtype=float).reshape(-1, 2)


# ───────────────────────────── sidebar controls ──────────────────────────────
# Every widget key carries the system name. Switching systems then lands on a
# fresh widget with that system's defaults, instead of trying to keep a stale
# state/sensor selection valid against a different model's names.

sb = st.sidebar
sb.title('Observability Explorer')

sys_label = sb.selectbox('System', [lab for lab, _ in SYSTEM_CHOICES],
                         key='system')
system = dict((lab, key) for lab, key in SYSTEM_CHOICES)[sys_label]
K = lambda name: f'{system}:{name}'          # per-system widget key

_probe = core.get_spec(system)
d = core._est_defaults(_probe)
base_hz = core._BASE_HZ[system]

with sb.expander('Trajectory', expanded=True):
    hz_labels = {v: lab for lab, v in core.DT_HZ}
    hz_opts = [v for _, v in core.DT_HZ]
    dt_hz = st.selectbox('sample rate', hz_opts,
                         index=hz_opts.index(base_hz) if base_hz in hz_opts
                         else len(hz_opts) - 1,
                         format_func=lambda v: hz_labels[v], key=K('hz'))
    v0 = st.number_input('initial speed v₀', value=float(_probe.v0_default),
                         step=0.1, key=K('v0'))
    has_wind = 'zeta' in _probe.state_names
    if has_wind:
        c1, c2 = st.columns(2)
        wind = c1.number_input('wind speed', value=float(_probe.wind_default),
                               step=0.1, key=K('wind'))
        zeta = c2.number_input('wind direction ζ',
                               value=float(_probe.zeta_default), step=0.1,
                               key=K('zeta'))
    else:
        wind, zeta = float(_probe.wind_default), float(_probe.zeta_default)

    st.caption('Leave the segment list empty to fly this system\'s default '
               'trajectory.')
    segs = st.session_state.setdefault(K('segs'), [])
    motif = st.selectbox('motif', MOTIFS, key=K('motif'))
    dur = st.number_input('duration [s]', value=float(_probe.seg_duration),
                          min_value=0.1, step=0.5, key=K('dur'))
    angle = speed = freq = amp = 0.0
    if motif == 'circle':
        angle = st.number_input('angle [rad]', value=float(2 * np.pi),
                                key=K('ang'))
    elif motif == 'turn':
        angle = st.number_input('angle [rad]', value=float(-np.pi / 2),
                                key=K('ang'))
    elif motif == 'speed':
        speed = st.number_input('speed change', value=0.5, key=K('spd'))
    elif motif == 'weave':
        cf, ca = st.columns(2)
        freq = cf.number_input('frequency', value=0.5, key=K('frq'))
        amp = ca.number_input('amplitude', value=0.5, key=K('amp'))
    b1, b2, b3 = st.columns(3)
    if b1.button('add', width='stretch', key=K('add')):
        segs.append(dict(motif=motif, duration=float(dur), angle=float(angle),
                         speed=float(speed), freq=float(freq), amp=float(amp)))
        st.rerun()
    if b2.button('undo', width='stretch', key=K('undo')) and segs:
        segs.pop()
        st.rerun()
    if b3.button('clear', width='stretch', key=K('clr')) and segs:
        segs.clear()
        st.rerun()
    if segs:
        st.markdown('  →  '.join(
            f'{i + 1}. {core.MOTIF_LABELS[s["motif"]](s)}'
            for i, s in enumerate(segs)))

# A drawn path takes precedence over the motif list (same rule as
# core.build_setpoint). Kept out of the sidebar because the canvas needs room.
recorded = st.session_state.get(K('recorded'))
if st_canvas is not None:
    with sb.expander('Draw a trajectory', expanded=False):
        st.caption(f'Drag to draw a path, or switch to *point* and click ≥2 '
                   f'waypoints. The box maps onto this system\'s arena, '
                   f'x {_probe.rec_xlim}, y {_probe.rec_ylim}. A drawn path '
                   f'overrides the motif list above.')
        mode = st.radio('mode', ['freedraw', 'point'], horizontal=True,
                        key=K('dmode'))
        draw_dur = st.number_input(
            'traverse in [s] (0 = hold v₀ instead)', value=0.0, min_value=0.0,
            step=0.5, key=K('ddur'),
            help='Fixes the step count N = duration/dt regardless of path '
                 'length, so a long path stays cheap to solve. 0 instead holds '
                 'v₀, and N grows with the path.')
        canvas = st_canvas(stroke_width=3, stroke_color='#e15759',
                           background_color='#ffffff', height=300, width=300,
                           drawing_mode=mode, display_toolbar=True,
                           point_display_radius=4, key=K('canvas'))
        pts = _canvas_points(canvas.json_data if canvas is not None else None)
        c_ap, c_cl = st.columns(2)
        if c_ap.button('apply drawing', width='stretch', key=K('dapply')):
            rec, note = core.canvas_to_recorded(_probe, pts, 300, 300, v0,
                                                draw_dur)
            st.session_state[K('recorded')] = (
                (tuple(rec[0]), tuple(rec[1])) if rec else None)
            st.session_state[K('drawnote')] = note
            segs.clear()                       # drawn path wins; drop motifs
            st.rerun()
        if c_cl.button('clear drawing', width='stretch', key=K('dclear')):
            st.session_state.pop(K('recorded'), None)
            st.session_state.pop(K('drawnote'), None)
            st.rerun()
        if st.session_state.get(K('drawnote')):
            st.markdown(st.session_state[K('drawnote')])
        elif len(pts):
            st.caption(f'{len(pts)} points captured — press **apply drawing**.')
else:
    sb.caption('Install `streamlit-drawable-canvas` to draw trajectories.')

with sb.expander('Observability', expanded=True):
    w = st.slider('sliding window w [steps]', 3, 60, int(_probe.w_default),
                  key=K('w'))
    lam = st.selectbox('λ (prior / regularizer)', core.LAM_VALS, index=2,
                       format_func=lambda v: f'{v:.0e}', key=K('lam'))
    eps_idx = min(range(len(core.EPS_VALS)),
                  key=lambda i: abs(core.EPS_VALS[i] - _probe.eps_default))
    eps = st.selectbox('ε (finite-difference step)', core.EPS_VALS,
                       index=eps_idx, format_func=lambda v: f'{v:.0e}',
                       key=K('eps'))
    fim_sensors = st.multiselect('sensors', list(_probe.measurement_names),
                                 default=list(_probe.fim_sensors),
                                 key=K('sens'))
    fim_states = st.multiselect('states', list(_probe.state_names),
                                default=list(_probe.fim_states),
                                key=K('fstates'))

with sb.expander('Measurement noise (R)'):
    r_mode = st.radio('R', ['uniform', 'custom'], horizontal=True,
                      index=1 if getattr(_probe, 'r_paper', None) else 0,
                      key=K('rmode'), label_visibility='collapsed')
    r_uniform = st.number_input('R for all sensors',
                                value=float(core._default_r(_probe)),
                                format='%.6g', key=K('runi'))
    r_table = None
    if r_mode == 'custom':
        r_table = st.data_editor(
            pd.DataFrame(core._r_rows_default(_probe),
                         columns=['sensor', 'R']),
            hide_index=True, width='stretch', key=K('rtbl'))

with sb.expander('Process noise (Q) & FIM basis'):
    do_stoch = st.checkbox('include process noise (stochastic gramian)',
                           value=False, key=K('stoch'))
    # the basis selection exists even while process noise is off, matching the
    # Gradio app (where these checkboxes keep their values but stay hidden), so
    # switching Q on lands on the same 'observability' default in both
    q_obs, q_con = True, False
    q_mode, q_uniform, q_table = 'uniform', 1e-4, None
    q_noise = 'uncorrelated'
    if do_stoch:
        st.caption('Pick either FIM basis, or both to compare them directly — '
                   'each adds its own trajectory panel and min-EV curve.')
        q_obs = st.checkbox('observability basis (initial state, ~smoother)',
                            value=True, key=K('qobs'))
        q_con = st.checkbox('constructability basis (final state, ~filter)',
                            value=False, key=K('qcon'))
        q_mode = st.radio('Q', ['uniform', 'custom'], horizontal=True,
                          index=1 if getattr(_probe, 'q_paper', None) else 0,
                          key=K('qmode'), label_visibility='collapsed')
        q_noise = st.selectbox(
            'noise type', ['uncorrelated', 'vanloan'],
            format_func=lambda v: ('uncorrelated' if v == 'uncorrelated'
                                   else 'correlated (Van Loan)'),
            key=K('qnoise'),
            help='uncorrelated: Q_k is the entered diagonal at every step. '
                 'correlated: Q_k is discretised through the local dynamics, '
                 'so it gains off-diagonal terms and varies along the '
                 'trajectory.')
        q_uniform = st.number_input('Q for all states', value=1e-4,
                                    format='%.6g', key=K('quni'))
        if q_mode == 'custom':
            q_table = st.data_editor(
                pd.DataFrame(core._q_rows_default(_probe),
                             columns=['state', 'Q']),
                hide_index=True, width='stretch', key=K('qtbl'))

_X0_HELP = ('Where each state\'s estimate **starts**. Fill a row to set that '
            'state\'s initial guess outright — any value, negative included. '
            'Leave it blank and the state starts at the truth plus a ~10% '
            'random error drawn from the seed. P₀ follows from this: pinning a '
            'guess far from the truth widens that state\'s prior variance to '
            'match.')


def _filter_controls(name):
    """ One estimator's own realization — seed, per-state initial guess and an
    injected input disturbance. Each filter carries a full copy, so the EKF and
    UKF can be initialized independently (matching seeds = identical data). """
    k = lambda s: K(f'{name}_{s}')
    show = st.checkbox(f'show {name}', value=True, key=k('show'))
    seed = st.number_input('noise seed', value=0, step=1, key=k('seed'))
    st.caption(_X0_HELP)
    x0 = st.data_editor(
        pd.DataFrame(core._x0_rows(_probe), columns=['state', 'initial guess']),
        hide_index=True, width='stretch', key=k('x0'))
    st.caption('Bias added to one measured acceleration between t₀ and t₁, plus '
               'optional zero-mean noise on those readings throughout. Filters '
               'consume accelerations in the prediction step, so they are '
               'perturbed here; the truth always flies the clean inputs.')
    ch = st.selectbox('input channel', ['none'] + list(_probe.input_names),
                      index=(['none'] + list(_probe.input_names)).index(
                          d['inj']['channel']),
                      key=k('ch'))
    c1, c2 = st.columns(2)
    mag = c1.number_input('bias magnitude', value=float(d['inj']['mag']),
                          key=k('mag'))
    unv = c2.number_input('input noise var', value=float(d['u_noise']),
                          format='%.6g', key=k('unv'))
    c3, c4 = st.columns(2)
    t0 = c3.number_input('start t₀ [s]', value=float(d['inj']['t0']), key=k('t0'))
    t1 = c4.number_input('end t₁ [s]', value=float(d['inj']['t1']), key=k('t1'))
    p0 = st.text_input('P0 diagonal (blank -> from initial guess)',
                       value='' if getattr(_probe, 'p0_paper', None) is None
                       else str(_probe.p0_paper), key=k('p0'))
    return show, core._realization(_probe, seed, ch, mag, t0, t1, unv, x0, p0)


with sb.expander('EKF'):
    s_ekf, r_ekf = _filter_controls('EKF')
with sb.expander('UKF'):
    s_ukf, r_ukf = _filter_controls('UKF')
    st.caption('Sigma points sit at ±√(α²(n+κ))·σ about the mean.')
    ca, cb, ck = st.columns(3)
    ukf_alpha = ca.number_input('α', value=float(d['ukf_alpha']), key=K('ua'))
    ukf_beta = cb.number_input('β', value=float(d['ukf_beta']), key=K('ub'))
    ukf_kappa = ck.number_input('κ', value=float(d['ukf_kappa']), key=K('uk'))

with sb.expander('Plot settings'):
    color = st.selectbox('trajectory color state', list(_probe.state_names),
                         index=list(_probe.state_names).index(
                             _probe.color_state_default), key=K('color'))
    ev_states = st.multiselect('min-EV curve states',
                               list(_probe.state_names),
                               default=list(_probe.ev_states_default),
                               key=K('ev'))
    traj_mode = st.radio('trajectory style', ['arrowhead', 'continuous'],
                         horizontal=True, key=K('tmode'))
    cmap = st.selectbox('colormap', core.CMAPS, key=K('cmap'))
    cv1, cv2 = st.columns(2)
    vmin = cv1.number_input('color vmin', value=1e-4, format='%.6g',
                            key=K('vmin'))
    vmax = cv2.number_input('color vmax', value=1e6, format='%.6g',
                            key=K('vmax'))
    st.markdown('**States plot**')
    est_states = st.multiselect('states shown', list(_probe.state_names),
                                default=list(_probe.ev_states_default),
                                key=K('eststates'))
    s_true = st.checkbox('show true state', value=True, key=K('strue'))
    sx1, sx2 = st.columns(2)

    st_ytbl = st.data_editor(
        pd.DataFrame(core._axis_rows(_probe.state_names),
                     columns=['axis', 'min', 'max']),
        hide_index=True, width='stretch', key=K('stytbl'))
    st.markdown('**Measurements plot**')
    meas_sel = st.multiselect('measurements shown',
                              list(_probe.measurement_names),
                              default=list(render.meas_default(_probe)),
                              key=K('meas'))
    m1, m2 = st.columns(2)
    m_noisy = m1.checkbox('noisy samples', value=True, key=K('mnoisy'))
    m_true = m2.checkbox('noise-free h(x,u)', value=True, key=K('mtrue'))
    m3, m4 = st.columns(2)
    m_pred = m3.checkbox('EKF prediction', value=True, key=K('mpred'))
    m_ukf = m4.checkbox('UKF prediction', value=False, key=K('mukf'))
    mx1, mx2 = st.columns(2)

    ms_ytbl = st.data_editor(
        pd.DataFrame(core._axis_rows(_probe.measurement_names),
                     columns=['axis', 'min', 'max']),
        hide_index=True, width='stretch', key=K('msytbl'))

_st_xlim, _st_ylims = core._axis_lims(st_ytbl)
_ms_xlim, _ms_ylims = core._axis_lims(ms_ytbl)


# ───────────────────────────── assemble + compute ────────────────────────────
# These three dicts mirror _compute() in app_custom.py exactly; core.py does the
# rest, so both front-ends produce identical numbers for identical settings.

with st.spinner('solving the MPC trajectory…'):
    try:
        engine = build_engine(system, dt_hz, _segs_key(segs), v0, wind, zeta,
                              recorded)
    except Exception as exc:
        st.error(f'trajectory failed: {exc}')
        st.stop()
spec = engine.spec

r_diag = (core._uniform_dict(r_uniform, spec.measurement_names)
          if r_mode == 'uniform'
          else core._table_dict(r_table, spec.measurement_names, r_uniform))
if not do_stoch:
    q_diag = {n: 1e-9 for n in spec.state_names}
elif q_mode == 'uniform':
    q_diag = core._uniform_dict(q_uniform, spec.state_names)
else:
    q_diag = core._table_dict(q_table, spec.state_names, q_uniform)

sensors = tuple(fim_sensors) or tuple(spec.measurement_names)
states = tuple(fim_states) or tuple(spec.state_names)
bases = tuple(b for b, on in (('observability', q_obs),
                              ('constructability', q_con)) if on)

kmax = max(engine.N - int(min(w, max(engine.N - 1, 3))), 0)
mat_k = st.sidebar.slider('𝒪 window start [step]', 0, max(kmax, 1), 0,
                          key=K('matk'))
fi_k = st.sidebar.slider('F⁻¹ window start [step]', 0, max(kmax, 1), 0,
                         key=K('fik'))

an, ak = d['ann'], d['aikf']
p = dict(w_raw=int(w), eps=float(eps), lam=float(lam), r_diag=r_diag,
         q_diag=q_diag, q_noise=str(q_noise), sensors=sensors, states=states,
         do_stoch=bool(do_stoch) and bool(bases), bases=bases,
         # ANN / AI-KF are work-in-progress and disabled in both front-ends
         do_ann=False,
         ann_target=an['target'], ann_layers=core._parse_layers(an['layers']),
         ann_steps=an['time_steps'], ann_traj=an['n_traj'],
         ann_epochs=an['epochs'], ann_batch=an['batch'], ann_noise=an['noise'],
         aikf_motif=ak['motif'], aikf_window=ak['window'],
         aikf_upper=ak['upper'], aikf_r_hi=ak['r_hi'], aikf_r_lo=ak['r_lo'])
est = dict(EKF=r_ekf, UKF=r_ukf, ukf_alpha=float(ukf_alpha),
           ukf_beta=float(ukf_beta), ukf_kappa=float(ukf_kappa))

with st.spinner('computing…'):
    try:
        payload = core.compute_payload(engine, ('update',), p, est)
    except np.linalg.LinAlgError as exc:
        payload = {'error': f'singular matrix ({exc}) — raise λ or adjust R/Q'}
    except Exception as exc:
        payload = {'error': repr(exc)}

if 'error' in payload:
    st.error(payload['error'])
    st.stop()

disp = dict(color=color, ev_states=ev_states, est_states=est_states,
            meas_sel=meas_sel, arrowhead=(traj_mode == 'arrowhead'),
            cmap=cmap, vmin=float(vmin), vmax=float(vmax),
            s_true=bool(s_true), s_ekf=bool(s_ekf), s_ukf=bool(s_ukf),
            s_ann=False, s_aikf=False,
            m_noisy=bool(m_noisy), m_true=bool(m_true), m_pred=bool(m_pred),
            m_ukf=bool(m_ukf), m_ann=False, m_aikf=False,
            st_xlim=_st_xlim, st_ylims=_st_ylims,
            ms_xlim=_ms_xlim, ms_ylims=_ms_ylims,
            mat_k=int(mat_k), fi_k=int(fi_k),
            do_stoch=p['do_stoch'], q_diag=q_diag, bases=bases,
            eps=float(eps), r_diag=r_diag, lam=float(lam),
            sensors=sensors, states=states)

fig_main, fig_O, fig_Finv, fig_minev, fig_est, fig_meas, fig_inputs = \
    core._figs_from_payload(engine, spec, payload, disp)


# ──────────────────────────────── main area ──────────────────────────────────

modes = (f'empirical {payload["t_emp"] * 1e3:.0f} ms'
         + (f' + {"+".join(bases)} gramian {payload["t_stoch"] * 1e3:.0f} ms'
            if p['do_stoch'] else ''))
msg = f'{engine.N} steps — {modes}'
if payload['lam'] < 10 * payload['floor']:
    msg += (f' | ⚠ λ = {payload["lam"]:.0e} at/below the round-off floor '
            f'(≈ {payload["floor"]:.0e}) — raise λ')
if bool(do_stoch) and not bases:
    msg += ' | ⚠ process noise on but no FIM basis selected'
if payload.get('note'):
    msg += f' | {payload["note"]}'
st.caption(msg)

st.pyplot(fig_main, width='stretch')

if fig_minev is not None:
    with st.expander('min error variance — state × time heat map'):
        st.pyplot(fig_minev, width='stretch')
if fig_O is not None:
    with st.expander('observability matrix 𝒪'):
        st.pyplot(fig_O, width='stretch')
if fig_Finv is not None:
    with st.expander('Fisher information inverse F⁻¹'):
        st.pyplot(fig_Finv, width='stretch')

st.markdown('## Estimation')
if fig_est is not None:
    with st.expander('States (true / EKF / UKF)', expanded=True):
        st.pyplot(fig_est, width='stretch')
if fig_meas is not None:
    if fig_inputs is not None:
        with st.expander('Inputs (measured accelerations)', expanded=True):
            st.pyplot(fig_inputs)
    with st.expander('Measurements', expanded=True):
        st.pyplot(fig_meas, width='stretch')
