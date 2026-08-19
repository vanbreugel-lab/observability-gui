"""
Observability explorer — web edition.

A Gradio front-end over the *identical* compute engine used by the PyQt desktop
app (``observability_gui.ObservabilityEngine``): MPC trajectory → linearized
Eq. (33) Gramian → pybounds empirical FIM → nonlinear Monte-Carlo Gramian →
EKF/UKF. The heavy numerics run in Python on the server; the browser only shows
HTML/CSS widgets and the rendered matplotlib figures.

Run locally:

    pip install -r requirements.txt
    python app.py            # → http://127.0.0.1:7860

This same file is the entry point when the repo is deployed as a Hugging Face
Space (see README.md).
"""
import os
import sys
import io
import re
import json
import time
import warnings
import contextlib
import socket
import threading
import importlib.util

# do-mpc prints ONNX/opcua "feature not available" UserWarnings on import; they
# are informational (those optional extras just aren't installed) — hush them.
warnings.filterwarnings('ignore', message='.*ONNX feature.*')
warnings.filterwarnings('ignore', message='.*opcua feature.*')

# engine/ holds the vendored compute layer copied from the pybounds Qanalysis/
# folder; put it on the path so its flat imports (drone_model,
# stochastic_observability) resolve.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'engine'))

import matplotlib
matplotlib.use('Agg')                 # headless: no display needed on a server
import numpy as np
from matplotlib.figure import Figure
import gradio as gr

import pandas as pd
from observability_gui import (ObservabilityEngine, SYSTEMS, build_setpoints,
                               MOTIF_LABELS, SystemSpec)
import render


# ── shared compute core (UI-agnostic; see core.py) ──
# System metadata, the set-point builder, the numerics of one refresh and the
# per-system defaults live in core.py so this Gradio app and streamlit_app.py
# compute identically. Imported by name rather than * so the wiring below still
# reads as a flat namespace.
import core
from core import (SYSTEM_LABELS, LAM_VALS, EPS_VALS, DT_HZ, CMAPS, _BASE_HZ,
                  ANN_ENABLED, get_spec, build_setpoint, compute_payload,
                  _figs_from_payload, _est_defaults, _realization,
                  _q_rows_default, _r_rows_default, _x0_rows, _x0_vec,
                  _axis_rows, _axis_lims, _ylim_dict, _lim, _parse_layers,
                  _uniform_dict,
                  _default_r, _r_rows, _q_rows, _table_dict)




# ─────────────── custom system + trajectory loaders (local use) ──────────────
# A custom system is an uploaded .py that defines the MODEL only — f, h,
# state_names, measurement_names, dt (input_names/angle_* optional). To enable
# MPC reconstruction from a measured (x,y) path it also declares position_states
# + input_bounds. The trajectory is uploaded SEPARATELY as a physical data file.

def load_system_py(path):
    """ Import an uploaded system script and build a SystemSpec from its
    module-level f, h, names, dt. No trajectory is read here. """
    mod_spec = importlib.util.spec_from_file_location('user_system', path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)                       # local-only: runs user code
    for attr in ('f', 'h', 'state_names', 'measurement_names', 'dt'):
        if not hasattr(mod, attr):
            raise ValueError(f"system script must define `{attr}`")
    sn = list(mod.state_names)
    inn = list(getattr(mod, 'input_names', []) or ['_u'])
    mn = list(mod.measurement_names)
    pos = list(getattr(mod, 'position_states', []) or [])
    bounds = dict(getattr(mod, 'input_bounds', {}) or {})
    spec = SystemSpec(
        name='custom', dt=float(mod.dt), f=mod.f, h=mod.h,
        state_names=sn, input_names=inn, measurement_names=mn,
        angle_states=tuple(getattr(mod, 'angle_states', ()) or ()),
        angle_measurements=tuple(getattr(mod, 'angle_measurements', ()) or ()),
        disc_method='rk4var', color_state_default=sn[0],
        ev_states_default=tuple(sn), position_states=pos, input_bounds=bounds)

    # If the script declares position_states, auto-build a generic
    # position-tracking MPC (objective f/h don't provide) so a measured (x,y)
    # path can be reconstructed. The measured columns map onto position_states.
    if pos:
        def make_setpoint(pos_dict, N):
            sp = {s: np.zeros(N) for s in sn}
            for s, arr in pos_dict.items():
                sp[s] = np.asarray(arr, dtype=float)
            return sp

        def default_setpoint(*_):
            return make_setpoint({}, 50)            # placeholder (hold origin)

        def build_mpc(sim):
            cost = sum((sim.model.x[s] - sim.model.tvp[s + '_set']) ** 2
                       for s in pos)
            sim.mpc.set_objective(mterm=cost, lterm=cost)
            sim.mpc.set_rterm(**{u: 1e-4 for u in inn})
            for u, (lo, hi) in bounds.items():
                sim.mpc.bounds['lower', '_u', u] = lo
                sim.mpc.bounds['upper', '_u', u] = hi

        spec.make_setpoint = make_setpoint
        spec.default_setpoint = default_setpoint
        spec.build_mpc = build_mpc
    return spec


def _read_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(path)
    if ext in ('.parquet', '.pq'):
        return pd.read_parquet(path)
    if ext in ('.h5', '.hdf', '.hdf5'):
        return pd.read_hdf(path)
    raise ValueError(f"unsupported file type '{ext}' (use csv/parquet/hdf)")


def on_upload_system(path):
    """ Upload handler: load a system .py → custom spec, repopulate the
    sensor/state selection controls and the R/Q noise tables with its names. """
    if not path:
        no = gr.update()
        return (None, no, no, no, no, no, no, no, no, no, 'no file')
    try:
        spec = load_system_py(path)
    except Exception as e:
        no = gr.update()
        return (None, no, no, no, no, no, no, no, no, no, f'⚠ system load error: {e}')
    sn, mn = spec.state_names, spec.measurement_names
    ev_d, meas_d = list(spec.ev_states_default), list(render.meas_default(spec))
    mpc = ' (position-tracking MPC ready)' if spec.position_states else \
          ' — no position_states declared, so MPC reconstruction is unavailable'
    return (
        spec,
        gr.update(choices=list(mn), value=list(mn)),      # fim_sensors (all)
        gr.update(choices=list(sn), value=list(sn)),      # fim_states (all)
        gr.update(choices=list(sn), value=spec.color_state_default),
        gr.update(choices=list(sn), value=ev_d),          # ev_states
        gr.update(choices=list(sn), value=ev_d),          # est_states
        gr.update(choices=list(mn), value=meas_d),        # meas_sel
        gr.update(value=_default_r(spec)),                # r_uniform
        gr.update(value=_r_rows(spec)),                   # r_table
        gr.update(value=_q_rows(spec)),                   # q_table
        f'✓ loaded custom system — states {sn}, measurements {mn}{mpc}. '
        'Now upload a physical (x, y) trajectory.')


# ── physical-trajectory upload (measured position → MPC reconstruct) ──
# The common real-data case: you have x, y over time. For a BUILT-IN vehicle we
# derive ground speed + course (the paper's method for measured fly flight); for
# a CUSTOM vehicle we map x, y onto its declared position_states. Either way the
# MPC reconstructs the full state + inputs.

def _derive_speed_heading(group, x_col, y_col, dt):
    x = group[x_col].to_numpy(dtype=float)
    y = group[y_col].to_numpy(dtype=float)
    xv_col, yv_col = x_col + 'vel', y_col + 'vel'          # e.g. x→xvel
    if xv_col in group.columns and yv_col in group.columns:
        vx = group[xv_col].to_numpy(dtype=float)
        vy = group[yv_col].to_numpy(dtype=float)
    else:                                                  # else finite-difference
        vx, vy = np.gradient(x, dt), np.gradient(y, dt)
    speed = np.sqrt(vx ** 2 + vy ** 2)
    heading = np.unwrap(np.arctan2(vy, vx))
    return speed.tolist(), heading.tolist()


def on_upload_physical(path, x_col, y_col, id_col, system, custom_spec):
    """ Read a physical-trajectory file and store per-trajectory (x, y, speed,
    heading). Populates the trajectory dropdown; no compute yet. """
    if not path:
        return None, gr.update(choices=[], value=None), 'no file'
    try:
        df = _read_table(path)
        dt = (custom_spec.dt if custom_spec is not None else get_spec(system).dt)
        for c in (x_col, y_col):
            if c not in df.columns:
                raise ValueError(f"column '{c}' not in file (has {list(df.columns)[:12]}…)")
        sort_col = 'frame' if 'frame' in df.columns else None

        def pack(g):
            x = g[x_col].to_numpy(dtype=float); y = g[y_col].to_numpy(dtype=float)
            sp, hd = _derive_speed_heading(g, x_col, y_col, dt)
            return (x.tolist(), y.tolist(), sp, hd)

        out = {}
        if id_col and id_col in df.columns:
            for oid, grp in df.groupby(id_col):
                out[str(oid)] = pack(grp.sort_values(sort_col) if sort_col else grp)
        else:
            out['(whole file)'] = pack(df.sort_values(sort_col) if sort_col else df)
    except Exception as e:
        return None, gr.update(choices=[], value=None), f'⚠ load error: {e}'
    ids = list(out.keys())
    return (out, gr.update(choices=ids, value=ids[0]),
            f'✓ {len(ids)} trajectories loaded from {os.path.basename(path)}. '
            'Pick one to reconstruct via MPC.')


def on_select_physical(phys_traj, obj_id, custom_spec):
    """ Build the `recorded` set-point from the selected trajectory. Built-in
    systems track (speed, heading); a custom system maps x, y onto its declared
    position_states. Clears segments so Simulate uses this via MPC. """
    if not phys_traj or obj_id not in phys_traj:
        return None, [], seg_text([]), 'select a trajectory'
    x, y, speed_sp, heading_sp = phys_traj[obj_id]
    n = len(speed_sp)
    if custom_spec is not None:                       # map x, y → position_states
        ps = custom_spec.position_states
        if len(ps) < 2:
            return None, [], seg_text([]), '⚠ custom system needs ≥2 position_states'
        recorded = {'pos': {ps[0]: x, ps[1]: y}}
    else:                                             # built-in: speed + course
        recorded = (speed_sp, heading_sp)
    return (recorded, [], seg_text([]),
            f'✓ trajectory {obj_id}: {n} steps. Reconstructing via MPC…')


def _compute(sess, custom_spec, segs, recorded, system, dt_hz, v0,
             wind, zeta, w, lam, eps, r_mode, r_uniform, r_table, do_stoch,
             q_obs, q_con, q_mode, q_noise, q_uniform, q_table,
             fim_sensors, fim_states, color,
             ev_states, est_states, meas_sel, traj_mode, cmap, vmin, vmax,
             s_true, s_ekf, s_ukf, s_ann, s_aikf,
             m_noisy, m_true, m_pred, m_ukf, m_ann, m_aikf,
             ekf_seed, ekf_x0, ekf_ch, ekf_mag, ekf_t0, ekf_t1, ekf_unv, ekf_p0,
             ukf_seed, ukf_x0, ukf_ch, ukf_mag, ukf_t0, ukf_t1, ukf_unv, ukf_p0,
             ukf_alpha, ukf_beta, ukf_kappa,
             ann_target, ann_layers, ann_steps, ann_traj, ann_epochs,
             ann_batch, ann_noise,
             aikf_motif, aikf_window, aikf_upper, aikf_r_hi, aikf_r_lo,
             st_ytbl, ms_ytbl,
             mat_k, fi_k, rebuild):
    sess = sess or {}
    spec = sess.get('spec')
    engine = sess.get('engine')

    # ── choose the spec: an uploaded custom system overrides the built-in ──
    if custom_spec is not None:
        spec = custom_spec
        if engine is None or sess.get('cid') != id(custom_spec):
            engine = ObservabilityEngine(spec)
            sess = dict(system='custom', spec=spec, engine=engine,
                        cid=id(custom_spec))
            rebuild = True
        if not spec.position_states:
            return (gr.update(),) * 6 + (
                '⚠ this custom system declares no position_states, so MPC '
                'reconstruction from a measured path is unavailable', sess)
        if recorded is None:
            return (gr.update(),) * 6 + (
                '⚠ upload a physical (x, y) trajectory for the custom system', sess)
    else:
        dt_val = 1.0 / float(dt_hz)
        if (engine is None or sess.get('system') != system
                or sess.get('dt') != dt_val):
            spec = get_spec(system, dt_val)
            engine = ObservabilityEngine(spec)
            sess = dict(system=system, dt=dt_val, spec=spec, engine=engine)
            rebuild = True

    # ── trajectory: unified MPC path (custom specs get an auto-built MPC) ──
    job = (('rebuild', build_setpoint(spec, segs, recorded, v0, wind, zeta))
           if (rebuild or not engine.N) else ('update',))

    # R per sensor (uniform or per-sensor table). Q enters ONLY the stochastic
    # gramian: ON → process noise; OFF → Q≈0 (deterministic observability/FIM).
    r_diag = (_uniform_dict(r_uniform, spec.measurement_names)
              if r_mode == 'uniform'
              else _table_dict(r_table, spec.measurement_names, r_uniform))
    if not do_stoch:
        q_diag = {n: 1e-9 for n in spec.state_names}
    elif q_mode == 'uniform':
        q_diag = _uniform_dict(q_uniform, spec.state_names)
    else:
        q_diag = _table_dict(q_table, spec.state_names, q_uniform)
    _st_xlim, _st_ylims = _axis_lims(st_ytbl)
    _ms_xlim, _ms_ylims = _axis_lims(ms_ytbl)
    sensors = tuple(fim_sensors) or tuple(spec.measurement_names)
    states = tuple(fim_states) or tuple(spec.state_names)
    # stochastic FIM bases, independently selectable: observability (initial
    # state, ~smoother) and/or constructability (final state, ~filter). Order is
    # fixed so panels and curves keep a stable position when one is toggled.
    bases = tuple(b for b, on in (('observability', q_obs),
                                  ('constructability', q_con)) if on)
    p = dict(w_raw=int(w), eps=float(eps), lam=float(lam), r_diag=r_diag,
             q_diag=q_diag, q_noise=str(q_noise), sensors=sensors, states=states,
             do_stoch=bool(do_stoch) and bool(bases), bases=bases,
             # hard gate: a stale browser session could still post True
             do_ann=ANN_ENABLED and (bool(s_ann) or bool(s_aikf)),
             ann_target=(ann_target if ann_target in spec.state_names
                         else spec.color_state_default),
             ann_layers=_parse_layers(ann_layers),
             ann_steps=max(1, int(ann_steps or 4)),
             ann_traj=max(2, int(ann_traj or 16)),
             ann_epochs=max(1, int(ann_epochs or 100)),
             ann_batch=max(8, int(ann_batch or 256)),
             ann_noise=max(0.0, float(ann_noise or 0.0)),
             aikf_motif=aikf_motif, aikf_window=max(1, int(aikf_window or 20)),
             aikf_upper=max(1e-9, float(aikf_upper or 0.5)),
             aikf_r_hi=max(1e-30, float(aikf_r_hi or 1e12)),
             aikf_r_lo=max(1e-30, float(aikf_r_lo or 1e-3)))
    # one independent estimation problem per filter (see _realization), plus the
    # UKF's sigma-point tuning, which the AI-KF also uses
    est = dict(
        EKF=_realization(spec, ekf_seed, ekf_ch, ekf_mag, ekf_t0, ekf_t1,
                         ekf_unv, ekf_x0, ekf_p0),
        UKF=_realization(spec, ukf_seed, ukf_ch, ukf_mag, ukf_t0, ukf_t1,
                         ukf_unv, ukf_x0, ukf_p0),
        ukf_alpha=float(ukf_alpha or 1.0), ukf_beta=float(ukf_beta or 0.0),
        ukf_kappa=float(ukf_kappa or 0.0))

    try:
        payload = compute_payload(engine, job, p, est)
    except np.linalg.LinAlgError as e:
        payload = {'error': f'singular matrix ({e}) — raise λ or adjust R/Q'}
    except Exception as e:                          # never crash the UI
        payload = {'error': repr(e)}

    if 'error' in payload:
        status = f'⚠ {payload["error"]}'
        no = gr.update()
        return (no,) * 6 + (status, sess)

    disp = dict(color=color, ev_states=ev_states, est_states=est_states,
                meas_sel=meas_sel, arrowhead=(traj_mode == 'arrowhead'),
                cmap=cmap, vmin=float(vmin), vmax=float(vmax),
                s_true=bool(s_true), s_ekf=bool(s_ekf), s_ukf=bool(s_ukf),
                s_ann=bool(s_ann), s_aikf=bool(s_aikf),
                m_noisy=bool(m_noisy), m_true=bool(m_true), m_pred=bool(m_pred),
                m_ukf=bool(m_ukf), m_ann=bool(m_ann), m_aikf=bool(m_aikf),
                st_xlim=_st_xlim, st_ylims=_st_ylims,
                ms_xlim=_ms_xlim, ms_ylims=_ms_ylims,
                mat_k=int(mat_k or 0), fi_k=int(fi_k or 0),
                do_stoch=p['do_stoch'], q_diag=q_diag, bases=bases,
                q_noise=p['q_noise'],
                eps=float(eps), r_diag=r_diag, lam=float(lam),
                sensors=sensors, states=states)
    fig_main, fig_O, fig_Finv, fig_minev, fig_est, fig_meas, fig_inputs = _figs_from_payload(
        engine, spec, payload, disp)

    floor_note = ''
    if payload['lam'] < 10 * payload['floor']:
        floor_note = (f' | ⚠ λ = {payload["lam"]:.0e} at/below round-off floor '
                      f'(≈ {payload["floor"]:.0e}) — raise λ')
    note = payload['note']
    modes = (f'empirical {payload["t_emp"] * 1e3:.0f} ms'
             + (f' + ANN/AI-KF {payload["t_ann"] * 1e3:.0f} ms'
                if payload.get('t_ann') else '')
             + (f' + {"+".join(bases)} gramian '
                f'{payload["t_stoch"] * 1e3:.0f} ms' if p['do_stoch'] else ''))
    if bool(do_stoch) and not bases:
        modes += ' | ⚠ process noise on but no FIM basis selected'
    status = f'✓ done — {note + " | " if note else ""}{modes}{floor_note}'

    # matrix figures return None when unselected → clears the plot; the core
    # est/meas panels keep their previous content when unavailable.
    return (fig_main, fig_O, fig_Finv, fig_minev,
            fig_est if fig_est is not None else gr.update(),
            fig_meas if fig_meas is not None else gr.update(),
            fig_inputs if fig_inputs is not None else gr.update(),
            status, sess)


# matplotlib and casadi are NOT thread-safe; Gradio runs every callback in a
# worker thread, so concurrent callbacks corrupt/crash them. Serialize all
# compute+render through one lock (fine for a single-user explorer; the engine
# caches make repeats fast).
_COMPUTE_LOCK = threading.Lock()


def simulate(*args):
    """ Re-solve the MPC trajectory, then compute + render. """
    with _COMPUTE_LOCK:
        return _compute(*args, rebuild=True)


def update(*args):
    """ Recompute from the existing trajectory (cheap — the engine caches the
    filter results by key) and render. """
    with _COMPUTE_LOCK:
        return _compute(*args, rebuild=False)


def on_system(system):
    """ Switch systems: reset every control to the new system's defaults AND
    recompute the figures in one call. The special 'custom' entry doesn't build
    a spec — it just reveals the system-.py uploader and waits. """
    if system == 'custom':
        no = gr.update()
        return (no, no, no, no, no, no,             # sensor/state controls: no-op
                no, no, no,                          # w, lam, eps: no-op
                no, no, no, no, no, no, no,          # r_mode..q_table: no-op
                no,                                  # dt_hz: no-op
                no, no, no,                          # v0, wind, zeta: no-op
                no,                                  # p_duration: no-op
                [], seg_text([]),                    # segs, seg_display
                None, no, no,                        # recorded, rec_status, draw_data
                None,                                # custom_spec cleared (upload sets it)
                gr.update(visible=True),             # reveal sys_file
                no, no, no, no, no, no,              # EKF x0, inj ch/mag/t0/t1, unv
                no, no, no, no, no, no,              # UKF x0, inj ch/mag/t0/t1, unv
                no, no, no,                          # ukf α, β, κ
                no, no, no, no, no, no, no,          # ANN controls
                no, no, no, no, no,                  # AI-KF controls
                no, no,                              # y-limit tables
                None, None, None, None, None, None,  # clear stale figures
                'Upload a system .py, then a physical (x, y) trajectory.', {})
    s = get_spec(system)
    lam_d = min(LAM_VALS, key=lambda v: abs(v - 1e-6))
    eps_d = min(EPS_VALS, key=lambda v: abs(v - s.eps_default))
    fim_sensors_d, fim_states_d = list(s.fim_sensors), list(s.fim_states)
    color_d = s.color_state_default
    ev_d = list(s.ev_states_default)
    meas_d = list(render.meas_default(s))
    v0_d, wind_d, zeta_d = (float(s.v0_default), float(s.wind_default),
                            float(s.zeta_default))
    r_d = _default_r(s)

    hz_d = _BASE_HZ[system]
    d = _est_defaults(s)
    an, ak, ij = d['ann'], d['aikf'], d['inj']
    q_rows_d, r_rows_d = _q_rows_default(s), _r_rows_default(s)
    # 'custom' Q/R mode so the per-state paper values in the tables are the ones
    # actually used, rather than the single uniform box
    qr_mode = 'custom' if getattr(s, 'q_paper', None) else 'uniform'
    figs = simulate({}, None, [], None, system, hz_d, v0_d, wind_d, zeta_d,
                    s.w_default, lam_d, eps_d, qr_mode, r_d, r_rows_d, False,
                    True, False, qr_mode, 'uncorrelated', 1e-4, q_rows_d,
                    fim_sensors_d, fim_states_d, color_d, ev_d, ev_d, meas_d,
                    'arrowhead', 'inferno_r', 1e-4, 1e6, True, True, True,
                    False, False, True, True, True, False, False, False,
                    # EKF then UKF, each with its own realization; the same
                    # defaults, so they start out on identical data
                    0, d['x0'], ij['channel'], ij['mag'], ij['t0'], ij['t1'],
                    d['u_noise'], d['p0'],
                    0, d['x0'], ij['channel'], ij['mag'], ij['t0'], ij['t1'],
                    d['u_noise'], d['p0'],
                    d['ukf_alpha'], d['ukf_beta'], d['ukf_kappa'],
                    an['target'], an['layers'], an['time_steps'], an['n_traj'],
                    an['epochs'], an['batch'], an['noise'],
                    ak['motif'], ak['window'], ak['upper'], ak['r_hi'],
                    ak['r_lo'],
                    _axis_rows(s.state_names), _axis_rows(s.measurement_names),
                    0, 0)
    (fig_main_, fig_O_, fig_Finv_, fig_minev_,
     fig_est_, fig_meas_, fig_inputs_, status_, sess_) = figs

    return (
        gr.update(choices=list(s.measurement_names), value=fim_sensors_d),
        gr.update(choices=list(s.state_names), value=fim_states_d),
        gr.update(choices=list(s.state_names), value=color_d),
        gr.update(choices=list(s.state_names), value=ev_d),
        gr.update(choices=list(s.state_names), value=ev_d),
        gr.update(choices=list(s.measurement_names), value=meas_d),
        gr.update(value=s.w_default),                # w
        gr.update(value=lam_d),                      # lam
        gr.update(value=eps_d),                      # eps
        gr.update(value=qr_mode),                    # r_mode
        gr.update(value=r_d),                        # r_uniform
        gr.update(value=r_rows_d),                   # r_table
        gr.update(value=False),                      # do_stoch
        gr.update(value=qr_mode),                    # q_mode
        gr.update(value='uncorrelated'),             # q_noise
        gr.update(value=1e-4),                       # q_uniform
        gr.update(value=q_rows_d),                   # q_table
        gr.update(value=hz_d),                       # dt_hz
        gr.update(value=v0_d), gr.update(value=wind_d), gr.update(value=zeta_d),
        gr.update(value=float(s.seg_duration)),      # segment duration
        [], seg_text([]),                            # segments state + display
        None, '', '',                                # drawing: recorded, status, JSON bridge
        None,                                        # clear custom system
        gr.update(visible=False),                    # hide sys_file for built-ins
        # EKF initial guess + disturbance, then the UKF's own copies. The
        # guess table MUST be reset: its rows are the new system's state names.
        gr.update(value=d['x0']),
        gr.update(choices=['none'] + list(s.input_names), value=ij['channel']),
        gr.update(value=ij['mag']), gr.update(value=ij['t0']),
        gr.update(value=ij['t1']), gr.update(value=d['u_noise']),
        gr.update(value=d['x0']),
        gr.update(choices=['none'] + list(s.input_names), value=ij['channel']),
        gr.update(value=ij['mag']), gr.update(value=ij['t0']),
        gr.update(value=ij['t1']), gr.update(value=d['u_noise']),
        gr.update(value=d['ukf_alpha']), gr.update(value=d['ukf_beta']),
        gr.update(value=d['ukf_kappa']),
        gr.update(choices=list(s.state_names), value=an['target']),
        gr.update(value=an['layers']), gr.update(value=an['time_steps']),
        gr.update(value=an['n_traj']), gr.update(value=an['epochs']),
        gr.update(value=an['batch']), gr.update(value=an['noise']),
        gr.update(choices=list(s.input_names), value=ak['motif']),
        gr.update(value=ak['window']), gr.update(value=ak['upper']),
        gr.update(value=ak['r_hi']), gr.update(value=ak['r_lo']),
        gr.update(value=_axis_rows(s.state_names)),
        gr.update(value=_axis_rows(s.measurement_names)),
        fig_main_, fig_O_, fig_Finv_, fig_minev_, fig_est_, fig_meas_,
        fig_inputs_, status_, sess_,
    )


# ─────────────────────── trajectory (motif) builder ─────────────────────────

MOTIFS = ['straight', 'turn', 'circle', 'speed', 'weave']


def seg_text(segs):
    """ Human-readable one-line summary of the current segment sequence. """
    if not segs:
        return ('*No segments — Simulate uses the system default trajectory. '
                'Add motifs below to build your own path.*')
    parts = '  →  '.join(f'{i + 1}. {MOTIF_LABELS[s["motif"]](s)}'
                         for i, s in enumerate(segs))
    total = sum(s['duration'] for s in segs)
    return f'**Path:** {parts}  \n*(total {total:g}s — press ▶ Simulate)*'


def on_motif(motif):
    """ Show only the shape parameter(s) the chosen motif uses; give turn and
    circle a sensible default angle. Returns updates for angle/speed/freq/amp. """
    if motif == 'circle':
        angle = gr.update(visible=True, value=float(2 * np.pi))
    elif motif == 'turn':
        angle = gr.update(visible=True, value=float(-np.pi / 2))
    else:
        angle = gr.update(visible=False)
    return (angle,
            gr.update(visible=motif == 'speed'),
            gr.update(visible=motif == 'weave'),
            gr.update(visible=motif == 'weave'))


def add_seg(segs, motif, duration, angle, speed, freq, amp):
    segs = list(segs or [])
    segs.append(dict(motif=motif, duration=float(duration), angle=float(angle),
                     speed=float(speed), freq=float(freq), amp=float(amp)))
    return segs, seg_text(segs)


def undo_seg(segs):
    segs = list(segs or [])
    if segs:
        segs.pop()
    return segs, seg_text(segs)


def clear_seg():
    return [], seg_text([])


# ─────────────────────── draw-a-trajectory recorder ─────────────────────────
# An HTML5 <canvas> (drawn/handled entirely in JS, see _APP_JS) captures the
# path: DRAG = freehand smooth stroke, CLICK = a waypoint. On pointer-up the JS
# serializes the points (in canvas-pixel coords) into a hidden textbox; Python
# reads that on "apply", maps pixels→arena data coords, and resamples the path
# into constant-speed (speed, heading) set-points.

_CANVAS_W, _CANVAS_H = 600, 360


def apply_drawing(draw_json, system, v0, dur):
    """ Parse the canvas JSON this app's own JS emits, then hand the pixels to
    core.canvas_to_recorded (shared with the Streamlit canvas). Stores the result
    as `recorded` and clears any motif segments so the drawn path takes
    precedence. """
    try:
        data = json.loads(draw_json) if draw_json else {}
    except (ValueError, TypeError):
        data = {}
    recorded, note = core.canvas_to_recorded(
        get_spec(system), data.get('pts', []), data.get('w', 1),
        data.get('h', 1), v0, dur)
    return recorded, note, [], seg_text([])


def clear_recorded():
    return None, '*Cleared — draw a new path.*'


# ─────────────────────────────── UI layout ──────────────────────────────────

APP_TITLE = 'Observability Explorer'
APP_TAGLINE = ('choose a system and motif, examine state observability, '
               'explore the relationship between observability and estimation')

# clean light theme
THEME = gr.themes.Soft(primary_hue='blue', neutral_hue='slate')

# App load JS: (1) lock light mode (matplotlib figures render on white), then
# (2) wire the drawing <canvas> — DRAG = freehand, CLICK = waypoint. On
# pointer-up it writes the points to the hidden #draw_data textbox for Python.
_APP_JS = """
() => {
  const u = new URL(location);
  if (u.searchParams.get('__theme') !== 'light') {
    u.searchParams.set('__theme', 'light'); location.href = u.href; return;
  }
  // Tutorial: glow the section(s) for the current step and scroll to them.
  const TUT_MAP = {0: ['sec_system', 'sec_traj'], 1: ['sec_mpc'],
                   2: ['sec_obs'], 3: ['sec_estimator'], 4: ['sec_plot'],
                   5: ['sec_reference']};
  window.__tutClear = () => document.querySelectorAll('.tut-highlight')
      .forEach(e => e.classList.remove('tut-highlight'));
  window.__tutHighlight = (s) => {
    window.__tutClear();
    let first = null;
    (TUT_MAP[s] || []).forEach(id => {
      const el = document.getElementById(id);
      if (el) { el.classList.add('tut-highlight'); if (!first) first = el; }
    });
    if (first) first.scrollIntoView({behavior: 'smooth', block: 'center'});
  };
  function setup() {
    const c = document.getElementById('draw_canvas');
    if (!c) { setTimeout(setup, 300); return; }
    if (c.dataset.ready) return; c.dataset.ready = '1';
    try {
      const ctx = c.getContext('2d');
      let pts = [], drawing = false;
      const grid = () => {
        ctx.clearRect(0, 0, c.width, c.height);
        ctx.lineWidth = 1; ctx.strokeStyle = '#eef2f7';
        for (let i = 1; i < 8; i++) { const x = c.width*i/8;
          ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, c.height); ctx.stroke(); }
        for (let i = 1; i < 6; i++) { const y = c.height*i/6;
          ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(c.width, y); ctx.stroke(); }
        ctx.strokeStyle = '#dbe2ea';
        ctx.beginPath(); ctx.moveTo(0, c.height/2); ctx.lineTo(c.width, c.height/2); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(c.width/2, 0); ctx.lineTo(c.width/2, c.height); ctx.stroke();
        ctx.fillStyle = '#94a3b8'; ctx.font = '15px sans-serif'; ctx.textAlign = 'center';
        ctx.fillText('drag to draw a path  ·  click to add waypoints', c.width/2, 22);
      };
      const redraw = () => {
        grid();
        if (!pts.length) return;
        ctx.strokeStyle = '#2563eb'; ctx.lineWidth = 2.5; ctx.beginPath();
        pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
        ctx.stroke();
        ctx.fillStyle = '#2563eb';
        pts.forEach(p => { ctx.beginPath(); ctx.arc(p[0], p[1], 2.5, 0, 7); ctx.fill(); });
        ctx.fillStyle = '#16a34a';
        ctx.beginPath(); ctx.arc(pts[0][0], pts[0][1], 5, 0, 7); ctx.fill();
      };
      const P = e => { const r = c.getBoundingClientRect();
        return [(e.clientX - r.left) * c.width / r.width,
                (e.clientY - r.top) * c.height / r.height]; };
      const commit = () => {
        const el = document.querySelector('#draw_data textarea, #draw_data input');
        if (el) { el.value = JSON.stringify({pts: pts, w: c.width, h: c.height});
                  el.dispatchEvent(new Event('input', {bubbles: true})); }
      };
      c.addEventListener('pointerdown', e => { drawing = true; pts.push(P(e)); redraw();
        try { c.setPointerCapture(e.pointerId); } catch (_) {} });
      c.addEventListener('pointermove', e => { if (drawing) { pts.push(P(e)); redraw(); } });
      c.addEventListener('pointerup', () => { drawing = false; commit(); });
      c.addEventListener('pointerleave', () => { if (drawing) { drawing = false; commit(); } });
      window.__clearDraw = () => { pts = []; redraw(); commit(); };
      grid();
    } catch (err) {
      const ctx = c.getContext('2d');
      ctx.fillStyle = 'crimson'; ctx.font = '12px monospace'; ctx.textAlign = 'left';
      ctx.fillText('draw JS error: ' + err.message, 10, 30);
    }
  }
  setup();
}
"""

# blue boxed headers for the two trajectory sub-sections; hide the JSON bridge;
# .tut-highlight = the glowing outline the tutorial puts on the active section
CSS = ('.blue-hdr { border: 1px solid var(--primary-500) !important; '
       'border-radius: 8px; }'
       '.blue-hdr > .label-wrap, .blue-hdr > .label-wrap span '
       '{ color: var(--primary-600) !important; font-weight: 600; }'
       '#draw_data { display: none !important; }'
       '.tut-highlight { outline: 3px solid var(--primary-500) !important; '
       'outline-offset: 3px; border-radius: 8px; '
       'box-shadow: 0 0 0 4px rgba(37,99,235,0.22) !important; }')

# render LaTeX ($…$ inline, $$…$$ display) inside the reference/equation Markdown
LATEX_DELIMS = [{'left': '$$', 'right': '$$', 'display': True},
                {'left': '$', 'right': '$', 'display': False}]

HERE = os.path.dirname(os.path.abspath(__file__))
TUT_DIR = os.path.join(HERE, 'tutorial_imgs')
# each step: image (or None), image height, caption, and which accordion keys to
# open. The glowing highlight targets are mapped by step index in _APP_JS.
TUT_ACC_KEYS = ['system', 'traj', 'obs', 'estimator', 'plot', 'reference']
TUT_STEPS = [
    dict(img=os.path.join(TUT_DIR, 'step1.png'), h=190,
         title='Step 1 — Choose a system & trajectory',
         body='Pick a **system** (or upload your own `f`, `h`). Then, make a '
              '**state trajectory** X. You can build one from motif segments, draw a '
              'path, or upload measured (x, y) data of your own. ',
         open={'system', 'traj'}),
    dict(img=os.path.join(TUT_DIR, 'mpc.png'), h=190,
         title='Step 2 — Reconstruct inputs with MPC',
         body='Since we have the dynamics and physical trajectory, we can '
              'recover the control inputs that move the system along the nominal trajectory. \\'  
            'Press **Simulate**. A model-predictive controller drives the '
              'prediction model $x_{k+1}=f(x_k,u_k)$ to follow your trajectory, '
              'recovering the control inputs U and a dynamically-consistent '
              'full-state trajectory to analyze.',
         open={'traj'}),
    dict(img=os.path.join(TUT_DIR, 'observability.png'), h=250,
         title='Step 3 — Investigate observability',
         body='Choose the sensor/state combination that you want for your system, '
              'empirical perturbation value ($\epsilon$), window size for each calculation ($w$), and noise '
              'levels R (and, optionally, Q). This pipeline perturbs each state by $\epsilon$, builds the '
              'observability matrix $\\mathcal{O}$, the Fisher information '
              '$\\mathcal{F}=\\mathcal{O}^{\\mathsf T}R^{-1}\\mathcal{O}$, and '
              'its inverse — whose diagonal is the minimum error variance of '
              'each state.',
         open={'obs'}),
    dict(img=os.path.join(TUT_DIR, 'kf_simple.png'), h=190,
         title='Additional functionality - Estimators: ',
         body='See how well an estimator (**EKF** / **UKF**) converges to the '
              'true state, with ±2σ bands in the **States** plot. '
              '**Measurement** plots (with and without noise) can also be viewed. Each filter has its own '
              'tuning noise (and the UKF its sigma-point parameters).',
         open={'estimator'}),
    dict(img=None, h=190,
         title='Plot settings',
         body='Adjust the visuals here: trajectory coloring & colormap, which '
              'states/measurements to show, and the sliding-window index for the '
              'matrices.',
         open={'plot'}),
    dict(img=None, h=190,
         title='Reference & equations',
         body='Look here for variable **names/definitions**, each system’s '
              'states & measurements, noise units, and the equations behind '
              'every plot and the estimators.',
         open={'reference'}),
]

# ─────────────────────── reference / equation content ───────────────────────
# Variable definitions live in the section where their equation is introduced.

# ---- Systems: system-level symbols + measurement/process noise (R, Q) --------
EQ_SYS_INTRO = r"""
A system is a dynamics model $x_{k+1}=f(x_k,u_k)$ and a measurement model
$y_k=h(x_k,u_k)$.

| symbol | meaning |
|---|---|
| $x$ | **state** vector — the quantities we want to estimate |
| $u$ | **inputs** (controls) |
| $y$ | **measurements** (sensor outputs) |
| $f$ | **dynamics** / prediction model |
| $h$ | **measurement** model |

**Measurement & process noise**
- **R** — measurement-noise covariance: a *per-sensor variance*; units =
  (sensor unit)². A heading sensor in rad → R in rad².
- **Q** — process-noise / model-uncertainty covariance **per step**; units =
  (state unit)². Q = 0 → deterministic; larger Q → more model uncertainty. How
  the entered numbers become the per-step $Q_k$ is set by **noise type** —
  *uncorrelated* (default) or *correlated (Van Loan)* — see
  *Observability → Fisher information → stochastic*. The empirical pipeline has
  no Q at all; both Gramian bases and the EKF/UKF use whichever setting is
  selected, so comparisons stay like-for-like.
"""

# ---- Observability: matrix, Fisher information, min error variance -----------
EQ_OBS_INTRO = r"""
Observability quantifies how the measurements over a short **sliding window**
respond to the state at the window's start — the basis for how well each state
can be inferred.

| symbol | meaning |
|---|---|
| $\mathcal{O}$ | observability matrix (rows = sensor × time-step, cols = states) |
| $w$ | sliding-window length (number of time-steps) |
"""

EQ_OBSMAT_EMP = r"""
**Empirical (Q = 0, pybounds).** Central finite difference of the measurements
under a small perturbation $\varepsilon$ of each initial state:
$$\mathcal{O}_{(i,k),\,j}=\frac{\partial y_i(t_k)}{\partial x_j(t_0)}\approx\frac{y_i^{+}-y_i^{-}}{2\varepsilon}.$$

- $\varepsilon$ — finite-difference perturbation size.

Reference: Cellini et al., *BOUNDS* — <https://doi.org/10.48550/arXiv.2511.08766>
"""

EQ_OBSMAT_STOCH = r"""
**Stochastic (analytic linearization, Q-independent).**
$$\mathcal{O}=\begin{bmatrix}C_0\\ C_1\Phi_{1,0}\\ \vdots\\ C_{w-1}\Phi_{w-1,0}\end{bmatrix}.$$

- $C_k=\left.\partial h/\partial x\right|_{x_k}$ — measurement Jacobian.
- $\Phi$ — discrete state-transition matrix ($\partial f/\partial x$, Van Loan).

Reference: Boyacıoğlu & van Breugel, *Duality of Stochastic Observability…*
(IEEE L-CSS) — <https://doi.org/10.1109/LCSYS.2025.3547297>
"""

EQ_FISHER_INTRO = r"""
The **Fisher information** $\mathcal{F}$ accumulates the window's information
about the initial state; its inverse lower-bounds the estimation error
covariance.
"""

EQ_FISHER_EMP = r"""
**Empirical (Q = 0).**
$$\mathcal{F}=\mathcal{O}^{\mathsf T} R^{-1}\mathcal{O}.$$

- $R$ — measurement-noise covariance (see *Systems*).

Reference: <https://doi.org/10.48550/arXiv.2511.08766>
"""

EQ_FISHER_STOCH = r"""
**Stochastic (process noise Q > 0).** The recursive stochastic observability
Gramian (Eq. 33) is used instead of stacking $\mathcal{O}$:
$$\mathcal{F}_j=\Phi_j^{\mathsf T}\big(Q_{d,j}+\mathcal{F}_{j+1}^{-1}\big)^{-1}\Phi_j+C_j^{\mathsf T}R^{-1}C_j,\qquad \mathcal{F}_{w-1}=C_{w-1}^{\mathsf T}R^{-1}C_{w-1},$$
which reduces to $\mathcal{F}=\mathcal{O}^{\mathsf T}R^{-1}\mathcal{O}$ as $Q\to0$.

- $Q_k$ — per-step discrete process covariance (units²), built from the entered
  Q in one of two ways, selected with **noise type** in
  *Investigate observability → process noise*:

  **uncorrelated** (default) — $Q_k=\operatorname{diag}(q)$: the entered numbers
  used verbatim, the same matrix at every step. Noise on one state never leaks
  into another, and $Q_k$ carries no trajectory dependence. This is the
  convention the recursions are written in — they consume a per-step covariance
  directly, so nothing is scaled by $dt$.

  **correlated (Van Loan)** — treat the entered numbers as a spectral density
  $Q_c=\operatorname{diag}(q)/dt$ and discretize it through the *local*
  dynamics,
  $$Q_k=\int_0^{dt}e^{A_k\tau}\,Q_c\,e^{A_k^{\mathsf T}\tau}\,d\tau .$$
  Over one step the dynamics mix the states, so $Q_k$ gains off-diagonal terms —
  noise on $\dot\phi$ leaks into $\phi$, velocity noise into position — and it
  varies along the trajectory through $A_k$. This is what System (22) means by a
  time-varying $Q_k$; the recursions have always indexed $Q_k$ per step, and
  this is the setting that actually exercises it.

  A state with no dynamics of its own ($\dot\zeta=0$, $\dot w=0$) is unaffected
  either way: its diagonal entry is exactly the number you entered and its cross
  terms stay zero, so the **wind entries mean the same thing under both
  settings** and only the coupled states change.

**FIM basis (observability vs constructability).** The recursion above is the
*observability* Gramian — it bounds the window's **initial** state $x_0$ (a
fixed-point smoother). Its dual, the *constructability* Gramian, bounds the
**final/current** state $x_{w-1}$ via the forward recursion
$$\mathcal F_{k+1}=\big(Q_{d,k}+\Phi_{k}\mathcal F_k^{-1}\Phi_{k}^\top\big)^{-1}+C_{k+1}^\top R^{-1}C_{k+1}.$$
Because a Kalman filter estimates the *current* state from past+present data,
the **constructability** min-EV lines up with the EKF/UKF error covariance far
better than observability does. Tick either basis — or both, to compare them —
in *Investigate observability → process noise*; each adds its own trajectory
panel and min-EV curve.

Both bases are plotted at each window's **center** time, so the curves share one
x-axis. Keep in mind what that does *not* mean: the state each bound refers to
still differs — observability the window's first state, constructability its
last — so the same feature appears at the same x in both curves even though it
physically occurred up to $w\,dt$ apart. Compare *levels*, not exact timing.

Reference: <https://doi.org/10.1109/LCSYS.2025.3547297>
"""

EQ_ANN = r"""
The **ANN** is $\mathcal H_i$ of Eq. 6 in Cellini et al.: a feed-forward
regressor mapping the measurements — augmented with an $\omega$-step time history
— onto a single state,
$$\check{x}_{i,k}=\mathcal H_i\big(\mathbf y_k,\dots,\mathbf y_{k-\omega},
\mathbf u_k,\dots,\mathbf u_{k-\omega}\big).$$

- ReLU hidden layers, linear output; default shape $8\!+\!16\!+\!8$ — the paper's
  best small net. A single 8-neuron layer is deliberately *too weak*, which is
  what makes the AI-KF gate worth having.
- For a **circular** state the paper's custom loss is used,
  $f(\zeta,\check\zeta)=(\sin\zeta-\sin\check\zeta)^2+(\cos\zeta-\cos\check\zeta)^2$,
  realized by predicting the $(\sin,\cos)$ pair and recovering the angle with
  $\mathrm{atan2}$ — so the net never has to represent the $\pm\pi$ jump.
- Inputs default to heading $\phi$, airflow angle $\gamma$ and course angle
  $\psi$ (the paper's Figure 6 choice). Airspeed and groundspeed are withheld on
  purpose: together they nearly hand you the wind vector.
- Trained on randomized trajectories drawn from the paper's parameter ranges,
  with a Gaussian input-noise layer active only during training. It never sees
  the trajectory being estimated. Nets are cached after the first use.
- The ANN has **no covariance**, so its trace carries no $\pm3\sigma$ band.
"""

EQ_AIKF = r"""
The **AI-KF** is the authors' repo estimator (`util/StateEstimator.m`): a
**UKF** on the real dynamics whose measurement vector is augmented with the
ANN's estimate as a pseudo-measurement, whose noise for that channel is set by
an active-sensing motif,
$$R_{\mathrm{ANN},k}=\mathrm{logmap}\big(\overline{|u|}_{k-w:k},\,
[0,\,u_{\max}],\,[R_{hi},R_{lo}]\big).$$
No motif → $R=10^{12}$ (the ANN is ignored and the filter coasts on its
dynamics); motif at or above the bound → $R=10^{-3}$ (the ANN is trusted).
Setting $R_{lo}=R_{hi}$ disables the channel and recovers the plain UKF — the
repo's own ablation (`r_sweep = [12, -3]`). It is UKF-based rather than
EKF-based deliberately: with $h=v_x/z$ an EKF base is far less stable (on the
$z_0=9$ initial-condition case, EKF error 2974 vs UKF 27).

Its sigma-point tuning is shared with the UKF section, defaulting to the repo's
$\alpha=10^{-3},\ \beta=1,\ \kappa=0$.

---

The **MIF** (a different scheme, from the earlier bioRxiv paper) is also
implemented in the engine as `run_ann_mif`: a one-line
complementary filter whose gain is set by **observability**, fusing the raw ANN
estimate with its own previous output (Eq. 5),
$$\hat{x}_{i,k}=\beta_k\,\check{x}_{i,k}+(1-\beta_k)\,\hat{x}_{i,k-1},$$
with the gain read off the sliding-window inverse Fisher information (Eq. 7)
$$\beta_k=\Big[\log_{10}\big(10+[\mathcal F_{\omega_k}^{-1}]_{i,i}\big)\Big]^{-1}.$$

Where the state is well observed, min-EV is small, $\beta\to1$ and the filter
takes the ANN at its word; where it is unobservable, min-EV explodes,
$\beta\to0$ and the filter **coasts** on its last estimate instead of tracking
ANN noise. $\beta$ is drawn in green on the right-hand axis of the *States* plot.

- $\beta$ uses the **constructability** Gramian, not observability: it is the
  *current* state being estimated, and the paper notes constructability is what
  statistical consistency calls for.
- $\beta$ is computed from the **same sensor set the ANN sees**. Using every
  measurement instead makes $\zeta$ look permanently well observed
  ($\beta\approx0.99$ throughout) and the gate stops doing anything.
- A circular state is blended on the unit circle — averaging raw angles would let
  a $+\pi$ / $-\pi$ pair cancel to $0$.
- Note $\beta$ never quite reaches $0$: Eq. 7 gives $\beta=1$ at min-EV $=0$ and
  decays only logarithmically ($\beta\approx0.17$ at min-EV $=10^6$).

Reference: Cellini, Boyacıoğlu, Stupski & van Breugel, bioRxiv
[2024.11.04.621976](https://doi.org/10.1101/2024.11.04.621976), Figure 6.
"""

EQ_MINEV = r"""
The **minimum error variance** (Cramér–Rao bound), Chernoff-regularized:
$$\mathrm{EV}=\mathrm{diag}\big[(\mathcal{F}+\lambda I)^{-1}\big].$$

- $\lambda$ — **Chernoff regularizer**: a small ridge added before inverting
  $\mathcal{F}$, so unobservable directions report a large-but-finite variance
  $\approx 1/\lambda$ instead of ∞.

The *min-EV matrix* stacks this per sliding window; the *$\mathcal{F}^{-1}$ map*
shows $|(\mathcal{F}+\lambda I)^{-1}|$ for one window (its diagonal = that
window's EV, on the same color scale).
"""

# ---- Estimators: EKF, UKF ----------------------------------------------------
EQ_EST_INTRO = r"""
Estimators recover the state from noisy measurements
$y_k=h(x_k,u_k)+v_k,\ v_k\sim\mathcal N(0,R_k)$ with process model
$x_{k+1}=f(x_k,u_k)+w_k,\ w_k\sim\mathcal N(0,Q_k)$. Both filters use the **same R and
Q as the observability analysis**; the Estimator controls only shape convergence
— the initial-estimate error/offset and an optional injected input disturbance
over a time window (paper Fig. 4b). The ±2σ bands in the *States* plot are
$2\sqrt{\mathrm{diag}\,P}$: where a state is unobservable the band stays wide,
mirroring a high min-EV.
"""

EQ_KF_CYCLE = r"""
Both filters run the same two-step cycle at every sample: **predict** the state
forward with the process model, then **correct** it with the measurement. The
supplied initial estimate is the **prior** at $k=0$ ($\hat x_{0|-1}$), so the
cycle is entered at the correction step and $y_0$ is used. That keeps each
filter aligned with the constructability Gramian, which starts the same way
($\mathcal F_0=C_0^{\mathsf T}R^{-1}C_0$) — the reason its min-EV curve is the
one comparable to a filter's $P$. (The BOUNDS reference driver instead predicts
first and drops $y_0$.)

They differ only in *how* the two steps are carried out: the EKF linearizes
$f,h$ and pushes a covariance through the Jacobians, while the UKF pushes a set
of sigma points through $f,h$ themselves. See the two subsections.
"""

EQ_EKF = r"""
The **Extended Kalman Filter** linearizes $f,h$ about the current estimate each
step — $\Phi_k=\partial f/\partial x$ (the one-step map, so $\Phi$ not $A$) and
$C_k=\partial h/\partial x$, both by central differences. Using the notation of
the diagram above, $^-$ is *prior* (before the measurement) and $^+$ is
*posterior*, the cycle at step $k$ is:

*Measurement update* — the cycle is entered here at $k=0$, so $y_0$ is used
$$S_k=C_kP^{-}_kC_k^{\mathsf T}+R_k,\qquad K_k=P^{-}_kC_k^{\mathsf T}S_k^{-1},$$
$$\hat{x}^{+}_k=\hat{x}^{-}_k+K_k\big(y_k-h(\hat{x}^{-}_k,u_k)\big),$$
$$P^{+}_k=(I-K_kC_k)P^{-}_k(I-K_kC_k)^{\mathsf T}+K_kR_kK_k^{\mathsf T}.$$

*Time update*
$$\hat{x}^{-}_{k+1}=f\big(\hat{x}^{+}_k,u_k\big),\qquad
P^{-}_{k+1}=\Phi_k\,P^{+}_k\,\Phi_k^{\mathsf T}+Q_k.$$

- $P$ — state-estimate covariance.  $K_k$ — Kalman gain.  $S_k$ — innovation
  covariance.
- The covariance update is the **Joseph form**, which stays symmetric positive
  definite where $(I-K_kC_k)P^{-}_k$ can drift indefinite.
- Innovations on angular channels are wrapped to $(-\pi,\pi]$ before use.
- Put $\mathcal F^{c}_k=(P^{+}_k)^{-1}$ and the time+measurement updates *are*
  the constructability recursion — same $\Phi$, same
  $C^{\mathsf T}R^{-1}C$, same additive $Q$. That is why the constructability
  min-EV is the one that lines up with this filter.
"""

EQ_UKF = r"""
The **Unscented Kalman Filter** runs the same cycle but propagates
$2n_x{+}1$ sigma points through $f,h$ instead of forming $\Phi_k,C_k$ — more
accurate under strong nonlinearity. With $n_x$ states the points sit at
$\hat{x}\pm[\sqrt{(n_x+\lambda_s)P}\,]_i$, i.e.
$\pm\sqrt{\alpha^{2}(n_x+\kappa)}\,\sigma$ about the mean, where
$\lambda_s=\alpha^{2}(n_x+\kappa)-n_x$ is the scaling parameter (unrelated to
the regularizer $\lambda$):

- $\alpha$ — **spread** of the sigma points about the mean.
- $\beta$ — **prior** knowledge of the distribution ($\beta=2$ is optimal for Gaussians).
- $\kappa$ — **secondary scaling** (usually 0).
"""

_MEAS_GLOSS = {'phi': 'heading [rad]',
               # fly: course angle; drone: yaw observed directly
               'psi': 'course angle atan2(v⊥,v∥) [rad] — fly; yaw [rad] — drone',
               'gamma': 'air-velocity angle [rad]', 'a': 'airspeed [m/s]',
               'g': 'groundspeed [m/s]', 'r': 'optic flow g/z [1/s]',
               'r_x': 'range/optic-flow x', 'r_y': 'range/optic-flow y',
               'beta': 'sideslip angle [rad]',
               'q_x': 'body-rate x', 'q_y': 'body-rate y', 'q_z': 'body-rate z'}
_STATE_GLOSS = {
    'x': 'x position [m]', 'y': 'y position [m]', 'z': 'altitude [m]',
    'v_para': 'parallel body velocity [m/s]',
    'v_perp': 'perpendicular body velocity [m/s]',
    'v_x': 'x velocity [m/s]', 'v_y': 'y velocity [m/s]',
    'v_z': 'z velocity [m/s]', 'phi': 'heading / roll [rad]',
    'phi_dot': 'angular velocity [rad/s]', 'theta': 'pitch [rad]',
    'psi': 'yaw [rad]', 'w': 'ambient wind speed [m/s]',
    'zeta': 'ambient wind direction [rad]', 'm': 'mass [kg]',
    'I': 'moment of inertia [kg·m²]', 'C_para': 'parallel damping [N·s/m]',
    'C_perp': 'perpendicular damping [N·s/m]',
    'C_phi': 'rotational damping [N·m·s/rad]',
    'km1': 'motor calibration coeff. 1', 'km2': 'motor calibration coeff. 2',
    'km3': 'motor calibration coeff. 3', 'km4': 'motor calibration coeff. 4',
    'g': 'ground speed [m/s]', 'd': 'camera distance to ground [m]'}
_SYS_DESC = {
    'fly7': 'fly (simple) — the 7 primary states: position, body-frame '
            'velocities, heading & rate, and ambient wind (speed & direction). '
            'Mass, inertia, damping and motor coefficients are fixed '
            'parameters rather than states, which keeps Q genuinely positive '
            'definite without a jitter floor. The default system.',
    'fly': 'fly (full state) — the same dynamics with the calibration '
           'parameters promoted to estimated states (18 in total: the pybounds '
           'example). Mass, inertia, the three damping coefficients and four '
           'motor coefficients are all unknown, so they need a small process-'
           'noise floor to keep the recursions invertible.',
    'drone': '3D kinematic quadcopter in ambient wind: position, body '
             'velocities, attitude (phi, theta, psi) and wind (w, zeta).',
    'alt2d': 'Minimal 2D altitude system (paper Fig. 4b): horizontal/vertical '
             'position & velocity with a single range-type measurement.'}


# Reference diagram per system. The *_web.png files are downscaled copies of the
# originals (which are 6k-12k px wide / up to 11 MB — far too big to ship to a
# browser on every page load).
SYS_IMG = {'fly': 'fly_system_web.png', 'fly7': 'fly_system_web.png',
           'drone': 'drone_system_web.png', 'alt2d': 'drone_system_2d_web.png'}

SYS_LIST = [('fly7', 'fly (simple)'), ('fly', 'fly (full state)'),
            ('drone', 'drone (3D)'), ('alt2d', 'altitude 2D'),
            ]


def _system_md(name):
    """ Markdown reference for one built-in system: description + a state
    definition table + inputs + measurements. """
    s = SYSTEMS[name]()
    st_rows = '\n'.join(
        f"| `{x}` | {_STATE_GLOSS.get(x, '—')} |" for x in s.state_names)
    meas = ', '.join(f'`{m}`' + (f' ({_MEAS_GLOSS[m]})' if m in _MEAS_GLOSS
                                 else '') for m in s.measurement_names)
    return (f"{_SYS_DESC[name]}\n\n"
            f"**states** ({len(s.state_names)}):\n\n"
            f"| state | definition |\n|---|---|\n{st_rows}\n\n"
            f"- **inputs**: {', '.join(f'`{u}`' for u in s.input_names)}\n"
            f"- **measurements**: {meas}\n")

_S0 = get_spec('fly7')     # defaults for the initial system, fly (simple)


def open_tutorial():
    return _tutorial_view(0)


def _tutorial_view(step):
    """ Show the tutorial overlay at `step`: set the image + caption, advance the
    step state, and open the section(s) that step points at. The glowing
    highlight itself is applied client-side by window.__tutHighlight. """
    step = max(0, min(int(step), len(TUT_STEPS) - 1))
    s = TUT_STEPS[step]
    cap = f"### {s['title']}  ·  {step + 1}/{len(TUT_STEPS)}\n\n{s['body']}"
    # show the image only if the file actually exists (some are added later)
    img = s['img'] if (s['img'] and os.path.exists(s['img'])) else None
    img_u = gr.update(value=img, height=s['h'], visible=img is not None)
    opens = [gr.update(open=(k in s['open'])) for k in TUT_ACC_KEYS]
    return (gr.update(visible=True), img_u, cap, step, *opens, str(step))

with gr.Blocks(title=APP_TITLE) as demo:   # theme/js/css passed at launch (Gradio 6)
    sess = gr.State({})
    segs = gr.State([])
    recorded = gr.State(None)      # applied drawn path: (speed_sp, heading_sp)
    custom_spec = gr.State(None)   # uploaded custom system (SystemSpec) or None
    phys_traj = gr.State(None)     # uploaded physical trajs {id: (x, y, speed, heading)}
    with gr.Row():
        gr.Markdown(f'# {APP_TITLE}\n{APP_TAGLINE}')
        btn_tutorial = gr.Button('📖 Tutorial', scale=0, min_width=150)

    # tutorial overlay (hidden until the button is pressed): a paper Fig-1 step
    # image + caption that also opens & glows the matching control section.
    # tut_hl is a hidden bridge — its .change reliably fires the client-side
    # highlight with the current step (more robust than a js-only .then).
    tut_step = gr.State(0)
    tut_hl = gr.Textbox('', visible=False, elem_id='tut_hl')
    with gr.Column(visible=False, elem_id='tutorial_panel') as tutorial_panel:
        with gr.Row():
            tut_img = gr.Image(TUT_STEPS[0]['img'], show_label=False,
                               container=False, height=190)
        tut_cap = gr.Markdown(latex_delimiters=LATEX_DELIMS)
        with gr.Row():
            btn_tut_prev = gr.Button('← Prev')
            btn_tut_next = gr.Button('Next →', variant='primary')
            btn_tut_close = gr.Button('✕ Close', variant='stop')

    with gr.Row():
        # ------------------------- controls ----------------------------------
        with gr.Column(scale=2, min_width=360):
            with gr.Accordion('Choose your system', open=True,
                              elem_id='sec_system') as acc_system:
                system = gr.Dropdown(SYSTEM_LABELS, value='fly7', label='system')
                sys_file = gr.File(label='upload a system .py (defines f, h, '
                                   'state_names, measurement_names, dt; add '
                                   'position_states + input_bounds for MPC)',
                                   file_types=['.py'], type='filepath',
                                   visible=False)   # shown when system = Custom

            with gr.Accordion('Build your trajectory', open=True,
                              elem_id='sec_traj') as acc_traj:
                with gr.Row():
                    with gr.Column(scale=2, min_width=280):
                        with gr.Accordion('Trajectory', open=False,
                                          elem_classes='blue-hdr'):
                            btn_default = gr.Button('reset', size='sm')
                            with gr.Accordion('Build a trajectory', open=False):
                                gr.Markdown('*Add motif segments in sequence.*')
                                v0 = gr.Number(value=float(_S0.v0_default),
                                               label='initial speed v₀ [m/s]')
                                motif = gr.Dropdown(MOTIFS, value='turn',
                                                    label='motif')
                                p_duration = gr.Number(value=float(_S0.seg_duration),
                                                       label='duration [s]')
                                p_angle = gr.Number(value=float(-np.pi / 2),
                                                    label='turn angle [rad]',
                                                    visible=True)
                                p_speed = gr.Number(value=float(_S0.v0_default),
                                                    label='target speed [m/s]',
                                                    visible=False)
                                p_freq = gr.Number(value=2.0,
                                                   label='weave frequency [Hz]',
                                                   visible=False)
                                p_amp = gr.Number(value=float(np.pi / 6),
                                                  label='weave amplitude [rad]',
                                                  visible=False)
                                with gr.Row():
                                    btn_add = gr.Button('+ add')
                                    btn_undo = gr.Button('undo')
                                    btn_clear = gr.Button('clear')
                                seg_display = gr.Markdown(seg_text([]))
                            with gr.Accordion('Draw a trajectory', open=False):
                                gr.Markdown('*Drag to draw a smooth path, or '
                                            'click to drop waypoints.*')
                                gr.HTML(
                                    f'<canvas id="draw_canvas" width="{_CANVAS_W}" '
                                    f'height="{_CANVAS_H}" style="width:100%;'
                                    'border:1px solid #cbd5e1;border-radius:8px;'
                                    'background:#fff;touch-action:none;'
                                    'cursor:crosshair;display:block;"></canvas>')
                                draw_data = gr.Textbox(value='', elem_id='draw_data')
                                draw_dur = gr.Number(
                                    value=0.0,
                                    label='path duration [s]  (0 = constant v₀; '
                                          'set a value to cap the step count / '
                                          'solve time on long paths)')
                                with gr.Row():
                                    btn_draw_apply = gr.Button('apply', size='sm')
                                    btn_draw_clear = gr.Button('clear', size='sm')
                                rec_status = gr.Markdown('')
                            with gr.Accordion('Upload a physical trajectory '
                                              '(x, y → MPC)', open=False):
                                gr.Markdown('*Real measured data: position (and '
                                            'velocity) over time. We derive speed '
                                            '+ course and reconstruct the full '
                                            'state via MPC on the built-in model '
                                            '(set your hypothesized wind below).*')
                                phys_file = gr.File(
                                    label='data file (csv / parquet / hdf)',
                                    file_types=['.csv', '.parquet', '.pq', '.h5', '.hdf', '.hdf5'],
                                    type='filepath')
                                with gr.Row():
                                    x_col = gr.Textbox('x', label='x column')
                                    y_col = gr.Textbox('y', label='y column')
                                id_col = gr.Textbox('obj_id_unique',
                                                    label='trajectory-id column (blank = whole file)')
                                obj_dd = gr.Dropdown([], label='trajectory')
                                phys_status = gr.Markdown('')
                            dt_hz = gr.Dropdown(DT_HZ, value=_BASE_HZ['fly7'],
                                                label='sampling rate')
                    with gr.Column(scale=1, min_width=160):
                        with gr.Accordion('External stimuli', open=False,
                                          elem_classes='blue-hdr'):
                            with gr.Accordion('Wind', open=False):
                                wind = gr.Number(value=float(_S0.wind_default),
                                                 label='speed [m/s]')
                                zeta = gr.Number(value=float(_S0.zeta_default),
                                                 label='direction ζ [rad]')

                btn_sim = gr.Button('▶ simulate (MPC)', variant='primary',
                                    elem_id='sec_mpc')

            with gr.Accordion('Investigate observability', open=True,
                              elem_id='sec_obs') as acc_obs:
                with gr.Accordion('Choose sensor & state combination', open=False):
                    gr.Markdown('*Sensors (rows of the observability matrix 𝒪) '
                                'and states (columns) used to build 𝒪 and the '
                                'FIM.*')
                    fim_sensors = gr.CheckboxGroup(
                        list(_S0.measurement_names), value=list(_S0.fim_sensors),
                        label='sensors')
                    fim_states = gr.CheckboxGroup(
                        list(_S0.state_names), value=list(_S0.fim_states),
                        label='states')

                with gr.Accordion('Set params', open=False):
                    w = gr.Slider(3, 60, value=_S0.w_default, step=1,
                                  label='window w [steps]')
                    lam = gr.Dropdown(
                        [(f'{v:.0e}', v) for v in LAM_VALS],
                        value=min(LAM_VALS, key=lambda v: abs(v - 1e-6)),
                        label='λ (regularizer)')
                    eps = gr.Dropdown(
                        [(f'{v:.1e}', v) for v in EPS_VALS],
                        value=min(EPS_VALS, key=lambda v: abs(v - _S0.eps_default)),
                        label='ε (empirical perturbation)')

                with gr.Accordion('Set noise levels', open=False):
                    with gr.Accordion('Measurement noise (R)', open=False):
                        r_mode = gr.Dropdown(['uniform', 'custom'],
                                             value='uniform', label='noise type')
                        r_uniform = gr.Number(value=_default_r(_S0),
                                              label='R for all sensors')
                        r_table = gr.Dataframe(
                            headers=['sensor', 'R'], datatype=['str', 'number'],
                            type='array', value=_r_rows(_S0), interactive=True,
                            visible=False, label='per-sensor R')

                    with gr.Accordion('Process noise / model uncertainty (Q)',
                                      open=False):
                        gr.Markdown(
                            'The empirical (pybounds) pipeline does **not** '
                            'incorporate process noise. To include it we add the '
                            'stochastic observability gramian (Eq. 33, Boyacıoğlu '
                            '& van Breugel, IEEE L-CSS 2025, '
                            'doi:10.1109/LCSYS.2025.3547297) as a second '
                            'pipeline, so its min error variance can be compared '
                            'against the empirical one. **If you include process '
                            'noise, Q must be non-zero**, otherwise the recursion '
                            'hits a singular-matrix error.\n\n'
                            'Pick either FIM basis, or both to compare them '
                            'directly — each selected basis adds its own '
                            'trajectory panel and min-EV curve.')
                        do_stoch = gr.Checkbox(
                            value=False,
                            label='include process noise (stochastic gramian)')
                        q_obs = gr.Checkbox(
                            value=True, visible=False,
                            label='observability basis (initial state)')
                        q_con = gr.Checkbox(
                            value=False, visible=False,
                            label='constructability basis (final state)')
                        q_mode = gr.Dropdown(['uniform', 'custom'],
                                             value='uniform',
                                             label='Q values', visible=False)
                        q_noise = gr.Dropdown(
                            [('uncorrelated', 'uncorrelated'),
                             ('correlated (Van Loan)', 'vanloan')],
                            value='uncorrelated', label='noise type',
                            visible=False,
                            info='uncorrelated: Q_k = the entered diagonal, '
                                 'identical at every step. correlated: Q_k is '
                                 'discretised through the local dynamics, so it '
                                 'gains off-diagonal terms and varies along the '
                                 'trajectory. See Reference & equations.')
                        q_uniform = gr.Number(value=1e-4,
                                              label='Q for all states',
                                              visible=False)
                        q_table = gr.Dataframe(
                            headers=['state', 'Q'], datatype=['str', 'number'],
                            type='array', value=_q_rows(_S0), interactive=True,
                            visible=False, label='per-state Q')

            with gr.Accordion('Estimators', open=False,
                              elem_id='sec_estimator') as acc_estimator:
                gr.Markdown(
                    '*How well each estimator converges to the true state. They '
                    'all use the **same R and Q as the observability analysis** '
                    'above, but each is initialized **independently** in its own '
                    'section below: its own noise seed, initial-estimate error, '
                    'P₀ and injected disturbance. Matching seeds put two filters '
                    'on identical data, which is what makes a head-to-head '
                    'comparison meaningful; differing seeds compare them on '
                    'different data instead.*')
                _INJ_MD = (
                    '*Bias added to one **measured acceleration** between t₀ '
                    'and t₁, plus optional zero-mean noise on those readings '
                    'throughout — probes how observability affects '
                    'reconvergence. The accelerations are IMU readings, but a '
                    'filter consumes them in its **prediction** step as '
                    'process-model inputs (u_z, u_x) rather than in the '
                    'correction step — exactly as the paper\'s own code does '
                    '(`U = [u_z_noise, u_x_noise]`) — so they are perturbed '
                    'here. The truth trajectory always flies the clean '
                    'accelerations.*')
                gr.Markdown(
                    '**R and Q come from the Observability section above** — the '
                    'estimators are deliberately given the same measurement and '
                    'process noise as the bound they are being compared against.\n\n'
                    '*If **process noise** is not switched on there, Q falls back to '
                    '10⁻⁹ on every state for the filters, because an EKF/UKF needs a '
                    'strictly positive Q (a singular one makes P collapse and the '
                    'filter stop listening to the data). That fallback is a floor, '
                    'not a tuning — set Q in the Observability section to control it.*')
                _X0_MD = ('*Where each state\'s estimate **starts**. Fill a row '
                          'to set that state\'s initial guess outright — any '
                          'value, negative included. Leave it blank and the '
                          'state starts at the truth plus a ~10% random error '
                          'drawn from the seed above.*\n\n'
                          '*P₀ follows from this rather than being set '
                          'separately: pinning a guess far from the truth '
                          'widens that state\'s prior variance to match, so the '
                          'filter starts out admitting an error the size of the '
                          'one you gave it instead of being overconfident. The '
                          'plotted ±2σ band is post-update, so a badly wrong '
                          'guess can still overshoot on the first step.*')

                with gr.Accordion('EKF', open=False):
                    with gr.Row():
                        s_ekf = gr.Checkbox(value=True, label='show EKF')
                        ekf_seed = gr.Number(value=0, precision=0,
                                             label='noise seed')
                    gr.Markdown(_X0_MD)
                    ekf_x0 = gr.Dataframe(
                        headers=['state', 'initial guess'],
                        datatype=['str', 'number'], type='array',
                        value=_x0_rows(_S0), interactive=True,
                        label='EKF initial estimate')
                    gr.Markdown(_INJ_MD)
                    with gr.Row():
                        ekf_ch = gr.Dropdown(
                            ['none'] + list(_S0.input_names), value='none',
                            label='input channel')
                        ekf_mag = gr.Number(value=0.0, label='bias magnitude')
                    with gr.Row():
                        ekf_t0 = gr.Number(value=0.0, label='start t₀ [s]')
                        ekf_t1 = gr.Number(value=0.0, label='end t₁ [s]')
                        ekf_unv = gr.Number(value=0.0,
                                            label='input noise var (all steps)')
                    ekf_p0 = gr.Number(
                        value=getattr(_S0, 'p0_paper', None),
                        label='P₀ diagonal (blank → from initial guess)')
                with gr.Accordion('UKF', open=False):
                    with gr.Row():
                        s_ukf = gr.Checkbox(value=True, label='show UKF')
                        ukf_seed = gr.Number(value=0, precision=0,
                                             label='noise seed')
                    gr.Markdown(_X0_MD)
                    ukf_x0 = gr.Dataframe(
                        headers=['state', 'initial guess'],
                        datatype=['str', 'number'], type='array',
                        value=_x0_rows(_S0), interactive=True,
                        label='UKF initial estimate')
                    gr.Markdown('*Sigma points sit at '
                                '$\\pm\\sqrt{\\alpha^2(n+\\kappa)}\\,\\sigma$ '
                                'about the mean.*', latex_delimiters=LATEX_DELIMS)
                    with gr.Row():
                        ukf_alpha = gr.Number(value=1.0, label='α (spread)')
                        ukf_beta = gr.Number(value=2.0, label='β (prior)')
                        ukf_kappa = gr.Number(value=0.0, label='κ (secondary)')
                    gr.Markdown(_INJ_MD)
                    with gr.Row():
                        ukf_ch = gr.Dropdown(
                            ['none'] + list(_S0.input_names), value='none',
                            label='input channel')
                        ukf_mag = gr.Number(value=0.0, label='bias magnitude')
                    with gr.Row():
                        ukf_t0 = gr.Number(value=0.0, label='start t₀ [s]')
                        ukf_t1 = gr.Number(value=0.0, label='end t₁ [s]')
                        ukf_unv = gr.Number(value=0.0,
                                            label='input noise var (all steps)')
                    ukf_p0 = gr.Number(
                        value=getattr(_S0, 'p0_paper', None),
                        label='P₀ diagonal (blank → from initial guess)')
                with gr.Accordion('ANN \u2014 work in progress', open=False):
                    gr.Markdown('**\u26a0 Work in progress \u2014 disabled.** These controls are '
                                'greyed out; the equations and repo defaults are '
                                'kept here for reference.')
                    gr.Markdown(
                        '*Feed-forward network estimating one state from the '
                        'measurements over a backward window (Cellini et al. '
                        'Eq. 6). Defaults are the authors\' repo values: '
                        '(64, 64, 64) layers, 4 time-steps for angular states, '
                        'batch 256, input-noise σ 0.01. Trained on MPC rollouts of '
                        'this system over perturbed set-points, never on the '
                        'trajectory being estimated; trains on first use, then '
                        'caches.*')
                    with gr.Row():
                        s_ann = gr.Checkbox(value=False, interactive=ANN_ENABLED,
                                            label='show ANN (raw)')
                        ann_target = gr.Dropdown(
                            list(_S0.state_names),
                            value=_S0.color_state_default,
                            label='state to estimate', interactive=ANN_ENABLED)
                    with gr.Row():
                        ann_layers = gr.Textbox(
                            value='64, 64, 64',
                            label='hidden layers (comma-separated)', interactive=ANN_ENABLED)
                        ann_steps = gr.Number(value=4, precision=0,
                                              label='time-steps in window', interactive=ANN_ENABLED)
                    with gr.Row():
                        ann_traj = gr.Number(
                            value=16, precision=0,
                            label='training MPC rollouts', interactive=ANN_ENABLED)
                        ann_epochs = gr.Number(value=100, precision=0,
                                               label='epochs', interactive=ANN_ENABLED)
                    with gr.Row():
                        ann_batch = gr.Number(value=256, precision=0,
                                              label='batch size', interactive=ANN_ENABLED)
                        ann_noise = gr.Number(
                            value=0.01, label='training input-noise σ', interactive=ANN_ENABLED)
                with gr.Accordion('AI-KF \u2014 work in progress', open=False):
                    gr.Markdown('**\u26a0 Work in progress \u2014 disabled.** These controls are '
                                'greyed out; the equations and repo defaults are '
                                'kept here for reference.')
                    gr.Markdown(
                        '*The repo\'s AI-KF: a **UKF** (their `method = \'ukf\'`) whose '
                        'measurement vector is '
                        'augmented with the ANN estimate, whose noise is set by an '
                        'active-sensing motif (mean |input| over a backward '
                        'window, log-mapped onto the R bounds). Repo values: '
                        'window 20, upper bound 0.5, R from 1e12 (motif 0 — ANN '
                        'ignored) to 1e-3 (full motif — ANN trusted). Setting '
                        '**R at full motif = 1e12 disables the ANN channel** and '
                        'recovers the plain UKF — the paper\'s own ablation. Sigma '
                        'points come from the UKF section above.*')
                    with gr.Row():
                        s_aikf = gr.Checkbox(value=False, interactive=ANN_ENABLED,
                                             label='show AI-KF')
                        aikf_motif = gr.Dropdown(
                            list(_S0.input_names),
                            value=_S0.input_names[-1],
                            label='motif input channel', interactive=ANN_ENABLED)
                    with gr.Row():
                        aikf_window = gr.Number(value=20, precision=0,
                                                label='motif window (steps)', interactive=ANN_ENABLED)
                        aikf_upper = gr.Number(value=0.5,
                                               label='motif upper bound', interactive=ANN_ENABLED)
                    with gr.Row():
                        aikf_r_hi = gr.Number(value=1e12,
                                              label='R at zero motif', interactive=ANN_ENABLED)
                        aikf_r_lo = gr.Number(value=1e-3,
                                              label='R at full motif', interactive=ANN_ENABLED)

            with gr.Accordion('Plot settings', open=False,
                              elem_id='sec_plot') as acc_plot:
                with gr.Accordion('Observability', open=False):
                    with gr.Accordion('Trajectory visualization', open=False):
                        with gr.Row():
                            color = gr.Dropdown(
                                list(_S0.state_names),
                                value=_S0.color_state_default,
                                label='color overlay (state/min ev)')
                            traj_mode = gr.Radio(['continuous', 'arrowhead'],
                                                 value='arrowhead',
                                                 label='trajectory style')
                        with gr.Row():
                            cmap = gr.Dropdown(CMAPS, value='inferno_r',
                                               label='colormap')
                            vmin = gr.Number(value=1e-4, label='colormap min')
                            vmax = gr.Number(value=1e6, label='colormap max')
                    with gr.Accordion('Min error variance time series',
                                      open=False):
                        ev_states = gr.CheckboxGroup(
                            list(_S0.state_names),
                            value=list(_S0.ev_states_default),
                            label='min-EV curve states')
                with gr.Accordion('Estimators', open=False):
                    with gr.Accordion('States', open=False):
                        est_states = gr.CheckboxGroup(
                            list(_S0.state_names),
                            value=list(_S0.ev_states_default),
                            label='states to show')
                        s_true = gr.Checkbox(value=True, label='true')
                        gr.Markdown('*Which estimators to draw is set in '
                                    '**Estimators** above, one per section.*')
                        gr.Markdown('*One row per axis \u2014 leave cells blank for '
                                    'autoscale. The `time [s]` row sets the shared x axis.*')
                        st_ytbl = gr.Dataframe(
                            headers=['axis', 'min', 'max'],
                            datatype=['str', 'number', 'number'],
                            type='array', value=_axis_rows(_S0.state_names),
                            interactive=True, label='axis limits')
                    with gr.Accordion('Measurements', open=False):
                        meas_sel = gr.CheckboxGroup(
                            list(_S0.measurement_names),
                            value=list(render.meas_default(_S0)),
                            label='measurements to show')
                        with gr.Row():
                            m_noisy = gr.Checkbox(value=True, label='noisy')
                            m_true = gr.Checkbox(value=True, label='true')
                        gr.Markdown('*Predicted measurements h(x̂,u) per '
                                    'estimator — ANN uses the AI-KF state with '
                                    'its own estimate substituted for the state '
                                    'it predicts.*')
                        with gr.Row():
                            m_pred = gr.Checkbox(value=True, label='EKF')
                            m_ukf = gr.Checkbox(value=False, label='UKF')
                            m_ann = gr.Checkbox(value=False, interactive=ANN_ENABLED,
                                                label='ANN')
                            m_aikf = gr.Checkbox(value=False, interactive=ANN_ENABLED,
                                                 label='AI-KF')
                        gr.Markdown('*One row per axis \u2014 leave cells blank for '
                                    'autoscale. The `time [s]` row sets the shared x axis.*')
                        ms_ytbl = gr.Dataframe(
                            headers=['axis', 'min', 'max'],
                            datatype=['str', 'number', 'number'],
                            type='array',
                            value=_axis_rows(_S0.measurement_names),
                            interactive=True,
                            label='axis limits')

            with gr.Accordion('📖 Reference & equations', open=False,
                              elem_id='sec_reference') as acc_reference:
                with gr.Accordion('Systems', open=False):
                    gr.Markdown(EQ_SYS_INTRO, latex_delimiters=LATEX_DELIMS)
                    for _name, _label in SYS_LIST:
                        with gr.Accordion(_label, open=False):
                            if _name in SYS_IMG:
                                gr.Image(os.path.join(TUT_DIR, SYS_IMG[_name]),
                                         show_label=False, container=False,
                                         height=300)
                            gr.Markdown(_system_md(_name),
                                        latex_delimiters=LATEX_DELIMS)
                    gr.Markdown('*Custom systems: the uploaded `.py` supplies '
                                'its own `f`, `h`, and names.*')
                with gr.Accordion('Observability', open=False):
                    gr.Markdown(EQ_OBS_INTRO, latex_delimiters=LATEX_DELIMS)
                    with gr.Accordion('Observability matrix 𝒪', open=False):
                        with gr.Accordion('empirical', open=False):
                            gr.Markdown(EQ_OBSMAT_EMP,
                                        latex_delimiters=LATEX_DELIMS)
                        with gr.Accordion('stochastic', open=False):
                            gr.Markdown(EQ_OBSMAT_STOCH,
                                        latex_delimiters=LATEX_DELIMS)
                    with gr.Accordion('Fisher information', open=False):
                        gr.Markdown(EQ_FISHER_INTRO,
                                    latex_delimiters=LATEX_DELIMS)
                        with gr.Accordion('empirical', open=False):
                            gr.Markdown(EQ_FISHER_EMP,
                                        latex_delimiters=LATEX_DELIMS)
                        with gr.Accordion('stochastic', open=False):
                            gr.Markdown(EQ_FISHER_STOCH,
                                        latex_delimiters=LATEX_DELIMS)
                    with gr.Accordion('Min error variance', open=False):
                        gr.Markdown(EQ_MINEV, latex_delimiters=LATEX_DELIMS)
                with gr.Accordion('Estimators', open=False):
                    gr.Markdown(EQ_EST_INTRO, latex_delimiters=LATEX_DELIMS)
                    with gr.Accordion('Kalman filter', open=False):
                        # the cycle both filters share; each subsection below
                        # fills in the equations that filter actually runs
                        gr.Image(os.path.join(TUT_DIR, 'kf_simple_web.png'),
                                 show_label=False, container=False, height=300)
                        gr.Markdown(EQ_KF_CYCLE, latex_delimiters=LATEX_DELIMS)
                        with gr.Accordion('EKF', open=False):
                            gr.Image(os.path.join(TUT_DIR, 'ekf_eqs_web.png'),
                                     show_label=False, container=False,
                                     height=300)
                            gr.Markdown(EQ_EKF, latex_delimiters=LATEX_DELIMS)
                        with gr.Accordion('UKF', open=False):
                            gr.Image(os.path.join(TUT_DIR, 'ukf_eqs_web.png'),
                                     show_label=False, container=False,
                                     height=330)
                            gr.Markdown(EQ_UKF, latex_delimiters=LATEX_DELIMS)
                    with gr.Accordion('ANN (artificial neural network)',
                                      open=False):
                        gr.Markdown(EQ_ANN, latex_delimiters=LATEX_DELIMS)
                    with gr.Accordion('AI-KF (motif informed filter)',
                                      open=False):
                        gr.Markdown(EQ_AIKF, latex_delimiters=LATEX_DELIMS)

        # ---------------------------- figures --------------------------------
        # Two visually separated sections — observability plots, then the
        # estimation (states/measurements) plots. Each plot lives in its own
        # collapsible accordion. When the stochastic gramian is on, the 𝒪 and
        # F⁻¹ plots show empirical | stochastic side by side.
        with gr.Column(scale=5):
            status = gr.Markdown('starting …')
            gr.Markdown('## 🔎 Observability')
            with gr.Accordion('Observability (min error variance)', open=True):
                gr.Markdown(r'Path colored by min error variance '
                            r'$\mathrm{diag}[(\mathcal{F}+\lambda I)^{-1}]$ per '
                            r'sliding window; the shaded band marks the window '
                            r'used by the matrices below. The curve underneath is '
                            r'the same quantity over time, one line style per '
                            r'pipeline.',
                            latex_delimiters=LATEX_DELIMS)
                fig_main = gr.Plot(show_label=False)
            with gr.Accordion('Observability / constructability matrix 𝒪',
                              open=False):
                gr.Markdown(r'$\mathcal{O}_{(i,k),j}=\partial y_i(t_k)/\partial '
                            r'x_j(t_0)$ — rows = sensor×time-step, cols = states '
                            r'(empirical $\approx\Delta Y/2\varepsilon$).',
                            latex_delimiters=LATEX_DELIMS)
                mat_k = gr.Number(value=0, precision=0,
                                  label='sliding-window index')
                fig_O = gr.Plot(show_label=False)
            with gr.Accordion('Inverse Fisher information F⁻¹', open=False):
                gr.Markdown(r'$\mathcal{F}=\mathcal{O}^{\mathsf T}R^{-1}'
                            r'\mathcal{O}$; shown $|(\mathcal{F}+\lambda I)^{-1}|$ '
                            r'— the diagonal is the min error variance.',
                            latex_delimiters=LATEX_DELIMS)
                fi_k = gr.Number(value=0, precision=0,
                                 label='sliding-window index')
                fig_Finv = gr.Plot(show_label=False)
            with gr.Accordion('Min error variance map (state × time)',
                              open=False):
                gr.Markdown(
                    r'The same $\mathrm{diag}[(\mathcal{F}+\lambda I)^{-1}]$ '
                    r'values as the min-EV curve above, as a heatmap: states on '
                    r'x, time down y. Rows are grouped by state, so the pipelines '
                    r'for one state (emp / obs / con) sit together.',
                    latex_delimiters=LATEX_DELIMS)
                fig_minev = gr.Plot(show_label=False)

            gr.Markdown('---\n## 📈 Estimation')
            with gr.Accordion('States (true / EKF / UKF)', open=False):
                fig_est = gr.Plot(show_label=False)
            with gr.Accordion('Measurements', open=False):
                fig_meas = gr.Plot(show_label=False)
            with gr.Accordion('Inputs (measured accelerations)', open=False):
                gr.Markdown('*Process-model inputs, not measurements: they enter '
                            'the prediction step, so a bias here walks the '
                            'estimate away with nothing to pull it back.*')
                fig_inputs = gr.Plot(show_label=False)

    # -------------------------------- wiring ---------------------------------
    COMPUTE_IN = [sess, custom_spec, segs, recorded, system, dt_hz,
                  v0, wind, zeta, w, lam, eps, r_mode, r_uniform, r_table,
                  do_stoch, q_obs, q_con, q_mode, q_noise, q_uniform, q_table,
                  fim_sensors, fim_states,
                  color, ev_states, est_states, meas_sel, traj_mode, cmap, vmin,
                  vmax, s_true, s_ekf, s_ukf, s_ann, s_aikf,
                  m_noisy, m_true, m_pred, m_ukf, m_ann, m_aikf,
                  ekf_seed, ekf_x0, ekf_ch, ekf_mag, ekf_t0, ekf_t1, ekf_unv,
                  ekf_p0,
                  ukf_seed, ukf_x0, ukf_ch, ukf_mag, ukf_t0, ukf_t1, ukf_unv,
                  ukf_p0,
                  ukf_alpha, ukf_beta, ukf_kappa,
                  ann_target, ann_layers, ann_steps, ann_traj, ann_epochs,
                  ann_batch, ann_noise,
                  aikf_motif, aikf_window, aikf_upper, aikf_r_hi, aikf_r_lo,
                  st_ytbl, ms_ytbl,
                  mat_k, fi_k]
    COMPUTE_OUT = [fig_main, fig_O, fig_Finv, fig_minev, fig_est, fig_meas,
                   fig_inputs,
                   status, sess]
    SYSTEM_OUT = [fim_sensors, fim_states, color, ev_states, est_states,
                  meas_sel, w, lam, eps, r_mode, r_uniform, r_table, do_stoch,
                  q_mode, q_noise, q_uniform, q_table, dt_hz,
                  v0, wind, zeta, p_duration, segs, seg_display,
                  recorded, rec_status, draw_data, custom_spec, sys_file,
                  ekf_x0, ekf_ch, ekf_mag, ekf_t0, ekf_t1, ekf_unv,
                  ukf_x0, ukf_ch, ukf_mag, ukf_t0, ukf_t1, ukf_unv,
                  ukf_alpha, ukf_beta, ukf_kappa,
                  ann_target, ann_layers, ann_steps, ann_traj, ann_epochs,
                  ann_batch, ann_noise,
                  aikf_motif, aikf_window, aikf_upper, aikf_r_hi, aikf_r_lo,
                  st_ytbl, ms_ytbl,
                  fig_main, fig_O, fig_Finv, fig_minev, fig_est, fig_meas,
                  fig_inputs,
                  status, sess]

    # measurement/process-noise mode → show uniform value or per-channel table
    r_mode.change(lambda m: (gr.update(visible=m == 'uniform'),
                             gr.update(visible=m == 'custom')),
                  r_mode, [r_uniform, r_table])
    q_mode.change(lambda m: (gr.update(visible=m == 'uniform'),
                             gr.update(visible=m == 'custom')),
                  q_mode, [q_uniform, q_table])
    do_stoch.change(lambda d, m: (gr.update(visible=d), gr.update(visible=d),
                                  gr.update(visible=d),
                                  gr.update(visible=d),
                                  gr.update(visible=d and m == 'uniform'),
                                  gr.update(visible=d and m == 'custom')),
                    [do_stoch, q_mode],
                    [q_obs, q_con, q_mode, q_noise, q_uniform, q_table])

    # custom system upload (revealed when system = Custom)
    sys_file.upload(on_upload_system, sys_file,
                    [custom_spec, fim_sensors, fim_states, color, ev_states,
                     est_states, meas_sel, r_uniform, r_table, q_table, status])

    # physical trajectory: measured x,y → MPC reconstruct (built-in or custom)
    phys_file.upload(on_upload_physical,
                     [phys_file, x_col, y_col, id_col, system, custom_spec],
                     [phys_traj, obj_dd, phys_status])
    obj_dd.change(on_select_physical, [phys_traj, obj_dd, custom_spec],
                  [recorded, segs, seg_display, phys_status]).then(
                      simulate, COMPUTE_IN, COMPUTE_OUT)

    # trajectory builder — edits the segment list, no recompute until Simulate
    motif.change(on_motif, motif, [p_angle, p_speed, p_freq, p_amp])
    btn_add.click(add_seg,
                  [segs, motif, p_duration, p_angle, p_speed, p_freq, p_amp],
                  [segs, seg_display])
    btn_undo.click(undo_seg, segs, [segs, seg_display])
    btn_clear.click(clear_seg, None, [segs, seg_display])
    # "reset" = clear the segment list (empty → default path)
    btn_default.click(clear_seg, None, [segs, seg_display])

    # draw-a-trajectory recorder (canvas capture lives in _APP_JS)
    _CLEAR_JS = '() => { if (window.__clearDraw) window.__clearDraw(); }'
    btn_draw_apply.click(apply_drawing, [draw_data, system, v0, draw_dur],
                         [recorded, rec_status, segs, seg_display])
    btn_draw_clear.click(clear_recorded, None, [recorded, rec_status],
                         js=_CLEAR_JS)
    # switching systems also wipes the canvas (extents differ per system)
    system.change(None, None, None, js=_CLEAR_JS)

    # things that change the MPC trajectory → rebuild (dt also rebuilds the
    # engine at the new timestep). .input on dt_hz = user-only, so the reset
    # in on_system doesn't re-trigger it.
    btn_sim.click(simulate, COMPUTE_IN, COMPUTE_OUT)
    for comp in (v0, wind, zeta):
        comp.submit(simulate, COMPUTE_IN, COMPUTE_OUT)
    dt_hz.input(simulate, COMPUTE_IN, COMPUTE_OUT)

    # system change: reset controls + recompute in one call (see on_system)
    system.change(on_system, system, SYSTEM_OUT)

    # everything else recomputes from the cached trajectory (cheap).
    # NOTE: .input (not .change) — .input fires only on real USER edits, so the
    # controls that on_system resets programmatically don't trigger a cascade of
    # updates that would read the CheckboxGroups mid-reset (value vs new choices
    # transiently inconsistent → "'phi' not in choices" crash).
    for comp in (w, lam, eps, r_mode, r_uniform, r_table, do_stoch,
                 q_obs, q_con, q_mode,
                 q_uniform, q_table, fim_sensors, fim_states, color, ev_states,
                 est_states, meas_sel, traj_mode, cmap, vmin, vmax, s_true,
                 s_ekf, s_ukf, s_ann, s_aikf, m_noisy, m_true, m_pred,
                 m_ukf, m_ann, m_aikf,
                 ekf_seed, ekf_x0, ekf_ch, ekf_mag, ekf_t0, ekf_t1, ekf_unv, ekf_p0,
                 ukf_seed, ukf_x0, ukf_ch, ukf_mag, ukf_t0, ukf_t1, ukf_unv, ukf_p0,
                 ukf_alpha, ukf_beta, ukf_kappa,
                 ann_target, ann_layers, ann_steps, ann_traj, ann_epochs,
                 ann_batch, ann_noise,
                 aikf_motif, aikf_window, aikf_upper, aikf_r_hi, aikf_r_lo,
                 st_ytbl, ms_ytbl):
        comp.input(update, COMPUTE_IN, COMPUTE_OUT)
    # window indices use .change (commit), not .input — a Number emits None
    # mid-edit, which .input would turn into a transient window-0 flicker
    for comp in (mat_k, fi_k):
        comp.change(update, COMPUTE_IN, COMPUTE_OUT)

    # tutorial: image + caption + section-opening from Python; the glow/scroll
    # is applied client-side by window.__tutHighlight, fired via tut_hl.change
    # (a Textbox change reliably passes its new value to the js each step).
    TUT_OUT = [tutorial_panel, tut_img, tut_cap, tut_step,
               acc_system, acc_traj, acc_obs, acc_estimator, acc_plot,
               acc_reference, tut_hl]
    btn_tutorial.click(open_tutorial, None, TUT_OUT)
    btn_tut_next.click(lambda s: _tutorial_view(s + 1), tut_step, TUT_OUT)
    btn_tut_prev.click(lambda s: _tutorial_view(s - 1), tut_step, TUT_OUT)
    btn_tut_close.click(lambda: (gr.update(visible=False), ''), None,
                        [tutorial_panel, tut_hl], js='() => window.__tutClear()')
    tut_hl.change(None, tut_hl, None,
                  js='(s) => window.__tutHighlight(parseInt(s))')

    # run the load JS (light-mode lock + wire the drawing canvas) as a
    # dedicated load-event handler — js passed to launch() does not execute on
    # load in Gradio 6, but a js-only demo.load handler does.
    demo.load(None, None, None, js=_APP_JS)
    # first paint
    demo.load(simulate, COMPUTE_IN, COMPUTE_OUT)


def _free_port(start, tries=20):
    """ first port at or after `start` with nothing listening on it, so a stale
    instance left running doesn't block a restart. Falls back to `start` (and
    lets Gradio raise) if the whole range is busy. """
    for port in range(start, start + tries):
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start


if __name__ == '__main__':
    # bind 0.0.0.0 and honor a platform-provided $PORT so the same app runs
    # locally and on any container host (Cloud Run, Fly, Render, Railway, …).
    # A platform $PORT is the one the host routes to, so it is used verbatim;
    # only the local default steps to the next free port.
    env_port = os.environ.get('PORT')
    port = int(env_port) if env_port else _free_port(7860)  # distinct from app.py
    # --share opens a temporary public gradio.live tunnel to THIS machine. A
    # command-line flag rather than an edit to this line, so a shared session
    # can't be committed by accident. Anything else on argv is a typo, not a
    # silent no-op.
    argv = sys.argv[1:]
    share = '--share' in argv
    unknown = [a for a in argv if a != '--share']
    if unknown:
        sys.exit(f'unknown argument(s): {" ".join(unknown)}\n'
                 f'usage: python {os.path.basename(__file__)} [--share]')
    demo.launch(server_name='0.0.0.0', server_port=port, share=share,
                theme=THEME, css=CSS, allowed_paths=[TUT_DIR])
