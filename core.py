"""
Shared compute core for the observability explorer.

Everything here is UI-agnostic: system metadata, the set-point builder, the
numerics of one refresh (compute_payload), the figure assembly
(_figs_from_payload), and the per-system control defaults. Both front-ends
import it —

    app_custom.py     Gradio
    streamlit_app.py  Streamlit

— so the science is defined once and cannot drift between them. Nothing in this
module may import a UI toolkit.
"""
import os
import sys
import io
import time
import warnings
import contextlib
import functools
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

from observability_gui import (ObservabilityEngine, SYSTEMS, build_setpoints,
                               MOTIF_LABELS, SystemSpec)
import render


# ─────────────────────────── system metadata ────────────────────────────────

SYSTEM_LABELS = [('fly (full state)', 'fly'),
                 ('fly (simple)', 'fly7'),
                 ('drone (kinematic 3D)', 'drone'),
                 ('altitude 2D (paper Fig. 4b)', 'alt2d'),
                 ('Custom system (upload f & h)', 'custom')]
# plain python floats — Gradio dropdowns can't match a value against numpy
# scalar choices (np.arange gives np.float64), which raised a "not in the list
# of choices" error on system switch
LAM_VALS = [float(10.0 ** e) for e in range(-8, -2)]                 # 1e-8 … 1e-3
EPS_VALS = [float(10.0 ** e) for e in np.arange(-6, -2 + 1e-9, 0.5)]  # 1e-6 … 1e-2


def get_spec(system, dt=None):
    return SYSTEMS[system](dt=float(dt)) if dt else SYSTEMS[system]()
# sampling rate control: dt = 1 / Hz. Higher Hz = finer dt = more steps =
# slower but higher-resolution. Each system defaults to its native rate.
DT_HZ = [('1 Hz', 1.0), ('5 Hz', 5.0), ('10 Hz', 10.0),
         ('20 Hz', 20.0), ('50 Hz', 50.0), ('100 Hz', 100.0)]
CMAPS = ['inferno_r', 'viridis', 'plasma', 'magma', 'cividis', 'copper',
         'turbo', 'jet']
_BASE_HZ = {name: float(round(1.0 / SYSTEMS[name]().dt)) for name in SYSTEMS}


# ── filter backend: this repo's NumPy EKF/UKF, or the same problem run through
# dynamax (JAX). See engine/dynamax_filters.py for what is shared and what
# necessarily differs. Availability is probed WITHOUT importing jax (seconds of
# start-up), so a front-end can label the control before anything is computed.
FILTER_BACKENDS = [('this repo (NumPy)', 'engine'), ('dynamax (JAX)', 'dynamax'),
                   ('square-root UKF (NumPy)', 'srukf')]


def _has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


DYNAMAX_AVAILABLE = _has_module('jax') and _has_module('dynamax')


def build_setpoint(spec, segments, recorded, v0, wind, zeta):
    """ Build the MPC set-point. custom position set-point → drawn/measured
    (speed, heading) → motif segments → the system's default trajectory. """
    if isinstance(recorded, dict) and 'pos' in recorded:      # custom position track
        pos = {s: np.asarray(a, dtype=float) for s, a in recorded['pos'].items()}
        N = len(next(iter(pos.values())))
        return spec.make_setpoint(pos, N)
    if segments:
        speed_sp, heading_sp = build_setpoints(segments, spec.dt, v0=float(v0),
                                               heading0=spec.heading0)
        return spec.make_setpoint(speed_sp, heading_sp, float(wind), float(zeta))
    if recorded is not None:
        speed_sp = np.asarray(recorded[0], dtype=float)
        heading_sp = np.asarray(recorded[1], dtype=float)
        return spec.make_setpoint(speed_sp, heading_sp, float(wind), float(zeta))
    return spec.default_setpoint(float(wind), float(zeta))

# ─────────────── compute (Qt-free, lifted from observability_app) ────────────

# Stage weights for progress reporting, measured on the fly system: MPC and the
# empirical FIM dominate and both grow with trajectory length, so a bar that
# treated stages equally would sit at 20% for most of a long run. Rough is fine;
# these only decide where the bar sits, never what is computed.
_STAGES = (('mpc', 0.45, 'solving MPC trajectory'),
           ('emp', 0.30, 'empirical Fisher information'),
           ('stoch', 0.05, 'stochastic gramians'),
           ('filt', 0.12, 'running EKF / UKF'),
           ('ann', 0.08, 'ANN / AI-KF'))


def compute_payload(engine, job, p, est, on_stage=None):
    """ All the heavy numerics for one refresh. Lifted from
    ``observability_app.compute_payload`` — it never touched Qt, only the
    engine. ``job`` is ('update',) or ('rebuild', setpoint).

    ``on_stage(fraction, label)`` is an optional progress callback, called before
    each stage starts. A long drawn trajectory puts tens of seconds into MPC and
    the empirical FIM, which otherwise looks like the app has hung. """
    spec = engine.spec
    note = ''
    done = [0.0]

    def tick(key):
        """ Report the stage about to start, then bank its weight. """
        for k, wt, label in _STAGES:
            if k != key:
                continue
            if on_stage is not None:
                try:
                    on_stage(done[0], label)
                except Exception:
                    pass                # progress must never break the compute
            done[0] += wt
        return time.time()

    if job[0] == 'rebuild':
        t0 = tick('mpc')
        with contextlib.redirect_stdout(io.StringIO()):     # hush IPOPT
            engine.simulate_mpc(job[1])
        note = f'trajectory: {engine.N} steps in {time.time() - t0:.1f} s'
    else:
        done[0] += _STAGES[0][1]        # nothing to solve; skip that slice
    if not engine.N:
        return {'error': 'no trajectory'}

    w = int(min(p['w_raw'], max(engine.N - 1, 3)))
    # empirical FIM (pybounds, Q=0) — always; stochastic gramian (Eq.33, Q>0) —
    # optional, so the two min-error-variance results can be compared.
    t0 = tick('emp')
    ev_emp, _ = engine.empirical_ev(w, p['eps'], p['r_diag'], p['lam'],
                                    sensors=p['sensors'], states=p['states'])
    t_emp = time.time() - t0
    # one min-EV frame per selected basis, keyed by basis name, so the curves
    # (and their trajectory panels) can be drawn together for comparison
    ev_stoch, t_stoch = {}, 0.0
    if p['do_stoch']:
        t0 = tick('stoch')
        for b in p['bases']:
            ev_stoch[b] = engine.linearized_ev(
                w, p['q_diag'], p['r_diag'], p['lam'],
                sensors=p['sensors'], states=p['states'], basis=b,
                q_noise=p.get('q_noise', 'uncorrelated'))
        t_stoch = time.time() - t0

    # Which implementation runs the filter recursion, resolved before anything
    # is computed so the note is written even if the filters then fail.
    # 'dynamax' hands the SAME problem (same seed → same Y, x̂₀, P₀, U; same
    # sensor subset; same Q convention) to dynamax's JAX filters, so a
    # disagreement is a disagreement about the filter and nothing else.
    # Anything dynamax cannot reproduce is listed in the note, not left
    # implicit. The ANN / AI-KF below always uses the engine's own UKF.
    # anything unrecognized is the NumPy path: a stale browser session or a
    # script can post any string here, and reporting back a backend that did
    # not run would put a wrong line in the exported report
    req = p.get('backend')
    backend = req if req in ('dynamax', 'srukf') else 'engine'
    dmx = None
    if backend == 'dynamax':
        try:
            import dynamax_filters as dmx
            if not dmx.AVAILABLE:
                raise dmx.IMPORT_ERROR
        except Exception as exc:
            note += (f'{" | " if note else ""}⚠ dynamax backend unavailable '
                     f'({type(exc).__name__}: {exc}) — ran the NumPy filters '
                     f'instead; pip install dynamax')
            backend, dmx = 'engine', None
    if backend == 'dynamax':
        caveats = dmx.limitations(spec, p.get('q_noise', 'uncorrelated'))
        note += (f'{" | " if note else ""}EKF/UKF via dynamax'
                 + (' — ' + '; '.join(caveats) if caveats else ''))
    if backend == 'srukf':
        note += (f'{" | " if note else ""}UKF via square-root UKF '
                 '(Van der Merwe); EKF unchanged (NumPy)')

    results, meas = None, None
    tick('filt')
    try:
        results = {}
        # both filters use the SAME R, Q and SENSORS as the observability
        # analysis, but each brings its OWN realization (seed, per-state initial
        # guess, injected disturbance) — see _realization. The UKF also takes
        # α, β, κ.
        #
        # The sensor set matters as much as R and Q: driving the estimators from
        # every channel of h() while the Gramians see only p['sensors'] compares
        # a filter that had extra information against a bound that did not, so
        # the min-error-variance curve and the estimator's actual error are then
        # answers to different questions.
        qd, rd = p['q_diag'], p['r_diag']
        ukf_kw = dict(alpha=est['ukf_alpha'], beta=est['ukf_beta'],
                      kappa=est['ukf_kappa'])
        if backend == 'dynamax':
            filt_specs = [
                ('EKF', functools.partial(dmx.run_ekf, engine), {}),
                ('UKF', functools.partial(dmx.run_ukf, engine), ukf_kw)]
        elif backend == 'srukf':
            # only the UKF has a square-root form; the EKF slot stays the
            # standard NumPy one, so toggling engine ↔ srukf is a clean A/B on
            # the UKF alone (both plot into the same UKF series and cache under
            # their own backend key)
            filt_specs = [
                ('EKF', engine.run_ekf, {}),
                ('UKF', engine.run_srukf, ukf_kw)]
        else:
            filt_specs = [
                ('EKF', engine.run_ekf, {}),
                ('UKF', engine.run_ukf, ukf_kw)]
        # Each filter is caught on its own: a UKF that goes singular must not
        # take the working EKF (and both estimator plots) down with it.
        for name, fn, extra in filt_specs:
            e = est[name]
            # the backend is part of the cache key: the two implementations are
            # allowed to disagree, so serving one's result for the other would
            # hide exactly what this option exists to show
            fkey = (name, backend, engine._version,
                    p.get('q_noise', 'uncorrelated'),
                    tuple(f'{qd[k]:.3e}' for k in spec.state_names),
                    tuple(f'{rd[m]:.3e}' for m in spec.measurement_names),
                    e['seed'], round(e['unv'], 9), e['ub'],
                    None if e['p0'] is None else tuple(np.round(e['p0'], 12)),
                    None if e['x0'] is None else tuple(np.round(e['x0'], 12)),
                    # the sensor set is part of the problem, so it must key the
                    # cache too — otherwise changing the selection would serve a
                    # stale estimate computed from the previous one
                    tuple(p['sensors']),
                    tuple(sorted(extra.items())))
            try:
                if fkey not in engine._filt_cache:
                    engine._filt_cache[fkey] = fn(
                        qd, rd, seed=e['seed'], u_noise_var=e['unv'],
                        u_bias=e['ub'], p0_diag=e['p0'],
                        x0_guess=e['x0'], sensors=p['sensors'],
                        q_noise=p.get('q_noise', 'uncorrelated'), **extra)
                results[name] = engine._filt_cache[fkey]
            # not just LinAlgError: a third-party backend can fail in its own
            # ways (an untraceable model, a jax error), and one filter's failure
            # must still leave the other one plotted
            except Exception as exc:
                note += (f'{" | " if note else ""}⚠ {name} failed: '
                         f'{type(exc).__name__}: {exc}')
        if not results:
            raise np.linalg.LinAlgError('no estimator converged')
        X, U = engine.X, engine.U
        # the plotted noisy measurements belong to ONE realization: use the
        # EKF's when it ran (the primary), else the UKF's. Y depends only on the
        # seed, so this matches whichever filter's samples are being shown.
        e_m = est['EKF' if 'EKF' in results else 'UKF']
        if est['EKF']['seed'] != est['UKF']['seed']:
            note += (f'{" | " if note else ""}measurements shown are the '
                     f"{'EKF' if 'EKF' in results else 'UKF'}'s realization "
                     f'(seeds differ)')
        # Ask for the SAME realization the filter saw, so the plotted
        # accelerations carry that filter's input noise and bias. (Y itself is
        # drawn first inside _estimation_problem, so it was already identical.)
        Y_noisy, _xh0, _P0, _R, U_used = engine._estimation_problem(
            p['r_diag'], e_m['seed'], u_noise_var=e_m['unv'],
            u_bias=e_m['ub'], p0_diag=e_m['p0'], x0_guess=e_m['x0'])
        Y_true = np.array([np.asarray(spec.h(X[k], U[k]), dtype=float)
                           for k in range(engine.N)])
        hx = lambda Xs: np.array([np.asarray(spec.h(Xs[k], U[k]), dtype=float)
                                  for k in range(engine.N)])
        meas = dict(Y_noisy=Y_noisy, Y_true=Y_true,
                    U_true=U.copy(), U_noisy=U_used)
        if 'EKF' in results:
            meas['Y_hat'] = hx(results['EKF'][0])
        if 'UKF' in results:
            meas['Y_hat_ukf'] = hx(results['UKF'][0])
    except np.linalg.LinAlgError:
        results, meas = None, None

    # ANN raw estimate (Eq. 6) + motif-informed filter of it (Eq. 5/7), one net
    # per plotted state. Opt-in, because the first call trains the net; nets are
    # cached per (spec, state, shape) and survive trajectory edits.
    ann, t_ann = None, 0.0
    if p['do_ann']:
        t0 = tick('ann')
        st = p['ann_target']
        e_u = est['UKF']            # the AI-KF is a UKF: same realization
        try:
            X_ai, P_ai, raw, R_ann = engine.run_aikf(
                st, p['q_diag'], p['r_diag'], seed=e_u['seed'],
                time_steps=p['ann_steps'], layers=p['ann_layers'],
                n_traj=p['ann_traj'], epochs=p['ann_epochs'],
                batch=p['ann_batch'], noise_std=p['ann_noise'],
                motif_input=p['aikf_motif'], motif_window=p['aikf_window'],
                motif_upper=p['aikf_upper'], R_ann_lo=p['aikf_r_lo'],
                R_ann_hi=p['aikf_r_hi'],
                # the AI-KF is a UKF, so it shares the UKF's sigma-point tuning
                alpha=est['ukf_alpha'], beta=est['ukf_beta'],
                kappa=est['ukf_kappa'], x0_guess=e_u['x0'],
                u_noise_var=e_u['unv'], u_bias=e_u['ub'], p0_diag=e_u['p0'])
            ann = {st: dict(raw=raw, X_hat=X_ai, P_diag=P_ai, R_ann=R_ann)}
            if meas is not None:
                U_m = engine.U
                hx = lambda Xs: np.array(
                    [np.asarray(spec.h(Xs[k], U_m[k]), dtype=float)
                     for k in range(engine.N)])
                meas['Y_hat_aikf'] = hx(X_ai)
                # the ANN estimates ONE state, so it has no state vector of its
                # own: substitute its estimate for that state into the AI-KF's
                # state and evaluate h on that.
                X_annsub = X_ai.copy()
                j_t = spec.state_names.index(st)
                ok = np.isfinite(raw)
                X_annsub[ok, j_t] = raw[ok]
                meas['Y_hat_ann'] = hx(X_annsub)
        except Exception as exc:                      # keep the rest of the page
            ann = {st: dict(error=f'{type(exc).__name__}: {exc}')}
        t_ann = time.time() - t0

    # say so when the estimators are running on fewer channels than h() offers —
    # the measurement panel still plots ŷ for EVERY channel, including ones the
    # filter never saw, and without this the two are easy to confuse
    if results and len(p['sensors']) < len(spec.measurement_names):
        note += (f'{" | " if note else ""}EKF/UKF driven by '
                 f'{", ".join(p["sensors"])} (the observability selection)')

    return dict(ev_emp=ev_emp, ev_lin=ev_stoch, bases=p['bases'], results=results,
                meas=meas, ann=ann, w=w, lam=p['lam'], backend=backend,
                q_zeta=p['q_diag'].get('zeta', 0.0),
                t_emp=t_emp, t_stoch=t_stoch, t_ann=t_ann,
                floor=engine.lam_noise_floor, note=note)


# ─────────────────────────── orchestration ──────────────────────────────────

def _figs_from_payload(engine, spec, payload, disp):
    """ Build the 3 figures (observability, states, measurements) from a payload
    and the display options dict `disp`. """
    # sliding window(s) feeding the matrix plots below, highlighted on the path
    w = payload['w']
    kmax = max(engine.N - w, 0)
    mk = int(np.clip(disp['mat_k'], 0, kmax))
    fk = int(np.clip(disp['fi_k'], 0, kmax))
    if mk == fk:
        windows = [(r'$\mathcal{O}$ & $F^{-1}$ window', mk, w, '#1f77b4')]
    else:
        windows = [(r'$\mathcal{O}$ window', mk, w, '#1f77b4'),
                   (r'$F^{-1}$ window', fk, w, '#2ca02c')]

    stoch, q_diag = disp['do_stoch'], disp['q_diag']
    bases = disp['bases'] if stoch else ()

    # shared log color range for the min-EV and |F⁻¹| maps, from the min-EV data
    # (so F⁻¹'s diagonal matches the min-EV map cell-for-cell)
    ev = payload['ev_emp']
    evcols = [s for s in disp['states'] if s in ev.columns]
    d = np.asarray(ev[evcols].values, dtype=float).ravel() if evcols else np.array([])
    d = d[np.isfinite(d) & (d > 0)]
    mv_vmin = float(d.min()) if d.size else 1e-1
    mv_vmax = float(d.max()) if d.size else 1e6

    # one trajectory panel per pipeline (empirical + each selected basis), so the
    # figure widens as bases are added instead of squeezing the panels
    n_traj = 1 + len(bases)
    # The browser fits the figure to the container WIDTH, so adding panels
    # widens the figure and scales the whole thing down — each panel ends up
    # smaller on screen even though the figure got bigger. Grow the height
    # with the panel count so the rendered image keeps at least the
    # single-panel aspect and the panels get their pixels back. The
    # one-panel default is deliberately unchanged.
    fig_main = Figure(figsize=(max(13.0, 6.0 * n_traj + 2.2),
                               11.5 + 3.4 * (n_traj - 1)))
    render.render_main(fig_main, engine, spec, payload, disp['color'],
                       disp['ev_states'], arrowhead=disp['arrowhead'],
                       cmap_name=disp['cmap'], vmin=disp['vmin'],
                       vmax=disp['vmax'], windows=windows)

    # same min-EV numbers as the curve above, heatmap form, in its own section so
    # neither plot crowds the other. Width follows the column count.
    hm_cols = max(1, len(disp['states']) * n_traj)
    fig_minev = Figure(figsize=(max(7.0, 0.34 * hm_cols + 3.0), 8.5))
    try:
        render.render_minev_map(fig_minev, engine, spec, payload,
                                list(disp['states']), cmap_name=disp['cmap'],
                                vmin=mv_vmin, vmax=mv_vmax)
    except Exception:
        fig_minev = None

    if payload['results'] is not None:
        nrows = len((disp['est_states'] or [spec.color_state_default])[:8])
        fig_est = Figure(figsize=(13, max(2.6 * nrows, 3.4)))
        render.render_estimates(fig_est, engine, spec, payload['results'],
                                disp['est_states'], show_true=disp['s_true'],
                                show_ekf=disp['s_ekf'], show_ukf=disp['s_ukf'],
                                ann=payload.get('ann'),
                                show_ann=disp['s_ann'],
                                show_aikf=disp['s_aikf'],
                                xlim=disp['st_xlim'], ylims=disp['st_ylims'])
    else:
        fig_est = None

    if payload['meas'] is not None:
        nrows = len((disp['meas_sel'] or list(render.meas_default(spec)))[:8])
        fig_meas = Figure(figsize=(13, max(2.4 * nrows, 3.2)))
        render.render_measurements(fig_meas, engine, spec, payload['meas'],
                                   disp['meas_sel'], show_noisy=disp['m_noisy'],
                                   show_true=disp['m_true'],
                                   show_pred=disp['m_pred'],
                                   show_ukf=disp['m_ukf'],
                                   show_ann=disp['m_ann'],
                                   show_aikf=disp['m_aikf'],
                                   xlim=disp['ms_xlim'], ylims=disp['ms_ylims'])
    else:
        fig_meas = None

    # measured process inputs get their own figure: sizing the measurement panel
    # for sensors and then squeezing input rows into it made both unreadable
    chans = render.inputs_default(spec)
    if payload['meas'] is not None and chans:
        fig_inputs = Figure(figsize=(13, max(2.4 * len(chans), 3.2)))
        render.render_inputs(fig_inputs, engine, spec, payload['meas'],
                             show_noisy=disp['m_noisy'], show_true=disp['m_true'],
                             xlim=disp['ms_xlim'], ylims=disp['ms_ylims'])
    else:
        fig_inputs = None

    # observability matrices (paper Fig 1e–h) — always rendered. Each selected
    # basis adds a stochastic panel beside the empirical one. 𝒪 and F⁻¹ are each
    # shown for one chosen sliding window.
    fig_O = fig_Finv = None
    try:
        O_emp, kk = engine.empirical_window_O(
            w, disp['eps'], disp['mat_k'],
            sensors=disp['sensors'], states=disp['states'])
        O_stoch = {b: engine.linearized_window_O(
            w, disp['mat_k'], q_diag, sensors=disp['sensors'],
            states=disp['states'], basis=b,
            q_noise=disp.get('q_noise', 'uncorrelated'))[0] for b in bases}
        fig_O = Figure(figsize=(6.5 + 4.4 * len(O_stoch), 5.5))
        render.render_obs_matrix(fig_O, O_emp, k=kk, O_stoch=O_stoch)
    except Exception:
        fig_O = None
    try:
        F_emp, fk = engine.empirical_window_fisher_inv(
            w, disp['eps'], disp['r_diag'], disp['lam'], disp['fi_k'],
            sensors=disp['sensors'], states=disp['states'])
        F_stoch = {b: engine.linearized_fisher_inv(
            q_diag, disp['r_diag'], disp['lam'], sensors=disp['sensors'],
            states=disp['states'], a=fk, b=fk + w, basis=b,
            q_noise=disp.get('q_noise', 'uncorrelated')) for b in bases}
        fig_Finv = Figure(figsize=(6.5 + 4.4 * len(F_stoch), 5.5))
        render.render_fisher_inv(fig_Finv, F_emp, F_stoch=F_stoch, k=fk,
                                 cmap_name=disp['cmap'], vmin=mv_vmin, vmax=mv_vmax)
    except Exception:
        fig_Finv = None
    return fig_main, fig_O, fig_Finv, fig_minev, fig_est, fig_meas, fig_inputs

def _est_defaults(s):
    """ Per-system defaults for the estimator controls.

    A spec may carry paper/repo values (alt2d does — see make_alt2d_spec); every
    other system falls back to generic ones. Returns a flat dict keyed by control
    name so on_system and the initial render stay in step. """
    ann = dict(target=s.color_state_default, layers='64, 64, 64', time_steps=4,
               n_traj=16, epochs=100, batch=256, noise=0.01)
    ann.update(getattr(s, 'ann_paper', {}) or {})
    if ann['target'] not in s.state_names:
        ann['target'] = s.color_state_default
    aikf = dict(motif=s.input_names[-1], window=20, upper=0.5,
                r_hi=1e12, r_lo=1e-3)
    aikf.update(getattr(s, 'aikf_paper', {}) or {})
    if aikf['motif'] not in s.input_names:
        aikf['motif'] = s.input_names[-1]
    ub = getattr(s, 'u_bias_default', None)
    inj = dict(channel='none', mag=0.0, t0=0.0, t1=0.0)
    if ub:                                   # ('u_z', 0.25, 20.0, 28.0)
        ch, mag, t0, t1 = ub
        if ch in s.input_names:
            inj = dict(channel=ch, mag=float(mag), t0=float(t0), t1=float(t1))
    a, b, k = getattr(s, 'ukf_paper', (1.0, 2.0, 0.0))
    return dict(ann=ann, aikf=aikf, inj=inj,
                # per-state P0 table, prefilled with the spec's paper P0 when it
                # has one and blank otherwise
                p0=_p0_rows(s),
                u_noise=float(getattr(s, 'u_noise_default', 0.0) or 0.0),
                ukf_alpha=float(a), ukf_beta=float(b), ukf_kappa=float(k),
                # blank initial-guess table: every state starts from the seeded
                # random draw until a row is filled in
                x0=_x0_rows(s))


def _realization(spec, seed, inj_ch, inj_mag, inj_t0, inj_t1, u_noise, x0_rows,
                 p0=None):
    """ One estimator's own estimation problem: which noise draw it sees, where
    its initial estimate starts, and any injected input disturbance.

    Each filter carries a full copy of this, so the EKF and UKF can be
    initialized independently. Give them the same seed to compare them on
    identical data; give them different seeds and they face different
    measurements, which is a different (and usually less informative)
    comparison.

    P₀ is a per-state diagonal (`p0` is the table, or a scalar to broadcast).
    Any state left blank follows the initial estimate, widening if you pin that
    guess far from the truth (see _estimation_problem), so pinned and
    guess-following states mix freely. An all-blank table falls back to the
    spec's paper P₀ (alt2d's p0_paper = 10) so the published defaults keep
    reproducing. """
    ub = None
    if inj_ch and inj_ch != 'none' and float(inj_mag or 0.0) != 0.0 \
            and float(inj_t1 or 0.0) > float(inj_t0 or 0.0) \
            and inj_ch in spec.input_names:
        ub = (spec.input_names.index(inj_ch), float(inj_mag),
              float(inj_t0), float(inj_t1))
    # an explicit P0 entry wins per state; a fully blank table falls back to the
    # spec's published default (alt2d's p0_paper = 10) so the paper figures keep
    # reproducing
    p0v = _p0_vec(p0, spec.state_names)
    if p0v is None:
        pp = getattr(spec, 'p0_paper', None)
        p0v = _p0_vec([[n, pp.get(n)] for n in spec.state_names]
                      if isinstance(pp, dict) else pp, spec.state_names)
    return dict(seed=int(seed or 0), unv=float(u_noise or 0.0), ub=ub,
                x0=_x0_vec(x0_rows, spec.state_names), p0=p0v)


def _q_rows_default(s):
    """ Per-state Q table, using the spec's paper values where it has them. """
    qp = getattr(s, 'q_paper', None)
    if not qp:
        return _q_rows(s)
    return [[n, float(qp.get(n, 1e-4))] for n in s.state_names]


def _r_rows_default(s):
    rp = getattr(s, 'r_paper', None)
    if not rp:
        return _r_rows(s)
    return [[m, float(rp.get(m, _default_r(s)))] for m in s.measurement_names]


def _x0_rows(s):
    """ Per-state initial-guess table: [[state, guess-or-blank], ...], blank
    everywhere by default (every state starts from the seeded random draw). """
    return [[n, None] for n in s.state_names]


def _p0_rows(s):
    """ Per-state P₀ table: [[state, variance-or-blank], ...].

    A spec's published P₀ (alt2d's p0_paper = 10) fills every row, so the paper
    figures keep reproducing what the single broadcast box used to give. Every
    other system starts blank, meaning each state's P₀ follows its initial
    guess (see _estimation_problem). """
    pp = getattr(s, 'p0_paper', None)
    if isinstance(pp, dict):                        # already per-state
        return [[n, (float(pp[n]) if pp.get(n) else None)]
                for n in s.state_names]
    v = float(pp) if pp else None
    return [[n, v] for n in s.state_names]


def _x0_vec(rows, names):
    """ Initial-guess table -> array aligned to `names`, NaN where the row was
    left blank, or None when the whole table is blank. Values pass through
    verbatim — negative guesses are ordinary. """
    if rows is None:
        return None
    if hasattr(rows, 'values'):                     # DataFrame
        rows = rows.values.tolist()
    v = np.full(len(names), np.nan)
    idx = {n: i for i, n in enumerate(names)}
    for row in rows or []:
        if not row or row[0] not in idx:
            continue
        try:
            v[idx[row[0]]] = (float(row[1])
                              if len(row) > 1 and row[1] not in (None, '')
                              else np.nan)
        except (TypeError, ValueError):
            pass                                    # unparseable -> stays blank
    return None if not np.any(np.isfinite(v)) else v


def _p0_vec(rows, names):
    """ P₀ table -> array aligned to `names`, NaN where the row was left blank
    (that state's P₀ then follows its initial guess — see _estimation_problem),
    or None when the whole table is blank.

    A variance must be > 0 to mean anything, so zero/negative/unparseable
    entries are treated as blank rather than silently producing a singular P₀.

    A bare scalar is still accepted and broadcasts over every state: that is
    what the Streamlit front-end's single P₀ box passes, and what the Gradio
    per-state table replaced. """
    if rows is None:
        return None
    if isinstance(rows, (int, float, str)):         # scalar broadcast
        try:
            v = float(rows)
        except (TypeError, ValueError):
            return None
        return np.full(len(names), v) if np.isfinite(v) and v > 0 else None
    if hasattr(rows, 'values'):                     # DataFrame
        rows = rows.values.tolist()
    v = np.full(len(names), np.nan)
    idx = {n: i for i, n in enumerate(names)}
    for row in rows or []:
        if not row or row[0] not in idx:
            continue
        try:
            val = (float(row[1])
                   if len(row) > 1 and row[1] not in (None, '') else np.nan)
        except (TypeError, ValueError):
            continue                                # unparseable -> stays blank
        v[idx[row[0]]] = val if np.isfinite(val) and val > 0 else np.nan
    return None if not np.any(np.isfinite(v)) else v


# The ANN / AI-KF estimators are implemented (engine/ann_estimator.py and
# ObservabilityEngine.run_aikf) but not yet validated well enough to ship: their
# accuracy depends heavily on the synthesized training set. Their controls stay
# visible and documented but greyed out, and the compute path is hard-gated.
# Flip this to True to re-enable them.
ANN_ENABLED = False

TIME_ROW = 'time [s]'


def _axis_rows(names):
    """ Blank axis-limit table: one shared time (x) row, then one row per
    channel. Folding x into the same table means one control instead of a pair
    of number boxes beside it. """
    return [[TIME_ROW, None, None]] + [[n, None, None] for n in names]


def _axis_lims(rows):
    """ axis-limit table -> (xlim, {name: ylim}). The time row is the shared x
    limit; every other row is that channel's y limit. """
    d = _ylim_dict(rows)
    return d.pop(TIME_ROW, None), d


def _ylim_rows(names):
    """ Blank per-row y-limit table: [[name, ymin, ymax], ...]. """
    return [[n, None, None] for n in names]


def _ylim_dict(rows):
    """ y-limit table -> {name: (lo, hi)}, skipping rows left blank. Accepts
    the list-of-lists Gradio sends, or a DataFrame. """
    out = {}
    if rows is None:
        return out
    if hasattr(rows, 'values'):                 # DataFrame
        rows = rows.values.tolist()
    for row in rows:
        if not row or row[0] in (None, ''):
            continue
        lim = _lim(row[1] if len(row) > 1 else None,
                   row[2] if len(row) > 2 else None)
        if lim:
            out[str(row[0])] = lim
    return out


def _lim(lo, hi):
    """ Two limit boxes -> a (lo, hi) pair for matplotlib, or None when both
    are blank. Either side may stay None to keep that end on autoscale. """
    def f(v):
        try:
            return None if v is None or v == '' else float(v)
        except (TypeError, ValueError):
            return None
    lo, hi = f(lo), f(hi)
    return None if lo is None and hi is None else (lo, hi)


def _parse_layers(text, default=(64, 64, 64)):
    """ '64, 64, 64' -> (64, 64, 64). Falls back to the repo default on any
    unparseable entry rather than raising inside a callback. """
    try:
        vals = tuple(int(float(t)) for t in str(text).replace(';', ',')
                     .split(',') if t.strip())
        vals = tuple(v for v in vals if v > 0)
        return vals or default
    except (TypeError, ValueError):
        return default


def _uniform_dict(default, names):
    return {n: max(float(default), 1e-30) for n in names}


def _default_r(spec):
    rd = getattr(spec, 'r_default', None)
    return float(list(rd.values())[0]) if rd else 0.1


def _r_rows(spec):
    d = _default_r(spec)
    return [[m, d] for m in spec.measurement_names]


def _q_rows(spec):
    return [[s, 1e-4] for s in spec.state_names]


def _table_dict(rows, names, default):
    """ Build a per-name noise dict from an editable table (list of
    [name, value] rows); names not in the table fall back to `default`. """
    d = _uniform_dict(default, names)
    if rows is None:
        return d
    if hasattr(rows, 'values'):                 # pandas DataFrame → list of rows
        rows = rows.values.tolist()
    for row in rows:
        try:
            name, val = str(row[0]), float(row[1])
        except (IndexError, TypeError, ValueError):
            continue
        if name in d:
            d[name] = max(val, 1e-30)
    return d


# ─────────────────────── drawn / clicked trajectories ────────────────────────

def canvas_to_recorded(spec, pts_px, W, H, v0, dur):
    """ Canvas pixels → a (speed, heading) set-point pair, shared by both
    front-ends' drawing tools.

    `pts_px` is an (n, 2) array in canvas pixel coordinates with y measured DOWN
    from the top, `W`/`H` the canvas size; they are mapped onto the system's
    rec_xlim/rec_ylim arena box (flipping y) and resampled to N steps.

    Step count N — which drives MPC solve time — is set by `dur`: if dur > 0 the
    whole path is traversed in dur seconds → N = dur/dt, independent of how long
    the path is (speed adjusts to cover it). If dur == 0, fall back to constant
    speed v0 → N grows with path length.

    Returns (recorded, note), or (None, message) if there aren't two distinct
    points to work with. """
    W, H = (W or 1), (H or 1)
    xr, yr = spec.rec_xlim, spec.rec_ylim
    # The points come from JavaScript, so they are whatever arrived: a ragged
    # list, strings, or not a sequence at all. Anything unreadable is "no path
    # drawn yet", which is a message rather than a traceback out of a callback.
    try:
        raw = (np.asarray(pts_px, dtype=float).reshape(-1, 2)
               if len(pts_px) else np.empty((0, 2)))
    except (ValueError, TypeError):
        raw = np.empty((0, 2))
    pts = (np.column_stack([xr[0] + raw[:, 0] / W * (xr[1] - xr[0]),
                            yr[1] - raw[:, 1] / H * (yr[1] - yr[0])])
           if len(raw) else np.empty((0, 2)))
    if len(pts) >= 2:                                  # drop repeated points
        keep = np.concatenate([[True],
                               np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9])
        pts = pts[keep]
    if len(pts) < 2:
        return None, '*Drag to draw a path, or click ≥2 waypoints, then apply.*'
    sarc = np.concatenate([[0.0],
                           np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    arc, dt, dur = sarc[-1], spec.dt, float(dur or 0.0)
    if dur > 0:                                        # fixed duration → fixed N
        N = int(np.clip(round(dur / dt), 8, 800))
    else:                                              # constant v0 → N ∝ length
        N = int(np.clip(round(arc / (max(float(v0), 1e-3) * dt)), 8, 800))
    si = np.linspace(0.0, arc, N)
    xi = np.interp(si, sarc, pts[:, 0])
    yi = np.interp(si, sarc, pts[:, 1])
    heading = np.unwrap(np.arctan2(np.gradient(yi), np.gradient(xi)))
    v_eff = arc / ((N - 1) * dt) if N > 1 else max(float(v0), 1e-3)
    note = (f'*Recorded {arc:.2f}-unit path → {N} steps ({N * dt:g}s, '
            f'v≈{v_eff:.2g}). Press ▶ Simulate.*')
    return ([v_eff] * N, heading.tolist()), note
