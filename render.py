"""
Matplotlib rendering for the observability explorer — adapted verbatim from
``observability_app.ObservabilityApp`` (the PyQt desktop app), with the Qt
plumbing removed. Every function takes a fresh :class:`matplotlib.figure.Figure`
and draws one panel into it, so the Gradio layer can just hand each figure to a
``gr.Plot``. The science (which arrays get plotted, colour scales, wind glyph,
±2σ bands) is unchanged from the desktop version.
"""
import io
import contextlib

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection

from pybounds import colorline

# larger, more readable fonts across every panel
mpl.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 13,
    'legend.fontsize': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'figure.titlesize': 15})

# legends sit outside on the right; reserve that strip when laying out
_LEG = dict(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=11,
            frameon=False)
_TIGHT = dict(rect=(0, 0, 0.82, 1))

# width of the plotted uncertainty band, in standard deviations
_NSIG = 2.0


def render_wind(ax, engine, spec, q_zeta, w):
    """ Small standalone wind panel: an arrow along ζ (magnitude w) plus the
    ±1σ(ζ) process-noise fan. Drawn beside the trajectory panels. """
    # limits hug the glyph so the arrow fills the panel instead of floating in it
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal'); ax.axis('off')
    if 'w' not in spec.state_names or 'zeta' not in spec.state_names:
        ax.set_title('wind', fontsize=12)
        ax.text(0, 0, 'no wind\nstate', ha='center', va='center',
                fontsize=10, color='gray')
        return
    X = engine.X
    w_wind = X[0, spec.state_names.index('w')]
    zeta = X[0, spec.state_names.index('zeta')]
    # q_zeta is the PER-STEP covariance Q_d (the UI's convention, cf. engine
    # _lin, which divides by dt to recover the PSD). ζ is a random walk, so its
    # drift variance over an n_w-step window is Q_d·n_w  ( = Q_c·T_w ). Using
    # T_w in place of n_w here would understate σ by a factor √dt.
    n_w = min(w, max(engine.N - 1, 1))
    half = min(np.sqrt(max(q_zeta, 0.0) * n_w), np.pi)
    if half > 0.01:
        ax.add_patch(mpatches.Wedge((0, 0), 1.0, np.degrees(zeta - half),
                                    np.degrees(zeta + half), facecolor='steelblue',
                                    alpha=0.25, lw=0))
    ax.add_patch(mpatches.FancyArrow(
        -1.0 * np.cos(zeta), -1.0 * np.sin(zeta),
        2.0 * np.cos(zeta), 2.0 * np.sin(zeta), width=0.13,
        head_width=0.38, head_length=0.42, color='steelblue',
        length_includes_head=True))
    lbl = f'wind {w_wind:.2g} m/s\nζ = {zeta:.2f} rad'
    if half > 0.01:
        lbl += f'\n±1σ(ζ)={half:.2f}'
    ax.set_title(lbl, fontsize=12, color='steelblue')


def _heading_wedges(x, y, color, orientation_rad, norm, cmap, size_radius,
                    size_angle=20.0, nskip=0, center_offset_fraction=0.75):
    """ Paper-style heading arrowheads: a PatchCollection of colored wedges
    pointing along `orientation_rad`. Vendored/simplified from fly_plot_lib's
    get_wedges_for_heading_plot (that module isn't in the installed pybounds). """
    idx = np.arange(0, len(x), nskip + 1)
    orient = np.asarray(orientation_rad, dtype=float) * 180.0 / np.pi + 180.0
    wedges = []
    for i in idx:
        cx = x[i] - np.cos(np.radians(orient[i])) * size_radius * center_offset_fraction
        cy = y[i] - np.sin(np.radians(orient[i])) * size_radius * center_offset_fraction
        wedges.append(mpatches.Wedge((cx, cy), size_radius,
                                     orient[i] - size_angle / 2.0,
                                     orient[i] + size_angle / 2.0))
    pc = PatchCollection(wedges, cmap=cmap, norm=norm)
    pc.set_edgecolors('none')
    pc.set_linewidth(0.25)
    pc.set_array(np.asarray(color)[idx])
    return pc


# ─────────────────────────── small helpers ──────────────────────────────────

def _apply_limits(ax, xlim=None, ylim=None):
    """ Apply user axis limits. Each of `xlim`/`ylim` is a (lo, hi) pair in
    which either entry may be None, meaning 'leave that side on autoscale'. """
    for setter, lim in ((ax.set_xlim, xlim), (ax.set_ylim, ylim)):
        if not lim:
            continue
        lo, hi = lim
        if lo is None and hi is None:
            continue
        if lo is not None and hi is not None and lo == hi:
            continue                  # degenerate range — ignore rather than error
        setter(lo, hi)


def wrap_pi(a):
    """ Wrap an angle (array) into (−π, π]. """
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


_PI_MARGIN = 0.1   # rad of head-room past ±π so edge cases at the wrap show


def _set_pi_yaxis(ax):
    ax.set_ylim(-np.pi - _PI_MARGIN, np.pi + _PI_MARGIN)
    ax.set_yticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
    ax.set_yticklabels(['−π', '−π/2', '0', 'π/2', 'π'])


def meas_default(spec):
    picks = tuple(m for m in ('gamma', 'a', 'g')
                  if m in spec.measurement_names)
    return picks or tuple(spec.measurement_names[:3])


def _center_shift(engine, w):
    """ Paper convention (Methods step 7): attribute each window's value to its
    center by shifting the window-start time forward by round(w/2) steps. """
    return int(np.round(w / 2.0)) * engine.dt


def _ev_along_path(engine, src, cs, shift=0.0, hold=True):
    """ Map a min-EV column onto the engine's time grid. `hold=True` extends the
    first/last value over the un-windowed ends so the trajectory is fully
    colored; `hold=False` leaves them NaN, which is what a heatmap wants — held
    values there would read as real measurements. """
    ev_path = np.full(engine.N, np.nan)
    vals = src[cs].values
    ti = src['time_initial'].values + shift
    ok = np.isfinite(ti) & np.isfinite(vals)
    idx = np.rint(ti[ok] / engine.dt).astype(int)
    keep = (idx >= 0) & (idx < engine.N)          # shifted window may run off the end
    ev_path[idx[keep]] = vals[ok][keep]
    fin = np.flatnonzero(np.isfinite(ev_path))
    if hold and len(fin):
        ev_path[:fin[0]] = ev_path[fin[0]]
        ev_path[fin[-1]:] = ev_path[fin[-1]]
    return ev_path


def _draw_wind_glyph(ax, engine, spec, q_zeta, w, show_sigma=True):
    if 'w' not in spec.state_names or 'zeta' not in spec.state_names:
        return
    X = engine.X
    w_wind = X[0, spec.state_names.index('w')]
    zeta = X[0, spec.state_names.index('zeta')]
    n_w = min(w, max(engine.N - 1, 1))              # window length in STEPS
    T_w = n_w * engine.dt                           # ... and in seconds (label)
    if show_sigma:
        # q_zeta is the PER-STEP covariance Q_d (the UI's convention, cf. engine
        # _lin, which divides by dt to recover the PSD). ζ is a random walk, so
        # its drift variance over the window is Q_d·n_w ( = Q_c·T_w ); using T_w
        # here instead of n_w would understate σ by a factor √dt.
        sigma = np.sqrt(max(q_zeta, 0.0) * n_w)
        half = min(sigma, np.pi)                     # ±1σ (68%) each side
    else:
        half = 0.0
    cx, cy, r = 0.11, 0.88, 0.08
    tf = ax.transAxes
    if half > 0.01:
        ax.add_patch(mpatches.Wedge(
            (cx, cy), r, np.degrees(zeta - half), np.degrees(zeta + half),
            transform=tf, facecolor='steelblue', alpha=0.25, lw=0,
            zorder=5, clip_on=False))
    ax.add_patch(mpatches.FancyArrow(
        cx - 0.85 * r * np.cos(zeta), cy - 0.85 * r * np.sin(zeta),
        1.7 * r * np.cos(zeta), 1.7 * r * np.sin(zeta),
        transform=tf, width=0.012, color='steelblue',
        length_includes_head=True, zorder=6, clip_on=False))
    label = f'wind {w_wind:.2g} m/s, ζ = {zeta:.2f} rad'
    if half > 0.01:
        label += f'\n±1σ(ζ) = {half:.2f} rad over w = {T_w:.2g} s'
    ax.text(cx + r + 0.03, cy, label, transform=tf, ha='left',
            va='center', fontsize=6.5, color='steelblue', zorder=6,
            clip_on=False,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none',
                      alpha=0.6))


def _heading_of(engine, spec):
    """ Orientation for the arrowhead markers.

    Prefer a real heading state: yaw `psi` when the model has both psi and phi
    (in a 3D model phi is ROLL, which points the arrowheads nowhere useful),
    otherwise phi — which IS the heading in the planar fly models. Models with no
    heading state at all (alt2d) fall back to the direction of travel along the
    plotted path, so the arrowhead style still works there instead of silently
    reverting to a continuous line. """
    names = list(spec.state_names)
    order = ('psi', 'phi') if ('psi' in names and 'phi' in names) else ('phi', 'psi')
    for nm in order:
        if nm in names:
            return engine.X[:, names.index(nm)]
    X = engine.X
    if len(X) < 2:
        return None
    return np.arctan2(np.gradient(X[:, 1]), np.gradient(X[:, 0]))


def _plot_colored_traj(fig, ax, engine, spec, ev_path, norm, cs, src_lab,
                       q_zeta, w, show_sigma=True, arrowhead=False,
                       cmap_name='inferno_r', wind=True, windows=None,
                       colorbar=True):
    """ Returns the color mappable (or None) so a caller drawing several panels
    can attach ONE shared colorbar instead of one per panel. """
    X = engine.X
    cmap = plt.get_cmap(cmap_name)
    ax.plot(X[:, 0], X[:, 1], 'k-', lw=0.3, alpha=0.4)
    # highlight the sliding window(s) feeding the matrix plots below — a bold
    # translucent halo under the colored line, with start/end markers
    if windows:
        N = len(X)
        for lab, k, ww, col in windows:
            k = int(max(0, min(k, N - 1))); e = int(min(k + ww, N))
            ax.plot(X[k:e, 0], X[k:e, 1], '-', color=col, lw=12, alpha=0.6,
                    solid_capstyle='butt', zorder=2, label=lab)
    mappable = None
    if ev_path is not None and norm is not None:
        heading = _heading_of(engine, spec) if arrowhead else None
        if heading is not None:                       # paper-style arrowheads
            span = max(np.ptp(X[:, 0]), np.ptp(X[:, 1]), 1e-6)
            nskip = max(0, len(X) // 60)
            pc = _heading_wedges(X[:, 0], X[:, 1], ev_path, heading, norm, cmap,
                                 size_radius=0.035 * span, size_angle=28.0,
                                 nskip=nskip)
            ax.add_collection(pc)
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
            mappable = sm
        else:                                         # continuous colored line
            with contextlib.redirect_stdout(io.StringIO()):
                lc = colorline(X[:, 0], X[:, 1], ev_path, ax=ax, cmap=cmap,
                               norm=norm, linewidth=4)
            mappable = lc
        if colorbar:
            cb = fig.colorbar(mappable, ax=ax, pad=0.02)
            cb.set_label(f'min EV: {cs}', fontsize=12)
            cb.ax.tick_params(labelsize=10)
    else:
        ax.annotate(f'no EV to show for "{cs}" —\nadd it to the sensor & state '
                    'combination', xy=(0.5, 0.5), xycoords='axes fraction',
                    ha='center', va='center', fontsize=12, color='gray')
    ax.plot(X[0, 0], X[0, 1], 'go', ms=8, zorder=4, label='start')
    if wind:                          # inlaid glyph (off when a wind panel is shown)
        _draw_wind_glyph(ax, engine, spec, q_zeta, w, show_sigma=show_sigma)
    # label with the actual first two states: they are x/y only for the spatial
    # systems — alt2d's pair is (x, z) and mono's is (g, d)
    nm = list(getattr(spec, 'state_names', []))[:2] or ['x', 'y']
    ax.set_xlabel(nm[0]); ax.set_ylabel(nm[-1])
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title(f'min EV of {cs} — {src_lab}', fontsize=13)
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    return mappable


# ─────────────────────────── panel renderers ────────────────────────────────

# line style per min-EV pipeline, so one legend entry explains every curve
_PIPE_STYLE = {'empirical': ('-', 1.8),
               'observability': ('--', 1.4),
               'constructability': (':', 1.9)}
_PIPE_LABEL = {'empirical': 'empirical (Q=0)',
               'observability': 'stochastic observability (Q>0)',
               'constructability': 'stochastic constructability (Q>0)'}
_PIPE_SHORT = {'empirical': 'emp', 'observability': 'obs',
               'constructability': 'con'}


def _pipes_of(payload):
    """ (key, dataframe) for each present min-EV pipeline, empirical first. """
    ev_emp = payload.get('ev_emp')
    ev_lin = payload.get('ev_lin') or {}
    if not isinstance(ev_lin, dict):              # tolerate the old single-df form
        ev_lin = {'observability': ev_lin} if ev_lin is not None else {}
    pipes = ([('empirical', ev_emp)] if ev_emp is not None else [])
    return pipes + [(b, df) for b, df in ev_lin.items() if df is not None]


def render_main(fig, engine, spec, payload, color_state, ev_states,
                arrowhead=False, cmap_name='inferno_r', vmin=1e-4, vmax=1e6,
                windows=None):
    """ Observability panel: one colored-trajectory panel per min-EV pipeline
    (empirical always, plus one per selected stochastic basis), with the min-EV
    sliding-window curves below overlaying all of them for comparison.

    With 2+ trajectory panels the colorbar is shared (one bar between the panels
    and the wind glyph) instead of one per panel, which returns that width to the
    paths. The state x time heatmap of the same data is a separate figure —
    `render_minev_map`. """
    pipes = _pipes_of(payload)
    if not pipes:
        return
    cs = color_state
    ev_states = ev_states or [spec.color_state_default]
    qz, w = payload['q_zeta'], payload['w']
    shift = _center_shift(engine, w)     # paper ω/2 centering (Methods step 7)
    norm = mpl.colors.LogNorm(vmin, vmax)
    n = len(pipes)
    # column order: trajectory panels | shared colorbar | wind. The trailing
    # column is always reserved — it holds the wind glyph when the system has
    # one, and otherwise stays empty so the min-EV legend (drawn outside its
    # axes) still has somewhere to go instead of being clipped.
    share_cb = n >= 2
    has_wind = ('w' in spec.state_names and 'zeta' in spec.state_names)
    widths = [1.0] * n + ([0.05] if share_cb else []) + [0.46]
    cb_col = n if share_cb else None
    wind_col = (n + 1 if share_cb else n) if has_wind else None
    gs = fig.add_gridspec(2, len(widths), width_ratios=widths,
                          height_ratios=[2.25, 1.35],
                          wspace=0.13 if share_cb else 0.24, hspace=0.20)
    mappable = None
    for k, (key, ev) in enumerate(pipes):
        ax = fig.add_subplot(gs[0, k])
        path = _ev_along_path(engine, ev, cs, shift) if cs in ev else None
        m = _plot_colored_traj(fig, ax, engine, spec, path, norm, cs,
                               _PIPE_LABEL.get(key, key), qz, w,
                               show_sigma=False, arrowhead=arrowhead,
                               cmap_name=cmap_name, wind=False, windows=windows,
                               colorbar=not share_cb)
        mappable = mappable if mappable is not None else m
        if share_cb and k:            # only the leftmost panel needs the y label
            ax.set_ylabel('')
    if share_cb and mappable is not None:
        cb = fig.colorbar(mappable, cax=fig.add_subplot(gs[0, cb_col]))
        # caption above the bar, not beside it: a side label lands on top of the
        # tick labels in a column this narrow
        cb.ax.set_title(f'min EV\n{cs}', fontsize=11, pad=8)
        cb.ax.tick_params(labelsize=10)
    if wind_col is not None:
        render_wind(fig.add_subplot(gs[0, wind_col]), engine, spec, qz, w)

    cmap = plt.get_cmap('tab10')
    # curve spans the trajectory columns only — the wind/colorbar columns to its
    # right are where the (outside-the-axes) legend goes, so it isn't clipped
    ax_ev = fig.add_subplot(gs[1, :n])
    handles = []
    for i, st in enumerate(ev_states):
        c = cmap(i % 10)
        for key, df in pipes:
            if st not in df:
                continue
            ls, lw = _PIPE_STYLE.get(key, ('-', 1.4))
            ax_ev.semilogy(df['time_initial'] + shift, df[st], ls, color=c, lw=lw)
        handles.append(Line2D([], [], color=c, label=st))
    for key, _ in pipes:
        ls, lw = _PIPE_STYLE.get(key, ('-', 1.4))
        handles.append(Line2D([], [], color='k', ls=ls,
                              label=_PIPE_LABEL.get(key, key)))
    ax_ev.set_xlabel('time step (s)')
    ax_ev.set_ylabel('min error variance')
    ax_ev.set_title(f'min EV, sliding window w={w}', fontsize=13)
    ax_ev.legend(handles=handles, ncol=1, **{**_LEG, 'fontsize': 10})
    ax_ev.grid(alpha=0.3, which='both')
    # no tight_layout: it discards the wspace/hspace set on the gridspec above,
    # which is what keeps the gutters small.
    fig.subplots_adjust(left=0.055, right=0.945, top=0.94, bottom=0.07)


def render_minev_map(fig, engine, spec, payload, states, cmap_name='inferno_r',
                     vmin=None, vmax=None):
    """ Standalone state x time min-EV heatmap — the same numbers as the min-EV
    curve in `render_main`, on the same 'time step (s)' axis, kept in its own
    figure so neither plot crowds the other. """
    ev_emp = payload.get('ev_emp')
    ev_lin = payload.get('ev_lin') or {}
    if not isinstance(ev_lin, dict):
        ev_lin = {'observability': ev_lin} if ev_lin is not None else {}
    pipes = ([('empirical', ev_emp)] if ev_emp is not None else [])
    pipes += [(b, df) for b, df in ev_lin.items() if df is not None]
    if not pipes:
        return
    shift = _center_shift(engine, payload['w'])
    _minev_heatmap(fig, fig.subplots(1, 1), engine, pipes, list(states),
                   shift, cmap_name, vmin, vmax)
    # tight_layout (not subplots_adjust) so the colorbar keeps the width it
    # claimed — a manual right margin crops its ticks off the canvas.
    fig.tight_layout()


def _minev_heatmap(fig, ax, engine, pipes, states, shift, cmap_name,
                   vmin=None, vmax=None):
    """ min-EV as a log heatmap in the pybounds example's orientation: states on
    the x-axis, time steps down the y-axis, one shared log color scale.

    Columns are grouped STATE-major so each state's pipelines sit next to each
    other — that is the comparison this panel exists for. The ω/2-centering pad at
    each end holds the first/last valid value, matching what the pybounds examples
    do (`EV_aligned.fillna(method='bfill').fillna(method='ffill')`). """
    cols, labels, bounds = [], [], []
    for c in states:
        present = [(k, df) for k, df in pipes if c in df.columns]
        if not present:
            continue
        for key, df in present:
            # the pipelines have different row counts (pybounds pads for its ω/2
            # centering, the recursion does not), so resample each series onto the
            # engine's time grid — the same mapping the trajectory coloring uses.
            cols.append(_ev_along_path(engine, df, c, shift, hold=True))
            labels.append(f'{c} · {_PIPE_SHORT.get(key, key[:3])}')
        bounds.append(len(cols))              # group boundary for a separator
    if not cols:
        ax.set_visible(False)
        return
    # np.vstack gives (series, time); transpose so time runs down the y-axis
    data = np.clip(np.vstack(cols), 1e-30, None).T
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad('#d9d9d9')       # only reachable if a state is NaN throughout
    # y extent from the engine time grid, time increasing downward
    ext = [-0.5, len(cols) - 0.5, float(engine.t[-1]), float(engine.t[0])]
    im = ax.imshow(np.ma.masked_invalid(data), aspect='auto', cmap=cmap,
                   extent=ext, norm=mpl.colors.LogNorm(vmin, vmax))
    for b in bounds[:-1]:                     # thin rule between state groups
        ax.axvline(b - 0.5, color='w', lw=1.2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.tick_params(axis='x', length=0, pad=2)
    ax.set_xlabel('state'); ax.set_ylabel('time step (s)')
    ax.set_title('min error variance — state × time '
                 '(same data as the min-EV curve)', fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='min EV')


def render_position(fig, engine, spec):
    t, X = engine.t, engine.X
    n0, n1 = spec.state_names[0], spec.state_names[1]
    axs = fig.subplots(2, 1, sharex=True, squeeze=False)[:, 0]
    axs[0].plot(t, X[:, 0], color='#1f77b4', lw=1.7)
    axs[0].set_ylabel(f'{n0}')
    axs[1].plot(t, X[:, 1], color='#d62728', lw=1.7)
    axs[1].set_ylabel(f'{n1}')
    for a in axs:
        a.grid(alpha=0.3)
    axs[1].set_xlabel('time [s]')
    fig.suptitle('trajectory position over time', fontsize=10)
    fig.tight_layout()


def render_measurements(fig, engine, spec, meas, meas_sel,
                        show_noisy=True, show_true=True, show_pred=True,
                        show_ukf=False, show_ann=False, show_aikf=False,
                        xlim=None, ylims=None):
    """ Measured vs predicted sensor traces. Each estimator's prediction is
    h(x̂, u) evaluated on ITS OWN state estimate, so a sensor channel shows which
    estimator explains the data. The ANN estimates a single state, so it has no
    full state vector of its own: its prediction substitutes the ANN's estimate
    for that one state into the AI-KF's state vector (labelled accordingly). """
    t = engine.t
    sensors = (meas_sel or list(meas_default(spec)))[:8]
    # Y_hat may be absent if that filter failed — the loop below skips None
    Y_noisy, Y_true = meas['Y_noisy'], meas['Y_true']
    Y_hat = meas.get('Y_hat')
    # (flag, key in `meas`, style, color, label)
    preds = [(show_pred, None, '--', 'royalblue', 'EKF prediction h(x̂)'),
             (show_ukf, 'Y_hat_ukf', '-.', 'darkorange', 'UKF prediction h(x̂)'),
             (show_ann, 'Y_hat_ann', ':', '0.35', 'ANN prediction h(x̌)'),
             (show_aikf, 'Y_hat_aikf', '-', 'crimson', 'AI-KF prediction h(x̂)')]
    labels = getattr(spec, 'measurement_labels', {}) or {}
    axs = fig.subplots(max(len(sensors), 1), 1, sharex=True, squeeze=False)[:, 0]
    for i, m in enumerate(sensors):
        ax = axs[i]
        j = spec.measurement_names.index(m)
        ang = m in spec.angle_measurements
        wr = wrap_pi if ang else (lambda a: a)
        if show_noisy:
            ax.plot(t, wr(Y_noisy[:, j]), '.', color='0.55', ms=3.5,
                    label='noisy samples (R)')
        if show_true:
            ax.plot(t, wr(Y_true[:, j]), 'k-', lw=1.6, label='noise-free h(x,u)')
        for on, key, ls, col, lab in preds:
            if not on:
                continue
            Yp = Y_hat if key is None else meas.get(key)
            if Yp is None:
                continue
            ax.plot(t, wr(Yp[:, j]), ls, color=col, lw=1.4, label=lab)
        ax.set_ylabel(labels.get(m, m + (' [rad]' if ang else '')))
        ax.grid(alpha=0.3)
        if ang:
            _set_pi_yaxis(ax)
        # per-sensor y limits, shared x (time)
        _apply_limits(ax, xlim, (ylims or {}).get(m))
        if i == 0:
            ax.legend(**_LEG)
    for ax in axs[len(sensors):]:
        ax.set_visible(False)
    axs[min(len(sensors), len(axs)) - 1].set_xlabel('time [s]')
    shown = ['EKF'] * bool(show_pred) + ['UKF'] * bool(show_ukf) \
        + ['ANN'] * bool(show_ann) + ['AI-KF'] * bool(show_aikf)
    fig.suptitle('measurements: noisy (R) | true'
                 + (' | ' + '/'.join(shown) if shown else ''), fontsize=14)
    fig.tight_layout(**_TIGHT)


def inputs_default(spec):
    """ Every input channel. For alt2d these are measured accelerations; for the
    fly/drone they are commands. Either way the panel shows the clean signal
    against whatever noise/bias the estimator was actually driven with. """
    return tuple(spec.input_names)


def render_inputs(fig, engine, spec, meas, show_noisy=True, show_true=True,
                  xlim=None, ylims=None):
    """ Measured process inputs: the clean signal against the noisy/biased
    reading the filters were actually driven with (paper Fig. 4b's vertical and
    forward acceleration panels).

    These are NOT measurements in the filter's sense — they enter the prediction
    step, not the correction — so there is no h and no predicted trace. Whatever
    corruption is shown here propagates through the process model, which is why a
    bias on one of them walks the state estimate away with nothing to pull it
    back. """
    t = engine.t
    chans = inputs_default(spec)
    labels = getattr(spec, 'input_labels', {}) or {}
    axs = fig.subplots(max(len(chans), 1), 1, sharex=True, squeeze=False)[:, 0]
    for i, u in enumerate(chans):
        ax = axs[i]
        j = spec.input_names.index(u)
        if show_noisy and meas.get('U_noisy') is not None:
            ax.plot(t, meas['U_noisy'][:, j], '.', color='0.55', ms=3.5,
                    label='measured (noise + bias)')
        if show_true and meas.get('U_true') is not None:
            ax.plot(t, meas['U_true'][:, j], 'k-', lw=1.8, label='true')
        ax.set_ylabel(labels.get(u, u))
        ax.grid(alpha=0.3)
        _apply_limits(ax, xlim, (ylims or {}).get(u))
        if i == 0:
            ax.legend(**_LEG)
    for ax in axs[len(chans):]:
        ax.set_visible(False)
    axs[min(len(chans), len(axs)) - 1].set_xlabel('time [s]')
    fig.suptitle('inputs: true | as the filters received them', fontsize=14)
    fig.tight_layout(**_TIGHT)


def render_estimates(fig, engine, spec, results, est_states,
                     show_true=True, show_ekf=True, show_ukf=True,
                     ann=None, show_ann=False, show_aikf=False,
                     xlim=None, ylims=None):
    """ Per-state estimate traces. The Kalman filters carry a ±2σ band from their
    covariance; the ANN and AI-KF do not produce a covariance, so they are drawn
    as bare lines. `ann` = {state: {raw, filt, beta}} (or {'error': msg}). """
    t, X = engine.t, engine.X
    states = (est_states or [spec.color_state_default])[:8]
    axs = fig.subplots(max(len(states), 1), 1, sharex=True, squeeze=False)[:, 0]
    colors = {'EKF': 'royalblue', 'UKF': 'darkorange'}
    styles = {'EKF': '--', 'UKF': '-.'}
    keep = ({'EKF'} if show_ekf else set()) | ({'UKF'} if show_ukf else set())
    for i, st in enumerate(states):
        ax = axs[i]
        j = spec.state_names.index(st)
        ang = st in spec.angle_states
        wr = wrap_pi if ang else (lambda a: a)
        for name, (X_hat, P_diag) in results.items():
            if name not in keep:
                continue
            sd = np.sqrt(np.clip(P_diag[:, j], 0.0, None))
            m = wr(X_hat[:, j])
            ax.fill_between(t, m - _NSIG * sd, m + _NSIG * sd,
                            color=colors[name], alpha=0.12, lw=0)
            ax.plot(t, m, styles[name], color=colors[name], lw=1.5,
                    label=f'{name} ±{_NSIG:g}σ')
        # ANN raw has no covariance, so no band. The AI-KF is a full filter and
        # does — plus the motif-driven R it assigns to the ANN channel, drawn on a
        # twin log axis so the gate behind the trace is visible.
        a = (ann or {}).get(st)
        if a and 'error' not in a:
            if show_ann:
                ax.plot(t, wr(a['raw']), ':', color='0.45', lw=1.4,
                        label=r'ANN raw $\check{x}$')
            if show_aikf:
                sd = np.sqrt(np.clip(a['P_diag'][:, j], 0.0, None))
                m = wr(a['X_hat'][:, j])
                ax.fill_between(t, m - _NSIG * sd, m + _NSIG * sd,
                                color='crimson', alpha=0.10, lw=0)
                ax.plot(t, m, '-', color='crimson', lw=1.7,
                        label=rf'AI-KF $\hat{{x}}$ ±{_NSIG:g}σ')
        elif a and (show_ann or show_aikf):
            ax.annotate(f'ANN unavailable — {a["error"]}', xy=(0.02, 0.06),
                        xycoords='axes fraction', fontsize=8, color='crimson')
        if show_true:
            ax.plot(t, wr(X[:, j]), 'k-', lw=1.8, label='true')
        ax.set_ylabel(st + (' [rad]' if ang else ''))
        ax.grid(alpha=0.3)
        if ang:
            _set_pi_yaxis(ax)
        # y limits are PER STATE (states have unrelated units and ranges); x is
        # shared because it is time. Applied after the π axis so the user wins.
        _apply_limits(ax, xlim, (ylims or {}).get(st))
        if i == 0:
            ax.legend(**_LEG)
    for ax in axs[len(states):]:
        ax.set_visible(False)
    axs[min(len(states), len(axs)) - 1].set_xlabel('time [s]')
    shown = ['true'] * bool(show_true) + ['EKF'] * bool(show_ekf) \
        + ['UKF'] * bool(show_ukf) + ['ANN'] * bool(show_ann) \
        + ['AI-KF'] * bool(show_aikf)
    fig.suptitle('states — ' + ' / '.join(shown or ['(none selected)']),
                 fontsize=14)
    fig.tight_layout(**_TIGHT)


def _diverging_matrix(fig, ax, df, title, vmax=None, colorbar=True):
    """ Colored matrix on a diverging scale symmetric about 0 (paper style).
    Pass `vmax` to share one scale across side-by-side panels, and
    `colorbar=False` when the caller draws a single shared bar for the row. """
    v = np.asarray(df.values, dtype=float)
    a = vmax if vmax is not None else (np.nanpercentile(np.abs(v), 99) or 1.0)
    im = ax.imshow(v, aspect='auto', cmap='bwr', vmin=-a, vmax=a)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels([str(c) for c in df.columns], rotation=90, fontsize=9)
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def _meas_rows(ax, df):
    """ Label matrix rows by measurement = sensor name + time-step. """
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels([f'{s}[{t}]' for s, t in df.index], fontsize=8)
    ax.set_ylabel('measurements'); ax.set_xlabel('state')


def _shared_vmax(*dfs):
    return max((np.nanpercentile(np.abs(d.values), 99) for d in dfs), default=1.0) or 1.0


def _panel_row(fig, npanels):
    """ A row of matrix panels plus a dedicated colorbar column, laid out
    explicitly. tight_layout cannot place a colorbar that spans several axes — it
    warns and then draws the bar on top of the last panel — so the geometry is
    set here instead. Returns (panel axes, colorbar axes). """
    gs = fig.add_gridspec(1, npanels + 1,
                          width_ratios=[1.0] * npanels + [0.05], wspace=0.12)
    axs = [fig.add_subplot(gs[0, i]) for i in range(npanels)]
    cax = fig.add_subplot(gs[0, npanels])
    # bottom leaves room for the rotated state labels, left for the row labels
    fig.subplots_adjust(left=0.075, right=0.945, top=0.90, bottom=0.21)
    return axs, cax


def render_obs_matrix(fig, O_emp, k=None, O_stoch=None):
    """ Paper Fig 1e: observability matrix 𝒪 for one sliding window. Rows =
    measurements (one sensor at one time-step), columns = states. `O_stoch` is a
    {basis: dataframe} mapping; each selected basis gets a panel beside the
    empirical one, all on a shared color scale. """
    kd = f' — window k = {k}' if k is not None else ''
    O_stoch = O_stoch or {}
    if not isinstance(O_stoch, dict):                 # tolerate the old single-df form
        O_stoch = {'stochastic': O_stoch}
    if not O_stoch:
        ax = fig.subplots(1, 1)
        _diverging_matrix(fig, ax, O_emp, r'observability matrix $\mathcal{O}$' + kd)
        _meas_rows(ax, O_emp)
        fig.tight_layout()
    else:
        # all panels are on one shared scale, so they get ONE colorbar; the row
        # labels are identical too, so only the leftmost panel carries them
        vmax = _shared_vmax(O_emp, *O_stoch.values())
        axs, cax = _panel_row(fig, 1 + len(O_stoch))
        im = _diverging_matrix(fig, axs[0], O_emp, r'empirical $\mathcal{O}$' + kd,
                               vmax=vmax, colorbar=False)
        _meas_rows(axs[0], O_emp)
        for ax, (basis, df) in zip(axs[1:], O_stoch.items()):
            _diverging_matrix(fig, ax, df, rf'$\mathcal{{O}}$ — {basis}' + kd,
                              vmax=vmax, colorbar=False)
            _meas_rows(ax, df)
            ax.set_yticklabels([]); ax.set_ylabel('')
        fig.colorbar(im, cax=cax)


def render_fisher_inv(fig, F_emp, F_stoch=None, k=None, cmap_name='inferno_r',
                      vmin=1e-1, vmax=1e6):
    """ Paper Fig 1g: inverse Fisher information |F⁻¹| for a single sliding
    window (state × state), on the SAME log color scale as the min-error-
    variance map — so the diagonal (the per-state min error variance) matches
    that map cell-for-cell. Magnitude only: off-diagonal covariance signs are
    not shown (a log scale can't carry them). `F_stoch` is a {basis: dataframe}
    mapping; each selected basis gets a panel beside the empirical one. """
    kd = f' — window k = {k}' if k is not None else ''
    norm = mpl.colors.LogNorm(vmin, vmax)

    def one(ax, df, title, colorbar=True, ylabels=True):
        data = np.clip(np.abs(np.asarray(df.values, dtype=float)), 1e-30, None)
        im = ax.imshow(data, aspect='auto', cmap=cmap_name, norm=norm)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels([str(c) for c in df.columns], rotation=90, fontsize=9)
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels([str(r) for r in df.index] if ylabels else [], fontsize=9)
        ax.set_xlabel('state'); ax.set_ylabel('state' if ylabels else '')
        if colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r'$|F^{-1}|$')
        return im

    F_stoch = F_stoch or {}
    if not isinstance(F_stoch, dict):                 # tolerate the old single-df form
        F_stoch = {'stochastic': F_stoch}
    if not F_stoch:
        one(fig.subplots(1, 1), F_emp,
            r'inverse Fisher information $|F^{-1}|$' + kd)
        fig.tight_layout()
    else:
        # every panel already shares `norm`, so one colorbar serves them all
        axs, cax = _panel_row(fig, 1 + len(F_stoch))
        im = one(axs[0], F_emp, r'empirical $|F^{-1}|$' + kd, colorbar=False)
        for ax, (basis, df) in zip(axs[1:], F_stoch.items()):
            one(ax, df, rf'$|F^{{-1}}|$ — {basis}' + kd, colorbar=False,
                ylabels=False)
        fig.colorbar(im, cax=cax, label=r'$|F^{-1}|$')
