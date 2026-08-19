"""
Interactive GUI for exploring observability / minimum error variance.

Built on ipywidgets, designed to run in a notebook (Python311 kernel):

    from observability_gui import ObservabilityGUI
    gui = ObservabilityGUI()          # system='fly' by default
    gui.display()

What it does
------------
* **Two dynamical systems** — the insect-in-wind model from the pybounds
  example (`fly_wind_example.ipynb`, 18 states) and the 3D kinematic drone
  (11 states, `drone_model.py`). The fly's default trajectory is *exactly*
  the pybounds example (MPC tracking v_para/v_perp ramps + a -90 deg heading
  step at t = 0.2 s in wind w = 0.4, zeta = 180 deg) for direct comparison.
* **Trajectory builder** — compose maneuvers from preset motifs (straight,
  turn, circle, speed change, sinusoidal weave) or quick flight-command
  buttons; control inputs always come from the pybounds MPC.
* **Observability panels** — the trajectory colored by min EV of a chosen
  state (linearized Eq. 33 | pybounds empirical, shared color scale), the
  min-EV time series, and min EV vs window length at a chosen start.
* **Interactive parameters** — window size `w`, empirical perturbation `ε`,
  regularizer `λ` (range restricted to stay above the floating-point noise
  floor of F), per-sensor `R` (default 1e-3 each), per-state `Q` presets.

Caching: the linearized pipeline (~0.1 s) recomputes on every change; the
pybounds empirical pipeline is cached per (trajectory, w, ε, R, λ).
"""

import io
import time
import contextlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
try:
    # Only ObservabilityGUI (the notebook front-end at the bottom of this file)
    # touches these, and every use is inside a method. The Gradio and Streamlit
    # apps import this module for the engine alone, so a deployment must not be
    # forced to install the whole Jupyter stack — and Streamlit Cloud would fail
    # the build if it were listed in requirements.txt for no reason.
    import ipywidgets as W
    from IPython.display import display
except ImportError:                     # no notebook GUI; the engine is fine
    W = display = None

from drone_model import (f as drone_f, h as drone_h,
                         STATE_NAMES as DRONE_STATES,
                         INPUT_NAMES as DRONE_INPUTS,
                         MEASUREMENT_NAMES as DRONE_MEAS)
from ann_estimator import (ANNEstimator, mif_beta,
                           motif_informed_filter, motif_R)
from stochastic_observability import (linearize_along_trajectory,
                                      stochastic_observability_gramian,
                                      stochastic_constructability_gramian)
from pybounds import (Simulator, SlidingEmpiricalObservabilityMatrix,
                      SlidingFisherObservability, EmpiricalObservabilityMatrix,
                      FisherObservability, colorline)

# The nonlinear Monte-Carlo stochastic Gramian was removed from the web app.
# Keep a stub so the engine's (now unused) mc_ev* methods still import; calling
# them raises instead of silently doing nothing.
def gramian_recursive_nonlinear(*_a, **_k):
    raise NotImplementedError(
        'nonlinear MC Gramian removed from the observability-gui app')


# ───────────── fly-in-wind model (verbatim from fly_wind_example) ─────────────

FLY_STATES = ['x', 'y', 'z', 'v_para', 'v_perp', 'phi', 'phi_dot', 'w', 'zeta',
              'm', 'I', 'C_para', 'C_perp', 'C_phi', 'km1', 'km2', 'km3', 'km4']
FLY_INPUTS = ['u_para', 'u_perp', 'u_phi']
FLY_MEAS = ['phi', 'psi', 'gamma', 'a', 'g', 'r']

FLY_M = 0.25e-6 * 1e6                     # [mg]
FLY_I = 5.2e-13 * 1e6 * (1e3) ** 2        # [mg*mm^2]
FLY_C_PHI = 27.36e-12 * 1e6 * (1e3) ** 2
FLY_C_PARA = (0.25e-6 / 0.170) * 1e6


def fly_f(X, U):
    x, y, z, v_para, v_perp, phi, phi_dot, w, zeta, m, I, C_para, C_perp, C_phi, km1, km2, km3, km4 = X
    u_para, u_perp, u_phi = U
    a_para = v_para - w * np.cos(phi - zeta)
    a_perp = v_perp + w * np.sin(phi - zeta)
    v_para_dot = ((km1 * u_para - C_para * a_para) / m) + (v_perp * phi_dot)
    v_perp_dot = ((km3 * u_perp - C_perp * a_perp) / m) - (v_para * phi_dot)
    phi_ddot = (km4 * u_phi / I) - (C_phi * phi_dot / I) + (km2 * u_para / I)
    x_dot = v_para * np.cos(phi) - v_perp * np.sin(phi)
    y_dot = v_para * np.sin(phi) + v_perp * np.cos(phi)
    z0 = 0 * x
    return [x_dot, y_dot, z0, v_para_dot, v_perp_dot, phi_dot, phi_ddot,
            z0, z0, z0, z0, z0, z0, z0, z0, z0, z0, z0]


def fly_h(X, U):
    x, y, z, v_para, v_perp, phi, phi_dot, w, zeta, m, I, C_para, C_perp, C_phi, km1, km2, km3, km4 = X
    u_para, u_perp, u_phi = U
    a_para = v_para - w * np.cos(phi - zeta)
    a_perp = v_perp + w * np.sin(phi - zeta)
    a = np.sqrt(a_para ** 2 + a_perp ** 2)
    gamma = np.arctan2(a_perp, a_para)
    g = np.sqrt(v_para ** 2 + v_perp ** 2)
    psi = np.arctan2(v_perp, v_para)
    r = g / z
    if np.array(phi).ndim > 0 and np.array(phi).shape[0] > 1:
        phi = np.unwrap(phi); psi = np.unwrap(psi); gamma = np.unwrap(gamma)
    return [phi, psi, gamma, a, g, r]


# ────── fly-in-wind, 7 primary states (constants folded into parameters) ──────
# Same dynamics/measurements as fly_f/fly_h, but m, I, C_para, C_perp, C_phi and
# km1-4 are fixed parameters instead of states (cf. drone_model.py). This keeps
# the process-noise covariance Q genuinely positive definite for the recursive
# stochastic Gramian — no jitter needed on constant parameter states — at the
# cost of assuming perfect calibration. States: the CDC-2021 paper's 7 primary
# states plus x, y for trajectory plotting.

FLY7_STATES = ['x', 'y', 'z', 'v_para', 'v_perp', 'phi', 'phi_dot', 'w', 'zeta']
FLY_KM1, FLY_KM2, FLY_KM3, FLY_KM4 = 1.0, 0.0, 1.0, 1.0


def fly7_f(X, U):
    x, y, z, v_para, v_perp, phi, phi_dot, w, zeta = X
    u_para, u_perp, u_phi = U
    m, I = FLY_M, FLY_I
    C_para, C_perp, C_phi = FLY_C_PARA, FLY_C_PARA, FLY_C_PHI
    a_para = v_para - w * np.cos(phi - zeta)
    a_perp = v_perp + w * np.sin(phi - zeta)
    v_para_dot = ((FLY_KM1 * u_para - C_para * a_para) / m) + (v_perp * phi_dot)
    v_perp_dot = ((FLY_KM3 * u_perp - C_perp * a_perp) / m) - (v_para * phi_dot)
    phi_ddot = (FLY_KM4 * u_phi / I) - (C_phi * phi_dot / I) + (FLY_KM2 * u_para / I)
    x_dot = v_para * np.cos(phi) - v_perp * np.sin(phi)
    y_dot = v_para * np.sin(phi) + v_perp * np.cos(phi)
    z0 = 0 * x
    return [x_dot, y_dot, z0, v_para_dot, v_perp_dot, phi_dot, phi_ddot, z0, z0]


def fly7_h(X, U):
    x, y, z, v_para, v_perp, phi, phi_dot, w, zeta = X
    u_para, u_perp, u_phi = U
    a_para = v_para - w * np.cos(phi - zeta)
    a_perp = v_perp + w * np.sin(phi - zeta)
    a = np.sqrt(a_para ** 2 + a_perp ** 2)
    gamma = np.arctan2(a_perp, a_para)
    g = np.sqrt(v_para ** 2 + v_perp ** 2)
    psi = np.arctan2(v_perp, v_para)
    r = g / z
    if np.array(phi).ndim > 0 and np.array(phi).shape[0] > 1:
        phi = np.unwrap(phi); psi = np.unwrap(psi); gamma = np.unwrap(gamma)
    return [phi, psi, gamma, a, g, r]


# ── altitude 2D model (Cellini et al. 2025, Fig. 4, Eq. 6-7; + x for plots) ──
# States [x, z, v_z, v_x]; inputs are the (measured) accelerations u_z, u_x;
# the only measurement is forward ventral optic flow r_x = v_x / z. Altitude
# is observable only during nonzero acceleration (paper Fig. 2c-v / Fig. 4b).

ALT2D_STATES = ['x', 'z', 'v_z', 'v_x']
ALT2D_INPUTS = ['u_z', 'u_x']
ALT2D_MEAS = ['r_x']


def alt2d_f(X, U):
    x, z, v_z, v_x = X
    u_z, u_x = U
    return [v_x, v_z, u_z, u_x]


def alt2d_h(X, U):
    x, z, v_z, v_x = X
    # Paper Eq. 7 is r_x = -v_x/z, and the sign is not cosmetic for plotting:
    # Fig. 4b shows forward optic flow running from about -2.5 to -0.7 s^-1.
    # (It IS cosmetic for the filters and the Gramian -- flipping h flips both
    # the innovation and C, and F = C^T R^-1 C is unchanged.)
    return [-v_x / z]


# ───────────────────────── generic RK4 simulator ──────────────────────────

class GenericRK4Simulator:
    """ Minimal fixed-step RK4 simulator, duck-typed for pybounds'
    EmpiricalObservabilityMatrix / SlidingEmpiricalObservabilityMatrix. """

    def __init__(self, f, h, dt, state_names, input_names, measurement_names):
        self.f, self.h, self.dt = f, h, dt
        self.state_names = list(state_names)
        self.input_names = list(input_names)
        self.measurement_names = list(measurement_names)
        self.n, self.m, self.p = (len(self.state_names), len(self.input_names),
                                  len(self.measurement_names))

    def _rk4_step(self, x, u):
        dt, f = self.dt, self.f
        k1 = np.asarray(f(x, u), dtype=float)
        k2 = np.asarray(f(x + 0.5 * dt * k1, u), dtype=float)
        k3 = np.asarray(f(x + 0.5 * dt * k2, u), dtype=float)
        k4 = np.asarray(f(x + dt * k3, u), dtype=float)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def simulate(self, x0=None, u=None, aux=None):
        if isinstance(x0, dict):
            x0 = np.array([x0[k] for k in self.state_names], dtype=float)
        else:
            x0 = np.array(x0, dtype=float).squeeze()
        if isinstance(u, dict):
            u = np.vstack([u[k] for k in self.input_names]).T
        u = np.atleast_2d(np.array(u, dtype=float))
        w = u.shape[0]
        x = np.zeros((w, self.n))
        y = np.zeros((w, self.p))
        x[0] = x0
        y[0] = np.asarray(self.h(x[0], u[0]), dtype=float)
        for k in range(w - 1):
            x[k + 1] = self._rk4_step(x[k], u[k])
            y[k + 1] = np.asarray(self.h(x[k + 1], u[k + 1]), dtype=float)
        return y


# ───────────────────────────── system specs ────────────────────────────────

class SystemSpec:
    """ Everything the engine/GUI needs to know about one dynamical system. """

    def __init__(self, **kw):
        self.__dict__.update(kw)


def make_fly_spec(dt=0.01):

    def make_setpoint(speed_sp, heading_sp, w_wind, zeta):
        ones = np.ones_like(speed_sp)
        return {'x': 0.0 * ones, 'y': 0.0 * ones, 'z': 0.2 * ones,
                'v_para': speed_sp, 'v_perp': 0.0 * ones,
                'phi': heading_sp, 'phi_dot': 0.0 * ones,
                'w': w_wind * ones, 'zeta': zeta * ones,
                'm': FLY_M * ones, 'I': FLY_I * ones,
                'C_para': FLY_C_PARA * ones, 'C_perp': FLY_C_PARA * ones,
                'C_phi': FLY_C_PHI * ones,
                'km1': 1.0 * ones, 'km2': 0.0 * ones,
                'km3': 1.0 * ones, 'km4': 1.0 * ones}

    def default_setpoint(w_wind, zeta):
        # exactly the pybounds fly_wind_example trajectory set-points
        tsim = np.arange(0, 0.4, step=dt)
        sp = make_setpoint(0.3 * np.ones_like(tsim) + 0.01 * tsim,
                           (np.pi / 4) * np.ones_like(tsim), w_wind, zeta)
        sp['v_perp'] = 0.01 * np.ones_like(tsim) + 0.01 * tsim
        k_turn = min(int(round(0.2 / dt)), len(tsim) - 1)
        sp['phi'][k_turn:] = sp['phi'][k_turn] - np.pi / 2   # turn at t = 0.2 s
        return sp

    def build_mpc(sim):
        cost = ((sim.model.x['v_para'] - sim.model.tvp['v_para_set']) ** 2 +
                (sim.model.x['v_perp'] - sim.model.tvp['v_perp_set']) ** 2 +
                (sim.model.x['phi'] - sim.model.tvp['phi_set']) ** 2)
        sim.mpc.set_objective(mterm=cost, lterm=cost)
        sim.mpc.set_rterm(u_para=1e-6, u_perp=1e-6, u_phi=1e-6)

    q_tiny = {s: 1e-8 for s in FLY_STATES}          # jitter floor (see fly nb §4b)
    q_real = dict(q_tiny)
    q_real.update(dict(v_para=2e-2, v_perp=2e-2, phi=1e-5, phi_dot=5.0,
                       w=2e-1, zeta=2e-1))
    return SystemSpec(
        name='fly', dt=dt, f=fly_f, h=fly_h,
        fim_sensors=('phi', 'psi', 'gamma'),
        fim_states=('v_para', 'v_perp', 'phi', 'phi_dot', 'w', 'zeta',
                    'z', 'm', 'I', 'C_para', 'C_perp', 'C_phi'),
        state_names=FLY_STATES, input_names=FLY_INPUTS,
        measurement_names=FLY_MEAS,
        disc_method='rk4var',      # frozen-Jacobian expm is off during the turn
        make_setpoint=make_setpoint, default_setpoint=default_setpoint,
        build_mpc=build_mpc,
        wind_default=0.4, zeta_default=np.pi,
        v0_default=0.3, heading0=np.pi / 4,
        w_default=4, eps_default=1e-4,
        r_default={s: 0.1 for s in FLY_MEAS},   # matches fly_wind_example.ipynb
        q_tiny=q_tiny, q_realistic=q_real,
        color_state_default='zeta',
        ev_states_default=('phi', 'zeta', 'v_para', 'phi_dot'),
        angle_measurements=('phi', 'psi', 'gamma'),
        angle_states=('phi', 'zeta'),
        seg_duration=0.1, quick_turn_dur=0.06, quick_straight_dur=0.12,
        speed_step=0.1, rec_xlim=(-1.0, 1.0), rec_ylim=(-0.5, 0.5),
        default_traj_label='pybounds fly_wind_example trajectory '
                           '(hold π/4, −π/2 step at t = 0.2 s)')


def make_fly7_spec(dt=0.01):

    def make_setpoint(speed_sp, heading_sp, w_wind, zeta):
        ones = np.ones_like(speed_sp)
        return {'x': 0.0 * ones, 'y': 0.0 * ones, 'z': 0.2 * ones,
                'v_para': speed_sp, 'v_perp': 0.0 * ones,
                'phi': heading_sp, 'phi_dot': 0.0 * ones,
                'w': w_wind * ones, 'zeta': zeta * ones}

    def default_setpoint(w_wind, zeta):
        # same trajectory set-points as the pybounds fly_wind_example
        tsim = np.arange(0, 0.4, step=dt)
        sp = make_setpoint(0.3 * np.ones_like(tsim) + 0.01 * tsim,
                           (np.pi / 4) * np.ones_like(tsim), w_wind, zeta)
        sp['v_perp'] = 0.01 * np.ones_like(tsim) + 0.01 * tsim
        k_turn = min(int(round(0.2 / dt)), len(tsim) - 1)
        sp['phi'][k_turn:] = sp['phi'][k_turn] - np.pi / 2
        return sp

    def build_mpc(sim):
        cost = ((sim.model.x['v_para'] - sim.model.tvp['v_para_set']) ** 2 +
                (sim.model.x['v_perp'] - sim.model.tvp['v_perp_set']) ** 2 +
                (sim.model.x['phi'] - sim.model.tvp['phi_set']) ** 2)
        sim.mpc.set_objective(mterm=cost, lterm=cost)
        sim.mpc.set_rterm(u_para=1e-6, u_perp=1e-6, u_phi=1e-6)

    q_tiny = {s: 1e-9 for s in FLY7_STATES}
    q_real = dict(q_tiny)
    q_real.update(dict(v_para=2e-2, v_perp=2e-2, phi=1e-5, phi_dot=5.0,
                       w=2e-1, zeta=2e-1))
    return SystemSpec(
        name='fly7', dt=dt, f=fly7_f, h=fly7_h,
        fim_sensors=('phi', 'psi', 'gamma'),
        fim_states=('z', 'v_para', 'v_perp', 'phi', 'phi_dot', 'w', 'zeta'),
        state_names=FLY7_STATES, input_names=FLY_INPUTS,
        measurement_names=FLY_MEAS,
        disc_method='rk4var',
        make_setpoint=make_setpoint, default_setpoint=default_setpoint,
        build_mpc=build_mpc,
        wind_default=0.4, zeta_default=np.pi,
        v0_default=0.3, heading0=np.pi / 4,
        w_default=4, eps_default=1e-4,
        r_default={s: 0.1 for s in FLY_MEAS},   # matches fly_wind_example.ipynb
        q_tiny=q_tiny, q_realistic=q_real,
        color_state_default='zeta',
        ev_states_default=('phi', 'zeta', 'v_para', 'phi_dot'),
        angle_measurements=('phi', 'psi', 'gamma'),
        angle_states=('phi', 'zeta'),
        seg_duration=0.1, quick_turn_dur=0.06, quick_straight_dur=0.12,
        speed_step=0.1, rec_xlim=(-1.0, 1.0), rec_ylim=(-0.5, 0.5),
        default_traj_label='pybounds fly_wind_example trajectory '
                           '(hold π/4, −π/2 step at t = 0.2 s)')


def make_alt2d_spec(dt=0.1):

    def make_setpoint(speed_sp, heading_sp, w_wind, zeta):
        # heading and wind are meaningless for this 1-D longitudinal model
        ones = np.ones_like(speed_sp)
        return {'x': 0.0 * ones, 'z': 1.5 * ones, 'v_z': 0.0 * ones,
                'v_x': speed_sp}

    def default_setpoint(w_wind, zeta):
        """ Paper Fig. 4b (Cellini et al., arXiv 2511.08766): straight flight
        CLIMBING from z = 1.5 m to 4.5 m, with a brief 3 s deceleration and a
        later 3 s acceleration back to the original speed — altitude is
        observable only while accelerating. The biased vertical-acceleration
        episode is `u_bias_default` below, and the 10^-2 noise variance carried
        by every channel is `u_noise_default` / `r_paper`. """
        # Transcribed from the authors' make/make_altitude_trajectories_initial.ipynb
        # (trajectory index 4, the one plot/plot_altitude_trajectories.ipynb draws
        # for Fig. 4b). Nothing here is a reconstruction any more.
        #
        #   z  = 3*sin(2*pi*(0.5/T)*t) + 1.5, frozen from t = 25 s   -> CLIMBS 1.5 -> 4.5 m
        #   v_x: g0 = 3.5, ramped over 3 s from t = 10 by g_delta = -1.5 (-> 2.0),
        #        then ramped back over 3 s from t = 35 (-> 3.5)
        #
        # The climb is the part that had been missed: altitude is NOT constant, which
        # is why Fig. 4b's optic flow sweeps from about -2.3 toward -0.5 before the
        # manoeuvre even happens (r_x = -v_x/z, and z is growing).
        T = 50.0
        tsim = np.arange(0.0, T + dt / 2, dt)
        z = 3.0 * np.sin(2 * np.pi * (0.5 / T) * tsim) + 1.5
        i_hold = min(250, len(z) - 1)                # notebook: z[250:] = z[250]
        z[i_hold:] = z[i_hold]

        g0, g_delta = 3.5, -1.5
        v = np.full_like(tsim, g0)
        m1 = (tsim >= 10.0) & (tsim < 13.0)          # 3 s deceleration
        v[m1] = g0 + g_delta * (tsim[m1] - 10.0) / 3.0
        v[tsim >= 13.0] = g0 + g_delta
        m2 = (tsim >= 35.0) & (tsim < 38.0)          # 3 s acceleration, back to g0
        v[m2] = (g0 + g_delta) - g_delta * (tsim[m2] - 35.0) / 3.0
        v[tsim >= 38.0] = g0

        sp = make_setpoint(v, None, w_wind, zeta)
        sp['z'] = z
        # make_setpoint pins v_z = 0, which contradicts a climbing z and makes the
        # MPC fight itself — it spends a large u_z transient at t=0 getting the
        # climb started. The authors' notebook only ever sets z and v_x
        # (simulator.update_setpoint(z=z, v_x=g)), so give v_z the value the climb
        # implies. The drone then ascends at essentially constant vertical
        # velocity and the true vertical acceleration stays ~0, as in Fig. 4b.
        sp['v_z'] = np.gradient(z, dt)
        return sp

    def build_mpc(sim):
        mx, mtvp = sim.model.x, sim.model.tvp
        cost = ((mx['z'] - mtvp['z_set']) ** 2 +
                (mx['v_z'] - mtvp['v_z_set']) ** 2 +
                (mx['v_x'] - mtvp['v_x_set']) ** 2)
        sim.mpc.set_objective(mterm=cost, lterm=cost)
        sim.mpc.set_rterm(u_z=1e-4, u_x=1e-4)

    return SystemSpec(
        name='alt2d', dt=dt, f=alt2d_f, h=alt2d_h,
        fim_sensors=('r_x',), fim_states=('z', 'v_z', 'v_x'),
        state_names=ALT2D_STATES, input_names=ALT2D_INPUTS,
        measurement_names=ALT2D_MEAS,
        # Paper Fig. 4b plots three measured signals: forward optic flow,
        # vertical acceleration and forward acceleration. Only the first is a
        # measurement in the filter's sense (Eq. 7); the two accelerations are
        # consumed as process-model inputs. Naming them here lets the
        # measurement panel show them the way the figure does, with the clean
        # signal against the noisy/biased one the filters actually receive.
        measured_inputs=('u_z', 'u_x'),
        # axis labels in the paper's own words, so the panels read like Fig. 4b
        measurement_labels={'r_x': 'forward optic flow\n$r_x$  [1/s]'},
        input_labels={'u_z': 'vertical accel\n$u_z$  [m/s²]',
                      'u_x': 'forward accel\n$u_x$  [m/s²]'},
        disc_method='vanloan',
        make_setpoint=make_setpoint, default_setpoint=default_setpoint,
        build_mpc=build_mpc,
        wind_default=0.0, zeta_default=0.0,
        v0_default=3.5, heading0=0.0,    # g0 in the authors' notebook
        w_default=20, eps_default=1e-5,
        # The paper's Methods text says the optic-flow baseline variance was 10^-3,
        # but both source files say 10^-2 (`noise = 0.01` ->  R_base in
        # make_estimator_monocular_initial_sweep_simulated.m; `base_noise = 1e-2` in
        # make_altitude_trajectories_initial.ipynb). Following the code.
        r_default={'r_x': 1e-2},
        # From make_altitude_trajectories_initial.ipynb, verbatim:
        #   base_noise = 1e-2 on every channel (z, v_z, v_x, r_x, u_z, u_x)
        #   noise_mask = (t > 20.0) & (t < 25.0);  u_z_noise[mask] += 0.2
        # u_z is an IMU reading the filter consumes as a process-model input, so the
        # bias is applied there rather than in the measurement vector.
        u_noise_default=1e-2,
        u_bias_default=('u_z', 0.2, 20.0, 25.0),
        # ── defaults from the authors' repo, altitude/monocular experiment ──
        # make_estimator_monocular_initial_sweep_simulated.m, at the operating
        # point named by the figure file aikf_initial_condition_traj=4_Q=-8.0_P=1.0:
        #   Q = diag([q, q*1e2, q*1e2]) with q = 1e-8   (states z, v_z, v_x)
        #   P = diag([p, p, p])         with p = 1e1
        #   R = diag([R_base, 1e12])    with R_base = 0.01 (their `noise`)
        #   R_bounds = [1e12, 1e-3], motif = |u_x|, window = 20, upper = 0.5
        # UKF tuning from StateEstimator.m; ANN shape/window from
        # make_altitude_trajectories_initial.ipynb (64,64,64 / 20 steps / noise 0).
        q_paper={'x': 1e-8, 'z': 1e-8, 'v_z': 1e-6, 'v_x': 1e-6},
        r_paper={'r_x': 1e-2},
        p0_paper=10.0,          # absolute P0 diagonal, repo p = 10^1
        ukf_paper=(1e-3, 1.0, 0.0),          # Alpha, Beta, Kappa
        # Fig. 4: 'two-second windows of optic flow and forward acceleration'
        # -> 20 steps at dt = 0.1 s; the (r_x, u_x) input pair is what
        # ANNEstimator._default_sensors already derives for this model
        ann_paper=dict(target='z', layers='64, 64, 64', time_steps=20,
                       n_traj=16, epochs=100, batch=256, noise=0.0),
        aikf_paper=dict(motif='u_x', window=20, upper=0.5,
                        r_hi=1e12, r_lo=1e-3),
        q_tiny={s: 1e-9 for s in ALT2D_STATES},
        q_realistic={'x': 1e-6, 'z': 1e-4, 'v_z': 1e-2, 'v_x': 1e-2},
        color_state_default='z',
        ev_states_default=('z', 'v_x'),
        angle_measurements=(), angle_states=(),
        seg_duration=3.0, quick_turn_dur=3.0, quick_straight_dur=5.0,
        speed_step=1.0, rec_xlim=(-250.0, 250.0), rec_ylim=(-125.0, 125.0),
        default_traj_label='paper Fig. 4b: straight flight at z = 1.5 m, '
                           '3 s decel @ t=12, 3 s accel @ t=35 (±1 m/s²)')


def make_drone_spec(dt=0.1):

    def make_setpoint(speed_sp, heading_sp, w_wind, zeta):
        ones = np.ones_like(speed_sp)
        return {'x': 0.0 * ones, 'y': 0.0 * ones, 'z': 10.0 * ones,
                'v_x': speed_sp, 'v_y': 0.0 * ones, 'v_z': 0.0 * ones,
                'phi': 0.0 * ones, 'theta': 0.0 * ones, 'psi': heading_sp,
                'w': w_wind * ones, 'zeta': zeta * ones}

    def default_setpoint(w_wind, zeta):
        """ The Figure 2 trajectory of Cellini et al., arXiv 2511.08766,
        transcribed from the authors' make/make_drone_trajectory_custom.ipynb:
        11 s at z = 2 m holding ground speed g = 1 m/s on a global course of
        -pi/4, with

          t = 1.5 s   decelerate to g/4
          t = 4.0 s   accelerate back to g
          t = 5.0 s   heading turn, course +pi/2
          t = 6.5 s   course +pi/2, and heading offset -pi/2 until t = 7.9 s
                      (the "offset turn": the agent flies one way, points another)
          t = 7.9 s   course +(pi/1.3 - pi/2)
          t = 9.0 s   course +pi/2 plus a 0.6 Hz weave

        Body-frame velocities follow from the course/heading difference,
        v_x = g cos(beta - psi), v_y = g sin(beta - psi).

        NOTE their notebook computes v_y as `-v_x_global*sin(psi) + v_y_global`,
        which is missing the cos(psi) on the second term — it makes v_y nonzero
        even when heading equals course. That looks like a transcription slip, so
        the correct rotation is used here. """
        tsim = np.arange(0.0, 11.0 + dt / 2, dt)
        i = lambda t: int(round((1.0 + t) / dt))       # their t0 = 1 s offset
        g = np.ones_like(tsim)
        g[i(0.5):] -= 0.75                              # decelerate
        g[i(3.0):] += 0.75                              # accelerate
        beta = np.full_like(tsim, -np.pi / 4)           # global course
        beta[i(4.0):] += np.pi / 2
        beta[i(5.5):] += np.pi / 2
        beta[i(6.9):] += np.pi / 1.3 - np.pi / 2
        beta[i(8.0):] += np.pi / 2
        tw = tsim[i(8.0):] - tsim[i(8.0)]               # trailing weave
        weave = -(np.pi / 3) * np.sin(2 * np.pi * 0.6 * tw + np.pi / 2)
        beta[i(8.0):] += weave - weave[0]
        psi = beta.copy()                               # heading follows course
        psi[i(5.5):i(6.9)] -= np.pi / 2                 # except the offset turn
        ones = np.ones_like(tsim)
        d = beta - psi
        return {'x': 0.0 * ones, 'y': 0.0 * ones, 'z': 2.0 * ones,
                'v_x': g * np.cos(d), 'v_y': g * np.sin(d), 'v_z': 0.0 * ones,
                'phi': 0.0 * ones, 'theta': 0.0 * ones, 'psi': psi,
                'w': w_wind * ones, 'zeta': zeta * ones}

    def build_mpc(sim):
        mx, mtvp = sim.model.x, sim.model.tvp
        cost = ((mx['v_x'] - mtvp['v_x_set']) ** 2 +
                (mx['v_y'] - mtvp['v_y_set']) ** 2 +
                (mx['z'] - mtvp['z_set']) ** 2 +
                (mx['psi'] - mtvp['psi_set']) ** 2 +
                (mx['w'] - mtvp['w_set']) ** 2 +
                (mx['zeta'] - mtvp['zeta_set']) ** 2)
        sim.mpc.set_objective(mterm=cost, lterm=cost)
        sim.mpc.bounds['lower', '_x', 'z'] = 0
        sim.mpc.bounds['lower', '_x', 'phi'] = -np.pi / 4
        sim.mpc.bounds['upper', '_x', 'phi'] = np.pi / 4
        sim.mpc.bounds['lower', '_x', 'theta'] = -np.pi / 4
        sim.mpc.bounds['upper', '_x', 'theta'] = np.pi / 4
        sim.mpc.bounds['lower', '_u', 'u_thrust'] = 0
        sim.mpc.set_rterm(u_thrust=1e-4, u_phi=1e-4, u_theta=1e-4,
                          u_psi=1e-2, u_w=1e-2, u_zeta=1e-2)

    q_real = {'x': 1e-6, 'y': 1e-6, 'z': 1e-4, 'v_x': 5e-2, 'v_y': 5e-2,
              'v_z': 1e-2, 'phi': 1e-4, 'theta': 1e-4, 'psi': 1e-4,
              'w': 1e-2, 'zeta': 1e-2}
    return SystemSpec(
        name='drone', dt=dt, f=drone_f, h=drone_h,
        fim_sensors=tuple(DRONE_MEAS), fim_states=tuple(DRONE_STATES),
        state_names=DRONE_STATES, input_names=DRONE_INPUTS,
        measurement_names=DRONE_MEAS,
        disc_method='vanloan',
        make_setpoint=make_setpoint, default_setpoint=default_setpoint,
        build_mpc=build_mpc,
        wind_default=1.0, zeta_default=-np.pi,   # Fig. 2: w = 1 m/s
        v0_default=1.0, heading0=-np.pi / 4,
        # paper Fig. 2c: 'observability was calculated for 0.5 s time windows,
        # corresponding to 5 discrete measurements (omega = 5)'. A longer
        # window straddles the manoeuvres and makes everything look
        # observable — at w = 20 the cruise/turn contrast collapses from ~1e6
        # to ~12.
        w_default=5, eps_default=1e-5,
        r_default={s: 1e-2 for s in DRONE_MEAS},   # Fig. 2 run: R = 1e-2
        q_tiny={s: 1e-12 for s in DRONE_STATES}, q_realistic=q_real,
        color_state_default='zeta',
        ev_states_default=('w', 'zeta', 'z', 'v_x'),
        angle_measurements=('beta', 'gamma', 'psi'),
        angle_states=('phi', 'theta', 'psi', 'zeta'),
        seg_duration=3.0, quick_turn_dur=2.0, quick_straight_dur=3.0,
        speed_step=1.0, rec_xlim=(-120.0, 120.0), rec_ylim=(-60.0, 60.0),
        default_traj_label='paper Fig. 2: decelerate @1.5 s → accelerate @4 s '
                           '→ heading turn @5 s → offset turn @6.5–7.9 s '
                           '→ turn + 0.6 Hz weave @9 s (z = 2 m, w = 1 m/s)')


SYSTEMS = {'fly': make_fly_spec, 'fly7': make_fly7_spec,
           'drone': make_drone_spec, 'alt2d': make_alt2d_spec}

DRONE_DEFAULT_SEGMENTS = [
    dict(motif='straight', duration=4.0),
    dict(motif='turn', duration=3.0, angle=-np.pi / 2),
    dict(motif='straight', duration=3.0),
    dict(motif='speed', duration=3.0, speed=4.0),
    dict(motif='straight', duration=3.0),
]


# ─────────────────────────── trajectory building ───────────────────────────

def build_setpoints(segments, dt, v0, heading0=0.0):
    """ Compile a list of motif segments into per-step (speed, heading)
    set-point arrays. Each segment dict has keys: motif ('straight' | 'turn' |
    'circle' | 'speed' | 'weave'), duration [s], and per-motif parameters
    angle [deg], speed [m/s], freq [Hz], amp [deg]. Set-points continue from
    where the previous segment ends. """
    v, hd = [v0], [heading0]
    for seg in segments:
        n_k = max(int(round(seg['duration'] / dt)), 1)
        tau = np.arange(1, n_k + 1) * dt
        v_prev, h_prev = v[-1], hd[-1]
        motif = seg['motif']
        if motif == 'straight':
            v += [v_prev] * n_k
            hd += [h_prev] * n_k
        elif motif in ('turn', 'circle'):
            dh = seg['angle']
            v += [v_prev] * n_k
            hd += list(h_prev + dh * tau / tau[-1])
        elif motif == 'speed':
            v += list(v_prev + (seg['speed'] - v_prev) * tau / tau[-1])
            hd += [h_prev] * n_k
        elif motif == 'weave':
            amp = seg['amp']
            v += [v_prev] * n_k
            hd += list(h_prev + amp * np.sin(2 * np.pi * seg['freq'] * tau))
        else:
            raise ValueError(f'unknown motif {motif!r}')
    return np.array(v), np.array(hd)


# ───────────────────────────── compute engine ──────────────────────────────

class ObservabilityEngine:
    """ MPC trajectory simulation + both EV pipelines for one SystemSpec,
    with caching keyed on the trajectory version. """

    def __init__(self, spec):
        self.spec = spec
        self.dt = spec.dt
        self.t = self.X = self.U = None
        self.N = 0
        self._version = 0
        self._lin_cache = {}      # (version, Qc key)             -> (Phis, Cs, Qds)
        self._seom_cache = {}     # (version, w, eps)             -> SEOM
        self._ev_emp_cache = {}   # (version, w, eps, R, lam)     -> EV DataFrame
        self._eom_cache = {}      # (version, k0, W, eps)         -> EOM
        self._emp_vs_w_cache = {} # (version, k0, W, eps, R, lam) -> EV DataFrame
        self._mc_cache = {}       # nonlinear-MC sliding EV DataFrames
        self._mc_vs_w_cache = {}  # nonlinear-MC EV-vs-window DataFrames
        self._lin_ev_cache = {}   # linearized sliding EV DataFrames
        self._filt_cache = {}     # (filter, Q, R, seed, scale) -> (X_hat, P_diag)
        # trained ANNs. NOT cleared by set_trajectory: the net is trained on
        # randomized trajectories of the spec, not on the current one.
        self._ann_cache = {}
        self._mpc_sim = None
        self.lam_noise_floor = 0.0

    def _rk4_sim(self):
        s = self.spec
        return GenericRK4Simulator(s.f, s.h, s.dt, s.state_names,
                                   s.input_names, s.measurement_names)

    # -- trajectory ----------------------------------------------------------
    def set_trajectory(self, t, X, U):
        self.t, self.X, self.U = t, X, U
        self.N = len(t)
        self._version += 1
        for c in (self._lin_cache, self._seom_cache, self._ev_emp_cache,
                  self._eom_cache, self._emp_vs_w_cache, self._mc_cache,
                  self._mc_vs_w_cache, self._lin_ev_cache, self._filt_cache):
            c.clear()

    def _get_mpc_sim(self):
        if self._mpc_sim is None:
            s = self.spec
            sim = Simulator(s.f, s.h, dt=s.dt, state_names=s.state_names,
                            input_names=s.input_names,
                            measurement_names=s.measurement_names,
                            mpc_horizon=10)
            s.build_mpc(sim)
            self._mpc_sim = sim
        return self._mpc_sim

    def simulate_mpc(self, setpoint):
        """ Closed-loop MPC rollout (x0 defaults to the t=0 set-point). """
        sim = self._get_mpc_sim()
        sim.update_dict(setpoint, name='setpoint')
        t, x_sim, u_sim, _ = sim.simulate(x0=None, mpc=True,
                                          return_full_output=True)
        X = np.vstack(list(x_sim.values())).T
        U = np.vstack(list(u_sim.values())).T
        self.set_trajectory(t, X, U)
        return t, X, U

    # -- linearized pipeline (fast) -------------------------------------------
    def _lin(self, q_diag, q_noise='uncorrelated'):
        s = self.spec
        key = (self._version, q_noise,
               tuple(f'{q_diag[n]:.3e}' for n in s.state_names))
        if key not in self._lin_cache:
            # The UI enters Q as the per-step process covariance Q_d, which is
            # exactly the Q of the discrete Eq.30/33 recursion and of the Kalman
            # update — so it is used VERBATIM. No PSD, no dt scaling. (This used to
            # convert to a PSD Q/dt and Van Loan-discretize back; that round trip is
            # gone, which also drops the intra-step cross-state terms it added.)
            Qd = np.diag([q_diag[n] for n in s.state_names])
            self._lin_cache[key] = linearize_along_trajectory(
                s.f, s.h, self.X, self.U, self.dt, Qd,
                method=s.disc_method, q_noise=q_noise)
        return self._lin_cache[key]

    def _fim_subsets(self, sensors, states):
        s = self.spec
        sensors = list(sensors) if sensors else list(s.measurement_names)
        states = list(states) if states else list(s.state_names)
        sidx = [s.measurement_names.index(m) for m in sensors]
        sub = np.array([s.state_names.index(n) for n in states], dtype=int)
        return sensors, states, sidx, sub

    def _ev_diag(self, F, lam):
        """ diag((F + lam I)^-1) via eigendecomposition — never raises on
        singular input. Eigenvalues below the floating-point noise floor of F
        (n * eps * lam_max) are numerically indistinguishable from zero and are
        truncated to 0, so unresolvable directions honestly report 1/lam
        instead of round-off garbage. """
        evals, evecs = np.linalg.eigh(F)
        self._last_fmax = float(max(evals[-1], 0.0))
        tol = F.shape[0] * np.finfo(float).eps * self._last_fmax
        evals = np.where(evals > tol, evals, 0.0)
        return ((evecs ** 2) / (evals + lam)).sum(axis=1)

    def _stoch_gramian(self, basis):
        """ Pick the sliding-window FIM: 'observability' (Eq.33, bounds the
        window's INITIAL state — a smoother) or 'constructability' (Eq.30,
        bounds the FINAL/current state — matches a filter). """
        return (stochastic_constructability_gramian if basis == 'constructability'
                else stochastic_observability_gramian)

    def linearized_ev(self, w, q_diag, r_diag, lam, sensors=None, states=None,
                      basis='observability', q_noise='uncorrelated'):
        """ Sliding min EV via the recursive Gramian, using only the selected
        sensors (rows of C) and reported for the selected states (conditional
        FIM sub-block). `basis` selects the observability (initial-state, Eq.33)
        or constructability (final-state, Eq.30) Gramian; both are reported at
        the window's CENTER time so the two curves share one x-axis. Also records
        `lam_noise_floor` = machine-eps * max eigenvalue of the inverted block. """
        sensors, states, sidx, sub = self._fim_subsets(sensors, states)
        key = (self._version, w, q_noise,
               tuple(f'{q_diag[k]:.3e}' for k in self.spec.state_names),
               tuple(f'{r_diag[m]:.3e}' for m in sensors), f'{lam:.3e}',
               tuple(sensors), tuple(states), basis, 'center')
        if key in self._lin_ev_cache:
            df, self.lam_noise_floor = self._lin_ev_cache[key]
            return df.copy()
        gramian = self._stoch_gramian(basis)
        Phis, Cs, Qds = self._lin(q_diag, q_noise)
        Cs_sel = [C[sidx, :] for C in Cs]
        Rinv = np.diag(1.0 / np.array([r_diag[m] for m in sensors]))
        Rinvs = [Rinv] * w
        starts = np.arange(0, self.N - w + 1)
        EV, fmax = [], 0.0
        for k0 in starts:
            F = gramian(Phis[k0:k0 + w], Cs_sel[k0:k0 + w],
                        Qds[k0:k0 + w], Rinvs)
            # Invert the FULL FIM, then report the selected states'
            # entries. Cropping F to the selection first would instead
            # give the 'all other states known' variance, so unselecting
            # one state would change another's min-EV.
            EV.append(self._ev_diag(F, lam)[sub])
            fmax = max(fmax, self._last_fmax)
        self.lam_noise_floor = np.finfo(float).eps * fmax
        df = pd.DataFrame(np.array(EV), columns=states)
        # Both bases are reported at the window CENTER so the curves are directly
        # comparable on one time axis (render adds +round(w/2)·dt to
        # time_initial). The state each bound actually refers to still differs —
        # observability the window's first state, constructability its last — so
        # a feature in the constructability curve physically occurred up to
        # w·dt earlier than the same feature in the observability curve.
        df.insert(0, 'time_initial', self.t[starts])
        # window CENTRE, matching pybounds SlidingFisherObservability.shift_index
        df.insert(1, 'time', self.t[starts] + int(round(0.5 * w)) * self.dt)
        self._lin_ev_cache[key] = (df, self.lam_noise_floor)
        return df.copy()

    def linearized_ev_vs_window(self, k0, ws, q_diag, r_diag, lam,
                                sensors=None, states=None,
                                q_noise='uncorrelated'):
        """ Min EV for growing window lengths starting at k0 (same sensor /
        state restrictions as linearized_ev). """
        sensors, states, sidx, sub = self._fim_subsets(sensors, states)
        Phis, Cs, Qds = self._lin(q_diag, q_noise)
        Cs_sel = [C[sidx, :] for C in Cs]
        Rinv = np.diag(1.0 / np.array([r_diag[m] for m in sensors]))
        EV = []
        for wv in ws:
            F = stochastic_observability_gramian(
                Phis[k0:k0 + wv], Cs_sel[k0:k0 + wv], Qds[k0:k0 + wv],
                [Rinv] * wv)
            # Invert the FULL FIM, then report the selected states'
            # entries. Cropping F to the selection first would instead
            # give the 'all other states known' variance, so unselecting
            # one state would change another's min-EV.
            EV.append(self._ev_diag(F, lam)[sub])
        return pd.DataFrame(EV, columns=states, index=np.array(ws) * self.dt)

    # -- pybounds empirical pipeline (slow, cached) ----------------------------
    def empirical_ev(self, w, eps, r_diag, lam, sensors=None, states=None):
        s = self.spec
        sensors, states, _, _ = self._fim_subsets(sensors, states)
        r_key = tuple(f'{r_diag[m]:.3e}' for m in s.measurement_names)
        ev_key = (self._version, w, f'{eps:.3e}', r_key, f'{lam:.3e}',
                  tuple(sensors), tuple(states))
        if ev_key in self._ev_emp_cache:
            return self._ev_emp_cache[ev_key], True
        seom_key = (self._version, w, f'{eps:.3e}')
        if seom_key not in self._seom_cache:
            self._seom_cache[seom_key] = SlidingEmpiricalObservabilityMatrix(
                self._rk4_sim(), self.t, self.X, self.U,
                w=w, eps=eps, parallel_sliding=True)
        SEOM = self._seom_cache[seom_key]
        # pybounds' `states` argument crops O's columns, i.e. it inverts the
        # FIM of that sub-block — the "all other states known" variance. Pass ALL
        # states so the inverse is the full one, then keep the selected columns:
        # the state selection then only chooses what is DISPLAYED, and dropping
        # one state cannot change another's min-EV.
        SFO = SlidingFisherObservability(SEOM.O_df_sliding, R=dict(r_diag),
                                         lam=lam, time=SEOM.O_time,
                                         sensors=list(sensors),
                                         states=list(s.state_names),
                                         time_steps=np.arange(0, w), w=None)
        EV = SFO.get_minimum_error_variance()
        keep = [c for c in EV.columns
                if c in ('time', 'time_initial') or c in states]
        EV = EV[keep]
        self._ev_emp_cache[ev_key] = EV
        return EV, False

    def empirical_window_O(self, w, eps, k=0, sensors=None, states=None):
        """ Empirical observability matrix 𝒪 for a single sliding window k
        (paper Fig 1e), restricted to the selected sensors (rows) and states
        (columns). Reuses the cached SEOM from ``empirical_ev``. Returns
        (O_df, k_clamped); O_df rows are a (sensor, time_step) MultiIndex. """
        sensors, states, _, _ = self._fim_subsets(sensors, states)
        seom_key = (self._version, w, f'{eps:.3e}')
        if seom_key not in self._seom_cache:
            self._seom_cache[seom_key] = SlidingEmpiricalObservabilityMatrix(
                self._rk4_sim(), self.t, self.X, self.U,
                w=w, eps=eps, parallel_sliding=True)
        O_list = self._seom_cache[seom_key].O_df_sliding
        k = int(np.clip(k, 0, len(O_list) - 1))
        O_df = O_list[k][list(states)]                 # selected state columns
        keep = O_df.index.get_level_values('sensor').isin(list(sensors))
        return O_df[keep], k                           # selected sensor rows

    def _span(self, a, b):
        """ Clamp a trajectory step span [a, b) to valid bounds. """
        a = int(np.clip(a, 0, max(self.N - 1, 0)))
        b = self.N if b is None else int(np.clip(b, a + 1, self.N))
        return a, b

    def empirical_window_fisher_inv(self, w, eps, r_diag, lam, k=0,
                                    sensors=None, states=None):
        """ Inverse Fisher information F⁻¹ for a single sliding window k, from
        the empirical observability matrix (reuses the cached SEOM). Returns an
        (n_states × n_states) DataFrame plus the clamped k (paper Fig 1g). """
        sensors, states, _, _ = self._fim_subsets(sensors, states)
        seom_key = (self._version, w, f'{eps:.3e}')
        if seom_key not in self._seom_cache:
            self._seom_cache[seom_key] = SlidingEmpiricalObservabilityMatrix(
                self._rk4_sim(), self.t, self.X, self.U,
                w=w, eps=eps, parallel_sliding=True)
        O_list = self._seom_cache[seom_key].O_df_sliding
        k = int(np.clip(k, 0, len(O_list) - 1))
        # full inverse, then crop to the selection (see empirical_ev)
        FO = FisherObservability(O_list[k], R=dict(r_diag), lam=lam,
                                 sensors=list(sensors),
                                 states=list(self.spec.state_names),
                                 time_steps=np.arange(0, w))
        return FO.F_inv.loc[list(states), list(states)], k

    def linearized_window_O(self, w, k, q_diag, sensors=None, states=None,
                            q_noise='uncorrelated',
                            basis='observability'):
        """ Analytic observability/constructability matrix for the sliding
        window at k, from the trajectory linearization (Q-independent). Rows are
        a (sensor, time_step) MultiIndex, mirroring the empirical 𝒪.
        observability: 𝒪 = [C_j · Φ_{j←k}] (relates measurements to the window's
        INITIAL state). constructability: [C_{e-j} · Φ⁻¹_{e-j←e}] in reverse-time
        order (relates measurements to the FINAL state e = k+w-1). Returns
        (O_df, k_clamped). """
        sensors, states, sidx, _ = self._fim_subsets(sensors, states)
        Phis, Cs, _ = self._lin(q_diag, q_noise)
        n = len(self.spec.state_names)
        k = int(np.clip(k, 0, max(self.N - w, 0)))
        rows, index, Psi = [], [], np.eye(n)
        if basis == 'constructability':
            e = k + w - 1                            # final state index
            for j in range(w):                       # reverse-time from the end
                CP = Cs[e - j] @ Psi                 # C_{e-j} · Φ⁻¹_{e-j←e}
                for si, sname in zip(sidx, sensors):
                    rows.append(CP[si, :]); index.append((sname, j))
                if j < w - 1:
                    Psi = np.linalg.inv(Phis[e - j - 1]) @ Psi
        else:
            for j in range(w):
                CP = Cs[k + j] @ Psi                 # C_j · Φ_{j←k}
                for si, sname in zip(sidx, sensors):
                    rows.append(CP[si, :]); index.append((sname, j))
                if j < w - 1:
                    Psi = Phis[k + j] @ Psi
        O_df = pd.DataFrame(
            np.array(rows), columns=list(self.spec.state_names),
            index=pd.MultiIndex.from_tuples(index, names=['sensor', 'time_step']))
        return O_df[list(states)], k

    def linearized_fisher_inv(self, q_diag, r_diag, lam, sensors=None,
                              states=None, a=0, b=None, basis='observability',
                              q_noise='uncorrelated'):
        """ Inverse stochastic Gramian over trajectory steps [a, b) — the
        stochastic analog of the empirical F⁻¹. `basis` = 'observability'
        (initial-state, Eq.33) or 'constructability' (final-state, Eq.30).
        b=None → the whole trajectory. Returns an (n_states × n_states) frame. """
        sensors, states, sidx, sub = self._fim_subsets(sensors, states)
        a, b = self._span(a, b)
        Phis, Cs, Qds = self._lin(q_diag, q_noise)
        Cs_sel = [C[sidx, :] for C in Cs]
        Rinv = np.diag(1.0 / np.array([r_diag[m] for m in sensors]))
        L = b - a
        F = self._stoch_gramian(basis)(
            Phis[a:b], Cs_sel[a:b], Qds[a:b], [Rinv] * L)
        # invert the full FIM, then crop the INVERSE to the selection —
        # cropping first would condition on the unselected states
        n_all = len(self.spec.state_names)
        F_inv = np.linalg.inv(F + lam * np.eye(n_all))[np.ix_(sub, sub)]
        return pd.DataFrame(F_inv, index=list(states), columns=list(states))

    def empirical_ev_vs_window(self, k0, ws, eps, r_diag, lam,
                               sensors=None, states=None):
        """ Min EV vs window length at fixed start k0, from the pybounds
        empirical O: one EmpiricalObservabilityMatrix over the longest window,
        then F = O^T R^{-1} O restricted to the first wv time steps and the
        selected sensors / states (pybounds FisherObservability semantics). """
        s = self.spec
        sensors, states, _, _ = self._fim_subsets(sensors, states)
        W_max = int(max(ws))
        r_key = tuple(f'{r_diag[m]:.3e}' for m in s.measurement_names)
        key = (self._version, k0, W_max, f'{eps:.3e}', r_key, f'{lam:.3e}',
               tuple(sensors), tuple(states))
        if key in self._emp_vs_w_cache:
            return self._emp_vs_w_cache[key]
        eom_key = (self._version, k0, W_max, f'{eps:.3e}')
        if eom_key not in self._eom_cache:
            self._eom_cache[eom_key] = EmpiricalObservabilityMatrix(
                self._rk4_sim(), self.X[k0], self.U[k0:k0 + W_max], eps=eps)
        O_df = self._eom_cache[eom_key].O_df
        O = O_df[list(states)].values
        tsteps = O_df.index.get_level_values('time_step').values
        snames = O_df.index.get_level_values('sensor')
        smask = np.array([m in set(sensors) for m in snames])
        rinv = np.array([1.0 / r_diag[m] for m in snames])
        EV = []
        for wv in ws:
            rows = smask & (tsteps < wv)
            Om = O[rows]
            F = (Om * rinv[rows, None]).T @ Om
            evals, evecs = np.linalg.eigh(F)
            EV.append(((evecs ** 2) / (np.clip(evals, 0.0, None) + lam))
                      .sum(axis=1))
        df = pd.DataFrame(EV, columns=list(states),
                          index=np.array(ws) * self.dt)
        self._emp_vs_w_cache[key] = df
        return df

    # -- nonlinear Monte-Carlo stochastic Gramian (repo-root implementation) ------
    def _mc_callables(self, sidx):
        """ Wrap the system as the discrete-time x_{k+1} = f_k(x_k) + w_k,
        y_k = h_k(x_k) + v_k expected by gramian_recursive_nonlinear:
        f_k = RK4 step with the recorded input u_{k0+k} baked in; Jacobians by
        central finite differences (same stencil as the EKF). """
        s = self.spec
        sim = self._rk4_sim()
        n = len(s.state_names)

        def jac(func, x, eps=1e-6):
            cols = []
            for i in range(n):
                dx = np.zeros(n); dx[i] = eps * max(abs(x[i]), 1.0)
                cols.append((np.asarray(func(x + dx), dtype=float)
                             - np.asarray(func(x - dx), dtype=float))
                            / (2 * dx[i]))
            return np.array(cols).T

        def make(k0):
            f_k = lambda x, k: sim._rk4_step(np.asarray(x, float),
                                             self.U[k0 + k])
            Fj = lambda x, k: jac(lambda z: sim._rk4_step(z, self.U[k0 + k]),
                                  np.asarray(x, float))
            h_k = lambda x, k: np.asarray(s.h(np.asarray(x, float),
                                              self.U[k0 + k]),
                                          dtype=float)[sidx]
            Hj = lambda x, k: jac(lambda z: np.asarray(
                s.h(z, self.U[k0 + k]), dtype=float)[sidx],
                np.asarray(x, float))
            return f_k, Fj, h_k, Hj
        return make

    def _mc_blocks(self, k0, W, Qds, R, sidx, n_mc, rng):
        """ Monte-Carlo expected blocks over [k0, k0+W), conditioned on the
        nominal x0 = X[k0]: M_k = E[HᵀR⁻¹H] (W entries), G_k = E[F] and
        Dev_k = E[(F−G)ᵀQ⁻¹(F−G)] (W−1 entries each). Same expectations as
        gramian_recursive_nonlinear(mode='mc'), accumulated in a form that
        avoids the huge-minus-huge cancellation of M + S − GᵀQ⁻¹(·)Q⁻¹G when
        any Qd entry is tiny (jittered constant states): algebraically
        S − GᵀQ⁻¹(I+Q⁻¹)⁻¹Q⁻¹G = Dev + Gᵀ[Q⁻¹ − Q⁻¹(I+Q⁻¹)⁻¹Q⁻¹]G, and Dev
        is small×Q⁻¹×small (exactly zero in constant-state directions). """
        n = len(self.spec.state_names)
        f_k, Fj, h_k, Hj = self._mc_callables(sidx)(k0)
        Qsl = Qds[k0:k0 + W - 1]
        Qinvs = [np.linalg.inv(Q) for Q in Qsl]
        chols = [np.linalg.cholesky(Q) for Q in Qsl]
        Rinv = np.linalg.inv(R)
        Ms = [np.zeros((n, n)) for _ in range(W)]
        F_samp = [[] for _ in range(W - 1)]
        for _ in range(n_mc):
            x = self.X[k0].copy()
            for k in range(W):
                H = Hj(x, k)
                Ms[k] += H.T @ Rinv @ H
                if k < W - 1:
                    F_samp[k].append(Fj(x, k))
                    x = f_k(x, k) + chols[k] @ rng.standard_normal(n)
        Ms = [0.5 * (M + M.T) / n_mc for M in Ms]
        Gs, Devs = [], []
        for k in range(W - 1):
            Fs = np.array(F_samp[k])
            G = Fs.mean(axis=0)
            D = Fs - G
            Dev = sum(d.T @ Qinvs[k] @ d for d in D) / n_mc
            Gs.append(G)
            Devs.append(0.5 * (Dev + Dev.T))
        return Ms, Gs, Devs, Qinvs

    @staticmethod
    def _mc_recursion(Ms, Gs, Devs, Qinvs, w):
        """ Backward recursion of the nonlinear stochastic Gramian on the
        first w measurement blocks (stable bracket form). """
        I = Ms[w - 1].copy()
        for k in range(w - 2, -1, -1):
            Qi = Qinvs[k]
            B = Qi - Qi @ np.linalg.solve(I + Qi, Qi)
            I = Ms[k] + Devs[k] + Gs[k].T @ B @ Gs[k]
            I = 0.5 * (I + I.T)
        return I

    def mc_ev(self, w, q_diag, r_diag, lam, sensors=None, states=None,
              q_noise='uncorrelated',
              n_mc=100, seed=0):
        """ Sliding min EV via the nonlinear Monte-Carlo stochastic Gramian
        (the repo-root gramian_recursive_nonlinear math, mode='mc', in the
        stiff-Q-stable form — see _mc_blocks). Conditioned on the nominal
        window-initial state; discrete Q_k is the same Van-Loan Qd the
        linearized pipeline uses, so the methods share the noise model. """
        sensors, states, sidx, sub = self._fim_subsets(sensors, states)
        r_key = tuple(f'{r_diag[m]:.3e}' for m in sensors)
        q_key = (q_noise,) + tuple(f'{q_diag[k]:.3e}'
                                  for k in self.spec.state_names)
        key = (self._version, w, q_key, r_key, f'{lam:.3e}', tuple(sensors),
               tuple(states), int(n_mc), int(seed))
        if key in self._mc_cache:
            return self._mc_cache[key], True
        _, _, Qds = self._lin(q_diag, q_noise)
        R = np.diag([r_diag[m] for m in sensors])
        starts = np.arange(0, self.N - w + 1)
        EV = []
        for k0 in starts:
            Ms, Gs, Devs, Qinvs = self._mc_blocks(
                k0, w, Qds, R, sidx, n_mc,
                np.random.default_rng(100003 * seed + k0))
            F = self._mc_recursion(Ms, Gs, Devs, Qinvs, w)
            # Invert the FULL FIM, then report the selected states'
            # entries. Cropping F to the selection first would instead
            # give the 'all other states known' variance, so unselecting
            # one state would change another's min-EV.
            EV.append(self._ev_diag(F, lam)[sub])
        df = pd.DataFrame(np.array(EV), columns=states)
        df.insert(0, 'time_initial', self.t[starts])
        # window CENTRE, matching pybounds SlidingFisherObservability.shift_index
        df.insert(1, 'time', self.t[starts] + int(round(0.5 * w)) * self.dt)
        self._mc_cache[key] = df
        return df, False

    def mc_ev_vs_window(self, k0, ws, q_diag, r_diag, lam, sensors=None,
                        q_noise='uncorrelated',
                        states=None, n_mc=100, seed=0):
        """ Min EV vs window length via the nonlinear MC Gramian. The MC
        M/S/G blocks along [k0, k0+max(ws)) are estimated once (block k's
        distribution does not depend on the window horizon), then the
        backward recursion runs on each prefix — same math as
        gramian_recursive_nonlinear, factored for shared sampling. """
        sensors, states, sidx, sub = self._fim_subsets(sensors, states)
        W_max = int(max(ws))
        r_key = tuple(f'{r_diag[m]:.3e}' for m in sensors)
        q_key = (q_noise,) + tuple(f'{q_diag[k]:.3e}'
                                  for k in self.spec.state_names)
        key = (self._version, k0, W_max, q_key, r_key, f'{lam:.3e}',
               tuple(sensors), tuple(states), int(n_mc), int(seed))
        if key in self._mc_vs_w_cache:
            return self._mc_vs_w_cache[key]
        _, _, Qds = self._lin(q_diag, q_noise)
        R = np.diag([r_diag[m] for m in sensors])
        Ms, Gs, Devs, Qinvs = self._mc_blocks(
            k0, W_max, Qds, R, sidx, n_mc,
            np.random.default_rng(200003 * seed + k0))
        EV = []
        for wv in ws:   # backward recursion on each prefix of the blocks
            F = self._mc_recursion(Ms, Gs, Devs, Qinvs, wv)
            # Invert the FULL FIM, then report the selected states'
            # entries. Cropping F to the selection first would instead
            # give the 'all other states known' variance, so unselecting
            # one state would change another's min-EV.
            EV.append(self._ev_diag(F, lam)[sub])
        df = pd.DataFrame(EV, columns=states, index=np.array(ws) * self.dt)
        self._mc_vs_w_cache[key] = df
        return df

    # -- estimators: true vs estimated states ------------------------------------
    def _estimation_problem(self, r_diag, seed, x0_scale=1.0, x0_offset=None,
                            u_noise_var=0.0, u_bias=None, p0_diag=None,
                            x0_guess=None):
        """ Sample one estimation-problem realization: noisy measurements
        y_k = h(x_k, u_k) + v_k with v ~ N(0, R) from the true trajectory, plus
        an initial estimate with ~10%% error scaled by x0_scale, and optional
        per-state offsets (e.g. {'z': 7.5} for the paper's z0 = 9 experiment).
        P0 stays consistent with the total initial spread.

        x0_guess pins the initial estimate of individual states outright
        (length-n, NaN where a state should keep its sampled value) — e.g. the
        paper's z0 = 9 stated as a value rather than as an offset. Values are
        used verbatim, sign included; a state whose guess is -2.5 starts at
        -2.5.

        u_noise_var / u_bias model the *measured accelerations* (paper Fig. 4b).
        These are IMU readings, but a filter consumes them in its prediction
        step as process-model inputs, so they are perturbed here rather than in
        the measurement vector: the truth flies the clean U, while the filters
        receive U + N(0, u_noise_var)
        (+ a constant bias on one channel inside a time window,
        u_bias = (channel_index, magnitude, t0, t1)). Same seed -> same
        realization, so the EKF and UKF face identical data. """
        s = self.spec
        rng = np.random.default_rng(seed + 1000 * self._version)
        R = np.diag([r_diag[m] for m in s.measurement_names])
        Y = np.array([np.asarray(s.h(self.X[k], self.U[k]), dtype=float)
                      for k in range(self.N)])
        Y += rng.normal(0.0, np.sqrt(np.diag(R)), size=Y.shape)
        sig0 = x0_scale * 0.1 * np.maximum(np.abs(self.X[0]), 0.1)
        xh0 = self.X[0] + rng.normal(0.0, sig0)
        if x0_offset:
            for name, off in x0_offset.items():
                i = s.state_names.index(name)
                xh0[i] += off
                sig0[i] = np.hypot(sig0[i], off)   # keep P0 honest
        if x0_guess is not None:
            v = np.asarray(x0_guess, dtype=float).ravel()
            if v.size != len(sig0):
                raise ValueError(f'x0_guess has {v.size} entries, '
                                 f'expected {len(sig0)}')
            use = np.isfinite(v)
            # a pinned guess replaces the sampled one verbatim (signs included);
            # its DISTANCE from the truth widens that state's P0 exactly as an
            # offset does, so the filter starts out admitting an error as large
            # as the one you handed it
            sig0 = np.where(use, np.hypot(sig0, v - self.X[0]), sig0)
            xh0 = np.where(use, v, xh0)
        U_used = self.U.copy()
        if u_noise_var > 0:
            U_used = U_used + rng.normal(0.0, np.sqrt(u_noise_var),
                                         size=U_used.shape)
        if u_bias is not None:
            ch, mag, t0, t1 = u_bias
            U_used[(self.t >= t0) & (self.t < t1), ch] += mag
        # P0 is an ABSOLUTE per-state diagonal: p0_diag[i] is the initial
        # variance of state i. A missing/non-positive/non-finite entry falls back
        # to that state's sampled initial spread, so states can be mixed freely.
        # A scalar broadcasts over all states (the repo's P = diag([p, p, p])).
        var0 = sig0 ** 2
        if p0_diag is not None:
            v = np.asarray(p0_diag, dtype=float).ravel()
            if v.size == 1:
                v = np.full(len(sig0), v[0])
            if v.size != len(sig0):
                raise ValueError(f'p0_diag has {v.size} entries, '
                                 f'expected {len(sig0)}')
            use = np.isfinite(v) & (v > 0)
            var0 = np.where(use, v, var0)
        return Y, xh0, np.diag(var0), R, U_used

    def _ann_training_data(self, ann, n_traj, seed):
        """ Training trajectories for the ANN, the way the authors' repo makes
        them: run the system's OWN MPC over perturbed set-points and keep every
        window. Their altitude notebook sweeps the speed change g_delta and lets
        the altitude set-point vary so the net sees a range of altitudes; the
        same two knobs are applied generically here.

        Runs on a throwaway engine so the user's current trajectory is untouched.
        Returns [(YU, target_series)] in the ANN's own column layout. """
        s = self.spec
        rng = np.random.default_rng(seed)
        sp0 = s.default_setpoint(getattr(s, 'wind_default', 0.0),
                                 getattr(s, 'zeta_default', 0.0))
        sub = ObservabilityEngine(s)
        speeds = [k for k in ('v_para', 'v_x', 'g') if k in sp0]
        out = []
        for i in range(max(2, int(n_traj))):
            sp = {k: np.array(v, dtype=float).copy() for k, v in sp0.items()}
            # (1) scale the manoeuvre amplitude about its own baseline — the
            #     repo's g_delta sweep
            amp = rng.uniform(-1.5, 2.0)
            for k in speeds:
                v = sp[k]
                sp[k] = v[0] + (v - v[0]) * amp
            # (2) offset the estimated state's set-point so the net sees a range
            #     of its values rather than the single default level
            if ann.target in sp:
                v = sp[ann.target]
                lo = float(np.min(v))
                sp[ann.target] = v * rng.uniform(0.4, 2.5) if lo > 0 else \
                    v + rng.uniform(-1.0, 1.0) * max(abs(lo), 1.0)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    _, X, U = sub.simulate_mpc(sp)
            except Exception:
                continue                       # a set-point the MPC can't fly
            Y = np.array([np.asarray(s.h(X[k], U[k]), dtype=float)
                          for k in range(len(X))])
            YU = ann.columns(Y, U)
            tv = X[:, s.state_names.index(ann.target)]
            if np.all(np.isfinite(YU)) and np.all(np.isfinite(tv)):
                out.append((YU, tv))
        return out

    def _ann(self, target, time_steps, layers, n_traj, epochs, seed,
             batch=256, noise_std=0.01):
        """ Train-once-and-cache the ANN of Eq. 6 for one (target, shape) combo.
        The net is trained on MPC rollouts of this spec's own default set-point
        family, never on the trajectory being estimated, so it survives
        trajectory edits — hence the cache key deliberately excludes
        self._version. """
        key = (self.spec.name, target, time_steps, tuple(layers), n_traj,
               epochs, batch, round(float(noise_std), 6), seed)
        if key not in self._ann_cache:
            ann = ANNEstimator(self.spec, target, time_steps=time_steps,
                               layers=layers, seed=seed)
            ann.train(self._ann_training_data(ann, n_traj, seed),
                      epochs=epochs, batch=batch, noise_std=noise_std)
            self._ann_cache[key] = ann
        return self._ann_cache[key]

    def run_aikf(self, target, q_diag, r_diag, seed=0, time_steps=4,
                 layers=(64, 64, 64), n_traj=16, epochs=100, batch=256,
                 noise_std=0.01, motif_input=None, motif_window=20,
                 motif_upper=0.5, R_ann_lo=1e-3, R_ann_hi=1e12,
                 alpha=1e-3, beta=1.0, kappa=0.0,
                 x0_scale=1.0, x0_offset=None, u_noise_var=0.0, u_bias=None,
                 p0_diag=None, x0_guess=None):
        """ The repo's AI-KF (util/StateEstimator.m): a UKF on the real dynamics
        whose measurement vector is AUGMENTED with the ANN's estimate of `target`
        as a pseudo-measurement, and whose noise entry for that channel is
        modulated by an active-sensing motif.

            R[ann,ann]_k = clip(logarithmic_map(
                backward_window_mean(motif, motif_window),
                [0, motif_upper], (R_ann_hi, R_ann_lo)))

        No motif -> R = R_ann_hi (1e12: the channel is ignored and the filter
        coasts on its dynamics); motif at/above `motif_upper` -> R = R_ann_lo
        (1e-3: the ANN is trusted). Set R_ann_lo = R_ann_hi to disable the channel
        and recover the plain EKF — the repo's own ablation (r_sweep = [12, -3]).

        `motif_input` names the input channel used as the motif (repo:
        motif = |u_x|); it defaults to the last input, which is the horizontal
        acceleration for the altitude/monocular models.

        Returns (X_hat, P_diag, raw, R_ann): filter output over all states, the
        raw ANN estimate, and the per-step noise assigned to the ANN channel. """
        s = self.spec
        n = len(s.state_names)
        j_t = s.state_names.index(target)
        sim = self._rk4_sim()
        Y, xh, P, R, U = self._estimation_problem(r_diag, seed, x0_scale,
                                                  x0_offset, u_noise_var, u_bias,
                                                  p0_diag, x0_guess)
        ann = self._ann(target, time_steps, layers, n_traj, epochs, seed,
                        batch, noise_std)
        raw = ann.predict_series(Y, U)
        # the motif: mean |input| over a backward window (repo: |u_x|)
        mi = (s.input_names.index(motif_input)
              if motif_input in s.input_names else len(s.input_names) - 1)
        R_ann = motif_R(np.abs(U[:, mi]), motif_window, motif_upper,
                        (float(R_ann_hi), float(R_ann_lo)))
        # augmented measurement: [real sensors..., ANN estimate of `target`]
        p_real = len(s.measurement_names)
        ang_m = [i for i, m in enumerate(s.measurement_names)
                 if m in s.angle_measurements]
        circ = target in s.angle_states
        ang_aug = ang_m + ([p_real] if circ else [])   # angular cols of the
        wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi   # augmented vector
        Qd = np.diag([q_diag[k] for k in s.state_names])

        # ── sigma-point machinery, identical to run_ukf ──────────────────────
        lam_s = alpha ** 2 * (n + kappa) - n     # Merwe scaling parameter
        scale = max(n + lam_s, 1e-6)             # = α²(n+κ); sigma spread √scale
        Wm = np.full(2 * n + 1, 1.0 / (2 * scale)); Wm[0] = lam_s / scale
        Wc = Wm.copy(); Wc[0] = lam_s / scale + (1.0 - alpha ** 2 + beta)

        def sqrt_psd(M):
            try:
                return np.linalg.cholesky(M)
            except np.linalg.LinAlgError:
                e, V = np.linalg.eigh((M + M.T) / 2)
                return V @ np.diag(np.sqrt(np.clip(e, 1e-15, None)))

        def sigma_points(x, P):
            L = sqrt_psd(scale * P)
            pts = np.empty((2 * n + 1, n))
            pts[0] = x
            pts[1:n + 1] = x + L.T
            pts[n + 1:] = x - L.T
            return pts

        def y_stats(Ys, cols):
            ym = Ys.T @ Wm
            for i in cols:               # circular mean for angular channels
                ym[i] = np.arctan2(np.sum(Wm * np.sin(Ys[:, i])),
                                   np.sum(Wm * np.cos(Ys[:, i])))
            dY = Ys - ym
            if cols:
                dY[:, cols] = wrap(dY[:, cols])
            return ym, dY

        X_hat = np.zeros_like(self.X)
        P_diag = np.zeros((self.N, n))
        for k in range(self.N):
            # ── measurement update over the augmented vector ────────────────
            use_ann = bool(np.isfinite(raw[k]))
            pts = sigma_points(xh, P)
            Ys = np.array([np.asarray(s.h(pt, U[k]), dtype=float) for pt in pts])
            if use_ann:
                # the ANN channel observes the target state directly, so its
                # sigma-point image is just that component
                Ys = np.hstack([Ys, pts[:, [j_t]]])
                Rk = np.zeros((p_real + 1, p_real + 1))
                Rk[:p_real, :p_real] = R
                Rk[-1, -1] = R_ann[k]
                zk = np.append(Y[k], raw[k])
                cols = ang_aug
            else:
                Rk, zk, cols = R, Y[k], ang_m
            ym, dY = y_stats(Ys, cols)
            dX = pts - xh
            S_cov = dY.T @ (Wc[:, None] * dY) + Rk
            Cxy = dX.T @ (Wc[:, None] * dY)
            K = np.linalg.solve(S_cov.T, Cxy.T).T
            innov = zk - ym
            if cols:
                innov[cols] = wrap(innov[cols])
            xh = xh + K @ innov
            P = P - K @ S_cov @ K.T
            P = (P + P.T) / 2
            X_hat[k] = xh
            P_diag[k] = np.diag(P)
            # ── time update ─────────────────────────────────────────────────
            if k < self.N - 1:
                pts = sigma_points(xh, P)
                Xs = np.array([sim._rk4_step(pt, U[k]) for pt in pts])
                xh = Xs.T @ Wm
                dXp = Xs - xh
                P = dXp.T @ (Wc[:, None] * dXp) + Qd
                P = (P + P.T) / 2
        return X_hat, P_diag, raw, R_ann

    def run_ann_mif(self, target, q_diag, r_diag, lam, seed=0, time_steps=4,
                    layers=(64, 64, 64), n_traj=16, epochs=100, batch=256,
                    noise_std=0.01,
                    basis='constructability', x0_scale=1.0, x0_offset=None,
                    u_noise_var=0.0, u_bias=None, sensors=None):
        """ ANN raw estimate (Eq. 6) and the motif-informed filter of it
        (Eq. 5 with beta from Eq. 7), for a single state `target`.

        The ANN sees the SAME noisy measurement realization as the EKF/UKF (same
        seed), so the three are directly comparable. beta comes from the sliding
        min error variance of `target` in the requested basis — constructability
        by default, which is what the paper says statistical consistency wants.

        Returns (raw, filtered, beta): three length-N arrays; the first `omega`
        steps of `raw` are NaN (no measurement history yet). """
        s = self.spec
        Y, _, _, _, _ = self._estimation_problem(r_diag, seed, x0_scale,
                                                 x0_offset, u_noise_var, u_bias)
        ann = self._ann(target, time_steps, layers, n_traj, epochs, seed,
                        batch, noise_std)
        raw = ann.predict_series(Y, U)
        # beta must describe the observability available to THIS estimator, so the
        # FIM uses the ANN's own sensor set unless told otherwise. Feeding it every
        # measurement instead makes zeta look permanently well observed (min-EV
        # ~0.1 flat, beta ~ 0.99) because airspeed + groundspeed nearly hand you
        # the wind vector — and the gate stops doing anything.
        if sensors is None:
            sensors = tuple(ann.sensors)
        # Eq. 7 needs the min-EV of the target on the same time grid as the
        # trajectory; linearized_ev reports one row per window, centered.
        w = int(min(max(int(time_steps), 2), max(self.N - 1, 2)))
        # Eq. 7 wants the i-th DIAGONAL of the full window inverse, so the FIM is
        # inverted over all states and the target's element read off. Restricting
        # the FIM to the target alone instead would give the "all other states
        # known" variance — far smaller, and nearly constant, which pins beta ~ 1.
        ev = self.linearized_ev(w, q_diag, r_diag, lam, sensors=sensors,
                                states=None, basis=basis)
        ev_path = np.full(self.N, np.nan)
        idx = np.rint((ev['time_initial'].values
                       + int(np.round(w / 2.0)) * self.dt) / self.dt).astype(int)
        keep = (idx >= 0) & (idx < self.N)
        ev_path[idx[keep]] = ev[target].values[keep]
        fin = np.flatnonzero(np.isfinite(ev_path))
        if len(fin):                          # hold the ends, as pybounds does
            ev_path[:fin[0]] = ev_path[fin[0]]
            ev_path[fin[-1]:] = ev_path[fin[-1]]
        beta = mif_beta(ev_path)
        filt = motif_informed_filter(raw, beta,
                                     circular=target in s.angle_states)
        return raw, filt, beta

    def run_ekf(self, q_diag, r_diag, seed=0, full_cov=False,
                x0_scale=1.0, x0_offset=None, u_noise_var=0.0, u_bias=None, p0_diag=None,
                x0_guess=None, f_jac_at='post', fd_eps_scaled=False,
                q_noise='uncorrelated'):
        """ EKF along the true trajectory (finite-difference Jacobians of the
        RK4 step and of h). Returns (X_hat, P_diag), post-update values;

        Aligned with the BOUNDS reference EKF (util/extended_kalman_filter.py::EKF):
        `f_jac_at='post'` linearizes F about the PROPAGATED state as the reference
        does (textbook EKF uses 'pre'); `fd_eps_scaled=False` uses the reference's
        ABSOLUTE 1e-6 perturbation (True scales it by max(|xᵢ|,1), which is more
        accurate for models with large-magnitude states). Q here is already the
        per-step covariance, which matches the reference's convention.
        with full_cov=True returns (X_hat, P_diag, [P_k]) for consistency
        tests (NEES). """
        s = self.spec
        n = len(s.state_names)
        sim = self._rk4_sim()
        Y, xh, P, R, U = self._estimation_problem(r_diag, seed, x0_scale,
                                                  x0_offset, u_noise_var,
                                                  u_bias, p0_diag, x0_guess)
        # per-step process covariance. 'uncorrelated' -> the entered Q as a
        # diagonal, identical at every step; 'vanloan' -> rebuilt from the LOCAL
        # dynamics, so it is correlated and varies along the trajectory. Same
        # stack the stochastic gramian uses, so the two stay consistent.
        # 'uncorrelated' keeps the filters self-contained (no trajectory
        # linearization, as before); 'vanloan' needs A_k, so it reuses the same
        # cached stack the stochastic gramian sees.
        Qds = (self._lin(q_diag, q_noise)[2] if q_noise == 'vanloan' else
               [np.diag([q_diag[nm] for nm in s.state_names])] * self.N)
        ang_m = [i for i, m in enumerate(s.measurement_names)
                 if m in s.angle_measurements]
        wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi

        def jac(func, x, u, eps=1e-6):
            cols = []
            for i in range(n):
                step = eps * max(abs(x[i]), 1.0) if fd_eps_scaled else eps
                dx = np.zeros(n); dx[i] = step
                fp = np.asarray(func(x + dx, u), dtype=float)
                fm = np.asarray(func(x - dx, u), dtype=float)
                cols.append((fp - fm) / (2 * dx[i]))
            return np.array(cols).T

        X_hat = np.zeros_like(self.X)
        P_diag = np.zeros((self.N, n))
        P_hist = []
        I_n = np.eye(n)
        for k in range(self.N):
            # measurement update
            C = jac(s.h, xh, U[k])
            innov = Y[k] - np.asarray(s.h(xh, U[k]), dtype=float)
            if ang_m:
                innov[ang_m] = wrap(innov[ang_m])
            S = C @ P @ C.T + R
            K = np.linalg.solve(S.T, (P @ C.T).T).T
            xh = xh + K @ innov
            IKC = I_n - K @ C
            P = IKC @ P @ IKC.T + K @ R @ K.T   # Joseph form
            X_hat[k] = xh
            P_diag[k] = np.diag(P)
            if full_cov:
                P_hist.append(P.copy())
            # time update
            if k < self.N - 1:
                xnew = sim._rk4_step(xh, U[k])
                # reference linearizes about the PROPAGATED state
                Phi = jac(sim._rk4_step, xnew if f_jac_at == 'post' else xh, U[k])
                xh = xnew
                P = Phi @ P @ Phi.T + Qds[k]
        if full_cov:
            return X_hat, P_diag, P_hist
        return X_hat, P_diag

    def run_ukf(self, q_diag, r_diag, seed=0, alpha=1.0, beta=2.0, kappa=0.0,
                full_cov=False, x0_scale=1.0, x0_offset=None, u_noise_var=0.0,
                u_bias=None, p0_diag=None, x0_guess=None,
                q_noise='uncorrelated'):
        """ UKF on the identical problem realization as run_ekf (same seed →
        same measurements and initial estimate). Merwe scaled unscented
        transform with tunable spread α, prior β, secondary scaling κ (defaults
        α = 1, κ = 0 → sigma points at sqrt(n)·σ), with circular means / wrapped
        residuals for the arctan2 measurements. """
        s = self.spec
        n = len(s.state_names)
        sim = self._rk4_sim()
        Y, xh, P, R, U = self._estimation_problem(r_diag, seed, x0_scale,
                                                  x0_offset, u_noise_var,
                                                  u_bias, p0_diag, x0_guess)
        # per-step process covariance. 'uncorrelated' -> the entered Q as a
        # diagonal, identical at every step; 'vanloan' -> rebuilt from the LOCAL
        # dynamics, so it is correlated and varies along the trajectory. Same
        # stack the stochastic gramian uses, so the two stay consistent.
        # 'uncorrelated' keeps the filters self-contained (no trajectory
        # linearization, as before); 'vanloan' needs A_k, so it reuses the same
        # cached stack the stochastic gramian sees.
        Qds = (self._lin(q_diag, q_noise)[2] if q_noise == 'vanloan' else
               [np.diag([q_diag[nm] for nm in s.state_names])] * self.N)
        ang_m = [i for i, m in enumerate(s.measurement_names)
                 if m in s.angle_measurements]
        wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi

        lam_s = alpha ** 2 * (n + kappa) - n     # Merwe scaling parameter
        scale = max(n + lam_s, 1e-6)             # = α²(n+κ); sigma spread √scale
        Wm = np.full(2 * n + 1, 1.0 / (2 * scale)); Wm[0] = lam_s / scale
        Wc = Wm.copy(); Wc[0] = lam_s / scale + (1.0 - alpha ** 2 + beta)

        def sqrt_psd(M):
            try:
                return np.linalg.cholesky(M)
            except np.linalg.LinAlgError:
                e, V = np.linalg.eigh((M + M.T) / 2)
                return V @ np.diag(np.sqrt(np.clip(e, 1e-15, None)))

        def sigma_points(x, P):
            L = sqrt_psd(scale * P)
            pts = np.empty((2 * n + 1, n))
            pts[0] = x
            pts[1:n + 1] = x + L.T
            pts[n + 1:] = x - L.T
            return pts

        def y_stats(Ys):
            ym = Ys.T @ Wm
            for i in ang_m:   # circular mean for angular measurements
                ym[i] = np.arctan2(np.sum(Wm * np.sin(Ys[:, i])),
                                   np.sum(Wm * np.cos(Ys[:, i])))
            dY = Ys - ym
            if ang_m:
                dY[:, ang_m] = wrap(dY[:, ang_m])
            return ym, dY

        X_hat = np.zeros_like(self.X)
        P_diag = np.zeros((self.N, n))
        P_hist = []
        for k in range(self.N):
            # measurement update
            pts = sigma_points(xh, P)
            Ys = np.array([np.asarray(s.h(pt, U[k]), dtype=float)
                           for pt in pts])
            ym, dY = y_stats(Ys)
            dX = pts - xh
            S_cov = dY.T @ (Wc[:, None] * dY) + R
            Cxy = dX.T @ (Wc[:, None] * dY)
            K = np.linalg.solve(S_cov.T, Cxy.T).T
            innov = Y[k] - ym
            if ang_m:
                innov[ang_m] = wrap(innov[ang_m])
            xh = xh + K @ innov
            P = P - K @ S_cov @ K.T
            P = (P + P.T) / 2
            X_hat[k] = xh
            P_diag[k] = np.diag(P)
            if full_cov:
                P_hist.append(P.copy())
            # time update
            if k < self.N - 1:
                pts = sigma_points(xh, P)
                Xs = np.array([sim._rk4_step(pt, U[k]) for pt in pts])
                xh = Xs.T @ Wm
                dXp = Xs - xh
                P = dXp.T @ (Wc[:, None] * dXp) + Qds[k]
                P = (P + P.T) / 2
        if full_cov:
            return X_hat, P_diag, P_hist
        return X_hat, P_diag


# ─────────────────────────────────── GUI ───────────────────────────────────

MOTIF_LABELS = {
    'straight': lambda s: f"straight {s['duration']:g}s",
    'turn': lambda s: f"turn {s['angle']:g} rad / {s['duration']:g}s",
    'circle': lambda s: f"circle {s['angle']:g} rad / {s['duration']:g}s",
    'speed': lambda s: f"speed → {s['speed']:g} m/s / {s['duration']:g}s",
    'weave': lambda s: f"weave ±{s['amp']:g} rad @ {s['freq']:g}Hz / {s['duration']:g}s",
}


class _LiveFig:
    """ A matplotlib panel rendered into an ipywidgets Output area.

    By default (``interactive=False``) it renders each update to a static PNG
    Image widget — this is deliberately NON-interactive: the ipympl live canvas
    streams every redraw to the browser and made the in-notebook GUI sluggish,
    so the main panels use plain static images (for full zoom/pan
    interactivity run the desktop app `observability_app.py`). Pass
    ``interactive=True`` to opt back into the ipympl zoom/pan canvas. """

    def __init__(self, figsize, interactive=False):
        self._figsize = tuple(figsize)
        if interactive:
            try:
                from matplotlib.figure import Figure as _F
                from ipympl.backend_nbagg import (Canvas as _C,
                                                  FigureManager as _FM)
                self.fig = _F(figsize=figsize)
                self.canvas = _C(self.fig)
                self.manager = _FM(self.canvas, 0)
                self.canvas.header_visible = False
                self.canvas.footer_visible = False
                self.canvas.toolbar_position = 'right'
                self.canvas.capture_scroll = False
                self.widget = self.canvas
                self.live = True
                return
            except Exception:
                pass
        self.fig = None
        self.widget = W.Image(format='png',
                              layout=W.Layout(width='100%',
                                              max_width='1150px'))
        self.live = False

    def draw(self, render_fn, figsize=None):
        if self.live:
            if figsize is not None and tuple(figsize) != self._figsize:
                self._figsize = tuple(figsize)
                self.fig.set_size_inches(*figsize, forward=True)
            self.fig.clf()
            render_fn(self.fig)
            self.canvas.draw_idle()
        else:
            fig = plt.figure(figsize=figsize or self._figsize)
            render_fn(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=105, bbox_inches='tight')
            plt.close(fig)
            self.widget.value = buf.getvalue()


class ObservabilityGUI:
    def __init__(self, system='fly', dt=None):
        """ :param dt: optional time-step override [s]; None uses the
        system's default (fly/fly7: 0.01, drone: 0.1). """
        self._dt_override = dt
        self.spec = self._make_spec(system)
        self.engine = ObservabilityEngine(self.spec)
        self.segments = []            # empty -> system's default trajectory
        self._suspend = False

        self._build_widgets(system)
        self._wire_callbacks()
        self._apply_system_defaults()
        self._rebuild_trajectory()    # initial simulate + compute + draw

    # -- widget construction ---------------------------------------------------
    def _build_widgets(self, system):
        L = dict(style={'description_width': '52px'})
        self.w_system = W.Dropdown(options=[('fly (pybounds example, 18 states)', 'fly'),
                                            ('fly (7-state, calibrated)', 'fly7'),
                                            ('drone (kinematic 3D)', 'drone'),
                                            ('altitude 2D (paper Fig. 4b)', 'alt2d')],
                                   value=system, description='system', **L)
        # trajectory builder
        self.w_motif = W.Dropdown(options=['straight', 'turn', 'circle',
                                           'speed', 'weave'],
                                  value='turn', description='motif', **L)
        self.w_duration = W.FloatText(value=0.1, description='dur [s]', **L)
        self.w_angle = W.FloatText(value=-np.pi / 2, description='angle rad', **L)
        self.w_speed = W.FloatText(value=0.4, description='v [m/s]', **L)
        self.w_freq = W.FloatText(value=2.0, description='f [Hz]', **L)
        self.w_amp = W.FloatText(value=np.pi / 6, description='amp rad', **L)
        self.b_add = W.Button(description='+ add segment', button_style='primary',
                              layout=W.Layout(width='120px'))
        self.b_undo = W.Button(description='undo', layout=W.Layout(width='70px'))
        self.b_clear = W.Button(description='clear (→ default traj)',
                                layout=W.Layout(width='150px'))
        self.b_left = W.Button(description='↰ +π/2 left', layout=W.Layout(width='95px'))
        self.b_right = W.Button(description='↱ −π/2 right', layout=W.Layout(width='95px'))
        self.b_fwd = W.Button(description='↑ straight', layout=W.Layout(width='95px'))
        self.b_faster = W.Button(description='+Δv', layout=W.Layout(width='60px'))
        self.b_slower = W.Button(description='−Δv', layout=W.Layout(width='60px'))
        self.html_segments = W.HTML()
        self.w_v0 = W.FloatText(value=0.3, description='v₀', **L)
        self.w_wind = W.FloatText(value=0.4, description='wind w', **L)
        self.w_zeta = W.FloatText(value=np.pi, description='ζ [rad]', **L)
        self.b_sim = W.Button(description='▶ simulate (MPC)',
                              button_style='success', layout=W.Layout(width='160px'))

        # observability params — λ range floor (1e-8) stays above the
        # round-off noise floor eps_mach * λ_max(F) for these systems
        self.w_win = W.IntSlider(value=4, min=3, max=60, description='window w',
                                 continuous_update=False, **L)
        self.w_eps = W.FloatLogSlider(value=1e-4, base=10, min=-6, max=-2, step=0.5,
                                      description='ε', continuous_update=False,
                                      readout_format='.1e', **L)
        self.w_lam = W.FloatLogSlider(value=1e-6, base=10, min=-8, max=-3, step=1,
                                      description='λ', continuous_update=False,
                                      readout_format='.0e', **L)
        self.w_color_state = W.Dropdown(options=list(self.spec.state_names),
                                        value=self.spec.color_state_default,
                                        description='color by', **L)
        self.w_ev_states = W.SelectMultiple(options=list(self.spec.state_names),
                                            value=self.spec.ev_states_default,
                                            rows=6, description='EV states', **L)
        self.w_est_states = W.SelectMultiple(options=list(self.spec.state_names),
                                             value=self.spec.ev_states_default,
                                             rows=6, description='est. states',
                                             **L)
        self.w_meas = W.SelectMultiple(options=list(self.spec.measurement_names),
                                       value=self._meas_default(self.spec),
                                       rows=6, description='sensors', **L)
        self.w_fim_sensors = W.SelectMultiple(
            options=list(self.spec.measurement_names),
            value=tuple(self.spec.fim_sensors), rows=6,
            description='FIM sens.', **L)
        self.w_fim_states = W.SelectMultiple(
            options=list(self.spec.state_names),
            value=tuple(self.spec.fim_states), rows=8,
            description='FIM states', **L)
        self.w_empirical = W.Checkbox(value=True, indent=False,
                                      description='pybounds empirical (slower, cached)')
        self.w_mc = W.Checkbox(value=False, indent=False,
                               description='nonlinear MC Gramian (slow, cached)')
        self.w_nmc = W.BoundedIntText(value=100, min=10, max=2000, step=10,
                                      description='MC traj.', **L)
        self.w_init = W.BoundedFloatText(value=1.0, min=0.0, max=100.0,
                                         step=0.5, description='init err ×',
                                         tooltip='scales the EKF/UKF initial '
                                         'estimate error and P0 (filters '
                                         'only — unrelated to the MC Gramian, '
                                         'whose knob is "MC traj.")', **L)
        self.w_seed = W.BoundedIntText(value=0, min=0, max=2**31 - 1,
                                       description='seed', **L)
        self.w_x0off = W.Text(value='', description='x̂₀ offset',
                              placeholder='e.g. z:7.5',
                              tooltip='deterministic offsets added to the '
                              'filters\' initial estimate, "state:value" '
                              'comma-separated (paper Fig. 4d: z:7.5 for the '
                              'z₀=9 guess, z:-1 for 0.5); P₀ widens '
                              'consistently', **L)
        self.w_unoise = W.BoundedFloatText(value=0.0, min=0.0, max=1e6,
                                           step=0.01, description='u noise',
                                           tooltip='variance of the noise on '
                                           'the measured inputs the filters '
                                           'receive (paper: 1e-2); truth '
                                           'flies clean inputs', **L)
        self.w_ubias = W.FloatText(value=0.0, description='u bias',
                                   tooltip='constant bias added to one input '
                                   'channel inside [t0, t1)', **L)
        self.w_ubias_ch = W.Dropdown(options=list(self.spec.input_names),
                                     description='bias ch.', **L)
        self.w_ubias_t0 = W.FloatText(value=20.0, description='bias t₀', **L)
        self.w_ubias_t1 = W.FloatText(value=28.0, description='bias t₁', **L)

        self.gb_r = W.GridBox([], layout=W.Layout(
            width='100%', grid_template_columns='repeat(auto-fill, 128px)'))
        self.gb_q = W.GridBox([], layout=W.Layout(
            width='100%', grid_template_columns='repeat(auto-fill, 128px)'))
        self.r_boxes = {}
        self.q_boxes = {}
        self.b_q_tiny = W.Button(description='Q → 0', layout=W.Layout(width='80px'))
        self.b_q_real = W.Button(description='realistic Q', layout=W.Layout(width='100px'))
        self.b_r_default = W.Button(description='reset R', layout=W.Layout(width='80px'))

        self.status = W.HTML()
        self.w_prog = W.FloatProgress(value=0.0, min=0.0, max=1.0,
                                      bar_style='info',
                                      layout=W.Layout(width='320px',
                                                      height='22px'))
        # persistent interactive figures (zoom/pan via the ipympl toolbar),
        # redrawn in place — no Output-widget duplication possible
        self.fig_main = _LiveFig((11, 7.6))
        self.fig_meas = _LiveFig((11, 4.4))
        self.fig_est = _LiveFig((11, 4.4))
        self.b_dice = W.Button(description='🎲 new noise realization',
                               layout=W.Layout(width='190px'))

        # ── trackpad trajectory recorder (interactive ipympl canvas) ──
        self.recorded = None        # (speed_sp, heading_sp) from a drawn path
        self._rec_pts = []
        self._rec_drawing = False
        self._rec_speed = self.spec.v0_default
        try:
            from matplotlib.figure import Figure as _Figure
            from ipympl.backend_nbagg import (Canvas as _Canvas,
                                              FigureManager as _FigureManager)
            rfig = _Figure(figsize=(5, 5))
            self._rec_canvas = _Canvas(rfig)
            # the canvas needs a FigureManager: frontend messages (resize,
            # dpi, events) are dispatched via canvas.manager, which is None
            # otherwise and crashes on the first resize message
            self._rec_manager = _FigureManager(self._rec_canvas, 0)
            self._rec_canvas.header_visible = False
            self._rec_canvas.footer_visible = False
            self._rec_canvas.toolbar_visible = False
            self._rec_canvas.capture_scroll = False
            self._rec_ax = rfig.add_subplot(111)
            self._rec_line, = self._rec_ax.plot([], [], 'b-', lw=2)
            self._rec_start, = self._rec_ax.plot([], [], 'go', ms=6)
            self._rec_canvas.mpl_connect('button_press_event', self._rec_on_press)
            self._rec_canvas.mpl_connect('motion_notify_event', self._rec_on_move)
            self._rec_canvas.mpl_connect('button_release_event', self._rec_on_release)
            self._rec_canvas.mpl_connect('key_press_event', self._rec_on_key)
            rec_widget = self._rec_canvas
        except Exception as e:                      # ipympl missing/broken
            self._rec_canvas = None
            rec_widget = W.HTML(f'<i>trackpad recorder unavailable ({e})</i>')
        self.b_rec_apply = W.Button(description='✓ use drawn path',
                                    button_style='success',
                                    layout=W.Layout(width='150px'))
        self.b_rec_clear = W.Button(description='✗ clear drawing',
                                    layout=W.Layout(width='130px'))
        self.rec_status = W.HTML()
        rec_help = W.HTML(
            'Draw the flight path with the mouse/trackpad (click-drag; extra '
            'strokes extend the path). Click the canvas once to give it '
            'keyboard focus, then: <b>↑/↓</b> speed, <b>c</b> clear, '
            '<b>enter</b> apply. The MPC tracks the drawn path\'s '
            '<i>heading</i> at constant speed, so the flown trajectory follows '
            'the drawn shape (wind and dynamics permitting), not its exact '
            'position.')
        rec_row = W.Layout(display='flex', flex_flow='row wrap')
        self.acc_rec = W.Accordion(children=[W.VBox(
            [rec_help, rec_widget,
             W.HBox([self.b_rec_apply, self.b_rec_clear, self.rec_status],
                    layout=rec_row)])], selected_index=None)
        self.acc_rec.set_title(0, '🎨 record trajectory (trackpad + keyboard)')

        wrap = W.Layout(display='flex', flex_flow='row wrap', width='100%')
        motif_params = W.HBox([self.w_motif, self.w_duration, self.w_angle,
                               self.w_speed, self.w_freq, self.w_amp],
                              layout=wrap)
        seg_buttons = W.HBox([self.b_add, self.b_undo, self.b_clear,
                              W.Label('  quick:'), self.b_left, self.b_right,
                              self.b_fwd, self.b_faster, self.b_slower],
                             layout=wrap)
        sim_row = W.HBox([self.w_system, self.w_v0, self.w_wind,
                          self.w_zeta, self.b_sim], layout=wrap)
        traj_box = W.VBox([W.HTML('Motif parameters: turn/circle→angle, '
                                  'speed→v, weave→f &amp; amp; empty segment '
                                  'list = the system\'s default trajectory'),
                           sim_row, motif_params, seg_buttons,
                           self.acc_rec, self.html_segments],
                          layout=W.Layout(width='100%'))

        # ── section 1: computation settings (these change the numbers) ──
        col_params = W.VBox([W.HTML('<b>analysis parameters</b>'),
                             self.w_win, self.w_eps, self.w_lam])
        col_methods = W.VBox([W.HTML('<b>EV methods</b>'),
                              self.w_empirical, self.w_mc, self.w_nmc])
        fim_col = W.VBox([W.HTML('<b>FIM subset</b> (sensors & states used '
                                 'by all EV pipelines)'),
                          self.w_fim_sensors, self.w_fim_states])
        r_grid = W.VBox([W.HTML('<b>R</b> (sensor noise variance — '
                                'angle sensors in <b>rad²</b>)'),
                         self.gb_r, self.b_r_default],
                        layout=W.Layout(flex='1 1 280px', max_width='420px'))
        q_grid = W.VBox([W.HTML('<b>Q</b> (process noise PSD — angle states '
                                'in <b>rad²/s</b>; e.g. Q<sub>ζ</sub> = 0.1 '
                                '⇒ σ<sub>ζ</sub> ≈ 0.32·√t rad)'),
                         self.gb_q,
                         W.HBox([self.b_q_tiny, self.b_q_real])],
                        layout=W.Layout(flex='1 1 280px', max_width='420px'))
        comp_box = W.HBox([col_params, col_methods, fim_col, r_grid, q_grid],
                          layout=W.Layout(display='flex', flex_flow='row wrap',
                                          width='100%', padding='6px',
                                          border='1px solid #4682b4'))

        # ── section 2: display settings (plot selection only) ──
        disp_box = W.HBox([W.VBox([self.w_color_state]), self.w_ev_states],
                          layout=W.Layout(display='flex', flex_flow='row wrap',
                                          width='100%', padding='6px',
                                          border='1px solid #999999'))

        # per-panel control rows (live next to the panel they configure)
        panel_row = W.Layout(display='flex', flex_flow='row wrap',
                             width='100%', padding='4px',
                             border='1px solid #cccccc')
        meas_row = W.HBox([self.w_meas], layout=panel_row)
        bias_box = W.VBox([W.HTML('<b>biased input noise</b> (constant '
                                  'offset on one measured-input channel '
                                  'inside [t₀, t₁))'),
                           W.HBox([self.w_ubias, self.w_ubias_ch,
                                   self.w_ubias_t0, self.w_ubias_t1],
                                  layout=W.Layout(display='flex',
                                                  flex_flow='row wrap'))],
                          layout=W.Layout(border='1px solid #cc8888',
                                          padding='4px', margin='2px'))
        est_row = W.HBox([self.w_est_states,
                          W.VBox([self.w_seed, W.HBox([self.b_dice])]),
                          W.VBox([self.w_init, self.w_x0off, self.w_unoise]),
                          bias_box],
                         layout=panel_row)

        def _acc(title, box, open_=True):
            a = W.Accordion(children=[box],
                            selected_index=0 if open_ else None,
                            layout=W.Layout(width='100%'))
            a.set_title(0, title)
            return a

        acc_traj = _acc('✈ Trajectory — defines the nominal trajectory '
                        '(every result depends on it)', traj_box)
        acc_comp = _acc('⚙ Computation settings — change the calculations '
                        '(windows, λ/ε, R, Q, FIM subsets, EV methods)',
                        comp_box)
        acc_disp = _acc('🎨 Display settings — plot selection only '
                        '(cached; nothing recomputed)', disp_box, open_=False)
        acc_meas = _acc('Measurements panel — sensor selection (display '
                        'only). Gray dots: noisy samples the filters see '
                        '(R + estimator seed); black: noise-free h(x,u); '
                        'blue dashed: EKF prediction h(x̂)',
                        meas_row, open_=False)
        acc_est = _acc('⚙ Estimator settings — change the filter results: '
                       'seed/🎲 realization, init err ×, x̂₀ offset, '
                       'u noise / biased input noise (est. states is '
                       'display-only)', est_row, open_=False)

        self.ui = W.VBox([acc_traj, acc_comp, acc_disp,
                          W.HBox([self.w_prog, self.status],
                                 layout=W.Layout(display='flex',
                                                 flex_flow='row wrap',
                                                 align_items='center',
                                                 width='100%')),
                          self.fig_main.widget,
                          acc_meas, self.fig_meas.widget,
                          acc_est, self.fig_est.widget],
                         layout=W.Layout(width='100%'))

    def _wire_callbacks(self):
        self.w_system.observe(self._on_system_change, names='value')
        self.b_add.on_click(lambda b: self._add_segment_from_widgets())
        self.b_undo.on_click(lambda b: (self.segments and self.segments.pop(),
                                        self._refresh_segment_list()))
        self.b_clear.on_click(lambda b: (self.segments.clear(),
                                         setattr(self, 'recorded', None),
                                         self._refresh_segment_list()))
        self.b_rec_apply.on_click(lambda b: self._rec_apply())
        self.b_rec_clear.on_click(lambda b: self._rec_clear())
        # a turn needs follow-through: the set-point ramp alone would end the
        # trajectory before the vehicle finishes responding, so append a
        # straight hold after the ramp
        self.b_left.on_click(lambda b: self._quick(
            dict(motif='turn', duration=self.spec.quick_turn_dur, angle=np.pi / 2),
            dict(motif='straight', duration=self.spec.quick_straight_dur)))
        self.b_right.on_click(lambda b: self._quick(
            dict(motif='turn', duration=self.spec.quick_turn_dur, angle=-np.pi / 2),
            dict(motif='straight', duration=self.spec.quick_straight_dur)))
        self.b_fwd.on_click(lambda b: self._quick(dict(
            motif='straight', duration=self.spec.quick_straight_dur)))
        self.b_faster.on_click(lambda b: self._quick_speed(+self.spec.speed_step))
        self.b_slower.on_click(lambda b: self._quick_speed(-self.spec.speed_step))
        self.b_sim.on_click(lambda b: self._rebuild_trajectory())
        self.b_q_tiny.on_click(lambda b: self._set_boxes(self.q_boxes,
                                                         self.spec.q_tiny))
        self.b_q_real.on_click(lambda b: self._set_boxes(self.q_boxes,
                                                         self.spec.q_realistic))
        self.b_r_default.on_click(lambda b: self._set_boxes(self.r_boxes,
                                                            self.spec.r_default))
        self.b_dice.on_click(lambda b: setattr(
            self.w_seed, 'value', int(np.random.default_rng().integers(1e6))))
        for wd in (self.w_win, self.w_eps, self.w_lam,
                   self.w_color_state, self.w_ev_states, self.w_est_states,
                   self.w_meas, self.w_fim_sensors, self.w_fim_states,
                   self.w_empirical, self.w_mc, self.w_nmc, self.w_init,
                   self.w_seed, self.w_unoise, self.w_ubias, self.w_ubias_ch,
                   self.w_ubias_t0, self.w_ubias_t1, self.w_x0off):
            wd.observe(self._on_param_change, names='value')

    def _make_spec(self, name):
        if self._dt_override is not None:
            return SYSTEMS[name](dt=float(self._dt_override))
        return SYSTEMS[name]()

    @staticmethod
    def _meas_default(spec):
        picks = tuple(m for m in ('gamma', 'a', 'g')
                      if m in spec.measurement_names)
        return picks or tuple(spec.measurement_names[:3])

    # -- system switching --------------------------------------------------------
    def _apply_system_defaults(self):
        """ Push the current spec's defaults into all widgets (no recompute). """
        s = self.spec
        self._suspend = True
        self.w_v0.value = s.v0_default
        self.w_wind.value = s.wind_default
        self.w_zeta.value = s.zeta_default
        self.w_win.value = s.w_default
        self.w_eps.value = s.eps_default
        self.w_duration.value = s.seg_duration
        self.w_speed.value = s.v0_default
        self.w_color_state.options = list(s.state_names)
        self.w_color_state.value = s.color_state_default
        self.w_ev_states.options = list(s.state_names)
        self.w_ev_states.value = s.ev_states_default
        self.w_est_states.options = list(s.state_names)
        self.w_est_states.value = s.ev_states_default
        self.w_meas.options = list(s.measurement_names)
        self.w_meas.value = self._meas_default(s)
        self.w_fim_sensors.options = list(s.measurement_names)
        self.w_fim_sensors.value = tuple(s.fim_sensors)
        self.w_ubias_ch.options = list(s.input_names)
        self.w_unoise.value = getattr(s, 'u_noise_default', 0.0)
        ub = getattr(s, 'u_bias_default', None)
        if ub:
            ch, mag, t0, t1 = ub
            self.w_ubias_ch.value = ch
            self.w_ubias.value = mag
            self.w_ubias_t0.value = t0
            self.w_ubias_t1.value = t1
        else:
            self.w_ubias_ch.value = s.input_names[0]
            self.w_ubias.value = 0.0
        self.w_x0off.value = ''
        self.w_fim_states.options = list(s.state_names)
        self.w_fim_states.value = tuple(s.fim_states)
        txt = dict(layout=W.Layout(width='118px'),
                   style={'description_width': '48px'})
        self.r_boxes = {
            m: W.FloatText(value=s.r_default[m], description=m,
                           tooltip=(f'{m} noise variance '
                                    + ('[rad²]' if m in s.angle_measurements
                                       else '[native units²]')), **txt)
            for m in s.measurement_names}
        self.q_boxes = {
            n: W.FloatText(value=s.q_tiny[n], description=n,
                           tooltip=(f'{n} process-noise PSD '
                                    + ('[rad²/s]' if n in s.angle_states
                                       else '[native units²/s]')), **txt)
            for n in s.state_names}
        for wd in list(self.r_boxes.values()) + list(self.q_boxes.values()):
            wd.observe(self._on_param_change, names='value')
        self.gb_r.children = list(self.r_boxes.values())
        self.gb_q.children = list(self.q_boxes.values())
        self._suspend = False
        self._rec_reset()

    def _on_system_change(self, change):
        if self._suspend:
            return
        self.spec = self._make_spec(change['new'])
        self.engine = ObservabilityEngine(self.spec)
        self.segments = []
        self._apply_system_defaults()
        self._rebuild_trajectory()

    # -- trackpad recorder --------------------------------------------------------
    def _rec_reset(self):
        """ Configure the recorder canvas for the current system. """
        self.recorded = None
        self._rec_pts = []
        self._rec_drawing = False
        self._rec_speed = self.spec.v0_default
        self.rec_status.value = ''
        if self._rec_canvas is None:
            return
        ax = self._rec_ax
        ax.set_xlim(*self.spec.rec_xlim)
        ax.set_ylim(*self.spec.rec_ylim)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        self._rec_line.set_data([], [])
        self._rec_start.set_data([], [])
        self._rec_set_title()
        self._rec_canvas.draw_idle()

    def _rec_set_title(self):
        self._rec_ax.set_title(
            f'draw path — speed {self._rec_speed:.2f} m/s (↑/↓ to change)',
            fontsize=9)

    def _rec_redraw(self):
        pts = np.asarray(self._rec_pts)
        self._rec_line.set_data(pts[:, 0], pts[:, 1])
        self._rec_start.set_data(pts[:1, 0], pts[:1, 1])
        self._rec_canvas.draw_idle()

    def _rec_on_press(self, ev):
        if ev.inaxes is self._rec_ax and ev.xdata is not None:
            self._rec_drawing = True
            self._rec_pts.append((ev.xdata, ev.ydata))
            self._rec_redraw()

    def _rec_on_move(self, ev):
        if (self._rec_drawing and ev.inaxes is self._rec_ax
                and ev.xdata is not None):
            self._rec_pts.append((ev.xdata, ev.ydata))
            self._rec_redraw()

    def _rec_on_release(self, ev):
        self._rec_drawing = False

    def _rec_on_key(self, ev):
        if ev.key == 'up':
            self._rec_speed *= 1.25
        elif ev.key == 'down':
            self._rec_speed = max(self._rec_speed / 1.25, 1e-3)
        elif ev.key == 'c':
            self._rec_pts = []
            self._rec_line.set_data([], [])
            self._rec_start.set_data([], [])
        elif ev.key == 'enter':
            self._rec_apply()
            return
        self._rec_set_title()
        self._rec_canvas.draw_idle()

    def _rec_clear(self):
        self._rec_pts = []
        self.recorded = None
        self.rec_status.value = ''
        if self._rec_canvas is not None:
            self._rec_line.set_data([], [])
            self._rec_start.set_data([], [])
            self._rec_canvas.draw_idle()
        self._refresh_segment_list()

    def _rec_apply(self):
        """ Convert the drawn path into (speed, heading) set-points at the
        recorder speed, resampled uniformly in arc length, and simulate. """
        pts = np.asarray(self._rec_pts, dtype=float)
        if len(pts) >= 2:
            d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            pts = pts[np.concatenate([[True], d > 1e-12])]
        if len(pts) < 5:
            self.rec_status.value = ('<span style="color:firebrick">draw a '
                                     'longer path first</span>')
            return
        seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        sarc = np.concatenate([[0.0], np.cumsum(seglen)])
        v, dt = self._rec_speed, self.spec.dt
        N = int(np.clip(round(sarc[-1] / (v * dt)), 8, 800))
        si = np.linspace(0.0, sarc[-1], N)
        xi = np.interp(si, sarc, pts[:, 0])
        yi = np.interp(si, sarc, pts[:, 1])
        heading_sp = np.unwrap(np.arctan2(np.gradient(yi), np.gradient(xi)))
        self.recorded = (v * np.ones(N), heading_sp)
        self.segments.clear()
        self.rec_status.value = (f'recorded {sarc[-1]:.2f} m path → {N} steps '
                                 f'({N * dt:.2f} s at {v:.2f} m/s)')
        self._refresh_segment_list()
        self._rebuild_trajectory()

    # -- segment helpers --------------------------------------------------------
    def _add_segment_from_widgets(self):
        seg = dict(motif=self.w_motif.value, duration=float(self.w_duration.value),
                   angle=float(self.w_angle.value), speed=float(self.w_speed.value),
                   freq=float(self.w_freq.value), amp=float(self.w_amp.value))
        self.segments.append(seg)
        self._refresh_segment_list()

    def _quick(self, *segs):
        self.segments.extend(segs)
        self._refresh_segment_list()

    def _quick_speed(self, dv):
        v_last = float(self.w_v0.value)
        for s in self.segments:
            if s['motif'] == 'speed':
                v_last = s['speed']
        self.segments.append(dict(motif='speed',
                                  duration=self.spec.seg_duration,
                                  speed=v_last + dv))
        self._refresh_segment_list()

    def _refresh_segment_list(self):
        if not self.segments:
            if self.recorded is not None:
                n_rec = len(self.recorded[0])
                self.html_segments.value = (
                    f'<i>🎨 recorded trackpad trajectory ({n_rec} steps, '
                    f'{n_rec * self.spec.dt:.2f} s) — clear to discard</i>')
            else:
                self.html_segments.value = (
                    f'<i>default: {self.spec.default_traj_label} '
                    f'— add segments to build your own</i>')
            return
        items = ' &nbsp;→&nbsp; '.join(
            f'<b>{i+1}.</b> {MOTIF_LABELS[s["motif"]](s)}'
            for i, s in enumerate(self.segments))
        total = sum(s['duration'] for s in self.segments)
        self.html_segments.value = (f'{items} &nbsp; <i>(total {total:g} s — press '
                                    f'▶ simulate to apply)</i>')

    def _set_boxes(self, boxes, values):
        for k, wd in boxes.items():
            wd.unobserve(self._on_param_change, names='value')
            wd.value = values[k]
            wd.observe(self._on_param_change, names='value')
        self._on_param_change(None)

    def _parse_x0_offset(self):
        txt = self.w_x0off.value.strip()
        if not txt:
            return None
        out = {}
        for part in txt.split(','):
            if ':' in part:
                k, v = part.split(':', 1)
                k = k.strip()
                if k in self.spec.state_names:
                    try:
                        out[k] = float(v)
                    except ValueError:
                        pass
        return out or None

    def _set_status(self, txt, frac=None):
        """ Large status line + stage progress bar (updates render live
        mid-callback). """
        self.status.value = ('<div style="font-size:135%; font-weight:600; '
                             'padding:3px 0">' + txt + '</div>')
        if frac is not None:
            self.w_prog.value = float(frac)
            self.w_prog.bar_style = 'success' if frac >= 1.0 else 'info'

    # -- parameter collection ----------------------------------------------------
    def _params(self):
        r_diag = {m: max(float(wd.value), 1e-30) for m, wd in self.r_boxes.items()}
        q_diag = {n: max(float(wd.value), 1e-30) for n, wd in self.q_boxes.items()}
        w = int(min(self.w_win.value, max(self.engine.N - 1, 3)))
        sensors = tuple(self.w_fim_sensors.value) or tuple(
            self.spec.measurement_names)
        states = tuple(self.w_fim_states.value) or tuple(self.spec.state_names)
        return dict(w=w, eps=float(self.w_eps.value), lam=float(self.w_lam.value),
                    r_diag=r_diag, q_diag=q_diag, sensors=sensors,
                    states=states)

    # -- actions -----------------------------------------------------------------
    def _rebuild_trajectory(self):
        self._refresh_segment_list()
        s = self.spec
        w_wind = float(self.w_wind.value)
        zeta = float(self.w_zeta.value)
        if self.segments:
            speed_sp, heading_sp = build_setpoints(
                self.segments, s.dt, v0=float(self.w_v0.value),
                heading0=s.heading0)
            setpoint = s.make_setpoint(speed_sp, heading_sp, w_wind, zeta)
        elif self.recorded is not None:
            speed_sp, heading_sp = self.recorded
            setpoint = s.make_setpoint(speed_sp, heading_sp, w_wind, zeta)
        else:
            setpoint = s.default_setpoint(w_wind, zeta)
        self._set_status('⏳ solving MPC trajectory …', 0.05)
        t0 = time.time()
        self.engine.simulate_mpc(setpoint)
        self._set_status('trajectory solved — computing observability …', 0.2)
        t_sim = time.time() - t0
        self._update(f'trajectory: {self.engine.N} steps in {t_sim:.1f} s')

    def _on_param_change(self, change):
        if self._suspend or not self.engine.N:
            return
        self._update()

    # -- compute + draw ------------------------------------------------------------
    def _update(self, note=''):
        try:
            self._update_inner(note)
        except np.linalg.LinAlgError as e:
            self._set_status('<span style="color:firebrick">⚠ singular '
                             f'matrix at these settings ({e}) — typically '
                             'λ far below the noise floor of F, or a '
                             'degenerate R/Q combination. Raise λ or '
                             'adjust R/Q; plots show the previous '
                             'result.</span>', 1.0)

    def _update_inner(self, note=''):
        p = self._params()
        self._set_status('⏳ linearized Eq. (33) pipeline …', 0.25)
        t0 = time.time()
        ev_lin = self.engine.linearized_ev(p['w'], p['q_diag'], p['r_diag'],
                                           p['lam'], sensors=p['sensors'],
                                           states=p['states'])
        t_lin = time.time() - t0

        ev_mc, mc_note = None, ''
        if self.w_mc.value:
            self._set_status('⏳ nonlinear Monte-Carlo Gramian … '
                             '(scales with MC traj. × windows; cached)', 0.4)
            t0 = time.time()
            n_mc = int(self.w_nmc.value)
            ev_mc, mc_cached = self.engine.mc_ev(
                p['w'], p['q_diag'], p['r_diag'], p['lam'],
                sensors=p['sensors'], states=p['states'],
                n_mc=n_mc, seed=0)
            mc_note = (' | MC: cached' if mc_cached
                       else f' | MC: {time.time() - t0:.1f} s')

        ev_emp, emp_note = None, ''
        if self.w_empirical.value:
            self._set_status('⏳ pybounds empirical pipeline … (slower the '
                             'first time for a given trajectory / w / ε; '
                             'cached afterwards)', 0.55)
            t0 = time.time()
            ev_emp, cached = self.engine.empirical_ev(
                p['w'], p['eps'], p['r_diag'], p['lam'],
                sensors=p['sensors'], states=p['states'])
            emp_note = (' | empirical: cached' if cached
                        else f' | empirical: {time.time() - t0:.1f} s')

        self._set_status('⏳ filters (EKF / UKF) + rendering …', 0.8)
        self._draw(ev_lin, ev_emp, p, ev_mc=ev_mc)
        try:
            seed = int(self.w_seed.value)
            scale = float(self.w_init.value)
            x0off = self._parse_x0_offset()
            unv = max(float(self.w_unoise.value), 0.0)
            ub = None
            if float(self.w_ubias.value) != 0.0:
                ub = (self.spec.input_names.index(self.w_ubias_ch.value),
                      float(self.w_ubias.value),
                      float(self.w_ubias_t0.value),
                      float(self.w_ubias_t1.value))
            results = {}
            for name, fn in (('EKF', self.engine.run_ekf),
                             ('UKF', self.engine.run_ukf)):
                fkey = (name, self.engine._version,
                        tuple(f'{p["q_diag"][k]:.3e}'
                              for k in self.spec.state_names),
                        tuple(f'{p["r_diag"][m]:.3e}'
                              for m in self.spec.measurement_names),
                        seed, round(scale, 6), round(unv, 9), ub,
                        tuple(sorted(x0off.items())) if x0off else None)
                if fkey not in self.engine._filt_cache:
                    self.engine._filt_cache[fkey] = fn(
                        p['q_diag'], p['r_diag'], seed=seed, x0_scale=scale,
                        x0_offset=x0off, u_noise_var=unv, u_bias=ub)
                results[name] = self.engine._filt_cache[fkey]
            self._draw_estimates(results)
            self._draw_measurements(results['EKF'][0], p)
        except np.linalg.LinAlgError:
            pass  # keep the previous panel if a filter breaks down
        floor = self.engine.lam_noise_floor
        floor_note = ''
        if p['lam'] < 10 * floor:
            floor_note = (f' | <span style="color:firebrick">⚠ λ = {p["lam"]:.0e} '
                          f'is at/below the round-off noise floor of F '
                          f'(eps·λ_max ≈ {floor:.0e}) — EVs may be unreliable; '
                          f'raise λ</span>')
        self._set_status(f'✓ done — {note + " | " if note else ""}'
                         f'linearized: {t_lin * 1e3:.0f} ms{emp_note}'
                         f'{mc_note}{floor_note}', 1.0)

    def _draw(self, ev_lin, ev_emp, p, ev_mc=None):
        self.fig_main.draw(
            lambda f: self._render(f, ev_lin, ev_emp, p, ev_mc=ev_mc))

    def _draw_estimates(self, results):
        s = self.spec
        t, X = self.engine.t, self.engine.X
        states = list(self.w_est_states.value) or [self.spec.color_state_default]
        states = states[:8]
        ncols = 2 if len(states) > 3 else 1
        nrows = int(np.ceil(len(states) / ncols))

        def render(fig):
            axs = fig.subplots(nrows, ncols, sharex=True, squeeze=False)
            colors = {'EKF': 'royalblue', 'UKF': 'darkorange'}
            styles = {'EKF': '--', 'UKF': '-.'}
            for i, st in enumerate(states):
                ax = axs.ravel()[i]
                j = s.state_names.index(st)
                for name, (X_hat, P_diag) in results.items():
                    sd = np.sqrt(np.clip(P_diag[:, j], 0.0, None))
                    ax.fill_between(t, X_hat[:, j] - 3 * sd,
                                    X_hat[:, j] + 3 * sd,
                                    color=colors[name], alpha=0.12, lw=0)
                    ax.plot(t, X_hat[:, j], styles[name], color=colors[name],
                            lw=1.5, label=f'{name} ±3σ')
                ax.plot(t, X[:, j], 'k-', lw=1.8, label='true')
                ax.set_ylabel(st)
                ax.grid(alpha=0.3)
                if i == 0:
                    ax.legend(fontsize=7, loc='best')
            for ax in axs.ravel()[len(states):]:
                ax.set_visible(False)
            for ax in axs[-1, :]:
                ax.set_xlabel('time [s]')
            fig.suptitle('true vs estimated states — EKF vs UKF', fontsize=10)
            fig.tight_layout()

        self.fig_est.draw(render, figsize=(11, max(2.2 * nrows, 2.4)))

    def _draw_measurements(self, X_hat_ekf, p):
        """ Selected sensors vs time: the noisy samples the filters consume
        (drawn with the current R and seed — identical to the filter input),
        the noise-free h along the true trajectory, and the EKF's measurement
        prediction h(x_hat) (its smoothness/aggressiveness is set by Q). """
        s = self.spec
        t, X, U = self.engine.t, self.engine.X, self.engine.U
        sensors = list(self.w_meas.value) or list(self._meas_default(s))
        sensors = sensors[:8]
        Y_noisy, *_ = self.engine._estimation_problem(
            p['r_diag'], int(self.w_seed.value))
        Y_true = np.array([np.asarray(s.h(X[k], U[k]), dtype=float)
                           for k in range(self.engine.N)])
        Y_hat = np.array([np.asarray(s.h(X_hat_ekf[k], U[k]), dtype=float)
                          for k in range(self.engine.N)])
        ncols = 2 if len(sensors) > 3 else 1
        nrows = int(np.ceil(len(sensors) / ncols))

        def render(fig):
            axs = fig.subplots(nrows, ncols, sharex=True, squeeze=False)
            for i, m in enumerate(sensors):
                ax = axs.ravel()[i]
                j = s.measurement_names.index(m)
                ax.plot(t, Y_noisy[:, j], '.', color='0.55', ms=3.5,
                        label='noisy samples (R)')
                ax.plot(t, Y_true[:, j], 'k-', lw=1.6,
                        label='noise-free h(x,u)')
                ax.plot(t, Y_hat[:, j], '--', color='royalblue', lw=1.4,
                        label='EKF prediction h(x̂)')
                unit = ' [rad]' if m in s.angle_measurements else ''
                ax.set_ylabel(m + unit)
                ax.grid(alpha=0.3)
                if i == 0:
                    ax.legend(fontsize=7, loc='best')
            for ax in axs.ravel()[len(sensors):]:
                ax.set_visible(False)
            for ax in axs[-1, :]:
                ax.set_xlabel('time [s]')
            fig.suptitle('measurements: noisy (R) | true | EKF prediction',
                         fontsize=10)
            fig.tight_layout()

        self.fig_meas.draw(render, figsize=(11, max(2.0 * nrows, 2.2)))

    def _ev_along_path(self, src, cs):
        """ Map a pipeline's min-EV of state `cs` onto trajectory points
        (window-start alignment, edges padded). """
        ev_path = np.full(self.engine.N, np.nan)
        vals = src[cs].values
        ti = src['time_initial'].values
        ok = np.isfinite(ti) & np.isfinite(vals)
        ev_path[np.rint(ti[ok] / self.engine.dt).astype(int)] = vals[ok]
        fin = np.flatnonzero(np.isfinite(ev_path))
        if len(fin):
            ev_path[:fin[0]] = ev_path[fin[0]]
            ev_path[fin[-1]:] = ev_path[fin[-1]]
        return ev_path

    def _plot_colored_traj(self, fig, ax, ev_path, norm, cs, src_lab):
        X = self.engine.X
        ax.plot(X[:, 0], X[:, 1], 'k-', lw=0.3, alpha=0.4)
        if ev_path is not None and norm is not None:
            with contextlib.redirect_stdout(io.StringIO()):  # colorline prints its norm
                lc = colorline(X[:, 0], X[:, 1], ev_path, ax=ax,
                               cmap=plt.get_cmap('inferno_r'), norm=norm,
                               linewidth=4)
            cb = fig.colorbar(lc, ax=ax, pad=0.02)
            cb.set_label(f'min EV: {cs}', fontsize=8)
            cb.ax.tick_params(labelsize=7)
        else:
            ax.annotate(f'no EV to show for "{cs}" —\nadd it to "FIM states"'
                        '\n(or enable pybounds empirical)',
                        xy=(0.5, 0.5), xycoords='axes fraction',
                        ha='center', va='center', fontsize=9, color='gray')
        ax.plot(X[0, 0], X[0, 1], 'go', ms=7, zorder=4, label='start')
        self._draw_wind_glyph(ax)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title(f'min EV of {cs} — {src_lab}', fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    def _draw_wind_glyph(self, ax):
        """ Corner glyph: arrow along the wind direction zeta, with a shaded fan
        spanning ±1σ where σ = sqrt(Q_ζ · n_w) is the wind-direction drift the
        current per-step Q implies over one w-step analysis window. """
        s = self.spec
        if 'w' not in s.state_names or 'zeta' not in s.state_names:
            return
        X = self.engine.X
        w_wind = X[0, s.state_names.index('w')]
        zeta = X[0, s.state_names.index('zeta')]
        q_zeta = max(float(self.q_boxes['zeta'].value), 0.0)
        n_w = min(int(self.w_win.value), max(self.engine.N - 1, 1))   # in STEPS
        T_w = n_w * self.engine.dt                                    # in seconds
        # the Q boxes hold the PER-STEP covariance Q_d (cf. _lin, which divides by
        # dt to recover the PSD). ζ is a random walk, so its drift variance over
        # the window is Q_d·n_w ( = Q_c·T_w ); using T_w here instead of n_w would
        # understate σ by a factor √dt.
        sigma = np.sqrt(q_zeta * n_w)   # drift over one analysis window
        half = min(sigma, np.pi)                # fan half-angle = 1σ, capped
        cx, cy, r = 0.13, 0.86, 0.10            # axes-fraction center/radius
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
        label = f'wind {w_wind:.2g} m/s, ζ={zeta:.2f} rad'
        if half > 0.01:
            label += f'\n±1σ(ζ) = {half:.2f} rad over window w = {T_w:.2g} s'
        ax.text(cx, cy - r - 0.02, label, transform=tf, ha='center', va='top',
                fontsize=6.5, color='steelblue', zorder=6)

    def _render(self, fig, ev_lin, ev_emp, p, ev_mc=None):
        cs = self.w_color_state.value
        ev_states = list(self.w_ev_states.value) or [self.spec.color_state_default]
        gs = fig.add_gridspec(2, 2)
        ax_lin = fig.add_subplot(gs[0, 0])
        ax_emp = fig.add_subplot(gs[0, 1])
        ax_ev = fig.add_subplot(gs[1, :])

        # (top row) trajectory colored by min EV — linearized | pybounds,
        # on a SHARED log color scale so the two colorings are comparable
        path_lin = self._ev_along_path(ev_lin, cs) if cs in ev_lin else None
        path_emp = (self._ev_along_path(ev_emp, cs)
                    if ev_emp is not None and cs in ev_emp else None)
        arrays = [v[np.isfinite(v)] for v in (path_lin, path_emp)
                  if v is not None]
        allv = np.concatenate(arrays) if arrays else np.array([])
        allv = allv[allv > 0]
        norm = None
        if len(allv):
            lo = int(np.floor(np.log10(allv.min())))
            hi = int(np.ceil(np.log10(allv.max())))
            norm = mpl.colors.LogNorm(10.0 ** lo, 10.0 ** max(hi, lo + 1))
        self._plot_colored_traj(fig, ax_lin, path_lin, norm, cs,
                                'linearized Eq. (33)')
        self._plot_colored_traj(fig, ax_emp, path_emp,
                                norm if path_emp is not None else None,
                                cs, 'pybounds empirical')

        # (bottom row) per-state COLOR, per-method LINESTYLE:
        # solid = linearized Eq.(33), dashed = pybounds empirical,
        # dotted = nonlinear Monte-Carlo Gramian
        from matplotlib.lines import Line2D
        cmap = plt.get_cmap('tab10')

        def plot_methods(ax, x_of, lin_df, emp_df, mc_df, plotter):
            handles = []
            for i, s in enumerate(ev_states):
                c = cmap(i % 10)
                if s in lin_df:
                    plotter(ax, x_of(lin_df), lin_df[s], '-', c, 1.8)
                if emp_df is not None and s in emp_df:
                    plotter(ax, x_of(emp_df), emp_df[s], '--', c, 1.3)
                if mc_df is not None and s in mc_df:
                    plotter(ax, x_of(mc_df), mc_df[s], ':', c, 1.6)
                handles.append(Line2D([], [], color=c, label=s))
            handles.append(Line2D([], [], color='k', ls='-',
                                  label='linearized'))
            if emp_df is not None:
                handles.append(Line2D([], [], color='k', ls='--',
                                      label='pybounds'))
            if mc_df is not None:
                handles.append(Line2D([], [], color='k', ls=':',
                                      label='nonlinear MC'))
            return handles

        semi = lambda ax, x, y, ls, c, lw: ax.semilogy(x, y, ls, color=c, lw=lw)

        h1 = plot_methods(ax_ev, lambda d: d['time'],
                          ev_lin, ev_emp, ev_mc, semi)
        ax_ev.axhline(1.0 / p['lam'], color='gray', ls=':', lw=1)
        h1.append(Line2D([], [], color='gray', ls=':', label='1/λ floor'))
        ax_ev.set_xlabel('window start time [s]')
        ax_ev.set_ylabel('min error variance')
        ax_ev.set_title(f'min EV, sliding window w={p["w"]}', fontsize=9)
        ax_ev.legend(handles=h1, fontsize=6, ncol=3)
        ax_ev.grid(alpha=0.3, which='both')

        fig.tight_layout()

    def display(self):
        display(self.ui)
