"""
Self-contained 3D kinematic drone model (11 physical states) with a lightweight
RK4 simulator that duck-types the interface expected by pybounds'
EmpiricalObservabilityMatrix / SlidingEmpiricalObservabilityMatrix:

    y = simulator.simulate(x0, u, aux=None)   ->  (w x p) array
    simulator.state_names, simulator.measurement_names

The dynamics/measurements follow drone_model_kinematic_3d.py from the project,
with the constant parameter states (k_phi, k_theta, k_psi, k_thrust, C) folded
into fixed parameters. This keeps the process-noise covariance Q strictly
positive definite, which the recursive stochastic observability Gramian
requires (see notes in stochastic_observability.py and the notebook).

States (n = 11):
    x, y, z          : position in inertial frame [m]
    v_x, v_y, v_z    : velocity in body-level frame [m/s]
    phi, theta, psi  : roll, pitch, yaw [rad]
    w, zeta          : wind speed [m/s] and wind direction [rad]

Inputs (m = 6):
    u_thrust, u_phi, u_theta, u_psi, u_w, u_zeta

Measurements (p = 10):
    r_x, r_y  : optic flow (v_x / z, v_y / z)  [1/s]
    g         : ground speed  [m/s]
    beta      : course direction in body-level frame [rad]
    a         : airspeed magnitude [m/s]
    gamma     : apparent airflow angle [rad]
    q_x, q_y, q_z : accelerometer [m/s^2]
    psi       : heading / yaw (e.g. magnetometer) [rad] — the same signal as
                the psi state, observed directly; named psi here to match the
                paper (arXiv 2511.08766 Fig. 2b) and the fly model, which
                likewise reuses phi for both its state and its measurement
"""

import numpy as np

GRAVITY = 9.81
DRAG_C = 0.01

STATE_NAMES = ['x', 'y', 'z', 'v_x', 'v_y', 'v_z', 'phi', 'theta', 'psi', 'w', 'zeta']
INPUT_NAMES = ['u_thrust', 'u_phi', 'u_theta', 'u_psi', 'u_w', 'u_zeta']
MEASUREMENT_NAMES = ['r_x', 'r_y', 'g', 'beta', 'a', 'gamma',
                     'q_x', 'q_y', 'q_z', 'psi']


def f(X, U):
    """ Continuous-time dynamics x_dot = f(x, u). Vectorized over scalars. """
    x, y, z, v_x, v_y, v_z, phi, theta, psi, w, zeta = X
    u_thrust, u_phi, u_theta, u_psi, u_w, u_zeta = U

    # Air-relative velocity in body-level frame
    a_x = v_x - w * np.cos(psi - zeta)
    a_y = v_y + w * np.sin(psi - zeta)

    # Attitude kinematics (rate control, gains k_* = 1)
    phi_dot = u_phi
    theta_dot = u_theta
    psi_dot = u_psi

    # Position kinematics in inertial frame
    x_dot = v_x * np.cos(psi) - v_y * np.sin(psi)
    y_dot = v_x * np.sin(psi) + v_y * np.cos(psi)
    z_dot = v_z

    # Body-level translational dynamics (k_thrust = 1)
    v_x_dot = u_thrust * np.cos(phi) * np.sin(theta) + v_y * psi_dot - DRAG_C * a_x
    v_y_dot = -u_thrust * np.sin(phi) - v_x * psi_dot - DRAG_C * a_y
    v_z_dot = -u_thrust * np.cos(phi) * np.cos(theta) + GRAVITY

    # Wind kinematics
    w_dot = u_w
    zeta_dot = u_zeta

    return np.array([x_dot, y_dot, z_dot,
                     v_x_dot, v_y_dot, v_z_dot,
                     phi_dot, theta_dot, psi_dot,
                     w_dot, zeta_dot])


def h(X, U):
    """ Measurement model y = h(x, u). No angle unwrapping (keeps the map smooth
    for both finite-difference perturbations and analytic linearization). """
    x, y, z, v_x, v_y, v_z, phi, theta, psi, w, zeta = X
    u_thrust, u_phi, u_theta, u_psi, u_w, u_zeta = U

    xdot = f(X, U)
    v_x_dot, v_y_dot = xdot[3], xdot[4]
    v_z_dot = xdot[5]
    psi_dot = xdot[8]

    # Optic flow & course
    g = np.sqrt(v_x ** 2 + v_y ** 2)
    r_x = v_x / z
    r_y = v_y / z
    beta = np.arctan2(v_y, v_x)

    # Airspeed & apparent airflow angle
    a_x = v_x - w * np.cos(psi - zeta)
    a_y = v_y + w * np.sin(psi - zeta)
    a = np.sqrt(a_x ** 2 + a_y ** 2)
    gamma = np.arctan2(a_y, a_x)

    # Accelerometer (specific-force style, as in the project model)
    q_x = v_x_dot - v_y * psi_dot
    q_y = v_y_dot + v_x * psi_dot
    q_z = v_z_dot

    return np.array([r_x, r_y, g, beta, a, gamma, q_x, q_y, q_z, psi])


class DroneSimulatorRK4:
    """ Minimal fixed-step RK4 simulator, pybounds-compatible.

    simulate(x0, u, aux=None) integrates the dynamics under the open-loop input
    sequence u ((w x m) array or dict) starting at x0, and returns the (w x p)
    measurement array with y_k = h(x_k, u_k).
    """

    def __init__(self, dt=0.1):
        self.dt = dt
        self.state_names = list(STATE_NAMES)
        self.input_names = list(INPUT_NAMES)
        self.measurement_names = list(MEASUREMENT_NAMES)
        self.n = len(self.state_names)
        self.m = len(self.input_names)
        self.p = len(self.measurement_names)

    def _rk4_step(self, x, u):
        dt = self.dt
        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)
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
        y[0] = h(x[0], u[0])
        for k in range(w - 1):
            x[k + 1] = self._rk4_step(x[k], u[k])
            y[k + 1] = h(x[k + 1], u[k + 1])

        self.x_last = x  # stash for convenience
        return y


def generate_trajectory(dt=0.1, T=60.0):
    """ Generate an example closed-loop drone trajectory with turns and
    accelerations (the maneuvers that make wind observable), then return the
    recorded state and *open-loop* input sequences.

    A simple cascaded PD controller tracks:
      - piecewise-constant forward-speed setpoints (acceleration motifs)
      - piecewise-linear heading setpoints (turn motifs)
      - constant altitude
    Wind is constant: w = 1.2 m/s from zeta = 45 deg.
    """
    sim = DroneSimulatorRK4(dt=dt)
    t = np.arange(0.0, T + dt / 2, dt)
    N = len(t)

    # --- setpoints -----------------------------------------------------------
    vx_sp = np.piecewise(t, [t < 15, (t >= 15) & (t < 30), (t >= 30) & (t < 45), t >= 45],
                         [2.0, 5.0, 2.0, 4.0])
    # heading: hold 0, turn to +90 deg over t in [10, 14], hold, turn to -30 deg over [35, 40]
    psi_sp = np.zeros_like(t)
    psi_sp[(t >= 10) & (t < 14)] = np.deg2rad(90) * (t[(t >= 10) & (t < 14)] - 10) / 4
    psi_sp[(t >= 14) & (t < 35)] = np.deg2rad(90)
    m2 = (t >= 35) & (t < 40)
    psi_sp[m2] = np.deg2rad(90) + (np.deg2rad(-120)) * (t[m2] - 35) / 5
    psi_sp[t >= 40] = np.deg2rad(-30)
    z_sp = 10.0

    # --- controller gains ----------------------------------------------------
    kp_psi = 2.0
    kp_v, kp_th, kd_th = 0.15, 8.0, 0.0
    kp_vy, kp_ph = 0.15, 8.0
    kp_z, kd_z = 1.0, 2.0
    theta_max = np.pi / 4

    # --- initial state -------------------------------------------------------
    x0 = np.zeros(sim.n)
    x0[STATE_NAMES.index('z')] = z_sp
    x0[STATE_NAMES.index('v_x')] = 2.0
    x0[STATE_NAMES.index('w')] = 1.2
    x0[STATE_NAMES.index('zeta')] = np.pi / 4

    # --- closed-loop rollout (records open-loop u) ---------------------------
    X = np.zeros((N, sim.n))
    U = np.zeros((N, sim.m))
    X[0] = x0
    for k in range(N):
        xk = X[k]
        z, v_x, v_y, v_z = xk[2], xk[3], xk[4], xk[5]
        phi, theta, psi = xk[6], xk[7], xk[8]

        # altitude loop -> thrust
        # note: v_z_dot = -u_thrust*cos(phi)*cos(theta) + g, so thrust *reduces* v_z
        acc_z_cmd = kp_z * (z_sp - z) - kd_z * v_z
        u_thrust = (GRAVITY - acc_z_cmd) / max(np.cos(phi) * np.cos(theta), 0.5)

        # forward-speed loop -> pitch setpoint -> pitch rate
        theta_sp = np.clip(kp_v * (vx_sp[k] - v_x), -theta_max, theta_max)
        u_theta = kp_th * (theta_sp - theta) - kd_th * 0.0

        # lateral-velocity loop -> roll setpoint -> roll rate
        phi_sp = np.clip(kp_vy * v_y, -theta_max, theta_max)
        u_phi = kp_ph * (phi_sp - phi)

        # heading loop -> yaw rate
        u_psi = kp_psi * (psi_sp[k] - psi)

        U[k] = [u_thrust, u_phi, u_theta, u_psi, 0.0, 0.0]
        if k < N - 1:
            X[k + 1] = sim._rk4_step(xk, U[k])

    return t, X, U, sim
