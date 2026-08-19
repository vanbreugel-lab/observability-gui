"""
Stochastic observability Gramian for nonlinear systems via trajectory
linearization, following:

    B. Boyacioglu & F. van Breugel, "Duality of Stochastic Observability and
    Constructability and their Relation to the Fisher Information"
    (IEEE L-CSS / bioRxiv), Lemma 1 / Eq. (33).

Idea
----
Given a nonlinear system with process noise w and measurement noise v

    x_dot = f(x, u) + w(t),      w(t) white noise, PSD Q_c
    y_k   = h(x_k, u_k) + v_k,    v_k ~ N(0, R_k)

(continuous-time process noise: E[w(t) w(s)^T] = Q_c * delta(t - s); it is a
spectral density, not a per-step covariance. Discretizing over one step dt via
Q is the PER-STEP discrete covariance Qd, used verbatim — these recursions are
discrete, so no PSD and no dt scaling enters the Q path.) Linearize about a nominal trajectory {x_k, u_k} at every step:

    A_k = df/dx |_(x_k, u_k),     C_k = dh/dx |_(x_k, u_k)

then discretize (Van Loan) to obtain the DT-LTV system of the paper's Eq. (22):

    dx_{k+1} = Phi_k dx_k + w_k,   w_k ~ N(0, Qd_k),   dy_k = C_k dx_k + v_k

The w-step stochastic observability Gramian F^{x_0}_{down,w} — the Fisher
information of the window's measurements w.r.t. the window's initial state,
whose inverse Cramer-Rao-bounds the error covariance of any (unbiased)
fixed-point smoother of x_0 — is computed with the numerically stable backward
recursion obtained from duality (Lemma 1, Eq. (33)).  Written in original
(non-dual) indices, stepping backward from the end of the window, the
recursion is

    F_j = Phi_j^T [ Qd_j^{-1} - Qd_j^{-1} (F_{j+1} + Qd_j^{-1})^{-1} Qd_j^{-1} ] Phi_j
          + C_j^T R_j^{-1} C_j                                            (*)

initialized with F_{w-1} = C_{w-1}^T R_{w-1}^{-1} C_{w-1}, where Phi_j maps
x_j -> x_{j+1} and Qd_j is the process noise injected on that step.  F_0 is
the w-step stochastic observability Gramian.  By the Sherman-Morrison-Woodbury
identity the bracket equals (Qd_j + F_{j+1}^{-1})^{-1}, so (*) is exactly the
information-form backward pass of a forward-backward smoother; as Qd -> 0 it
reduces to the deterministic recursion F_j = Phi_j^T F_{j+1} Phi_j + C_j^T
R_j^{-1} C_j, i.e. F_0 -> O^T blkdiag(R)^{-1} O — the Fisher information that
pybounds computes from the empirical observability matrix.  This is the
consistency check exploited in the accompanying notebook.
"""

import numpy as np
import pandas as pd
from scipy.linalg import expm


# ----------------------------------------------------------------------------
# Jacobians (central finite differences; matches pybounds' empirical approach)
# ----------------------------------------------------------------------------
def fd_jacobian(func, x, u, eps=1e-6):
    """ Central-difference Jacobian of func(x, u) w.r.t. x. Returns (dim_out, n). """
    x = np.asarray(x, dtype=float)
    n = x.size
    f0 = np.asarray(func(x, u), dtype=float)
    J = np.zeros((f0.size, n))
    for i in range(n):
        dx = np.zeros(n)
        step = eps * max(1.0, abs(x[i]))
        dx[i] = step
        J[:, i] = (np.asarray(func(x + dx, u)) - np.asarray(func(x - dx, u))) / (2 * step)
    return J


# ----------------------------------------------------------------------------
# Discretization
# ----------------------------------------------------------------------------
def transition(A, dt, method='vanloan'):
    """ One-step state-transition matrix Phi over dt.

    'vanloan' : Phi = expm(A dt) — exact for LTI over the step; the standard
                choice, and Phi is guaranteed invertible as the duality
                construction requires.
    'euler'   : Phi = I + A dt (cheaper; may be poorly conditioned /
                non-invertible for stiff A and large dt).

    NOTE this used to also return a process-noise matrix, built by Van Loan from
    a continuous PSD Q_c as Q_d = ∫₀^dt e^{Aτ} Q_c e^{Aᵀτ} dτ. That is gone: the
    recursions this module implements (Lemma 1 / Eq. 30 / Eq. 33) are DISCRETE, so
    they take the per-step covariance Q_d directly and no dt conversion belongs
    anywhere in the Q path. Q_d is now supplied by the caller and used verbatim.
    """
    if method == 'vanloan':
        return expm(A * dt)
    if method == 'euler':
        return np.eye(A.shape[0]) + A * dt
    raise ValueError("method must be 'vanloan' or 'euler'")


def _rk4_variational_step(f, x, u, dt, fd_eps):
    """ One RK4 step of the joint (state, variational) system:
        x_dot = f(x, u),   Phi_dot = A(x(t), u) Phi,   Phi(0) = I.
    Returns Phi over the step, consistent with the RK4 flow of the nonlinear
    system (captures the within-step variation of A that a frozen-Jacobian
    matrix exponential misses). """
    n = np.asarray(x).size
    I_n = np.eye(n)
    fa = lambda xx, uu: np.asarray(f(xx, uu), dtype=float)
    k1x = fa(x, u);                   k1P = fd_jacobian(f, x, u, fd_eps) @ I_n
    x2 = x + 0.5 * dt * k1x
    k2x = fa(x2, u);                  k2P = fd_jacobian(f, x2, u, fd_eps) @ (I_n + 0.5 * dt * k1P)
    x3 = x + 0.5 * dt * k2x
    k3x = fa(x3, u);                  k3P = fd_jacobian(f, x3, u, fd_eps) @ (I_n + 0.5 * dt * k2P)
    x4 = x + dt * k3x
    k4x = fa(x4, u);                  k4P = fd_jacobian(f, x4, u, fd_eps) @ (I_n + dt * k3P)
    Phi = I_n + (dt / 6.0) * (k1P + 2 * k2P + 2 * k3P + k4P)
    return Phi



def van_loan_process_noise(A, Qc, dt):
    """ Correlated per-step process noise from a continuous spectral density:

        Q_k = integral_0^dt  e^{A tau} Qc e^{A^T tau}  dtau        (Van Loan 1978)

    computed from the matrix exponential of the block matrix
    [[-A, Qc], [0, A^T]] * dt.  Unlike a diagonal Qc * dt this is generally NOT
    diagonal: over one step the dynamics mix the states, so noise injected on one
    state leaks into every state it drives (phi_dot into phi, velocities into
    position), and Q_k inherits the trajectory's time-variation through A_k.

    States with no dynamics of their own (zeta_dot = 0, w_dot = 0) are unaffected:
    their diagonal entry is exactly Qc[i,i] * dt and their cross terms stay 0.

    NOTE this is a different use of "Van Loan" from ``transition(method=...)``,
    which builds Phi only.
    """
    n = A.shape[0]
    M = dt * np.block([[-A, Qc], [np.zeros((n, n)), A.T]])
    E = expm(M)
    Phi = E[n:, n:].T
    Qd = Phi @ E[:n, n:]
    return 0.5 * (Qd + Qd.T)


def linearize_along_trajectory(f, h, x_traj, u_traj, dt, Qd, method='vanloan', eps=1e-6,
                               q_noise='uncorrelated'):
    """ Linearize a nonlinear system at every point of a trajectory and build the
    per-step transition matrices.

    :param Qd: the PER-STEP process-noise covariance [units²] — the Q of the
        discrete recursions, used VERBATIM at every step. It is NOT derived from a
        continuous spectral density and is never scaled by dt; see transition().
    :param q_noise: how the per-step process covariance is built from `Qd`:
        'uncorrelated' : Q_k = Qd at every step — diagonal in, diagonal out, and
                         identical for all k. The default, and what every result
                         predating 2026-08-18 used.
        'vanloan'      : Q_k = integral_0^dt e^{A_k tau} (Qd/dt) e^{A_k^T tau} dtau,
                         i.e. treat the entered Qd as a spectral density Qd/dt and
                         discretise it through the LOCAL dynamics. Gives correlated
                         (non-diagonal) and genuinely time-varying Q_k. States with
                         no dynamics keep their entered value exactly.
    :param method: how to build the per-step transition matrix Phi_k:
        'vanloan' : Phi = expm(A_k dt) with A_k frozen at the left endpoint.
                    O(dt^2) local error when A(t) varies within a step.
        'magnus2' : Phi = expm(0.5 (A_k + A_{k+1}) dt) — 2nd-order Magnus with
                    trapezoidal A. Cheap fix that captures most within-step
                    variation of A (important for fast maneuvers).
        'rk4var'  : integrate the variational equation Phi_dot = A(x(t)) Phi
                    alongside the state with RK4 — matched-flow Jacobian,
                    consistent with an RK4-simulated empirical O.
        'euler'   : Phi = I + A_k dt.
    :returns: (Phis, Cs, Qds) lists of length N (Phis[k]: x_k -> x_{k+1});
        every entry of Qds is the same supplied Qd.
    """
    N = x_traj.shape[0]
    As = [fd_jacobian(f, x_traj[k], u_traj[k], eps=eps) for k in range(N)]
    Cs = [fd_jacobian(h, x_traj[k], u_traj[k], eps=eps) for k in range(N)]
    Qd = np.asarray(Qd, dtype=float)
    Phis = []
    for k in range(N):
        if method == 'magnus2':
            A_eff = 0.5 * (As[k] + As[min(k + 1, N - 1)])
            Phi = transition(A_eff, dt, method='vanloan')
        elif method == 'rk4var':
            Phi = _rk4_variational_step(f, x_traj[k], u_traj[k], dt, eps)
        else:
            Phi = transition(As[k], dt, method=method)
        Phis.append(Phi)
    if q_noise == 'vanloan':
        Qc = Qd / dt
        Qds = [van_loan_process_noise(As[k], Qc, dt) for k in range(N)]
    elif q_noise == 'uncorrelated':
        Qds = [Qd] * N
    else:
        raise ValueError("q_noise must be 'uncorrelated' or 'vanloan'")
    return Phis, Cs, Qds


# ----------------------------------------------------------------------------
# Recursive stochastic observability Gramian  (paper Lemma 1 / Eq. 33)
# ----------------------------------------------------------------------------
def stochastic_observability_gramian(Phis, Cs, Qds, Rinvs, return_sequence=False):
    """ w-step stochastic observability Gramian F^{x_0}_{down,w} via the
    backward recursion (*) in the module docstring.

    :param Phis:  list of w-1 (or w; last unused) state-transition matrices,
                  Phis[j] maps x_j -> x_{j+1} within the window
    :param Cs:    list of w measurement Jacobians C_j
    :param Qds:   list of w-1 (or w; last unused) discrete process covariances
    :param Rinvs: list of w inverse measurement covariances R_j^{-1}
    :param return_sequence: if True also return [F^{x_j}_{down, w-j}] for all j
    """
    w = len(Cs)
    F = Cs[-1].T @ Rinvs[-1] @ Cs[-1]
    seq = [F]
    for j in range(w - 2, -1, -1):
        Phi, Qd, C, Rinv = Phis[j], Qds[j], Cs[j], Rinvs[j]
        Qinv = np.linalg.inv(Qd)
        # bracket = Qinv - Qinv (F + Qinv)^{-1} Qinv  ==  (Qd + F^{-1})^{-1}  (SMW)
        S = np.linalg.solve(F + Qinv, Qinv)          # (F + Qinv)^{-1} Qinv
        bracket = Qinv - Qinv @ S
        F = Phi.T @ bracket @ Phi + C.T @ Rinv @ C
        F = 0.5 * (F + F.T)
        seq.append(F)
    if return_sequence:
        return F, seq[::-1]
    return F


def stochastic_constructability_gramian(Phis, Cs, Qds, Rinvs):
    """ w-step stochastic constructability Gramian F^{x_{w-1}} — the Fisher
    information of the window's *final* state (vs. observability, which bounds
    the *initial* state). Its inverse is the a-posteriori Cramer-Rao bound for a
    filter estimating the current state, so it lines up with a Kalman filter's
    error covariance far better than the observability Gramian.

    Forward recursion (Boyacioglu & van Breugel, Eq. 30 — the dual of the
    observability Eq. 33). With Phi = Phi_{k+1,k} and process noise Q_k on the
    step, each update is
        F_{k+1} = (Q_k + Phi F_k^{-1} Phi^T)^{-1} + C_{k+1}^T R_{k+1}^{-1} C_{k+1},
    written in the Sherman-Morrison-Woodbury form that avoids F^{-1}:
        (Q + Phi F^{-1} Phi^T)^{-1} = Qinv - Qinv Phi (F + Phi^T Qinv Phi)^{-1} Phi^T Qinv.

    :param Phis:  Phis[k] maps x_k -> x_{k+1} within the window
    :param Cs:    measurement Jacobians C_k
    :param Qds:   discrete per-step process covariances Q_k
    :param Rinvs: inverse measurement covariances R_k^{-1}
    """
    w = len(Cs)
    F = Cs[0].T @ Rinvs[0] @ Cs[0]          # F^{x_0}
    for k in range(w - 1):
        Phi, Qd = Phis[k], Qds[k]
        Qinv = np.linalg.inv(Qd)
        M = Phi.T @ Qinv @ Phi
        X = np.linalg.solve(F + M, Phi.T @ Qinv)     # (F + M)^{-1} Phi^T Qinv
        bracket = Qinv - Qinv @ Phi @ X              # (Q + Phi F^{-1} Phi^T)^{-1}
        F = bracket + Cs[k + 1].T @ Rinvs[k + 1] @ Cs[k + 1]
        F = 0.5 * (F + F.T)
    return F                                 # F^{x_{w-1}} (final-state FIM)


def deterministic_observability_gramian(Phis, Cs, Rinvs):
    """ Q -> 0 limit: F = sum_j Phi_{j,0}^T C_j^T R_j^{-1} C_j Phi_{j,0}.
    Equals the Fisher information pybounds builds from the empirical O
    (up to linearization/discretization error). Provided for validation. """
    w = len(Cs)
    n = Cs[0].shape[1]
    F = np.zeros((n, n))
    Phi_j0 = np.eye(n)
    for j in range(w):
        CP = Cs[j] @ Phi_j0
        F += CP.T @ Rinvs[j] @ CP
        if j < w - 1:
            Phi_j0 = Phis[j] @ Phi_j0
    return F


# ----------------------------------------------------------------------------
# Sliding-window wrapper mirroring pybounds' SlidingFisherObservability output
# ----------------------------------------------------------------------------
class SlidingStochasticObservability:
    """ Compute the stochastic observability Gramian (and the min error
    variance diag(F^{-1})) in sliding windows along a nonlinear trajectory,
    by linearizing at each time step.

    Produces an EV_aligned DataFrame with the same layout as
    pybounds.SlidingFisherObservability (time, time_initial, one column per
    state) so results can be compared directly.
    """

    def __init__(self, f, h, t_sim, x_sim, u_sim, w, Qd, R_diag,
                 state_names, measurement_names, sensors=None, states=None,
                 lam=1e-6, disc_method='vanloan', fd_eps=1e-6):
        """
        :param f, h: continuous dynamics and measurement functions f(x,u), h(x,u)
        :param disc_method: 'vanloan' | 'magnus2' | 'rk4var' | 'euler'
            (see linearize_along_trajectory; prefer 'magnus2' or 'rk4var' when
            the linearization changes quickly within one time step, e.g. fast
            maneuvers relative to dt)
        :param t_sim, x_sim, u_sim: nominal trajectory (N,), (N,n), (N,m)
        :param w: window size in time steps
        :param Qd: PER-STEP discrete process-noise covariance (n,n), must be > 0.
            Used verbatim in the recursion — never divided or multiplied by dt.
        :param R_diag: dict {sensor_name: variance} (per-step measurement noise)
        :param sensors: subset of measurement names to use (default: all)
        :param states: subset of state names for which to report error variance.
            To match pybounds' FisherObservability semantics, the error variance
            is diag((F[states, states] + lam I)^{-1}) — the *conditional* FIM
            submatrix (remaining initial states treated as known), which is
            exactly what pybounds computes when it column-slices O before
            forming O^T R^{-1} O.
        :param lam: Chernoff regularizer used when inverting F (matches pybounds)
        """
        self.t = np.asarray(t_sim)
        if isinstance(x_sim, dict):
            x_sim = np.vstack([x_sim[k] for k in state_names]).T
        if isinstance(u_sim, dict):
            u_sim = np.vstack(list(u_sim.values())).T
        self.x = np.asarray(x_sim)
        self.u = np.asarray(u_sim)
        self.N = self.t.shape[0]
        self.w = int(w)
        self.dt = float(np.mean(np.diff(self.t)))
        self.lam = lam
        self.state_names = list(state_names)
        self.n = len(self.state_names)

        # sensor selection -> row mask into h output & per-step R
        self.measurement_names = list(measurement_names)
        self.sensors = list(sensors) if sensors is not None else list(self.measurement_names)
        sensor_idx = [self.measurement_names.index(s) for s in self.sensors]
        r_vec = np.array([R_diag[s] for s in self.sensors], dtype=float)
        self.Rinv_step = np.diag(1.0 / r_vec)
        self._sensor_idx = np.array(sensor_idx, dtype=int)

        self.states = list(states) if states is not None else list(self.state_names)

        # 1) linearize + discretize once along the entire trajectory
        self.Phis, Cs_full, self.Qds = linearize_along_trajectory(
            f, h, self.x, self.u, self.dt, Qd, method=disc_method, eps=fd_eps)
        self.Cs = [C[self._sensor_idx, :] for C in Cs_full]

        # 2) slide windows, run the backward recursion in each
        self.window_starts = np.arange(0, self.N - self.w + 1)
        self.F_sliding = []
        EV = []
        Rinvs = [self.Rinv_step] * self.w
        sub = np.array([self.state_names.index(s) for s in self.states], dtype=int)
        n_sub = len(sub)
        for k0 in self.window_starts:
            sl = slice(k0, k0 + self.w)
            F = stochastic_observability_gramian(
                self.Phis[sl], self.Cs[sl], self.Qds[sl], Rinvs)
            self.F_sliding.append(F)
            F_sub = F[np.ix_(sub, sub)]  # conditional FIM (pybounds semantics)
            # PSD projection: F is PSD in exact arithmetic, but the recursion's
            # Q^{-1} cancellations leave a numerical noise floor that can create
            # small negative eigenvalues comparable to lam. pybounds' Gram-form
            # O^T R^{-1} O is PSD by construction, so clip for a fair inverse.
            evals, evecs = np.linalg.eigh(F_sub)
            F_sub = (evecs * np.clip(evals, 0.0, None)) @ evecs.T
            F_inv = np.linalg.inv(F_sub + self.lam * np.eye(n_sub))
            EV.append(np.diag(F_inv))

        EV = np.array(EV)
        ev_df = pd.DataFrame(EV, columns=self.states)
        ev_df.insert(0, 'time_initial', self.t[self.window_starts])

        # align like pybounds: shift index forward by half the window
        self.shift_index = int(np.round(0.5 * self.w))
        ev_df.index = np.arange(self.shift_index, EV.shape[0] + self.shift_index)
        time_df = pd.DataFrame(np.atleast_2d(self.t).T, columns=['time'])
        self.EV = ev_df
        self.EV_aligned = pd.concat((time_df, ev_df), axis=1)

    def get_minimum_error_variance(self):
        return self.EV_aligned.copy()
