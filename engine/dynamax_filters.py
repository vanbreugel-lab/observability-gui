"""
dynamax (JAX) EKF / UKF backend — an independent second implementation of the
two filters ObservabilityEngine.run_ekf / run_ukf provide in NumPy.

Why have both: the min-error-variance bound is compared against what an
estimator actually achieves, so it matters that the estimator is right. Running
the *same estimation problem* through a widely-used third-party filter
(probml/dynamax, `dynamax.nonlinear_gaussian_ssm`) and getting the same
trajectory is real evidence that neither implementation is fooling us.

"Same estimation problem" is meant literally. Everything upstream of the filter
recursion comes from the engine, unchanged:

    Y, x̂₀, P₀, R, U  ←  engine._estimation_problem(...)   (same seed → same draw)
    sensor subset     ←  the observability selection
    Q per step        ←  the same 'uncorrelated' / 'vanloan' stack

so a run differs from the engine's only in *how* the recursion is carried out.

Angular measurements ARE wrapped, even though dynamax has no notion of a
circular channel — see `_wrap_to`. dynamax computes the innovation as the plain
difference y − h(x̂), so the wrapping is folded into h itself: each angular
channel of h is shifted by whole turns until it lands within ±π of that step's
measurement, which makes the plain difference exactly the wrapped one. The
measurement is handed to h through the input vector (dynamax passes u_t to h at
step t, so the augmented input [u_t, y_t] is available), and jnp.round has a
zero derivative, so C = ∂h/∂x is unchanged by the shift. In the UKF the same
shift puts every sigma point's measurement in one branch around y_k, which is
what the engine's circular sigma-point means achieve.

One deliberate difference remains: **the EKF's Jacobians are exact (JAX
autodiff)** rather than the reference implementation's finite differences, and
dynamax linearizes the prediction about the *filtered* mean where the engine's
default (`f_jac_at='post'`) linearizes about the propagated one.

Requires ``pip install dynamax`` (which pulls in jax). The import is guarded:
``AVAILABLE`` is False and ``IMPORT_ERROR`` holds the reason when it is missing,
so the app can offer the option and explain itself rather than crash.
"""
import types

import numpy
import numpy as np

try:
    import jax
    # float64 everywhere: JAX defaults to float32, which would put a
    # single-precision filter next to a double-precision one and make every
    # comparison against the NumPy engine meaningless. Must be set before any
    # array is created.
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from dynamax.nonlinear_gaussian_ssm import (ParamsNLGSSM, UKFHyperParams,
                                                extended_kalman_filter,
                                                unscented_kalman_filter)
    AVAILABLE, IMPORT_ERROR = True, None
except Exception as _exc:                       # ImportError, or a jax/CUDA fault
    AVAILABLE, IMPORT_ERROR = False, _exc
    jax = jnp = None


def _require():
    if not AVAILABLE:
        raise ImportError(
            'the dynamax filter backend needs jax + dynamax: '
            f'pip install dynamax  ({type(IMPORT_ERROR).__name__}: {IMPORT_ERROR})')


# ───────────────────── making a NumPy model JAX-traceable ────────────────────
# dynamax traces f and h under jit/lax.scan, and its EKF differentiates them.
# The system models here are plain NumPy (`np.cos`, `np.sqrt`, …), which a JAX
# tracer cannot flow through. Two ways out, tried in that order:
#
#   'jax'      rebuild the function with its module's numpy alias rebound to
#              jax.numpy. jax.numpy is a drop-in for the ufunc-level numpy these
#              models use, so f/h become genuinely traceable and differentiable —
#              fast, and the EKF gets exact Jacobians.
#   'callback' leave the function in NumPy and call it from JAX through
#              jax.pure_callback, supplying a central-difference JVP so the EKF
#              can still linearize. Works for *any* model (an uploaded custom
#              system may use scipy, `math`, control flow on values …) at the
#              cost of one host call per evaluation.
#
# The rebind is done on a COPY of the globals dict, never on the module itself:
# patching `observability_gui.np` in place would be visible to every other
# thread for the duration of a trace.

def _jaxified(fn, _memo=None):
    """ `fn` with every numpy alias in its globals rebound to jax.numpy.

    Recurses into same-module helper functions so a model split across a few
    functions still converts. Returns `fn` unchanged if it has no globals (a
    builtin or a C callable). """
    _memo = {} if _memo is None else _memo
    if id(fn) in _memo:
        # `or fn`: a None entry is an in-progress conversion (a cycle), and the
        # original is a safer stand-in than a hole — it will simply fail to
        # trace, which the caller already handles.
        return _memo[id(fn)] or fn
    g_old = getattr(fn, '__globals__', None)
    if g_old is None:
        return fn
    g_new = dict(g_old)
    _memo[id(fn)] = None                       # placeholder: breaks f→g→f cycles
    for k, v in g_old.items():
        if v is numpy:
            g_new[k] = jnp
        elif (isinstance(v, types.FunctionType) and v is not fn
                and getattr(v, '__globals__', None) is g_old):
            g_new[k] = _jaxified(v, _memo)     # helper in the same module
    out = types.FunctionType(fn.__code__, g_new, fn.__name__, fn.__defaults__,
                             fn.__closure__)
    out.__kwdefaults__ = fn.__kwdefaults__
    _memo[id(fn)] = out
    return out


def _as_vec(v):
    """ A model's return value as a 1-D JAX array. f/h may return a list of
    scalars (fly, alt2d) or an array (drone); both must come back as one flat
    vector, since dynamax multiplies the result by a matrix. """
    if isinstance(v, (list, tuple)):
        return jnp.concatenate([jnp.atleast_1d(jnp.asarray(e, dtype=float))
                                .ravel() for e in v])
    return jnp.asarray(v, dtype=float).ravel()


def _callback(pyfn, out_dim, fd_eps=1e-6):
    """ A NumPy `pyfn(x, u)` as a JAX-callable with a finite-difference JVP.

    The JVP is a central difference along the requested tangent direction, the
    same scheme (and the same absolute 1e-6 step) the engine's EKF uses for its
    Jacobians — so in this mode the two EKFs differentiate identically and only
    the recursion differs. `vmap_method='sequential'` is what lets the UKF's
    vmap over sigma points call back into Python. """
    def _host(x, u):
        return np.asarray(pyfn(np.asarray(x, dtype=float),
                               np.asarray(u, dtype=float)), dtype=float).ravel()

    @jax.custom_jvp
    def g(x, u):
        return jax.pure_callback(
            _host, jax.ShapeDtypeStruct((out_dim,), jnp.float64), x, u,
            vmap_method='sequential')

    @g.defjvp
    def _g_jvp(primals, tangents):
        x, u = primals
        dx, du = tangents
        # normalize the direction so the step is fd_eps regardless of how the
        # tangent is scaled; a zero tangent gives step→∞ but x ± step·0 = x, so
        # the difference is exactly 0 and the ratio stays 0.
        nrm = jnp.sqrt(jnp.sum(dx ** 2) + jnp.sum(du ** 2))
        step = fd_eps / jnp.maximum(nrm, 1e-300)
        plus = g(x + step * dx, u + step * du)
        minus = g(x - step * dx, u - step * du)
        return g(x, u), (plus - minus) / (2.0 * step)

    return g


def _model_fns(engine, sensors, mode='auto'):
    """ (f_step, h_sel, sidx, mode) — the RK4 step and the sensor-restricted
    measurement function as JAX callables, plus which mechanism was used.

    `mode='auto'` tries the jax.numpy rebind and verifies it against the
    engine's own NumPy RK4 step on the current trajectory; anything that fails
    to trace, or that traces to a different answer, falls back to the callback
    path rather than silently filtering a different model. """
    s = engine.spec
    n, dt = len(s.state_names), float(s.dt)
    names = list(s.measurement_names)
    sel = set(sensors) if sensors else set(names)
    sidx = np.array([i for i, m in enumerate(names) if m in sel], dtype=int)
    p_full = len(names)
    jidx = jnp.asarray(sidx)
    # the NumPy twin the JAX h is checked against
    h_ref = engine._sensor_subset(sensors, np.zeros((1, p_full)),
                                  np.eye(p_full))[0]
    p_out = len(sidx)

    def _rk4(f):
        def step(x, u):
            k1 = _as_vec(f(x, u))
            k2 = _as_vec(f(x + 0.5 * dt * k1, u))
            k3 = _as_vec(f(x + 0.5 * dt * k2, u))
            k4 = _as_vec(f(x + dt * k3, u))
            return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return step

    if mode in ('auto', 'jax'):
        try:
            f_step = _rk4(_jaxified(s.f))
            h_j = _jaxified(s.h)
            h_sel = lambda x, u: _as_vec(h_j(x, u))[jidx]
            # verify against the NumPy model at two points on the trajectory,
            # not one: a rebind that happens to agree at k=0 but not further
            # along (a branch taken on a value, a stray np.unwrap) would
            # otherwise pass. jit here so the check runs through the same trace
            # the filters will.
            f_jit, h_jit = jax.jit(f_step), jax.jit(h_sel)
            np_step = engine._rk4_sim()._rk4_step
            ok = True
            for k in {0, engine.N // 2, engine.N - 1}:
                xk = np.asarray(engine.X[k], dtype=float)
                uk = np.asarray(engine.U[k], dtype=float)
                got = np.asarray(f_jit(jnp.asarray(xk), jnp.asarray(uk)),
                                 dtype=float)
                gh = np.asarray(h_jit(jnp.asarray(xk), jnp.asarray(uk)),
                                dtype=float)
                wh = np.asarray(h_ref(xk, uk), dtype=float)
                ok = ok and (got.shape == (n,) and gh.shape == (p_out,)
                             and np.allclose(got, np_step(xk, uk),
                                             rtol=1e-9, atol=1e-12)
                             and np.allclose(gh, wh, rtol=1e-9, atol=1e-12))
            if ok:
                return f_step, h_sel, sidx, 'jax'
            if mode == 'jax':
                raise ValueError('the jax.numpy rebuild of f/h does not '
                                 'reproduce the NumPy model')
        except Exception:
            if mode == 'jax':
                raise

    # callback path: the model stays in NumPy
    f_np = engine._rk4_sim()._rk4_step
    h_np = lambda x, u: np.asarray(s.h(x, u), dtype=float)
    f_step = _callback(f_np, n)
    h_full = _callback(h_np, p_full)
    h_sel = lambda x, u: h_full(x, u)[jidx]
    return f_step, h_sel, sidx, 'callback'


def _wrap_to(y, raw, mask):
    """ `raw` with each masked (angular) channel shifted by whole turns so it
    lands within ±π of the corresponding entry of `y`.

    This is what gives dynamax wrapped innovations without touching dynamax:
    it computes y − h(x̂) as a plain difference, and after this shift that plain
    difference IS the difference wrapped into (−π, π]. `mask` is 1.0 on angular
    channels and 0.0 elsewhere, so the non-angular ones pass through untouched.

    jnp.round carries a zero derivative, so the shift contributes nothing to
    ∂h/∂x — the EKF's C is exactly the unshifted model's. """
    return raw + mask * (2.0 * np.pi) * jnp.round((y - raw) / (2.0 * np.pi))


def _problem(engine, q_diag, r_diag, seed, u_noise_var, u_bias, p0_diag,
             x0_guess, sensors, q_noise, mode):
    """ One estimation problem, assembled exactly as the engine's own filters
    assemble it, with the model functions handed over to JAX. """
    _require()
    s = engine.spec
    Y, xh0, P0, R, U = engine._estimation_problem(
        r_diag, seed, u_noise_var=u_noise_var, u_bias=u_bias,
        p0_diag=p0_diag, x0_guess=x0_guess)
    # _sensor_subset validates the selection, slices Y and R, and reports which
    # channels are angular WITHIN the subset; its own h_sel is NumPy, so the JAX
    # one is rebuilt from the same indices in _model_fns.
    _h_np, Y, R, ang = engine._sensor_subset(sensors, Y, R)
    f_core, h_core, _sidx, used = _model_fns(engine, sensors, mode)
    n_u = U.shape[1]

    if ang:      # wrap the angular channels; see _wrap_to
        # Augment the input with that step's measurement so h can wrap against
        # it (see _wrap_to). dynamax hands u_t to both f and h at step t, so
        # this is the one channel available for it; f simply ignores the tail.
        mask = np.zeros(Y.shape[1])
        mask[list(ang)] = 1.0
        jmask = jnp.asarray(mask)
        f_step = lambda x, uy: f_core(x, uy[:n_u])
        h_sel = lambda x, uy: _wrap_to(uy[n_u:], h_core(x, uy[:n_u]), jmask)
        inputs = np.concatenate([U, Y], axis=1)
    else:
        f_step, h_sel, inputs = f_core, h_core, U

    # per-step Q, same two conventions the engine's filters offer: the entered
    # diagonal repeated, or Van Loan's correlated stack from the local dynamics.
    if q_noise == 'vanloan':
        Qds = np.asarray(engine._lin(q_diag, q_noise)[2], dtype=float)
    else:
        Qds = np.diag([q_diag[nm] for nm in s.state_names])
    return dict(Y=jnp.asarray(Y), U=jnp.asarray(inputs), Q=jnp.asarray(Qds),
                R=jnp.asarray(R), x0=jnp.asarray(xh0), P0=jnp.asarray(P0),
                f=f_step, h=h_sel, mode=used, wrapped=bool(ang))


def _unpack(post, ang_s=()):
    """ dynamax posterior → the engine's (X_hat, P_diag) contract: post-update
    means and the diagonal of the post-update covariance, one row per step.

    dynamax has no notion of a circular state, so it lets an angular state drift
    in whatever 2π branch it started in. `ang_s` names the angular-state columns
    (spec.angle_states); they are canonicalized back into (−π, π] here with the
    SAME wrap the engine's filters apply after each update, so the reported
    estimate lands in the engine's branch instead of one an integer number of
    turns away. This is representation only — variance is branch-invariant, so P
    is untouched — and it is the one lever dynamax exposes: its UKF takes plain
    sigma-point means with no hook for a circular mean (see `limitations`). """
    X_hat = np.array(post.filtered_means, dtype=float)   # copy: JAX out is read-only
    if len(ang_s):
        a = list(ang_s)
        X_hat[:, a] = (X_hat[:, a] + np.pi) % (2.0 * np.pi) - np.pi
    P = np.asarray(post.filtered_covariances, dtype=float)
    return X_hat, np.diagonal(P, axis1=1, axis2=2).copy()


def _angular_state_idx(spec):
    """ Column indices of spec.angle_states within the state vector. """
    return [i for i, nm in enumerate(spec.state_names)
            if nm in getattr(spec, 'angle_states', ())]


def run_ekf(engine, q_diag, r_diag, seed=0, u_noise_var=0.0, u_bias=None,
            p0_diag=None, x0_guess=None, sensors=None,
            q_noise='uncorrelated', num_iter=1, mode='auto',
            **_ignored):
    """ dynamax's extended Kalman filter on the engine's estimation problem.
    Returns (X_hat, P_diag), post-update, matching ObservabilityEngine.run_ekf.

    Angular innovations are wrapped by construction (see
    `_wrap_to`); pass False to see what an unwrapped filter does to a model with
    arctan2 channels. What remains different from the reference EKF: the
    Jacobians are exact (autodiff) instead of finite differences unless the
    model had to fall back to the callback path, and dynamax linearizes the
    prediction about the filtered mean rather than the propagated one.
    `num_iter` > 1 re-linearizes the update about the posterior (dynamax's
    iterated EKF); 1 is the plain EKF.

    Extra keyword arguments the NumPy runners accept (full_cov, x0_scale, …) are
    ignored rather than rejected, so the two backends stay call-compatible. """
    P = _problem(engine, q_diag, r_diag, seed, u_noise_var, u_bias, p0_diag,
                 x0_guess, sensors, q_noise, mode)
    params = ParamsNLGSSM(
        initial_mean=P['x0'], initial_covariance=P['P0'],
        dynamics_function=P['f'], dynamics_covariance=P['Q'],
        emission_function=P['h'], emission_covariance=P['R'])
    post = extended_kalman_filter(params, P['Y'], inputs=P['U'],
                                  num_iter=max(1, int(num_iter)))
    return _unpack(post, _angular_state_idx(engine.spec))


def run_ukf(engine, q_diag, r_diag, seed=0, alpha=1.0, beta=2.0, kappa=0.0,
            u_noise_var=0.0, u_bias=None, p0_diag=None, x0_guess=None,
            sensors=None, q_noise='uncorrelated', mode='auto',
            **_ignored):
    """ dynamax's unscented Kalman filter on the same problem. Returns
    (X_hat, P_diag), matching ObservabilityEngine.run_ukf.

    α, β, κ mean what they mean in the engine's UKF: sigma points at
    ±√(α²(n+κ))·σ, β the prior weight — dynamax builds the identical scaled
    unscented transform.

    dynamax takes plain (non-circular) means and residuals, so the shift in
    (see `_wrap_to`) puts every sigma point's angular measurement in one branch
    around y_k — the same job the engine's circular sigma-point means do.

    One limit remains: dynamax reads the state dimension off
    ``dynamics_covariance.shape[0]``, so a per-step Q stack cannot be passed —
    with q_noise='vanloan' the step-0 covariance is used for every step, and the
    caller is told so via the returned note (see core.compute_payload). """
    P = _problem(engine, q_diag, r_diag, seed, u_noise_var, u_bias, p0_diag,
                 x0_guess, sensors, q_noise, mode)
    Q = P['Q']
    if Q.ndim == 3:                 # see the docstring: constant Q only
        Q = Q[0]
    params = ParamsNLGSSM(
        initial_mean=P['x0'], initial_covariance=P['P0'],
        dynamics_function=P['f'], dynamics_covariance=Q,
        emission_function=P['h'], emission_covariance=P['R'])
    hp = UKFHyperParams(alpha=float(alpha), beta=float(beta),
                        kappa=float(kappa))
    post = unscented_kalman_filter(params, P['Y'], hp, inputs=P['U'])
    return _unpack(post, _angular_state_idx(engine.spec))


def limitations(spec, q_noise='uncorrelated'):
    """ The caveats that apply to running THIS system through dynamax, as short
    phrases for the status line. Empty when there are none — which is the usual
    case now that angular channels are wrapped. """
    out = []
    if getattr(spec, 'angle_states', ()):
        out.append('angular states (' + ', '.join(spec.angle_states) +
                   ') are wrapped to (−π, π] after filtering, not by a circular '
                   'unscented mean — matches the engine unless a state’s spread '
                   'straddles ±π')
    if q_noise == 'vanloan':
        out.append("dynamax's UKF takes a constant Q, so the Van Loan stack's "
                   'k=0 covariance is used at every step')
    return out
