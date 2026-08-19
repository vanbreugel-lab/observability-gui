"""
Feed-forward ANN state estimator and the motif-informed filter (MIF, "AI-KF")
that fuses it, following

    B. Cellini, B. Boyacioglu, S. D. Stupski, F. van Breugel,
    "Active sensing motifs ..." bioRxiv 2024.11.04.621976, Figure 6 + Methods
    ("Artificial neural network", "Motif informed filter").

Verified against the authors' repository
    Discovering-and-exploiting-active-sensing-motifs-for-estimation
      util/network_model.py     (NetworkModel, custom_loss_circular)
      util/filter_custom.py     (TimeSeriesFilter -- the Eq. 5 blend)
      util/StateEstimator.m     (the AI-KF: ANN estimate as a pseudo-measurement)
      util/logarithmic_map.m, util/sliding_window_func_backward.m
      make/make_estimator_monocular_initial_sweep_simulated.m (its parameters)

ANN (Methods "Artificial neural network"; repo util/network_model.py)
--------------------------------------------------------------------
A feed-forward regressor whose inputs are the sensory measurements augmented
with an omega-step time history, and whose output is a single state (wind
direction zeta in the paper). ReLU hidden layers, linear output. For a circular
target the paper's custom loss is used,

    f(zeta, zeta_check) = (sin zeta - sin zeta_check)^2
                        + (cos zeta - cos zeta_check)^2      (Methods, unnumbered)

The repo (custom_loss_circular) applies this to a SINGLE linear output: y_pred is
the raw angle, and the loss wraps it through sin/cos. That is what is implemented
here -- its gradient collapses neatly to

    dL/dy_pred = 2 sin(y_pred - y_true)

so the network is free to leave the +-pi branch and predictions are wrapped to
[-pi, pi] afterwards (repo: util.wrapToPi). Non-circular targets use plain MSE.

Repo defaults reproduced here: ReLU hidden layers of (64, 64, 64), linear output,
a GaussianNoise layer on the inputs active during training only, adam at its Keras
default lr = 1e-3, an 80/20 train/test split plus validation_split = 0.2 inside
the fit, and the repo's feature layout from get_sequential_inputs -- columns
grouped NAME-major with time ascending inside each group
(a_0..a_{T-1}, b_0..b_{T-1}), over a BACKWARD-looking window (the repo's
time_direction = 'backward' for the altitude network).

Training data come from the system's OWN MPC over perturbed set-points -- the
repo's approach (make_altitude_trajectories_initial.ipynb sweeps the speed change
g_delta and the altitude profile, one MPC rollout per trajectory). Each rollout
yields one window per step, so a couple of dozen rollouts is plenty. The engine
builds them, since it owns the MPC: ObservabilityEngine._ann_training_data. An
earlier version synthesized randomized kinematics instead; it was endless to tune
and never matched the regime the net is actually asked to work in.

Two different fusion schemes live here, because the two papers do different
things and the repo implements the second:

MIF (bioRxiv Methods "Motif informed filter", Eq. 5-7; repo util/filter_custom.py)
----------------------------------------------------------------------------------
    x_hat_k = beta_k * x_check_k + (1 - beta_k) * x_hat_{k-1}          (Eq. 5)
    x_check_k = H(y_k..y_{k-omega}, u_k..u_{k-omega})                  (Eq. 6)
    beta_k = [ log10(10 + [F_omega^-1]_{i,i}) ]^-1                     (Eq. 7)

Eq. 7 turns the sliding-window min error variance of the target state into a
0-1 gain: where the state is well observed (small min-EV) beta -> 1 and the raw
ANN estimate is trusted; where it is unobservable (large min-EV) beta -> 0 and
the filter coasts on its previous estimate. The paper notes that for statistical
consistency beta should come from the CONSTRUCTABILITY Gramian, not the
observability one, since it is the current state being estimated -- so that is
the default here.

For a circular state the Eq. 5 blend is done on the unit circle (blend sin/cos,
then atan2). Blending the raw angles would let a +pi/-pi pair average to 0. NOTE
the repo's TimeSeriesFilter blends linearly and does not special-case angles.

AI-KF (repo util/StateEstimator.m + make_estimator_monocular_initial_sweep_*.m)
------------------------------------------------------------------------------
The repo's actual "AI-KF" is NOT the Eq. 5 complementary filter. It is an
EKF/UKF on the real dynamics whose measurement vector is augmented with the ANN's
estimate as a pseudo-measurement, and whose noise entry for that channel is
modulated by an active-sensing "motif":

    Y = [r_x_noise, z_predict]          % real sensor + ANN estimate
    motif = |u_x|                       % the motif that makes z observable
    R[ann, ann] = logarithmic_map(backward_window_mean(motif, w),
                                  [0, upper_bound], R_bounds)

`logarithmic_map` interpolates in log10 space, so a motif of 0 maps to
R_bounds[0] (1e12 -- the ANN is effectively ignored and the filter coasts on its
dynamics) and a motif at or above `upper_bound` maps to R_bounds[1] (1e-3 -- the
ANN is trusted). Setting R_bounds[1] = 1e12 too disables the ANN channel entirely
and recovers the plain UKF, which is how the repo ablates it (r_sweep = [12, -3]).
"""
import numpy as np

_RANGE_PAD = 0.6   # how far past the default set-point range to sample
# for synthesizing training inputs: input name -> the state it is the rate of
_INPUT_OF = {'u_x': 'v_x', 'u_z': 'v_z', 'u_y': 'v_y'}

__all__ = ['ANNEstimator', 'motif_informed_filter', 'mif_beta',
           'motif_R', 'logarithmic_map', 'backward_window_mean']


# ──────────────────────────── tiny MLP (numpy) ──────────────────────────────
# A dependency-free Adam-trained MLP. tensorflow/keras (what the paper used) and
# torch are both importable in this environment, but the networks here are tiny
# (the paper's best was 8+16+8 = 32 neurons) and training happens inside a Gradio
# callback, so avoiding a multi-hundred-MB framework import on the request path
# is worth more than the framework's features.

class _MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for a, b in zip(sizes[:-1], sizes[1:]):
            # He initialization, appropriate for ReLU
            self.W.append(rng.normal(0.0, np.sqrt(2.0 / a), size=(a, b)))
            self.b.append(np.zeros(b))

    def forward(self, X, cache=False):
        acts = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.maximum(z, 0.0) if i < len(self.W) - 1 else z   # linear head
            acts.append(h)
        return (h, acts) if cache else h

    def fit(self, X, Y, epochs=100, batch=256, lr=1e-3, seed=0,
            noise_std=0.0, val=None, circular=False):
        """ Adam (lr = 1e-3, the Keras default the repo relies on).

        `noise_std` adds Gaussian noise to the inputs during training only — the
        repo's GaussianNoise layer, added before the first Dense layer.

        `circular=True` uses the repo's custom_loss_circular on a single linear
        output, L = (sin y - sin yhat)^2 + (cos y - cos yhat)^2, whose gradient is
        dL/dyhat = 2 sin(yhat - y). """
        rng = np.random.default_rng(seed)
        mW = [np.zeros_like(w) for w in self.W]
        vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]
        vb = [np.zeros_like(b) for b in self.b]
        b1, b2, eps = 0.9, 0.999, 1e-8
        n, step = len(X), 0
        hist = []
        for ep in range(epochs):
            for idx in np.array_split(rng.permutation(n), max(1, n // batch)):
                xb, yb = X[idx], Y[idx]
                if noise_std > 0:
                    xb = xb + rng.normal(0.0, noise_std, size=xb.shape)
                out, acts = self.forward(xb, cache=True)
                g = (2.0 * np.sin(out - yb) if circular
                     else 2.0 * (out - yb)) / len(idx)
                step += 1
                for i in range(len(self.W) - 1, -1, -1):
                    gW = acts[i].T @ g
                    gb = g.sum(axis=0)
                    if i:
                        g = (g @ self.W[i].T) * (acts[i] > 0)
                    for p, gp, m, v in ((self.W[i], gW, mW, vW),
                                        (self.b[i], gb, mb, vb)):
                        m[i] = b1 * m[i] + (1 - b1) * gp
                        v[i] = b2 * v[i] + (1 - b2) * gp * gp
                        p -= lr * (m[i] / (1 - b1 ** step)) / \
                            (np.sqrt(v[i] / (1 - b2 ** step)) + eps)
            if val is not None and (ep + 1) % max(1, epochs // 8) == 0:
                d = self.forward(val[0]) - val[1]
                hist.append(float(np.mean(2.0 - 2.0 * np.cos(d)) if circular
                                  else np.mean(d ** 2)))
        return hist


# ─────────────────────── measurement-history features ───────────────────────

def _window_features(Y, time_steps):
    """ One feature row per step from a BACKWARD-looking window of `time_steps`
    samples (Eq. 6's Y_omega_k; repo time_direction='backward').

    Column layout follows the repo's get_sequential_inputs: NAME-major with time
    ascending inside each name — a_0..a_{T-1}, b_0..b_{T-1}, ... — not
    time-major. Rows without a full window are NaN, matching the repo, which
    trains only on steps that have the full history available. """
    Y = np.asarray(Y, dtype=float)
    N, p = Y.shape
    T = int(time_steps)
    F = np.full((N, p * T), np.nan)
    for k in range(T - 1, N):
        win = Y[k - T + 1:k + 1]                     # oldest .. newest
        F[k] = win.T.ravel()                         # name-major, time ascending
    return F


class ANNEstimator:
    """ H_i of Eq. 6: measurements (+ omega history) -> one state.

    Trained on randomized trajectories of the *same* system spec, so it is
    reusable across the app's trajectories; it never sees the trajectory being
    estimated. """

    def __init__(self, spec, target, time_steps=4, layers=(64, 64, 64),
                 sensors=None, seed=0):
        """ `time_steps` = samples in the backward window (repo convention: 4 for
        the wind networks, 20 for the altitude network). `layers` defaults to the
        repo's (64, 64, 64). """
        self.spec = spec
        self.target = target
        self.time_steps = max(1, int(time_steps))
        self.omega = self.time_steps - 1        # prior steps, the paper's omega
        self.layers = tuple(layers)
        self.sensors = list(sensors) if sensors else self._default_sensors(spec)
        # a sensor name may be a measurement OR an input: Eq. 6 admits u, and
        # the paper's altitude ANN uses optic flow together with forward
        # acceleration, which is an input of this model rather than an output
        # of h. Columns are taken from the [Y | U] stack.
        p_m = len(spec.measurement_names)
        self.sidx = [spec.measurement_names.index(m) if m in spec.measurement_names
                     else p_m + spec.input_names.index(m) for m in self.sensors]
        self.circular = target in getattr(spec, 'angle_states', ())
        self.seed = seed
        self.net = None
        self.x_mu = self.x_sd = None
        self.val_loss = None

    @staticmethod
    def _default_sensors(spec):
        """ The paper's Figure 6 ANN takes heading phi, apparent-airflow angle
        gamma and ground-velocity angle psi — deliberately NOT airspeed a or
        groundspeed g, from which the wind vector is almost directly recoverable.
        Fall back to every measurement for systems without that trio. """
        for trio in (('psi', 'beta', 'gamma'),      # repo's wind networks
                     ('phi', 'gamma', 'psi')):     # bioRxiv Figure 6 text
            if all(m in spec.measurement_names for m in trio):
                return list(trio)
        # optic-flow models: the paper's altitude ANN reads 'two-second windows
        # of optic flow and forward acceleration'
        picks = list(spec.measurement_names)
        if 'r_x' in picks and 'u_x' in spec.input_names:
            picks = picks + ['u_x']
        return picks

    def _targets(self, vals):
        # ONE output neuron either way: the repo's circular loss wraps a single
        # linear output through sin/cos rather than regressing a (sin, cos) pair.
        return np.asarray(vals, dtype=float)[:, None]

    def _decode(self, out):
        y = out[:, 0]
        return np.arctan2(np.sin(y), np.cos(y)) if self.circular else y

    def train(self, trajectories, epochs=100, noise_std=0.01, batch=256,
              seed=None):
        """ Fit on `trajectories`, a list of (YU, target_series) pairs where YU is
        (steps, n_sensors) over this estimator's sensor columns. Every window with
        a full backward history becomes one sample. 80/20 split as in the repo. """
        rng = np.random.default_rng(self.seed if seed is None else seed)
        feats, tgts = [], []
        T = self.time_steps
        for YU, tv in trajectories:
            for k in range(T - 1, len(YU)):
                feats.append(YU[k - T + 1:k + 1].T.ravel())  # name-major, ascending
                tgts.append(tv[k])
        if len(feats) < 50:
            raise RuntimeError(f'too few ANN training windows ({len(feats)})')
        Xf = np.asarray(feats, dtype=float)
        Yt = self._targets(tgts)
        ok = np.all(np.isfinite(Xf), axis=1) & np.all(np.isfinite(Yt), axis=1)
        Xf, Yt = Xf[ok], Yt[ok]
        order = rng.permutation(len(Xf))       # windows arrive trajectory-major
        Xf, Yt = Xf[order], Yt[order]
        self.x_mu, self.x_sd = Xf.mean(0), Xf.std(0) + 1e-9
        Xn = (Xf - self.x_mu) / self.x_sd
        cut = int(0.8 * len(Xn))
        self.net = _MLP((Xn.shape[1],) + self.layers + (Yt.shape[1],),
                        seed=self.seed)
        hist = self.net.fit(Xn[:cut], Yt[:cut], epochs=epochs, batch=batch,
                            seed=self.seed, noise_std=noise_std,
                            val=(Xn[cut:], Yt[cut:]), circular=self.circular)
        self.val_loss = hist[-1] if hist else None
        return self.val_loss

    def columns(self, Y, U):
        """ Stack measurements and inputs, then take this estimator's sensor
        columns — the layout both training and prediction use. """
        return np.hstack([np.asarray(Y, dtype=float),
                          np.asarray(U, dtype=float)])[:, self.sidx]

    def predict_series(self, Y, U=None):
        """ Raw estimate x_check_k for every step of a measurement series.
        `Y` is (N, p) over the spec's measurements and `U` (N, m) its inputs —
        both are needed because a sensor name may refer to either. The first
        time_steps-1 steps have no full backward window and come back NaN. """
        if U is None:
            U = np.zeros((len(Y), len(self.spec.input_names)))
        Y = self.columns(Y, U)
        F = _window_features(Y, self.time_steps)
        out = np.full(len(F), np.nan)
        ok = np.all(np.isfinite(F), axis=1)
        if ok.any():
            Xn = (F[ok] - self.x_mu) / self.x_sd
            out[ok] = self._decode(self.net.forward(Xn))
        return out


# ───────────────── repo AI-KF helpers (StateEstimator.m) ─────────────────────

def backward_window_mean(arr, w, nan_val=0.0):
    """ util/sliding_window_func_backward.m with func=@mean: result[i] is the mean
    of arr[i-w+1 : i+1], NaN (then nan_val) where the window is incomplete. """
    a = np.asarray(arr, dtype=float).ravel()
    out = np.full(a.shape, np.nan)
    w = max(1, int(w))
    for i in range(w - 1, len(a)):
        out[i] = a[i - w + 1:i + 1].mean()
    if nan_val is not None:
        out[~np.isfinite(out)] = nan_val
    return out


def logarithmic_map(data, original_bounds, target_bounds, reverse=False):
    """ util/logarithmic_map.m: normalize `data` onto [0,1] over original_bounds,
    then interpolate in log10 space between the target bounds. Positive target
    bounds only, which is all the repo uses. """
    d = np.asarray(data, dtype=float)
    o0, o1 = original_bounds
    a, b = target_bounds
    if reverse:
        a, b = b, a
    norm = (d - o0) / (o1 - o0)
    return 10.0 ** (np.log10(abs(a)) + norm * (np.log10(abs(b)) - np.log10(abs(a))))


def motif_R(motif, window, upper_bound, R_bounds):
    """ The AI-KF's time-varying noise for the ANN pseudo-measurement channel, as
    StateEstimator.set_motif builds it: backward window mean of the motif, mapped
    logarithmically from [0, upper_bound] onto R_bounds, then clipped to them.

    R_bounds = (R_at_zero_motif, R_at_full_motif) — the repo uses (1e12, 1e-3):
    no motif => the ANN channel is effectively ignored; full motif => trusted.
    Setting both to 1e12 disables the channel and recovers the plain filter. """
    win = backward_window_mean(motif, window, nan_val=0.0)
    r = logarithmic_map(win, (0.0, float(upper_bound)), R_bounds, reverse=False)
    hi, lo = max(R_bounds), min(R_bounds)
    return np.clip(r, lo, hi)


# ───────────────────────────── MIF / "AI-KF" ────────────────────────────────

def mif_beta(min_ev):
    """ Eq. 7:  beta_k = [ log10(10 + [F_omega^-1]_{i,i}) ]^-1.

    min-EV 0 -> beta = 1 (fully trust the ANN); min-EV large -> beta -> 0 (coast
    on the previous estimate). NaNs (window pad) give beta = 0. """
    ev = np.asarray(min_ev, dtype=float)
    beta = 1.0 / np.log10(10.0 + np.clip(ev, 0.0, None))
    return np.where(np.isfinite(beta), beta, 0.0)


def motif_informed_filter(raw, beta, circular=False, x0=None):
    """ Eq. 5:  x_hat_k = beta_k * x_check_k + (1 - beta_k) * x_hat_{k-1}.

    A circular state is blended on the unit circle — averaging the raw angles
    would let a +pi / -pi pair cancel to 0. Steps where the raw estimate is NaN
    (no omega history yet) hold the previous estimate, i.e. beta = 0. """
    raw = np.asarray(raw, dtype=float)
    beta = np.asarray(beta, dtype=float)
    n = len(raw)
    out = np.full(n, np.nan)
    if circular:
        s = c = None
        for k in range(n):
            b = 0.0 if not np.isfinite(raw[k]) else float(np.clip(beta[k], 0, 1))
            rs = np.sin(raw[k]) if np.isfinite(raw[k]) else 0.0
            rc = np.cos(raw[k]) if np.isfinite(raw[k]) else 0.0
            if s is None:
                if b == 0.0:
                    continue                 # nothing to initialize from yet
                s, c = rs, rc
            else:
                s, c = b * rs + (1 - b) * s, b * rc + (1 - b) * c
            out[k] = np.arctan2(s, c)
    else:
        m = None
        for k in range(n):
            b = 0.0 if not np.isfinite(raw[k]) else float(np.clip(beta[k], 0, 1))
            if m is None:
                if b == 0.0:
                    continue
                m = raw[k] if x0 is None else b * raw[k] + (1 - b) * x0
            else:
                m = b * raw[k] + (1 - b) * m
            out[k] = m
    return out
