#!/opt/anaconda3/envs/Python311/bin/python
"""Regenerate the EKF and UKF cycle diagrams used in Reference & equations.

    python tutorial_imgs/make_kf_eqs.py
        -> ekf_eqs.png / ekf_eqs_web.png
        -> ukf_eqs.png / ukf_eqs_web.png

The generic predict/correct loop is `kf_simple.png` (the main infographic); these
two fill in the equations each filter actually runs.

Notation matches the rest of the project — engine, equations section and the Q
write-up — rather than the F / H / z / k|k-1 convention of the stock diagram:

    Phi_k, C_k   Jacobians of f and h          (not F, H)
    y_k          measurement                   (not z)
    P^-_k, P^+_k prior / posterior covariance   (not P_{k|k-1}, P_{k|k})
    Q_k, R_k     per-step covariances, indexed as in System (22)
    lambda_s     unscented scaling parameter, distinct from the ridge lambda

The loop is drawn in the standard predict -> correct direction. The ENTRY
point is the correction step, because `run_ekf` / `run_ukf` take the supplied
initial estimate as the PRIOR at k = 0 (x_hat_{0|-1}) and so consume y_0. That
also keeps the filters aligned with the constructability Gramian, which starts
the same way (F_0 = C_0^T R^-1 C_0).
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

INK = '#1f2a37'
GREY = '#6b7280'
RED = '#c1442f'           # time update / predict  (matches the app's --world)
BLUE = '#2f6fb5'          # measurement update     (matches --belief)
EDGE = '#8fa8c4'
FILL = '#ffffff'

plt.rcParams.update({'mathtext.fontset': 'dejavusans', 'font.size': 11})

# ---- layout in INCHES, converted to figure fractions in build(). Keeping it
# absolute is what stops the taller UKF box from pushing its loop arrow into the
# footnotes: every figure gets the same margins and the height follows the
# content instead of being guessed.
HEAD_IN = 0.62            # title block inside a box
LINE_IN = 0.30            # label row
EQ_IN = 0.54              # equation row
PAD_IN = 0.18             # slack at the box bottom
TOPMARGIN_IN = 1.02       # room for the two header lines
BOTMARGIN_IN = 1.62       # room for the return arrow, its label and 2 footnotes


def box_height_in(lines):
    return HEAD_IN + len(lines) * (LINE_IN + EQ_IN) + PAD_IN


def box(ax, x, y, w, h, color, title, lines, f, eq_size=13.0):
    """f = figure fractions per inch, so the text metrics stay put."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.016',
        linewidth=2.0, edgecolor=color, facecolor=FILL, zorder=2))
    ax.plot([x + 0.012, x + w - 0.012], [y + h - HEAD_IN * f + 0.020 * f] * 2,
            color=color, lw=1.4, zorder=3)
    ax.text(x + w / 2, y + h - 0.30 * f, title, ha='center', va='center',
            fontsize=12.5, color=color, weight='bold', zorder=4)
    yy = y + h - HEAD_IN * f
    for label, eq in lines:
        ax.text(x + 0.020, yy, label, ha='left', va='top', fontsize=10.5,
                color=INK, zorder=4)
        yy -= LINE_IN * f
        ax.text(x + 0.038, yy, eq, ha='left', va='top', fontsize=eq_size,
                color=INK, zorder=4)
        yy -= EQ_IN * f


EKF_PREDICT = [
    ('1.  Propagate the estimate',
     r'$\hat{x}^{-}_{k+1} = f(\hat{x}^{+}_k,\, u_k)$'),
    ('2.  Propagate the covariance',
     r'$P^{-}_{k+1} = \Phi_k P^{+}_k \Phi_k^{\mathsf{T}} + Q_k$'),
]
EKF_CORRECT = [
    ('1.  Innovation covariance and gain',
     r'$S_k = C_k P^{-}_k C_k^{\mathsf{T}} + R_k ,\ \ '
     r'K_k = P^{-}_k C_k^{\mathsf{T}} S_k^{-1}$'),
    ('2.  Correct the estimate with $y_k$',
     r'$\hat{x}^{+}_k = \hat{x}^{-}_k + K_k'
     r'\left(y_k - h(\hat{x}^{-}_k, u_k)\right)$'),
    ('3.  Correct the covariance  (Joseph form)',
     r'$P^{+}_k = (I - K_k C_k) P^{-}_k (I - K_k C_k)^{\mathsf{T}}'
     r' + K_k R_k K_k^{\mathsf{T}}$'),
]

UKF_PREDICT = [
    ('1.  Draw sigma points from the posterior',
     r'$\chi_i = \sigma(\hat{x}^{+}_k,\, P^{+}_k)$'),
    ('2.  Push them through the dynamics',
     r'$\mathcal{X}_i = f(\chi_i, u_k) ,\ \ '
     r'\hat{x}^{-}_{k+1} = \sum_i W^{m}_i \mathcal{X}_i$'),
    ('3.  Propagate the covariance',
     r'$P^{-}_{k+1} = \sum_i W^{c}_i\, \delta\mathcal{X}_i\, '
     r'\delta\mathcal{X}_i^{\mathsf{T}} + Q_k$'),
]
UKF_CORRECT = [
    ('1.  Push sigma points through the sensors',
     r'$\mathcal{Y}_i = h(\chi_i, u_k) ,\ \ '
     r'\hat{y}_k = \sum_i W^{m}_i \mathcal{Y}_i$'),
    ('2.  Innovation and cross covariance',
     r'$S_k = \sum_i W^{c}_i \delta\mathcal{Y}_i \delta\mathcal{Y}_i'
     r'^{\mathsf{T}} + R_k ,\ \ '
     r'C^{xy}_k = \sum_i W^{c}_i \delta\chi_i \delta\mathcal{Y}_i'
     r'^{\mathsf{T}}$'),
    ('3.  Gain, then correct with $y_k$',
     r'$K_k = C^{xy}_k S_k^{-1} ,\ \ '
     r'\hat{x}^{+}_k = \hat{x}^{-}_k + K_k(y_k - \hat{y}_k)$'),
    ('4.  Correct the covariance',
     r'$P^{+}_k = P^{-}_k - K_k S_k K_k^{\mathsf{T}}$'),
]

EKF_FOOT = (r'$\Phi_k = \partial f/\partial x$  and  '
            r'$C_k = \partial h/\partial x$, by central differences about the '
            r'current estimate.  Angular innovations are wrapped to $(-\pi,\pi]$.')
UKF_FOOT = (r'$\chi_0 = \hat{x}$,  '
            r'$\chi_{i\pm} = \hat{x} \pm [\sqrt{(n_x+\lambda_s)P}\,]_i$  with  '
            r'$\lambda_s = \alpha^2(n_x+\kappa) - n_x$;  '
            r'$W^{m}_0 = \lambda_s/(n_x+\lambda_s)$,  '
            r'$W^{c}_0 = W^{m}_0 + (1-\alpha^2+\beta)$,  '
            r'$W^{m}_i = W^{c}_i = 1/[2(n_x+\lambda_s)]$.')


def build(kind):
    predict, correct, foot, title = (
        (EKF_PREDICT, EKF_CORRECT, EKF_FOOT, 'Extended Kalman filter')
        if kind == 'ekf' else
        (UKF_PREDICT, UKF_CORRECT, UKF_FOOT, 'Unscented Kalman filter'))

    LH_in, RH_in = box_height_in(predict), box_height_in(correct)
    fig_h = TOPMARGIN_IN + max(LH_in, RH_in) + BOTMARGIN_IN
    fig = plt.figure(figsize=(13.6, fig_h), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    f = 1.0 / fig_h                       # figure fractions per inch

    def yof(inches_from_top):
        return 1.0 - inches_from_top * f

    TOP = yof(TOPMARGIN_IN)
    LH, RH = LH_in * f, RH_in * f
    LX, LW = 0.030, 0.410
    RX, RW = 0.560, 0.410

    box(ax, LX, TOP - LH, LW, LH, RED, 'Time update  ("predict")', predict, f)
    box(ax, RX, TOP - RH, RW, RH, BLUE,
        'Measurement update  ("correct")', correct, f)

    y_fwd = TOP - min(LH, RH) * 0.55
    ax.add_patch(FancyArrowPatch((LX + LW + 0.008, y_fwd), (RX - 0.008, y_fwd),
                                 arrowstyle='-|>', mutation_scale=22,
                                 lw=2.6, color=RED, shrinkA=0, shrinkB=0))
    ax.text((LX + LW + RX) / 2, y_fwd + 0.10 * f,
            r'$\hat{x}^{-}_{k+1}$', ha='center', va='bottom', fontsize=12,
            color=RED)
    ax.text((LX + LW + RX) / 2, y_fwd - 0.13 * f, r'$P^{-}_{k+1}$',
            ha='center', va='top', fontsize=12, color=RED)

    ax.add_patch(FancyArrowPatch((RX + RW / 2, TOP - RH - 0.05 * f),
                                 (LX + LW / 2, TOP - LH - 0.05 * f),
                                 connectionstyle='arc3,rad=-0.26',
                                 arrowstyle='-|>', mutation_scale=22,
                                 lw=2.6, color=BLUE, shrinkA=4, shrinkB=4))
    ax.text(0.500, yof(fig_h - 0.86),
            r'posterior $\hat{x}^{+}_k,\ P^{+}_k$   into the next prediction',
            ha='center', va='center', fontsize=12.5, color=BLUE)

    ax.add_patch(FancyArrowPatch((RX + 0.050, yof(0.70)),
                                 (RX + 0.078, TOP + 0.06 * f),
                                 arrowstyle='-|>', mutation_scale=20,
                                 lw=2.0, color=EDGE, shrinkA=2, shrinkB=2))
    ax.text(0.030, yof(0.26), r'initial estimate   $\hat{x}^{-}_0,\ P^{-}_0$',
            ha='left', va='center', fontsize=12.5, color=INK)
    ax.text(0.030, yof(0.55),
            'it is the PRIOR at $k=0$, so the cycle is entered at the '
            'correction step and $y_0$ is used',
            ha='left', va='center', fontsize=10.5, color=GREY, style='italic')
    ax.text(0.970, yof(0.26), title, ha='right', va='center', fontsize=13,
            color=INK, weight='bold')

    ax.text(0.030, yof(fig_h - 0.44), foot, ha='left', va='center',
            fontsize=10.0, color=GREY)
    ax.text(0.030, yof(fig_h - 0.17),
            r'$Q_k,\ R_k$ — per-step process and measurement covariances, '
            r'indexed as in System (22).',
            ha='left', va='center', fontsize=10.0, color=GREY)
    return fig


if __name__ == '__main__':
    for kind in ('ekf', 'ukf'):
        fig = build(kind)
        for name, dpi in ((f'{kind}_eqs.png', 320), (f'{kind}_eqs_web.png', 118)):
            path = os.path.join(HERE, name)
            fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight')
            print(f'  wrote {path}  ({os.path.getsize(path)/1e6:.2f} MB)')
        plt.close(fig)
