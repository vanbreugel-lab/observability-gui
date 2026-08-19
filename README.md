# pybounds interactive app

An interactive browser GUI for observability analysis with
[pybounds](https://github.com/vanbreugel-lab/pybounds). Build a trajectory, recover controls that produce that trajectory with
MPC, then see how observable each state is along it — empirical Fisher
information, the stochastic observability and constructability Gramians, the
sliding-window minimum error variance, and EKF/UKF state estimates.

The numerics run in **Python on the server** (MPC needs casadi/do-mpc, which are
compiled); the browser only shows the controls and the rendered figures. That is
why this cannot be a static page.

## Walkthrough

A one-minute tour of the app — build a trajectory, simulate, read the
observability panels. No audio.

<video src="https://github.com/vanbreugel-lab/observability-gui/raw/main/tutorial_imgs/tutorial_video_web.mp4" controls muted playsinline width="100%"></video>

If the player above does not load, [download the walkthrough](tutorial_imgs/tutorial_video_web.mp4).

## Run it locally

Requires Python 3.11.

```bash
git clone https://github.com/vanbreugel-lab/pybounds-interactive-app.git
cd pybounds-interactive-app

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python app_custom.py
```

Then open the URL it prints — <http://localhost:7860> by default. If that port is
busy the app steps to the next free one.

To stop it, `Ctrl-C`. If a stale instance is holding the port:

```bash
lsof -ti tcp:7863 | xargs kill        # macOS / Linux
```
or

```bash
 pkill -f "python.*app_custom.py"      # macOS / Linux
```


## Share a temporary public link

Gradio can tunnel your locally running app to a public URL — useful for showing
a colleague without permanantly deploying. Pass `--share`:

```bash
python app_custom.py --share
```

It prints a second URL like `https://xxxxxxxx.gradio.live` alongside the local
one. The first such launch downloads a small tunnel binary (`frpc`) into the
Gradio package directory, so it needs network access and takes a few extra
seconds; later launches reuse it. 

**Note:**

- **Your machine is serving the link.** The link dies when you stop the process or
  close the laptop. The link expires after about a week.
- **The link is public and unauthenticated** — anyone with the URL can use the app and
  the CPU behind it. Every visitor's Simulate runs MPC on your machine. To
  require a password, add `auth=` to the `launch()` call at the bottom of
  `app_custom.py`:

  ```python
  demo.launch(server_name='0.0.0.0', server_port=port, share=share,
              auth=('username', 'password'),
              theme=THEME, css=CSS, allowed_paths=[TUT_DIR])
  ```


## What's in here

| path | |
|---|---|
| `core.py` | **the shared compute core** — set-point builder, `compute_payload`, figure assembly, per-system defaults. UI-agnostic; imports no toolkit |
| `app_custom.py` | the Gradio app — layout, controls, callbacks |
| `render.py` | every matplotlib figure |
| `smoke_test.py` | end-to-end checks, incl. front-end parity (see below) |
| `engine/observability_gui.py` | the compute engine: MPC, Gramians, EKF/UKF |
| `engine/stochastic_observability.py` | the stochastic observability & constructability Gramians |
| `engine/drone_model.py` | the 3D drone dynamics/measurement model |
| `engine/ann_estimator.py` | ANN estimator + AI-KF (see below) |
| `tutorial_imgs/` | figures for the in-app tutorial and reference sections |
| `streamlit-app/streamlit_app.py` | Streamlit app — same core, different UI, work in progress as a potential way to permanently deploy online |

## Work in progress

The **ANN** and **AI-KF** estimators are implemented but not yet validated —
their accuracy depends heavily on how the training trajectories are generated.
Their controls are greyed out in the app and their compute path is disabled;
the equations and defaults are still shown for reference. To re-enable them for
development, set `ANN_ENABLED = True` in `app_custom.py`.


## References

- Cellini, Boyacıoğlu, Stupski & van Breugel, *Discovering and exploiting active
  sensing motifs for estimation*, [arXiv:2511.08766](https://arxiv.org/abs/2511.08766)
- Boyacıoğlu & van Breugel, *Duality of stochastic observability and
  constructability and their relation to the Fisher information*,
  [doi:10.1109/LCSYS.2025.3547297](https://doi.org/10.1109/LCSYS.2025.3547297)
