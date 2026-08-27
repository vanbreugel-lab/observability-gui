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

<video src="https://github.com/user-attachments/assets/77dfe85e-c1d3-4567-81ee-04271cd675d9" controls="controls" style="max-width: 730px;"></video>

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
| `tests/` | the unit-test suite (see *Tests* below) |
| `smoke_test.py` | end-to-end checks, incl. front-end parity (see below) |
| `engine/observability_gui.py` | the compute engine: MPC, Gramians, EKF/UKF |
| `engine/stochastic_observability.py` | the stochastic observability & constructability Gramians |
| `engine/drone_model.py` | the 3D drone dynamics/measurement model |
| `engine/dynamax_filters.py` | second EKF/UKF implementation, via dynamax (JAX) |
| `engine/ann_estimator.py` | ANN estimator + AI-KF (see below) |
| `tutorial_imgs/` | figures for the in-app tutorial and reference sections |
| `streamlit-app/streamlit_app.py` | Streamlit app — same core, different UI, work in progress as a potential way to permanently deploy online |

## Tests

```bash
pip install pytest
python -m pytest                  # the whole suite, ~1 minute
python -m pytest -m "not slow"    # skip the full-trajectory solves, ~40 s
python smoke_test.py              # the end-to-end script, incl. front-end parity
```

The suite is organised by what it protects, rather than by module:

| file | what it pins |
|---|---|
| `tests/test_tables.py` | the entry parsers — blank rows, stray letters, negative variances, names left over from another system. This is where a user's typing lands, and none of it may raise out of a callback |
| `tests/test_specs.py` | every built-in system's self-consistency: `f`/`h` return their declared dimensions, and every declared sensor/state actually exists. Nothing checks this at import time |
| `tests/test_trajectory.py` | set-point precedence, and the drawing canvas given degenerate input (one click, a repeated point, a 400-point scribble, a zero-size canvas) |
| `tests/test_engine.py` | the invariants the app's claims rest on: same seed → identical problem, P stays positive (the ±2σ band), min-EV scales with R and responds to λ, more sensors never loosen the bound, caches invalidate when they must |
| `tests/test_compute.py` | one refresh end to end, with control values that fight the numerics — a window longer than the trajectory, no sensors selected, λ below the round-off floor, R = 0. Also pins the Gradio wiring against `_compute`'s signature, by position and by name |
| `tests/test_app_robustness.py` | the upload paths: a system `.py` missing `f`, a syntax error, a CSV without the named columns, a corrupted canvas payload |
| `tests/test_dynamax_backend.py` | the second filter implementation agrees with the first (skipped when dynamax is not installed) |
| `tests/test_render_and_export.py` | every figure actually *draws* (matplotlib defers its work to render time), and the PDF export writes a real PDF |

Two conventions worth knowing if you add to it:

* **Callbacks may not raise.** A Gradio callback that throws shows the user a red
  box with no explanation. At that layer the requirement is a returned status
  string starting with ⚠ — so the tests are mostly "give it something wrong,
  assert it comes back with a message".
* **Engines are expensive and shared.** `conftest.py` solves each system once, on
  a short hand-made set-point, and hands the same engine to every test. Ask for
  the `fresh_engine` fixture if your test mutates one.

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
