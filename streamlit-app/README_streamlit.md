

## Deploy on Streamlit Community Cloud

Unlike a `--share` link, this survives closing your laptop: the app sleeps when
idle and wakes on the next visit.

1. Push this repo to GitHub (it must be public for the free tier).
2. At <https://share.streamlit.io> → **New app**, pick the repo and set the main
   file to `streamlit_app.py`.
3. Deploy. The first build takes several minutes — it compiles nothing, but
   `casadi`/`do-mpc`/`pybounds` are a big install.

The free tier gives ~1 GB of RAM, which the four built-in systems fit in, but
MPC is CPU-heavy — expect a few seconds per trajectory rebuild, and longer for
`alt2d` (501 steps). Changing only display or noise controls reuses the cached
engine and is fast.

One thing is deliberately **not** in the Streamlit build: **uploading a custom
system `.py`**. It `exec()`s the uploaded file, which is fine on your own machine
and remote code execution on a public URL. Use the Gradio app locally for that.

To slim the deploy, drop the `gradio` line from `requirements.txt` —
`streamlit_app.py` never imports it.

### Test the deploy locally first

Streamlit Cloud installs *only* `requirements.txt` into an empty environment, so
the usual way a deploy fails is a package your own environment happens to have.
Reproduce that with a throwaway venv:

```bash
python -m venv /tmp/deploytest
/tmp/deploytest/bin/pip install -r requirements.txt
/tmp/deploytest/bin/streamlit run streamlit_app.py
```

This is worth doing — it is how the `ipywidgets` import at the top of
`engine/observability_gui.py` was caught. Only the notebook GUI at the bottom of
that file uses it, but the module-level import made it mandatory, and it is not
in `requirements.txt`. It now degrades to `W = display = None`. A local conda
env with Jupyter installed hides that failure completely.

(A `Dockerfile` is also present if you want to test against Debian rather than
macOS — but note its `CMD` targets `app.py`.)

## Run the Streamlit version locally

```bash
streamlit run streamlit_app.py
```
