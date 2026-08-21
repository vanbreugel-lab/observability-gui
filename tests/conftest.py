"""
Shared fixtures for the test suite.

    pip install pytest
    python -m pytest -q            # from the repo root

Design notes:

* **MPC dominates the runtime.** Every engine here is built from a SHORT
  hand-made set-point (a couple of dozen steps) rather than a system's default
  trajectory, and engines are cached for the whole session. A suite that solved
  the real trajectories would take minutes and get skipped, which is worse than
  a suite that exercises the same code paths on 24 steps.
* **Shared engines are read-only.** Tests that mutate an engine (re-solving a
  trajectory, clearing caches) must ask for `fresh_engine`, because
  `set_trajectory` bumps the version and wipes every cache the other tests are
  relying on.
* IPOPT and do-mpc are chatty on stdout; `silent()` swallows that so a failure
  is readable.
"""
import contextlib
import io
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'engine'))

import matplotlib
matplotlib.use('Agg')                  # no display in CI
import numpy as np

import core                                                    # noqa: E402
import render                                                   # noqa: E402

# every built-in system, in the order the app lists them. 'custom' is not here:
# it has no spec until a file is uploaded (see test_custom_systems.py).
BUILTINS = ('fly', 'fly7', 'drone', 'alt2d')

# how many trajectory steps the shared engines get. Big enough that a sliding
# window fits several times over, small enough that the MPC solve is quick.
N_STEPS = 24


def silent():
    """ Swallow the solver's stdout chatter. """
    return contextlib.redirect_stdout(io.StringIO())


def short_setpoint(spec, n=N_STEPS):
    """ An n-step constant-speed, constant-heading set-point for any built-in
    spec (they all take make_setpoint(speed, heading, wind, zeta)). """
    return spec.make_setpoint(
        float(spec.v0_default) * np.ones(n),
        float(getattr(spec, 'heading0', 0.0)) * np.ones(n),
        float(spec.wind_default), float(spec.zeta_default))


def build_engine(system, n=N_STEPS):
    """ A solved engine for `system`. Not cached — see the `engine` fixture. """
    spec = core.get_spec(system)
    eng = core.ObservabilityEngine(spec)
    with silent():
        eng.simulate_mpc(short_setpoint(spec, n))
    return eng


_CACHE = {}


@pytest.fixture(scope='session')
def engines():
    """ {system: solved engine}, built once. READ-ONLY: do not re-solve or
    clear caches on these — use `fresh_engine` for that. """
    for name in BUILTINS:
        if name not in _CACHE:
            _CACHE[name] = build_engine(name)
    return dict(_CACHE)


@pytest.fixture(scope='session')
def engine(engines):
    """ One representative engine. alt2d: 4 states, one measurement, no angular
    channels and linear dynamics — the system whose numbers are easiest to
    reason about when something breaks. """
    return engines['alt2d']


@pytest.fixture
def fresh_engine():
    """ A throwaway solved engine for tests that mutate one. """
    return build_engine('alt2d')


@pytest.fixture(scope='session')
def app():
    """ app_custom, imported once (importing it builds the whole Gradio
    Blocks, which is not free). """
    import app_custom
    return app_custom


def q_of(spec, v=1e-4):
    """ A per-state Q dict, using a spec's own jitter floor when it has one:
    fly's constant-parameter states need it to keep the recursions invertible.
    """
    return dict(getattr(spec, 'q_tiny', None) or {s: v for s in spec.state_names})


def r_of(spec, v=0.1):
    return {m: v for m in spec.measurement_names}


def rel_diff(a, b):
    """ Max per-column relative difference between two (N, k) trajectories,
    scaled by each column's own magnitude — the comparison used throughout for
    'do these two implementations agree'. """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    scale = np.maximum(np.abs(b).max(axis=0), 1e-9)
    return float((np.abs(a - b).max(axis=0) / scale).max())
