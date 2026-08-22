"""
Tests for gen_stream_Chen25 (streamhelpers.py).

The CPU active-set integrator (method="auto" on CPU, or method="activeset") must
reproduce the vmapped diffrax path (method="vmap") for the same Chen+25 stream, at
matched tolerances, to integrator accuracy. This is the drop-in guarantee: the fast
path changes *how* the released particles are integrated, not the result.

Self-contained (no agama): a bound progenitor sprayed in a smooth NFW halo, integrated
forward from t=0 (past) to the present.

Run with:
    python tests/test_gen_stream_chen25_activeset.py     (or: pytest tests/…)
"""
import traceback

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import diffrax
import streamsculptor as ssc
from streamsculptor import potential as P
from streamsculptor.main import usys


# ---------------------------------------------------------------------------
# Shared setup: smooth NFW, a bound progenitor, forward stream over 3 Gyr
# ---------------------------------------------------------------------------
POT        = P.NFWPotential(m=1e12, r_s=15.0, units=usys)
TS         = jnp.linspace(0.0, 3000.0, 300)               # Myr, past -> present
PROG_W0    = jnp.array([30.0, 0.0, 0.0, 0.0, 0.16, 0.03])  # kpc, kpc/Myr (bound orbit)
MSAT       = 1e5
KEY        = jax.random.PRNGKey(0)
RTOL, ATOL = 5e-8, 1e-9                                    # matched-accuracy regime

# tolerance on the drop-in agreement: both integrators are converged to a few e-6 kpc
# at these settings on this smooth problem, so require agreement well below 1e-3 kpc.
MATCH_ATOL = 1e-4   # kpc


def _stream(method, solver=diffrax.Dopri8()):
    lead, trail = ssc.gen_stream_Chen25(
        pot_base=POT, ts=TS, prog_w0=PROG_W0, Msat=MSAT, key=KEY,
        solver=solver, rtol=RTOL, atol=ATOL, method=method,
    )
    return np.asarray(lead), np.asarray(trail)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestActiveSetMatchesVmap:

    def test_activeset_matches_vmap_positions_and_velocities(self):
        """method='activeset' reproduces method='vmap' (all 6 phase-space coords)."""
        lead_as, trail_as = _stream("activeset")
        lead_vm, trail_vm = _stream("vmap")

        assert np.isfinite(lead_as).all() and np.isfinite(trail_as).all()
        e_lead  = np.max(np.abs(lead_as  - lead_vm))
        e_trail = np.max(np.abs(trail_as - trail_vm))
        assert e_lead  < MATCH_ATOL, f"lead disagreement {e_lead:.2e} kpc"
        assert e_trail < MATCH_ATOL, f"trail disagreement {e_trail:.2e} kpc"

    def test_generic_solver_dopri5(self):
        """Generic tableau extraction: the same guarantee holds for Dopri5."""
        lead_as, trail_as = _stream("activeset", solver=diffrax.Dopri5())
        lead_vm, trail_vm = _stream("vmap",      solver=diffrax.Dopri5())
        assert np.max(np.abs(lead_as  - lead_vm))  < MATCH_ATOL
        assert np.max(np.abs(trail_as - trail_vm)) < MATCH_ATOL

    def test_output_shapes_match_vmap(self):
        """(lead, trail) shapes are (N_release - 1, 6), same as the vmapped path."""
        lead_as, trail_as = _stream("activeset")
        lead_vm, trail_vm = _stream("vmap")
        expected = (TS.shape[0] - 1, 6)
        assert lead_as.shape == expected and trail_as.shape == expected
        assert lead_as.shape == lead_vm.shape and trail_as.shape == trail_vm.shape


class TestAutoDispatch:

    def test_auto_equals_activeset_on_cpu(self):
        """On CPU, method='auto' must select the active-set path exactly."""
        if jax.default_backend() != "cpu":
            pytest.skip("auto selects vmap off CPU; this asserts the CPU branch")
        lead_auto, trail_auto = _stream("auto")
        lead_as,   trail_as   = _stream("activeset")
        assert np.array_equal(lead_auto, lead_as)
        assert np.array_equal(trail_auto, trail_as)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            ssc.gen_stream_Chen25(pot_base=POT, ts=TS, prog_w0=PROG_W0, Msat=MSAT,
                                  key=KEY, method="not_a_method")


if __name__ == "__main__":
    test_classes = [TestActiveSetMatchesVmap, TestAutoDispatch]
    passed = failed = 0
    for cls in test_classes:
        obj = cls()
        for name in [m for m in dir(cls) if m.startswith("test_")]:
            method = getattr(obj, name)
            try:
                method()
                print(f"  PASS  {cls.__name__}.{name}")
                passed += 1
            except Exception:
                print(f"  FAIL  {cls.__name__}.{name}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
