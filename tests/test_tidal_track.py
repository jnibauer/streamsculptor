"""
Tests for the Du+2024 tidal track (tidal_track.py).

Checks agreement with pyHalo's compute_r_te_and_f_t, exact mass conservation of
the truncated profile, and that the track is jittable, vmappable and differentiable.

Run with:
    pytest tests/test_tidal_track.py
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from streamsculptor.tidal_track import (
    _nfw_params_from_infall,
    _tidally_evolved_nfw_params,
    _tnfw_total_mass_dimensionless,
)


def _grid():
    c, f = np.meshgrid([2.0, 3.0, 5.0, 10.0, 20.0, 40.0, 80.0, 120.0], np.logspace(-5, 0, 26))
    c, f = c.ravel(), f.ravel()
    rng = np.random.default_rng(0)
    m = 10 ** rng.uniform(5.0, 11.0, c.size)
    z = rng.uniform(0.0, 6.0, c.size)
    return m, c, z, f


def test_matches_pyhalo():
    pytest.importorskip("pyHalo")
    from pyHalo.Halos.galacticus_truncation.transfer_function_density_profile import compute_r_te_and_f_t

    m, c, z, f = _grid()
    _, _, ft, rt = (np.asarray(a) for a in _tidally_evolved_nfw_params(m, c, z, f))
    _, _, r200 = (np.asarray(a) for a in _nfw_params_from_infall(m, c, z))
    rt_ph, ft_ph = compute_r_te_and_f_t(f * m, m, r200, c)

    # rt tolerance is set by pyHalo's own cubic lookup table (~1.5e-6)
    np.testing.assert_allclose(ft, ft_ph, rtol=1e-12)
    np.testing.assert_allclose(rt, rt_ph, rtol=1e-5)


def test_profile_mass_equals_bound_mass():
    m, c, z, f = _grid()
    rhos, rs, ft, rt = _tidally_evolved_nfw_params(m, c, z, f)
    m_profile = 4 * jnp.pi * rhos * ft * rs**3 * _tnfw_total_mass_dimensionless(rt / rs)
    np.testing.assert_allclose(np.asarray(m_profile), f * m, rtol=1e-10)


def test_jit_vmap_grad():
    m, c, z, f = (jnp.asarray(a) for a in _grid())
    eager = _tidally_evolved_nfw_params(m, c, z, f)
    jitted = jax.jit(jax.vmap(_tidally_evolved_nfw_params))(m, c, z, f)
    for a, b in zip(eager, jitted):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12)

    drt_df = jax.vmap(jax.grad(lambda ff, cc: _tidally_evolved_nfw_params(1e8, cc, 1.0, ff)[3]))(f, c)
    assert bool(jnp.all(jnp.isfinite(drt_df)))
    assert bool(jnp.all(drt_df > 0))  # more bound mass -> larger truncation radius
