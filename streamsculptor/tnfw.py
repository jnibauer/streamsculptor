"""
tnfw.py — Truncated NFW (tidally-stripped) potentials via bfeax BFE.
Courtesy of Daniel Gilman, adapted from his original implementation.

Implements:
    TNFWPotential              — single TNFW, delegates to BFEPotential.from_density
    TNFWSubhaloLinePotential   — N subhalos on straight-line trajectories

Each subhalo potential is a TNFWPotential (BFEPotential) built at construction
time. Evaluation loops over them with jax.lax.cond per subhalo.

Requires:
    pip install git+https://github.com/jnibauer/bfeax.git
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from streamsculptor.main import Potential, usys
from streamsculptor.bfe import BFEPotential
# Tidal track helpers (Du+2024), shared with tnfw_analytic.py
from streamsculptor.tidal_track import _nfw_params_from_infall, _tidally_evolved_nfw_params
from functools import partial

jax.config.update("jax_enable_x64", True)


# =============================================================================
# TNFWPotential — single subhalo, delegates entirely to BFEPotential
# =============================================================================

class TNFWPotential(BFEPotential):
    """
    Truncated NFW (tidally-stripped) potential via BFEPotential.from_density.

    Density profile:
        rho(r) = rhos * ft / (u * (1+u)^2 * (1 + (r/rt)^2)),  u = r/rs

    Inherits potential(), gradient() (via _exp.force()), and density() from
    BFEPotential. Construction uses a spherical (l=0) BFE with bfeax.

    Parameters
    ----------
    rhos : float — characteristic density [Msun/kpc^3]
    rs   : float — NFW scale radius [kpc]
    ft   : float — tidal normalisation (dimensionless, <= 1)
    rt   : float — tidal truncation radius [kpc]
    n_r  : int   — BFE radial grid points (default 128)
    """

    @classmethod
    def from_profile(cls, rhos, rs, ft, rt, n_r=128, units=usys):
        """Construct from density profile parameters directly."""
        rhos_f, rs_f, ft_f, rt_f = rhos, rs, ft, rt

        def tnfw_density(x, y, z):
            r = jnp.sqrt(x**2 + y**2 + z**2)
            u = r / rs_f
            return rhos_f * ft_f / (u * (1.0 + u)**2 * (1.0 + (r / rt_f)**2))

        r_min = 1e-4 * rs_f
        r_max = jnp.maximum(50.0 * rt_f, 1e3 * rs_f)
        return cls.from_density(
            tnfw_density, r_min=r_min, r_max=r_max,
            n_r=n_r, l_max=0, symmetry="spherical", units=units, prune_modes=False
        )

    @classmethod
    def from_infall(cls, m_infall, c_infall, z_infall, f_bound, n_r=128, units=usys):
        """Construct from infall properties and bound mass fraction (Du+2024 tidal track)."""
        rhos, rs, ft, rt = _tidally_evolved_nfw_params(
            m_infall, c_infall, z_infall, f_bound
        )
        return cls.from_profile(rhos, rs, ft, rt, n_r=n_r, units=units)




# =============================================================================
# TNFWSubhaloLinePotential — N subhalos on straight-line trajectories
# =============================================================================

class TNFWSubhaloLinePotential(Potential):
    # 1. 'pots' is now a single object (a batched Pytree), not a list.
    # 2. It is NOT static because it contains the physical parameters (rhos, rs).
    pots: TNFWPotential  
    subhalo_x0: jnp.ndarray
    subhalo_v:  jnp.ndarray
    subhalo_t0: jnp.ndarray
    t_window: float

    def __init__(self, rhos, rs, ft, rt,
                 subhalo_x0, subhalo_v, subhalo_t0, t_window,
                 n_r=128, units=usys):
        super().__init__(units)
        
        # Vectorize the initialization. This creates one TNFWPotential 
        # instance where every internal leaf (rho_s, r_s, etc.) is an array of shape (N,).
        self.pots = jax.vmap(lambda _rhos, _rs, _ft, _rt: 
            TNFWPotential.from_profile(_rhos, _rs, _ft, _rt, n_r=n_r, units=units)
        )(jnp.asarray(rhos), jnp.asarray(rs), jnp.asarray(ft), jnp.asarray(rt))

        self.subhalo_x0 = jnp.asarray(subhalo_x0)
        self.subhalo_v  = jnp.asarray(subhalo_v)
        self.subhalo_t0 = jnp.asarray(subhalo_t0)
        self.t_window   = t_window

    @classmethod
    def from_infall(cls, m_infall, c_infall, z_infall, f_bound,
                    subhalo_x0, subhalo_v, subhalo_t0, t_window,
                    n_r=128, units=usys):
        # Ensure inputs are JAX arrays before calling the helper
        rhos, rs, ft, rt = _tidally_evolved_nfw_params(
            jnp.asarray(m_infall), jnp.asarray(c_infall),
            jnp.asarray(z_infall), jnp.asarray(f_bound),
        )
        return cls(rhos, rs, ft, rt, subhalo_x0, subhalo_v, subhalo_t0, t_window, n_r, units)

    def potential_per_SH(self, xyz, t):
        def compute_single_phi(pot, x0, v, t0):
            rel = xyz - (x0 + v * (t - t0))
            active = jnp.abs(t - t0) < self.t_window
            return jax.lax.cond(
                active,
                lambda r: pot.potential(r, t),
                lambda r: jnp.array(0.0),
                rel,
            )

        return jax.vmap(compute_single_phi)(
            self.pots, self.subhalo_x0, self.subhalo_v, self.subhalo_t0
        )

    def gradient_per_SH(self, xyz, t):
        def compute_single_grad(pot, x0, v, t0):
            rel = xyz - (x0 + v * (t - t0))
            active = jnp.abs(t - t0) < self.t_window
            return jax.lax.cond(
                active,
                lambda r: pot.gradient(r, t),   # analytic force from BFEPotential
                lambda r: jnp.zeros(3),
                rel,
            )

        return jax.vmap(compute_single_grad)(
            self.pots, self.subhalo_x0, self.subhalo_v, self.subhalo_t0
        )

    def potential(self, xyz, t):
        return jnp.sum(self.potential_per_SH(xyz, t))

    def gradient(self, xyz, t):
        # Analytic gradient via BFEPotential.gradient (spline derivatives),
        # bypassing reverse-mode AD through potential_per_SH entirely.
        return jnp.sum(self.gradient_per_SH(xyz, t), axis=0)

