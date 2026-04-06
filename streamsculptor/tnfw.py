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

jax.config.update("jax_enable_x64", True)



# =============================================================================
# Tidal track helpers (Du+2024, NFW alpha=1 beta=3 gamma=1 delta=2)
# =============================================================================

def _nfw_params_from_infall(m_infall, c_infall, z_infall,
                             H0=67.4, Omega_m=0.315, Omega_L=0.685):
    """Compute rhos, rs, R200 from infall properties (Planck 2018 cosmology by default)."""
    G = 4.498e-12  # kpc^3 / (Msun * Myr^2)
    H0_myr = H0 * 1e3 / 3.0856e22 * 3.15576e13
    Ez = jnp.sqrt(Omega_m * (1 + z_infall)**3 + Omega_L)
    rho_crit = 3 * (H0_myr * Ez)**2 / (8 * jnp.pi * G)
    R200 = (3 * m_infall / (4 * jnp.pi * 200 * rho_crit))**(1.0 / 3.0)
    rs = R200 / c_infall
    rhos = m_infall / (4 * jnp.pi * rs**3 * (jnp.log(1 + c_infall) - c_infall / (1 + c_infall)))
    return rhos, rs, R200


def _tidally_evolved_nfw_params(m_infall, c_infall, z_infall, f_bound):
    """
    Du+2024 tidal track parameters for a TNFW given infall properties and
    bound mass fraction. Returns rhos, rs, ft, rt.
    """
    rhos, rs, r200 = _nfw_params_from_infall(m_infall, c_infall, z_infall)
    A, B, C = 0.68492777, 0.66438857, 2.07766512
    D, E    = 0.75826635, 0.23376409
    ft = jnp.minimum((1 + D) * f_bound**E / (1 + D * f_bound**(2 * E)), 1.0)
    rt = (1 + A) * f_bound**B / (1 + A * f_bound**(2 * B)) / jnp.exp(C * (1 - f_bound)) * r200
    return rhos, rs, ft, rt


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

