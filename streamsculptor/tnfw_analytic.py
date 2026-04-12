"""
tnfw_new.py — Truncated NFW potentials using the analytic Baltz+2009 formula.

Mirrors the interface of tnfw.py but replaces the bfeax BFE multipole expansion
with the closed-form 3D potential from tNFWPotential (potential.py). No bfeax
dependency; faster construction and evaluation.

Implements:
    AnalyticTNFWPotential            — single TNFW, wraps tNFWPotential directly
    AnalyticTNFWSubhaloLinePotential — N subhalos on straight-line trajectories

Tidal-track helpers (_nfw_params_from_infall, _tidally_evolved_nfw_params) are
identical to those in tnfw.py.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from streamsculptor.main import Potential, usys
from streamsculptor.potential import tNFWPotential

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
# AnalyticTNFWPotential — single subhalo, pure analytic potential
# =============================================================================

class TNFWPotential(Potential):
    """
    Truncated NFW potential using the analytic Baltz+2009 formula.

    Wraps tNFWPotential directly; no BFE grid is built.

    Parameters
    ----------
    rhos : float — characteristic density [Msun/kpc^3]
    r_s  : float — NFW scale radius [kpc]
    r_t  : float — tidal truncation radius [kpc]
    f_t  : float — tidal normalisation (dimensionless, <= 1)
    """
    _pot: tNFWPotential

    def __init__(self, rhos, r_s, r_t, f_t, units=usys):
        super().__init__(units)
        self._pot = tNFWPotential(rhos=rhos, r_s=r_s, r_t=r_t, f_t=f_t, units=units)

    @classmethod
    def from_profile(cls, rhos, r_s, f_t, r_t, units=usys):
        """Construct from density profile parameters directly."""
        return cls(rhos=rhos, r_s=r_s, r_t=r_t, f_t=f_t, units=units)

    @classmethod
    def from_infall(cls, m_infall, c_infall, z_infall, f_bound, units=usys):
        """Construct from infall properties and bound mass fraction (Du+2024 tidal track)."""
        rhos, rs, ft, rt = _tidally_evolved_nfw_params(
            m_infall, c_infall, z_infall, f_bound
        )
        return cls(rhos=rhos, r_s=rs, r_t=rt, f_t=ft, units=units)

    def potential(self, xyz, t):
        return self._pot.potential(xyz, t)


# =============================================================================
# AnalyticTNFWSubhaloLinePotential — N subhalos on straight-line trajectories
# =============================================================================

class TNFWSubhaloLinePotential(Potential):
    """
    N truncated-NFW subhalos moving on straight-line trajectories,
    evaluated with the analytic Baltz+2009 potential.

    Each subhalo is active only within ±t_window of its encounter time t0.

    Per-subhalo constants (c0, a1, b1, a2, b2, a3, b3) are precomputed at
    construction so the hot-path evaluation only does:
        phi = c0 + (a1 + b1/r)*log(1+r/rs) + (a2 + b2/r)*log(1+(r/rt)^2) + (a3 + b3/r)*arctan(r/rt)

    Parameters
    ----------
    rhos, r_s, r_t, f_t   : array-like, shape (N,) — TNFW profile parameters
    subhalo_x0             : array, shape (N, 3) — position at t0 [kpc]
    subhalo_v              : array, shape (N, 3) — constant velocity [kpc/Myr]
    subhalo_t0             : array, shape (N,)   — encounter time [Myr]
    t_window               : float               — half-duration of active window [Myr]
    """
    # Precomputed per-subhalo constants (shape N each)
    _c0: jnp.ndarray
    _a1: jnp.ndarray
    _b1: jnp.ndarray
    _a2: jnp.ndarray
    _b2: jnp.ndarray
    _a3: jnp.ndarray
    _b3: jnp.ndarray
    _r_s: jnp.ndarray   # still needed as log/arctan argument scale
    _r_t: jnp.ndarray
    subhalo_x0: jnp.ndarray
    subhalo_v:  jnp.ndarray
    subhalo_t0: jnp.ndarray
    t_window: float

    def __init__(self, rhos, r_s, r_t, f_t,
                 subhalo_x0, subhalo_v, subhalo_t0, t_window,
                 units=usys):
        super().__init__(units)
        rhos = jnp.asarray(rhos)
        r_s  = jnp.asarray(r_s)
        r_t  = jnp.asarray(r_t)
        f_t  = jnp.asarray(f_t)

        tau   = r_t / r_s
        t2    = tau ** 2
        t2_m1 = t2 - 1.0
        t2_p1 = t2 + 1.0
        P     = units.G * 4.0 * jnp.pi * rhos * f_t * r_s**2 * t2 / t2_p1**2

        self._c0 = P * (jnp.pi * t2_m1 / (2.0 * tau) - jnp.log(t2))
        self._a1 = 2.0 * P
        self._b1 = -P * t2_m1 * r_s
        self._a2 = -P
        self._b2 = P * t2_m1 * r_s / 2.0
        self._a3 = -P * t2_m1 / tau
        self._b3 = -2.0 * P * r_t
        self._r_s = r_s
        self._r_t = r_t

        self.subhalo_x0 = jnp.asarray(subhalo_x0)
        self.subhalo_v  = jnp.asarray(subhalo_v)
        self.subhalo_t0 = jnp.asarray(subhalo_t0)
        self.t_window   = float(t_window)

    @classmethod
    def from_infall(cls, m_infall, c_infall, z_infall, f_bound,
                    subhalo_x0, subhalo_v, subhalo_t0, t_window,
                    units=usys):
        rhos, rs, ft, rt = _tidally_evolved_nfw_params(
            jnp.asarray(m_infall), jnp.asarray(c_infall),
            jnp.asarray(z_infall), jnp.asarray(f_bound),
        )
        return cls(rhos, rs, rt, ft,
                   subhalo_x0, subhalo_v, subhalo_t0, t_window, units)

    def single_subhalo_potential(self, xyz, c0, a1, b1, a2, b2, a3, b3, r_s, r_t, t):
        r = jnp.clip(jnp.linalg.norm(xyz), a_min=1e-10)
        inv_r = 1.0 / r
        return (c0
                + (a1 + b1 * inv_r) * jnp.log(1.0 + r / r_s)
                + (a2 + b2 * inv_r) * jnp.log(1.0 + (r / r_t)**2)
                + (a3 + b3 * inv_r) * jnp.arctan(r / r_t))

    def potential_per_SH(self, xyz, t):
        def true_func(x0, v, t0, c0, a1, b1, a2, b2, a3, b3, r_s, r_t, t):
            rel = xyz - (x0 + v * (t - t0))
            return self.single_subhalo_potential(rel, c0, a1, b1, a2, b2, a3, b3, r_s, r_t, t)

        def false_func(x0, v, t0, c0, a1, b1, a2, b2, a3, b3, r_s, r_t, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window
        vmapped_cond = jax.vmap(jax.lax.cond,
                                in_axes=(0, None, None,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None))
        return vmapped_cond(pred, true_func, false_func,
                            self.subhalo_x0, self.subhalo_v, self.subhalo_t0,
                            self._c0, self._a1, self._b1,
                            self._a2, self._b2,
                            self._a3, self._b3,
                            self._r_s, self._r_t, t)

    def potential(self, xyz, t):
        return jnp.sum(self.potential_per_SH(xyz, t))
