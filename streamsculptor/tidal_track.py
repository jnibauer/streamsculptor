"""
tidal_track.py — Du+2024 tidal track for truncated NFW subhalos.

Shared by tnfw.py (bfeax BFE) and tnfw_analytic.py (Baltz+2009 closed form).
Pure JAX with no optional dependencies, so importing it never requires bfeax.

Follows pyHalo's compute_r_te_and_f_t. The Du+2024 fits are calibrated on a
c_vir = 20.6 reference halo, so the bound fraction is re-expressed relative to
the mass inside the infall r_max before entering the f_t fit, and r_t is then
solved so that the truncated profile's total mass equals the bound mass.
"""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# =============================================================================
# Tidal track helpers (Du+2024, NFW alpha=1 beta=3 gamma=1 delta=2)
# =============================================================================

_X_RMAX = 2.162581587064612     # r_max / r_s for NFW
_Y_SCALE = 0.19019440734112852  # M_mx,ref / M_bound,ref,0 (pyHalo Convert_to_reference_model)


def _mu_nfw(x):
    """NFW enclosed mass M(<x r_s) / (4 pi rhos r_s^3)."""
    return jnp.log1p(x) - x / (1.0 + x)


def _tnfw_total_mass_dimensionless(tau):
    """Total mass of rhos / (u (1+u)^2 (1 + (u/tau)^2)) in units of 4 pi rhos r_s^3."""
    t2 = tau**2
    return t2 / (t2 + 1.0)**2 * ((t2 - 1.0) * jnp.log(tau) + jnp.pi * tau - (t2 + 1.0))


# log-log inversion table for tau(M_tot), built once
_LOG_TAU_TAB = np.linspace(np.log(1e-8), np.log(1e6), 4000)
_LOG_MTOT_TAB = np.log(
    np.asarray(_tnfw_total_mass_dimensionless(jnp.exp(jnp.asarray(_LOG_TAU_TAB))))
)


def _tau_from_total_mass(m_tot, n_newton=2):
    """Invert _tnfw_total_mass_dimensionless: table lookup, then Newton steps in log tau."""
    log_m = jnp.log(m_tot)
    log_tau = jnp.interp(log_m, jnp.asarray(_LOG_MTOT_TAB), jnp.asarray(_LOG_TAU_TAB))
    f = lambda lt: jnp.log(_tnfw_total_mass_dimensionless(jnp.exp(lt)))
    for _ in range(n_newton):
        val, slope = jax.jvp(f, (log_tau,), (jnp.ones_like(log_tau),))
        log_tau = log_tau - (val - log_m) / slope
    return jnp.exp(log_tau)


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
    bound mass fraction f_bound = M_bound / M_200,infall. Returns rhos, rs, ft, rt.

    The density rhos ft / (u (1+u)^2 (1 + (r/rt)^2)) integrates to exactly
    f_bound * m_infall. Matches pyHalo's compute_r_te_and_f_t.
    """
    rhos, rs, _ = _nfw_params_from_infall(m_infall, c_infall, z_infall)
    mu_c = _mu_nfw(c_infall)

    # bound mass relative to the infall M(<r_max), mapped onto the reference halo
    x = f_bound * mu_c / _mu_nfw(_X_RMAX) * _Y_SCALE
    D, E = 0.75826635, 0.23376409
    ft = jnp.minimum((1 + D) * x**E / (1 + D * x**(2 * E)), 1.0)

    # f_t * M_tot(tau) = f_bound * mu(c)
    rt = rs * _tau_from_total_mass(f_bound * mu_c / ft)
    return rhos, rs, ft, rt
