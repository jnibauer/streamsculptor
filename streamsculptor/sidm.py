from functools import partial
from astropy.constants import G
import astropy.coordinates as coord
import astropy.units as u
# gala
from gala.units import dimensionless, UnitSystem

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import jax.random as random 
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from diffrax import diffeqsolve, ODETerm, Dopri5,SaveAt,PIDController,DiscreteTerminatingEvent, DirectAdjoint, RecursiveCheckpointAdjoint, ConstantStepSize, Euler, StepTo
import diffrax
import equinox as eqx
from jax.scipy import special
usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)

from streamsculptor import Potential
import interpax
import os


class SIDMPotential(Potential):
    def __init__(self, M_cdm, r_s_cdm, tf, interp=None, units=None):
        """
        Initializes a SIDM potential with an interpolated function.
        """
        super().__init__(units, {'M_cdm': M_cdm, 'r_s_cdm': r_s_cdm, 'tf': tf})
        if interp is None:
            data_path = os.path.join(os.path.dirname(__file__), 'data/sidm', 'sidm_interpolant_Menc.npy')
            interp = jnp.load(data_path, allow_pickle=True).item()
        self.interp = interp
        
    @partial(jax.jit, static_argnums=(0,))
    def t_c(self, t, rho_s_cdm, r_s_cdm):
        sig_eff_over_m_chi = (147.1) * (u.cm**2 / u.g).to(u.kpc**2 / u.Msun)
        C = 0.75
        denom1 = C * sig_eff_over_m_chi * rho_s_cdm * r_s_cdm 
        denom2 = jnp.sqrt( 4 * jnp.pi * rho_s_cdm * self._G)
        return 150 / (denom1 * denom2)
    
    @partial(jax.jit, static_argnums=(0,))
    def t_tilde(self, t,tf, rho_s_cdm, r_s_cdm):
        return (t-tf) / self.t_c(t, rho_s_cdm, r_s_cdm)


    @partial(jax.jit, static_argnums=(0,))
    def r_s_sidm(self, t, r_s_cdm, rho_s_cdm, tf):
        """Calculate the SIDM scale radius."""
        t_tilde_value = self.t_tilde(t, tf, rho_s_cdm, r_s_cdm)
        r_s_sidm_value = 0.7178 + (0.1026*t_tilde_value) + 0.2474 * t_tilde_value**2 - 0.4079 * t_tilde_value**6 + (1-0.7178)*jnp.log(t_tilde_value + 0.001)/jnp.log(0.001)
        return r_s_sidm_value * r_s_cdm

    @partial(jax.jit, static_argnums=(0,))
    def r_c_sidm(self, t, r_s_cdm, rho_s_cdm, tf):
        """Calculate the SIDM core radius."""
        t_tilde_value = self.t_tilde(t, tf, rho_s_cdm, r_s_cdm)
        r_c_sidm_value = 2.555*jnp.sqrt(t_tilde_value) - 3.632*t_tilde_value + 2.131 * t_tilde_value**2 - 1.415 * t_tilde_value**3 + 0.4683 * t_tilde_value**4
        return r_c_sidm_value * r_s_cdm

    @partial(jax.jit, static_argnums=(0,))
    def get_nfw_rho0(self, M,r_s):
        c_NFW = 15. # following https://arxiv.org/pdf/2211.04495
        denom = jnp.log(1. + c_NFW) - (c_NFW/(1+c_NFW))
        fac = 1./denom
        rho0 = (M/(4*jnp.pi*r_s**3))*fac
        return rho0

    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz, t):
        raise NotImplementedError
    
    eqx.filter_jit
    def gradient(self, xyz, t):
        """Calculate the SIDM acceleration."""
        r = jnp.sqrt(jnp.sum(xyz**2))
        rho_s_cdm = self.get_nfw_rho0(self.M_cdm, self.r_s_cdm)
        r_s = self.r_s_sidm(t, self.r_s_cdm, rho_s_cdm, self.tf)
        r_c = self.r_c_sidm(t, self.r_s_cdm, rho_s_cdm, self.tf)
        shape = self.interp(r_c, r_s, r)
        M_enc = rho_s_cdm * shape
        return self._G * M_enc / r**2

    eqx.filter_jit
    def acceleration(self,xyz,t):
        return -self.gradient(xyz, t)