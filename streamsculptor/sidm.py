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
from diffrax import diffeqsolve, ODETerm, Dopri5,SaveAt,PIDController,DiscreteTerminatingEvent, DirectAdjoint, RecursiveCheckpointAdjoint, ConstantStepSize, Euler, StepTo
import diffrax
import equinox as eqx
from jax.scipy import special
usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)

from .main import Potential
import interpax
import os


class SIDMPotential(Potential):
    def __init__(self, M_cdm, r_s_cdm, tf, interp=None, units=None):
        """
        Initializes a SIDM potential with an interpolated function.
        Subhalo profile from Shin’ichiro Ando et al. 2025.
        https://iopscience.iop.org/article/10.1088/1475-7516/2025/02/053
        Includes a time evolving scale radius and core radius, cosmologically motivated.
        """
        super().__init__(units, {'M_cdm': M_cdm, 'r_s_cdm': r_s_cdm, 'tf': tf})
        if interp is None:
            ###data_path = os.path.join(os.path.dirname(__file__), 'data/sidm', 'sidm_interpolant_Menc.npy')
            data_path = os.path.join(os.path.dirname(__file__), 'data/sidm', 'jax_interp.npy')
            ###interp = jnp.load(data_path, allow_pickle=True).item()
            interp = jnp.load(data_path, allow_pickle=True).item()
        self.interp = interp
        self.density = self.density_func
    
    @partial(jax.jit, static_argnums=(0,))
    def interp_func(self, r_c, r_s, r):
        """
        Interpolates the enclosed mass using the provided interpolant.
        """
        xx = jnp.array([r_c, r_s, r])
        return self.interp(xx)

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

    @partial(jax.jit, static_argnums=(0,))
    def rho_s_SIDM(self, t, r_s_cdm, rho_s_cdm, tf):
        """Calculate the SIDM scale density."""
        t_tilde_value = self.t_tilde(t, tf, rho_s_cdm, r_s_cdm)
        rho_s_sidm_value = 2.033 + 0.7381 * t_tilde_value + 7.264 * t_tilde_value**5 - 12.73 * t_tilde_value**7 + 9.915 * t_tilde_value**9 + (1-2.033)*(jnp.log(t_tilde_value + 0.001)/jnp.log(0.001))
        return rho_s_sidm_value * rho_s_cdm

    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz, t):
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def density_func(self, xyz, t):
        """
           Calculate the SIDM density.
           Equation 2.11 of https://arxiv.org/pdf/2403.16633
        """
        rho_s_cdm = self.get_nfw_rho0(self.M_cdm, self.r_s_cdm)
        r_s = self.r_s_sidm(t, self.r_s_cdm, rho_s_cdm, self.tf)
        r_c = self.r_c_sidm(t, self.r_s_cdm, rho_s_cdm, self.tf)
        rho_s = self.rho_s_SIDM(t, self.r_s_cdm, rho_s_cdm, self.tf)

        r = jnp.sqrt(jnp.sum(xyz**2))
        beta = 4.0

        denom_bracket1 = ( (r**beta + r_c**beta)**(1/beta) ) / r_s
        denom_bracket2 = ((r/r_s)  + 1)**2

        denom = denom_bracket1 * denom_bracket2

        return rho_s / denom


    
    
    #eqx.filter_jit
    @partial(jax.jit, static_argnums=(0,))
    def gradient(self, xyz, t):
        """Calculate the SIDM acceleration."""
        r = jnp.sqrt(jnp.sum(xyz**2))
        rho_s_cdm = self.get_nfw_rho0(self.M_cdm, self.r_s_cdm)
        r_s = self.r_s_sidm(t, self.r_s_cdm, rho_s_cdm, self.tf)
        r_c = self.r_c_sidm(t, self.r_s_cdm, rho_s_cdm, self.tf)
        rho_s = self.rho_s_SIDM(t, self.r_s_cdm, rho_s_cdm, self.tf)
        
        shape = self.interp_func(r_c,r_s,r)#self.interp(r_c, r_s, r)
        M_enc = rho_s * shape
        return self._G * M_enc / r**2

    #eqx.filter_jit
    @partial(jax.jit, static_argnums=(0,))
    def acceleration(self,xyz,t):
        return -self.gradient(xyz, t)



class SIDMLinePotential(Potential):
    def __init__(self, M_cdm, r_s_cdm, tf, subhalo_x0, subhalo_v, subhalo_t0, t_window, units=None):
        super().__init__(units, {'M_cdm': M_cdm, 
                                 'r_s_cdm': r_s_cdm, 
                                 'tf': tf, 
                                 'subhalo_x0': subhalo_x0, 
                                 'subhalo_v': subhalo_v, 
                                 'subhalo_t0': subhalo_t0, 
                                 't_window': t_window})
        ####data_path = os.path.join(os.path.dirname(__file__), 'data/sidm', 'sidm_interpolant_Menc.npy')
        ####self.interp = jnp.load(data_path, allow_pickle=True).item()
        data_path = os.path.join(os.path.dirname(__file__), 'data/sidm', 'jax_interp.npy')
        self.interp = jnp.load(data_path, allow_pickle=True).item()
    @partial(jax.jit,static_argnums=(0,))
    def single_subhalo_acceleration(self, xyz, M_cdm, r_s_cdm, tf, t):
        return SIDMPotential(M_cdm=M_cdm, r_s_cdm = r_s_cdm, tf=tf, interp=self.interp, units=self.units).acceleration(xyz,t)
        
    @partial(jax.jit, static_argnums=(0,))
    def acceleration_per_SH(self, xyz, t):


        def true_func(subhalo_x0, subhalo_v, subhalo_t0, M_cdm, r_s_cdm, tf, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v*(t - subhalo_t0))
            acceleration_vals = self.single_subhalo_acceleration(relative_position, M_cdm, r_s_cdm, tf, t) 
            r_hat = relative_position / jnp.linalg.norm(relative_position)
            return acceleration_vals * r_hat
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, M_cdm, r_s_cdm, tf, t):
            return jnp.zeros(3)

        pred1 = jnp.abs(t - self.subhalo_t0) < self.t_window # True if in window, false otherwise
        pred2 = t >= self.tf # True if after tf, false otherwise
        pred = pred1 & pred2
        vmapped_cond = jax.vmap(jax.lax.cond,in_axes=((0,None,None,0,0,0,0,0,0,None)))
        acc_per_subhalo = vmapped_cond(pred,true_func,false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, self.M_cdm, self.r_s_cdm, self.tf, t) 
        return acc_per_subhalo
    
    @partial(jax.jit, static_argnums=(0,))
    def gradient(self, xyz, t):
        """
        Calculate the total gradient from all subhalos.
        """
        return -jnp.sum(self.acceleration_per_SH(xyz, t), axis=0)

    @partial(jax.jit, static_argnums=(0,))
    def acceleration(self, xyz, t):
        """
        Calculate the total acceleration from all subhalos.
        """
        return jnp.sum(self.acceleration_per_SH(xyz, t), axis=0)