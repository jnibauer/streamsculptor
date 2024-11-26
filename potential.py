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

from StreamSculptor import Potential


class LMCPotential(Potential):
    def __init__(self, LMC_internal, LMC_orbit, units=None):
        """
        LMC_internal: dictionary of LMC internal (i.e., structural) parameters
        LMC_orbit: {x,y,z,t}
        """
        super().__init__(units, {'LMC_internal': LMC_internal, 'LMC_orbit': LMC_orbit})
        self.spl_x = InterpolatedUnivariateSpline(self.LMC_orbit['t'], self.LMC_orbit['x'],k=3)
        self.spl_y = InterpolatedUnivariateSpline(self.LMC_orbit['t'], self.LMC_orbit['y'],k=3)
        self.spl_z = InterpolatedUnivariateSpline(self.LMC_orbit['t'], self.LMC_orbit['z'],k=3)

    
    @partial(jax.jit, static_argnums=(0,))
    def potential(self, xyz, t):
        LMC_pos = jnp.array([ self.spl_x(t), self.spl_y(t), self.spl_z(t) ])
        xyz_adjust = xyz - LMC_pos
        
        
        potential_lmc = NFWPotential(m=self.LMC_internal['m_NFW'], r_s=self.LMC_internal['r_s_NFW'],units=usys)
        #pot_bar = BarPotential(m=self.LMC_internal['bar_m'], a=self.LMC_internal['bar_a'],
        #                         b=self.LMC_internal['bar_b'], c=self.LMC_internal['bar_c'],Omega=self.LMC_internal['bar_Omega'],units=usys)
        #potential_list = [pot_NFW,pot_bar]
        #potential_lmc = Potential_Combine(potential_list=potential_list,units=usys)
        return potential_lmc.potential(xyz_adjust, t)


class MiyamotoNagaiDisk(Potential):
    def __init__(self, m, a, b, units=None):
        super().__init__(units, {'m': m, 'a': a, 'b': b,})
    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz,t):
        R2 = xyz[0]**2 + xyz[1]**2
        return -self._G*self.m / jnp.sqrt(R2 + jnp.square(jnp.sqrt(xyz[2]**2 + self.b**2) + self.a))

class NFWPotential(Potential):
    """
    standard def see spherical model @ https://github.com/adrn/gala/blob/main/gala/potential/potential/builtin/builtin_potentials.c
    """
    def __init__(self, m, r_s, units=None):
        super().__init__(units, {'m': m, 'r_s': r_s})
    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz,t):
        v_h2 = -self._G*self.m/self.r_s
        m = jnp.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2 )/self.r_s ##removed softening! used to be .001 after xyz[2]**2
        return v_h2*jnp.log(1.0+ m) / m

class Isochrone(Potential):
    
    def __init__(self, m, a, units=None):
        super().__init__(units, {'m': m, 'a': a})
    
    @partial(jax.jit, static_argnums=(0,))
    def potential(self, xyz, t):
        r = jnp.linalg.norm(xyz, axis=0)
        return - self._G * self.m / (self.a + jnp.sqrt(r**2 + self.a**2))
    
class PlummerPotential(Potential):
    def __init__(self, m, r_s, units=None):
        super().__init__(units, {'m': m, 'r_s': r_s})
    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz,t):
        r_squared = xyz[0]**2 + xyz[1]**2 + xyz[2]**2
        return -self._G*self.m / jnp.sqrt(r_squared + self.r_s**2)

class HernquistPotential(Potential):
    def __init__(self, m, r_s, units=None):
        super().__init__(units, {'m': m, 'r_s': r_s})
    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz,t):
        r = jnp.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2 + 0.00005)
        return -self._G*self.m / (r + self.r_s) 
class ProgenitorPotential(Potential):
    def __init__(self, m, r_s, interp_func, units=None):
        super().__init__(units, {'m': m, 'r_s': r_s, 'interp_func':interp_func})
        self.prog_pot = PlummerPotential(m=self.m,r_s=self.r_s,units=units)
    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz,t):
        eval_pt = xyz - self.interp_func.evaluate(t)[:3]
        return self.prog_pot.potential(eval_pt,t)



    
class BarPotential(Potential):
    """
    Rotating bar potentil, with hard-coded rotation.
    Eq 8a in https://articles.adsabs.harvard.edu/pdf/1992ApJ...397...44L
    Rz according to https://en.wikipedia.org/wiki/Rotation_matrix
    """
    def __init__(self, m, a, b, c, Omega, units=None):
        super().__init__(units, {'m': m, 'a': a, 'b': b, 'c': c, 'Omega': Omega})
    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz,t):
        ## First take the simulation frame coordinates and rotate them by Omega*t
        ang = -self.Omega*t
        Rot_mat = jnp.array([[jnp.cos(ang), -jnp.sin(ang), 0], [jnp.sin(ang), jnp.cos(ang), 0.], [0.0, 0.0, 1.0] ])
        Rot_inv = jnp.linalg.inv(Rot_mat)
        xyz_corot = jnp.matmul(Rot_mat,xyz)
        
        T_plus = jnp.sqrt( (self.a + xyz_corot[0])**2 + xyz_corot[1]**2 + ( self.b + jnp.sqrt(self.c**2 + xyz_corot[2]**2) )**2 )
        T_minus = jnp.sqrt( (self.a - xyz_corot[0])**2 + xyz_corot[1]**2 + ( self.b + jnp.sqrt(self.c**2 + xyz_corot[2]**2) )**2 )
        
        pot_corot_frame = (self._G*self.m/(2.0*self.a))*jnp.log( (xyz_corot[0] - self.a + T_minus)/(xyz_corot[0] + self.a + T_plus) )
        return pot_corot_frame

class DehnenBarPotential(Potential):
    def __init__(self, alpha, v0, R0, Rb, phib, Omega, units=None):
        super().__init__(units, {'alpha':alpha,'v0':v0, 'R0':R0, 'Rb':Rb, 'phib':phib, 'Omega':Omega})

    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz,t):
        phi = jnp.arctan2(xyz[1],xyz[0])
        R = jnp.sqrt(xyz[0]**2 + xyz[1]**2)
        r = jnp.sqrt(jnp.sum(xyz**2))

        def U_func(r):
            def gtr_func():
                return -(r / self.Rb)**(-3)
            def less_func():
                return (r/self.Rb)**3 - 2.0
            bool_eval = r >= self.Rb
            return jax.lax.cond(bool_eval, gtr_func, less_func)
        
        U_eval = U_func(r)
        prefacs = self.alpha*( (self.v0**2)/3 )*( (self.R0 / self.Rb)**3 )
        pot_eval = prefacs*((R**2/r**2))*U_eval*jnp.cos(2*(phi - self.phib - self.Omega*t))
        return pot_eval

class PowerLawCutoffPotential(Potential):
    """
    Galpy potential, following the implementation from gala
    galpy source: https://github.com/jobovy/galpy/blob/main/galpy/potential/PowerSphericalPotentialwCutoff.py
    gala source: https://github.com/adrn/gala/blob/main/gala/potential/potential/builtin/builtin_potentials.c

    """
    def __init__(self, m, alpha, r_c, units=None):
        super().__init__(units, {'m':m,'alpha':alpha,'r_c':r_c})
        self.gradient = self.gradient_func

    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz,t):
        r = jnp.sqrt(jnp.sum(xyz**2))
        tmp_0 = (1/2.)*self.alpha
        tmp_1 = -tmp_0
        tmp_2 = tmp_1 + 1.5
        tmp_3 = r**2
        tmp_4 = tmp_3/self.r_c**2
        tmp_5 = self._G*self.m
        tmp_6 = tmp_5*special.gammainc(tmp_2,tmp_4)*special.gamma(tmp_2)/(jnp.sqrt(tmp_3)*special.gamma(tmp_1 + 2.5))
        return tmp_0*tmp_6 - 3./2.0*tmp_6 + tmp_5*special.gammainc(tmp_1 + 1, tmp_4)*special.gamma(tmp_1 + 1)/(self.r_c*special.gamma(tmp_2))
    
    @partial(jax.jit,static_argnums=(0,))
    def gradient_func(self, xyz, t):
        r = jnp.sqrt(jnp.sum(xyz**2))
        dPhi_dr = (self._G*self.m/(r**2) * 
                    special.gammainc(0.5*(3-self.alpha), r*r/(self.r_c*self.r_c))    )
        grad0 = dPhi_dr * xyz[0] / r
        grad1 = dPhi_dr * xyz[1] / r
        grad2 = dPhi_dr * xyz[2] / r
        return jnp.array([grad0, grad1, grad2])
    
class GalaMilkyWayPotential(Potential):
    def __init__(self,units=None):
        super().__init__(units,{'params':None})
        #Disk: Miytamoto-nagai
        self.m_disk = 6.80e10
        self.a_disk = 3.0
        self.b_disk = 0.28

        #Bulge: HernquistPotential
        self.m_bulge = 5e9
        self.c_bulge = 1.0

        #Nucleus: HernquistPotential
        self.m_nucleus = 1.71e9
        self.c_nucleus = 0.07

        #Halo: NFWPotential
        self.m_halo = 5.4e11
        self.r_s_halo = 15.62
        pot_disk = MiyamotoNagaiDisk(m=self.m_disk, a=self.a_disk, b=self.b_disk, units=self.units)
        pot_bulge = HernquistPotential(m=self.m_bulge, r_s=self.c_bulge, units=self.units)
        pot_nucleus = HernquistPotential(m=self.m_nucleus, r_s=self.c_nucleus, units=self.units)
        pot_halo = NFWPotential(m=self.m_halo, r_s=self.r_s_halo, units=self.units)
        potential_list = [pot_disk,pot_bulge, pot_nucleus, pot_halo]
        self.pot = Potential_Combine(potential_list=potential_list,units=self.units)

    @partial(jax.jit,static_argnums=(0,))
    def potential(self, xyz,t):
        return self.pot.potential(xyz,t)

class BovyMWPotential2014(Potential):
    def __init__(self,units=None):
        super().__init__(units,{'params':None})
        #Disk: Miytamoto-nagai
        self.m_disk = 6.82e10
        self.a_disk = 3.0
        self.b_disk = 0.28

        #Bulge: HernquistPotential
        self.m_bulge = 4.50e9
        self.alpha_bulge = 1.80
        self.r_c_bulge = 1.90


        #Halo: NFWPotential
        self.m_halo = 4.37e11
        self.r_s_halo = 16.0

        pot_disk = MiyamotoNagaiDisk(m=self.m_disk, a=self.a_disk, b=self.b_disk, units=self.units)
        pot_bulge = PowerLawCutoffPotential(m=self.m_bulge, alpha=self.alpha_bulge, r_c=self.r_c_bulge, units=self.units)
        pot_halo = NFWPotential(m=self.m_halo, r_s=self.r_s_halo, units=self.units)
        potential_list = [pot_disk,pot_bulge,pot_halo]
        self.pot = Potential_Combine(potential_list=potential_list,units=self.units)

    @partial(jax.jit,static_argnums=(0,))
    def potential(self, xyz,t):
        return self.pot.potential(xyz,t)



########################## SUBHALOS ###########################

class SubhaloLinePotential(Potential):
    def __init__(self, m, a, subhalo_x0, subhalo_v, subhalo_t0, t_window, units=None):
        super().__init__(units, {'m': m, 'a': a, 'subhalo_x0': subhalo_x0, 'subhalo_v': subhalo_v, 
        'subhalo_t0':subhalo_t0, 't_window':t_window,})
    

    @partial(jax.jit,static_argnums=(0,))
    def single_subhalo_potential(self, xyz, m, a, t):
        return PlummerPotential(m=m, r_s=a,units=usys).potential(xyz,t) ##Was NFWPotential

    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz, t):
        """
        xyz is where we want to evalaute the potential due to the ensemble of subhalos
        t is evaluation time.
        """
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v*(t - subhalo_t0))
            pot_values = self.single_subhalo_potential(relative_position, m,a,t)#pot_all_subhalos_func(relative_position,self.m,self.a,t)
            return pot_values
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window # True if in window, false otherwise

        vmapped_cond = jax.vmap(jax.lax.cond,in_axes=((0,None,None,0,0,0,0,0,None)))
        pot_per_subhalo = vmapped_cond(pred,true_func,false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, self.m, self.a, t)#jax.lax.cond(pred, true_func, false_func)
        return jnp.sum(pot_per_subhalo)

    @partial(jax.jit,static_argnums=(0,))
    def potential_per_SH(self,xyz, t):
        """
        xyz is where we want to evalaute the potential due to the ensemble of subhalos
        t is evaluation time.
        """
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v*(t - subhalo_t0))
            pot_values = self.single_subhalo_potential(relative_position, m,a,t)#pot_all_subhalos_func(relative_position,self.m,self.a,t)
            return pot_values
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window # True if in window, false otherwise

        vmapped_cond = jax.vmap(jax.lax.cond,in_axes=((0,None,None,0,0,0,0,0,None)))
        pot_per_subhalo = vmapped_cond(pred,true_func,false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, self.m, self.a, t)#jax.lax.cond(pred, true_func, false_func)
        return pot_per_subhalo

class SubhaloLinePotential_dRadius(Potential):
    def __init__(self, m, a, subhalo_x0, subhalo_v, subhalo_t0, t_window, units=None):
        super().__init__(units, {'m': m, 'a': a, 'subhalo_x0': subhalo_x0, 'subhalo_v': subhalo_v, 'subhalo_t0':subhalo_t0, 't_window':t_window})
    """
    Strategy is to use d/dtheta( dphi/dx )  = d/dx( dphi / dtheta ). Assuming theta is 1d, dphi/dtheta is a new potential that we will use to obtain
    the radius corrections to the EOM.
    For many structural parameters (multivariate) need to take a jacobian, but the same principle will apply.
    """

    @partial(jax.jit,static_argnums=(0,))
    def single_subhalo_potential(self, xyz, m, a, t):
        func = lambda m, r_s: PlummerPotential(m=m, r_s=r_s,units=usys).potential(xyz,t)
        return jax.grad(func,argnums=(1))(m, a) # returns gradient of potential with respect to scale radius. Output is still a 1d potential evaluation (scalar)

    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz, t):
        """
        xyz is where we want to evalaute the potential due to the ensemble of subhalos
        t is evaluation time.
        """
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v*(t - subhalo_t0))
            pot_values = self.single_subhalo_potential(relative_position, m,a,t)#pot_all_subhalos_func(relative_position,self.m,self.a,t)
            return pot_values
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window # True if in window, false otherwise

        vmapped_cond = jax.vmap(jax.lax.cond,in_axes=((0,None,None,0,0,0,0,0,None)))
        pot_per_subhalo = vmapped_cond(pred,true_func,false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, self.m, self.a, t)#jax.lax.cond(pred, true_func, false_func)
        return jnp.sum(pot_per_subhalo)

    @partial(jax.jit,static_argnums=(0,))
    def potential_per_SH(self,xyz, t):
        """
        xyz is where we want to evalaute the potential due to the ensemble of subhalos
        t is evaluation time.
        """
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v*(t - subhalo_t0))
            pot_values = self.single_subhalo_potential(relative_position, m,a,t)#pot_all_subhalos_func(relative_position,self.m,self.a,t)
            return pot_values
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window # True if in window, false otherwise

        vmapped_cond = jax.vmap(jax.lax.cond,in_axes=((0,None,None,0,0,0,0,0,None)))
        pot_per_subhalo = vmapped_cond(pred,true_func,false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, self.m, self.a, t)#jax.lax.cond(pred, true_func, false_func)
        return pot_per_subhalo
    

        
################ HELPER FUNCTIONS ##################
@jax.jit
def interp_func(t,ind,stream_func):
    arr, narr = eqx.partition(stream_func, eqx.is_array)
    arr = jax.tree_util.tree_map(lambda x: x[ind], arr)
    interp = eqx.combine(arr, narr)
    w0_at_t = interp.evaluate(t)
    return w0_at_t

class Potential_Combine(Potential):
    def __init__(self, potential_list, units=None):
        super().__init__(units, {'potential_list': potential_list })
        self.gradient = self.gradient_func

    @partial(jax.jit,static_argnums=(0,))
    def potential(self, xyz, t,):
        output = []
        for i in range(len(self.potential_list)):
            output.append(self.potential_list[i].potential(xyz,t))
        return jnp.sum(jnp.array(output))

    @partial(jax.jit,static_argnums=(0,))
    def gradient_func(self, xyz, t,):
        output = []
        for i in range(len(self.potential_list)):
            output.append(self.potential_list[i].gradient(xyz,t))
        return jnp.sum( jnp.array(output), axis = 0)
