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
from .InterpAGAMA import AGAMA_Spheroid
import interpax
import os



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

    
    @partial(jax.jit,static_argnums=(0,))
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

class TriaxialNFWPotential(Potential):
    """
    Flattening in the potential, not the density.
    """
    def __init__(self, m, r_s, q1=1.0, q2=1.0, q3=1.0, units=None):
        super().__init__(units, {'m': m, 'r_s': r_s, 'q1': q1, 'q2': q2, 'q3': q3})
    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz,t):
        xyz = jnp.array([xyz[0]/self.q1,xyz[1]/self.q2,xyz[2]/self.q3])
        v_h2 = -self._G*self.m/self.r_s
        m = jnp.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2 )/self.r_s ##removed softening! used to be .001 after xyz[2]**2
        return v_h2*jnp.log(1.0+ m) / m


class Isochrone(Potential):
    
    def __init__(self, m, a, units=None):
        super().__init__(units, {'m': m, 'a': a})
    
    @partial(jax.jit,static_argnums=(0,))
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
    """
    Progenitor potential centered on a moving spline-interpolated track.
    prog_pot is the functional form of the potential, e.g., PlummerPotential
    Must take mass and scale radius parameters: m, r_s
    interp_func is a diffrax interpolated solution to the progenitor's trajectory
    """
    def __init__(self, m, r_s, interp_func, prog_pot, units=None):
        super().__init__(units, {'m': m, 'r_s': r_s, 'interp_func':interp_func, 'prog_pot':prog_pot})
        self.prog_pot = prog_pot(m=self.m,r_s=self.r_s,units=units)
    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz,t):
        eval_pt = xyz - self.interp_func.evaluate(t)[:3]
        return self.prog_pot.potential(eval_pt,t)

class TimeDepProgenitorPotential(Potential):
    """
    Time dependent progenitor potential in the location, mass, and scale-radius of the progenitor
    prog_pot is the functional form of the potential, e.g., PlummerPotential
    Must take mass and scale radius parameters: m, r_s
    mass_spl and r_s_spl are spline-interpolated functions that take a single argument [time]
    and output a scalar [mass, radius]
    interp_func is a diffrax interpolated solution to the progenitor's trajectory
    """
    def __init__(self, mass_spl, r_s_spl, interp_func, prog_pot, units=None):
        super().__init__(units, {'mass_spl': mass_spl, 'r_s_spl': r_s_spl, 'interp_func':interp_func, 'prog_pot':prog_pot})
    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz,t):
        eval_pt = xyz - self.interp_func.evaluate(t)[:3]
        mass_curr = self.mass_spl(t)
        r_s_curr = self.r_s_spl(t)
        pot_curr = self.prog_pot(m=mass_curr,r_s=r_s_curr,units=self.units)
        return pot_curr.potential(eval_pt,t)




    
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

class TimeDepTranslatingPotential(Potential):
    """
    Time dependent potential that translates with a spline-interpolated track.
    pot: potential object
    center_spl: Jax differentiable spline-interpolated track of the center of the potential. Must take a single argument [time]
    --> center_spl(t) returns the center of the potential at time t
    """
    def __init__(self, pot, center_spl, units=None):
        super().__init__(units,{'pot':pot, 'center_spl':center_spl})

    @partial(jax.jit,static_argnums=(0,))
    def potential(self, xyz, t):
        center = self.center_spl(t)
        xyz_adjust = xyz - center
        return self.pot.potential(xyz_adjust,t)

class GrowingPotential(Potential):
    """
    Time dependent potential that grows with a time-dependent growth factor.
    Inputs include
    pot: potential object
    growth_func: a function that takes a single argument [time] and returns a scalar growth factor
    """
    def __init__(self, pot, growth_func, units=None):
        super().__init__(units,{'pot':pot, 'growth_func':growth_func})
    
    @partial(jax.jit, static_argnums=(0,))
    def potential(self, xyz, t):
        growth_factor = self.growth_func(t)
        return self.pot.potential(xyz,t) * growth_factor


class UniformAcceleration(Potential):
    """
    Spatially uniform acceleration field
    """
    def __init__(self, velocity_func=None, units=None):
        """
        velocity_func: spline function that takes a single argument [time] and returns a 3d vector [vx,vy,vz] in kpc/Myr
        Derivative of this function is the acceleration of the frame
        Minus the derivative is the exterted spatially uniform acceleration
        """
        super().__init__(units,{'velocity_func':velocity_func})
        #self.gradient = gradient
        #self.acceleration = acceleration

    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz, t):
        raise NotImplementedError
    @partial(jax.jit,static_argnums=(0,))
    def gradient(self,xyz,t):
        return jax.jacfwd(self.velocity_func)(t)
    @partial(jax.jit,static_argnums=(0,))
    def acceleration(self,xyz,t):
        return -self.gradient(xyz, t)
    

class MW_LMC_Potential(Potential):
    """
    This potential is implemented from the AGAMA script: https://github.com/GalacticDynamics-Oxford/Agama/blob/c507fc3e703513ae4a41bb705e171a4d036754a8/py/example_lmc_mw_interaction.py
    Approximation for the Milky Way and LMC potentials, evolving in time as rigid bodies.
    This "potential" is non-conservative, so only the force is implemented.
    The LMC experiences chandrasekhar dynamical friction, with a spatially dependent drag force (velocity dispersion compute form MW potential using AGAMA)
    For interactive notebook implementation, see examples/mw_lmc.ipynb
    Crucially, force field assumes we are in the non-inertial frame of the Milky Way.
    Therefore, integration must be done in the non-inertial frame. 
    This means the MW's reflex motion will be incorporated in all integrated velocity vectors
    Must integrate from a negative time > -14_000 Myr ago to present day (t=0).

    Caution: all splines are C2 (i.e., twice differentiable). Only a single round of automatic diffrentation should be applied 
    to orbits in this potential. 
    TODO: use higher-order splines.
    """
    def __init__(self, units=None):
        super().__init__(units,{'params':None})
        # Load the data for MW and LMC motion
        data_path_MW = os.path.join(os.path.dirname(__file__), 'data/LMC_MW_potential', 'MW_motion_dict.npy')
        data_path_LMC = os.path.join(os.path.dirname(__file__), 'data/LMC_MW_potential', 'LMC_motion_dict.npy')
        MW_motion_dict = jnp.load(data_path_MW, allow_pickle=True).item()
        LMC_motion_dict = jnp.load(data_path_LMC, allow_pickle=True).item()
        
        # LMC spatial track
        self.LMC_x = interpax.Interpolator1D(x=LMC_motion_dict['flip_tsave'], f=LMC_motion_dict['flip_trajLMC'][:,0], method='cubic2')
        self.LMC_y = interpax.Interpolator1D(x=LMC_motion_dict['flip_tsave'], f=LMC_motion_dict['flip_trajLMC'][:,1], method='cubic2')
        self.LMC_z = interpax.Interpolator1D(x=LMC_motion_dict['flip_tsave'], f=LMC_motion_dict['flip_trajLMC'][:,2], method='cubic2')

        # MW velocity track
        self.velocity_func_x = interpax.Interpolator1D(x=MW_motion_dict['flip_tsave'], f=MW_motion_dict['flip_traj'][:,3], method='cubic2')
        self.velocity_func_y = interpax.Interpolator1D(x=MW_motion_dict['flip_tsave'], f=MW_motion_dict['flip_traj'][:,4], method='cubic2')
        self.velocity_func_z = interpax.Interpolator1D(x=MW_motion_dict['flip_tsave'], f=MW_motion_dict['flip_traj'][:,5], method='cubic2')

        # Create a simple but realistic model of the Milky Way with a bulge, a single disk,
        # and a spherical dark halo
        paramBulge = dict(
            type              = 'Spheroid',
            mass              = 1.2e10,
            scaleRadius       = 0.2,
            outerCutoffRadius = 1.8,
            gamma             = 0.0,
            beta              = 1.8)
        paramDisk  = dict(
            type='MiyamotoNagai',
            mass              = 5.0e10,
            scaleRadius       = 3.0,
            scaleHeight       = 0.3)
        paramHalo  = dict(
            type              = 'Spheroid',
            densityNorm       = 1.35e7,
            scaleRadius       = 14,
            outerCutoffRadius = 300,
            cutoffStrength    = 4,
            gamma             = 1,
            beta              = 3)

        # LMC params
        massLMC    = 1.5e11
        radiusLMC  = (massLMC/1e11)**0.6 * 8.5
        paramLMC = dict(
            type              = 'spheroid',
            mass              = massLMC,
            scaleRadius       = radiusLMC,
            outerCutoffRadius = radiusLMC*10,
            gamma             = 1,
            beta              = 3
            )
            
        # Create the Milky Way model
        pot_bulge = AGAMA_Spheroid(**paramBulge)
        pot_disk = MiyamotoNagaiDisk(m=paramDisk['mass'], a=paramDisk['scaleRadius'], b=paramDisk['scaleHeight'], units=units)
        pot_halo = AGAMA_Spheroid(**paramHalo)
        pot_MW_lst = [pot_bulge, pot_disk, pot_halo]
        self.pot_MW = Potential_Combine(pot_MW_lst, units=units)
        # Create the LMC model
        self.pot_LMC = AGAMA_Spheroid(**paramLMC)
        self.translating_LMC_pot = TimeDepTranslatingPotential(pot=self.pot_LMC, center_spl=self.LMC_center_spline,units=units)
        # Uniform acceleration: we will assume integration in the *non-intertial* frame of the MW
        # Uniform acceleration is the negative of the derivative of the MW velocity function (MW's velocity track in intertial frame)
        self.unif_acc = UniformAcceleration(velocity_func=self.MW_velocity_func,units=units)
        pot_total_lst = [self.pot_MW, self.translating_LMC_pot, self.unif_acc]
        self.total_pot = Potential_Combine(pot_total_lst, units=units)
        
        self.gradient = self.total_pot.gradient
        self.acceleration = self.total_pot.acceleration


    @partial(jax.jit,static_argnums=(0,))
    def LMC_center_spline(self,t):
        return jnp.array([self.LMC_x(t), self.LMC_y(t), self.LMC_z(t)])
        
    @partial(jax.jit,static_argnums=(0,))
    def MW_velocity_func(self,t):
        return jnp.array([self.velocity_func_x(t), self.velocity_func_y(t), self.velocity_func_z(t)])

    @partial(jax.jit,static_argnums=(0,))
    def potential(self, xyz, t):
        # Raise error with message
        raise NotImplementedError("Potential not implemented, force is non-conservative")

 
class CustomPotential(Potential):
    """
    Class to define a custom potential function
    potential_func must take arguments (xyz, t) and return a scalar potential value
    --> def potential_func(xyz, t):
    -->     potential_value = ...   
    -->     return potential_value
    """
    def __init__(self, potential_func=None, units=None):
        super().__init__(units,{'potential_func':potential_func})
        self.potential = self.potential_func


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
    


class SubhaloLinePotential_Custom(Potential):
    def __init__(self, pot, subhalo_x0, subhalo_v, subhalo_t0, t_window, units=None):
        super().__init__(units, {'pot': pot, 'subhalo_x0': subhalo_x0, 'subhalo_v': subhalo_v, 
        'subhalo_t0':subhalo_t0, 't_window':t_window,})
    

    @partial(jax.jit,static_argnums=(0,))
    def single_subhalo_potential(self, xyz, t):
        return self.pot.potential(xyz,t) 

    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz, t):
        """
        xyz is where we want to evalaute the potential due to the ensemble of subhalos
        t is evaluation time.
        """
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v*(t - subhalo_t0))
            pot_values = self.single_subhalo_potential(relative_position,t)
            return pot_values
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, t):
            return jnp.array([0.0])

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window # True if in window, false otherwise

        vmapped_cond = jax.vmap(jax.lax.cond,in_axes=((0,None,None,0,0,0,None)))
        pot_per_subhalo = vmapped_cond(pred,true_func,false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, t)#jax.lax.cond(pred, true_func, false_func)
        return jnp.sum(pot_per_subhalo)

    @partial(jax.jit,static_argnums=(0,))
    def potential_per_SH(self,xyz, t):
        """
        xyz is where we want to evalaute the potential due to the ensemble of subhalos
        t is evaluation time.
        """
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v*(t - subhalo_t0))
            pot_values = self.single_subhalo_potential(relative_position,t)
            return pot_values
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, t):
            return jnp.array([0.0])

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window # True if in window, false otherwise

        vmapped_cond = jax.vmap(jax.lax.cond,in_axes=((0,None,None,0,0,0,None)))
        pot_per_subhalo = vmapped_cond(pred,true_func,false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, t)#jax.lax.cond(pred, true_func, false_func)
        return pot_per_subhalo

class SubhaloLinePotential_dRadius_Custom(Potential):
    def __init__(self, dpot_dRadius, subhalo_x0, subhalo_v, subhalo_t0, t_window, units=None):
        super().__init__(units, {'dpot_dRadius': dpot_dRadius, 'subhalo_x0': subhalo_x0, 'subhalo_v': subhalo_v, 
        'subhalo_t0':subhalo_t0, 't_window':t_window,})
    

    @partial(jax.jit,static_argnums=(0,))
    def single_subhalo_potential(self, xyz, t):
        return self.dpot_dRadius(xyz,t) 

    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz, t):
        """
        xyz is where we want to evalaute the potential due to the ensemble of subhalos
        t is evaluation time.
        """
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v*(t - subhalo_t0))
            pot_values = self.single_subhalo_potential(relative_position,t)
            return pot_values
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, t):
            return jnp.array([0.0])

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window # True if in window, false otherwise

        vmapped_cond = jax.vmap(jax.lax.cond,in_axes=((0,None,None,0,0,0,None)))
        pot_per_subhalo = vmapped_cond(pred,true_func,false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, t)#jax.lax.cond(pred, true_func, false_func)
        return jnp.sum(pot_per_subhalo)

    @partial(jax.jit,static_argnums=(0,))
    def potential_per_SH(self,xyz, t):
        """
        xyz is where we want to evalaute the potential due to the ensemble of subhalos
        t is evaluation time.
        """
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v*(t - subhalo_t0))
            pot_values = self.single_subhalo_potential(relative_position,t)
            return pot_values
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, t):
            return jnp.array([0.0])

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window # True if in window, false otherwise

        vmapped_cond = jax.vmap(jax.lax.cond,in_axes=((0,None,None,0,0,0,None)))
        pot_per_subhalo = vmapped_cond(pred,true_func,false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, t)#jax.lax.cond(pred, true_func, false_func)
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

