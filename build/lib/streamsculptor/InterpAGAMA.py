from functools import partial
from astropy.constants import G
import astropy.coordinates as coord
import astropy.units as u
# gala
from gala.units import dimensionless, UnitSystem

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from diffrax import diffeqsolve, ODETerm, Dopri5,SaveAt,PIDController,DiscreteTerminatingEvent, DirectAdjoint, RecursiveCheckpointAdjoint, ConstantStepSize, Euler, StepTo
import diffrax
import equinox as eqx
usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)
from streamsculptor import Potential
# Necessary for the interpolation
import agama
import interpax
import numpy as np

# all agama dimensional quantities are given in the units of 1 kpc, 1 km/s, 1 Msun;
agama.setUnits(length=1, velocity=1, mass=1)
timeUnitMyr = agama.getUnits()['time']

class AGAMA_Spheroid(Potential):
    def __init__(self,type=None,densityNorm=None, scaleRadius=None, outerCutoffRadius=None, cutoffStrength=2.0, gamma=None, beta=None, alpha=1.0, rgrid=jnp.logspace(-6,3,1000),mass=None,units=None):
        super().__init__(units,{'densityNorm':densityNorm, 'scaleRadius':scaleRadius, 'outerCutoffRadius':outerCutoffRadius, 'cutoffStrength':cutoffStrength, 'gamma':gamma, 'beta':beta,'alpha':alpha ,'rgrid':rgrid, 'mass':mass})

        if self.densityNorm is None:
            self.density = self.density_func
            self.densityNorm = self.solve_for_rho0_from_mass()

        if self.mass is None:
            self.mass = 1.0
            self.density = lambda xyz, t: self.density_func(xyz, t, self.densityNorm)
            self.mass = 1./self.solve_for_rho0_from_mass()


        self.agama_params  = dict(
        type              = 'Spheroid',
        densityNorm       = self.densityNorm,
        scaleRadius       = self.scaleRadius,
        outerCutoffRadius = self.outerCutoffRadius,
        cutoffStrength    = self.cutoffStrength,
        gamma             = self.gamma,
        beta              = self.beta,
        alpha             = self.alpha)
        self.pot_agama = agama.Potential(self.agama_params)
        zeros = np.zeros_like(self.rgrid)[:,None]
        inp = np.hstack([self.rgrid[:, None], zeros, zeros])
        self.pot_target = self.pot_agama.potential(inp)*( (u.km/u.s)**2 ).to(u.kpc**2/u.Myr**2)
        self.spl_pot_func  = interpax.Interpolator1D(x=self.rgrid, f=jnp.array(self.pot_target),method='monotonic')
        self.density = lambda xyz, t: self.density_func(xyz, t, self.densityNorm)


    @eqx.filter_jit
    def potential(self, xyz, t):
        r = jnp.sqrt(jnp.sum(xyz**2))
        return self.spl_pot_func(r)


    @eqx.filter_jit
    def density_func(self, xyz, t, densityNorm=1.0):
        r = jnp.sqrt(jnp.sum(xyz**2))
        r_over_rs = r/self.scaleRadius
        return densityNorm*( r_over_rs**(-self.gamma) )*( 1 + r_over_rs**self.alpha )**((self.gamma - self.beta)/self.alpha)*jnp.exp( -(r/self.outerCutoffRadius)**self.cutoffStrength )
       
    @eqx.filter_jit
    def solve_for_rho0_from_mass(self):
        r_grid = jnp.logspace(-5,jnp.log10(5*self.outerCutoffRadius),10_000)
        #stack the 1d array so we have 3d array with r_grid along x, zeros otherwise
        zeros = jnp.zeros_like(r_grid)[:,None]
        inp = jnp.hstack([r_grid[:, None], zeros, zeros])
        density_vals = jax.vmap(self.density,in_axes=(0,None))(inp,0.0)
        integral = jnp.trapezoid(density_vals*4*jnp.pi*r_grid**2,x=r_grid)
        rho0 = self.mass/integral
        return rho0
       



