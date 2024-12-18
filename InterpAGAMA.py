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
# Necessary for the interpolation
import agama
import interpax
import numpy as np

# all agama dimensional quantities are given in the units of 1 kpc, 1 km/s, 1 Msun;
agama.setUnits(length=1, velocity=1, mass=1)
timeUnitMyr = agama.getUnits()['time']

class AGAMA_Spheroid(Potential):
    def __init__(self,densityNorm=None, scaleRadius=None, outerCutoffRadius=None, cutoffStrength=None, gamma=None, beta=None, rgrid=jnp.logspace(-6,3,1000),units=None):
        super().__init__(units,{'densityNorm':densityNorm, 'scaleRadius':scaleRadius, 'outerCutoffRadius':outerCutoffRadius, 'cutoffStrength':cutoffStrength, 'gamma':gamma, 'beta':beta, 'rgrid':rgrid})
      
        self.agama_params  = dict(
        type              = 'Spheroid',
        densityNorm       = self.densityNorm,
        scaleRadius       = self.scaleRadius,
        outerCutoffRadius = self.outerCutoffRadius,
        cutoffStrength    = self.cutoffStrength,
        gamma             = self.gamma,
        beta              = self.beta)
        self.pot_agama = agama.Potential(self.agama_params)
        zeros = np.zeros_like(self.rgrid)[:,None]
        inp = np.hstack([self.rgrid[:, None], zeros, zeros])
        self.pot_target = self.pot_agama.potential(inp)*( (u.km/u.s)**2 ).to(u.kpc**2/u.Myr**2)
        self.spl_pot_func  = interpax.Interpolator1D(x=self.rgrid, f=jnp.array(self.pot_target),method='monotonic')


    @partial(jax.jit, static_argnums=(0,))
    def potential(self, xyz, t):
        r = jnp.sqrt(jnp.sum(xyz**2))
        return self.spl_pot_func(r)
