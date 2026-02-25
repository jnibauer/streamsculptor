import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import agama
import interpax
from astropy import units as u
from typing import Any

from streamsculptor.main import Potential, usys

jax.config.update("jax_enable_x64", True)

# Set AGAMA units
agama.setUnits(length=1, velocity=1, mass=1)
timeUnitMyr = agama.getUnits()['time']

class AGAMA_Spheroid(Potential):
    densityNorm: float
    scaleRadius: float
    outerCutoffRadius: float
    cutoffStrength: float
    gamma: float
    beta: float
    alpha: float
    mass: float
    rgrid: jnp.ndarray
    spl_pot_func: Any 
    

    agama_params: dict = eqx.field(static=True) 

    def __init__(self, type='Spheroid', densityNorm=None, scaleRadius=1.0, outerCutoffRadius=100.0, cutoffStrength=2.0, gamma=1.0, beta=3.0, alpha=1.0, rgrid=None, mass=None, units=usys):
        super().__init__(units)
        
        if rgrid is None:
            rgrid = jnp.logspace(-6, 3, 1000)
            
        self.rgrid = jnp.asarray(rgrid)
        self.scaleRadius = float(scaleRadius)
        self.outerCutoffRadius = float(outerCutoffRadius)
        self.cutoffStrength = float(cutoffStrength)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.alpha = float(alpha)
        
        # ---------------------------------------------------------------------
        # Pre-compute the unnormalized mass integral to resolve rho0/mass.
        # This replaces the need to dynamically assign lambda functions to self,
        # which breaks JAX PyTree serialization.
        # ---------------------------------------------------------------------
        r_eval = jnp.logspace(-5, jnp.log10(5 * self.outerCutoffRadius), 10_000)
        r_over_rs = r_eval / self.scaleRadius
        unnorm_density = (r_over_rs**(-self.gamma)) * (1 + r_over_rs**self.alpha)**((self.gamma - self.beta) / self.alpha) * jnp.exp(-(r_eval / self.outerCutoffRadius)**self.cutoffStrength)
        integral = jnp.trapezoid(unnorm_density * 4 * jnp.pi * r_eval**2, x=r_eval)
        
        if densityNorm is None and mass is not None:
            self.mass = float(mass)
            self.densityNorm = float(self.mass / integral)
        elif mass is None and densityNorm is not None:
            self.densityNorm = float(densityNorm)
            self.mass = float(self.densityNorm * integral)
        elif mass is not None and densityNorm is not None:
            self.mass = float(mass)
            self.densityNorm = float(densityNorm)
        else:
            # Fallback if neither is provided
            self.mass = 1.0
            self.densityNorm = float(self.mass / integral)

        self.agama_params = dict(
            type='Spheroid',
            densityNorm=self.densityNorm,
            scaleRadius=self.scaleRadius,
            outerCutoffRadius=self.outerCutoffRadius,
            cutoffStrength=self.cutoffStrength,
            gamma=self.gamma,
            beta=self.beta,
            alpha=self.alpha
        )
        
        # Generate AGAMA potential in standard Python/Numpy
        pot_agama = agama.Potential(self.agama_params)
        
        # AGAMA requires numpy arrays, not jax arrays
        zeros = np.zeros_like(np.array(self.rgrid))[:, None]
        inp = np.hstack([np.array(self.rgrid)[:, None], zeros, zeros])
        
        # Compute and convert units
        conversion_factor = ((u.km / u.s)**2).to(u.kpc**2 / u.Myr**2)
        pot_target = pot_agama.potential(inp) * conversion_factor
        
        # Convert the numpy array back to a JAX array for the Interpolator
        self.spl_pot_func = interpax.Interpolator1D(x=self.rgrid, f=jnp.array(pot_target), method='monotonic')

    @eqx.filter_jit
    def potential(self, xyz, t):
        r = jnp.sqrt(jnp.sum(xyz**2))
        return self.spl_pot_func(r)

    @eqx.filter_jit
    def density(self, xyz, t):
        """
        Replaces the old dynamically assigned lambda function with a standard, 
        JIT-compatible class method.
        """
        r = jnp.sqrt(jnp.sum(xyz**2))
        r_over_rs = r / self.scaleRadius
        return self.densityNorm * (r_over_rs**(-self.gamma)) * (1 + r_over_rs**self.alpha)**((self.gamma - self.beta) / self.alpha) * jnp.exp(-(r / self.outerCutoffRadius)**self.cutoffStrength)
 