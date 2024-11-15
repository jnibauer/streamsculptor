### Put all perturbation theory potentials here
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
usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)
from StreamSculptor import Potential


class GenerateMassRadiusPerturbations(Potential):
    """
    Class to define a perturbation object.
    potiential_base: potential function for the base potential, in H_base
    potential_perturbation: gravitation potential of the perturbation(s)
    potential_structural: defined as the derivative of the perturbing potential wrspct to the structural parameter.
    dPhi_alpha / dstructural. For instance, dPhi_alpha(x, t) / dr.
    """
    def __init__(self, potential_base, potential_perturbation, potential_structural, units=None):
        super().__init__(units,{'potential_base':potential_base, 'potential_perturbation':potential_perturbation, 'potential_structural':potential_structural})
        self.gradient = None
        self.gradientPotentialBase = potential_base.gradient
        self.gradientPotentialPerturbation = potential_perturbation.gradient
        self.gradientPotentialStructural = potential_structural.gradient

        self.gradientPotentialPerturbation_per_SH = jax.jit(jax.jacfwd(potential_perturbation.potential_per_SH))
        self.gradientPotentialStructural_per_SH = jax.jit(jax.jacfwd(potential_structural.potential_per_SH))
        
    @partial(jax.jit,static_argnums=(0,))
    def potential(self, xyz, t,):
        raise NotImplementedError


class GenerateMassPerturbation(Potential):
    """
    Class to define a perturbation object, **at fixed subhalo radius**
    potiential_base: potential function for the base potential, in H_base
    potential_perturbation: gravitation potential of the perturbation(s)
    """
    def __init__(self, potential_base, potential_perturbation, units=None):
        super().__init__(units,{'potential_base':potential_base, 'potential_perturbation':potential_perturbation, 'potential_structural':potential_structural})
        self.gradient = None
        self.gradientPotentialBase = potential_base.gradient
        self.gradientPotentialPerturbation = potential_perturbation.gradient

        self.gradientPotentialPerturbation_per_SH = jax.jit(jax.jacfwd(potential_perturbation.potential_per_SH))
        
    @partial(jax.jit,static_argnums=(0,))
    def potential(self, xyz, t,):
        raise NotImplementedError