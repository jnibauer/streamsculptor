import fields
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
import jaxopt
import optax

from . import fields
from streamsculptor.fields import integrate_field

class RestrictedNbody_generator:
    def __init__(self, potential=None, progenitor_potential=None, interp_prog=None, init_mass=None, init_rs=None, r_esc=1.0, maxiter=250, lr=1e-3):
        """
        Class to generate a restricted N-body model of a stream.
        Similar to AGAMA's restricted N-body model
        The stream is modelled as a progenitor with a user-supplied potential.
        Currently, the progenitor potential must have parameters 'm' and 'r_s'(mass, scale radius)
        Inputs:
        potential: Potential object representing the external potential
        progenitor_potential: Potential object representing the progenitor potential
        interp_prog: InterpolatedUnivariateSpline object representing the progenitor's orbit
        init_mass: Initial guess for the progenitor mass
        init_rs: Initial guess for the progenitor scale radius
        """
        self.potential = potential
        self.progenitor_potential = progenitor_potential
        self.interp_prog = interp_prog
        self.r_esc = r_esc
        self.lr = lr
        opt = optax.adam(learning_rate=lr)
        self.init_mass = init_mass
        self.init_rs = init_rs
        self.solver = jaxopt.OptaxSolver(opt=opt, fun=self.cost_func, maxiter=maxiter)

        self.pot_prog_curr = self.progenitor_potential(m=init_mass, r_s=init_rs, units=self.potential.units)

    @eqx.filter_jit
    def cost_func(self, params ,locs, t, inside_bool):
        mass_param, r_s_param = 10**params
        pot_prog_curr = self.progenitor_potential(m=mass_param, r_s=r_s_param, units=self.potential.units)
        density_at_locs = jax.vmap(pot_prog_curr.density,in_axes=(0,None))(locs,t)
        log_density = jnp.where(inside_bool, jnp.log(density_at_locs), 0.0)
        log_like = -mass_param + jnp.sum(log_density)
        return -log_like

    @eqx.filter_jit
    def fit_monopole(self, x, t, inside_bool):
        """
        Approximate self gravity of the progenitor
        Returns mass, radius
        """
        def mass_left():
            init_mass = self.init_mass
            init_rs = self.init_rs
            init_params = jnp.array([jnp.log10(init_mass), jnp.log10(init_rs)])
            m, rs =  10**self.solver.run(init_params, locs=x, t=t, inside_bool=inside_bool).params
            return jnp.array([m, rs])
        def dissolved():
            return jnp.array([0.0, 1.0])
        mass_left_bool = inside_bool.sum() > 0
        return jax.lax.cond(mass_left_bool, mass_left, dissolved)

    #@eqx.filter_jit
    @partial(jax.jit, static_argnums=(0,))
    def get_params(self,t, coords, args):
        """
        coords: N_particles x 6
        """
        x, v = coords[:,:3], coords[:,3:]
        prog_center = self.interp_prog.evaluate(t)[:3]
        x_rel = x - prog_center
        r_rel = jnp.sqrt(jnp.sum(x_rel**2, axis=1))
        inside_bool = r_rel < self.r_esc
        mass_fit, r_s_fit = self.fit_monopole(x_rel, t, inside_bool)
       
        return mass_fit, r_s_fit
    
    #@eqx.filter_jit
    @partial(jax.jit, static_argnums=(0,))
    def term(self,t, coords, args):
        """
        coords: N_particles x 6
        """
        x, v = coords[:,:3], coords[:,3:]
        acceleration_external = -jax.vmap(self.potential.gradient,in_axes=(0,None))(x,t)
        prog_center = self.interp_prog.evaluate(t)[:3]
        x_rel = x - prog_center
        r_rel = jnp.sqrt(jnp.sum(x_rel**2, axis=1))
        
        acceleration_internal = -jax.vmap(self.pot_prog_curr.gradient,in_axes=(0,None))(x_rel,t)
        acceleration = acceleration_external + acceleration_internal

        return jnp.hstack([v,acceleration])


def initialize_prog_params(w0=None, t0=None, field=None, maxiter=5_000):
    """
    Initialize progenitor parameters
    field is a RestrictedNbody_generator object
    Returns mass, r_s after optimzation
    """
    init_state = RestrictedNbody_generator(potential=field.potential, progenitor_potential=field.progenitor_potential, interp_prog=field.interp_prog, r_esc=field.r_esc, init_mass=field.init_mass, init_rs=field.init_rs, maxiter=maxiter)
    mass_curr, r_s_curr = init_state.get_params(t=t0, coords=w0, args=None)
    return mass_curr, r_s_curr


def integrate_restricted_Nbody(w0=None,ts=None, interrupt_ts=None,solver=diffrax.Dopri8(scan_kind='bounded'),field=None, args=None, rtol=1e-7, atol=1e-7, dtmin=0.05, dtmax=None, maxiter=5, max_steps=1_000, mass_init=None, r_s_init=None):
    """
    Integrates a restricted N-body model of a stream
    w0: initial conditions
    ts: length 2 arrary with: ts[0] is inital time, ts[1] is final time
    interrupt_ts: array of times at which to update the progenitor parameters and return model state
    field: RestrictedNbody_generator object
    maxiter: maximum number of iterations for optimization of progenitor parameters at each interrupt_ts
    mass_init: initial guess for progenitor mass
    r_s_init: initial guess for progenitor scale radius
    """
    @eqx.filter_jit
    def body_func(carry, idx):
        wcurr, tcurr, tstop, param_mass, param_rs =  carry
        ts_curr = jnp.array([tcurr, jnp.clip(tstop, -jnp.inf, ts[-1])])
        curr_state = RestrictedNbody_generator(potential=field.potential, progenitor_potential=field.progenitor_potential, interp_prog=field.interp_prog, r_esc=field.r_esc, init_mass=param_mass, init_rs=param_rs, maxiter=maxiter)
        mass_curr, r_s_curr = curr_state.get_params(t=tcurr, coords=wcurr, args=None)
        #update field with new params
        new_field = RestrictedNbody_generator(potential=field.potential, progenitor_potential=field.progenitor_potential, interp_prog=field.interp_prog, r_esc=field.r_esc, init_mass=mass_curr, init_rs=r_s_curr)
        #integrate field with new params
        w_at_tstop = integrate_field(w0=wcurr,ts=ts_curr, solver=solver,field=new_field, args=args, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps).ys[-1]
        return [w_at_tstop, tstop, interrupt_ts[idx+1], mass_curr, r_s_curr], [tstop, mass_curr, r_s_curr, w_at_tstop]

    interrupt_ts = jnp.hstack([interrupt_ts, ts.max()])    
    init_carry = [w0, ts[0], interrupt_ts[0], mass_init, r_s_init]
    ids = jnp.arange(len(interrupt_ts))
    final_state, all_states = jax.lax.scan(body_func, init_carry,ids)
    return all_states


