from streamsculptor import potential
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

@jax.jit
def eval_dense_stream(t_eval=None, dense_stream=None):
    """
    Evaluate dense interpolation of stream model. Returns leading and trailing arm at time t_eval.
    Must supply dense_stream – an interpolation of the stream model.
    """
    output = jax.vmap(jax.vmap(lambda s: s.evaluate(t_eval)))(dense_stream)
    return output[:,0,:], output[:,1,:] #lead, trail

@jax.jit
def eval_dense_stream_id(time=None, interp_func=None, idx=None, lead=True):
    """
    Evaluate the trajectory of a dense interpolation stream, returning only the 
    trajectory of particle with index label idx. 
    When lead = True, the leading arm is evaluated.
    When lead = False, the trailing arm is evaluated.
    """
    def lead_func():
        arr, narr = eqx.partition(interp_func, eqx.is_array)
        #index x[idx, 0] is lead, x[idx, 1] is trail
        arr = jax.tree_util.tree_map(lambda x: x[idx,0], arr)
        interp = eqx.combine(arr, narr)
        return interp.evaluate(time)
    def trail_func():
        arr, narr = eqx.partition(interp_func, eqx.is_array)
        #index x[idx, 0] is lead, x[idx, 1] is trail
        arr = jax.tree_util.tree_map(lambda x: x[idx,1], arr)
        interp = eqx.combine(arr, narr)
        return interp.evaluate(time)
    
    return jax.lax.cond(lead, lead_func, trail_func)

@partial(jax.jit,static_argnums=(0,1,6))
def gen_stream_ics_pert(pot_base=None, pot_pert=None, ts=None, prog_w0=None, Msat=None, seed_num=None, solver=diffrax.Dopri5(scan_kind='bounded'),kval_arr=1.0,**kwargs):
    """
    Generate stream initial conditions for the case of direct impacts or near misses with massive subhalos.
    This function exists purely for numerical reasons: when computing the particle spray release function,
    we need to compute the jacobi radius, which depends on two deriv of the potential.
    Numerical, with a massive near/direct subhalo passage this derivative can lead to nans.
    To avoid nans, we use this function to compute the release conditions for the particles
    in the smooth potential specified by pot_base, while the trajectory is in the total potential (with perturbations),
    specified by pot_base + pot_pert.
    """
    pot_total_lst = [pot_base, pot_pert]
    pot_total = potential.Potential_Combine(potential_list=pot_total_lst, units=usys)
    # Integrate progenitor in full potential, base + perturbation
    ws_jax = pot_total.integrate_orbit(w0=prog_w0,ts=ts,solver=solver, **kwargs).ys
    Msat = Msat*jnp.ones(len(ts))
    
    def scan_fun(carry, t):
        # compute release model derivs (tidal tensor along prog's orbit) in base potential only, to avoid numerical errors with perturbation flyby
        i, pos_close, pos_far, vel_close, vel_far = carry
        pos_close_new, pos_far_new, vel_close_new, vel_far_new = pot_base.release_model(x=ws_jax[i,:3], v=ws_jax[i,3:], Msat=Msat[i],i=i, t=t, seed_num=seed_num, kval_arr=kval_arr)
        return [i+1, pos_close_new, pos_far_new, vel_close_new, vel_far_new], [pos_close_new, pos_far_new, vel_close_new, vel_far_new]
        
        
    init_carry = [0, jnp.array([0.0,0.0,0.]), jnp.array([0.0,0.0,0.]), jnp.array([0.0,0.0,0.]), jnp.array([0.0,0.0,0.])] 
    final_state, all_states = jax.lax.scan(scan_fun, init_carry, ts)
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = all_states
    return pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr

@partial(jax.jit,static_argnums=((0,1,6)))
def gen_stream_vmapped_with_pert(pot_base=None, pot_pert=None, ts=None, prog_w0=None, Msat=None, seed_num=None, solver=diffrax.Dopri5(scan_kind='bounded'), kval_arr=1.0, **kwargs):
    """
    Generate perturbed stream with vmap. Better for GPU usage.
    """
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = gen_stream_ics_pert(pot_base=pot_base, pot_pert=pot_pert, ts=ts, prog_w0=prog_w0, Msat=Msat, seed_num=seed_num, solver=solver,kval_arr=kval_arr,**kwargs)
    pot_total_lst = [pot_base, pot_pert]
    pot_total = potential.Potential_Combine(potential_list=pot_total_lst, units=usys)
    orb_integrator = lambda w0, ts: pot_total.integrate_orbit(w0=w0, ts=ts, solver=solver, **kwargs).ys[-1]
    orb_integrator_mapped = jax.jit(jax.vmap(orb_integrator,in_axes=(0,None,)))
    @jax.jit
    def single_particle_integrate(particle_number,pos_close_curr,pos_far_curr,vel_close_curr,vel_far_curr):
        curr_particle_w0_close = jnp.hstack([pos_close_curr,vel_close_curr])
        curr_particle_w0_far = jnp.hstack([pos_far_curr,vel_far_curr])
        t_release = ts[particle_number]
        t_final = ts[-1]
        ts_arr = jnp.array([t_release,t_final])
        
        curr_particle_loc = jnp.vstack([curr_particle_w0_close,curr_particle_w0_far])
        w_particle = orb_integrator_mapped(curr_particle_loc, ts_arr)
        w_particle_close = w_particle[0]
        w_particle_far =   w_particle[1]

        return w_particle_close, w_particle_far
    particle_ids = jnp.arange(len(pos_close_arr)-1)
    
    return jax.vmap(single_particle_integrate,in_axes=(0,0,0,0,0,))(particle_ids,pos_close_arr[:-1], pos_far_arr[:-1], vel_close_arr[:-1], 
    vel_far_arr[:-1])


@partial(jax.jit,static_argnums=((0,1,6)))
def gen_stream_scan_with_pert(pot_base=None, pot_pert=None, ts=None, prog_w0=None, Msat=None, seed_num=None, solver=diffrax.Dopri5(scan_kind='bounded') ,kval_arr=1.0, **kwargs):
    """
    Generate perturbed stream with scan. Better for CPU usage.
    """
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = gen_stream_ics_pert(pot_base=pot_base, pot_pert=pot_pert, ts=ts, prog_w0=prog_w0, Msat=Msat, seed_num=seed_num, solver=solver,kval_arr=kval_arr, **kwargs)
    pot_total_lst = [pot_base, pot_pert]
    pot_total = potential.Potential_Combine(potential_list=pot_total_lst, units=usys)
    orb_integrator = lambda w0, ts: pot_total.integrate_orbit(w0=w0, ts=ts, solver=solver, **kwargs).ys[-1]
    orb_integrator_mapped = jax.jit(jax.vmap(orb_integrator,in_axes=(0,None,)))

    @jax.jit
    def scan_fun(carry, particle_idx):
        i, pos_close_curr, pos_far_curr, vel_close_curr, vel_far_curr = carry
        curr_particle_w0_close = jnp.hstack([pos_close_curr,vel_close_curr])
        curr_particle_w0_far = jnp.hstack([pos_far_curr,vel_far_curr])

        ts_arr = jnp.array([ts[i],ts[-1]])
        curr_particle_loc = jnp.vstack([curr_particle_w0_close,curr_particle_w0_far])
        w_particle = orb_integrator_mapped(curr_particle_loc, ts_arr)

        w_particle_close = w_particle[0]
        w_particle_far =   w_particle[1]
        
        return [i+1, pos_close_arr[i+1,:], pos_far_arr[i+1,:], vel_close_arr[i+1,:], vel_far_arr[i+1,:]], [w_particle_close, w_particle_far]
    init_carry = [0, pos_close_arr[0,:], pos_far_arr[0,:], vel_close_arr[0,:], vel_far_arr[0,:]]

    particle_ids = jnp.arange(len(pos_close_arr)-1)
    final_state, all_states = jax.lax.scan(scan_fun, init_carry, particle_ids)
    lead_arm, trail_arm = all_states
    
    return lead_arm, trail_arm



@jax.jit
def custom_release_model(pos_prog=None, vel_prog=None, pos_rel=None, vel_rel=None):
    """
    Custom release model for the stream.
    all inputs are length 3 arrays or shape N x 3 
    pos_prog: 3d position of progenitor 
    vel_prog: 3d velocity of progenitor
    pos_rel: 3d position of released particle w/ origin on progenitor
    vel_rel: 3d velocity of released particle w/ origin on progenitor
    """
    pos_init = pos_prog + pos_rel
    vel_init = vel_prog + vel_rel
    return pos_init, vel_init