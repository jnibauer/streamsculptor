from . import potential
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
from .interpolation import *


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

#@eqx.filter_jit
@partial(jax.jit,static_argnames=('pot_base','pot_pert','solver','max_steps','rtol','atol','dtmin'))
def gen_stream_ics_pert(pot_base=None, pot_pert=None, ts=None, prog_w0=None, Msat=None, seed_num=None, solver=diffrax.Dopri5(scan_kind='bounded'),kval_arr=1.0,max_steps=10_000,rtol=1e-7,atol=1e-7,dtmin=0.1):
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
    ws_jax = pot_total.integrate_orbit(w0=prog_w0,ts=ts,solver=solver,max_steps=max_steps, rtol=rtol, atol=atol, dtmin=dtmin).ys
    Msat = Msat*jnp.ones(len(ts))

    @jax.jit
    def body_func(i):
        """
        body function to vmap over
        """
        pos_close_new, pos_far_new, vel_close_new, vel_far_new = pot_base.release_model(x=ws_jax[i,:3], v=ws_jax[i,3:], Msat=Msat[i],i=i, t=ts[i], seed_num=seed_num, kval_arr=kval_arr)
        return [pos_close_new, pos_far_new, vel_close_new, vel_far_new]

    iterator_arange = jnp.arange(len(ts))
    all_states = jax.vmap(body_func)(iterator_arange)
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = all_states
    return pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr
    


#@eqx.filter_jit
@partial(jax.jit,static_argnames=('pot_base','pot_pert','solver','max_steps','rtol','atol','dtmin'))
def gen_stream_vmapped_with_pert(pot_base=None, pot_pert=None, ts=None, prog_w0=None, Msat=None, seed_num=None, solver=diffrax.Dopri5(scan_kind='bounded'), kval_arr=1.0,max_steps=10_000,rtol=1e-7,atol=1e-7,dtmin=0.1):
    """
    Generate perturbed stream with vmap. Better for GPU usage.
    """
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = gen_stream_ics_pert(pot_base=pot_base, pot_pert=pot_pert, ts=ts, prog_w0=prog_w0, Msat=Msat, seed_num=seed_num, solver=solver,kval_arr=kval_arr,max_steps=max_steps,rtol=rtol,atol=atol,dtmin=dtmin)
    pot_total_lst = [pot_base, pot_pert]
    pot_total = potential.Potential_Combine(potential_list=pot_total_lst, units=usys)
    orb_integrator = lambda w0, ts: pot_total.integrate_orbit(w0=w0, ts=ts, solver=solver,max_steps=max_steps, rtol=rtol, atol=atol, dtmin=dtmin).ys[-1]
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

#@eqx.filter_jit
@partial(jax.jit,static_argnames=('pot_base','pot_pert','solver','max_steps','rtol','atol','dtmin'))
def gen_stream_vmapped_with_pert_fixed_prog(pot_base=None, pot_pert=None, ts=None, prog_w0=None, Msat=None, seed_num=None, solver=diffrax.Dopri5(scan_kind='bounded'), kval_arr=1.0,max_steps=10_000,rtol=1e-7,atol=1e-7,dtmin=0.1):
    """
    Generate perturbed stream with vmap. Assume the progenitor's orbit is unperturbed, i.e. Bovy+2017 J0 progenitor is invariant to perturbations. This simplfies impact sampling.
    """
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = pot_base.gen_stream_ics(ts=ts, prog_w0=prog_w0, Msat=Msat, seed_num=seed_num, solver=solver, kval_arr=kval_arr, max_steps=max_steps, rtol=rtol, atol=atol, dtmin=dtmin)
    pot_total_lst = [pot_base, pot_pert]
    pot_total = potential.Potential_Combine(potential_list=pot_total_lst, units=usys)
    orb_integrator = lambda w0, ts: pot_total.integrate_orbit(w0=w0, ts=ts, solver=solver,max_steps=max_steps, rtol=rtol, atol=atol, dtmin=dtmin).ys[-1]
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



#@eqx.filter_jit
@partial(jax.jit,static_argnames=('pot_base','pot_pert','solver','max_steps','rtol','atol','dtmin'))
def gen_stream_scan_with_pert(pot_base=None, pot_pert=None, ts=None, prog_w0=None, Msat=None, seed_num=None, solver=diffrax.Dopri5(scan_kind='bounded') ,kval_arr=1.0, max_steps=10_000, rtol=1e-7, atol=1e-7, dtmin=0.1):
    """
    Generate perturbed stream with scan. Better for CPU usage.
    """
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = gen_stream_ics_pert(pot_base=pot_base, pot_pert=pot_pert, ts=ts, prog_w0=prog_w0, Msat=Msat, seed_num=seed_num, solver=solver,kval_arr=kval_arr,max_steps=max_steps,rtol=rtol,atol=atol,dtmin=dtmin)
    pot_total_lst = [pot_base, pot_pert]
    pot_total = potential.Potential_Combine(potential_list=pot_total_lst, units=usys)
    orb_integrator = lambda w0, ts: pot_total.integrate_orbit(w0=w0, ts=ts, solver=solver, max_steps=max_steps, rtol=rtol, atol=atol, dtmin=dtmin).ys[-1]
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


@partial(jax.jit,static_argnums=(2,))
def computed_binned_track(stream: jnp.ndarray, phi1: jnp.ndarray, bins=20):
    """
    Compute the binned track of the stream based on the phase-space coordinates.
    Assumes stream is in the form of a 2D array with shape (N, 6) where N is the number of particles.
    phi1 is a 1D array of the same length as the number of particles in the stream.
    """
    # bin the stream in phi1 and compute the mean phase-space position in phi1 bins
    phi1_bins = jnp.linspace(jnp.min(phi1), jnp.max(phi1), bins)
    phi1_digitized = jnp.digitize(phi1, phi1_bins)

    # compute mean pos in each bin by scanning over bins
    def compute_bin_mean(carry, bin_idx):
        stream, phi1_digitized = carry
        mask = phi1_digitized == bin_idx
        bin_count = mask.sum()
        # Filter the elements using jnp.where
        masked_stream = jnp.where(mask[:, None], stream, jnp.nan)
        bin_mean = jnp.where(bin_count > 0, jnp.nanmean(masked_stream, axis=0), jnp.zeros(6))
        return carry, bin_mean

    _, bin_means = jax.lax.scan(
        compute_bin_mean, 
        (stream, phi1_digitized), 
        jnp.arange(len(phi1_bins) - 1)
    )
    bin_means = jnp.where(bin_means == 0, jnp.nan, bin_means)   
    return bin_means

@partial(jax.jit,static_argnums=(2,))
def compute_stream_length(stream: jnp.ndarray, phi1: jnp.ndarray, bins=20):
    """
    Compute the length of the stream based on the phase-space coordinates.
    Assumes stream is in the form of a 2D array with shape (N, 6) where N is the number of particles.
    phi1 is a 1D array of the same length as the number of particles in the stream.
    """
    bin_means = computed_binned_track(stream, phi1, bins)
    # Calculate the segment lengths between consecutive bin means (position only)
    segment_lengths = jnp.linalg.norm(bin_means[:,:3][1:] - bin_means[:,:3][:-1], axis=1)

    # Return the total length of the stream
    return jnp.nansum(segment_lengths)


@partial(jax.jit, static_argnames=["pot"])
def compute_length_oscillations(pot: callable,
                                prog_today: jnp.array,
                                first_stripped_lead: jnp.array, 
                                first_stripped_trail: jnp.array,
                                t_age: float,
                                length_today: float,
                                ):
    """
    Compute the time evolution of a stream's length due to orbital oscillations.

    Integrates the orbits of the progenitor and its leading and trailing
    stripped components backward in time within a specified gravitational potential.
    It then estimates the oscillatory behavior of the stream's length over time, 
    normalized by the present-day stream length.

    Parameters
    ----------
    pot : callable
        A potential object with an `integrate_orbit` method, used to evolve phase-space coordinates.
    prog_today : jnp.array
        The current phase-space coordinates of the progenitor (6-element array).
    first_stripped_lead : jnp.array
        The current phase-space coordinates of the first leading stripped star.
    first_stripped_trail : jnp.array
        The current phase-space coordinates of the first trailing stripped star.
    t_age : float
        Total time (in the past) to integrate orbits over, typically the age of the stream [same time units as pot].
    length_today : float
        Present-day length of the stream [same spatial units as used in pot].

    Returns
    -------
    dict
        A dictionary containing:
        - `ts` : jnp.array
            Time array (reversed, from -t_age to 0), shape (2000,)
        - `length_func` : jnp.array
            Stream length as a function of time, normalized to present-day length. Shape (2000,)

    Notes
    -----
    The stream length at each time step is approximated as the Euclidean distance between 
    the progenitor and its leading and trailing stripped components, added in quadrature.
    """
    ts = jnp.linspace(0,-t_age,2_000)
    l_back = pot.integrate_orbit(w0=first_stripped_lead, t0=0, t1=-t_age, ts=ts)
    t_back = pot.integrate_orbit(w0=first_stripped_trail, t0=0, t1=-t_age, ts=ts)
    prog_back = pot.integrate_orbit(w0=prog_today,t0=0, t1=-t_age, ts=ts)
    diff = jnp.sqrt(jnp.sum( (l_back.ys[:,:3] - prog_back.ys[:,:3])**2 + (t_back.ys[:,:3] - prog_back.ys[:,:3])**2, axis=1 ) )
    
    
    diff_flip = jnp.flip(diff)
    ts_flip = jnp.flip(l_back.ts)

    amplitude_osc = diff_flip / diff_flip[-1]

    length_func = length_today * amplitude_osc
    
    return dict(ts=ts_flip, length_func=length_func)

@partial(jax.jit,static_argnames=["num_samples"])
def sample_from_1D_pdf(x: jnp.array, y: jnp.array, key: jax.random.PRNGKey, num_samples: int):
    """
    Sample random values from a 1D probability distribution using the inverse CDF method.

    Parameters
    ----------
    x : jax.Array
        1D array of shape (n,) representing the x-axis values. Must be sorted in ascending order.
    y : jax.Array
        1D array of shape (n,) representing the function values at each `x`. Values should be non-negative.
    key : jax.random.PRNGKey
        JAX PRNG key used for random number generation.
    num_samples : int
        Number of random samples to generate.

    Returns
    -------
    jax.Array
        1D array of shape (num_samples,) containing values sampled from the probability distribution
        defined by `x` and `y`.

    Notes
    -----
    - The input `y` is automatically normalized to form a valid probability density function (PDF).
    - The cumulative distribution function (CDF) is computed numerically, and its inverse is
      approximated using linear interpolation.
    - `x` should densely cover the domain of the distribution for accurate sampling.
    """
    # Step 1: Normalize y to make it a PDF
    pdf = y / jnp.trapezoid(y, x)

    # Step 2: Compute the CDF
    cdf = jnp.cumsum(pdf)
    cdf = cdf / cdf[-1]  # Normalize to 1

    # Step 3: Create inverse CDF by interpolation
    def inverse_cdf(u):
        return jnp.interp(u, cdf, x)
    
    # Step 4: Sample uniform values and map through inverse CDF
    u = jax.random.uniform(key, shape=(num_samples,))
    samples = inverse_cdf(u)
    return samples


@partial(jax.jit,static_argnums=(0,))
def release_model_Chen25(   pot_base: callable,
                            key: jax.random.PRNGKey,
                            x: jnp.ndarray,
                            v: jnp.ndarray,
                            Msat: float,
                            t: float,
                            ):
    
    r_tidal = pot_base.tidalr(x,v,Msat,t)
    mean = jnp.array([1.6, -30, 0, 1, 20, 0])
    cov = jnp.array([
        [0.1225,   0,   0, 0, -4.9,   0],
        [     0, 529,   0, 0,    0,   0],
        [     0,   0, 144, 0,    0,   0],
        [     0,   0,   0, 0,    0,   0],
        [  -4.9,   0,   0, 0,  400,   0],
        [     0,   0,   0, 0,    0, 484],
    ])

    # x_new-hat
    r = jnp.linalg.norm(x)
    x_new_hat = x / r

    # z_new-hat
    L_vec = jnp.cross(x, v)
    z_new_hat = L_vec / jnp.linalg.norm(L_vec)

    # y_new-hat
    phi_vec = v - jnp.sum(v * x_new_hat) * x_new_hat
    y_new_hat = phi_vec / jnp.linalg.norm(phi_vec)

    posvel = jax.random.multivariate_normal(
            key, mean, cov, shape=r_tidal.shape, method="svd"
        )
    
    
    Dr = posvel.at[0].get() * r_tidal

    v_esc = jnp.sqrt(2 * pot_base._G * Msat / Dr)
    Dv = posvel.at[3].get() * v_esc

    # convert degrees to radians
    phi = posvel.at[1].get() * 0.017453292519943295
    theta = posvel.at[2].get() * 0.017453292519943295
    alpha = posvel.at[4].get() * 0.017453292519943295
    beta = posvel.at[5].get() * 0.017453292519943295

    ctheta, stheta = jnp.cos(theta), jnp.sin(theta)
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    calpha, salpha = jnp.cos(alpha), jnp.sin(alpha)
    cbeta, sbeta = jnp.cos(beta), jnp.sin(beta)
    # Trailing arm
    x_trail = (
        x
        + (Dr * ctheta * cphi) * x_new_hat
        + (Dr * ctheta * sphi) * y_new_hat
        + (Dr * stheta) * z_new_hat
    )
    v_trail = (
        v
        + (Dv * cbeta * calpha) * x_new_hat
        + (Dv * cbeta * salpha) * y_new_hat
        + (Dv * sbeta) * z_new_hat
        )
    
    # Leading arm
    x_lead = (
        x
        - (Dr * ctheta * cphi) * x_new_hat
        - (Dr * ctheta * sphi) * y_new_hat
        + (Dr * stheta) * z_new_hat
    )
    v_lead = (
        v
        - (Dv * cbeta * calpha) * x_new_hat
        - (Dv * cbeta * salpha) * y_new_hat
        + (Dv * sbeta) * z_new_hat
    )

    return x_lead, x_trail, v_lead, v_trail

@partial(jax.jit,static_argnames=('pot_base','solver','rtol','atol','max_steps'))
def gen_stream_ics_Chen25(pot_base=None, ts=None, prog_w0=None, Msat=None, key=None, solver=diffrax.Dopri5(scan_kind='bounded'), rtol=1e-7, atol=1e-7, dtmin=0.3,dtmax=None,max_steps=10_000):
    ws_jax_out = pot_base.integrate_orbit(w0=prog_w0,ts=ts,solver=solver, rtol=rtol, atol=atol, dtmin=dtmin,dtmax=dtmax,max_steps=max_steps)
    ws_jax = ws_jax_out.ys
    Msat = Msat*jnp.ones(len(ts))

    @jax.jit
    def body_func(i, key):
        """
        body function to vmap over
        """
        pos_close_new, pos_far_new, vel_close_new, vel_far_new = release_model_Chen25(      pot_base=pot_base,
                                                                                            x=ws_jax[i,:3],
                                                                                            v=ws_jax[i,3:],
                                                                                            Msat=Msat[i], 
                                                                                            t=ts[i], 
                                                                                            key=key)
        return [pos_close_new, pos_far_new, vel_close_new, vel_far_new]
    
        
    iterator_arange = jnp.arange(len(ts))
    keys = jax.random.split(key, len(ts))
    all_states = jax.vmap(body_func)(iterator_arange, keys)
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = all_states

    return [pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr], ws_jax_out

@partial(jax.jit,static_argnames=('pot_base','pot_pert','solver','rtol','atol','max_steps'))
def gen_stream_ics_pert_Chen25(pot_base=None, pot_pert=None, ts=None, prog_w0=None, Msat=None, key=None, solver=diffrax.Dopri5(scan_kind='bounded'), rtol=1e-7, atol=1e-7, dtmin=0.3,dtmax=None,max_steps=10_000):
    pot_total_lst = [pot_base, pot_pert]
    pot_total = potential.Potential_Combine(potential_list=pot_total_lst, units=usys)
    
    ws_jax_out = pot_total.integrate_orbit(w0=prog_w0,ts=ts,solver=solver, rtol=rtol, atol=atol, dtmin=dtmin,dtmax=dtmax,max_steps=max_steps)
    ws_jax = ws_jax_out.ys
    Msat = Msat*jnp.ones(len(ts))

    @jax.jit
    def body_func(i, key):
        """
        body function to vmap over
        """
        pos_close_new, pos_far_new, vel_close_new, vel_far_new = release_model_Chen25(      pot_base=pot_base,
                                                                                            x=ws_jax[i,:3],
                                                                                            v=ws_jax[i,3:],
                                                                                            Msat=Msat[i], 
                                                                                            t=ts[i], 
                                                                                            key=key)
        return [pos_close_new, pos_far_new, vel_close_new, vel_far_new]
    
        
    iterator_arange = jnp.arange(len(ts))
    keys = jax.random.split(key, len(ts))
    all_states = jax.vmap(body_func)(iterator_arange, keys)
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = all_states

    return [pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr], ws_jax_out

@partial(jax.jit,static_argnames=('pot_base','solver','rtol','atol','max_steps', 'throw','prog_pot'))
def gen_stream_vmapped_Chen25(  pot_base: callable, 
                                ts: jnp.array, 
                                prog_w0: jnp.array, 
                                Msat: float, 
                                key: jax.random.PRNGKey, 
                                solver=diffrax.Dopri5(scan_kind='bounded'),
                                rtol=1e-7, 
                                atol=1e-7, 
                                dtmin=0.3,
                                dtmax=None,
                                max_steps=10_000, 
                                throw=False,
                                prog_pot=potential.PlummerPotential(m=0.0, r_s=1.0,units=usys)):
    """
    Generate stellar stream by vmapping over the release model/integration. Better for GPU usage.
    """
    stream_ics, orb_fwd = gen_stream_ics_Chen25(pot_base=pot_base, ts=ts, prog_w0=prog_w0, Msat=Msat, key=key, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps)

    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = stream_ics
    
    # Interpolate progenitor forward. This is pointless when prog_pot is Null (m=0), but will not break anything. 
    prog_spline = UniformCubicInterpolator(orb_fwd.ts, orb_fwd.ys[:,:3])
    prog_pot_translating = potential.TimeDepTranslatingPotential(pot=prog_pot, center_spl=prog_spline, units=usys)
    pot_tot = potential.Potential_Combine(potential_list=[pot_base, prog_pot_translating], units=usys)
    
    orb_integrator = lambda w0, ts: pot_tot.integrate_orbit(w0=w0, ts=ts, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin,dtmax=dtmax,max_steps=max_steps, throw=throw).ys[-1]
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

@partial(jax.jit,static_argnames=('pot_base','pot_pert','prog_pot','solver','max_steps','rtol','atol','dtmin'))
def gen_stream_vmapped_with_pert_Chen25(pot_base=None, 
                                        pot_pert=None, 
                                        prog_pot=potential.PlummerPotential(m=0.0, r_s=1.0,units=usys),
                                        ts=None, 
                                        prog_w0=None, 
                                        Msat=None, 
                                        key=None, 
                                        solver=diffrax.Dopri5(scan_kind='bounded'),
                                        max_steps=10_000,
                                        rtol=1e-7,
                                        atol=1e-7,
                                        dtmin=0.1):
    """
    Generate perturbed stream with vmap. Better for GPU usage.
    """
    stream_ics, orb_fwd = gen_stream_ics_pert_Chen25(pot_base=pot_base, pot_pert=pot_pert, ts=ts, prog_w0=prog_w0, Msat=Msat, key=key, solver=solver,max_steps=max_steps,rtol=rtol,atol=atol,dtmin=dtmin)
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = stream_ics
    
    # Interpolate progenitor forward. This is pointless when prog_pot is Null (m=0), but will not break anything. 
    prog_spline = UniformCubicInterpolator(orb_fwd.ts, orb_fwd.ys[:,:3])
    prog_pot_translating = potential.TimeDepTranslatingPotential(pot=prog_pot, center_spl=prog_spline, units=usys)
    pot_total = potential.Potential_Combine(potential_list=[pot_base, pot_pert, prog_pot_translating], units=usys)
    # Integrate progenitor in full potential: base + prog + perturbation
    orb_integrator = lambda w0, ts: pot_total.integrate_orbit(w0=w0, ts=ts, solver=solver,max_steps=max_steps, rtol=rtol, atol=atol, dtmin=dtmin).ys[-1]
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


@partial(jax.jit,static_argnames=('pot_base','pot_pert','prog_pot','solver','max_steps','rtol','atol','dtmin'))
def gen_stream_vmapped_with_pert_Chen25_fixed_prog(pot_base=None, 
                                        pot_pert=None, 
                                        prog_pot=potential.PlummerPotential(m=0.0, r_s=1.0,units=usys),
                                        ts=None, 
                                        prog_w0=None, 
                                        Msat=None, 
                                        key=None, 
                                        solver=diffrax.Dopri5(scan_kind='bounded'),
                                        max_steps=10_000,
                                        rtol=1e-7,
                                        atol=1e-7,
                                        dtmin=0.1):
    """
    Generate perturbed stream with vmap. Better for GPU usage.
    Progenitor's orbit is unperturbed, i.e. Bovy+2017 J0 progenitor is invariant to perturbations. This simplfies impact sampling.
    """
    stream_ics, orb_fwd = gen_stream_ics_Chen25(pot_base=pot_base, ts=ts, prog_w0=prog_w0, Msat=Msat, key=key, solver=solver,max_steps=max_steps,rtol=rtol,atol=atol,dtmin=dtmin)
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = stream_ics
    
    # Interpolate progenitor forward. This is pointless when prog_pot is Null (m=0), but will not break anything. 
    prog_spline = UniformCubicInterpolator(orb_fwd.ts, orb_fwd.ys[:,:3])
    prog_pot_translating = potential.TimeDepTranslatingPotential(pot=prog_pot, center_spl=prog_spline, units=usys)
    pot_total = potential.Potential_Combine(potential_list=[pot_base, pot_pert, prog_pot_translating], units=usys)
    # Integrate progenitor in full potential: base + prog + perturbation
    orb_integrator = lambda w0, ts: pot_total.integrate_orbit(w0=w0, ts=ts, solver=solver,max_steps=max_steps, rtol=rtol, atol=atol, dtmin=dtmin).ys[-1]
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



########## Streakline model #########
def get_Streakline_ICs(pot: callable, 
                       prog_w0: jnp.array,
                       Msat: float,
                       t0: float, 
                       t1: float, 
                       Nstrip: int, 
                       solver=diffrax.Dopri8(), 
                       rtol=1e-6,
                       atol=1e-6):
    """
    Generate initial conditions for a streakline stellar stream,
    by releasing particles from L1, L2 with the same angular velocity (omega)
    as the parent cluster.
    """
    ts = jnp.linspace(t0, t1, Nstrip)
    prog_orb = pot.integrate_orbit(w0=prog_w0, ts=ts,solver=solver, rtol=rtol, atol=atol)
    
    L_close, L_far = jax.vmap(pot.lagrange_pts,in_axes=(0,0,None,0))(prog_orb.ys[:,:3],prog_orb.ys[:,3:],Msat, prog_orb.ts) # each is an xyz array

    omega_val = jax.vmap(pot.omega)(prog_orb.ys[:,:3],prog_orb.ys[:,3:])

    r = jnp.sqrt(jnp.sum(prog_orb.ys[:,:3]**2,axis=1)) 
    r_hat = prog_orb.ys[:,:3]/r[:,None]
    r_tidal = jax.vmap(pot.tidalr,in_axes=(0,0,None,0))(prog_orb.ys[:,:3],prog_orb.ys[:,3:],Msat, prog_orb.ts)

    v_hat = prog_orb.ys[:,3:] / jnp.linalg.norm(prog_orb.ys[:,3:], axis=1)[:,None]

    @partial(jax.vmap)
    def release_velocity(q_rel, q_prog, p_prog, omega):
        r_prog_hat = q_prog/jnp.sqrt(jnp.sum(q_prog**2))
        p_prog_hat = p_prog/jnp.sqrt(jnp.sum(p_prog**2))
        sintheta = jnp.linalg.norm(jnp.cross(r_prog_hat,p_prog_hat))
        
        
        p_rel = omega*jnp.sqrt(jnp.sum(q_rel**2)) / sintheta
        vel_out = p_rel*p_prog_hat
        return vel_out

    v_lead = release_velocity(L_close, prog_orb.ys[:,:3], prog_orb.ys[:,3:], omega_val)
    v_trail =release_velocity(L_far, prog_orb.ys[:,:3], prog_orb.ys[:,3:], omega_val)

    return L_close, v_lead, L_far, v_trail, ts

    
@partial(jax.jit,static_argnums=(0,5))
def gen_streakline( pot: callable, 
                    prog_w0: jnp.array,
                    Msat: float,
                    t0: float, 
                    t1: float, 
                    Nstrip: int, 
                    solver=diffrax.Dopri8(), 
                    rtol=1e-6,
                    atol=1e-6):
    """
    Generate streakline stellar stream by realisng particles from L1, L2
    with the same angular velocity (omega) as the parent cluster.
    """
    
    pos_lead, vel_lead, pos_trail, vel_trail, tstrip = get_Streakline_ICs(
                       pot=pot, 
                       prog_w0=prog_w0,
                       Msat=Msat,
                       t0=t0,
                       t1=t1,
                       Nstrip=Nstrip,
                       solver=solver,
                       rtol=rtol,
                       atol=atol
                       )
    orb_func = lambda w0, t0: pot.integrate_orbit(w0=w0, t0=t0, t1=t1, solver=solver, ts=jnp.array([t1]), rtol=rtol, atol=atol).ys[0]
    w0_lead = jnp.hstack([ pos_lead, vel_lead ])
    w0_trail = jnp.hstack([ pos_trail, vel_trail ])
    mapped_orb_func = jax.vmap(orb_func)
    lead = mapped_orb_func(w0_lead, tstrip)
    trail = mapped_orb_func(w0_trail, tstrip)
    return lead, trail, tstrip
