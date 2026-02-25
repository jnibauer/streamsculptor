from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController, DiscreteTerminatingEvent, DirectAdjoint, RecursiveCheckpointAdjoint, ConstantStepSize, Euler, StepTo
import interpax
from jax.scipy.interpolate import RegularGridInterpolator

# Import from the newly refactored modules
from streamsculptor import potential
from streamsculptor.main import GalacticUnitSystem, usys

jax.config.update("jax_enable_x64", True)

@eqx.filter_jit
def eval_dense_stream(t_eval=None, dense_stream=None):
    """
    Evaluate dense interpolation of stream model. Returns leading and trailing arm at time t_eval.
    Must supply dense_stream – an interpolation of the stream model.
    """
    output = jax.vmap(jax.vmap(lambda s: s.evaluate(t_eval)))(dense_stream)
    return output[:,0,:], output[:,1,:] #lead, trail

@eqx.filter_jit
def eval_dense_stream_id(time=None, interp_func=None, idx=None, lead=True):
    """
    Evaluate the trajectory of a dense interpolation stream, returning only the 
    trajectory of particle with index label idx. 
    """
    def lead_func():
        arr, narr = eqx.partition(interp_func, eqx.is_array)
        arr = jax.tree_util.tree_map(lambda x: x[idx,0], arr)
        interp = eqx.combine(arr, narr)
        return interp.evaluate(time)
    
    def trail_func():
        arr, narr = eqx.partition(interp_func, eqx.is_array)
        arr = jax.tree_util.tree_map(lambda x: x[idx,1], arr)
        interp = eqx.combine(arr, narr)
        return interp.evaluate(time)
    
    return jax.lax.cond(lead, lead_func, trail_func)

@eqx.filter_jit
def gen_stream_ics_pert(pot_base=None, pot_pert=None, ts=None, prog_w0=None, Msat=None, seed_num=None, solver=diffrax.Dopri5(scan_kind='bounded'),kval_arr=1.0,max_steps=10_000,rtol=1e-7,atol=1e-7,dtmin=0.1):
    """
    Generate stream initial conditions for the case of direct impacts or near misses with massive subhalos.
    """
    pot_total_lst = [pot_base, pot_pert]
    pot_total = potential.Potential_Combine(potential_list=pot_total_lst, units=usys)
    
    ws_jax = pot_total.integrate_orbit(w0=prog_w0,ts=ts,solver=solver,max_steps=max_steps, rtol=rtol, atol=atol, dtmin=dtmin).ys
    Msat = Msat * jnp.ones(len(ts))

    def body_func(i):
        pos_close_new, pos_far_new, vel_close_new, vel_far_new = pot_base.release_model(x=ws_jax[i,:3], v=ws_jax[i,3:], Msat=Msat[i],i=i, t=ts[i], seed_num=seed_num, kval_arr=kval_arr)
        return [pos_close_new, pos_far_new, vel_close_new, vel_far_new]

    all_states = jax.vmap(body_func)(jnp.arange(len(ts)))
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = all_states
    return pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr
    
@eqx.filter_jit
def gen_stream_vmapped_with_pert(pot_base=None, pot_pert=None, ts=None, prog_w0=None, Msat=None, seed_num=None, solver=diffrax.Dopri5(scan_kind='bounded'), kval_arr=1.0,max_steps=10_000,rtol=1e-7,atol=1e-7,dtmin=0.1):
    """
    Generate perturbed stream with vmap. Better for GPU usage.
    """
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = gen_stream_ics_pert(pot_base=pot_base, pot_pert=pot_pert, ts=ts, prog_w0=prog_w0, Msat=Msat, seed_num=seed_num, solver=solver,kval_arr=kval_arr,max_steps=max_steps,rtol=rtol,atol=atol,dtmin=dtmin)
    pot_total_lst = [pot_base, pot_pert]
    pot_total = potential.Potential_Combine(potential_list=pot_total_lst, units=usys)
    
    orb_integrator = lambda w0, t_arr: pot_total.integrate_orbit(w0=w0, ts=t_arr, solver=solver,max_steps=max_steps, rtol=rtol, atol=atol, dtmin=dtmin).ys[-1]
    orb_integrator_mapped = jax.vmap(orb_integrator, in_axes=(0, None))
    
    def single_particle_integrate(particle_number, pos_close_curr, pos_far_curr, vel_close_curr, vel_far_curr):
        curr_particle_w0_close = jnp.hstack([pos_close_curr, vel_close_curr])
        curr_particle_w0_far = jnp.hstack([pos_far_curr, vel_far_curr])
        ts_arr = jnp.array([ts[particle_number], ts[-1]])
        
        curr_particle_loc = jnp.vstack([curr_particle_w0_close, curr_particle_w0_far])
        w_particle = orb_integrator_mapped(curr_particle_loc, ts_arr)

        return w_particle[0], w_particle[1]
    
    particle_ids = jnp.arange(len(pos_close_arr)-1)
    return jax.vmap(single_particle_integrate, in_axes=(0,0,0,0,0))(particle_ids, pos_close_arr[:-1], pos_far_arr[:-1], vel_close_arr[:-1], vel_far_arr[:-1])

@eqx.filter_jit
def gen_stream_vmapped_with_pert_fixed_prog(pot_base=None, pot_pert=None, ts=None, prog_w0=None, Msat=None, seed_num=None, solver=diffrax.Dopri5(scan_kind='bounded'), kval_arr=1.0,max_steps=10_000,rtol=1e-7,atol=1e-7,dtmin=0.1):
    """
    Generate perturbed stream with vmap. Assume the progenitor's orbit is unperturbed.
    """
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = pot_base.gen_stream_ics(ts=ts, prog_w0=prog_w0, Msat=Msat, seed_num=seed_num, solver=solver, kval_arr=kval_arr, max_steps=max_steps, rtol=rtol, atol=atol, dtmin=dtmin)
    pot_total_lst = [pot_base, pot_pert]
    pot_total = potential.Potential_Combine(potential_list=pot_total_lst, units=usys)
    
    orb_integrator = lambda w0, t_arr: pot_total.integrate_orbit(w0=w0, ts=t_arr, solver=solver,max_steps=max_steps, rtol=rtol, atol=atol, dtmin=dtmin).ys[-1]
    orb_integrator_mapped = jax.vmap(orb_integrator, in_axes=(0, None))
    
    def single_particle_integrate(particle_number, pos_close_curr, pos_far_curr, vel_close_curr, vel_far_curr):
        curr_particle_w0_close = jnp.hstack([pos_close_curr, vel_close_curr])
        curr_particle_w0_far = jnp.hstack([pos_far_curr, vel_far_curr])
        ts_arr = jnp.array([ts[particle_number], ts[-1]])
        
        curr_particle_loc = jnp.vstack([curr_particle_w0_close, curr_particle_w0_far])
        w_particle = orb_integrator_mapped(curr_particle_loc, ts_arr)

        return w_particle[0], w_particle[1]
    
    particle_ids = jnp.arange(len(pos_close_arr)-1)
    return jax.vmap(single_particle_integrate, in_axes=(0,0,0,0,0))(particle_ids, pos_close_arr[:-1], pos_far_arr[:-1], vel_close_arr[:-1], vel_far_arr[:-1])

@eqx.filter_jit
def gen_stream_scan_with_pert(pot_base=None, pot_pert=None, ts=None, prog_w0=None, Msat=None, seed_num=None, solver=diffrax.Dopri5(scan_kind='bounded') ,kval_arr=1.0, max_steps=10_000, rtol=1e-7, atol=1e-7, dtmin=0.1):
    """
    Generate perturbed stream with scan. Better for CPU usage.
    """
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = gen_stream_ics_pert(pot_base=pot_base, pot_pert=pot_pert, ts=ts, prog_w0=prog_w0, Msat=Msat, seed_num=seed_num, solver=solver,kval_arr=kval_arr,max_steps=max_steps,rtol=rtol,atol=atol,dtmin=dtmin)
    pot_total_lst = [pot_base, pot_pert]
    pot_total = potential.Potential_Combine(potential_list=pot_total_lst, units=usys)
    
    orb_integrator = lambda w0, t_arr: pot_total.integrate_orbit(w0=w0, ts=t_arr, solver=solver, max_steps=max_steps, rtol=rtol, atol=atol, dtmin=dtmin).ys[-1]
    orb_integrator_mapped = jax.vmap(orb_integrator, in_axes=(0, None))

    def scan_fun(carry, i):
        pos_close_curr, pos_far_curr, vel_close_curr, vel_far_curr = carry
        curr_particle_w0_close = jnp.hstack([pos_close_curr, vel_close_curr])
        curr_particle_w0_far = jnp.hstack([pos_far_curr, vel_far_curr])

        ts_arr = jnp.array([ts[i], ts[-1]])
        curr_particle_loc = jnp.vstack([curr_particle_w0_close, curr_particle_w0_far])
        w_particle = orb_integrator_mapped(curr_particle_loc, ts_arr)
        
        return [pos_close_arr[i+1,:], pos_far_arr[i+1,:], vel_close_arr[i+1,:], vel_far_arr[i+1,:]], [w_particle[0], w_particle[1]]
    
    init_carry = [pos_close_arr[0,:], pos_far_arr[0,:], vel_close_arr[0,:], vel_far_arr[0,:]]
    particle_ids = jnp.arange(len(pos_close_arr)-1)
    
    _, all_states = jax.lax.scan(scan_fun, init_carry, particle_ids)
    return all_states[0], all_states[1]

@eqx.filter_jit
def custom_release_model(pos_prog=None, vel_prog=None, pos_rel=None, vel_rel=None):
    """
    Custom release model for the stream.
    """
    pos_init = pos_prog + pos_rel
    vel_init = vel_prog + vel_rel
    return pos_init, vel_init

@eqx.filter_jit
def computed_binned_track(stream: jnp.ndarray, phi1: jnp.ndarray, bins=20):
    """
    Compute the binned track of the stream based on the phase-space coordinates.
    """
    phi1_bins = jnp.linspace(jnp.min(phi1), jnp.max(phi1), bins)
    phi1_digitized = jnp.digitize(phi1, phi1_bins)

    def compute_bin_mean(carry, bin_idx):
        stream, phi1_digitized = carry
        mask = phi1_digitized == bin_idx
        bin_count = mask.sum()
        masked_stream = jnp.where(mask[:, None], stream, jnp.nan)
        bin_mean = jnp.where(bin_count > 0, jnp.nanmean(masked_stream, axis=0), jnp.zeros(6))
        return carry, bin_mean

    _, bin_means = jax.lax.scan(
        compute_bin_mean, 
        (stream, phi1_digitized), 
        jnp.arange(len(phi1_bins) - 1)
    )
    return jnp.where(bin_means == 0, jnp.nan, bin_means)   

@eqx.filter_jit
def compute_stream_length(stream: jnp.ndarray, phi1: jnp.ndarray, bins=20):
    """
    Compute the length of the stream based on the phase-space coordinates.
    """
    bin_means = computed_binned_track(stream, phi1, bins)
    segment_lengths = jnp.linalg.norm(bin_means[:,:3][1:] - bin_means[:,:3][:-1], axis=1)
    return jnp.nansum(segment_lengths)

@eqx.filter_jit
def compute_length_oscillations(pot: callable, prog_today: jnp.array, first_stripped_lead: jnp.array, first_stripped_trail: jnp.array, t_age: float, length_today: float):
    """
    Compute the time evolution of a stream's length due to orbital oscillations.
    """
    ts = jnp.linspace(0, -t_age, 2_000)
    l_back = pot.integrate_orbit(w0=first_stripped_lead, t0=0, t1=-t_age, ts=ts)
    t_back = pot.integrate_orbit(w0=first_stripped_trail, t0=0, t1=-t_age, ts=ts)
    prog_back = pot.integrate_orbit(w0=prog_today, t0=0, t1=-t_age, ts=ts)
    
    diff = jnp.sqrt(jnp.sum((l_back.ys[:,:3] - prog_back.ys[:,:3])**2 + (t_back.ys[:,:3] - prog_back.ys[:,:3])**2, axis=1))
    
    diff_flip = jnp.flip(diff)
    ts_flip = jnp.flip(l_back.ts)
    amplitude_osc = diff_flip / diff_flip[-1]
    length_func = length_today * amplitude_osc
    
    return dict(ts=ts_flip, length_func=length_func)

@eqx.filter_jit
def sample_from_1D_pdf(x: jnp.array, y: jnp.array, key: jax.random.PRNGKey, num_samples: int):
    """
    Sample random values from a 1D probability distribution using the inverse CDF method.
    """
    pdf = y / jnp.trapezoid(y, x)
    cdf = jnp.cumsum(pdf)
    cdf = cdf / cdf[-1]
    
    def inverse_cdf(u):
        return jnp.interp(u, cdf, x)
    
    u = jax.random.uniform(key, shape=(num_samples,))
    return inverse_cdf(u)

@eqx.filter_jit
def release_model_Chen25(pot_base: callable, key: jax.random.PRNGKey, x: jnp.ndarray, v: jnp.ndarray, Msat: float, t: float):
    r_tidal = pot_base.tidalr(x, v, Msat, t)
    mean = jnp.array([1.6, -30, 0, 1, 20, 0])
    cov = jnp.array([
        [0.1225,   0,   0, 0, -4.9,   0],
        [     0, 529,   0, 0,    0,   0],
        [     0,   0, 144, 0,    0,   0],
        [     0,   0,   0, 0,    0,   0],
        [  -4.9,   0,   0, 0,  400,   0],
        [     0,   0,   0, 0,    0, 484],
    ])

    r = jnp.linalg.norm(x)
    x_new_hat = x / r

    L_vec = jnp.cross(x, v)
    z_new_hat = L_vec / jnp.linalg.norm(L_vec)

    phi_vec = v - jnp.sum(v * x_new_hat) * x_new_hat
    y_new_hat = phi_vec / jnp.linalg.norm(phi_vec)

    posvel = jax.random.multivariate_normal(key, mean, cov, shape=r_tidal.shape, method="svd")
    
    Dr = posvel.at[0].get() * r_tidal
    # Updated _G to units.G
    v_esc = jnp.sqrt(2 * pot_base.units.G * Msat / Dr)
    Dv = posvel.at[3].get() * v_esc

    phi = posvel.at[1].get() * 0.017453292519943295
    theta = posvel.at[2].get() * 0.017453292519943295
    alpha = posvel.at[4].get() * 0.017453292519943295
    beta = posvel.at[5].get() * 0.017453292519943295

    ctheta, stheta = jnp.cos(theta), jnp.sin(theta)
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    calpha, salpha = jnp.cos(alpha), jnp.sin(alpha)
    cbeta, sbeta = jnp.cos(beta), jnp.sin(beta)
    
    x_trail = x + (Dr * ctheta * cphi) * x_new_hat + (Dr * ctheta * sphi) * y_new_hat + (Dr * stheta) * z_new_hat
    v_trail = v + (Dv * cbeta * calpha) * x_new_hat + (Dv * cbeta * salpha) * y_new_hat + (Dv * sbeta) * z_new_hat
    
    x_lead = x - (Dr * ctheta * cphi) * x_new_hat - (Dr * ctheta * sphi) * y_new_hat + (Dr * stheta) * z_new_hat
    v_lead = v - (Dv * cbeta * calpha) * x_new_hat - (Dv * cbeta * salpha) * y_new_hat + (Dv * sbeta) * z_new_hat

    return x_lead, x_trail, v_lead, v_trail

@eqx.filter_jit
def gen_stream_ics_Chen25(pot_base=None, ts=None, prog_w0=None, Msat=None, key=None, solver=diffrax.Dopri5(scan_kind='bounded'), rtol=1e-7, atol=1e-7, dtmin=0.3,dtmax=None,max_steps=10_000):
    ws_jax_out = pot_base.integrate_orbit(w0=prog_w0,ts=ts,solver=solver, rtol=rtol, atol=atol, dtmin=dtmin,dtmax=dtmax,max_steps=max_steps)
    ws_jax = ws_jax_out.ys
    Msat = Msat * jnp.ones(len(ts))

    def body_func(i, key):
        pos_close_new, pos_far_new, vel_close_new, vel_far_new = release_model_Chen25(pot_base=pot_base, x=ws_jax[i,:3], v=ws_jax[i,3:], Msat=Msat[i], t=ts[i], key=key)
        return [pos_close_new, pos_far_new, vel_close_new, vel_far_new]
    
    iterator_arange = jnp.arange(len(ts))
    keys = jax.random.split(key, len(ts))
    all_states = jax.vmap(body_func)(iterator_arange, keys)
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = all_states

    return [pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr], ws_jax_out

@eqx.filter_jit
def gen_stream_ics_pert_Chen25(pot_base=None, pot_pert=None, ts=None, prog_w0=None, Msat=None, key=None, solver=diffrax.Dopri5(scan_kind='bounded'), rtol=1e-7, atol=1e-7, dtmin=0.3,dtmax=None,max_steps=10_000):
    pot_total_lst = [pot_base, pot_pert]
    pot_total = potential.Potential_Combine(potential_list=pot_total_lst, units=usys)
    
    ws_jax_out = pot_total.integrate_orbit(w0=prog_w0,ts=ts,solver=solver, rtol=rtol, atol=atol, dtmin=dtmin,dtmax=dtmax,max_steps=max_steps)
    ws_jax = ws_jax_out.ys
    Msat = Msat * jnp.ones(len(ts))

    def body_func(i, key):
        pos_close_new, pos_far_new, vel_close_new, vel_far_new = release_model_Chen25(pot_base=pot_base, x=ws_jax[i,:3], v=ws_jax[i,3:], Msat=Msat[i], t=ts[i], key=key)
        return [pos_close_new, pos_far_new, vel_close_new, vel_far_new]
    
    iterator_arange = jnp.arange(len(ts))
    keys = jax.random.split(key, len(ts))
    all_states = jax.vmap(body_func)(iterator_arange, keys)
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = all_states

    return [pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr], ws_jax_out

@eqx.filter_jit
def gen_stream_vmapped_Chen25(pot_base: callable, ts: jnp.array, prog_w0: jnp.array, Msat: float, key: jax.random.PRNGKey, solver=diffrax.Dopri5(scan_kind='bounded'), rtol=1e-7, atol=1e-7, dtmin=0.3, dtmax=None, max_steps=10_000, throw=False, prog_pot=potential.PlummerPotential(m=0.0, r_s=1.0,units=usys)):
    stream_ics, orb_fwd = gen_stream_ics_Chen25(pot_base=pot_base, ts=ts, prog_w0=prog_w0, Msat=Msat, key=key, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps)
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = stream_ics
    
    prog_spline = interpax.Interpolator1D(x=orb_fwd.ts, f=orb_fwd.ys[:,:3], method='cubic')
    prog_pot_translating = potential.TimeDepTranslatingPotential(pot=prog_pot, center_spl=prog_spline, units=usys)
    pot_tot = potential.Potential_Combine(potential_list=[pot_base, prog_pot_translating], units=usys)
    
    orb_integrator = lambda w0, t_arr: pot_tot.integrate_orbit(w0=w0, ts=t_arr, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin,dtmax=dtmax,max_steps=max_steps, throw=throw).ys[-1]
    orb_integrator_mapped = jax.vmap(orb_integrator, in_axes=(0, None))
    
    def single_particle_integrate(particle_number, pos_close_curr, pos_far_curr, vel_close_curr, vel_far_curr):
        curr_particle_w0_close = jnp.hstack([pos_close_curr, vel_close_curr])
        curr_particle_w0_far = jnp.hstack([pos_far_curr, vel_far_curr])
        ts_arr = jnp.array([ts[particle_number], ts[-1]])
        
        curr_particle_loc = jnp.vstack([curr_particle_w0_close, curr_particle_w0_far])
        w_particle = orb_integrator_mapped(curr_particle_loc, ts_arr)

        return w_particle[0], w_particle[1]
    
    particle_ids = jnp.arange(len(pos_close_arr)-1)
    return jax.vmap(single_particle_integrate, in_axes=(0,0,0,0,0))(particle_ids, pos_close_arr[:-1], pos_far_arr[:-1], vel_close_arr[:-1], vel_far_arr[:-1])

@eqx.filter_jit
def gen_stream_vmapped_with_pert_Chen25(pot_base=None, pot_pert=None, prog_pot=potential.PlummerPotential(m=0.0, r_s=1.0,units=usys), ts=None, prog_w0=None, Msat=None, key=None, solver=diffrax.Dopri5(scan_kind='bounded'), max_steps=10_000, rtol=1e-7, atol=1e-7, dtmin=0.1):
    stream_ics, orb_fwd = gen_stream_ics_pert_Chen25(pot_base=pot_base, pot_pert=pot_pert, ts=ts, prog_w0=prog_w0, Msat=Msat, key=key, solver=solver,max_steps=max_steps,rtol=rtol,atol=atol,dtmin=dtmin)
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = stream_ics
    
    prog_spline = interpax.Interpolator1D(x=orb_fwd.ts, f=orb_fwd.ys[:,:3], method='cubic')
    prog_pot_translating = potential.TimeDepTranslatingPotential(pot=prog_pot, center_spl=prog_spline, units=usys)
    pot_total = potential.Potential_Combine(potential_list=[pot_base, pot_pert, prog_pot_translating], units=usys)
    
    orb_integrator = lambda w0, t_arr: pot_total.integrate_orbit(w0=w0, ts=t_arr, solver=solver,max_steps=max_steps, rtol=rtol, atol=atol, dtmin=dtmin).ys[-1]
    orb_integrator_mapped = jax.vmap(orb_integrator, in_axes=(0, None))
    
    def single_particle_integrate(particle_number, pos_close_curr, pos_far_curr, vel_close_curr, vel_far_curr):
        curr_particle_w0_close = jnp.hstack([pos_close_curr, vel_close_curr])
        curr_particle_w0_far = jnp.hstack([pos_far_curr, vel_far_curr])
        ts_arr = jnp.array([ts[particle_number], ts[-1]])
        
        curr_particle_loc = jnp.vstack([curr_particle_w0_close, curr_particle_w0_far])
        w_particle = orb_integrator_mapped(curr_particle_loc, ts_arr)

        return w_particle[0], w_particle[1]
    
    particle_ids = jnp.arange(len(pos_close_arr)-1)
    return jax.vmap(single_particle_integrate, in_axes=(0,0,0,0,0))(particle_ids, pos_close_arr[:-1], pos_far_arr[:-1], vel_close_arr[:-1], vel_far_arr[:-1])

@eqx.filter_jit
def gen_stream_vmapped_with_pert_Chen25_fixed_prog(pot_base=None, pot_pert=None, prog_pot=potential.PlummerPotential(m=0.0, r_s=1.0,units=usys), ts=None, prog_w0=None, Msat=None, key=None, solver=diffrax.Dopri5(scan_kind='bounded'), max_steps=10_000, rtol=1e-7, atol=1e-7, dtmin=0.1):
    stream_ics, orb_fwd = gen_stream_ics_Chen25(pot_base=pot_base, ts=ts, prog_w0=prog_w0, Msat=Msat, key=key, solver=solver,max_steps=max_steps,rtol=rtol,atol=atol,dtmin=dtmin)
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = stream_ics
    
    prog_spline = interpax.Interpolator1D(x=orb_fwd.ts, f=orb_fwd.ys[:,:3], method='cubic')
    prog_pot_translating = potential.TimeDepTranslatingPotential(pot=prog_pot, center_spl=prog_spline, units=usys)
    pot_total = potential.Potential_Combine(potential_list=[pot_base, pot_pert, prog_pot_translating], units=usys)
    
    orb_integrator = lambda w0, t_arr: pot_total.integrate_orbit(w0=w0, ts=t_arr, solver=solver,max_steps=max_steps, rtol=rtol, atol=atol, dtmin=dtmin).ys[-1]
    orb_integrator_mapped = jax.vmap(orb_integrator, in_axes=(0, None))
    
    def single_particle_integrate(particle_number, pos_close_curr, pos_far_curr, vel_close_curr, vel_far_curr):
        curr_particle_w0_close = jnp.hstack([pos_close_curr, vel_close_curr])
        curr_particle_w0_far = jnp.hstack([pos_far_curr, vel_far_curr])
        ts_arr = jnp.array([ts[particle_number], ts[-1]])
        
        curr_particle_loc = jnp.vstack([curr_particle_w0_close, curr_particle_w0_far])
        w_particle = orb_integrator_mapped(curr_particle_loc, ts_arr)

        return w_particle[0], w_particle[1]
    
    particle_ids = jnp.arange(len(pos_close_arr)-1)
    return jax.vmap(single_particle_integrate, in_axes=(0,0,0,0,0))(particle_ids, pos_close_arr[:-1], pos_far_arr[:-1], vel_close_arr[:-1], vel_far_arr[:-1])

@eqx.filter_jit
def get_Streakline_ICs(pot: callable, prog_w0: jnp.array, Msat: float, t0: float, t1: float, Nstrip: int, solver=diffrax.Dopri8(), rtol=1e-6, atol=1e-6):
    ts = jnp.linspace(t0, t1, Nstrip)
    prog_orb = pot.integrate_orbit(w0=prog_w0, ts=ts, solver=solver, rtol=rtol, atol=atol)
    
    L_close, L_far = jax.vmap(pot.lagrange_pts, in_axes=(0,0,None,0))(prog_orb.ys[:,:3], prog_orb.ys[:,3:], Msat, prog_orb.ts)

    omega_val = jax.vmap(pot.omega)(prog_orb.ys[:,:3], prog_orb.ys[:,3:])

    @jax.vmap
    def release_velocity(q_rel, q_prog, p_prog, omega):
        r_prog_hat = q_prog / jnp.sqrt(jnp.sum(q_prog**2))
        p_prog_hat = p_prog / jnp.sqrt(jnp.sum(p_prog**2))
        sintheta = jnp.linalg.norm(jnp.cross(r_prog_hat, p_prog_hat))
        
        p_rel = omega * jnp.sqrt(jnp.sum(q_rel**2)) / sintheta
        return p_rel * p_prog_hat

    v_lead = release_velocity(L_close, prog_orb.ys[:,:3], prog_orb.ys[:,3:], omega_val)
    v_trail = release_velocity(L_far, prog_orb.ys[:,:3], prog_orb.ys[:,3:], omega_val)

    return L_close, v_lead, L_far, v_trail, ts

@eqx.filter_jit
def gen_streakline(pot: callable, prog_w0: jnp.array, Msat: float, t0: float, t1: float, Nstrip: int, solver=diffrax.Dopri8(), rtol=1e-6, atol=1e-6):
    pos_lead, vel_lead, pos_trail, vel_trail, tstrip = get_Streakline_ICs(pot=pot, prog_w0=prog_w0, Msat=Msat, t0=t0, t1=t1, Nstrip=Nstrip, solver=solver, rtol=rtol, atol=atol)
    
    orb_func = lambda w0, t_start: pot.integrate_orbit(w0=w0, t0=t_start, t1=t1, solver=solver, ts=jnp.array([t1]), rtol=rtol, atol=atol).ys[0]
    
    w0_lead = jnp.hstack([pos_lead, vel_lead])
    w0_trail = jnp.hstack([pos_trail, vel_trail])
    
    mapped_orb_func = jax.vmap(orb_func)
    lead = mapped_orb_func(w0_lead, tstrip)
    trail = mapped_orb_func(w0_trail, tstrip)
    
    return lead, trail, tstrip