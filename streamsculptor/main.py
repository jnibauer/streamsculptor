import jax
import jax.numpy as jnp
from jax import random 
import equinox as eqx
import diffrax
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt, PIDController, ForwardMode
from functools import partial
jax.config.update("jax_enable_x64", True)

# =============================================================================
# Unit System
# =============================================================================

class GalacticUnitSystem(eqx.Module):
    """
    A lightweight, JAX-friendly unit system.
    Default: kpc, Myr, Msun, radian.
    """
    G: float = eqx.field(static=True)
    
    def __init__(self):
        # Gravitational constant in kpc^3 / (Msun * Myr^2)
        self.G = 4.498502151469554e-12

usys = GalacticUnitSystem()

# =============================================================================
# Potential Base Class
# =============================================================================

class Potential(eqx.Module):
    """
    Base Potential class as an eqx.Module.
    Arrays assigned as attributes are treated as dynamic PyTree leaves.
    """
    units: GalacticUnitSystem = eqx.field(static=True)

    def __init__(self, units=usys):
        self.units = units

    def potential(self, xyz, t):
        raise NotImplementedError("Subclasses must implement the potential function.")

    def gradient(self, xyz, t):
        return jax.grad(self.potential)(xyz, t)
    
    def density(self, xyz, t):
        lap = jnp.trace(jax.hessian(self.potential)(xyz, t))
        return lap / (4 * jnp.pi * self.units.G)
    
    def acceleration(self, xyz, t):
        return -self.gradient(xyz, t)
    
    def local_circular_velocity(self, xyz, t):
        r = jnp.linalg.norm(xyz)
        r_hat = xyz / r
        grad_phi = self.gradient(xyz, t)
        dphi_dr = jnp.sum(grad_phi * r_hat)
        return jnp.sqrt(jnp.abs(r * dphi_dr))
   
    def jacobian_force(self, xyz, t):
        return jax.jacfwd(self.gradient)(xyz, t)

    def dphidr(self, x, t):
        rad = jnp.linalg.norm(x)
        r_hat = x / rad
        return jnp.sum(self.gradient(x, t) * r_hat)

    def d2phidr2(self, x, t):
        rad = jnp.linalg.norm(x)
        r_hat = x / rad
        dphi_dr_func = lambda pos: jnp.sum(self.gradient(pos, t) * r_hat)
        return jnp.sum(jax.grad(dphi_dr_func)(x) * r_hat)

    def omega(self, x, v):
        rad_sq = jnp.sum(x**2)
        omega_vec = jnp.cross(x, v) / rad_sq
        return jnp.linalg.norm(omega_vec)

    def tidalr(self, x, v, Msat, t):
        denom = self.omega(x, v)**2 - self.d2phidr2(x, t)
        denom = jnp.clip(denom, a_min=1e-12)
        return (self.units.G * Msat / denom) ** (1.0 / 3.0)
    
    def lagrange_pts(self, x, v, Msat, t):
        r_tidal = self.tidalr(x, v, Msat, t)
        r_hat = x / jnp.linalg.norm(x)
        L_close = x - r_hat * r_tidal
        L_far = x + r_hat * r_tidal
        return L_close, L_far  
    
    #@partial(jax.jit, static_argnames=['self'])
    def velocity_acceleration(self, t, xv, args):
        x, v = xv[:3], xv[3:]
        acceleration = -self.gradient(x, t)
        return jnp.hstack([v, acceleration])
    
    # =============================================================================
    # Orbit Integrators
    # =============================================================================

    @eqx.filter_jit
    def integrate_orbit(self, w0, ts, dense=False, solver=Dopri5(scan_kind='bounded'), rtol=1e-7, atol=1e-7, dtmin=0.3, dtmax=None, max_steps=10_000, t0=None, t1=None, steps=False, jump_ts=None, throw=False):
        #vel_acc = jax.jit(self.velocity_acceleration)
        term = ODETerm(self.velocity_acceleration)
        saveat = SaveAt(t0=False, t1=False, ts=ts if not dense else None, dense=dense, steps=steps)
        stepsize_controller = PIDController(rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, force_dtmin=True, jump_ts=jump_ts)
        
        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=ts.min() if t0 is None else t0,
            t1=ts.max() if t1 is None else t1,
            y0=w0,
            dt0=None,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=int(max_steps),
            adjoint=ForwardMode(),
            throw=throw
        )
        return solution

    @eqx.filter_jit
    def integrate_orbit_batch_scan(self, w0, ts, dense=False, solver=Dopri5(scan_kind='bounded'), rtol=1e-7, atol=1e-7, dtmin=0.3, dtmax=None, max_steps=10_000, t0=None, t1=None, steps=False, jump_ts=None, throw=False):
        def body(carry, i):
            w0_curr = w0[i]
            ts_curr = ts if len(ts.shape) == 1 else ts[i]
            sol = self.integrate_orbit(w0=w0_curr, ts=ts_curr, dense=dense, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps, t0=t0, t1=t1, steps=steps, jump_ts=jump_ts, throw=throw)
            return None, sol 
        
        _, all_states = jax.lax.scan(body, None, jnp.arange(len(w0)))
        return all_states

    @eqx.filter_jit
    def integrate_orbit_batch_vmapped(self, w0, ts, dense=False, solver=Dopri5(scan_kind='bounded'), rtol=1e-7, atol=1e-7, dtmin=0.3, dtmax=None, max_steps=10_000, t0=None, t1=None, steps=False, jump_ts=None, throw=False):
        integrator = lambda w, t_arr: self.integrate_orbit(w0=w, ts=t_arr, dense=dense, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps, t0=t0, t1=t1, steps=steps, jump_ts=jump_ts, throw=throw)
        
        if len(ts.shape) == 1:
            return jax.vmap(integrator, in_axes=(0, None))(w0, ts)
        else:
            return jax.vmap(integrator, in_axes=(0, 0))(w0, ts)

    # =============================================================================
    # Stream Model
    # =============================================================================
    
    def release_model(self, x, v, Msat, i, t, seed_num, kval_arr=1.0):
        pred = jnp.isscalar(kval_arr)
        def true_func():
            return jnp.array([2.0, 0.3, 0.0, 0.0, 0.4, 0.4, 0.5, 0.5])
        def false_func():
            return jnp.ones(8) * kval_arr
        
        kval_arr = jax.lax.cond(pred, true_func, false_func)
        kr_bar, kvphi_bar, kz_bar, kvz_bar, sigma_kr, sigma_kvphi, sigma_kz, sigma_kvz = kval_arr
        
        key_master = random.PRNGKey(seed_num)
        random_ints = random.randint(key=key_master, shape=(5,), minval=0, maxval=1000)

        keya, keyb, keyc, keyd, _ = random.split(random.PRNGKey(i * random_ints[0]), 5)
        
        r = jnp.linalg.norm(x)
        r_hat = x / r
        r_tidal = self.tidalr(x, v, Msat, t)
        rel_v = self.omega(x, v) * r_tidal 
        v_circ = rel_v
        
        L_vec = jnp.cross(x, v)
        z_hat = L_vec / jnp.linalg.norm(L_vec)
        
        phi_vec = v - jnp.sum(v * r_hat) * r_hat
        phi_hat = phi_vec / jnp.linalg.norm(phi_vec)
        
        kr_samp = kr_bar + random.normal(keya, shape=(1,)) * sigma_kr
        kvphi_samp = kr_samp * (kvphi_bar + random.normal(keyb, shape=(1,)) * sigma_kvphi)
        kz_samp = kz_bar + random.normal(keyc, shape=(1,)) * sigma_kz
        kvz_samp = kvz_bar + random.normal(keyd, shape=(1,)) * sigma_kvz
        
        pos_trail = x + kr_samp * r_hat * r_tidal
        pos_trail = pos_trail + z_hat * kz_samp * r_tidal
        v_trail = v + (kvphi_samp * v_circ) * phi_hat
        v_trail = v_trail + (kvz_samp * v_circ) * z_hat
        
        pos_lead = x - kr_samp * r_hat * r_tidal
        pos_lead = pos_lead - z_hat * kz_samp * r_tidal
        v_lead = v - (kvphi_samp * v_circ) * phi_hat
        v_lead = v_lead - (kvz_samp * v_circ) * z_hat
        
        return pos_lead, pos_trail, v_lead, v_trail

    @eqx.filter_jit
    def gen_stream_ics(self, ts, prog_w0, Msat, seed_num, solver=Dopri5(scan_kind='bounded'), kval_arr=1.0, rtol=1e-7, atol=1e-7, dtmin=0.3, dtmax=None, max_steps=10_000, throw=False):
        ws_jax = self.integrate_orbit(w0=prog_w0, ts=ts, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps, throw=throw).ys
        Msat_arr = Msat * jnp.ones(len(ts))

        def body_func(i):
            return self.release_model(x=ws_jax[i, :3], v=ws_jax[i, 3:], Msat=Msat_arr[i], i=i, t=ts[i], seed_num=seed_num, kval_arr=kval_arr)
        
        return jax.vmap(body_func)(jnp.arange(len(ts)))
            
    @eqx.filter_jit
    def gen_stream_scan(self, ts, prog_w0, Msat, seed_num, solver=Dopri5(scan_kind='bounded'), kval_arr=1.0, rtol=1e-7, atol=1e-7, dtmin=0.3, dtmax=None, max_steps=10_000, throw=False):
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts=ts, prog_w0=prog_w0, Msat=Msat, seed_num=seed_num, solver=solver, kval_arr=kval_arr, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps, throw=throw)
        
        orb_integrator = lambda w0, t_arr: self.integrate_orbit(w0=w0, ts=t_arr, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps, throw=throw).ys[-1]
        orb_integrator_mapped = jax.vmap(orb_integrator, in_axes=(0, None))

        def scan_fun(carry, i):
            pos_close_curr, pos_far_curr, vel_close_curr, vel_far_curr = carry
            w0_close = jnp.hstack([pos_close_curr, vel_close_curr])
            w0_far = jnp.hstack([pos_far_curr, vel_far_curr])

            ts_arr = jnp.array([ts[i], ts[-1]])
            curr_locs = jnp.vstack([w0_close, w0_far])
            
            w_particle = orb_integrator_mapped(curr_locs, ts_arr)
            
            next_carry = [pos_close_arr[i+1, :], pos_far_arr[i+1, :], vel_close_arr[i+1, :], vel_far_arr[i+1, :]]
            return next_carry, [w_particle[0], w_particle[1]]

        init_carry = [pos_close_arr[0, :], pos_far_arr[0, :], vel_close_arr[0, :], vel_far_arr[0, :]]
        particle_ids = jnp.arange(len(pos_close_arr) - 1)
        
        _, all_states = jax.lax.scan(scan_fun, init_carry, particle_ids)
        return all_states[0], all_states[1]
    
    @eqx.filter_jit
    def gen_stream_vmapped(self, ts, prog_w0, Msat, seed_num, solver=Dopri5(scan_kind='bounded'), kval_arr=1.0, rtol=1e-7, atol=1e-7, dtmin=0.3, dtmax=None, max_steps=10_000, throw=False):
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts=ts, prog_w0=prog_w0, Msat=Msat, seed_num=seed_num, solver=solver, kval_arr=kval_arr, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps, throw=throw)
        
        orb_integrator = lambda w0, t_arr: self.integrate_orbit(w0=w0, ts=t_arr, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps, throw=throw).ys[-1]
        orb_integrator_mapped = jax.vmap(orb_integrator, in_axes=(0, None))

        def single_particle_integrate(particle_number, p_close, p_far, v_close, v_far):
            w0_close = jnp.hstack([p_close, v_close])
            w0_far = jnp.hstack([p_far, v_far])
            ts_arr = jnp.array([ts[particle_number], ts[-1]])
            
            curr_locs = jnp.vstack([w0_close, w0_far])
            w_particle = orb_integrator_mapped(curr_locs, ts_arr)

            return w_particle[0], w_particle[1]
        
        particle_ids = jnp.arange(len(pos_close_arr) - 1)
        return jax.vmap(single_particle_integrate, in_axes=(0, 0, 0, 0, 0))(particle_ids, pos_close_arr[:-1], pos_far_arr[:-1], vel_close_arr[:-1], vel_far_arr[:-1])

    # =============================================================================
    # Dense Stream Model
    # =============================================================================
  
    @eqx.filter_jit
    def gen_stream_scan_dense(self, ts, prog_w0, Msat, seed_num, solver=Dopri5(scan_kind='bounded'), kval_arr=1.0, rtol=1e-7, atol=1e-7, dtmin=0.3, dtmax=None, max_steps=10_000, throw=False):
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts=ts, prog_w0=prog_w0, Msat=Msat, seed_num=seed_num, solver=solver, kval_arr=kval_arr, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps, throw=throw)
        
        orb_integrator = lambda w0, t_arr: self.integrate_orbit(w0=w0, ts=t_arr, dense=True, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps, throw=throw)
        orb_integrator_mapped = jax.vmap(orb_integrator, in_axes=(0, None))

        def scan_fun(carry, i):
            pos_close_curr, pos_far_curr, vel_close_curr, vel_far_curr = carry
            w0_close = jnp.hstack([pos_close_curr, vel_close_curr])
            w0_far = jnp.hstack([pos_far_curr, vel_far_curr])

            ts_arr = jnp.array([ts[i], ts[-1]])
            curr_locs = jnp.vstack([w0_close, w0_far])
            w_particle = orb_integrator_mapped(curr_locs, ts_arr)

            next_carry = [pos_close_arr[i+1, :], pos_far_arr[i+1, :], vel_close_arr[i+1, :], vel_far_arr[i+1, :]]
            return next_carry, [w_particle]

        init_carry = [pos_close_arr[0, :], pos_far_arr[0, :], vel_close_arr[0, :], vel_far_arr[0, :]]
        particle_ids = jnp.arange(len(pos_close_arr) - 1)
        
        _, all_states = jax.lax.scan(scan_fun, init_carry, particle_ids)
        return all_states[0]

    @eqx.filter_jit
    def gen_stream_vmapped_dense(self, ts, prog_w0, Msat, seed_num, solver=Dopri5(scan_kind='bounded'), kval_arr=1.0, rtol=1e-7, atol=1e-7, dtmin=0.3, dtmax=None, max_steps=10_000, throw=False):
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts=ts, prog_w0=prog_w0, Msat=Msat, seed_num=seed_num, solver=solver, kval_arr=kval_arr, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps, throw=throw)
        
        orb_integrator = lambda w0, t_arr: self.integrate_orbit(w0=w0, ts=t_arr, dense=True, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps, throw=throw)
        orb_integrator_mapped = jax.vmap(orb_integrator, in_axes=(0, None))

        def single_particle_integrate(particle_number, p_close, p_far, v_close, v_far):
            w0_close = jnp.hstack([p_close, v_close])
            w0_far = jnp.hstack([p_far, v_far])
            ts_arr = jnp.array([ts[particle_number], ts[-1]])
            
            curr_locs = jnp.vstack([w0_close, w0_far])
            return orb_integrator_mapped(curr_locs, ts_arr)
        
        particle_ids = jnp.arange(len(pos_close_arr) - 1)
        return jax.vmap(single_particle_integrate, in_axes=(0, 0, 0, 0, 0))(particle_ids, pos_close_arr[:-1], pos_far_arr[:-1], vel_close_arr[:-1], vel_far_arr[:-1])