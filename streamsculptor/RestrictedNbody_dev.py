from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import jaxopt
import optax
from typing import Any

from streamsculptor.fields import integrate_field
from streamsculptor.main import usys

jax.config.update("jax_enable_x64", True)
class RestrictedNbody_generator(eqx.Module):
    potential: Any
    progenitor_potential_cls: type = eqx.field(static=True)
    interp_prog: Any
    init_mass: Any 
    init_rs: Any   
    r_s_max: Any  
    pot_prog_curr: Any
    m_particle: float = eqx.field(static=True)

    def __init__(self, potential, progenitor_potential, interp_prog, 
                 init_mass=1e5, init_rs=1.0, r_s_max=5.0, m_particle=1.0):
        self.potential = potential
        self.progenitor_potential_cls = progenitor_potential
        self.interp_prog = interp_prog
        self.r_s_max = r_s_max
        self.init_mass = init_mass
        self.init_rs = init_rs
        self.m_particle = float(m_particle)

        # Ensure safe initial values for the potential
        rs_safe = jnp.maximum(self.init_rs, 1e-6)
        m_safe = jnp.maximum(self.init_mass, 1e-4)

        self.pot_prog_curr = self.progenitor_potential_cls(
            m=m_safe, r_s=rs_safe, units=self.potential.units
        )

    @eqx.filter_jit
    def get_params(self, t, coords):
        """
        Calculates bound mass and scale radius deterministically based on particle energies.
        """
        # 1. Get relative kinematics
        prog_state = self.interp_prog.evaluate(t)
        x_prog, v_prog = prog_state[:3], prog_state[3:6]
        
        x_rel = coords[:, :3] - x_prog
        v_rel = coords[:, 3:] - v_prog
        
        # 2. Compute specific energies (E = Phi + K)
        # Using the current Plummer potential for Phi
        phi = jax.vmap(self.pot_prog_curr.potential, in_axes=(0, None))(x_rel, t)
        kin = 0.5 * jnp.sum(v_rel**2, axis=1)
        energy = phi + kin
        
        # 3. Identify bound particles
        bound_mask = energy < 0.0
        N_bound = jnp.sum(bound_mask)
        
        # 4. Deterministic Mass
        new_mass = N_bound * self.m_particle
        
        # 5. Deterministic Scale Radius
        r_rel = jnp.linalg.norm(x_rel, axis=1)
        
        # Push unbound particles to infinity so they sort to the end
        r_bound_padded = jnp.where(bound_mask, r_rel, jnp.inf)
        r_sorted = jnp.sort(r_bound_padded)
        
        # Find the index of the half-mass radius (median of bound particles)
        # We use floor division // so it remains a valid integer index for JAX
        half_idx = jnp.maximum(0, N_bound // 2 - 1)
        r_half = r_sorted[half_idx]
        
        # For a Plummer sphere, r_half ~ 1.305 * r_s
        new_rs = r_half / 1.305
        
        # 6. Fallback conditions for a dissolved cluster
        # If fewer than 5 particles remain, lock the radius and set mass to trace levels
        is_alive = N_bound > 5
        new_mass = jnp.where(is_alive, new_mass, 1e-3)
        new_rs = jnp.where(is_alive, new_rs, self.init_rs)
        
        # Clip outputs to safe ranges
        new_mass = jnp.maximum(new_mass, 0.0)
        new_rs = jnp.clip(new_rs, 1e-6, self.r_s_max)
        
        return new_mass, new_rs

    @eqx.filter_jit
    def term(self, t, coords, args):
        """
        Evaluates the combined external + internal force field.
        """
        x, v = coords[:, :3], coords[:, 3:]
        
        # Vmap external gradient
        acc_ext = -jax.vmap(self.potential.gradient, in_axes=(0, None))(x, t)
        
        # Vmap internal (progenitor) gradient
        prog_center = self.interp_prog.evaluate(t)[:3]
        x_rel = x - prog_center
        acc_int = -jax.vmap(self.pot_prog_curr.gradient, in_axes=(0, None))(x_rel, t)

        return jnp.hstack([v, acc_ext + acc_int])

# =============================================================================
# Integration Wrappers
# =============================================================================

@eqx.filter_jit
def integrate_restricted_Nbody(w0, ts, interrupt_ts, field, solver=diffrax.Dopri8(scan_kind='bounded'), 
                               rtol=1e-7, atol=1e-7, dtmin=0.01, dtmax=None, max_steps=5000, 
                               mass_init=1e5, r_s_init=1.0, save_full_traj=False):
    
    # Calculate mass per particle ONCE at the start
    N_total = w0.shape[0]
    m_particle = mass_init / N_total
    
    def body_func(carry, idx):
        wcurr, tcurr, next_idx, param_mass, param_rs = carry
        # Clip t_tstop between -inf and the end of the main integration interval to ensure we don't overshoot
        t_stop = jnp.clip(interrupt_ts[next_idx], -jnp.inf, ts[-1])
        
        
        # Evaluate current state to get new parameters based on energy
        curr_state = RestrictedNbody_generator(
            potential=field.potential, 
            progenitor_potential=field.progenitor_potential_cls, 
            interp_prog=field.interp_prog, 
            init_mass=param_mass, 
            init_rs=param_rs,
            m_particle=m_particle
        )
        
        new_mass, new_rs = curr_state.get_params(tcurr, wcurr)
        
        # Build field with updated parameters for the upcoming interval
        new_field = RestrictedNbody_generator(
            potential=field.potential, 
            progenitor_potential=field.progenitor_potential_cls, 
            interp_prog=field.interp_prog, 
            init_mass=new_mass, 
            init_rs=new_rs,
            m_particle=m_particle
        )
        
        ts_slice = jnp.array([tcurr, t_stop])
        sol = integrate_field(
            w0=wcurr, ts=ts_slice, solver=solver, field=new_field, 
            rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps
        )
        
        w_next = sol.ys[-1]
        new_carry = [w_next, t_stop, next_idx + 1, new_mass, new_rs]

        if save_full_traj:
            # If we want to save the full trajectory, we can concatenate the results
            # This will create a large array, so be cautious with memory usage
            saved_state = [t_stop, new_mass, new_rs, w_next]

        else:
            saved_state = [t_stop, new_mass, new_rs]
        return new_carry, saved_state


        #return new_carry, [t_stop, new_mass, new_rs, w_next]

    # Process interrupt schedule
    init_carry = [w0, ts[0], 0, mass_init, r_s_init]
    ids = jnp.arange(len(interrupt_ts))
    
    final_carry, all_states = jax.lax.scan(body_func, init_carry, ids)
    w_final = final_carry[0]
    return w_final, all_states


# class RestrictedNbody_generator(eqx.Module):
#     potential: Any
#     progenitor_potential_cls: type = eqx.field(static=True)
#     interp_prog: Any
#     r_esc: Any
#     lr: float = eqx.field(static=True)
#     maxiter: int = eqx.field(static=True)
#     init_mass: Any
#     init_rs: Any
#     pot_prog_curr: Any

#     def __init__(self, potential=None, progenitor_potential=None, interp_prog=None, 
#                  init_mass=1e5, init_rs=1.0, r_esc=1.0, maxiter=250, lr=1e-3):
#         self.potential = potential
#         self.progenitor_potential_cls = progenitor_potential
#         self.interp_prog = interp_prog
#         self.r_esc = r_esc
#         self.lr = float(lr)
#         self.maxiter = int(maxiter)
#         self.init_mass = init_mass
#         self.init_rs = init_rs

#         # Initialize the active potential instance
#         self.pot_prog_curr = self.progenitor_potential_cls(
#             m=self.init_mass, r_s=self.init_rs, units=self.potential.units
#         )

#     def cost_func(self, params, locs, t, inside_bool):
#         # EXACT copy of your original cost function
#         mass_param, r_s_param = 10**params[0], 10**params[1]
#         pot_prog_curr = self.progenitor_potential_cls(m=mass_param, r_s=r_s_param, units=self.potential.units)
#         density_at_locs = jax.vmap(pot_prog_curr.density, in_axes=(0, None))(locs, t)
        
#         # log(0) handled via jnp.where
#         log_density = jnp.where(inside_bool, jnp.log(density_at_locs), 0.0)
#         log_like = -mass_param + jnp.sum(log_density)
#         return -log_like

#     @eqx.filter_jit
#     def fit_monopole(self, x, t, inside_bool):
#         def mass_left():
#             # Warm start from the carry parameters
#             init_params = jnp.array([jnp.log10(self.init_mass), jnp.log10(self.init_rs)])
            
#             # Setup the exact ADAM solver you had before
#             opt = optax.adam(learning_rate=self.lr)
#             solver = jaxopt.OptaxSolver(opt=opt, fun=self.cost_func, maxiter=self.maxiter)
            
#             # Run solver
#             res = solver.run(init_params, locs=x, t=t, inside_bool=inside_bool)
#             return 10**res.params

#         def dissolved():
#             return jnp.array([0.0, 1.0])

#         mass_left_bool = inside_bool.sum() > 0
#         return jax.lax.cond(mass_left_bool, mass_left, dissolved)

#     @eqx.filter_jit
#     def get_params(self, t, coords, args=None):
#         x = coords[:, :3]
#         prog_center = self.interp_prog.evaluate(t)[:3]
#         x_rel = x - prog_center
#         r_rel = jnp.sqrt(jnp.sum(x_rel**2, axis=1))
        
#         inside_bool = r_rel < self.r_esc
#         mass_fit, r_s_fit = self.fit_monopole(x_rel, t, inside_bool)
       
#         return mass_fit, r_s_fit
    
#     @eqx.filter_jit
#     def term(self, t, coords, args):
#         x, v = coords[:, :3], coords[:, 3:]
        
#         acceleration_external = -jax.vmap(self.potential.gradient, in_axes=(0, None))(x, t)
        
#         prog_center = self.interp_prog.evaluate(t)[:3]
#         x_rel = x - prog_center
        
#         acceleration_internal = -jax.vmap(self.pot_prog_curr.gradient, in_axes=(0, None))(x_rel, t)
#         acceleration = acceleration_external + acceleration_internal

#         return jnp.hstack([v, acceleration])

# # =============================================================================
# # Integration Wrappers
# # =============================================================================

# def initialize_prog_params(w0=None, t0=None, field=None, maxiter=5_000):
#     init_state = RestrictedNbody_generator(
#         potential=field.potential, 
#         progenitor_potential=field.progenitor_potential_cls, 
#         interp_prog=field.interp_prog, 
#         r_esc=field.r_esc, 
#         init_mass=field.init_mass, 
#         init_rs=field.init_rs, 
#         maxiter=maxiter,
#         lr=field.lr
#     )
#     mass_curr, r_s_curr = init_state.get_params(t=t0, coords=w0)
#     return mass_curr, r_s_curr

# @eqx.filter_jit
# def integrate_restricted_Nbody(w0, ts, interrupt_ts, field, solver=diffrax.Dopri8(scan_kind='bounded'), 
#                                rtol=1e-7, atol=1e-7, dtmin=0.05, dtmax=None, maxiter=250, 
#                                max_steps=1_000, mass_init=1e5, r_s_init=1.0):
    
#     def body_func(carry, idx):
#         wcurr, tcurr, tstop, param_mass, param_rs = carry
#         ts_curr = jnp.array([tcurr, jnp.clip(tstop, -jnp.inf, ts[-1])])
        
#         curr_state = RestrictedNbody_generator(
#             potential=field.potential, 
#             progenitor_potential=field.progenitor_potential_cls, 
#             interp_prog=field.interp_prog, 
#             r_esc=field.r_esc, 
#             init_mass=param_mass, 
#             init_rs=param_rs, 
#             maxiter=maxiter,
#             lr=field.lr
#         )
        
#         mass_curr, r_s_curr = curr_state.get_params(t=tcurr, coords=wcurr)
        
#         new_field = RestrictedNbody_generator(
#             potential=field.potential, 
#             progenitor_potential=field.progenitor_potential_cls, 
#             interp_prog=field.interp_prog, 
#             r_esc=field.r_esc, 
#             init_mass=mass_curr, 
#             init_rs=r_s_curr,
#             lr=field.lr
#         )
        
#         w_at_tstop = integrate_field(
#             w0=wcurr, ts=ts_curr, solver=solver, field=new_field, 
#             rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps
#         ).ys[-1]
        
#         new_carry = [w_at_tstop, tstop, interrupt_ts[idx+1], mass_curr, r_s_curr]
#         saved_state = [tstop, mass_curr, r_s_curr, w_at_tstop]
        
#         return new_carry, saved_state

#     interrupt_ts = jnp.hstack([interrupt_ts, ts.max()])    
#     init_carry = [w0, ts[0], interrupt_ts[0], mass_init, r_s_init]
#     ids = jnp.arange(len(interrupt_ts) - 1)
    
#     _, all_states = jax.lax.scan(body_func, init_carry, ids)
#     return all_states