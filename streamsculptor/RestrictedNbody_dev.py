from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
import optimistix as optx

# Assuming streamsculptor imports remain the same
from streamsculptor.fields import integrate_field

jax.config.update("jax_enable_x64", True)

class RestrictedNbody_generator(eqx.Module):
    """
    Subclassing eqx.Module turns this into a registered JAX PyTree.
    This prevents JAX from recompiling or failing when you pass this object 
    into jitted functions or scan loops.
    """
    potential: any
    progenitor_potential: any
    interp_prog: any
    r_esc: float

    def __init__(self, potential, progenitor_potential, interp_prog, r_esc=1.0):
        self.potential = potential
        self.progenitor_potential = progenitor_potential
        self.interp_prog = interp_prog
        self.r_esc = r_esc

    def cost_func(self, params, args):
        """
        Optimistix expects fn(y, args). 
        params: [log10(mass), log10(r_s)]
        args: (locs, t, inside_bool)
        """
        locs, t, inside_bool = args
        mass_param, r_s_param = 10**params[0], 10**params[1]
        
        pot_prog_curr = self.progenitor_potential(m=mass_param, r_s=r_s_param, units=self.potential.units)
        density_at_locs = jax.vmap(pot_prog_curr.density, in_axes=(0, None))(locs, t)
        
        # Avoid log(0) by using jnp.where
        log_density = jnp.where(inside_bool, jnp.log(density_at_locs + 1e-12), 0.0)
        
        # Negative log-likelihood
        log_like = -mass_param + jnp.sum(log_density)
        return -log_like

    def fit_monopole(self, init_params, x, t, inside_bool, maxiter):
        """
        Uses Optimistix's Newton solver to find the exact root.
        """
        # optx.Newton computes the exact Hessian. If that becomes too slow 
        # for a very complex potential, you can instantly swap this to optx.BFGS(rtol, atol)
        solver = optx.Newton(rtol=1e-5, atol=1e-5)
        args = (x, t, inside_bool)

        def mass_left(p_init):
            sol = optx.minimise(
                self.cost_func,
                solver,
                y0=p_init,
                args=args,
                max_steps=maxiter,
                throw=False # Prevents crashing if max_steps is hit before tolerance
            )
            return 10**sol.value # Convert back from log10

        def dissolved(p_init):
            return jnp.array([0.0, 1.0])

        mass_left_bool = inside_bool.sum() > 0
        return jax.lax.cond(mass_left_bool, mass_left, dissolved, init_params)

    def get_params(self, t, coords, current_mass, current_rs, maxiter=5):
        """
        Extracts particles inside r_esc and triggers the Newton fit.
        """
        x = coords[:, :3]
        prog_center = self.interp_prog.evaluate(t)[:3]
        x_rel = x - prog_center
        r_rel = jnp.sqrt(jnp.sum(x_rel**2, axis=1))
        inside_bool = r_rel < self.r_esc
        
        init_params = jnp.array([jnp.log10(current_mass), jnp.log10(current_rs)])
        mass_fit, r_s_fit = self.fit_monopole(init_params, x_rel, t, inside_bool, maxiter)
       
        return mass_fit, r_s_fit

    def term(self, t, coords, args):
        """
        The diffrax ODETerm right-hand side.
        Note: requires mass and rs to be passed in args so the state isn't hardcoded.
        """
        mass, rs = args
        x, v = coords[:, :3], coords[:, 3:]
        
        acceleration_external = -jax.vmap(self.potential.gradient, in_axes=(0, None))(x, t)
        
        prog_center = self.interp_prog.evaluate(t)[:3]
        x_rel = x - prog_center
        
        pot_prog_curr = self.progenitor_potential(m=mass, r_s=rs, units=self.potential.units)
        acceleration_internal = -jax.vmap(pot_prog_curr.gradient, in_axes=(0, None))(x_rel, t)
        
        acceleration = acceleration_external + acceleration_internal
        return jnp.hstack([v, acceleration])


def integrate_restricted_Nbody(w0, ts, interrupt_ts, field, solver=diffrax.Dopri8(scan_kind='bounded'), args=None, rtol=1e-7, atol=1e-7, dtmin=0.05, dtmax=None, maxiter=5, max_steps=1_000, mass_init=None, r_s_init=None):
    """
    Much cleaner scan loop. We don't recreate the field object, we just update the 
    mass and r_s parameters flowing through the carry.
    """
    @eqx.filter_jit
    def body_func(carry, idx):
        wcurr, tcurr, tstop, param_mass, param_rs = carry
        
        # 1. Fit the new parameters at the current interruption step
        new_mass, new_rs = field.get_params(tcurr, wcurr, param_mass, param_rs, maxiter=maxiter)
        
        # 2. Integrate to the next interruption step
        ts_curr = jnp.array([tcurr, jnp.clip(tstop, -jnp.inf, ts[-1])])
        
        # Pass the dynamic mass/rs as args so integrate_field can give them to field.term
        # If your integrate_field routine doesn't natively accept dynamic args for the field, 
        # you may need to tweak it to pass (new_mass, new_rs) into the ODETerm.
        field_args = (new_mass, new_rs) 
        
        sol = integrate_field(
            w0=wcurr, ts=ts_curr, solver=solver, field=field, 
            args=field_args, rtol=rtol, atol=atol, dtmin=dtmin, 
            dtmax=dtmax, max_steps=max_steps
        )
        w_next = sol.ys[-1]
        
        new_carry = [w_next, tstop, interrupt_ts[idx+1], new_mass, new_rs]
        saved_state = [tstop, new_mass, new_rs, w_next]
        
        return new_carry, saved_state

    # Setup the interrupt schedule
    interrupt_ts = jnp.hstack([interrupt_ts, ts.max()])    
    init_carry = [w0, ts[0], interrupt_ts[0], mass_init, r_s_init]
    ids = jnp.arange(len(interrupt_ts))
    
    # Run the scan
    final_carry, all_states = jax.lax.scan(body_func, init_carry, ids)
    return all_states