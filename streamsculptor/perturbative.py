import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx
import interpax
from typing import Any

from streamsculptor.main import Potential, usys
from streamsculptor import fields
from streamsculptor.fields import integrate_field
from streamsculptor.streamhelpers import custom_release_model
import streamsculptor as ssc

jax.config.update("jax_enable_x64", True)


class GenerateMassRadiusPerturbation(Potential):
    """
    Class to define a perturbation object.
    potiential_base: potential function for the base potential, in H_base
    potential_perturbation: gravitation potential of the perturbation(s)
    potential_structural: defined as the derivative of the perturbing potential wrspct to the structural parameter.
    dPhi_alpha / dstructural. For instance, dPhi_alpha(x, t) / dr.
    """
    potential_base: Any
    potential_perturbation: Any
    potential_structural: Any
    base_stream: Any = eqx.field(static=True)
    
    gradientPotentialBase: Any = eqx.field(static=True)
    gradientPotentialPerturbation: Any = eqx.field(static=True)
    gradientPotentialStructural: Any = eqx.field(static=True)
    gradientPotentialPerturbation_per_SH: Any = eqx.field(static=True)
    gradientPotentialStructural_per_SH: Any = eqx.field(static=True)
    
    num_pert: int = eqx.field(static=True)
    field_wobs: list
    jump_ts: Any
    prog_base: Any
    prog_fieldICs: jnp.ndarray
    perturbation_ICs_lead: jnp.ndarray
    perturbation_ICs_trail: jnp.ndarray
    base_realspace_ICs_lead: jnp.ndarray
    base_realspace_ICs_trail: jnp.ndarray

    def __init__(self, potential_base, potential_perturbation, potential_structural, BaseStreamModel=None, units=usys, **kwargs):
        super().__init__(units)
        self.potential_base = potential_base
        self.potential_perturbation = potential_perturbation
        self.potential_structural = potential_structural
        
        self.gradientPotentialBase = potential_base.gradient
        self.gradientPotentialPerturbation = potential_perturbation.gradient
        self.gradientPotentialStructural = potential_structural.gradient
        self.gradientPotentialPerturbation_per_SH = jax.jit(jax.jacfwd(potential_perturbation.potential_per_SH))
        self.gradientPotentialStructural_per_SH = jax.jit(jax.jacfwd(potential_structural.potential_per_SH))
        
        if BaseStreamModel is not None:
            self.base_stream = BaseStreamModel
            self.num_pert = self.potential_perturbation.subhalo_x0.shape[0]
            self.field_wobs = [BaseStreamModel.prog_loc_fwd[-1], jnp.zeros((self.num_pert, 12))]
            self.jump_ts = None
            
            flipped_times = jnp.flip(BaseStreamModel.ts)
            prog_fieldICs = integrate_field(w0=self.field_wobs, ts=flipped_times, field=fields.MassRadiusPerturbation_OTF(self), backwards_int=True, **kwargs)
            
            self.prog_base = prog_fieldICs
            self.prog_fieldICs = jnp.flipud(prog_fieldICs.ys[1])
            self.base_realspace_ICs_lead = jnp.hstack([self.base_stream.streamICs[0], self.base_stream.streamICs[2]]) 
            self.base_realspace_ICs_trail =  jnp.hstack([self.base_stream.streamICs[1], self.base_stream.streamICs[3]]) 
            self.perturbation_ICs_lead, self.perturbation_ICs_trail = self.compute_perturbation_ICs() 

    @eqx.filter_jit
    def compute_base_stream(self, cpu=True):
        def cpu_func():
            return self.potential_base.gen_stream_scan(ts=self.base_stream.ts, prog_w0=self.base_stream.prog_w0, Msat=self.base_stream.Msat, seed_num=self.base_stream.seednum, solver=self.base_stream.solver)
        def gpu_func():
            return self.potential_base.gen_stream_vmapped(ts=self.base_stream.ts, prog_w0=self.base_stream.prog_w0, Msat=self.base_stream.Msat, seed_num=self.base_stream.seednum, solver=self.base_stream.solver)
        return jax.lax.cond(cpu, cpu_func, gpu_func)

    @eqx.filter_jit
    def compute_perturbation_ICs(self):
        lead_pert_ICs_deps = jnp.einsum('ijk,ilk->ilj', self.base_stream.dRel_dIC[:, 0, :, :], self.prog_fieldICs[:, :, :6])
        trail_pert_ICs_deps = jnp.einsum('ijk,ilk->ilj', self.base_stream.dRel_dIC[:, 1, :, :], self.prog_fieldICs[:, :, :6])

        lead_pert_ICs_depsdr =  jnp.einsum('ijk,ilk->ilj', self.base_stream.dRel_dIC[:, 0, :, :], self.prog_fieldICs[:, :, 6:])
        trail_pert_ICs_depsdr = jnp.einsum('ijk,ilk->ilj', self.base_stream.dRel_dIC[:, 1, :, :], self.prog_fieldICs[:, :, 6:])

        lead_deriv_ICs = jnp.dstack([lead_pert_ICs_deps, lead_pert_ICs_depsdr])
        trail_deriv_ICs = jnp.dstack([trail_pert_ICs_deps, trail_pert_ICs_depsdr])

        return lead_deriv_ICs, trail_deriv_ICs

    @eqx.filter_jit
    def compute_perturbation_OTF(self, cpu=True, solver=diffrax.Dopri8(scan_kind='bounded'), rtol=1e-6, atol=1e-6, dtmin=0.05, dtmax=None, max_steps=10_000):
        integrator = lambda w0, ts: integrate_field(w0=w0, ts=ts, field=fields.MassRadiusPerturbation_OTF(self), jump_ts=self.jump_ts, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps)
        
        def cpu_func():
            def scan_fun(carry, particle_idx):
                i, lead_deriv_ICs_curr, trail_deriv_ICs_curr = carry
                ICs_total_lead = [self.base_realspace_ICs_lead[i], lead_deriv_ICs_curr]
                ICs_total_trail = [self.base_realspace_ICs_trail[i], trail_deriv_ICs_curr]
                ts_arr = jnp.array([self.base_stream.ts[i], self.base_stream.ts[-1]])
                
                lead_space_and_derivs = integrator(ICs_total_lead, ts_arr)
                lead_out = [lead_space_and_derivs.ys[0][-1, :], lead_space_and_derivs.ys[1][-1, :]]
                
                trail_space_and_derivs = integrator(ICs_total_trail, ts_arr)
                trail_out = [trail_space_and_derivs.ys[0][-1, :], trail_space_and_derivs.ys[1][-1, :]]
                
                return [i+1, self.perturbation_ICs_lead[i+1], self.perturbation_ICs_trail[i+1]], [lead_out, trail_out]

            init_carry = [0, self.perturbation_ICs_lead[0], self.perturbation_ICs_trail[0]]
            particle_ids = self.base_stream.IDs[:-1]
            _, all_states = jax.lax.scan(scan_fun, init_carry, particle_ids)
            return all_states[0], all_states[1]

        def gpu_func():
            def single_particle_integrate(idx):
                ts_arr = jnp.array([self.base_stream.ts[idx], self.base_stream.ts[-1]])
                lead_space_and_derivs = integrator([self.base_realspace_ICs_lead[idx], self.perturbation_ICs_lead[idx]], ts_arr)
                lead_out = [lead_space_and_derivs.ys[0][-1, :], lead_space_and_derivs.ys[1][-1, :]]
                
                trail_space_and_derivs = integrator([self.base_realspace_ICs_trail[idx], self.perturbation_ICs_trail[idx]], ts_arr)
                trail_out = [trail_space_and_derivs.ys[0][-1, :], trail_space_and_derivs.ys[1][-1, :]]
                return lead_out, trail_out
                
            particle_ids = self.base_stream.IDs[:-1]
            return jax.vmap(single_particle_integrate)(particle_ids)
        
        return jax.lax.cond(cpu, cpu_func, gpu_func)
        
    @eqx.filter_jit
    def compute_perturbation_jacobian_OTF(self, cpu=True, solver=diffrax.Dopri8(scan_kind='bounded'), rtol=1e-6, atol=1e-6, dtmin=0.05, dtmax=None):
        integrator = lambda realspace_w0, pert_w0, ts: integrate_field(w0=[realspace_w0, pert_w0], ts=ts, field=fields.MassRadiusPerturbation_OTF(self), jump_ts=self.jump_ts, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax)
        jacobian_integrator = jax.jacfwd(integrator, argnums=(0,))
        
        def cpu_func():
            def scan_fun(carry, particle_idx):
                i, lead_deriv_ICs_curr, trail_deriv_ICs_curr = carry
                ts_arr = jnp.array([self.base_stream.ts[i], self.base_stream.ts[-1]])
                
                lead_space_and_derivs = jacobian_integrator(self.base_realspace_ICs_lead[i], lead_deriv_ICs_curr, ts_arr)
                lead_out = [lead_space_and_derivs, lead_space_and_derivs]
                
                trail_space_and_derivs = jacobian_integrator(self.base_realspace_ICs_trail[i], trail_deriv_ICs_curr, ts_arr)
                trail_out = [trail_space_and_derivs, trail_space_and_derivs]
                
                return [i+1, self.perturbation_ICs_lead[i+1], self.perturbation_ICs_trail[i+1]], [lead_out, trail_out]

            init_carry = [0, self.perturbation_ICs_lead[0], self.perturbation_ICs_trail[0]]
            particle_ids = self.base_stream.IDs[:-1]
            _, all_states = jax.lax.scan(scan_fun, init_carry, particle_ids)
            return all_states[0], all_states[1]

        def gpu_func():
            def single_particle_integrate(idx):
                ts_arr = jnp.array([self.base_stream.ts[idx], self.base_stream.ts[-1]])
                lead_space_and_derivs = jacobian_integrator(self.base_realspace_ICs_lead[idx], self.perturbation_ICs_lead[idx], ts_arr)
                lead_out = [lead_space_and_derivs, lead_space_and_derivs]
                
                trail_space_and_derivs = jacobian_integrator(self.base_realspace_ICs_trail[idx], self.perturbation_ICs_trail[idx], ts_arr)
                trail_out = [trail_space_and_derivs, trail_space_and_derivs]
                return lead_out, trail_out
                
            particle_ids = self.base_stream.IDs[:-1]
            return jax.vmap(single_particle_integrate)(particle_ids)
        
        return jax.lax.cond(cpu, cpu_func, gpu_func)
    
    @eqx.filter_jit
    def compute_perturbation_Interp(self, cpu=True, solver=diffrax.Dopri8(scan_kind='bounded'), rtol=1e-6, atol=1e-6, dtmin=0.05, dtmax=None):
        integrator = lambda w0, ts, args: integrate_field(w0=w0, ts=ts, field=fields.MassRadiusPerturbation_Interp(self), jump_ts=self.jump_ts, args=args, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax)
        
        def cpu_func():
            def scan_fun(carry, particle_idx):
                i, lead_deriv_ICs_curr, trail_deriv_ICs_curr = carry
                args_lead = {'idx': i, 'tail_bool': True}
                args_trail = {'idx': i, 'tail_bool': False}
                ts_arr = jnp.array([self.base_stream.ts[i], self.base_stream.ts[-1]])

                lead_derivs = integrator(lead_deriv_ICs_curr, ts_arr, args_lead).ys[-1, :]
                trail_derivs = integrator(trail_deriv_ICs_curr, ts_arr, args_trail).ys[-1, :]
                
                return [i+1, self.perturbation_ICs_lead[i+1], self.perturbation_ICs_trail[i+1]], [lead_derivs, trail_derivs]

            init_carry = [0, self.perturbation_ICs_lead[0], self.perturbation_ICs_trail[0]]
            particle_ids = self.base_stream.IDs[:-1]
            _, all_states = jax.lax.scan(scan_fun, init_carry, particle_ids)
            return all_states[0], all_states[1]

        def gpu_func():
            def single_particle_integrate(idx):
                ts_arr = jnp.array([self.base_stream.ts[idx], self.base_stream.ts[-1]])
                args_lead = {'idx': idx, 'tail_bool': True}
                args_trail = {'idx': idx, 'tail_bool': False}
                
                lead_derivs = integrator(self.perturbation_ICs_lead[idx], ts_arr, args_lead).ys[-1, :]
                trail_derivs = integrator(self.perturbation_ICs_trail[idx], ts_arr, args_trail).ys[-1, :]
                return lead_derivs, trail_derivs
                
            particle_ids = self.base_stream.IDs[:-1]
            return jax.vmap(single_particle_integrate)(particle_ids)

        return jax.lax.cond(cpu, cpu_func, gpu_func)


class GenerateMassPerturbation(Potential):
    potential_base: Any
    potential_perturbation: Any
    gradientPotentialBase: Any = eqx.field(static=True)
    gradientPotentialPerturbation: Any = eqx.field(static=True)
    gradientPotentialPerturbation_per_SH: Any = eqx.field(static=True)

    def __init__(self, potential_base, potential_perturbation, units=usys):
        super().__init__(units)
        self.potential_base = potential_base
        self.potential_perturbation = potential_perturbation
        self.gradientPotentialBase = potential_base.gradient
        self.gradientPotentialPerturbation = potential_perturbation.gradient
        self.gradientPotentialPerturbation_per_SH = jax.jit(jax.jacfwd(potential_perturbation.potential_per_SH))
        
    @eqx.filter_jit
    def potential(self, xyz, t):
        raise NotImplementedError

class BaseStreamModel(Potential):
    potential_base: Any
    prog_w0: jnp.ndarray
    ts: jnp.ndarray
    Msat: float
    seednum: int = eqx.field(static=True)
    solver: Any = eqx.field(static=True)
    dense: bool = eqx.field(static=True)
    
    streamICs: jnp.ndarray
    IDs: jnp.ndarray
    prog_loc_fwd: jnp.ndarray
    dRel_dIC: jnp.ndarray
    stream_interp: Any

    def __init__(self, potential_base, prog_w0, ts, Msat, seednum, solver=None, units=usys, dense=False, cpu=True, **kwargs):
        super().__init__(units)
        self.potential_base = potential_base
        self.prog_w0 = prog_w0
        self.ts = ts
        self.Msat = float(Msat)
        self.seednum = int(seednum)
        self.dense = dense
        self.solver = solver if solver is not None else diffrax.Dopri5(scan_kind='bounded')
        
        self.streamICs = potential_base.gen_stream_ics(ts=self.ts, prog_w0=self.prog_w0, Msat=self.Msat, seed_num=self.seednum, solver=self.solver, **kwargs)
        self.IDs = jnp.arange(len(self.ts))
        self.prog_loc_fwd = potential_base.integrate_orbit(w0=self.prog_w0, ts=self.ts, t0=self.ts.min(), t1=self.ts.max(), solver=self.solver, **kwargs).ys        
        self.dRel_dIC = self.release_func_jacobian()

        if self.dense:
            if cpu:
                self.stream_interp = potential_base.gen_stream_scan_dense(ts=self.ts, prog_w0=self.prog_w0, Msat=self.Msat, seed_num=self.seednum, solver=self.solver, **kwargs)
            else:
                self.stream_interp = potential_base.gen_stream_vmapped_dense(ts=self.ts, prog_w0=self.prog_w0, Msat=self.Msat, seed_num=self.seednum, solver=self.solver, **kwargs)
        else:
            self.stream_interp = None

    @eqx.filter_jit
    def release_func_jacobian(self):     
        def release_func(prog_loc, M_sat, stripping_num, t, seed_num):
            pos_close_new, pos_far_new, vel_close_new, vel_far_new = self.potential_base.release_model(prog_loc[:3], prog_loc[3:6], M_sat, stripping_num, t, seed_num)
            return jnp.vstack([jnp.hstack([pos_close_new, vel_close_new]),
                               jnp.hstack([pos_far_new, vel_far_new])])

        mapped_release_jacobian = jax.vmap(jax.jacfwd(release_func), in_axes=((0, None, 0, 0, None)))
        return mapped_release_jacobian(self.prog_loc_fwd, self.Msat, self.IDs, self.ts, self.seednum)
    

class CustomBaseStreamModel(Potential):
    potential_base: Any
    prog_w0: jnp.ndarray
    ts: jnp.ndarray
    pos_rel: jnp.ndarray
    vel_rel: jnp.ndarray
    solver: Any = eqx.field(static=True)
    dense: bool = eqx.field(static=True)
    
    prog_at_ts: jnp.ndarray
    streamICs: jnp.ndarray
    IDs: jnp.ndarray
    prog_loc_fwd: jnp.ndarray
    dRel_dIC: jnp.ndarray
    stream_interp: Any

    def __init__(self, potential_base=None, prog_w0=None, ts=None, pos_rel=None, vel_rel=None, solver=None, units=usys, dense=False, cpu=True, **kwargs):
        super().__init__(units)
        self.potential_base = potential_base
        self.prog_w0 = prog_w0
        self.ts = ts
        self.pos_rel = pos_rel
        self.vel_rel = vel_rel
        self.solver = solver if solver is not None else diffrax.Dopri5(scan_kind='bounded')
        self.dense = dense

        self.prog_at_ts = self.potential_base.integrate_orbit(w0=self.prog_w0, ts=self.ts, t0=self.ts.min(), t1=self.ts.max(), solver=self.solver, **kwargs).ys 
        streamICs = custom_release_model(pos_prog=self.prog_at_ts[:,:3], vel_prog=self.prog_at_ts[:,3:], pos_rel=self.pos_rel, vel_rel=self.vel_rel)
        self.streamICs = jnp.hstack([streamICs[0], streamICs[1]])

        self.IDs = jnp.arange(len(self.ts))
        self.prog_loc_fwd = potential_base.integrate_orbit(w0=self.prog_w0, ts=self.ts, t0=self.ts.min(), t1=self.ts.max(), solver=self.solver, **kwargs).ys        
        self.dRel_dIC = self.release_func_jacobian()

        if self.dense:
            self.stream_interp = jax.vmap(potential_base.integrate_orbit, in_axes=(0,0,None,None,None))(self.streamICs, self.ts, self.ts[-1], solver=self.solver, dense=True)
        else:
            self.stream_interp = None

    @eqx.filter_jit
    def gen_stream(self):
        integrator = lambda w0, t0, t1: self.potential_base.integrate_orbit(w0=w0, t0=t0, t1=t1, dense=False, solver=self.solver, ts=jnp.array([t1]))
        return jax.vmap(integrator, in_axes=((0, 0, None)))(self.streamICs[:-1, :], self.ts[:-1], self.ts[-1])

    @eqx.filter_jit
    def release_func_jacobian(self):     
        def release_func(prog_loc, pos_r, vel_r):
            pos_out, vel_out = custom_release_model(pos_prog=prog_loc[:3], vel_prog=prog_loc[3:], pos_rel=pos_r, vel_rel=vel_r)
            return jnp.hstack([pos_out, vel_out])
        
        mapped_release_jacobian = jax.vmap(jax.jacfwd(release_func), in_axes=((0, 0, 0)))
        return mapped_release_jacobian(self.prog_at_ts, self.pos_rel, self.vel_rel) 


class GenerateMassRadiusPerturbation_CustomBase(Potential):
    potential_base: Any
    potential_perturbation: Any
    potential_structural: Any
    base_stream: Any = eqx.field(static=True)
    
    gradientPotentialBase: Any = eqx.field(static=True)
    gradientPotentialPerturbation: Any = eqx.field(static=True)
    gradientPotentialStructural: Any = eqx.field(static=True)
    gradientPotentialPerturbation_per_SH: Any = eqx.field(static=True)
    gradientPotentialStructural_per_SH: Any = eqx.field(static=True)
    
    num_pert: int = eqx.field(static=True)
    field_wobs: list
    jump_ts: Any
    prog_base: Any
    prog_fieldICs: jnp.ndarray
    perturbation_ICs: jnp.ndarray
    base_realspace_ICs: jnp.ndarray

    def __init__(self, potential_base, potential_perturbation, potential_structural, BaseStreamModel=None, units=usys, **kwargs):
        super().__init__(units)
        self.potential_base = potential_base
        self.potential_perturbation = potential_perturbation
        self.potential_structural = potential_structural
        self.gradientPotentialBase = potential_base.gradient
        self.gradientPotentialPerturbation = potential_perturbation.gradient
        self.gradientPotentialStructural = potential_structural.gradient

        self.gradientPotentialPerturbation_per_SH = jax.jit(jax.jacfwd(potential_perturbation.potential_per_SH))
        self.gradientPotentialStructural_per_SH = jax.jit(jax.jacfwd(potential_structural.potential_per_SH))
        
        self.base_stream = BaseStreamModel
        self.num_pert = self.potential_perturbation.subhalo_x0.shape[0]
        self.field_wobs = [BaseStreamModel.prog_loc_fwd[-1], jnp.zeros((self.num_pert, 12))]
        self.jump_ts = None
        
        flipped_times = jnp.flip(BaseStreamModel.ts)
        prog_fieldICs = integrate_field(w0=self.field_wobs, ts=flipped_times, field=fields.MassRadiusPerturbation_OTF(self), backwards_int=True, **kwargs)
        
        self.prog_base = prog_fieldICs
        self.prog_fieldICs = jnp.flipud(prog_fieldICs.ys[1])
        
        self.base_realspace_ICs = self.base_stream.streamICs 
        self.perturbation_ICs = self.compute_perturbation_ICs() 
            
    @eqx.filter_jit
    def compute_base_stream(self, cpu=True):
        return self.base_stream.gen_stream()

    @eqx.filter_jit
    def compute_perturbation_ICs(self):
        pert_ICs_deps = jnp.einsum('ijk,ilk->ilj', self.base_stream.dRel_dIC, self.prog_fieldICs[:, :, :6])
        pert_ICs_depsdr =  jnp.einsum('ijk,ilk->ilj', self.base_stream.dRel_dIC, self.prog_fieldICs[:, :, 6:])
        deriv_ICs = jnp.dstack([pert_ICs_deps, pert_ICs_depsdr])
        return deriv_ICs 

    @eqx.filter_jit
    def compute_perturbation_OTF(self, cpu=True, solver=diffrax.Dopri8(scan_kind='bounded'), rtol=1e-6, atol=1e-6, dtmin=0.05, max_steps=10_000, dtmax=None):
        integrator = lambda w0, ts: integrate_field(w0=w0, ts=ts, field=fields.MassRadiusPerturbation_OTF(self), jump_ts=self.jump_ts, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps)
        
        def cpu_func():
            def scan_fun(carry, particle_idx):
                i, deriv_ICs_curr = carry
                ICs_total = [self.base_realspace_ICs[i], deriv_ICs_curr]
                ts_arr = jnp.array([self.base_stream.ts[i], self.base_stream.ts[-1]])
                space_and_derivs = integrator(ICs_total, ts_arr)
                space_out = [space_and_derivs.ys[0][-1, :], space_and_derivs.ys[1][-1, :]]
                return [i+1, self.perturbation_ICs[i+1]], space_out

            init_carry = [0, self.perturbation_ICs[0]] 
            particle_ids = self.base_stream.IDs[:-1]
            _, all_states = jax.lax.scan(scan_fun, init_carry, particle_ids)
            return all_states

        def gpu_func():
            def single_particle_integrate(idx):
                ts_arr = jnp.array([self.base_stream.ts[idx], self.base_stream.ts[-1]])
                space_and_derivs = integrator([self.base_realspace_ICs[idx], self.perturbation_ICs[idx]], ts_arr)
                return [space_and_derivs.ys[0][-1, :], space_and_derivs.ys[1][-1, :]]
                
            particle_ids = self.base_stream.IDs[:-1]
            return jax.vmap(single_particle_integrate)(particle_ids)
        
        return jax.lax.cond(cpu, cpu_func, gpu_func)
        
    @eqx.filter_jit
    def compute_perturbation_jacobian_OTF(self, cpu=True, solver=diffrax.Dopri8(scan_kind='bounded'), rtol=1e-6, atol=1e-6, dtmin=0.05, dtmax=None):
        integrator = lambda realspace_w0, pert_w0, ts: integrate_field(w0=[realspace_w0, pert_w0], ts=ts, field=fields.MassRadiusPerturbation_OTF(self), jump_ts=self.jump_ts, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax)
        jacobian_integrator = jax.jacfwd(integrator, argnums=(0,))
        
        def cpu_func():
            def scan_fun(carry, particle_idx):
                i, deriv_ICs_curr = carry
                ts_arr = jnp.array([self.base_stream.ts[i], self.base_stream.ts[-1]])
                space_and_derivs = jacobian_integrator(self.base_realspace_ICs[i], deriv_ICs_curr, ts_arr)
                space_out = [space_and_derivs.ys[0][-1, :], space_and_derivs.ys[1][-1, :]]
                return [i+1, self.perturbation_ICs[i+1]], [space_out]

            init_carry = [0, self.perturbation_ICs[0]] 
            particle_ids = self.base_stream.IDs[:-1]
            _, all_states = jax.lax.scan(scan_fun, init_carry, particle_ids)
            return all_states

        def gpu_func():
            def single_particle_integrate(idx):
                ts_arr = jnp.array([self.base_stream.ts[idx], self.base_stream.ts[-1]])
                space_and_derivs = jacobian_integrator(self.base_realspace_ICs[idx], self.perturbation_ICs[idx], ts_arr)
                return [space_and_derivs.ys[0][-1, :], space_and_derivs.ys[1][-1, :]]
                
            particle_ids = self.base_stream.IDs[:-1]
            return jax.vmap(single_particle_integrate)(particle_ids)
        
        return jax.lax.cond(cpu, cpu_func, gpu_func)


class GenerateMassRadiusPerturbation_CustomBase_SecondOrder(Potential):
    potential_base: Any
    potential_perturbation: Any
    potential_structural: Any
    base_stream: Any = eqx.field(static=True)
    
    gradientPotentialBase: Any = eqx.field(static=True)
    gradientPotentialPerturbation: Any = eqx.field(static=True)
    gradientPotentialStructural: Any = eqx.field(static=True)
    gradientPotentialPerturbation_per_SH: Any = eqx.field(static=True)
    gradientPotentialStructural_per_SH: Any = eqx.field(static=True)
    
    num_pert: int = eqx.field(static=True)
    field_wobs: list
    jump_ts: Any
    prog_base: Any
    prog_fieldICs_first_order_mass: jnp.ndarray
    prog_fieldICs_second_order_mass: jnp.ndarray
    perturbation_ICs: list
    base_realspace_ICs: jnp.ndarray

    def __init__(self, potential_base, potential_perturbation, potential_structural, BaseStreamModel=None, units=usys, **kwargs):
        super().__init__(units)
        self.potential_base = potential_base
        self.potential_perturbation = potential_perturbation
        self.potential_structural = potential_structural
        
        self.gradientPotentialBase = potential_base.gradient
        self.gradientPotentialPerturbation = potential_perturbation.gradient
        self.gradientPotentialStructural = potential_structural.gradient

        self.gradientPotentialPerturbation_per_SH = jax.jit(jax.jacfwd(potential_perturbation.potential_per_SH))
        self.gradientPotentialStructural_per_SH = jax.jit(jax.jacfwd(potential_structural.potential_per_SH))
        
        self.base_stream = BaseStreamModel
        self.num_pert = self.potential_perturbation.subhalo_x0.shape[0]
        self.field_wobs = [BaseStreamModel.prog_loc_fwd[-1], jnp.zeros((self.num_pert, 12)), jnp.zeros((self.num_pert, 6))]
        self.jump_ts = None
        
        flipped_times = jnp.flip(BaseStreamModel.ts)
        prog_fieldICs = integrate_field(w0=self.field_wobs, ts=flipped_times, field=fields.MassRadiusPerturbation_OTF_SecondOrder(self), backwards_int=True, **kwargs)
        
        self.prog_base = prog_fieldICs
        self.prog_fieldICs_first_order_mass = jnp.flipud(prog_fieldICs.ys[1])
        self.prog_fieldICs_second_order_mass = jnp.flipud(prog_fieldICs.ys[2])
        
        self.base_realspace_ICs = self.base_stream.streamICs 
        self.perturbation_ICs = self.compute_perturbation_ICs() 
            
    @eqx.filter_jit
    def compute_base_stream(self, cpu=False):
        return self.base_stream.gen_stream()

    @eqx.filter_jit
    def compute_perturbation_ICs(self):
        pert_ICs_deps = jnp.einsum('ijk,ilk->ilj', self.base_stream.dRel_dIC, self.prog_fieldICs_first_order_mass[:, :, :6])
        pert_ICs_depsdr =  jnp.einsum('ijk,ilk->ilj', self.base_stream.dRel_dIC, self.prog_fieldICs_first_order_mass[:, :, 6:])
        deriv_ICs_first_order_mass = jnp.dstack([pert_ICs_deps, pert_ICs_depsdr]) 

        pert_ICs_second_order_mass = jnp.einsum('ijk,ilk->ilj', self.base_stream.dRel_dIC, self.prog_fieldICs_second_order_mass[:, :, :]) 
        deriv_ICs = [deriv_ICs_first_order_mass, pert_ICs_second_order_mass] 
        return deriv_ICs 

    @eqx.filter_jit
    def compute_perturbation_OTF(self, cpu=False, solver=diffrax.Dopri8(scan_kind='bounded'), rtol=1e-8, atol=1e-8, dtmin=0.05, max_steps=10_000, dtmax=None):
        integrator = lambda w0, ts: integrate_field(w0=w0, ts=ts, field=fields.MassRadiusPerturbation_OTF_SecondOrder(self), jump_ts=self.jump_ts, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps)
        
        def cpu_func():
            def scan_fun(carry, particle_idx):
                i, deriv_ICs_curr = carry
                ICs_total = [self.base_realspace_ICs[i], deriv_ICs_curr[0], deriv_ICs_curr[1]]
                ts_arr = jnp.array([self.base_stream.ts[i], self.base_stream.ts[-1]])
                
                space_and_derivs = integrator(ICs_total, ts_arr)
                space_out = [space_and_derivs.ys[0][-1, :], space_and_derivs.ys[1][-1, :], space_and_derivs.ys[2][-1, :]]
                
                pert_ics_next = [self.perturbation_ICs[0][i+1], self.perturbation_ICs[1][i+1]]
                return [i+1, pert_ics_next], space_out

            init_carry = [0, [self.perturbation_ICs[0][0], self.perturbation_ICs[1][0]]] 
            particle_ids = self.base_stream.IDs[:-1]
            _, all_states = jax.lax.scan(scan_fun, init_carry, particle_ids)
            return all_states

        def gpu_func():
            def single_particle_integrate(idx):
                ts_arr = jnp.array([self.base_stream.ts[idx], self.base_stream.ts[-1]])
                space_and_derivs = integrator([self.base_realspace_ICs[idx], self.perturbation_ICs[0][idx], self.perturbation_ICs[1][idx]], ts_arr)
                return [space_and_derivs.ys[0][-1, :], space_and_derivs.ys[1][-1, :], space_and_derivs.ys[2][-1, :]]
                
            particle_ids = self.base_stream.IDs[:-1]
            return jax.vmap(single_particle_integrate)(particle_ids)
        
        return jax.lax.cond(cpu, cpu_func, gpu_func)
        

class BaseStreamModelChen25(Potential):
    pot_base: Any
    ts: jnp.ndarray
    prog_w0: jnp.ndarray
    Msat: float
    key: jnp.ndarray
    solver: Any = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    dtmin: float = eqx.field(static=True)
    dtmax: Any = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    throw: bool = eqx.field(static=True)
    prog_pot: Any

    pot_tot: Any
    BaseModel: CustomBaseStreamModel

    def __init__(self, pot_base, ts, prog_w0, Msat, key, solver=diffrax.Dopri5(scan_kind='bounded'), rtol=1e-7, atol=1e-7, dtmin=0.3, dtmax=None, max_steps=10_000, throw=False, prog_pot=None, units=usys):
        super().__init__(units)
        self.pot_base = pot_base
        self.ts = jnp.asarray(ts)
        self.prog_w0 = jnp.asarray(prog_w0)
        self.Msat = float(Msat)
        self.key = key
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.dtmin = dtmin
        self.dtmax = dtmax
        self.max_steps = max_steps
        self.throw = throw
        self.prog_pot = prog_pot if prog_pot is not None else ssc.potential.PlummerPotential(m=0.0, r_s=1.0, units=usys)

        stream_ics, orb_fwd = ssc.gen_stream_ics_Chen25(pot_base=self.pot_base, ts=self.ts, prog_w0=self.prog_w0, Msat=self.Msat, key=self.key, solver=self.solver, rtol=self.rtol, atol=self.atol, dtmin=self.dtmin, dtmax=self.dtmax, max_steps=self.max_steps)
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = stream_ics
        
        pos_close_rel = pos_close_arr - orb_fwd.ys[:, :3]
        pos_far_rel = pos_far_arr - orb_fwd.ys[:, :3]
        vel_close_rel = vel_close_arr - orb_fwd.ys[:, 3:]
        vel_far_rel = vel_far_arr - orb_fwd.ys[:, 3:]

        pos_rel = jnp.vstack([pos_close_rel, pos_far_rel])  
        vel_rel = jnp.vstack([vel_close_rel, vel_far_rel])

        ts_stack = jnp.clip(jnp.hstack([self.ts, self.ts + 1e-12]), self.ts.min(), self.ts.max()) 
        args = jnp.argsort(ts_stack)  
        ts_sorted = ts_stack[args]
        pos_rel = pos_rel[args]
        vel_rel = vel_rel[args]

        prog_spline = interpax.Interpolator1D(x=orb_fwd.ts, f=orb_fwd.ys[:, :3], method='cubic')
        prog_pot_translating = ssc.potential.TimeDepTranslatingPotential(pot=self.prog_pot, center_spl=prog_spline, units=usys)
        self.pot_tot = ssc.potential.Potential_Combine(potential_list=[self.pot_base, prog_pot_translating], units=usys)
        
        self.BaseModel = CustomBaseStreamModel(
            potential_base=self.pot_tot, 
            prog_w0=self.prog_w0, 
            ts=ts_sorted,  
            pos_rel=pos_rel, 
            vel_rel=vel_rel, 
            solver=self.solver, 
            units=self.pot_tot.units,
            dense=False, 
            cpu=False
        )

class GenerateMassRadiusPerturbation_Chen25(Potential):
    potential_base: Any
    potential_perturbation: Any
    potential_structural: Any
    base_stream: Any = eqx.field(static=True)
    
    gradientPotentialBase: Any = eqx.field(static=True)
    gradientPotentialPerturbation: Any = eqx.field(static=True)
    gradientPotentialStructural: Any = eqx.field(static=True)
    gradientPotentialPerturbation_per_SH: Any = eqx.field(static=True)
    gradientPotentialStructural_per_SH: Any = eqx.field(static=True)
    
    num_pert: int = eqx.field(static=True)
    field_wobs: list
    jump_ts: Any
    prog_base: Any
    prog_fieldICs: jnp.ndarray
    perturbation_ICs: jnp.ndarray
    base_realspace_ICs: jnp.ndarray
    second_order_pert_ICs: list
    BaseStreamModel: Any = eqx.field(static=True)

    def __init__(self, potential_base, potential_perturbation, BaseStreamModel, units=usys, **kwargs):
        super().__init__(units)
        self.potential_base = potential_base
        self.potential_perturbation = potential_perturbation
        self.BaseStreamModel = BaseStreamModel
        
        self.gradientPotentialBase = BaseStreamModel.pot_tot.gradient 
        self.gradientPotentialPerturbation = potential_perturbation.gradient
        
        self.potential_structural = ssc.potential.SubhaloLinePotentialCustom_dRadius_fromFunc(
                                                        func=potential_perturbation.func,
                                                        m=potential_perturbation.m,
                                                        r_s=potential_perturbation.r_s,
                                                        subhalo_x0=potential_perturbation.subhalo_x0,
                                                        subhalo_v=potential_perturbation.subhalo_v,
                                                        subhalo_t0=potential_perturbation.subhalo_t0,
                                                        t_window=potential_perturbation.t_window,
                                                        units=potential_perturbation.units)
        
        self.gradientPotentialStructural = self.potential_structural.gradient
        self.gradientPotentialPerturbation_per_SH = jax.jit(jax.jacfwd(potential_perturbation.potential_per_SH))
        self.gradientPotentialStructural_per_SH = jax.jit(jax.jacfwd(self.potential_structural.potential_per_SH))
        
        self.base_stream = BaseStreamModel.BaseModel 
        self.num_pert = self.potential_perturbation.subhalo_x0.shape[0]
        self.field_wobs = [self.base_stream.prog_loc_fwd[-1], jnp.zeros((self.num_pert, 12))]
        self.jump_ts = None
        
        flipped_times = jnp.flip(self.base_stream.ts)
        prog_fieldICs = integrate_field(w0=self.field_wobs, ts=flipped_times, field=fields.MassRadiusPerturbation_OTF(self), backwards_int=True, **kwargs)
        
        self.prog_base = prog_fieldICs
        self.prog_fieldICs = jnp.flipud(prog_fieldICs.ys[1])
        self.perturbation_ICs = jnp.zeros((len(flipped_times), self.num_pert, 12)) 
        self.base_realspace_ICs = self.base_stream.streamICs 
        
        self.second_order_pert_ICs = [self.perturbation_ICs, jnp.zeros((len(flipped_times), self.num_pert, 6))] 

    @eqx.filter_jit
    def compute_base_stream(self, cpu=True):
        return self.base_stream.gen_stream()

    @eqx.filter_jit
    def compute_perturbation_OTF(self, cpu=True, solver=diffrax.Dopri8(scan_kind='bounded'), rtol=1e-6, atol=1e-6, dtmin=0.05, max_steps=10_000, dtmax=None):
        integrator = lambda w0, ts: integrate_field(w0=w0, ts=ts, field=fields.MassRadiusPerturbation_OTF(self), jump_ts=self.jump_ts, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps)
        
        def cpu_func():
            def scan_fun(carry, particle_idx):
                i, deriv_ICs_curr = carry
                ICs_total = [self.base_realspace_ICs[i], deriv_ICs_curr]
                ts_arr = jnp.array([self.base_stream.ts[i], self.base_stream.ts[-1]])
                space_and_derivs = integrator(ICs_total, ts_arr)
                space_out = [space_and_derivs.ys[0][-1, :], space_and_derivs.ys[1][-1, :]]
                return [i+1, self.perturbation_ICs[i+1]], space_out

            init_carry = [0, self.perturbation_ICs[0]] 
            particle_ids = self.base_stream.IDs[:-1]
            _, all_states = jax.lax.scan(scan_fun, init_carry, particle_ids)
            return all_states

        def gpu_func():
            def single_particle_integrate(idx):
                ts_arr = jnp.array([self.base_stream.ts[idx], self.base_stream.ts[-1]])
                space_and_derivs = integrator([self.base_realspace_ICs[idx], self.perturbation_ICs[idx]], ts_arr)
                return [space_and_derivs.ys[0][-1, :], space_and_derivs.ys[1][-1, :]]
                
            particle_ids = self.base_stream.IDs[:-1]
            return jax.vmap(single_particle_integrate)(particle_ids)
        
        return jax.lax.cond(cpu, cpu_func, gpu_func)

    @eqx.filter_jit
    def compute_perturbation_second_order_OTF(self, cpu=True, solver=diffrax.Dopri8(scan_kind='bounded'), rtol=1e-6, atol=1e-6, dtmin=0.05, max_steps=10_000, dtmax=None):
        integrator = lambda w0, ts: integrate_field(w0=w0, ts=ts, field=fields.MassRadiusPerturbation_OTF_SecondOrder(self), jump_ts=self.jump_ts, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps)
      
        def gpu_func():
            def single_particle_integrate(idx):
                ts_arr = jnp.array([self.base_stream.ts[idx], self.base_stream.ts[-1]])
                space_and_derivs = integrator([self.base_realspace_ICs[idx], self.second_order_pert_ICs[0][idx], self.second_order_pert_ICs[1][idx]], ts_arr)
                return [space_and_derivs.ys[0][-1, :], space_and_derivs.ys[1][-1, :], space_and_derivs.ys[2][-1, :]]
                
            particle_ids = self.base_stream.IDs[:-1]
            return jax.vmap(single_particle_integrate)(particle_ids)
        
        return gpu_func()

    @eqx.filter_jit
    def run_nonlinear_sim(self, pot_pert=None, solver=diffrax.Dopri8(scan_kind='bounded'), rtol=1e-6, atol=1e-6, dtmin=0.05, max_steps=10_000):
        if pot_pert is None:
            pot_pert = self.potential_perturbation
    
        l, t = ssc.gen_stream_vmapped_with_pert_Chen25_fixed_prog(
            pot_base=self.potential_base,
            pot_pert=pot_pert,
            prog_pot=self.BaseStreamModel.prog_pot,
            prog_w0=self.base_stream.prog_w0, 
            ts=self.BaseStreamModel.ts, 
            key=self.BaseStreamModel.key, 
            Msat=self.BaseStreamModel.Msat, 
            max_steps=max_steps,
            atol=atol,
            rtol=rtol, 
            solver=solver, 
            dtmin=dtmin
        )
        return jnp.vstack([l, t])