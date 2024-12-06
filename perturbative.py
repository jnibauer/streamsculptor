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
import fields
from fields import integrate_field


class GenerateMassRadiusPerturbation(Potential):
    """
    Class to define a perturbation object.
    potiential_base: potential function for the base potential, in H_base
    potential_perturbation: gravitation potential of the perturbation(s)
    potential_structural: defined as the derivative of the perturbing potential wrspct to the structural parameter.
    dPhi_alpha / dstructural. For instance, dPhi_alpha(x, t) / dr.
    """
    def __init__(self, potential_base, potential_perturbation, potential_structural, BaseStreamModel=None, units=None, **kwargs):
        super().__init__(units,{'potential_base':potential_base, 'potential_perturbation':potential_perturbation, 'potential_structural':potential_structural, 'BaseStreamModel':BaseStreamModel})
        self.gradient = None
        self.potential_base = potential_base
        self.gradientPotentialBase = potential_base.gradient
        self.gradientPotentialPerturbation = potential_perturbation.gradient
        self.gradientPotentialStructural = potential_structural.gradient

        self.gradientPotentialPerturbation_per_SH = jax.jit(jax.jacfwd(potential_perturbation.potential_per_SH))
        self.gradientPotentialStructural_per_SH = jax.jit(jax.jacfwd(potential_structural.potential_per_SH))
        
        if BaseStreamModel is not None:
            """
            TODO: remove the if statement by creating a NullBaseStreamModel
            This way the definitions below can be made, without breaking jit compilation.
            Alternative: just assume a BaseStreamModel is always passed, or use a standard base model as a placeholder.
            """
            self.base_stream = BaseStreamModel
            self.num_pert = self.potential_perturbation.subhalo_x0.shape[0]
            ## Note: prog_w0 is the *intial condition*, that we must integrate forward from
            ## .prog_loc_fwd[-1] is the final (observed) coordinate.
            self.field_wobs = [BaseStreamModel.prog_loc_fwd[-1], jnp.zeros((self.num_pert, 12))]
            self.jump_ts = None
            # Need to integrate backwards in time. BaseStreamModel.ts is in increasing time order, from past to present day [past, ..., observation time]
            flipped_times = jnp.flip(BaseStreamModel.ts)
            prog_fieldICs = integrate_field(w0=self.field_wobs,ts=flipped_times,field=fields.MassRadiusPerturbation_OTF(self), backwards_int=True, **kwargs)
            # Flip back to original time order, since we will integrate from [past, ..., observation time]
            self.prog_base = prog_fieldICs### for testing
            self.prog_fieldICs = jnp.flipud(prog_fieldICs.ys[1])
            self.perturbation_ICs_lead, self.perturbation_ICs_trail = self.compute_perturbation_ICs() # N_lead/trail x N_sh x 12

            self.base_realspace_ICs_lead = jnp.hstack([self.base_stream.streamICs[0], self.base_stream.streamICs[2]]) # N_lead x 6
            self.base_realspace_ICs_trail =  jnp.hstack([self.base_stream.streamICs[1], self.base_stream.streamICs[3]]) # N_trail x 6
            
        
    @partial(jax.jit,static_argnums=(0,1))
    def compute_base_stream(self,cpu=True):
        """
        Compute the unperturbed stream.
        If cpu = True: use scan to compute stream
        If cpu = False: use vmap to compute stream [better for gpu usage]
        """
        def cpu_func():
            return self.potential_base.gen_stream_scan(ts=self.base_stream.ts, prog_w0=self.base_stream.prog_w0, Msat=self.base_stream.Msat, seed_num=self.base_stream.seednum, solver=self.base_stream.solver)
        def gpu_func():
            return self.potential_base.gen_stream_vmapped(ts=self.base_stream.ts, prog_w0=self.base_stream.prog_w0, Msat=self.base_stream.Msat, seed_num=self.base_stream.seednum, solver=self.base_stream.solver)

        lead, trail = jax.lax.cond(cpu, cpu_func, gpu_func)
        return lead, trail

    @partial(jax.jit,static_argnums=(0,))
    def compute_perturbation_ICs(self,):
        """
        Compute the initial conditions for the perturbation vector field.
        """
        lead_pert_ICs_deps = jnp.einsum('ijk,ilk->ilj',self.base_stream.dRel_dIC[:,0,:,:],self.prog_fieldICs[:,:,:6])
        trail_pert_ICs_deps = jnp.einsum('ijk,ilk->ilj',self.base_stream.dRel_dIC[:,1,:,:],self.prog_fieldICs[:,:,:6])

        lead_pert_ICs_depsdr =  jnp.einsum('ijk,ilk->ilj',self.base_stream.dRel_dIC[:,0,:,:],self.prog_fieldICs[:,:,6:])
        trail_pert_ICs_depsdr = jnp.einsum('ijk,ilk->ilj',self.base_stream.dRel_dIC[:,1,:,:],self.prog_fieldICs[:,:,6:])

        lead_deriv_ICs = jnp.dstack([lead_pert_ICs_deps,lead_pert_ICs_depsdr])
        trail_deriv_ICs = jnp.dstack([trail_pert_ICs_deps,trail_pert_ICs_depsdr])

        return lead_deriv_ICs, trail_deriv_ICs # N_lead x N_sh x 12

    @partial(jax.jit,static_argnums=(0,1,2))
    def compute_perturbation_OTF(self, cpu=True, solver=diffrax.Dopri8(scan_kind='bounded'), rtol=1e-6, atol=1e-6, dtmin=0.05, dtmax=None):
        integrator = lambda w0, ts: integrate_field(w0=w0,ts=ts,field=fields.MassRadiusPerturbation_OTF(self),jump_ts=self.jump_ts, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax)
        integrator = jax.jit(integrator)
        def cpu_func():
            def scan_fun(carry, particle_idx):
                i, lead_deriv_ICs_curr, trail_deriv_ICs_curr = carry
                ICs_total_lead = [self.base_realspace_ICs_lead[i], lead_deriv_ICs_curr]
                ICs_total_trail = [self.base_realspace_ICs_trail[i], trail_deriv_ICs_curr]
                ts_arr = jnp.array([self.base_stream.ts[i], self.base_stream.ts[-1]])
                lead_space_and_derivs = integrator(ICs_total_lead,ts_arr)
                lead_space_and_derivs = [lead_space_and_derivs.ys[0][-1,:], lead_space_and_derivs.ys[1][-1,:]]
                trail_space_and_derivs = integrator(ICs_total_trail,ts_arr)
                trail_space_and_derivs = [trail_space_and_derivs.ys[0][-1,:], trail_space_and_derivs.ys[1][-1,:]]
                return [i+1, self.perturbation_ICs_lead[i+1], self.perturbation_ICs_trail[i+1]], [lead_space_and_derivs, trail_space_and_derivs]

            init_carry = [0, self.perturbation_ICs_lead[0], self.perturbation_ICs_trail[0]] #perturbation_ICs_lead
            particle_ids = self.base_stream.IDs[:-1]
            final_state, all_states = jax.lax.scan(scan_fun,init_carry,particle_ids)
            lead_and_derivs, trail_and_derivs = all_states
            return lead_and_derivs, trail_and_derivs

        def gpu_func():
            def single_particle_integrate(idx):
                ts_arr = jnp.array([self.base_stream.ts[idx], self.base_stream.ts[-1]])
                lead_space_and_derivs = integrator([self.base_realspace_ICs_lead[idx], self.perturbation_ICs_lead[idx]],ts_arr)
                lead_space_and_derivs = [lead_space_and_derivs.ys[0][-1,:], lead_space_and_derivs.ys[1][-1,:]]
                trail_space_and_derivs = integrator([self.base_realspace_ICs_trail[idx], self.perturbation_ICs_trail[idx]],ts_arr)
                trail_space_and_derivs = [trail_space_and_derivs.ys[0][-1,:], trail_space_and_derivs.ys[1][-1,:]]
                return lead_space_and_derivs, trail_space_and_derivs
            particle_ids = self.base_stream.IDs[:-1]
            lead_and_derivs, trail_and_derivs = jax.vmap(single_particle_integrate)(particle_ids)
            return lead_and_derivs, trail_and_derivs
        
        return jax.lax.cond(cpu, cpu_func, gpu_func)
        
    ################## EXPERIMENTAL ##################
    @partial(jax.jit,static_argnums=(0,1,2))
    def compute_perturbation_jacobian_OTF(self, cpu=True, solver=diffrax.Dopri8(scan_kind='bounded'), rtol=1e-6, atol=1e-6, dtmin=0.05, dtmax=None):
        integrator = lambda realspace_w0, pert_w0, ts: integrate_field(w0=[realspace_w0, pert_w0],ts=ts,field=fields.MassRadiusPerturbation_OTF(self),jump_ts=self.jump_ts, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax)
        integrator = jax.jit(integrator)
        jacobain_integrator = jax.jacfwd(integrator,argnums=(0,))
        def cpu_func():
            def scan_fun(carry, particle_idx):
                i, lead_deriv_ICs_curr, trail_deriv_ICs_curr = carry
                ICs_total_lead = [self.base_realspace_ICs_lead[i], lead_deriv_ICs_curr]
                ICs_total_trail = [self.base_realspace_ICs_trail[i], trail_deriv_ICs_curr]
                ts_arr = jnp.array([self.base_stream.ts[i], self.base_stream.ts[-1]])
                lead_space_and_derivs = jacobain_integrator(self.base_realspace_ICs_lead[i], lead_deriv_ICs_curr, ts_arr)
                lead_space_and_derivs = [lead_space_and_derivs, lead_space_and_derivs]
                trail_space_and_derivs = jacobain_integrator(self.base_realspace_ICs_trail[i], trail_deriv_ICs_curr, ts_arr)
                trail_space_and_derivs = [trail_space_and_derivs, trail_space_and_derivs]
                return [i+1, self.perturbation_ICs_lead[i+1], self.perturbation_ICs_trail[i+1]], [lead_space_and_derivs, trail_space_and_derivs]

            init_carry = [0, self.perturbation_ICs_lead[0], self.perturbation_ICs_trail[0]] #perturbation_ICs_lead
            particle_ids = self.base_stream.IDs[:-1]
            final_state, all_states = jax.lax.scan(scan_fun,init_carry,particle_ids)
            lead_and_derivs, trail_and_derivs = all_states
            return lead_and_derivs, trail_and_derivs

        def gpu_func():
            def single_particle_integrate(idx):
                ts_arr = jnp.array([self.base_stream.ts[idx], self.base_stream.ts[-1]])
                lead_space_and_derivs = jacobain_integrator(self.base_realspace_ICs_lead[idx], self.perturbation_ICs_lead[idx],ts_arr)
                lead_space_and_derivs = [lead_space_and_derivs, lead_space_and_derivs]
                trail_space_and_derivs = jacobain_integrator(self.base_realspace_ICs_trail[idx], self.perturbation_ICs_trail[idx],ts_arr)
                trail_space_and_derivs = [trail_space_and_derivs, trail_space_and_derivs]
                return lead_space_and_derivs, trail_space_and_derivs
            particle_ids = self.base_stream.IDs[:-1]
            lead_and_derivs, trail_and_derivs = jax.vmap(single_particle_integrate)(particle_ids)
            return lead_and_derivs, trail_and_derivs
        
        return jax.lax.cond(cpu, cpu_func, gpu_func)
    ##################################################
    

    @partial(jax.jit,static_argnums=(0,1,2))
    def compute_perturbation_Interp(self,cpu=True,solver=diffrax.Dopri8(scan_kind='bounded'), rtol=1e-6, atol=1e-6, dtmin=0.05, dtmax=None):
        """
        Compute the perturbation field from the interpolated stream.
        """
        integrator = lambda w0, ts, args: integrate_field(w0=w0,ts=ts,field=fields.MassRadiusPerturbation_Interp(self),jump_ts=self.jump_ts, args=args, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax)
        integrator = jax.jit(integrator)
        def cpu_func():
            def scan_fun(carry, particle_idx):
                i, lead_deriv_ICs_curr, trail_deriv_ICs_curr = carry
                args_lead = {'idx':i, 'tail_bool':True}
                args_trail = {'idx':i, 'tail_bool':False}
                ts_arr = jnp.array([self.base_stream.ts[i], self.base_stream.ts[-1]])

                lead_derivs = integrator(lead_deriv_ICs_curr,ts_arr,args_lead)
                lead_derivs = lead_derivs.ys[-1,:]

                trail_derivs = integrator(trail_deriv_ICs_curr,ts_arr,args_trail)
                trail_derivs = trail_derivs.ys[-1,:]
                return [i+1, self.perturbation_ICs_lead[i+1], self.perturbation_ICs_trail[i+1]], [lead_derivs, trail_derivs]

            init_carry = [0, self.perturbation_ICs_lead[0], self.perturbation_ICs_trail[0]] #perturbation_ICs_lead
            particle_ids = self.base_stream.IDs[:-1]
            final_state, all_states = jax.lax.scan(scan_fun,init_carry,particle_ids)
            lead_and_derivs, trail_and_derivs = all_states
            return lead_and_derivs, trail_and_derivs

        def gpu_func():
            def single_particle_integrate(idx):
                ts_arr = jnp.array([self.base_stream.ts[idx], self.base_stream.ts[-1]])
                args_lead = {'idx':idx, 'tail_bool':True}
                args_trail = {'idx':idx, 'tail_bool':False}
                lead_derivs = integrator(self.perturbation_ICs_lead[idx],ts_arr,args_lead)
                lead_derivs = lead_derivs.ys[-1,:]

                trail_derivs = integrator(self.perturbation_ICs_trail[idx],ts_arr,args_trail)
                trail_derivs = trail_derivs.ys[-1,:]
                return lead_derivs, trail_derivs
            particle_ids = self.base_stream.IDs[:-1]
            lead_and_derivs, trail_and_derivs = jax.vmap(single_particle_integrate)(particle_ids)
            return lead_and_derivs, trail_and_derivs

        return jax.lax.cond(cpu, cpu_func, gpu_func)


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

class BaseStreamModel(Potential):
    """
    Class to define a stream model object, in the absence of linear perturbations.
    Zeroth order quantities are precomputed here, excluding the trajectory of the stream particles.
    potiential_base: potential function for the base potential, in H_base
    prog_w0: initial conditions of the progenitor. This will be integrated _forwards_ in time.
    ts: array of stripping times. ts[-1] is the observation time.
    Msat: mass of the satellite
    seednum: seed number for the random number generator
    """
    def __init__(self,  potential_base, prog_w0, ts, Msat, seednum, solver, units=None, dense=False, cpu=True, **kwargs):
        super().__init__(units,{'potential_base':potential_base, 'prog_w0':prog_w0, 'ts':ts, 'Msat':Msat, 'seednum':seednum, 'solver':solver, 'dense':dense, 'cpu':cpu})
        self.potential_base = potential_base
        self.prog_w0 = prog_w0
        self.ts = ts
        self.Msat = Msat
        self.seednum = seednum
        self.dense = dense
        if solver is None:
            self.solver = diffrax.Dopri5(scan_kind='bounded')
        else:
            self.solver = solver
        self.streamICs = potential_base.gen_stream_ics(ts=self.ts, prog_w0=self.prog_w0, Msat=self.Msat, seed_num=self.seednum, solver=self.solver, **kwargs)
        self.IDs = jnp.arange(len(self.ts))

        self.prog_loc_fwd = potential_base.integrate_orbit(w0=self.prog_w0,ts=self.ts, t0=self.ts.min(), t1=self.ts.max(),solver=self.solver, **kwargs).ys        
        self.dRel_dIC = self.release_func_jacobian()

        if dense:
            if cpu:
                #evaluate using: lead, trail = StreamSculptor.eval_dense_stream(time, stream_interp)
                self.stream_interp = potential_base.gen_stream_scan_dense(ts=self.ts, prog_w0=self.prog_w0, Msat=self.Msat, seed_num=self.seednum, solver=self.solver, **kwargs)
            if cpu is False:
                self.stream_interp = potential_base.gen_stream_vmapped_dense(ts=self.ts, prog_w0=self.prog_w0, Msat=self.Msat, seed_num=self.seednum, solver=self.solver, **kwargs)
        else:
            self.stream_interp = None

    @partial(jax.jit,static_argnums=(0,))
    def release_func_jacobian(self,):     
        """ 
        Compute the Jacobian of the release function with repsect to (q,p) for leading, trailing arms.
        prog_loc is in phasespace. A 6dim progenitor phase space location
        Output has len(ts) x 2 x 6 x 6 dimensions
        """
        @jax.jit
        def release_func(prog_loc, M_sat, stripping_num, t, seed_num):
            pos_close_new, pos_far_new, vel_close_new, vel_far_new = self.potential_base.release_model(prog_loc[:3], prog_loc[3:6], M_sat, stripping_num, t, seed_num)
            return jnp.vstack([jnp.hstack([pos_close_new,vel_close_new]),
                        jnp.hstack([pos_far_new, vel_far_new])])

        mapped_release = jax.vmap(release_func,in_axes=((0,None,0,0,None)))
        mapped_release_jacobian = jax.vmap(jax.jacfwd(release_func),in_axes=((0,None,0,0,None)))
        return mapped_release_jacobian(self.prog_loc_fwd, self.Msat, self.IDs, self.ts, self.seednum)
    


