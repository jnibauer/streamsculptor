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
from diffrax import diffeqsolve, ODETerm, Dopri5,SaveAt,PIDController,DiscreteTerminatingEvent, DirectAdjoint, RecursiveCheckpointAdjoint, ConstantStepSize, Euler, StepTo, ForwardMode
import diffrax
import equinox as eqx

usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)

class Potential:
    def __init__(self, units, params):
        if units is None:
            units = dimensionless
        self.units = UnitSystem(units)
        
        if self.units == dimensionless:
            self._G = 1
        else:
            self._G = G.decompose(self.units).value
        
        for name, param in params.items():
            if hasattr(param, 'unit'):
                param = param.decompose(self.units).value
            setattr(self, name, param)
    
    @partial(jax.jit,static_argnums=(0,))
    def gradient(self, xyz, t):
        grad_func = jax.grad(self.potential)
        return grad_func(xyz, t)
    
    @partial(jax.jit,static_argnums=(0,))
    def density(self, xyz, t):
        lap = jnp.trace(jax.hessian(self.potential)(xyz, t))
        return lap / (4 * jnp.pi * self._G)
    
    @partial(jax.jit,static_argnums=(0,))
    def acceleration(self, xyz, t):
        return -self.gradient(xyz, t)
    
    @partial(jax.jit,static_argnums=(0,))
    def local_circular_velocity(self,xyz,t):
        r = jnp.sqrt(jnp.sum(xyz**2))
        r_hat = xyz / r
        grad_phi = self.gradient(xyz,t)
        dphi_dr = jnp.sum(grad_phi*r_hat)
        return jnp.sqrt( r*dphi_dr )
   
    @partial(jax.jit,static_argnums=(0,))
    def jacobian_force(self, xyz, t):
        """
        from https://github.com/undark-lab/sstrax
        """
        jacobian_force = jax.jacfwd(self.gradient)
        return jacobian_force(xyz, t)

    @partial(jax.jit,static_argnums=(0,))
    def dphidr(self, x, t):
        """
        Radial derivative of the potential
        """
        rad = jnp.linalg.norm(x)
        r_hat = x/rad
        return jnp.sum(self.gradient(x,t)*r_hat)

    @partial(jax.jit,static_argnums=(0,))
    def d2phidr2(self, x, t):
        """
        Second radial derivative of the potential
        from https://github.com/undark-lab/sstrax
        """
        rad = jnp.linalg.norm(x)
        r_hat = x/rad
        dphi_dr_func = lambda x: jnp.sum(self.gradient(x,t)*r_hat)
        return jnp.sum(jax.grad(dphi_dr_func)(x)*r_hat)
        

    @partial(jax.jit,static_argnums=(0,))
    def omega(self, x,v):
        """
        Computes angular velocity 
        from https://github.com/undark-lab/sstrax
        """
        rad = jnp.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        omega_vec = jnp.cross(x, v) / (rad**2)
        return jnp.linalg.norm(omega_vec)

    @partial(jax.jit,static_argnums=(0,))
    def tidalr(self, x, v, Msat, t):
        """
        Computes the tidal radius of a cluster
        from https://github.com/undark-lab/sstrax
        """
        return (self._G * Msat / ( self.omega(x, v) ** 2 - self.d2phidr2(x, t)) ) ** (1.0 / 3.0)
    
    @partial(jax.jit,static_argnums=(0,))
    def lagrange_pts(self,x,v,Msat, t):
        r_tidal = self.tidalr(x,v,Msat, t)
        r_hat = x/jnp.linalg.norm(x)
        L_close = x - r_hat*r_tidal
        L_far = x + r_hat*r_tidal
        return L_close, L_far  
    

    ##################### STANDARD FIELD ###############################
    @partial(jax.jit,static_argnums=(0,))
    def velocity_acceleration(self,t,xv,args):
        x, v = xv[:3], xv[3:]
        acceleration = -self.gradient(x,t)
        return jnp.hstack([v,acceleration])
    
    #################### Orbit integrator ###########################

    #@eqx.filter_jit
    @partial(jax.jit,static_argnames=('self','dense','solver','max_steps','rtol','atol','steps','jump_ts', 'throw'))
    def integrate_orbit(self,w0=None,ts=None, dense=False, solver=diffrax.Dopri8(scan_kind='bounded'),rtol=1e-7, atol=1e-7, dtmin=0.3,dtmax=None,max_steps=10_000, t0=None, t1=None,steps=False,jump_ts=None,throw=False):
        """
        Integrate orbit associated with potential function.
        w0: length 6 array [x,y,z,vx,vy,vz]
        ts: array of saved times. Must be at least length 2, specifying a minimum and maximum time. This does _not_ determine the timestep
        dense: boolean array.  When False, return orbit at times ts. When True, return dense interpolation of orbit between ts.min() and ts.max()
        solver: integrator
        rtol, atol: tolerance for PIDController, adaptive timestep
        dtmin: minimum timestep (in Myr)
        max_steps: maximum number of allowed timesteps
        throw: whether to throw an error if the integrator fails. False by default. Returns will be infs upon failure. If throw is True and integration fails, raise exception.
        """
          
        term = ODETerm(self.velocity_acceleration)
        
        saveat = SaveAt(t0=False, t1=False,ts= ts if not dense else None,dense=dense, steps=steps)
        
        
        rtol: float = rtol  
        atol: float = atol  
        stepsize_controller = PIDController(rtol=rtol, atol=atol, dtmin=dtmin,dtmax=dtmax,force_dtmin=True, jump_ts=jump_ts)
        #max_steps: int = max_steps
        max_steps = int(max_steps)
        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=ts.min() if t0 is None else t0,
            t1=ts.max() if t1 is None else t1,
            y0=w0,
            dt0=None,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=None,
            max_steps=max_steps,
            adjoint=ForwardMode(),
            throw= throw
        )
        return solution

    #@eqx.filter_jit
    @partial(jax.jit,static_argnames=('self','dense','solver','max_steps','steps','jump_ts','rtol','atol'))
    def integrate_orbit_batch_scan(self, w0=None,ts=None, dense=False, solver=diffrax.Dopri8(scan_kind='bounded'),rtol=1e-7, atol=1e-7, dtmin=0.3,dtmax=None,max_steps=10_000, t0=None, t1=None,steps=False,jump_ts=None,):
        """
        Integrate a batch of orbits using scan [best for CPU usage]
        w0: shape (N,6) array of initial conditions
        ts: array of saved times. Can eithe be 1D array (same for all trajectories), or N x M array, where M is the number of saved times for each trajectory
        """
        @jax.jit
        def body(carry, t):
            i = carry[0]
            w0_curr = w0[i]
            ts_curr = ts if len(ts.shape) == 1 else ts[i]
            sol = self.integrate_orbit(w0=w0_curr,ts=ts_curr, dense=dense, solver=solver,rtol=rtol, atol=atol, dtmin=dtmin,dtmax=dtmax,max_steps=max_steps, t0=t0, t1=t1, steps=steps,jump_ts=jump_ts,)
            return [i+1], sol 
    
        init_carry = [0]
        final_state, all_states = jax.lax.scan(body, init_carry, jnp.arange(len(w0)))
        return all_states

    #@eqx.filter_jit
    @partial(jax.jit,static_argnames=('self','dense','solver','max_steps','steps','jump_ts','rtol','atol'))
    def integrate_orbit_batch_vmapped(self, w0=None, ts=None, dense=False, solver=diffrax.Dopri8(scan_kind='bounded'), rtol=1e-7, atol=1e-7, dtmin=0.3, dtmax=None, max_steps=10_000, t0=None, t1=None, steps=False, jump_ts=None):
        """
        Integrate a batch of orbits using vmap [best for GPU usage]
        w0: shape (N,6) array of initial conditions
        ts: array of saved times. Can either be 1D array (same for all trajectories), or N x M array, where M is the number of saved times for each trajectory
        """
        integrator = lambda w0, ts: self.integrate_orbit(w0=w0, ts=ts, dense=dense, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, dtmax=dtmax, max_steps=max_steps, t0=t0, t1=t1, steps=steps, jump_ts=jump_ts)
        
        if len(ts.shape) == 1:
            func = lambda w0: integrator(w0, ts)
            integrator_mapped = jax.vmap(func, in_axes=(0,))
        else:
            func = jax.vmap(integrator, in_axes=(0, 0))
            integrator_mapped = lambda w0: func(w0, ts)
        
        return integrator_mapped(w0)
            

   
    ################### Stream Model ######################
    
    #@eqx.filter_jit
    @partial(jax.jit,static_argnums=(0,))
    def release_model(self, x=None, v=None, Msat=None, i=None, t=None, seed_num=None, kval_arr = 1.0):
        # if kval_arr is a scalar, then we assume the default values of kvals
        pred = jnp.isscalar(kval_arr)
        def true_func():
            return jnp.array([2.0, 0.3, 0.0, 0.0, 0.4, 0.4, 0.5, 0.5]) #jnp.array([2.0, 0.3, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])#
        def false_func():
            return jnp.ones(8)*kval_arr
        kval_arr = jax.lax.cond(pred, true_func, false_func)
        kr_bar, kvphi_bar, kz_bar, kvz_bar, sigma_kr, sigma_kvphi, sigma_kz, sigma_kvz = kval_arr
        #kr_bar=2.0, kvphi_bar=0.3, kz_bar=0.0, kvz_bar=0.0, sigma_kr=0.5, sigma_kvphi=0.5, sigma_kz=0.5, sigma_kvz=0.5):
        key_master = jax.random.PRNGKey(seed_num)
        random_ints = jax.random.randint(key=key_master,shape=(5,),minval=0,maxval=1000)

        keya = jax.random.PRNGKey(i*random_ints[0])
        keyb = jax.random.PRNGKey(i*random_ints[1])
        
        keyc = jax.random.PRNGKey(i*random_ints[2])
        keyd = jax.random.PRNGKey(i*random_ints[3])
        keye = jax.random.PRNGKey(i*random_ints[4])
        
        L_close, L_far = self.lagrange_pts(x,v,Msat, t) # each is an xyz array
        
        omega_val = self.omega(x,v)
        
        
        r = jnp.linalg.norm(x)
        r_hat = x/r
        r_tidal = self.tidalr(x,v,Msat, t)
        rel_v = omega_val*r_tidal #relative velocity
        
        #circlar_velocity
        dphi_dr = jnp.sum(self.gradient(x, t)*r_hat)
        v_circ = rel_v##jnp.sqrt( r*dphi_dr )
        
        L_vec = jnp.cross(x,v)
        z_hat = L_vec / jnp.linalg.norm(L_vec)
        
        phi_vec = v - jnp.sum(v*r_hat)*r_hat
        phi_hat = phi_vec/jnp.linalg.norm(phi_vec)
        vt_sat = jnp.sum(v*phi_hat)
        
        
        ##kr_bar = 2.0
        ##kvphi_bar = 0.3
        
        ##kz_bar = 0.0
        ##kvz_bar = 0.0
        
        ##sigma_kr = 0.5
        ##sigma_kvphi = 0.5
        ##sigma_kz = 0.5
        ##sigma_kvz = 0.5
        
        kr_samp =  kr_bar + jax.random.normal(keya,shape=(1,))*sigma_kr
        kvphi_samp = kr_samp*(kvphi_bar  + jax.random.normal(keyb,shape=(1,))*sigma_kvphi)
        kz_samp = kz_bar + jax.random.normal(keyc,shape=(1,))*sigma_kz
        kvz_samp = kvz_bar + jax.random.normal(keyd,shape=(1,))*sigma_kvz
        
        ## Trailing arm
        pos_trail = x + kr_samp*r_hat*(r_tidal) #nudge out
        pos_trail  = pos_trail + z_hat*kz_samp*(r_tidal/1.0)#r #nudge above/below orbital plane
        v_trail = v + (0.0 + kvphi_samp*v_circ*(1.0))*phi_hat#v + (0.0 + kvphi_samp*v_circ*(-r_tidal/r))*phi_hat #nudge velocity along tangential direction
        v_trail = v_trail + (kvz_samp*v_circ*(1.0))*z_hat#v_trail + (kvz_samp*v_circ*(-r_tidal/r))*z_hat #nudge velocity along vertical direction
        
        ## Leading arm
        pos_lead = x + kr_samp*r_hat*(-r_tidal) #nudge in
        pos_lead  = pos_lead + z_hat*kz_samp*(-r_tidal/1.0)#r #nudge above/below orbital plane
        v_lead = v + (0.0 + kvphi_samp*v_circ*(-1.0))*phi_hat#v + (0.0 + kvphi_samp*v_circ*(r_tidal/r))*phi_hat #nudge velocity along tangential direction
        v_lead = v_lead + (kvz_samp*v_circ*(-1.0))*z_hat#v_lead + (kvz_samp*v_circ*(r_tidal/r))*z_hat #nudge velocity against vertical direction
        
        return pos_lead, pos_trail, v_lead, v_trail
    
    #@eqx.filter_jit
    @partial(jax.jit,static_argnames=('self','solver','rtol','atol','max_steps'))
    def gen_stream_ics(self, ts=None, prog_w0=None, Msat=None, seed_num=None, solver=diffrax.Dopri5(scan_kind='bounded'),kval_arr=1.0, rtol=1e-7, atol=1e-7, dtmin=0.3,dtmax=None,max_steps=10_000):
        ws_jax = self.integrate_orbit(w0=prog_w0,ts=ts,solver=solver, rtol=rtol, atol=atol, dtmin=dtmin,dtmax=dtmax,max_steps=max_steps).ys
        Msat = Msat*jnp.ones(len(ts))

    
        @jax.jit
        def body_func(i):
            """
            body function to vmap over
            """
            pos_close_new, pos_far_new, vel_close_new, vel_far_new = self.release_model(x=ws_jax[i,:3], v=ws_jax[i,3:], Msat=Msat[i], i=i, t=ts[i], seed_num=seed_num, kval_arr=kval_arr)
            return [pos_close_new, pos_far_new, vel_close_new, vel_far_new]
        
            
        iterator_arange = jnp.arange(len(ts))
        all_states = jax.vmap(body_func)(iterator_arange)
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = all_states

        return pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr
    
            
    #@eqx.filter_jit
    @partial(jax.jit,static_argnames=('self','solver','rtol','atol','max_steps'))
    def gen_stream_scan(self, ts=None, prog_w0=None, Msat=None, seed_num=None, solver=diffrax.Dopri5(scan_kind='bounded'), kval_arr=1.0, rtol=1e-7, atol=1e-7, dtmin=0.3,dtmax=None,max_steps=10_000):
        """
        Generate stellar stream by scanning over the release model/integration. Better for CPU usage.
        pass in kwargs for the orbit integrator
        """
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts=ts, prog_w0=prog_w0, Msat=Msat, seed_num=seed_num, kval_arr=kval_arr,rtol=rtol, atol=atol, dtmin=dtmin,dtmax=dtmax,max_steps=max_steps)
        orb_integrator = lambda w0, ts: self.integrate_orbit(w0=w0, ts=ts,solver=solver,rtol=rtol, atol=atol, dtmin=dtmin,dtmax=dtmax,max_steps=max_steps).ys[-1]
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
        # Particle ids is one less than len(ts): ts[-1] defines final time to integrate up to... the observed time
        particle_ids = jnp.arange(len(pos_close_arr)-1)
        final_state, all_states = jax.lax.scan(scan_fun, init_carry, particle_ids)
        lead_arm, trail_arm = all_states
        return lead_arm, trail_arm
    
    #@eqx.filter_jit
    @partial(jax.jit,static_argnames=('self','solver','rtol','atol','max_steps', 'throw'))
    def gen_stream_vmapped(self, ts=None, prog_w0=None, Msat=None, seed_num=None, solver=diffrax.Dopri5(scan_kind='bounded') , kval_arr=1.0, rtol=1e-7, atol=1e-7, dtmin=0.3,dtmax=None,max_steps=10_000, throw=False):
        """
        Generate stellar stream by vmapping over the release model/integration. Better for GPU usage.
        """
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts=ts, prog_w0=prog_w0, Msat=Msat, seed_num=seed_num, solver=solver, kval_arr=kval_arr, rtol=rtol, atol=atol, dtmin=dtmin,dtmax=dtmax,max_steps=max_steps)
        orb_integrator = lambda w0, ts: self.integrate_orbit(w0=w0, ts=ts, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin,dtmax=dtmax,max_steps=max_steps, throw=throw).ys[-1]
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

    ################### Dense Stream Model ######################
  
    #@eqx.filter_jit
    @partial(jax.jit,static_argnames=('self','solver','rtol','atol','max_steps'))
    def gen_stream_scan_dense(self, ts=None, prog_w0=None, Msat=None, seed_num=None,solver=diffrax.Dopri5(scan_kind='bounded'), kval_arr=1.0, rtol=1e-7, atol=1e-7, dtmin=0.3,dtmax=None,max_steps=10_000):
        """
        Generate dense stellar stream model by scanning over the release model/integration. Better for CPU usage.
        pass in kwargs for the orbit integrator
        Dense means we can access the stream model at anytime from ts.min() to ts.max() via an interpolation of orbits
        """
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts=ts, prog_w0=prog_w0, Msat=Msat, seed_num=seed_num, solver=solver, kval_arr=kval_arr, rtol=rtol, atol=atol, dtmin=dtmin,dtmax=dtmax,max_steps=max_steps)
        orb_integrator = lambda w0, ts: self.integrate_orbit(w0=w0, ts=ts, dense=True, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin,dtmax=dtmax,max_steps=max_steps)
        orb_integrator_mapped = jax.jit(jax.vmap(orb_integrator,in_axes=(0,None,)))
        @jax.jit
        def scan_fun(carry, particle_idx):
            i, pos_close_curr, pos_far_curr, vel_close_curr, vel_far_curr = carry
            curr_particle_w0_close = jnp.hstack([pos_close_curr,vel_close_curr])
            curr_particle_w0_far = jnp.hstack([pos_far_curr,vel_far_curr])

            ts_arr = jnp.array([ts[i],ts[-1]])
            curr_particle_loc = jnp.vstack([curr_particle_w0_close,curr_particle_w0_far])
            w_particle = orb_integrator_mapped(curr_particle_loc, ts_arr)

         
            
            return [i+1, pos_close_arr[i+1,:], pos_far_arr[i+1,:], vel_close_arr[i+1,:], vel_far_arr[i+1,:]], [w_particle]
        init_carry = [0, pos_close_arr[0,:], pos_far_arr[0,:], vel_close_arr[0,:], vel_far_arr[0,:]]
        # Particle ids is one less than len(ts): ts[-1] defines final time to integrate up to... the observed time
        particle_ids = jnp.arange(len(pos_close_arr)-1)
        final_state, all_states = jax.lax.scan(scan_fun, init_carry, particle_ids)
        lead_arm_trail_arm = all_states
        return lead_arm_trail_arm[0]


    #@eqx.filter_jit
    @partial(jax.jit,static_argnames=('self','solver','max_steps','rtol','atol'))
    def gen_stream_vmapped_dense(self, ts=None, prog_w0=None, Msat=None, seed_num=None,solver=diffrax.Dopri5(scan_kind='bounded'), kval_arr=1.0, rtol=1e-7, atol=1e-7, dtmin=0.3,dtmax=None,max_steps=10_000):
        """
        Generate dense stellar stream by vmapping over the release model/integration. Better for GPU usage.
        Dense means we can access the stream model at anytime from ts.min() to ts.max() via an interpolation of orbits
        """
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts=ts, prog_w0=prog_w0, Msat=Msat, seed_num=seed_num, solver=solver, kval_arr=kval_arr, rtol=rtol, atol=atol, dtmin=dtmin,dtmax=dtmax,max_steps=max_steps)
        orb_integrator = lambda w0, ts: self.integrate_orbit(w0=w0, ts=ts, dense=True, solver=solver, rtol=rtol, atol=atol, dtmin=dtmin,dtmax=dtmax,max_steps=max_steps)
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
            return w_particle
        particle_ids = jnp.arange(len(pos_close_arr)-1)
        
        return jax.vmap(single_particle_integrate,in_axes=(0,0,0,0,0,))(particle_ids,pos_close_arr[:-1], pos_far_arr[:-1],vel_close_arr[:-1], vel_far_arr[:-1])
    




