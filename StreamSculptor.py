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
    
    @partial(jax.jit, static_argnums=(0,))
    def gradient(self, xyz, t):
        grad_func = jax.grad(self.potential)
        return grad_func(xyz, t)
    
    @partial(jax.jit, static_argnums=(0,))
    def density(self, xyz, t):
        lap = jnp.trace(jax.hessian(self.potential)(xyz, t))
        return lap / (4 * jnp.pi * self._G)
    
    @partial(jax.jit, static_argnums=(0,))
    def acceleration(self, xyz, t):
        return -self.gradient(xyz, t)
    
    @partial(jax.jit, static_argnums=(0,))
    def local_circular_velocity(self,xyz,t):
        r = jnp.sqrt(jnp.sum(xyz**2))
        r_hat = xyz / r
        grad_phi = self.gradient(xyz,t)
        dphi_dr = jnp.sum(grad_phi*r_hat)
        return jnp.sqrt( r*dphi_dr )
   
    @partial(jax.jit,static_argnums=(0,))
    def jacobian_force_mw(self, xyz, t):
        jacobian_force_mw = jax.jacfwd(self.gradient)
        return jacobian_force_mw(xyz, t)
    
    @partial(jax.jit,static_argnums=(0,))
    def d2phidr2_mw(self, x, t):
        """
        Second radial derivative of the potential
        """
        rad = jnp.linalg.norm(x)
        r_hat = x/rad
        dphi_dr_func = lambda x: jnp.sum(self.gradient(x,t)*r_hat)
        return jnp.sum(jax.grad(dphi_dr_func)(x)*r_hat)
        

    @partial(jax.jit,static_argnums=(0,))
    def omega(self, x,v):
        """
        Computes the magnitude of the angular momentum in the simulation frame
        Args:
          x: 3d position (x, y, z) in [kpc]
          v: 3d velocity (v_x, v_y, v_z) in [kpc/Myr]
        Returns:
          Magnitude of angular momentum in [rad/Myr]
        Examples
        --------
        >>> omega(x=jnp.array([8.0, 0.0, 0.0]), v=jnp.array([8.0, 0.0, 0.0]))
        """
        rad = jnp.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        omega_vec = jnp.cross(x, v) / (rad**2)
        return jnp.linalg.norm(omega_vec)

    @partial(jax.jit,static_argnums=(0,))
    def tidalr_mw(self, x, v, Msat, t):
        """
        Computes the tidal radius of a cluster in the potential
        Args:
          x: 3d position (x, y, z) in [kpc]
          v: 3d velocity (v_x, v_y, v_z) in [kpc/Myr]
          Msat: Cluster mass in [Msol]
        Returns:
          Tidal radius of the cluster in [kpc]
        Examples
        --------
        >>> tidalr_mw(x=jnp.array([8.0, 0.0, 0.0]), v=jnp.array([8.0, 0.0, 0.0]), Msat=1e4)
        """
        return (self._G * Msat / ( self.omega(x, v) ** 2 - self.d2phidr2_mw(x, t)) ) ** (1.0 / 3.0)
    
    @partial(jax.jit,static_argnums=(0,))
    def lagrange_pts(self,x,v,Msat, t):
        r_tidal = self.tidalr_mw(x,v,Msat, t)
        r_hat = x/jnp.linalg.norm(x)
        L_close = x - r_hat*r_tidal
        L_far = x + r_hat*r_tidal
        return L_close, L_far  
    

    ##################### FIELDS ###############################
    @partial(jax.jit,static_argnums=(0,))
    def velocity_acceleration(self,t,xv,args):
        x, v = xv[:3], xv[3:]
        acceleration = -self.gradient(x,t)
        return jnp.hstack([v,acceleration])
    
    #################### Orbit integrator ###########################

    @partial(jax.jit,static_argnums=((0,3,4,5,6,7,8,9,16,17,18,19,)))
    def integrate_orbit(self,w0=None,ts=None, dense=False, solver=diffrax.Dopri8(scan_kind='bounded'),rtol=1e-7, atol=1e-7, dtmin=0.3,dtmax=None,max_steps=10_000, t0=None, t1=None,dt0=0.5,pcoeff=0.4, icoeff=0.3,dcoeff=0, factormin=.2,factormax=10.0,safety=0.9,steps=False,jump_ts=None,):
        """
        Integrate orbit associated with potential function.
        w0: length 6 array [x,y,z,vx,vy,vz]
        ts: array of saved times. Must be at least length 2, specifying a minimum and maximum time. This does _not_ determine the timestep
        dense: boolean array.  When False, return orbit at times ts. When True, return dense interpolation of orbit between ts.min() and ts.max()
        solver: integrator
        rtol, atol: tolerance for PIDController, adaptive timestep
        dtmin: minimum timestep (in Myr)
        max_steps: maximum number of allowed timesteps
        step_controller: 0 for PID (adaptive), 1 for constant timestep (must then specify dt0)
        """
        
        dt0_sign = jnp.sign(ts.max() - ts.min()) if t0 is None else jnp.sign(t1 - t0)
        dt0 = dt0*dt0_sign

       
        term = ODETerm(self.velocity_acceleration)
        
        saveat = SaveAt(t0=False, t1=False,ts= ts if not dense else None,dense=dense, steps=steps)
        
        
        rtol: float = rtol  
        atol: float = atol  
        stepsize_controller = PIDController(rtol=rtol, atol=atol, dtmin=dtmin,dtmax=dtmax,pcoeff=pcoeff, icoeff=icoeff, dcoeff=dcoeff,factormin=factormin,factormax=factormax,safety=safety,force_dtmin=True, jump_ts=jump_ts)
        max_steps: int = max_steps
        #max_steps = int(max_steps)
               
        

        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=ts.min() if t0 is None else t0,
            t1=ts.max() if t1 is None else t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=None,
            max_steps=max_steps,
            adjoint=DirectAdjoint()
        )
        return solution
    

    @partial(jax.jit,static_argnums=((0,2)))
    def get_adaptive_timesteps(self,w0=None,solver=diffrax.Dopri8(scan_kind='bounded'),rtol=1e-7, atol=1e-7, dtmin=0.05,dtmax=None,max_steps=10_000, t0=None, t1=None,dt0=0.5,pcoeff=0.4, icoeff=0.3,dcoeff=0):
        """
        Inputs are the same as orbit integrator
        Outputs the timesteps taken by the adpative timestepper, dependent on tolerances and the solver choice.
        """
        ts = self.integrate_orbit(w0=w0,solver=solver,rtol=rtol,atol=atol,dtmin=dtmin,dtmax=dtmax,max_steps=max_steps,t0=t0,t1=t1,dt0=dt0,pcoeff=pcoeff, icoeff=icoeff,dcoeff=dcoeff,steps=True).ts
        return jnp.hstack([t0,ts])
    

    @partial(jax.jit,static_argnums=((0,3,4,6,)))
    def integrate_orbit_const_step(self,w0=None,ts=None, dense=False, solver=diffrax.Dopri8(scan_kind='bounded'),dt0=None,max_steps=10_000, t0=None, t1=None):
        """
        Integrate orbit associated with potential function.
        w0: length 6 array [x,y,z,vx,vy,vz]
        ts: array of saved times. Must be at least length 2, specifying a minimum and maximum time. This does _not_ determine the timestep
        dense: boolean array.  When False, return orbit at times ts. When True, return dense interpolation of orbit between ts.min() and ts.max()
        solver: integrator
        rtol, atol: tolerance for PIDController, adaptive timestep
        dtmin: minimum timestep (in Myr)
        max_steps: maximum number of allowed timesteps
        """
       
        term = ODETerm(self.velocity_acceleration)
        
        saveat = SaveAt(t0=False, t1=False,ts= ts if not dense else None,dense=dense)
        
      
        stepsize_controller = diffrax.ConstantStepSize()
        max_steps: int = max_steps
       

        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=ts.min() if t0 is None else t0,
            t1=ts.max() if t1 is None else t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=None,
            max_steps=max_steps,
            adjoint=DirectAdjoint()
        )
        return solution

    @partial(jax.jit,static_argnums=((0,3,4,6,)))
    def integrate_orbit_stepto(self,w0=None,ts=None, dense=False, solver=diffrax.Dopri8(scan_kind='bounded'),dt0=None,max_steps=100_000, ts_stepto=None, t0=None, t1=None, t_terminate=None):
        """
        Integrate orbit associated with potential function.
        w0: length 6 array [x,y,z,vx,vy,vz]
        ts: array of saved times. Must be at least length 2, specifying a minimum and maximum time. This does _not_ determine the timestep
        dense: boolean array.  When False, return orbit at times ts. When True, return dense interpolation of orbit between ts.min() and ts.max()
        solver: integrator
        rtol, atol: tolerance for PIDController, adaptive timestep
        dtmin: minimum timestep (in Myr)
        max_steps: maximum number of allowed timesteps
        """
        t_terminate = t1 if t_terminate is None else t_terminate

        def event_fun(t,y,args,**kwargs):
            """
            solver will terminate when true
            """
            return t>=t_terminate

       
        term = ODETerm(self.velocity_acceleration)
        
        saveat = SaveAt(t0=False, t1=False,ts=ts if not dense else None,dense=dense)
        
      
        stepsize_controller = diffrax.StepTo(ts=ts_stepto)
        max_steps: int = max_steps
       
        terminating_event =  diffrax.Event(event_fun)

        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=ts.min() if t0 is None else t0,
            t1=ts.max() if t1 is None else t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            event=terminating_event,#DiscreteTerminatingEvent(event_fun),
            max_steps=max_steps,
            adjoint=DirectAdjoint()
        )
        return solution



    @partial(jax.jit,static_argnums=((0,4,6,)))
    def integrate_orbit_collisional(self,w0=None,ts=None,solver=diffrax.Dopri8(scan_kind='bounded'),tcollision=None,dt0_collision=None,collision_twindow=None,rtol=1e-7,atol=1e-7,dtmin=0.1,t0=None, t1=None):
        """
        Orbit integrator with an adaptive timestep routine to handle collisional dynamics
        Assuming we have a collision, where extra timesteps will be taken.
        Outside of collisional window, the adaptive timestepper takes over.
        w0: length 6 array [x,y,z,vx,vy,vz]
        ts: array of saved times. Must be at least length 2, specifying a minimum and maximum time. This does _not_ determine the timestep
        dense: boolean array.  When False, return orbit at times ts. When True, return dense interpolation of orbit between ts.min() and ts.max()
        solver: integrator
        rtol, atol: tolerance for PIDController, adaptive timestep
        dtmin: minimum timestep (in Myr)
        max_steps: maximum number of allowed timesteps
        """
        if t0 is None:
            t0 = ts.min()
        if t1 is None:
            t1 = ts.max()

        t_lower = jnp.clip(tcollision - collision_twindow,t0,t1)
        t_upper = jnp.clip(tcollision + collision_twindow,t0,t1)
        
        w_intermediate = self.integrate_orbit(w0=w0, ts=jnp.array([t0, t_lower]), dense=False, rtol=rtol, atol=atol, dtmin=dtmin).ys[1]
        w_after_collision = self.integrate_orbit_const_step(w0=w_intermediate, ts=jnp.array([t_lower, t_upper]), dense=False,dt0=dt0_collision).ys[1]
        w_final = self.integrate_orbit(w0=w_after_collision, ts=jnp.array([t_upper, t1]), dense=False,rtol=rtol, atol=atol, dtmin=dtmin)

        return w_final.ys[1]


    @partial(jax.jit,static_argnums=((0,3,)))
    def integrate_orbit_collisional_constant_stepper(self,w0=None,ts=None,solver=diffrax.Dopri8(scan_kind='bounded'),dt0=None, tcollision=None,dt0_collision=None,collision_twindow=None,rtol=1e-7,atol=1e-7,dtmin=0.1,t0=None, t1=None):
        """
        Orbit integrator with an adaptive timestep routine to handle collisional dynamics
        Assuming we have a collision, where extra timesteps will be taken.
        Outside of collisional window, the adaptive timestepper takes over.
        w0: length 6 array [x,y,z,vx,vy,vz]
        ts: array of saved times. Must be at least length 2, specifying a minimum and maximum time. This does _not_ determine the timestep
        dense: boolean array.  When False, return orbit at times ts. When True, return dense interpolation of orbit between ts.min() and ts.max()
        solver: integrator
        rtol, atol: tolerance for PIDController, adaptive timestep
        dtmin: minimum timestep (in Myr)
        max_steps: maximum number of allowed timesteps
        **Outside collision window assumes a constant timestep of dt0**
        **Inside collision window assumes a constant timestep of dt0_collision**
        """
        if t0 is None:
            t0 = ts.min()
        if t1 is None:
            t1 = ts.max()

        # check if final time is greater than initial
        # if not (backwards solve), ensure timesteps have correct sign
        # to take us from t0 --> t1, even if t0 > t1.
        pred = t1 > t0
        def true_func(dt0_dt0coll):
            """
            Apply this branch if pred is True
            """
            dt0, dt0_collision = dt0_dt0coll
            dt0 = jnp.abs(dt0)
            dt0_collision = jnp.abs(dt0_collision)
            return dt0, dt0_collision
        def false_func(dt0_dt0coll):
            """
            Apply this branch if pred is False
            """
            dt0, dt0_collision = dt0_dt0coll
            dt0 = jnp.abs(dt0)
            dt0_collision = jnp.abs(dt0_collision)
            return -dt0, -dt0_collision
        dt0, dt0_collision = jax.lax.cond(pred, true_func, false_func,jnp.array([dt0, dt0_collision]))


        t_lower = jnp.clip(tcollision - collision_twindow,t0,t1)
        t_upper = jnp.clip(tcollision + collision_twindow,t0,t1)

        @jax.jit
        def get_timestep(tcurr):
            def in_collision():
                return dt0_collision
            def out_collision():
                return dt0
            # check if tcurr is in any collision interval
            # true if in collision interval, false otherwise
            pred = jnp.any((t_lower <= tcurr) & (tcurr <= t_upper))
            dtcurr = jax.lax.cond(pred, in_collision, out_collision)
            return dtcurr

        term = ODETerm(self.velocity_acceleration)



        tprev = t0
        tnext = t0 + get_timestep(t0)
        args = None
        w = w0
        state = solver.init(term, tprev, tnext, w0, args)
        
        @jax.jit
        def cond_check(curr_carry):
            w, state, tprev, tnext = curr_carry
            return tprev < t1
        @jax.jit
        def body_func(carry):
            w, state, tprev, tnext = carry
            w, _, _, state, _ = solver.step(term, tprev, tnext, w, args, state, made_jump=False)
            tprev = tnext
            tnext = jnp.minimum(tprev + get_timestep(tprev), t1)
            next_carry = [w, state, tprev, tnext]
            return next_carry

        init_val = [w, state, tprev, tnext]
        return jax.lax.while_loop(cond_check, body_func, init_val)


    @partial(jax.jit,static_argnums=((0,3,)))
    def integrate_orbit_collisional_stepper(self,w0=None,ts=None,solver=diffrax.Dopri8(scan_kind='bounded'),ts_stepto_base=None, tcollision=None,dt0_collision=None,collision_twindow=None,t0=None, t1=None):
        """
        Orbit integrator with an adaptive timestep routine to handle collisional dynamics
        Assuming we have a collision, where extra timesteps will be taken.
        Outside of collisional window, the adaptive timestepper takes over.
        w0: length 6 array [x,y,z,vx,vy,vz]
        ts: array of saved times. Must be at least length 2, specifying a minimum and maximum time. This does _not_ determine the timestep
        dense: boolean array.  When False, return orbit at times ts. When True, return dense interpolation of orbit between ts.min() and ts.max()
        solver: integrator
        rtol, atol: tolerance for PIDController, adaptive timestep
        dtmin: minimum timestep (in Myr)
        max_steps: maximum number of allowed timesteps
        """
        if t0 is None:
            t0 = ts.min()
        if t1 is None:
            t1 = ts.max()

        # check if final time is greater than initial
        # if not (backwards solve), ensure timesteps have correct sign
        # to take us from t0 --> t1, even if t0 > t1.
        pred = t1 > t0
        def true_func(dt0coll):
            """
            Apply this branch if pred is True
            """
            dt0_collision = jnp.abs(dt0coll)
            return dt0_collision
        def false_func(dt0coll):
            """
            Apply this branch if pred is False
            """
            dt0_collision = jnp.abs(dt0coll)
            return -dt0_collision
        dt0_collision = jax.lax.cond(pred, true_func, false_func,dt0_collision)


        t_lower = jnp.clip(tcollision - collision_twindow,t0,t1)
        t_upper = jnp.clip(tcollision + collision_twindow,t0,t1)

        @jax.jit
        def get_timestep(tcurr):
            def in_collision(ts_stepto_base):
                return dt0_collision
            def out_collision(ts_stepto_base):
                curr_time_ind = jnp.argmin(jnp.abs(tcurr - ts_stepto_base))
                dt = ts_stepto_base[curr_time_ind+1] - ts_stepto_base[curr_time_ind]
                return dt
            # check if tcurr is in any collision interval
            # true if in collision interval, false otherwise
            pred = jnp.any((t_lower <= tcurr) & (tcurr <= t_upper))
            dtcurr = jax.lax.cond(pred, in_collision, out_collision, ts_stepto_base)
            return dtcurr

        term = ODETerm(self.velocity_acceleration)



        tprev = t0
        tnext = t0 + get_timestep(t0)
        args = None
        w = w0
        state = solver.init(term, tprev, tnext, w0, args)
        
        @jax.jit
        def cond_check(curr_carry):
            w, state, tprev, tnext = curr_carry
            return tprev < t1
        @jax.jit
        def body_func(carry):
            w, state, tprev, tnext = carry
            w, _, _, state, _ = solver.step(term, tprev, tnext, w, args, state, made_jump=False)
            tprev = tnext
            tnext = jnp.minimum(tprev + get_timestep(tprev), t1)
            next_carry = [w, state, tprev, tnext]
            return next_carry

        init_val = [w, state, tprev, tnext]
        return jax.lax.while_loop(cond_check, body_func, init_val)






    
   
    ################### Stream Model ######################
    
    @partial(jax.jit,static_argnums=(0,))
    def release_model(self, x=None, v=None, Msat=None, i=None, t=None, seed_num=None, kval_arr = 1.0):
        # if kval_arr is a scalar, then we assume the default values of kvals
        pred = jnp.isscalar(kval_arr)
        def true_func():
            return jnp.array([2.0, 0.3, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
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
        r_tidal = self.tidalr_mw(x,v,Msat, t)
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
    
    @partial(jax.jit,static_argnums=(0,5))
    def gen_stream_ics(self, ts=None, prog_w0=None, Msat=None, seed_num=None, solver=diffrax.Dopri5(scan_kind='bounded'),kval_arr=1.0, **kwargs):
        ws_jax = self.integrate_orbit(w0=prog_w0,ts=ts,solver=solver, **kwargs).ys
        Msat = Msat*jnp.ones(len(ts))

        def scan_fun(carry, t):
            i, pos_close, pos_far, vel_close, vel_far = carry
            pos_close_new, pos_far_new, vel_close_new, vel_far_new = self.release_model(x=ws_jax[i,:3], v=ws_jax[i,3:], Msat=Msat[i], i=i, t=t, seed_num=seed_num, kval_arr=kval_arr)
            return [i+1, pos_close_new, pos_far_new, vel_close_new, vel_far_new], [pos_close_new, pos_far_new, vel_close_new, vel_far_new]
            
            
        init_carry = [0, jnp.array([0.0,0.0,0.]), jnp.array([0.0,0.0,0.]), jnp.array([0.0,0.0,0.]), jnp.array([0.0,0.0,0.])] 
        final_state, all_states = jax.lax.scan(scan_fun, init_carry, ts)
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = all_states
        return pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr
    
            
    @partial(jax.jit,static_argnums=(0,5))
    def gen_stream_scan(self, ts=None, prog_w0=None, Msat=None, seed_num=None, solver=diffrax.Dopri5(scan_kind='bounded'), kval_arr=1.0, **kwargs):
        """
        Generate stellar stream by scanning over the release model/integration. Better for CPU usage.
        pass in kwargs for the orbit integrator
        """
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts=ts, prog_w0=prog_w0, Msat=Msat, seed_num=seed_num, kval_arr=kval_arr, **kwargs)
        orb_integrator = lambda w0, ts: self.integrate_orbit(w0=w0, ts=ts,solver=solver,**kwargs).ys[-1]
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
    
    @partial(jax.jit,static_argnums=((0,5)))
    def gen_stream_vmapped(self, ts=None, prog_w0=None, Msat=None, seed_num=None, solver=diffrax.Dopri5(scan_kind='bounded') , kval_arr=1.0, **kwargs):
        """
        Generate stellar stream by vmapping over the release model/integration. Better for GPU usage.
        """
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts=ts, prog_w0=prog_w0, Msat=Msat, seed_num=seed_num, solver=solver, kval_arr=kval_arr, **kwargs)
        orb_integrator = lambda w0, ts: self.integrate_orbit(w0=w0, ts=ts, solver=solver, **kwargs).ys[-1]
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

    
    
    @partial(jax.jit,static_argnums=((0,)))
    def gen_stream_vmapped_const_step(self, ts=None, prog_w0=None, Msat=None, dt0=None, seed_num=None, **kwargs):
        """
        Generate stellar stream by vmapping over the release model/integration. Better for GPU usage.
        Constant timestep dt0
        """
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts, prog_w0, Msat, seed_num, **kwargs)
        orb_integrator = lambda w0, ts: self.integrate_orbit_const_step(w0=w0, ts=ts, dt0=dt0,**kwargs).ys[-1]
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
    
    @partial(jax.jit,static_argnums=((0,)))
    def gen_stream_scan_stepto(self,ts=None, prog_w0=None, Msat=None, seed_num=None, solver=diffrax.Dopri8(),ts_stepto=None,t0=None,t1=None, **kwargs):
        """
        Generate stellar stream by scanning over the release model/integration. Better for CPU usage.
        stepto indicates that the timesteps are pre-specified with the ts_stepto argument
        """
        
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts, prog_w0, Msat, seed_num, **kwargs)
        t1_fix = t1
        #take .ys[-2] since .ys[-1] will be inf for ts_arr of length 3 due to terminating event
        orb_integrator = lambda w0, ts, t0, t1, ts_stepto: self.integrate_orbit_stepto(w0=w0, ts=ts, t0=t0,t1=t1, solver=solver, ts_stepto=ts_stepto, t_terminate=t1_fix,**kwargs).ys[-2]
        orb_integrator_mapped = jax.jit(jax.vmap(orb_integrator,in_axes=(0,None,None,None,None)))

        particle_spray_steps = lambda stripping_time: get_particle_stepto(ts_stepto, stripping_time, t1)

        @jax.jit
        def scan_fun(carry, particle_idx):
            i, pos_close_curr, pos_far_curr, vel_close_curr, vel_far_curr = carry
            curr_particle_w0_close = jnp.hstack([pos_close_curr,vel_close_curr])
            curr_particle_w0_far = jnp.hstack([pos_far_curr,vel_far_curr])

            curr_particle_loc = jnp.vstack([curr_particle_w0_close,curr_particle_w0_far])

            stepto_ts_curr = particle_spray_steps(ts[i])
            ts_arr = jnp.array([stepto_ts_curr[0],t1_fix,stepto_ts_curr[-1]])
            w_particle = orb_integrator_mapped(curr_particle_loc, ts_arr, ts_arr[0], ts_arr[-1], stepto_ts_curr)

            w_particle_close = w_particle[0]
            w_particle_far =   w_particle[1]
            
            return [i+1, pos_close_arr[i+1,:], pos_far_arr[i+1,:], vel_close_arr[i+1,:], vel_far_arr[i+1,:]], [w_particle_close, w_particle_far]
        init_carry = [0, pos_close_arr[0,:], pos_far_arr[0,:], vel_close_arr[0,:], vel_far_arr[0,:]]
        # Particle ids is one less than len(ts): ts[-1] defines final time to integrate up to... the observed time
        particle_ids = jnp.arange(len(pos_close_arr)-1)
        final_state, all_states = jax.lax.scan(scan_fun, init_carry, particle_ids)
        lead_arm, trail_arm = all_states
        return lead_arm, trail_arm

    @partial(jax.jit,static_argnums=((0,)))
    def gen_stream_vmapped_stepto(self,ts=None, prog_w0=None, Msat=None, seed_num=None, solver=diffrax.Dopri8(),ts_stepto=None,t0=None,t1=None, **kwargs):
        """
        Generate stellar stream by vmapping over the release model/integration. Better for GPU usage.
        stepto indicates that the timesteps are pre-specified with the ts_stepto argument
        """
        
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts, prog_w0, Msat, seed_num, **kwargs)
        t1_fix = t1
        #take .ys[-2] since .ys[-1] will be inf for ts_arr of length 3 due to terminating event
        orb_integrator = lambda w0, ts, t0, t1, ts_stepto: self.integrate_orbit_stepto(w0=w0, ts=ts, t0=t0,t1=t1, solver=solver, ts_stepto=ts_stepto, t_terminate=t1_fix,**kwargs).ys[-2]
        orb_integrator_mapped = jax.jit(jax.vmap(orb_integrator,in_axes=(0,None,None,None,None)))

        particle_spray_steps = lambda stripping_time: get_particle_stepto(ts_stepto, stripping_time, t1)


        @jax.jit
        def single_particle_integrate(particle_number,pos_close_curr,pos_far_curr,vel_close_curr,vel_far_curr):
            curr_particle_w0_close = jnp.hstack([pos_close_curr,vel_close_curr])
            curr_particle_w0_far = jnp.hstack([pos_far_curr,vel_far_curr])
            
            curr_particle_loc = jnp.vstack([curr_particle_w0_close,curr_particle_w0_far])

            stepto_ts_curr = particle_spray_steps(ts[particle_number])
            ts_arr = jnp.array([stepto_ts_curr[0],t1_fix,stepto_ts_curr[-1]])
            w_particle = orb_integrator_mapped(curr_particle_loc, ts_arr, ts_arr[0], ts_arr[-1], stepto_ts_curr)

            w_particle_close = w_particle[0]
            w_particle_far =   w_particle[1]

            return w_particle_close, w_particle_far

        # Particle ids is one less than len(ts): ts[-1] defines final time to integrate up to... the observed time
        particle_ids = jnp.arange(len(pos_close_arr)-1)
        
        return jax.vmap(single_particle_integrate,in_axes=(0,0,0,0,0,))(particle_ids,pos_close_arr[:-1], pos_far_arr[:-1], vel_close_arr[:-1], 
        vel_far_arr[:-1])

    
  
    @partial(jax.jit,static_argnums=(0,5))
    def gen_stream_scan_dense(self, ts=None, prog_w0=None, Msat=None, seed_num=None,solver=diffrax.Dopri5(scan_kind='bounded'), kval_arr=1.0, **kwargs):
        """
        Generate dense stellar stream model by scanning over the release model/integration. Better for CPU usage.
        pass in kwargs for the orbit integrator
        Dense means we can access the stream model at anytime from ts.min() to ts.max() via an interpolation of orbits
        """
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts=ts, prog_w0=prog_w0, Msat=Msat, seed_num=seed_num, solver=solver, kval_arr=kval_arr, **kwargs)
        orb_integrator = lambda w0, ts: self.integrate_orbit(w0=w0, ts=ts, dense=True, solver=solver, **kwargs)
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


    @partial(jax.jit,static_argnums=((0,5)))
    def gen_stream_vmapped_dense(self, ts=None, prog_w0=None, Msat=None, seed_num=None,solver=diffrax.Dopri5(scan_kind='bounded'), kval_arr=1.0, **kwargs):
        """
        Generate dense stellar stream by vmapping over the release model/integration. Better for GPU usage.
        Dense means we can access the stream model at anytime from ts.min() to ts.max() via an interpolation of orbits
        """
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts=ts, prog_w0=prog_w0, Msat=Msat, seed_num=seed_num, solver=solver, kval_arr=kval_arr, **kwargs)
        orb_integrator = lambda w0, ts: self.integrate_orbit(w0=w0, ts=ts, dense=True, solver=solver, **kwargs)
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
    



################## Some helpers ####################
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

def fill_in_collision_times(ts_stepto_base=None,tcollision=None,dt0_collision=None,collision_twindow=None,t0=None,t1=None):
    """
    Function outputs a ts_stepto_array accounting for a user-defined time-step during the collision.
    ts_stepto_base: timesteps in base potential. Outside of the collision window this will remain unchanged.
    tcollision: array of collision times
    dt0_collision: timestep in collision window
    collison_twindow: symmetric buffer around tcollision. At tcollision +/- collision_twindow, dt0_collision is the timestep
    t0: starting integration time
    t1: terminating time
    """
    if t0 is None:
            t0 = ts_stepto_base.min()
    if t1 is None:
        t1 = ts_stepto_base[ts_stepto_base<jnp.inf].max()

    t_lower = jnp.clip(tcollision - collision_twindow,t0,t1)
    t_upper = jnp.clip(tcollision + collision_twindow,t0,t1)

    @jax.jit
    def get_timestep(tcurr):
        def in_collision(ts_stepto_base):
            return dt0_collision
        def out_collision(ts_stepto_base):
            curr_time_ind = jnp.argmin(jnp.abs(tcurr - ts_stepto_base))
            dt = ts_stepto_base[curr_time_ind+1] - ts_stepto_base[curr_time_ind]
            return dt
        # check if tcurr is in any collision interval
        # true if in collision interval, false otherwise
        pred = jnp.any((t_lower <= tcurr) & (tcurr <= t_upper))
        dtcurr = jax.lax.cond(pred, in_collision, out_collision, ts_stepto_base)
        return dtcurr

    tprev = t0
    #tnext = t0 + get_timestep(t0)
    ts_arr = jnp.zeros(100_000) + jnp.inf
    count = 0
    ts_arr = ts_arr.at[count].set(t0)
    count += 1
    while tprev < t1:
        tnext = jnp.minimum(tprev + get_timestep(tprev), t1)
        ts_arr = ts_arr.at[count].set(tnext)
        tprev = tnext
        count += 1
    return ts_arr[ts_arr<jnp.inf]

@jax.jit
def get_particle_stepto(ts_stepto, stripping_time, final_time):
    """
    Given a precomputed ts_stepto array, stripping time, and final time
    this function rolls ts_stepto back, clips the value to the stripping time,
    and after final time, adds arbitrary increasing times. 
    All operations leave array shapes static. 
    """
    #final time has index len(ts_stepto)-1
    idx0 = len(ts_stepto)-1

    t_ind = jnp.argmin(jnp.abs(ts_stepto-stripping_time))
    rolled_back = jnp.roll(ts_stepto,-t_ind)
    sub = stripping_time-rolled_back[0]
    sub = jnp.clip(sub, -jnp.inf, -final_time) 
   
    new_time = rolled_back + sub
    new_time = jnp.where(new_time==new_time[0], stripping_time, new_time)
    new_time = jnp.where(new_time==final_time+sub, final_time, new_time)
    newfinal_ind = idx0 - t_ind #how much to roll back final time

    idx = jnp.arange(len(new_time))
    #at index after the final time, just arbitrarily add larger increasing times above final_time.
    new_time = jnp.where(idx>newfinal_ind,jnp.cumsum(jnp.abs(new_time/1000))+final_time, new_time)

    return new_time

import potential

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



    
