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


@partial(jax.jit,static_argnums=(0,))
def release_model(self, x, v, Msat, i, t, seed_num):
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
    
    
    kr_bar = 2.0
    kvphi_bar = 0.3
    ####################kvt_bar = 0.3 ## FROM GALA
    
    kz_bar = 0.0
    kvz_bar = 0.0
    
    sigma_kr = 0.5
    sigma_kvphi = 0.5
    sigma_kz = 0.5
    sigma_kvz = 0.5
    ##############sigma_kvt = 0.5 ##FROM GALA
    
    kr_samp =  kr_bar + jax.random.normal(keya,shape=(1,))*sigma_kr
    kvphi_samp = kr_samp*(kvphi_bar  + jax.random.normal(keyb,shape=(1,))*sigma_kvphi)
    kz_samp = kz_bar + jax.random.normal(keyc,shape=(1,))*sigma_kz
    kvz_samp = kvz_bar + jax.random.normal(keyd,shape=(1,))*sigma_kvz
    ########kvt_samp = kvt_bar + jax.random.normal(keye,shape=(1,))*sigma_kvt
    
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

@partial(jax.jit,static_argnums=(0,))
def gen_stream_ics(self, ts, prog_w0, Msat, seed_num):
    ws_jax = self.orbit_integrator_run_notdense_PID(prog_w0,jnp.min(ts),jnp.max(ts),ts,False)#self.orbit_integrator_run(prog_w0,jnp.min(ts),jnp.max(ts),ts,False)
    
    def scan_fun(carry, t):
        i, pos_close, pos_far, vel_close, vel_far = carry
        pos_close_new, pos_far_new, vel_close_new, vel_far_new = self.release_model(ws_jax[i,:3], ws_jax[i,3:], Msat,i, t, seed_num)
        return [i+1, pos_close_new, pos_far_new, vel_close_new, vel_far_new], [pos_close_new, pos_far_new, vel_close_new, vel_far_new]#[i+1, pos_close_new, pos_far_new, vel_close_new, vel_far_new]
        
        
    init_carry = [0, jnp.array([0.0,0.0,0.]), jnp.array([0.0,0.0,0.]), jnp.array([0.0,0.0,0.]), jnp.array([0.0,0.0,0.])] 
    final_state, all_states = jax.lax.scan(scan_fun, init_carry, ts)#[:-1])#ts[1:])
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = all_states
    return pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr

        
#####@partial(jax.jit,static_argnums=(0,))
def gen_stream_scan(self, ts, prog_w0, Msat, seed_num, steps):
    """
    Generate stellar stream by scanning over the release model/integration. Better for CPU usage.
    """
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts, prog_w0, Msat, seed_num)
    ########@jax.jit
    def scan_fun(carry, particle_idx):
        i, pos_close_curr, pos_far_curr, vel_close_curr, vel_far_curr = carry
        curr_particle_w0_close = jnp.hstack([pos_close_curr,vel_close_curr])
        curr_particle_w0_far = jnp.hstack([pos_far_curr,vel_far_curr])
        ########w0_lead_trail = jnp.vstack([curr_particle_w0_close,curr_particle_w0_far])
        
        minval, maxval =  ts[i],ts[-1]
        ####integrate_different_ics = lambda ics:  self.orbit_integrator_run(ics,minval,maxval,None, steps)#####[0]
        #####w_particle_close, w_particle_far = jax.vmap(integrate_different_ics,in_axes=(0,))(w0_lead_trail) #vmap over leading and trailing arm
        
        w_particle_close = self.orbit_integrator_run_dense(curr_particle_w0_close,minval,maxval,None,steps)###[0]
        w_particle_far = self.orbit_integrator_run_dense(curr_particle_w0_far,minval,maxval,None,steps)###[0]
        
        
        
        return [i+1, pos_close_arr[i+1,:], pos_far_arr[i+1,:], vel_close_arr[i+1,:], vel_far_arr[i+1,:]], [w_particle_close, w_particle_far]
    init_carry = [0, pos_close_arr[0,:], pos_far_arr[0,:], vel_close_arr[0,:], vel_far_arr[0,:]]
    particle_ids = jnp.arange(len(pos_close_arr))
    final_state, all_states = jax.lax.scan(scan_fun, init_carry, particle_ids)
    lead_arm, trail_arm = all_states
    return lead_arm, trail_arm

@partial(jax.jit,static_argnums=((0,5)))
def gen_stream_vmapped(self, ts, prog_w0, Msat, seed_num, steps):
    """
    Generate stellar stream by vmapping over the release model/integration. Better for GPU usage.
    """
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts, prog_w0, Msat, seed_num)
    @jax.jit
    def single_particle_integrate(particle_number,pos_close_curr,pos_far_curr,vel_close_curr,vel_far_curr):
        curr_particle_w0_close = jnp.hstack([pos_close_curr,vel_close_curr])
        curr_particle_w0_far = jnp.hstack([pos_far_curr,vel_far_curr])
        t_release = ts[particle_number]
        t_final = ts[-1] + .01
        
        w_particle_close = self.orbit_integrator_run_dense(curr_particle_w0_close,t_release,t_final,None,steps)###[0]
        w_particle_far = self.orbit_integrator_run_dense(curr_particle_w0_far,t_release,t_final,None,steps)###[0]
        
        return w_particle_close, w_particle_far
    particle_ids = jnp.arange(len(pos_close_arr))
    
    return jax.vmap(single_particle_integrate,in_axes=(0,0,0,0,0,))(particle_ids,pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr)


@partial(jax.jit,static_argnums=((0,5)))
def gen_stream_vmapped_notdense(self, ts, prog_w0, Msat, seed_num, steps):
    """
    Generate stellar stream by vmapping over the release model/integration. Better for GPU usage.
    """
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts, prog_w0, Msat, seed_num)
    @jax.jit
    def single_particle_integrate(particle_number,pos_close_curr,pos_far_curr,vel_close_curr,vel_far_curr):
        curr_particle_w0_close = jnp.hstack([pos_close_curr,vel_close_curr])
        curr_particle_w0_far = jnp.hstack([pos_far_curr,vel_far_curr])
        t_release = ts[particle_number]
        t_final = ts[-1] #################+ .01
        
        w_particle_close = self.orbit_integrator_run_notdense(curr_particle_w0_close,t_release,t_final,None,steps)###[0]
        w_particle_far = self.orbit_integrator_run_notdense(curr_particle_w0_far,t_release,t_final,None,steps)###[0]
        
        return w_particle_close, w_particle_far
    particle_ids = jnp.arange(len(pos_close_arr))
    
    return jax.vmap(single_particle_integrate,in_axes=(0,0,0,0,0,))(particle_ids,pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr)


@partial(jax.jit,static_argnums=((0,5)))
def gen_stream_vmapped_notdense_PID(self, ts, prog_w0, Msat, seed_num, steps):
    """
    Generate stellar stream by vmapping over the release model/integration. Better for GPU usage.
    """
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts, prog_w0, Msat, seed_num)
    @jax.jit
    def single_particle_integrate(particle_number,pos_close_curr,pos_far_curr,vel_close_curr,vel_far_curr):
        curr_particle_w0_close = jnp.hstack([pos_close_curr,vel_close_curr])
        curr_particle_w0_far = jnp.hstack([pos_far_curr,vel_far_curr])
        t_release = ts[particle_number]
        t_final = ts[-1] #################+ .01
        
        w_particle_close = self.orbit_integrator_run_notdense_PID(curr_particle_w0_close,t_release,t_final,None,steps)###[0]
        w_particle_far = self.orbit_integrator_run_notdense_PID(curr_particle_w0_far,t_release,t_final,None,steps)###[0]
        
        return w_particle_close, w_particle_far
    particle_ids = jnp.arange(len(pos_close_arr))
    
    return jax.vmap(single_particle_integrate,in_axes=(0,0,0,0,0,))(particle_ids,pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr)


@partial(jax.jit,static_argnums=((0,1,6)))
def gen_stream_vmapped_notdense_PID_selfGrav(self, pot_no_selfGrav,ts, prog_w0, Msat, seed_num, steps):
    """
    Generate stellar stream by vmapping over the release model/integration. Better for GPU usage.
    Incorp self gravity of progenitor. Must supply potential object for system without self gravity, 
    but the self potential object _does_ include self-gravity. If no self gravity is included in either,
    this reduces to the same function as gen_stream_vmapped_notdense_PID.
    """
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = pot_no_selfGrav.gen_stream_ics(ts, prog_w0, Msat, seed_num)
    @jax.jit
    def single_particle_integrate(particle_number,pos_close_curr,pos_far_curr,vel_close_curr,vel_far_curr):
        curr_particle_w0_close = jnp.hstack([pos_close_curr,vel_close_curr])
        curr_particle_w0_far = jnp.hstack([pos_far_curr,vel_far_curr])
        t_release = ts[particle_number]
        t_final = ts[-1] #################+ .01
        
        w_particle_close = self.orbit_integrator_run_notdense_PID(curr_particle_w0_close,t_release,t_final,None,steps)###[0]
        w_particle_far = self.orbit_integrator_run_notdense_PID(curr_particle_w0_far,t_release,t_final,None,steps)###[0]
        
        return w_particle_close, w_particle_far
    particle_ids = jnp.arange(len(pos_close_arr))
    
    return jax.vmap(single_particle_integrate,in_axes=(0,0,0,0,0,))(particle_ids,pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr)

@partial(jax.jit,static_argnums=((0,1,6)))
def gen_stream_vmapped_notdense_selfGrav(self, pot_no_selfGrav,ts, prog_w0, Msat, seed_num, steps):
    """
    Generate stellar stream by vmapping over the release model/integration. Better for GPU usage.
    Incorp self gravity of progenitor. Must supply potential object for system without self gravity, 
    but the self potential object _does_ include self-gravity. If no self gravity is included in either,
    this reduces to the same function as gen_stream_vmapped_notdense.

    Uses constant stepsize rather than PID controller for orbit integration!

    """
    pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = pot_no_selfGrav.gen_stream_ics(ts, prog_w0, Msat, seed_num)
    @jax.jit
    def single_particle_integrate(particle_number,pos_close_curr,pos_far_curr,vel_close_curr,vel_far_curr):
        curr_particle_w0_close = jnp.hstack([pos_close_curr,vel_close_curr])
        curr_particle_w0_far = jnp.hstack([pos_far_curr,vel_far_curr])
        t_release = ts[particle_number]
        t_final = ts[-1] #################+ .01
        
        w_particle_close = self.orbit_integrator_run_notdense(curr_particle_w0_close,t_release,t_final,None,steps)###[0]
        w_particle_far = self.orbit_integrator_run_notdense(curr_particle_w0_far,t_release,t_final,None,steps)###[0]
        
        return w_particle_close, w_particle_far
    particle_ids = jnp.arange(len(pos_close_arr))
    
    return jax.vmap(single_particle_integrate,in_axes=(0,0,0,0,0,))(particle_ids,pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr)


