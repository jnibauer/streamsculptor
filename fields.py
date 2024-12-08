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
from StreamSculptor import eval_dense_stream_id

"""
Not all integrations are hamiltonian. fields.py allows for the integration
of trajectories along arbitrary vector fields. 

Vector fields are defined as a class. The terms of the vector field
must be defined with a function term(t, coords, args).

coords do not need to represent position and velocity. For instance,
in perturbation theory the coordinate space is derivatives of position/velocity
coordinates. The derivatives are evolved via an update rule, specified by the 
term function.
"""


@partial(jax.jit,static_argnums=((2,3,4,10)))
def integrate_field(w0=None,ts=None, dense=False, solver=diffrax.Dopri8(scan_kind='bounded'),field=None, args=None, rtol=1e-7, atol=1e-7, dtmin=0.05, dtmax=None, max_steps=1_000,jump_ts=None, backwards_int=False):
    """
    Integrate field associated with potential function.
    w0: length 6 array [x,y,z,vx,vy,vz]
    ts: array of saved times. Must be at least length 2, specifying a minimum and maximum time. This does _not_ determine the timestep
    dense: boolean array.  When False, return orbit at times ts. When True, return dense interpolation of orbit between ts.min() and ts.max()
    solver: integrator
    field: instance of a potential function (i.e., pot.velocity_acceleration) specifying the field that we are integrating on
    rtol, atol: tolerance for PIDController, adaptive timestep
    dtmin: minimum timestep (in Myr)
    max_steps: maximum number of allowed timesteps
    """
    term = ODETerm(field.term)
    saveat = SaveAt(t0=False, t1=False,ts=ts,dense=dense)
    
    rtol: float = rtol  
    atol: float = atol  
    
    stepsize_controller = PIDController(rtol=rtol, atol=atol, dtmin=dtmin,dtmax=dtmax,force_dtmin=True,jump_ts=jump_ts)
    #max_steps: int = max_steps
    max_steps = int(max_steps)

    def false_func():
        """
        Integrating forward in time: t1 > t0
        """
        t0 = ts.min()
        t1 = ts.max()
        return t0, t1
    def true_func():
        """
        Integrating backwards in time: t1 < t0
        """
        t0 = ts.max()
        t1 = ts.min()
        return t0, t1
    t0, t1 = jax.lax.cond(backwards_int, true_func, false_func)
    

    solution = diffeqsolve(
        terms=term,
        solver=solver,
        t0= t0,
        t1= t1,
        y0=w0,
        dt0=None,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        discrete_terminating_event=None,
        max_steps=max_steps,
        adjoint=DirectAdjoint(),
        args=args
    )
    return solution

class hamiltonian_field:
    """
    Standard hamiltonian field: (q,p).
    This is the same as the velocity_acceleration term in integrate orbit.
    This class is redundant, and only included for pedagogical/tutorial purposes.
    """
    def __init__(self, pot):
        self.pot = pot
    @partial(jax.jit,static_argnums=(0,))
    def term(self,t,xv,args):
        x, v = xv[:3], xv[3:]
        acceleration = -self.pot.gradient(x,t)
        return jnp.hstack([v,acceleration])


class MassRadiusPerturbation_OTF:
    """
    Applying perturbation theory in the mass and radius of a 
    subhalo potential. 
    OTF = "On The Fly"
    The unperturbed orbits are computed in realtime, along
    with the perturbation equations. No pre-computation is utilized.
    coordinate vectors consist of a pytree: 
    coords[0]: length 6 (position and velocity field in base potential)
    coords[1]: length 12 (postion and veloicty derivatives wrspct to eps, radius)
    coords: [ [x,y,z, vx, vy, vz],
       [dx/deps,..., dvx/deps,..., d^2x/dthetadeps, ..., d^2vx/dthetadeps]  ]
    """
    def __init__(self, perturbation_generator):
        self.pertgen = perturbation_generator
    @partial(jax.jit,static_argnums=(0,))
    def term(self,t, coords, args):
        """
        x0,v0: base position and velocity
        x1, v1: mass perturbations in each coord
        dx1_dtheta, dv1_dtheta: second order mass*radius perturbations in each coord
        """
        # Unperturbed, base flow
        x0, v0 = coords[0][:3], coords[0][3:6] #length 6 array
        # Epsilon Derivs
        x1, v1 = coords[1][:,:3], coords[1][:,3:6] #nSHS x 3
        # Structural derivs
        dx1_dtheta, dv1_dtheta = coords[1][:,6:9], coords[1][:,9:]

        acceleration0 = -self.pertgen.gradientPotentialBase(x0,t)

        # acceleration due to perturbation
        acceleration1 = -self.pertgen.gradientPotentialPerturbation_per_SH(x0,t) # nSH x 3
        # tidal tensor
        d2H_dq2 = -jax.jacrev(self.pertgen.gradientPotentialBase)(x0,t) 
        # For the Hamiltonian considered, dp/deps is just the integrated acceleration. Relabling...
        d_qdot_d_eps = v1 
        # Next, injected acceleration: subhalo acceleration + matmul(tidal tensor, dq/deps)
        d_pdot_d_eps = acceleration1 + jnp.einsum('ij,kj->ki',d2H_dq2,x1)#,optimize='optimal') #nSH x 3
        
        # Now handle radius deviations
        acceleration1_r = -self.pertgen.gradientPotentialStructural_per_SH(x0,t) # nSH x 3
        d_qalpha1dot_dtheta = dv1_dtheta
        d_palpha1dot_dtheta = acceleration1_r + jnp.einsum('ij,kj->ki',d2H_dq2,dx1_dtheta)#,optimize='optimal')#jnp.matmul(d2H_dq2,dx1_dtheta)
        
        # Package the output: [6, Nsh x 12]
        return [jnp.hstack([v0,acceleration0]), 
                jnp.hstack([d_qdot_d_eps,d_pdot_d_eps, d_qalpha1dot_dtheta, d_palpha1dot_dtheta])]

class MassRadiusPerturbation_Interp:
    """
    Apply perturbation theory in the mass and radius of a subhalo potential.
    Interpolated version. BaseStreamModel must have dense=True in order to support
    this function. The base trajectories are saved via interpolation, and perturbation
    trajectories are computed along the interpolated particle trajectories.
    When sampling many batches of perturbations (order 1000s), this function 
    eliminates the need to recompute the base stream every time. Can lead to factor
    of a few speedup. The cost is increased memory usage.

    coordinate vectors consist of a pytree:
    coords: shape nSH x 12
    coords[0,:]: [dx/deps,..., dvx/deps,..., d^2x/dthetadeps, ..., d^2vx/dthetadeps] 
    """
    def __init__(self, perturbation_generator):
        self.pertgen = perturbation_generator
        self.base_stream = perturbation_generator.base_stream
    @partial(jax.jit,static_argnums=(0,))
    def term(self, t, coords, args):
        """
        args is a dictionary:  args['idx'] is the current particle index, and args['tail_bool'] specifies the stream arm
        If tail_bool is True, interpolate leading arm. If False, interpolate trailing arm.
        x1, v1: mass perturbations in each coord
        dx1_dtheta, dv1_dtheta: second order mass*radius perturbations in each coord
        """
        idx, tail_bool = args['idx'], args['tail_bool']
        x0v0 = eval_dense_stream_id(time=t, interp_func=self.base_stream.stream_interp, idx=idx, lead=tail_bool)
        # Unperturbed, base flow
        x0, v0 = x0v0[:3], x0v0[3:]
        # Epsilon Derivs
        x1, v1 = coords[:,:3], coords[:,3:6] #nSHS x 3
        # Structural derivs
        dx1_dtheta, dv1_dtheta = coords[:,6:9], coords[:,9:]

        # acceleration due to perturbation
        acceleration1 = -self.pertgen.gradientPotentialPerturbation_per_SH(x0,t) # nSH x 3
        # tidal tensor
        d2H_dq2 = -jax.jacrev(self.pertgen.gradientPotentialBase)(x0,t) 
        # For the Hamiltonian considered, dp/deps is just the integrated acceleration. Relabling...
        d_qdot_d_eps = v1 
        # Next, injected acceleration: subhalo acceleration + matmul(tidal tensor, dq/deps)
        d_pdot_d_eps = acceleration1 + jnp.einsum('ij,kj->ki',d2H_dq2,x1)#,optimize='optimal') #nSH x 3
        
        # Now handle radius deviations
        acceleration1_r = -self.pertgen.gradientPotentialStructural_per_SH(x0,t) # nSH x 3
        d_qalpha1dot_dtheta = dv1_dtheta
        d_palpha1dot_dtheta = acceleration1_r + jnp.einsum('ij,kj->ki',d2H_dq2,dx1_dtheta)#,optimize='optimal')#jnp.matmul(d2H_dq2,dx1_dtheta)
        
        # Package the output: [Nsh x 12]
        return jnp.hstack([d_qdot_d_eps,d_pdot_d_eps, d_qalpha1dot_dtheta, d_palpha1dot_dtheta])
