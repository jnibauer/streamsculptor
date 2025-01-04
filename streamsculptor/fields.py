from functools import partial
from astropy.constants import G
import astropy.coordinates as coord
import astropy.units as u
# gala
from gala.units import dimensionless, UnitSystem

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from diffrax import diffeqsolve, ODETerm, Dopri5,SaveAt,PIDController,DiscreteTerminatingEvent, DirectAdjoint, RecursiveCheckpointAdjoint, ConstantStepSize, Euler, StepTo, ForwardMode
import diffrax
import equinox as eqx
usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)
from streamsculptor import Potential
from streamsculptor import eval_dense_stream_id
from streamsculptor import potential

"""
Not all integrations are hamiltonian. fields.py allows for the integration
of trajectories along arbitrary vector fields. 

Vector fields are defined as a class. The terms of the vector field
must be defined with a function term(t, coords, args).

coords do not need to represent position and velocity. For instance,
when integrating perturbative equations the coordinate space is *derivatives* of position/velocity
coordinates. The derivatives are evolved via an update rule, specified by the 
term function.
"""


@eqx.filter_jit
def integrate_field(w0=None,ts=None, dense=False, solver=diffrax.Dopri8(scan_kind='bounded'),field=None, args=None, rtol=1e-7, atol=1e-7, dtmin=0.05, dtmax=None, max_steps=1_000,jump_ts=None, backwards_int=False,t0=0.0, t1=0.0):
    """
    Integrate a trajectory on a field.
    w0: length 6 array [x,y,z,vx,vy,vz]
    ts: array of saved times. Must be at least length 2, specifying a minimum and maximum time. This does _not_ determine the timestep
    dense: boolean array.  When False, return orbit at times ts. When True, return dense interpolation of orbit between ts.min() and ts.max()
    solver: integrator
    field: specifies the field that we are integrating on
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

    ## Below we specify how to handle t0, t1
    ## If t0 == t1, t0 and t1 are computed using ts array (default)
    ## If t0 != t1, the user-specified values are utilized
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

    def t0_t1_are_different(): 
        return t0, t1
    def t0_t1_are_same():
        t0, t1 = jax.lax.cond(backwards_int, true_func, false_func)
        return t0, t1
    
    t0, t1 = jax.lax.cond(t0 != t1, t0_t1_are_different, t0_t1_are_same)


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
        adjoint=ForwardMode(),
        args=args
    )
    return solution

class hamiltonian_field:
    """
    Standard hamiltonian field.
    This is the same as the velocity_acceleration term in integrate orbit.
    This class is redundant, and only included for pedagogical/tutorial purposes.
    """
    def __init__(self, pot):
        self.pot = pot
    @eqx.filter_jit
    def term(self,t,xv,args):
        x, v = xv[:3], xv[3:]
        acceleration = -self.pot.gradient(x,t)
        return jnp.hstack([v,acceleration])

class Nbody_field:
    """
    Nbody field
    The term computes pairwise forces between particles.
    """
    def __init__(self, ext_pot=None, masses=None, units=None, eps=1e-3):
        """
        ext_pot: external potential. If None, no external potential is used.
        masses: array of masses for the N particles
        units: astropy unit system
        eps: softening length [kpc]
        """
        if ext_pot is None:
            ext_pot = potential.PlummerPotential(m=0.0, r_s=1.0, units=usys)
        self.ext_pot = ext_pot
        self.masses = masses
        self._G = jnp.array(G.decompose(units).value)
        self.eps = eps
    @eqx.filter_jit
    def term(self, t, xv, args):
        """
        xv: (N,6) array of positions and velocities
        masses: (N,) array of masses
        """
        x, v = xv[:, :3], xv[:, 3:]
        # Compute pairwise displacement vectors
        displacements = x[:, None, :] - x[None, :, :]
        # Compute squared distances with softening
        distances_squared = jnp.sum(displacements**2, axis=-1) + self.eps**2
        distances = jnp.sqrt(distances_squared)
        # Compute pairwise forces
        forces = (self._G * self.masses[:, None] * self.masses[None, :] / distances_squared)[:, :, None] * displacements / distances[:, :, None]
        # Zero out self-interaction forces (diagonal elements)
        forces = jnp.where(jnp.eye(x.shape[0])[:, :, None] == 1, 0.0, forces)

        # Sum forces to get accelerations
        self_grav_acceleration = jnp.sum(forces, axis=0) / self.masses[:, None]
        # total acceleration is self-gravity + external acceleration 
        total_acceleration = self_grav_acceleration - jax.vmap(self.ext_pot.gradient, in_axes=(0, None))(x, t)

        return jnp.hstack([v, total_acceleration])
        


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
    @eqx.filter_jit
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
    --------------------------------------------------------------------------------
    coordinate vectors consist of a pytree:
    coords: shape nSH x 12
    coords[0,:]: [dx/deps,..., dvx/deps,..., d^2x/dthetadeps, ..., d^2vx/dthetadeps] 
    """
    def __init__(self, perturbation_generator):
        self.pertgen = perturbation_generator
        self.base_stream = perturbation_generator.base_stream
    @eqx.filter_jit
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



class MassRadiusPerturbation_OTF_SecondOrder:
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
        self.potential_base = perturbation_generator.potential_base
        self.potential_pert = perturbation_generator.potential_perturbation
        
        self.acceleration = jax.jit(self.potential_base.acceleration)
        self.dacceleration_dx = jax.jit(jax.jacfwd(self.acceleration,argnums=(0,)))
        self.d2acceleration_dx2 = jax.jit(jax.jacfwd(self.dacceleration_dx,argnums=(0,)))
        
        self.grad_per_SH = jax.jit(jax.jacfwd(self.potential_pert.potential_per_SH))
        self.minus_dapert_dx_perSH = jax.jit(jax.jacfwd(self.grad_per_SH))
        
        
        
        
    @eqx.filter_jit
    def term(self,t, coords, args):
        """
        coords[0] contain x0,v0: base position and velocity [length 6]
        coords[1] contain x1, v1, dx1_dtheta, dv1_dtheta: mass / structural perturbations in each coord [nSH x 12]
        coords[2] contain x2, v2: second order mass perturbations in each coord [nSH x 6]
        """
        
        x0, v0 = coords[0][:3], coords[0][3:]
        x1, v1 = coords[1][:,:3], coords[1][:,3:6] # nSH x 3
        dx1_dtheta, dv1_dtheta = coords[1][:,6:9], coords[1][:,9:] # nSH x 3
        x2, v2 = coords[2][:,:3], coords[2][:,3:] # nSH x 3
        
        a0 = self.pot_base.acceleration(x0, t) 
        a1 = -self.grad_per_SH(x0,t) # nSH x 3
        da_dx = self.dacceleration_dx(x0, t)[0] # 3 x 3
        d2a_dx2 = self.d2acceleration_dx2(x0,t)[0][0] # 3 x 3 x 3
        dapert_dx = -self.minus_dapert_dx_perSH(x0,t) # nSH x 3 x 3
        
        da = jnp.einsum('ij,nj->ni', da_dx, x1) + a1  # nSH x 3
        
        term1 = jnp.einsum('ij,kj->ki', da_dx, x2) # nSH x 3
        inner_term2 = jnp.einsum('ikj,nk->nij', d2a_dx2, x1) # nSH x 3 x 3
        term2 = jnp.einsum('nj,nij->ni', x1, inner_term2) # nSH x 3
        d2a = term1 + term2 + jnp.einsum('nij,nj->ni',dapert_dx,x1) # nSH x 3

        # Now handle radius deviations
        acceleration1_r = -self.pertgen.gradientPotentialStructural_per_SH(x0,t) # nSH x 3
        d_palpha1dot_dtheta = acceleration1_r + jnp.einsum('ij,kj->ki',da_dx,dx1_dtheta) # nSH x 3

        coord0 = jnp.hstack([v0, a0])
        coord1 = jnp.hstack([v1, da, dv1_dtheta, d_palpha1dot_dtheta])
        coord2 = jnp.hstack([v2, d2a])
        coords_out = [coord0, coord1, coord2]
        return coords_out
        


class MW_LMC_field:
    # The following field is based entirely on 
    # this script from AGAMA: 
    # https://github.com/GalacticDynamics-Oxford/Agama/blob/c507fc3e703513ae4a41bb705e171a4d036754a8/py/example_lmc_mw_interaction.py
    # Treat the MW and LMC as rigid body potentials.
    # Evolve the centroid of each in response to the other, with Chandrasekhar dynamical friction for the LMC.
    # sigma_func is the velocity dispersion function of the MW at the LMC position [sigma_func(xyz) = scalar].
    def __init__(self, pot_MW=None, pot_LMC=None, sigma_func=None, bminCouLog=None):
        self.pot_MW = pot_MW
        self.pot_LMC = pot_LMC
        self.sigma_func = sigma_func
        self.bminCouLog = bminCouLog
        self.massLMC = pot_LMC.mass
    
    @eqx.filter_jit
    def term(self, t, coords, args):
        x0, v0 = coords[0][:3], coords[0][3:] # MW position and velocity
        x1, v1 = coords[1][:3], coords[1][3:] # LMC position and velocity
        dx = x1 - x0 # relative position – from MW center
        dv = v1 - v0 # relative velocity - from MW center
        dist = jnp.sum(dx**2)**0.5
        vmag = jnp.sum(dv**2)**0.5
        f0 = self.pot_LMC.acceleration(-dx,t) # force from LMC on MW center
        f1 = self.pot_MW.acceleration(dx,t) # force from MW on LMC
        rho = self.pot_MW.density(dx,t) # MW density at LMC position
        sigma = self.sigma_func(dist) # velocity dispersion of MW at LMC position

        # distance-dependent Coulomb logarithm
        # (an approximation that best matches the results of N-body simulations)
        couLog = jnp.maximum(0.0, jnp.log(dist/self.bminCouLog)**0.5)
        X = vmag / (sigma * 2**0.5)
        drag  = -(4*jnp.pi * rho * dv / vmag *
        (jax.lax.erf(X) - 2/jnp.pi**.5 * X * jnp.exp(-X*X)) *
        self.massLMC * self.pot_MW._G**2 / vmag**2 * couLog)   # dynamical friction force
        
        force_on_MW = f0
        force_on_LMC = f1 + drag
        return [jnp.hstack([v0, force_on_MW]), jnp.hstack([v1, force_on_LMC])]

class CustomField:
    """
    - Custom field. User must define a function: term(t, coords, args).
    - term(t, coords, args) outputs a pytree of same shape as coords.
    - coords: list of arrays. Each array is a different coordinate in the field.
    - args: dictionary of additional arguments.
    ----------------------------------------------------------------
    Physically, term specifies an update rule for coords.
    For instance, in the case of orbit integration, term(t, coords, args) would
    output the time derivative of the position and velocity coordinates (i.e., velocity and acceleration).
    ----------------------------------------------------------------
    """
    def __init__(self, term=None):
        self.term = term
    @eqx.filter_jit
    def term(self, t, coords, args):
        return self.term(t, coords, args)