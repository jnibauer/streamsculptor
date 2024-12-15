import jax
import jax.numpy as jnp
from functools import partial
from astropy import units as u
class ImpactGenerator:
    def __init__(self, pot=None, tobs=None, stream=None, stream_phi1=None, phi1window=.1, NumImpacts=1,
    phi_bounds=[0,jnp.pi],beta_bounds=[0,jnp.pi/2],gamma_bounds=[0,jnp.pi/2],bImpact_bounds=[0,1.0],
    vImpact_bounds=[-400.0*(u.km/u.s).to(u.kpc/u.Myr),400.0*(u.km/u.s).to(u.kpc/u.Myr)], tImpactBounds=[-4000.,0.0], phi1_bounds=None, seednum=0):
        """
        pot: potential function
        stream: phase-space coordinates of stream particles, N x 6 array
        stream_phi1: phi1 values of stream particles, 1D array
        phi1window: scalar width of window in phi1 in which to average stream particles (deg)
        tobs: observation time
        ----------------------------------------------------------------------------------------
        Example usage:
        ->  stream = jnp.vstack([lead,trail])
        ->  imp_obj = ImpactGenerator(pot=pot_NFW, tobs=0.0, stream=stream, stream_phi1=stream[:,0],NumImpacts=20)
        ->  imp_obj.sample_impact_params()
        ->  out = imp_obj.get_subhalo_ImpactParams()
        ->  CartImpact = out['CartesianImpactParams']
        ->  ImpactFrameParams = out['ImpactFrameParams']
        """
        self.pot = pot
        self.stream = stream
        self.stream_phi1 = stream_phi1
        self.phi1window = phi1window
        self.NumImpacts = NumImpacts
        self.tobs = tobs
        if phi1_bounds is None:
            self.phi1_bounds = [jnp.min(stream_phi1), jnp.max(stream_phi1)]
        else:
            self.phi1_bounds = phi1_bounds
        # Bounds for impact parameters
        self.phi_bounds = phi_bounds
        self.beta_bounds = beta_bounds
        self.gamma_bounds = gamma_bounds
        self.bImpact_bounds = bImpact_bounds
        self.vImpact_bounds = vImpact_bounds
        self.tImpactBounds = tImpactBounds
        self.seednum = seednum
        
    @partial(jax.jit,static_argnums=(0,))
    def sample_impact_params(self):
        """
        Sample impact parameters
        Returns: dictionary of impact parameters
        Angles in radians, bImpact in kpc, vImpact in kpc/Myr
        """
        key = jax.random.PRNGKey(self.seednum)
        keys = jax.random.split(key, 7)

        phi = jax.random.uniform(minval=self.phi_bounds[0],maxval=self.phi_bounds[1],key=keys[0],shape=(self.NumImpacts,))
        beta = jax.random.uniform(minval=self.beta_bounds[0],maxval=self.beta_bounds[1],key=keys[1],shape=(self.NumImpacts,))
        gamma = jax.random.uniform(minval=self.gamma_bounds[0],maxval=self.gamma_bounds[1],key=keys[2],shape=(self.NumImpacts,))
        bImpact = jax.random.uniform(minval=self.bImpact_bounds[0],maxval=self.bImpact_bounds[1],key=keys[3],shape=(self.NumImpacts,))
        vImpact = jax.random.uniform(minval=self.vImpact_bounds[0],maxval=self.vImpact_bounds[1],key=keys[4],shape=(self.NumImpacts,))
        tImpact = jax.random.uniform(minval=self.tImpactBounds[0],maxval=self.tImpactBounds[1],key=keys[5],shape=(self.NumImpacts,))
        phi1_samples = jax.random.uniform(minval=self.phi1_bounds[0],maxval=self.phi1_bounds[1],key=keys[6],shape=(self.NumImpacts,))
        return {"phi":phi, "beta":beta, "gamma":gamma, "bImpact":bImpact, "vImpact":vImpact, 'tImpact':tImpact, 'phi1_samples':phi1_samples}

    @partial(jax.jit,static_argnums=(0,1))
    def get_particle_mean(self, phi1_0=None):
        """
        Get average phase-space location of particles in stream at the input time
        phi1_0: scalar phi1 value at which to center the window
        """
        phi1low, phi1high = phi1_0-self.phi1window, phi1_0 + self.phi1window
        bool_in = (self.stream_phi1 > phi1low) & (self.stream_phi1 < phi1high)
        bool_in = bool_in.astype(int)
        stream_mean = jnp.sum(self.stream*bool_in[:,None], axis= 0)/jnp.sum(bool_in)                       
        return stream_mean
        
        
    @partial(jax.jit,static_argnums=(0,1))
    def _get_subhalo_ImpactParamsCartesian(self, tobs=None, particle_mean=None, tImpact=None, bImpact=None, vImpact=None, phi=None, beta=None, gamma=None):
        """
        Get impact parameters in Cartesian coordinates
        _____________________________________________________
        pot: potential function
        tobs: the obervation time
        particle_mean: average phase-space location of stream particles at the observation time, tobs
        _____________________________________________________
        Basis vectors:
        T – is tangent to velocity
        N – is orthogonal to velocity and angular momentum
        B – is along the angular momentum vector
        _____________________________________________________
        Time-space Parameters to define impact: 
        tImpact – time of impact 
        bImpact – impact parameter 
        vImpact – impact velocity
        _____________________________________________________
        Angle parameters:
        phi – spherical angle between B-axis and location of impact 
        beta - azimuthal angle of impact velocity vector in TN plane
        gamma - angle of impact velocity vector from TN plane
        _____________________________________________________
        Returns:
        ImpactW and bImpact_hat [unit vector impact location from backwards integrated patch]
        """
        
        W0 = self.pot.integrate_orbit(w0=particle_mean,t0=tobs,t1=tImpact,ts=jnp.array([tImpact])).ys[0]
        # Define basis vecs
        T = W0[3:]
        T = T / jnp.sqrt(jnp.sum( T**2 ))
        B = jnp.cross(W0[:3],W0[3:])
        B = B / jnp.sqrt(jnp.sum( B**2 ))
        N = jnp.cross(B,T)

        # Impact unit vector (ray from integrated patch to impact location)
        bImpact_hat =  jnp.sin(phi)*N + jnp.cos(phi)*B
        # Impact location    
        X_I = W0[:3] + bImpact*bImpact_hat
        # Impact velocity vector
        v_hat = jnp.cos(gamma)*jnp.cos(beta)*T + jnp.cos(gamma)*jnp.sin(beta)*N + jnp.sin(gamma)*B
        V_I = vImpact*v_hat
        return jnp.hstack([X_I,V_I]), bImpact_hat

    #@partial(jax.jit,static_argnums=(0,))
    def get_subhalo_ImpactParams(self):
        """
        Get cartesian location of impactors at impact time
        """
        impact_param_dict = self.sample_impact_params()
        particle_means = jax.vmap(self.get_particle_mean)(impact_param_dict['phi1_samples'])

        batch_cart_impact = lambda particle_mean, tImpact, bImpact, vImpact, phi, beta, gamma: self._get_subhalo_ImpactParamsCartesian(tobs=self.tobs, particle_mean=particle_mean, tImpact=tImpact, bImpact=bImpact, vImpact=vImpact, phi=phi, beta=beta, gamma=gamma)
        cart_impact_params, bImpact_hat = jax.vmap(batch_cart_impact)(particle_means, impact_param_dict['tImpact'],impact_param_dict['bImpact'], impact_param_dict['vImpact'], impact_param_dict['phi'], impact_param_dict['beta'], impact_param_dict['gamma'])

        return {'CartesianImpactParams':cart_impact_params, 'ImpactFrameParams':impact_param_dict}


