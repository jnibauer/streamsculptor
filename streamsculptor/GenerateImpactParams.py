import jax
import jax.numpy as jnp
from functools import partial
from astropy import units as u
import equinox as eqx
import diffrax
from streamsculptor import compute_stream_length, compute_length_oscillations, sample_from_1D_pdf

class ImpactGenerator:
    """
    Impact generator using formalisms of Yoon+2011, Erkal+2016.
    """
    def __init__(self,
                 pot: callable,
                 tobs: jnp.array,
                 stream: jnp.ndarray,
                 stream_phi1: jnp.array,
                 stripping_times: jnp.array,
                 prog_today: jnp.array,
                 phi1window  = 0.1,
                 NumImpacts = 1,
                 tImpactBounds = None,
                 bImpact_bounds = [0,1.0],
                 sigma = (180*u.km/u.s).to(u.kpc/u.Myr).value,
                 phi1_bounds = None,
                 phi1_exclude = [1.,1.],
                 stream_length = None,
                 seednum = 0):

        self.pot = pot
        self.stream = stream
        self.stream_phi1 = stream_phi1
        self.stripping_times = stripping_times
        self.phi1window = phi1window
        self.NumImpacts = NumImpacts
        self.prog_today = prog_today
        self.tobs = tobs
        self.stream_length = stream_length
        self.bImpact_bounds = bImpact_bounds
        #self.tImpactBounds = tImpactBounds
        self.sigma = sigma
        self.seednum = seednum
        self.phi1_exclude = phi1_exclude
        self.keys = jax.random.split(jax.random.PRNGKey(seednum), 7)

        if tImpactBounds is None:
            self.tImpactBounds = [jnp.min(stripping_times), 0.0]
        else:
            self.tImpactBounds = tImpactBounds
        
        if phi1_bounds is None:
            self.phi1_bounds = [jnp.min(stream_phi1), jnp.max(stream_phi1)]
        else:
            self.phi1_bounds = phi1_bounds
        
        if len(self.bImpact_bounds) == 2:
            self.b_low = jnp.ones_like(self.NumImpacts) * self.bImpact_bounds[0]
            self.b_high = jnp.ones_like(self.NumImpacts) * self.bImpact_bounds[1]
        else:
            # If bImpact_bounds is N_impacts x 2
            self.b_low = self.bImpact_bounds[:,0]
            self.b_high = self.bImpact_bounds[:,1]

        if self.stream_length is None:
            length = compute_stream_length(stream=self.stream, phi1=self.stream_phi1)
    
        else:
            length = self.stream_length
            #self.growth_rate = length / jnp.max(jnp.abs(jnp.array(self.tImpactBounds)))  # length / time
        #else:
            #self.growth_rate = self.stream_length / jnp.max(jnp.abs(jnp.array(self.tImpactBounds))) 
        first_strip_inds = jnp.where(self.stripping_times == self.stripping_times.min())[0]
        lead_first = self.stream.at[first_strip_inds.at[0].get()].get()
        trail_first = self.stream.at[first_strip_inds.at[1].get()].get()
        self.length_osc = compute_length_oscillations(pot=self.pot, 
                                                        prog_today=self.prog_today,
                                                        first_stripped_lead=lead_first,
                                                        first_stripped_trail=trail_first,
                                                        t_age=jnp.abs(jnp.asarray(self.tImpactBounds).min()),
                                                        length_today=length) 



    @partial(jax.jit, static_argnums=(0,))
    def w_parallel_sample(self, vs):
        """
        Probability density function for w_parallel, the component of relative subhalo velocity parallel to the stream.
        vs: scalar velocity of the stream patch 
        """
        return jax.random.normal(key=self.keys[0], shape=(self.NumImpacts,)) * self.sigma - vs

    @partial(jax.jit, static_argnums=(0,))
    def w_perpendicular_sample(self):
        """
        Probability density function for w_perpendicular, the component of relative subhalo velocity perpendicular to the stream.
        """
        prefac = jnp.sqrt(2 / jnp.pi) / self.sigma**3
        prob_func = lambda w_perp: prefac * (w_perp**2) * jnp.exp(-w_perp**2 / (2*self.sigma**2))
        w_perp_vals = jnp.linspace(-7*self.sigma, 7*self.sigma, 10_000)
        prob_w_perp = prob_func(w_perp_vals)
        prob_w_perp /= jnp.sum(prob_w_perp)
        w_perp_samples = jax.random.choice(key=self.keys[1], a=w_perp_vals, shape=(self.NumImpacts,), p=prob_w_perp, replace=True)
        return w_perp_samples

    
    
    @partial(jax.jit, static_argnums=(0,))
    def sample_impact_params(self):
        """
        Sample impact parameters
        Returns: dictionary of impact parameters
        Angles in radians, bImpact in kpc, vImpact in kpc/Myr
        """
        key = self.keys[-1]
        keys = jax.random.split(key, 4)
        
        bImpact = jax.random.uniform(minval=self.b_low,maxval=self.b_high,key=keys[0],shape=(self.NumImpacts,))
    
        #t_sample = jnp.linspace(self.tImpactBounds[0], self.tImpactBounds[1], 10_000)
        #prob_timpact = self.growth_rate*(t_sample + jnp.abs(self.tImpactBounds[0]))
        #prob_timpact = prob_timpact / jnp.sum(prob_timpact)
        #tImpact = jax.random.choice(key=keys[1], a=t_sample, shape=(self.NumImpacts,), p=prob_timpact, replace=True)
        tImpact = sample_from_1D_pdf(x=self.length_osc['ts'], y=self.length_osc['length_func'], key=keys[1], num_samples=self.NumImpacts)
        def phi1_exclude_provided(phi1_exclude):
            seg1 = jnp.linspace(self.phi1_bounds[0], phi1_exclude[0], 1000)
            seg2 = jnp.linspace(phi1_exclude[1], self.phi1_bounds[1], 1000)
            phi1_eval = jnp.hstack([seg1, seg2])
            length1 = phi1_exclude[0] - self.phi1_bounds[0]
            length2 = self.phi1_bounds[1] - phi1_exclude[1]
            total_length = length1 + length2
            prob_1 = ( length1 / total_length ) * (1./len(seg1))
            prob_2 = ( length2 / total_length ) * (1./len(seg2))
            prob_phi1 = jnp.hstack([jnp.ones_like(seg1) * prob_1, jnp.ones_like(seg2) * prob_2])
            return jax.random.choice(key=keys[2], a=phi1_eval, shape=(self.NumImpacts,), replace=True, p=prob_phi1)  
                  
        def phi1_exclude_not_provided(phi1_exclude):
            return jax.random.uniform(minval=self.phi1_bounds[0], maxval=self.phi1_bounds[1], key=keys[2], shape=(self.NumImpacts,))
        phi1_exclude_diff = self.phi1_exclude[1] - self.phi1_exclude[0] 
        exclude_bool = jnp.abs(phi1_exclude_diff) > 0 #if gtr than zero an interval w/ support is provided
        phi1_samples = jax.lax.cond(exclude_bool, phi1_exclude_provided, phi1_exclude_not_provided, self.phi1_exclude)
        #phi1_samples = jax.random.uniform(minval=self.phi1_bounds[0],maxval=self.phi1_bounds[1],key=keys[2],shape=(self.NumImpacts,))
        perp_angle_samples = jax.random.uniform(minval=0, maxval=2*jnp.pi, key=keys[3], shape=(self.NumImpacts,))
        return { "bImpact":bImpact, 'tImpact':tImpact, 'phi1_samples': phi1_samples, 'perp_angle': perp_angle_samples}

    @partial(jax.jit, static_argnums=(0,))
    def get_particle_mean(self, phi1_0=None):
        """
        Get average phase-space location of particles in stream at the input time
        phi1_0: scalar phi1 value at which to center the window
        """
        phi1low, phi1high = phi1_0-self.phi1window, phi1_0 + self.phi1window
        bool_in = (self.stream_phi1 > phi1low) & (self.stream_phi1 < phi1high)
        bool_in = bool_in.astype(int)
        stream_mean = jnp.sum(self.stream*bool_in[:,None], axis= 0)/jnp.sum(bool_in)   
        # Compute mean stripping times
        mean_tstrip = jnp.sum(self.stripping_times * bool_in) / jnp.sum(bool_in)                  
        return stream_mean, mean_tstrip
        
        
    #@eqx.filter_jit
    @partial(jax.jit, static_argnums=(0,))
    def _get_subhalo_ImpactParamsCartesian(self, tobs=None, particle_mean=None, tstrip_mean=None, tImpact=None, bImpact=None, perp_angle=None, subidx=None):
        """
        Get impact parameters in Cartesian coordinates
        _____________________________________________________
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
        perp_angle – angle [rad] of perpendicular velocity vector from TN plane (and in the N-B plane)
        _____________________________________________________
        Returns:
        ImpactW and bImpact_hat [unit vector impact location from backwards integrated patch]
        """
        tImpact = jnp.maximum(tImpact, tstrip_mean)
        W0 = self.pot.integrate_orbit(w0=particle_mean,t0=tobs,t1=tImpact,ts=jnp.array([tImpact]),solver=diffrax.Dopri8(),atol=1e-7,rtol=1e-7,dtmin=0.1).ys[0]
        # Define basis vecs
        T = W0[3:]
        T = T / jnp.sqrt(jnp.sum( T**2 ))
        B = jnp.cross(W0[:3],W0[3:])
        B = B / jnp.sqrt(jnp.sum( B**2 ))
        N = jnp.cross(B,T)

        v_s = jnp.sqrt(jnp.sum(W0[3:]**2))
        V_parallel = (self.w_parallel_sample(v_s)[subidx] + v_s) * T
        V_perpendicular = self.w_perpendicular_sample()[subidx] * (-N*jnp.sin(perp_angle) + B*jnp.cos(perp_angle))
        V_I = V_parallel + V_perpendicular

        # Impact unit vector (ray from integrated patch to impact location)
        bImpact_hat =  N*jnp.cos(perp_angle) + B*jnp.sin(perp_angle)
        # Impact location    
        X_I = W0[:3] + bImpact*bImpact_hat
        return jnp.hstack([X_I,V_I]), bImpact_hat

    def get_subhalo_ImpactParams(self):
        """
        Get cartesian location of impactors at impact time
        """
        impact_param_dict = self.sample_impact_params()
        particle_means, mean_stripping_times = jax.vmap(self.get_particle_mean)(impact_param_dict['phi1_samples'])

        batch_cart_impact = lambda particle_mean, tstrip_mean, tImpact, bImpact, perp_angle, subidx: self._get_subhalo_ImpactParamsCartesian(tobs=self.tobs, particle_mean=particle_mean, tstrip_mean=tstrip_mean, tImpact=tImpact, bImpact=bImpact, perp_angle=perp_angle,subidx=subidx)
        subidx = jnp.arange(self.NumImpacts)
        cart_impact_params, bImpact_hat = jax.vmap(batch_cart_impact)(particle_means, mean_stripping_times, impact_param_dict['tImpact'],impact_param_dict['bImpact'], impact_param_dict['perp_angle'], subidx)

        return {'CartesianImpactParams':cart_impact_params, 'ImpactFrameParams':impact_param_dict}


