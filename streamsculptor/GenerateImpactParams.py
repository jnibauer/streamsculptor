import jax
import jax.numpy as jnp
from astropy import units as u
import equinox as eqx
import diffrax
from typing import Any

from streamsculptor.streamhelpers import compute_stream_length, compute_length_oscillations, sample_from_1D_pdf

jax.config.update("jax_enable_x64", True)

class ImpactGenerator(eqx.Module):
    pot: Any
    tobs: jnp.ndarray
    stream: jnp.ndarray
    stream_phi1: jnp.ndarray
    stripping_times: jnp.ndarray
    prog_today: jnp.ndarray
    phi1window: float = eqx.field(static=True)
    NumImpacts: int = eqx.field(static=True)
    bImpact_bounds: jnp.ndarray
    sigma: float = eqx.field(static=True)
    phi1_exclude: jnp.ndarray
    stream_length: float
    seednum: int # don't make this static so that it can be changed without re-initializing the whole class and re-compiling the jitted functions
    
    # Internal arrays created during init
    keys: jnp.ndarray
    tImpactBounds: jnp.ndarray
    phi1_bounds: jnp.ndarray
    b_low: jnp.ndarray
    b_high: jnp.ndarray
    length_osc: dict
    _t_impact_pdf: jnp.ndarray

    def __init__(self,
                 pot,
                 tobs,
                 stream,
                 stream_phi1,
                 stripping_times,
                 prog_today,
                 phi1window=0.1,
                 NumImpacts=1,
                 tImpactBounds=None,
                 bImpact_bounds=[0.0, 1.0],
                 sigma=(180 * u.km / u.s).to(u.kpc / u.Myr).value,
                 phi1_bounds=None,
                 phi1_exclude=[1.0, 1.0],
                 stream_length=None,
                 seednum=0,
                 nsub_times=None,
                 nsub_vals=None):

        self.pot = pot
        self.tobs = jnp.asarray(tobs)
        self.stream = stream
        self.stream_phi1 = stream_phi1
        self.stripping_times = stripping_times
        self.prog_today = prog_today
        

        # Explicitly cast to pure Python types to silence Equinox static array warnings
        self.phi1window = float(phi1window)
        self.NumImpacts = int(NumImpacts)
        self.sigma = float(sigma)
        self.seednum = seednum

        # Convert bounds to JAX arrays
        self.bImpact_bounds = jnp.asarray(bImpact_bounds)
        self.phi1_exclude = jnp.asarray(phi1_exclude)
        
        self.keys = jax.random.split(jax.random.PRNGKey(self.seednum), 7)

        if tImpactBounds is None:
            self.tImpactBounds = jnp.array([jnp.min(stripping_times), 0.0])
        else:
            self.tImpactBounds = jnp.asarray(tImpactBounds)
        
        if phi1_bounds is None:
            self.phi1_bounds = jnp.array([jnp.min(stream_phi1), jnp.max(stream_phi1)])
        else:
            self.phi1_bounds = jnp.asarray(phi1_bounds)
        
        if self.bImpact_bounds.ndim == 1 and len(self.bImpact_bounds) == 2:
            self.b_low = jnp.ones(self.NumImpacts) * self.bImpact_bounds[0]
            self.b_high = jnp.ones(self.NumImpacts) * self.bImpact_bounds[1]
        else:
            self.b_low = self.bImpact_bounds[0]
            self.b_high = self.bImpact_bounds[1]

        if stream_length is None:
            self.stream_length = compute_stream_length(stream=self.stream, phi1=self.stream_phi1)
        else:
            self.stream_length = float(stream_length)
            
        ind_break = len(stripping_times) // 2 
        first_strip_inds = [0, ind_break]
        lead_first = self.stream[first_strip_inds[0]]
        trail_first = self.stream[first_strip_inds[1]]
        
        self.length_osc = compute_length_oscillations(pot=self.pot,
                                                      prog_today=self.prog_today,
                                                      first_stripped_lead=lead_first,
                                                      first_stripped_trail=trail_first,
                                                      t_age=jnp.abs(self.tImpactBounds.min()),
                                                      length_today=self.stream_length)

        if nsub_times is not None and nsub_vals is not None:
            nsub_ts = jnp.asarray(nsub_times)
            nsub_vs = jnp.asarray(nsub_vals)
            ts = self.length_osc['ts']
            if float(jnp.min(nsub_ts)) > float(jnp.min(ts)) or float(jnp.max(nsub_ts)) < float(jnp.max(ts)):
                raise ValueError(
                    f"nsub_times range [{float(jnp.min(nsub_ts)):.2f}, {float(jnp.max(nsub_ts)):.2f}] Myr "
                    f"does not cover length_osc ts range [{float(jnp.min(ts)):.2f}, {float(jnp.max(ts)):.2f}] Myr. "
                    "Supply nsub_times spanning at least [-t_age, 0]."
                )
            nsub_interp = jnp.interp(ts, nsub_ts, nsub_vs)
            self._t_impact_pdf = self.length_osc['length_func'] * nsub_interp
        else:
            self._t_impact_pdf = self.length_osc['length_func']
    @eqx.filter_jit
    def w_parallel_sample(self, vs):
        """
        Probability density function for w_parallel, the component of relative subhalo velocity parallel to the stream.
        vs: scalar velocity of the stream patch 
        """
        return jax.random.normal(key=self.keys[0], shape=(self.NumImpacts,)) * self.sigma - vs

    @eqx.filter_jit
    def w_perpendicular_sample(self):
        """
        Probability density function for w_perpendicular, the component of relative subhalo velocity perpendicular to the stream.
        """
        prefac = jnp.sqrt(2 / jnp.pi) / self.sigma**3
        prob_func = lambda w_perp: prefac * (w_perp**2) * jnp.exp(-w_perp**2 / (2 * self.sigma**2))
        
        w_perp_vals = jnp.linspace(-7 * self.sigma, 7 * self.sigma, 10_000)
        prob_w_perp = prob_func(w_perp_vals)
        prob_w_perp /= jnp.sum(prob_w_perp)
        
        w_perp_samples = jax.random.choice(key=self.keys[1], a=w_perp_vals, shape=(self.NumImpacts,), p=prob_w_perp, replace=True)
        return w_perp_samples

    @eqx.filter_jit
    def sample_impact_params(self):
        """
        Sample impact parameters
        Returns: dictionary of impact parameters
        Angles in radians, bImpact in kpc, vImpact in kpc/Myr
        """
        key = self.keys[-1]
        keys = jax.random.split(key, 4)
        
        bImpact = jax.random.uniform(minval=self.b_low, maxval=self.b_high, key=keys[0], shape=(self.NumImpacts,))
    
        tImpact = sample_from_1D_pdf(x=self.length_osc['ts'], y=self._t_impact_pdf, key=keys[1], num_samples=self.NumImpacts)
        
        def phi1_exclude_provided(phi1_exc):
            seg1 = jnp.linspace(self.phi1_bounds[0], phi1_exc[0], 1000)
            seg2 = jnp.linspace(phi1_exc[1], self.phi1_bounds[1], 1000)
            phi1_eval = jnp.hstack([seg1, seg2])
            
            length1 = phi1_exc[0] - self.phi1_bounds[0]
            length2 = self.phi1_bounds[1] - phi1_exc[1]
            total_length = length1 + length2
            
            prob_1 = (length1 / total_length) * (1. / len(seg1))
            prob_2 = (length2 / total_length) * (1. / len(seg2))
            prob_phi1 = jnp.hstack([jnp.ones_like(seg1) * prob_1, jnp.ones_like(seg2) * prob_2])
            
            return jax.random.choice(key=keys[2], a=phi1_eval, shape=(self.NumImpacts,), replace=True, p=prob_phi1)  
                  
        def phi1_exclude_not_provided(phi1_exc):
            return jax.random.uniform(minval=self.phi1_bounds[0], maxval=self.phi1_bounds[1], key=keys[2], shape=(self.NumImpacts,))
        
        phi1_exclude_diff = self.phi1_exclude[1] - self.phi1_exclude[0] 
        exclude_bool = jnp.abs(phi1_exclude_diff) > 0 # if gtr than zero an interval w/ support is provided
        phi1_samples = jax.lax.cond(exclude_bool, phi1_exclude_provided, phi1_exclude_not_provided, self.phi1_exclude)
        
        perp_angle_samples = jax.random.uniform(minval=0, maxval=2 * jnp.pi, key=keys[3], shape=(self.NumImpacts,))
        return {"bImpact": bImpact, 'tImpact': tImpact, 'phi1_samples': phi1_samples, 'perp_angle': perp_angle_samples}

    @eqx.filter_jit
    def get_particle_mean(self, phi1_0):
        """
        Get average phase-space location of particles in stream at the input time
        phi1_0: scalar phi1 value at which to center the window
        """
        phi1low, phi1high = phi1_0 - self.phi1window, phi1_0 + self.phi1window
        bool_in = (self.stream_phi1 > phi1low) & (self.stream_phi1 < phi1high)
        bool_in_int = bool_in.astype(int)
        
        # Avoid division by zero by safely defaulting to 0 if no particles are in the window
        sum_bool = jnp.maximum(jnp.sum(bool_in_int), 1)
        
        stream_mean = jnp.sum(self.stream * bool_in_int[:, None], axis=0) / sum_bool   
        mean_tstrip = jnp.sum(self.stripping_times * bool_in_int) / sum_bool                  
        return stream_mean, mean_tstrip
        
    @eqx.filter_jit
    def _get_subhalo_ImpactParamsCartesian(self, tobs, particle_mean, tstrip_mean, tImpact, bImpact, perp_angle, subidx):
        """
        Get impact parameters in Cartesian coordinates
        """
        W0 = self.pot.integrate_orbit(w0=particle_mean, t0=tobs, t1=tImpact, ts=jnp.array([tImpact]), solver=diffrax.Dopri8(), atol=1e-7, rtol=1e-7, dtmin=0.1).ys[0]
        
        # Define basis vecs
        T = W0[3:]
        T = T / jnp.sqrt(jnp.sum(T**2))
        B = jnp.cross(W0[:3], W0[3:])
        B = B / jnp.sqrt(jnp.sum(B**2))
        N = jnp.cross(B, T)

        v_s = jnp.sqrt(jnp.sum(W0[3:]**2))
        V_parallel = (self.w_parallel_sample(v_s)[subidx] + v_s) * T
        V_perpendicular = self.w_perpendicular_sample()[subidx] * (-N * jnp.sin(perp_angle) + B * jnp.cos(perp_angle))
        V_I = V_parallel + V_perpendicular

        # Impact unit vector (ray from integrated patch to impact location)
        bImpact_hat = N * jnp.cos(perp_angle) + B * jnp.sin(perp_angle)
        
        # Impact location    
        X_I = W0[:3] + bImpact * bImpact_hat
        return jnp.hstack([X_I, V_I]), bImpact_hat

    @eqx.filter_jit
    def get_subhalo_ImpactParams(self):
        """
        Get cartesian location of impactors at impact time
        """
        impact_param_dict = self.sample_impact_params()
        
        # vmap over the phi1 samples to get means
        particle_means, mean_stripping_times = jax.vmap(self.get_particle_mean)(impact_param_dict['phi1_samples'])

        # vmap over the specific impact parameters
        def batch_cart_impact(p_mean, t_strip, t_imp, b_imp, p_angle, idx):
            return self._get_subhalo_ImpactParamsCartesian(self.tobs, p_mean, t_strip, t_imp, b_imp, p_angle, idx)
            
        subidx = jnp.arange(self.NumImpacts)
        
        cart_impact_params, bImpact_hat = jax.vmap(batch_cart_impact)(
            particle_means, 
            mean_stripping_times, 
            impact_param_dict['tImpact'],
            impact_param_dict['bImpact'], 
            impact_param_dict['perp_angle'], 
            subidx
        )

        return {'CartesianImpactParams': cart_impact_params, 'ImpactFrameParams': impact_param_dict}