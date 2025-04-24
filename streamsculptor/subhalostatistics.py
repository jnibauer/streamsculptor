import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import jax.random as random 
from streamsculptor import JaxCoords as jc
import equinox as eqx
from functools import partial
from jax.scipy.integrate import trapezoid
import astropy.units as u
import interpax
sigma = (180*u.km/u.s).to(u.kpc/u.Myr).value


class RateCalculator():
    """
    Class to sample subhalo impacts from SHMF given formalism from Yoon+2011, Erkal+2016.
    """
    def __init__(self,
                 orbit: object,
                 t_age: jnp.array,
                 b_max_fac: jnp.array,
                 l_obs: jnp.array,
                 sigma = sigma,
                 a0 = jnp.array(1.77e-5), #Msun^-1
                 c0 = jnp.array(2.02e-13), #Msun^-1 kpc^-3
                 m0 = jnp.array(2.52e7), #Msun
                 alpha = jnp.array(0.678),
                 r_minus2 = jnp.array(162.4), #kpc
                 disk_factor = jnp.array(1.), #Disk supresses number of subhalos by factor of 3 in Erkal+2016
                ):
        
        self.orbital_r = jnp.sqrt(jnp.sum(orbit.ys[:,0:3]**2, axis=1))
        self.orbit_ts = jnp.linspace(0, jnp.max(jnp.abs(orbit.ts)), len(orbit.ts))
        self.t_age = t_age
        self.b_max_fac = b_max_fac
        self.sigma = sigma
        self.l_obs = l_obs
        self.a0 = a0
        self.c0 = c0
        self.m0 = m0
        self.alpha = alpha
        self.r_minus2 = r_minus2
        self.disk_factor = disk_factor

    @partial(jax.jit, static_argnums=(0,))
    def r_s_func(self, log10M,concentration_fac=1.0):
        M = 10**log10M
        return 1.05*jnp.sqrt(M / 1e8) * concentration_fac

    @partial(jax.jit, static_argnums=(0,))
    def b_max_func(self, log10M: jnp.array, concentration_fac=1.0):
        """
        b_max as a function of log10M
        """
        r_s_expect = self.r_s_func(log10M, concentration_fac)
        return self.b_max_fac * r_s_expect


    @partial(jax.jit, static_argnums=(0,))
    def dN_dlog10M(self, log10M: jnp.array, slope=-1.9, gamma=2.7, M_hm=0.0, beta=0.99):
        """
        Differential number of subhalos per unit mass.
        WDM parameters from https://arxiv.org/pdf/1911.02663
        M_hm = 0 is CDM expectation (default)
        """
        M = 10**log10M
        fac = M*jnp.log(10)
        wdm_fac = (1.0 + gamma * (M_hm/M))**(-beta)
        return ( self.a0 * (M/self.m0)**slope ) * fac * wdm_fac
    
    @partial(jax.jit, static_argnums=(0,))
    def spatial_density(self, r: jnp.array):
        """
        Spatial density of subhalos (up to proportionality constant). [num per volume]
        """
        arg = -(2./self.alpha) * ( (r/self.r_minus2)**self.alpha - 1.0 )
        return jnp.exp(arg)

    @partial(jax.jit, static_argnums=(0,))
    def dn_dlog10M(self, r: jnp.array, log10M: jnp.array, slope=-1.9, gamma=2.7, M_hm=0.0, beta=0.99):
        """
        total number density of subhalos per masss
        """
        mass_func_part = (self.c0/self.a0) * self.dN_dlog10M(log10M=log10M, slope=slope, gamma=gamma, M_hm=M_hm, beta=beta) #div by a0 b/c we want to renomralize by c0
        spatial_part = self.spatial_density(r)
        return mass_func_part * spatial_part

   

    @partial(jax.jit, static_argnums=(0,))
    def dN_encounter_dlog10M(self, log10M: jnp.float64, normalization=1.0, slope=-1.9,concentration_fac=1.0, gamma=2.7, M_hm=0.0, beta=0.99):
        """
        Differential number of encounters with subhalos per mass.
        This integrates over time, but not mass. 
        Assumes single mass input. Batch evaluation can be performed w/ vmap.
        """
        dn_dlog10M = self.dn_dlog10M(log10M=log10M, r=self.orbital_r, slope=slope, gamma=gamma, beta=beta, M_hm=M_hm)
        b_max = self.b_max_func(log10M=log10M, concentration_fac=concentration_fac)
        integrand = b_max * dn_dlog10M * self.orbit_ts
        prefac = jnp.sqrt(2*jnp.pi) * self.sigma * self.disk_factor * ( self.l_obs / self.t_age )
        return trapezoid(y=integrand, x=self.orbit_ts, axis=0) * prefac * normalization

    @partial(jax.jit, static_argnums=(0,))
    def N_encounter(self, log10M_min: jnp.array, log10M_max: jnp.array, normalization=1.0, slope=-1.9,concentration_fac=1.0, gamma=2.7, M_hm=0.0, beta=0.99):
        """
        Number of encounters with subhalos in the mass range [M_min, M_max].
        Does not return masses, just the total number of encounters (Poisson rate).
        Integrates dN_encounter_dlog10M over the mass range.
        """
        log10_m_arr = jnp.linspace(log10M_min, log10M_max, 1000)
        dN_encounter_dlog10M_func = lambda log10_M: self.dN_encounter_dlog10M(log10M=log10_M, normalization=normalization, slope=slope, concentration_fac=concentration_fac, gamma=gamma, beta=beta, M_hm=M_hm)
        dN_encounter_dlog10M_eval = jax.vmap(dN_encounter_dlog10M_func)(log10_m_arr)
        mass_integral = trapezoid(y=dN_encounter_dlog10M_eval, x=log10_m_arr)
        return mass_integral 
        


        #log10_M, R = jnp.meshgrid(log10_m_arr, self.orbital_r,indexing='ij')
        #dn_dlog10M_eval = jax.vmap(self.dn_dlog10M, in_axes=(0,0,None,None,None,None))(R.ravel(), log10_M.ravel(), slope, gamma, M_hm, beta).reshape(R.shape)
        #b_max = self.b_max_func(log10M=log10_m_arr, concentration_fac=concentration_fac)
        #integrand = b_max[:,None] * dn_dlog10M_eval * self.orbit_ts[None, :]
        #mass_integral = trapezoid(y=integrand, x=log10_m_arr, axis=0)
        #assert len(mass_integral) == len(self.orbit_ts)
        #final_integral = trapezoid(y=mass_integral, x=self.orbit_ts)
        #prefac = jnp.sqrt(2*jnp.pi) * self.sigma * self.disk_factor * ( self.l_obs / self.t_age )
        #return prefac * final_integral * normalization
        

    @partial(jax.jit, static_argnums=(0,10))
    def sample_masses(self, log10M_min: jnp.array, log10M_max: jnp.array, key: jax.random.PRNGKey ,normalization=1.0, slope=-1.9, concentration_fac=1.0 , gamma=2.7, M_hm=0.0, beta=0.99, array_length=1_000):
        """
        Sample masses of subhalos from the mass function.
        """
        key1, key2, key3 = random.split(key,3)
        N_encounter_rate = self.N_encounter(log10M_min=log10M_min, log10M_max=log10M_max, normalization=normalization, slope=slope,concentration_fac=concentration_fac, gamma=gamma, beta=beta, M_hm=M_hm)
        N_encounter = jax.random.poisson(key1, N_encounter_rate)

        # Dense array that we will resample from
        log10_M_arr = jnp.linspace(log10M_min, log10M_max, 5_000)
        dN_dlog10M_func = lambda log10_M: self.dN_encounter_dlog10M(log10M=log10_M, normalization=normalization, slope=slope, concentration_fac=concentration_fac, gamma=gamma, beta=beta, M_hm=M_hm)
        dNenc_dlog10M = jax.vmap(dN_dlog10M_func)(log10_M_arr)
        dlog10M = log10_M_arr[1] - log10_M_arr[0]
        prob = dNenc_dlog10M * dlog10M / jnp.sum(dNenc_dlog10M * dlog10M)
        prob = prob / jnp.sum(prob)

        samps = jax.random.choice(a=log10_M_arr, p=prob, shape=(array_length,),key=key2)

        #cdf = jnp.cumsum(prob)
        #inverse_func = interpax.Interpolator1D(x=cdf, f=log10_M_arr, method='monotonic')
        #runif = jax.random.uniform(key2, shape=(array_length,), minval=cdf.min(), maxval=1)
        #samps = inverse_func(runif)
        #samps = jax.random.permutation(key3, samps)
        idx = jnp.arange(array_length)
        return dict(log10_mass=jnp.where(idx>=N_encounter, 0, samps), N_encounter=N_encounter, N_encounter_rate=N_encounter_rate)

       
    @partial(jax.jit, static_argnums=(0,))
    def compute_N_enclosed(self, log10M_min: jnp.array, log10M_max: jnp.array, r_min: jnp.array, r_max: jnp.array, slope=-1.9, gamma=2.7, M_hm=0.0, beta=0.99):
        """
        Compute the total number of subhalos in a given range of radii and masses.
        """

        log10mass_arr = jnp.linspace(log10M_min, log10M_max, 101)
        r_arr = jnp.linspace(r_min, r_max, 100)
        log10_M, R = jnp.meshgrid(log10mass_arr, r_arr, indexing='ij')
        func_map = lambda r, log10M: self.dn_dlog10M(r, log10M,slope=slope, gamma=gamma, M_hm=M_hm, beta=beta)
        inp = jax.vmap(func_map)(R.ravel(), log10_M.ravel()).reshape(R.shape)
        inp = inp * 4 * jnp.pi * R**2
        I1 = trapezoid(y=inp, x=log10mass_arr, axis=0)
        I2 = trapezoid(y=I1, x=r_arr)
        return I2