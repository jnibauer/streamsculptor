import jax
import jax.numpy as jnp
import jax.random as random 
import equinox as eqx
from jax.scipy.integrate import trapezoid
import astropy.units as u

# Assuming jc and interpax are accessible from your environment
from streamsculptor import JaxCoords as jc
import interpax

jax.config.update("jax_enable_x64", True)
sigma_val = (180 * u.km / u.s).to(u.kpc / u.Myr).value

class RateCalculator(eqx.Module):
    """
    Class to sample subhalo impacts from SHMF given formalism from Yoon+2011, Erkal+2016.
    """
    orbital_r: jnp.ndarray
    orbit_ts: jnp.ndarray
    t_age: jnp.ndarray
    b_max_fac: jnp.ndarray
    l_obs: jnp.ndarray
    
    sigma: jnp.ndarray
    a0: jnp.ndarray
    c0: jnp.ndarray
    m0: jnp.ndarray
    alpha: jnp.ndarray
    r_minus2: jnp.ndarray
    disk_factor: jnp.ndarray

    def __init__(self,
                 orbit,
                 t_age,
                 b_max_fac,
                 l_obs,
                 sigma=sigma_val,
                 a0=1.77e-5,      # Msun^-1
                 c0=2.02e-13,     # Msun^-1 kpc^-3
                 m0=2.52e7,       # Msun
                 alpha=0.678,
                 r_minus2=162.4,  # kpc
                 disk_factor=1.0  # Disk suppresses number of subhalos by factor of 3 in Erkal+2016
                ):
        
        self.orbital_r = jnp.sqrt(jnp.sum(orbit.ys[:, 0:3]**2, axis=1))
        self.orbit_ts = jnp.linspace(0, jnp.max(jnp.abs(orbit.ts)), len(orbit.ts))
        
        # Ensure all physical parameters are treated as traceable JAX arrays
        self.t_age = jnp.asarray(t_age)
        self.b_max_fac = jnp.asarray(b_max_fac)
        self.l_obs = jnp.asarray(l_obs)
        self.sigma = jnp.asarray(sigma)
        self.a0 = jnp.asarray(a0)
        self.c0 = jnp.asarray(c0)
        self.m0 = jnp.asarray(m0)
        self.alpha = jnp.asarray(alpha)
        self.r_minus2 = jnp.asarray(r_minus2)
        self.disk_factor = jnp.asarray(disk_factor)

    @eqx.filter_jit
    def r_s_func(self, log10M, concentration_fac=1.0):
        M = 10**log10M
        return 1.05 * jnp.sqrt(M / 1e8) * concentration_fac

    @eqx.filter_jit
    def b_max_func(self, log10M, concentration_fac=1.0):
        """
        b_max as a function of log10M
        """
        r_s_expect = self.r_s_func(log10M, concentration_fac)
        return self.b_max_fac * r_s_expect
    

    @eqx.filter_jit
    def dN_dlog10M(self, log10M, slope=-1.9, gamma=2.7, M_hm=0.0, beta=0.99):
        """
        Differential number of subhalos per unit mass.
        WDM parameters from https://arxiv.org/pdf/1911.02663
        M_hm = 0 is CDM expectation (default)
        """
        M = 10**log10M
        fac = M * jnp.log(10)
        wdm_fac = (1.0 + gamma * (M_hm / M))**(-beta)
        return (self.a0 * (M / self.m0)**slope) * fac * wdm_fac
    
    @eqx.filter_jit
    def spatial_density(self, r):
        """
        Spatial density of subhalos (up to proportionality constant). [num per volume]
        """
        arg = -(2. / self.alpha) * ((r / self.r_minus2)**self.alpha - 1.0)
        return jnp.exp(arg)

    @eqx.filter_jit
    def dn_dlog10M(self, r, log10M, slope=-1.9, gamma=2.7, M_hm=0.0, beta=0.99):
        """
        total number density of subhalos per masss
        """
        mass_func_part = (self.c0 / self.a0) * self.dN_dlog10M(log10M=log10M, slope=slope, gamma=gamma, M_hm=M_hm, beta=beta) 
        spatial_part = self.spatial_density(r)
        return mass_func_part * spatial_part

    @eqx.filter_jit
    def nsub(self, r):
        return (self.c0 / self.a0) * self.spatial_density(r)
        
    @eqx.filter_jit
    def dN_encounter_dlog10M(self, log10M, normalization=1.0, slope=-1.9, concentration_fac=1.0, gamma=2.7, M_hm=0.0, beta=0.99):
        """
        Differential number of encounters with subhalos per mass.
        This integrates over time, but not mass. 
        Assumes single mass input. Batch evaluation can be performed w/ vmap.
        """
        dn_dlog10M_val = self.dn_dlog10M(log10M=log10M, r=self.orbital_r, slope=slope, gamma=gamma, beta=beta, M_hm=M_hm)
        b_max = self.b_max_func(log10M=log10M, concentration_fac=concentration_fac)
        nsub_eval = self.nsub(r=self.orbital_r)
        
        integrand = b_max * dn_dlog10M_val  * self.orbit_ts
        prefac = jnp.sqrt(2 * jnp.pi) * self.sigma * self.disk_factor * (self.l_obs / self.t_age)
        
        return trapezoid(y=integrand, x=self.orbit_ts, axis=0) * prefac * normalization

    @eqx.filter_jit
    def N_encounter(self, log10M_min, log10M_max, normalization=1.0, slope=-1.9, concentration_fac=1.0, gamma=2.7, M_hm=0.0, beta=0.99):
        """
        Number of encounters with subhalos in the mass range [M_min, M_max].
        Does not return masses, just the total number of encounters (Poisson rate).
        Integrates dN_encounter_dlog10M over the mass range.
        """
        log10_m_arr = jnp.linspace(log10M_min, log10M_max, 1000)
        dN_encounter_dlog10M_func = lambda m: self.dN_encounter_dlog10M(log10M=m, normalization=normalization, slope=slope, concentration_fac=concentration_fac, gamma=gamma, beta=beta, M_hm=M_hm)
        dN_encounter_dlog10M_eval = jax.vmap(dN_encounter_dlog10M_func)(log10_m_arr)
        mass_integral = trapezoid(y=dN_encounter_dlog10M_eval, x=log10_m_arr)
        return mass_integral 

    @eqx.filter_jit
    def sample_masses(self, log10M_min, log10M_max, key, normalization=1.0, slope=-1.9, concentration_fac=1.0, gamma=2.7, M_hm=0.0, beta=0.99, array_length=1_000):
        """
        Sample masses of subhalos from the mass function.
        """
        key1, key2, key3 = random.split(key, 3)
        N_encounter_rate = self.N_encounter(log10M_min=log10M_min, log10M_max=log10M_max, normalization=normalization, slope=slope, concentration_fac=concentration_fac, gamma=gamma, beta=beta, M_hm=M_hm)
        N_encounter = jax.random.poisson(key1, N_encounter_rate)

        # Dense array that we will resample from
        log10_M_arr = jnp.linspace(log10M_min, log10M_max, 5_000)
        dN_dlog10M_func = lambda m: self.dN_encounter_dlog10M(log10M=m, normalization=normalization, slope=slope, concentration_fac=concentration_fac, gamma=gamma, beta=beta, M_hm=M_hm)
        dNenc_dlog10M = jax.vmap(dN_dlog10M_func)(log10_M_arr)
        
        dlog10M = log10_M_arr[1] - log10_M_arr[0]
        prob = dNenc_dlog10M * dlog10M / jnp.sum(dNenc_dlog10M * dlog10M)
        prob = prob / jnp.sum(prob)

        samps = jax.random.choice(a=log10_M_arr, p=prob, shape=(array_length,), key=key2)

        idx = jnp.arange(array_length)
        # Pad the unused array elements with 0 based on the sampled N_encounter
        return dict(log10_mass=jnp.where(idx >= N_encounter, 0.0, samps), N_encounter=N_encounter, N_encounter_rate=N_encounter_rate)

    @eqx.filter_jit
    def compute_N_enclosed(self, log10M_min, log10M_max, r_min, r_max, slope=-1.9, gamma=2.7, M_hm=0.0, beta=0.99):
        """
        Compute the total number of subhalos in a given range of radii and masses.
        """
        log10mass_arr = jnp.linspace(log10M_min, log10M_max, 101)
        r_arr = jnp.linspace(r_min, r_max, 100)
        log10_M, R = jnp.meshgrid(log10mass_arr, r_arr, indexing='ij')
        
        func_map = lambda r_val, m_val: self.dn_dlog10M(r_val, m_val, slope=slope, gamma=gamma, M_hm=M_hm, beta=beta)
        inp = jax.vmap(func_map)(R.ravel(), log10_M.ravel()).reshape(R.shape)
        inp = inp * 4 * jnp.pi * R**2
        
        I1 = trapezoid(y=inp, x=log10mass_arr, axis=0)
        I2 = trapezoid(y=I1, x=r_arr)
        return I2