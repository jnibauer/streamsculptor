import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import equinox as eqx
from jax.scipy.integrate import trapezoid
import astropy.units as u

# Assuming jc and interpax are accessible from your environment
from streamsculptor import JaxCoords as jc
import interpax

from streamsculptor.streamhelpers import compute_length_oscillations

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
    length_osc: dict = None

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
                 disk_factor=1.0,  # Disk suppresses number of subhalos by factor of 3 in Erkal+2016
                 prog_today = None,
                 first_strip_lead = None,
                 first_strip_trail = None,
                 pot = None,
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

        if prog_today is not None and first_strip_lead is not None and first_strip_trail is not None and pot is not None and t_age is not None:
            #self.prog_today = prog_today
            #self.first_strip_lead = first_strip_lead # at observation time
            #self.first_strip_trail = first_strip_trail # at observation time
            #self.pot = pot
            self.length_osc = compute_length_oscillations(prog_today=prog_today, 
                                                          first_stripped_lead=first_strip_lead, 
                                                          first_stripped_trail=first_strip_trail, 
                                                          pot=pot, 
                                                          t_age=t_age,
                                                          length_today=self.l_obs)
        
        




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
    def dN_encounter_dlog10M_general(self, log10M: jnp.array, b_max_func: callable, normalization=1.0, slope=-1.9, concentration_fac=1.0, gamma=2.7, M_hm=0.0, beta=0.99, linear_growth=False):
        """
        Generalized version of dN_encounter_dlog10M that incorporates stream length oscillations.
        Takes a function b_max_func that can depend on log10M and returns b_max. 
        log10M: a single input mass. Use vmap to vectorize over many masses

        - dN_enc/dlog10M proportional to integral over time of l(t) * dn/dlog10M(r(t), log10M)
        """
        dn_dlog10M_val = self.dn_dlog10M(log10M=log10M, r=self.orbital_r, slope=slope, gamma=gamma, beta=beta, M_hm=M_hm)
        b_max = b_max_func(log10M)
        
        length_osc_ts = self.length_osc['ts'] # this goes from e.g. -3000 Myr to 0
        length_osc_flip = jnp.flip(jnp.abs(length_osc_ts)) # this goes from 0 to 3000 Myr, and is the same length as legnth_osc_ts
        # interpolate this over the orbit times to get length as a function of time
        if linear_growth:
             l_of_t = self.l_obs * (self.orbit_ts / self.t_age)
        else:
            l_of_t = jnp.interp(self.orbit_ts, length_osc_flip, self.length_osc['length_func']) # same length as self.orbit_ts
        integrand = l_of_t * dn_dlog10M_val
        prefac = jnp.sqrt(2 * jnp.pi) * self.sigma * self.disk_factor * b_max 
        integral = trapezoid(y=integrand, x=self.orbit_ts) * prefac * normalization
        return integral



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
    def N_encounter_general(self, log10M_min, log10M_max, b_max_func, normalization=1.0, slope=-1.9, concentration_fac=1.0, gamma=2.7, M_hm=0.0, beta=0.99,linear_growth=False):
        """
        Generalized version of N_encounter that incorporates stream length oscillations.
        Takes a function b_max_func that can depend on log10M and returns b_max. 
        Integrates dN_encounter_dlog10M_general over the mass range. 
        """
        log10_m_arr = jnp.linspace(log10M_min, log10M_max, 1000)
        dN_encounter_dlog10M_func = lambda m: self.dN_encounter_dlog10M_general(log10M=m, b_max_func=b_max_func, normalization=normalization, slope=slope, concentration_fac=concentration_fac, gamma=gamma, beta=beta, M_hm=M_hm,linear_growth=linear_growth)
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
    def sample_masses_general(self, log10M_min, log10M_max, b_max_func, key, normalization=1.0, slope=-1.9, concentration_fac=1.0, gamma=2.7, M_hm=0.0, beta=0.99, array_length=1_000,linear_growth=False):
        """
        Sample masses of subhalos from the mass function.
        Utilizes dN_encounter_dlog10M_general, which integrates
        the product of the stream's length and dn/dlog10M over time, incporporating 
        stream length oscillations. 
        b_max_func: a function that takes log10M and returns b_max. 
        linear_growth: if True, assume stream length grows linearly with time. Default False, uses numerically estimated length oscillations.
        """
        key1, key2 = random.split(key, 2)
        N_encounter_rate = self.N_encounter_general(log10M_min=log10M_min, log10M_max=log10M_max, b_max_func=b_max_func, normalization=normalization, slope=slope, concentration_fac=concentration_fac, gamma=gamma, beta=beta, M_hm=M_hm, linear_growth=linear_growth)
        N_encounter = jax.random.poisson(key1, N_encounter_rate)

        # Dense array that we will resample from
        log10_M_arr = jnp.linspace(log10M_min, log10M_max, 5_000)
        dN_dlog10M_func = lambda m: self.dN_encounter_dlog10M_general(log10M=m, b_max_func=b_max_func, normalization=normalization, slope=slope, concentration_fac=concentration_fac, gamma=gamma, beta=beta, M_hm=M_hm,linear_growth=linear_growth)
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


class TNFWSampler:
    """
    Samples truncated NFW subhalo parameters (m_infall, c_infall, z_infall, f_bound)
    using pyhalo's tidal evolution and concentration models with a user-defined
    power-law subhalo mass function.

    pyhalo, colossus, and scipy are optional dependencies imported at instantiation.

    Parameters
    ----------
    z_eval : float
        Redshift at which the stream/host is evaluated (for infall distribution).
    logM_host : float
        log10(host halo mass / Msun).
    chost : float
        Host halo concentration (used by the tidal evolution model).
    log10M_low : float
        Lower bound on log10(infall mass / Msun) for SHMF sampling.
    log10M_high : float
        Upper bound on log10(infall mass / Msun) for SHMF sampling.
    slope : float
        Power-law slope of the SHMF (positive convention, default 1.9).
    bound_mass_cut : float
        Minimum surviving bound mass [Msun]; halos below this threshold are dropped.
    """

    def __init__(self,
                 z_eval=0.0,
                 logM_host=12.0,
                 chost=9.0,
                 log10M_low=6.0,
                 log10M_high=10.0,
                 slope=1.9,
                 concentration_fac=1.0,
                 bound_mass_cut=1e6):

        try:
            from pyHalo.Halos.accretion import InfallDistributionDirectMilkyWay30kpc
            from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
            from pyHalo.Halos.galacticus_truncation.interp_mass_loss import InterpGalacticusMW
            from astropy.cosmology import Planck18
            from colossus.cosmology import cosmology as colossus_cosmology
            from scipy.interpolate import interp1d
        except ImportError as e:
            raise ImportError(
                "TNFWSampler requires pyhalo and its dependencies. "
                "Install with: pip install pyHalo colossus scipy"
            ) from e


        colossus_cosmology.setCosmology('planck18')

        self.z_eval = z_eval
        self.logM_host = logM_host
        self.chost = chost
        self.log10M_low = log10M_low
        self.log10M_high = log10M_high
        self.slope = slope
        self.bound_mass_cut = bound_mass_cut
        self.concentration_fac = concentration_fac

        self._tidal_model = InterpGalacticusMW(rmax=30.0)
        self._infall_dist = InfallDistributionDirectMilkyWay30kpc(z_eval, logM_host)
        self._concentration_model = ConcentrationDiemerJoyce(Planck18)

        zvalues_interp = np.linspace(0.0, 20.0, 100)
        lookback_times = Planck18.lookback_time(zvalues_interp) #[Planck18.lookback_time(zi).value for zi in zvalues_interp]
        self._time_since_infall_interp = interp1d(zvalues_interp, lookback_times)

    def _sample_masses(self, N, seed):
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, 1.0, N)
        s = self.slope
        M_lo = 10**self.log10M_low
        M_hi = 10**self.log10M_high
        # inverse-CDF for dN/dM ~ M^{-s}
        masses = (x * (M_hi**(1 - s) - M_lo**(1 - s)) + M_lo**(1 - s))**(1.0 / (1 - s))
        return masses

    def sample(self, N, seed=0, verbose=True):
        """
        Sample N halos from the SHMF, apply tidal evolution, and return surviving halos.

        Parameters
        ----------
        N : int
            Number of halos to draw from the SHMF before the bound-mass filter.
        seed : int
            Seed for the numpy RNG used in mass sampling.
        verbose : bool
            Show a tqdm progress bar over the tidal-evolution loop.

        Returns
        -------
        dict of jnp.ndarray
            Keys: m_infall, c_infall, z_infall, f_bound (each shape [N_surviving]),
            and N_surviving (scalar int).
        """
        _m_infall = self._sample_masses(N, seed)

        m_infall_buf = np.zeros(N)
        c_infall_buf = np.zeros(N)
        z_infall_buf = np.zeros(N)
        f_bound_buf  = np.zeros(N)
        k = 0

        iterator = _m_infall
        if verbose:
            try:
                import tqdm
                iterator = tqdm.tqdm(_m_infall, desc="Sampling TNFW params")
            except ImportError:
                pass

        for m in iterator:
            z              = self._infall_dist(m)
            c              = self._concentration_model.nfw_concentration(m, z) * self.concentration_fac
            t_since_infall = self._time_since_infall_interp(z)
            log10_fbound   = self._tidal_model(np.log10(c), t_since_infall, self.chost)
            f              = 10**log10_fbound

            if m * f >= self.bound_mass_cut:
                m_infall_buf[k] = m
                c_infall_buf[k] = c
                z_infall_buf[k] = z
                f_bound_buf[k]  = f
                k += 1

        return dict(
            m_infall    = jnp.array(m_infall_buf[:k]),
            c_infall    = jnp.array(c_infall_buf[:k]),
            z_infall    = jnp.array(z_infall_buf[:k]),
            f_bound     = jnp.array(f_bound_buf[:k]),
            N_surviving = jnp.array(k),
        )

    def sample_fast(self, N, seed=0):
        """
        Vectorized version of ``sample`` using the same pyhalo call chain.

        np.vectorize is used where pyhalo functions require scalar inputs
        (_infall_dist, nfw_concentration, _tidal_model). The time-since-infall
        interpolator already accepts arrays and is called directly. The
        bound-mass filter is replaced by numpy boolean masking.

        Parameters
        ----------
        N : int
            Number of halos to draw before the bound-mass filter.
        seed : int
            Seed passed to ``_sample_masses`` for the SHMF draw.

        Returns
        -------
        dict of jnp.ndarray
            Same keys as ``sample``: m_infall, c_infall, z_infall, f_bound,
            N_surviving.
        """
        masses = self._sample_masses(N, seed)

        z_arr    = np.vectorize(self._infall_dist)(masses)
        c_arr    = np.vectorize(self._concentration_model.nfw_concentration)(masses, z_arr)
        t_si_arr = self._time_since_infall_interp(z_arr)
        log10f   = np.vectorize(self._tidal_model)(np.log10(c_arr), t_si_arr, self.chost)
        f_arr    = 10**log10f

        mask = masses * f_arr >= self.bound_mass_cut
        return dict(
            m_infall    = jnp.array(masses[mask]),
            c_infall    = jnp.array(c_arr[mask]),
            z_infall    = jnp.array(z_arr[mask]),
            f_bound     = jnp.array(f_arr[mask]),
            N_surviving = jnp.array(int(mask.sum())),
        )