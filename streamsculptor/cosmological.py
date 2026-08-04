"""
cosmological.py

Dark matter model in, subhalo impactors out.
Example usage:
    model = DarkMatterModel(c_scale=2.0, slope=-1.9, normalization=1.0)
    pop   = SubhaloPopulation(model, log10M_bound_range=(5.0, 8.3))
    xsec  = KickCrossSection(v_char=100.0, dV_min=0.1, b_ceiling=5.0)
    rates = ImpactRates(pop, xsec, orbit=prog_orb, t_age=t_age, l_obs=l_obs,
                        prog_today=prog_w, first_strip_lead=lead[0],
                        first_strip_trail=trail[0], pot=pot)

    rates.expected_impacts()          # Poisson rate
    cat = rates.sample_impactors(key) # masses AND matching TNFW structure

    # Get cartesian impact locs with ImpactGenerator and cat.b_bounds (for b_max sampling), then
    pot_sub = cat.to_subhalo_potential(cart, tImpact, t_window=150.0)


pyHalo, colossus and scipy are required for ``InfallPopulation`` (and hence
``SubhaloPopulation``); the rate machinery has no such dependency.
"""

from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import astropy.units as u

from streamsculptor.subhalostatistics import RateCalculator

jax.config.update("jax_enable_x64", True)

# G in the streamsculptor unit system: kpc^3 / (Msun Myr^2)
_G = 4.498502e-12
# km/s -> kpc/Myr
_KMS = float((1.0 * u.km / u.s).to(u.kpc / u.Myr).value)


# =============================================================================
# The dark matter model
# =============================================================================

@dataclass(frozen=True)
class PowerLawAmplitude:
    """
    Changing the infall mass-concentration relation changes the number of surviving 
    subhalos. More compact subhalos --> more numerous subhalos.
    We parametrize this survival enhancement as a power law in the concentration scaling factor, calibrated
    to the pyHalo population over a specific bound-mass window. That is, 
    dN/dlog10M_bound ~ amp * dN/dlog10M_bound|_CDM, where amp = c_scale ** exponent.

    The exact amplitude (``SubhaloPopulation.amplitude``) needs numpy and pyHalo, so
    it cannot be traced. This is the opt-in replacement when you need the amplitude
    inside a jit or a gradient. Not a default: the exponent depends on both the
    bound-mass window and the ``c_scale`` range it was fitted over (0.781 over
    1e5-2e8 Msun and c_scale in [0.5, 8]; 0.899 over c_scale in [0.5, 3]), and the
    amplitude is not a true power law -- 16 percent max residual over the wider range.
    Recalibrate with ``InfallPopulation.amplitude`` for the range you work in.
    """

    exponent: float = 0.781

    def __call__(self, c_scale):
        return c_scale ** self.exponent


@dataclass(frozen=True)
class DarkMatterModel:
    """
    A dark matter model: everything that distinguishes one universe from another.

    Parameters
    ----------
    c_scale : float
        Infall mass-concentration relation relative to pyHalo CDM. Acts on the
        survival amplitude (how many subhalos) and on the impactor structure (how
        compact they are) simultaneously -- which is why it must live in one place.
    slope : float
        SHMF slope, ``dN/dM ~ M**slope``, so CDM is -1.9. **Negative.** Sets both the
        Erkal mass function in the rate and the importance weights of the pyHalo
        population; a positive value raises rather than silently reweighting wrongly.
    normalization : float
        Overall multiplier on the impact rate. Changes how many impacts, not which.
    M_hm, gamma, beta : float
        WDM suppression of the mass function, ``(1 + gamma M_hm/M)**-beta``.
        ``M_hm = 0`` is CDM. Parameters from arXiv:1911.02663.
    amplitude_model : callable, optional
        Override for the survival amplitude, e.g. ``PowerLawAmplitude()``. Called as
        ``amplitude_model(c_scale)``. Default (None) computes it exactly from the
        pyHalo population.
    name : str, optional
        Label for plots and tables.
    """

    c_scale: float = 1.0
    slope: float = -1.9
    normalization: float = 1.0
    M_hm: float = 0.0
    gamma: float = 2.7
    beta: float = 0.99
    amplitude_model: Optional[Callable] = None
    name: Optional[str] = None

    def __post_init__(self):
        if self.slope > 0:
            raise ValueError(
                f"slope={self.slope} is positive; the convention here is "
                "dN/dM ~ M**slope, so CDM is slope=-1.9."
            )
        if self.c_scale <= 0:
            raise ValueError(f"c_scale={self.c_scale} must be positive.")
        if self.M_hm < 0:
            raise ValueError(f"M_hm={self.M_hm} must be non-negative.")

    def replace(self, **kwargs):
        """A copy with some fields changed, e.g. ``model.replace(c_scale=3.0)``."""
        return replace(self, **kwargs)

    @property
    def label(self):
        if self.name is not None:
            return self.name
        bits = [f"c={self.c_scale:g}", f"slope={self.slope:g}"]
        if self.normalization != 1.0:
            bits.append(f"norm={self.normalization:g}")
        if self.M_hm > 0:
            bits.append(f"M_hm={self.M_hm:.1e}")
        return ", ".join(bits)


CDM = DarkMatterModel(name="CDM")


# =============================================================================
# The pyHalo population
# =============================================================================

_DRAW_CACHE = {}


class InfallPopulation:
    """
    A frozen pyHalo infall population, tidally evolved on demand.

    Deliberately knows nothing about ``c_scale``: the concentration scaling is
    applied when ``f_bound`` is evaluated, not when the population is drawn. That is
    what makes a scan over ``c_scale`` free and a change of ``slope`` expensive --
    ``slope`` enters the importance weights, so it does require a redraw.

    Every halo shares common random numbers across concentration scalings (the tidal
    scatter quantiles are frozen at construction), so amplitudes are exactly 1.0 at
    ``c_scale = 1`` rather than 1 plus Monte Carlo noise.

    Parameters
    ----------
    slope : float
        SHMF slope, ``dN/dM ~ M**slope`` (negative).
    log10M_infall_range : (float, float)
        Infall-mass draw range [log10 Msun]. Must extend well below any bound-mass
        window you intend to use: median ``f_bound`` inside 30 kpc is ~3e-3, so a
        halo ending at 1e5 Msun typically fell in 2-3 dex higher.
    N : int
        Population size.
    z_eval, logM_host, chost, rmax :
        pyHalo model settings. ``rmax`` is the radial extent of the Galacticus tidal
        calibration -- results are only valid for streams inside it.
    seed : int
        Seed for the frozen draw.
    """

    def __init__(self,
                 slope=-1.9,
                 log10M_infall_range=(4.0, 11.0),
                 N=150_000,
                 z_eval=0.0,
                 logM_host=12.0,
                 chost=9.0,
                 rmax=30.0,
                 seed=0):

        try:
            from pyHalo.Halos.accretion import InfallDistributionDirectMilkyWay30kpc
            from pyHalo.Halos.concentration import ConcentrationDiemerJoyce
            from pyHalo.Halos.galacticus_truncation.interp_mass_loss import InterpGalacticusMW
            from astropy.cosmology import Planck18
            from colossus.cosmology import cosmology as colossus_cosmology
            from scipy.interpolate import interp1d
            from scipy.stats import johnsonsu
        except ImportError as e:
            raise ImportError(
                "InfallPopulation requires pyhalo and its dependencies."
            ) from e

        if slope > 0:
            raise ValueError(
                f"slope={slope} is positive; the convention here is dN/dM ~ M**slope, "
                "so CDM is slope=-1.9."
            )

        colossus_cosmology.setCosmology("planck18")

        self.slope = slope
        self.log10M_infall_range = tuple(log10M_infall_range)
        self.N = int(N)
        self.z_eval = z_eval
        self.logM_host = logM_host
        self.chost = chost
        self.rmax = rmax
        self.seed = seed
        self._johnsonsu = johnsonsu

        self._tidal = InterpGalacticusMW(rmax=rmax)
        infall_dist = InfallDistributionDirectMilkyWay30kpc(z_eval, logM_host)
        conc = ConcentrationDiemerJoyce(Planck18)

        zgrid = np.linspace(0.0, 20.0, 100)
        t_of_z = interp1d(zgrid, Planck18.lookback_time(zgrid))

        rng = np.random.default_rng(seed)
        lo, hi = self.log10M_infall_range
        self.m_infall = 10.0 ** rng.uniform(lo, hi, self.N)

        # SHMF importance weights: dN/dlog10M ~ M * dN/dM ~ M**(1 + slope)
        w = self.m_infall ** (1.0 + slope)
        self.weights = w / w.mean()

        # pyHalo's infall distribution draws from the global numpy RNG; seed it for
        # reproducibility without disturbing the caller's stream.
        _state = np.random.get_state()
        try:
            np.random.seed(seed)
            self.z_infall = np.vectorize(infall_dist)(self.m_infall)
            self.c_cdm = np.vectorize(conc.nfw_concentration)(self.m_infall, self.z_infall)
        finally:
            np.random.set_state(_state)

        self.t_since_infall = np.asarray(t_of_z(self.z_infall))
        self.u_scatter = rng.uniform(0.0, 1.0, self.N)  # frozen tidal-scatter quantiles

    # ------------------------------------------------------------------

    def f_bound(self, c_scale=1.0):
        """Bound mass fraction of every halo under ``c -> c_scale * c_CDM``."""
        c = c_scale * self.c_cdm
        log10c = np.clip(np.log10(c), np.log10(2.0), np.log10(128.0))
        t = np.clip(self.t_since_infall, 0.0, 12.9)
        chost = np.clip(self.chost, 6.0, 12.0)

        pts = np.column_stack([t, log10c, np.full_like(t, chost)])
        a, b = self._tidal._a_interp(pts), self._tidal._b_interp(pts)
        log10f = np.minimum(self._johnsonsu.ppf(self.u_scatter, a, b), 0.0)
        return np.clip(10.0 ** log10f, 0.0, 1.0)

    def m_bound(self, c_scale=1.0):
        """Present-day bound mass of every halo."""
        return self.m_infall * self.f_bound(c_scale)

    def n_survive(self, c_scale, log10M_bound_range):
        """SHMF-weighted count of halos landing inside the bound-mass window."""
        mb = self.m_bound(c_scale)
        lo, hi = log10M_bound_range
        inside = (mb >= 10.0 ** lo) & (mb <= 10.0 ** hi)
        return float(self.weights[inside].sum())

    def amplitude(self, c_scale, log10M_bound_range):
        """
        Survival enhancement relative to pyHalo CDM: the factor by which the subhalo
        *number density* changes when the infall MCR is scaled. Exactly 1.0 at
        ``c_scale = 1``.
        """
        return (self.n_survive(c_scale, log10M_bound_range)
                / self.n_survive(1.0, log10M_bound_range))


def infall_population(slope=-1.9, log10M_infall_range=(4.0, 11.0), N=150_000,
                      z_eval=0.0, logM_host=12.0, chost=9.0, rmax=30.0, seed=0):
    """
    Cached ``InfallPopulation``. Repeated calls with the same arguments return the
    same object, so scanning models that share a ``slope`` costs one pyHalo draw.
    """
    key = (float(slope), tuple(log10M_infall_range), int(N), float(z_eval),
           float(logM_host), float(chost), float(rmax), int(seed))
    if key not in _DRAW_CACHE:
        _DRAW_CACHE[key] = InfallPopulation(
            slope=slope, log10M_infall_range=log10M_infall_range, N=N,
            z_eval=z_eval, logM_host=logM_host, chost=chost, rmax=rmax, seed=seed,
        )
    return _DRAW_CACHE[key]


def clear_population_cache():
    """Drop every cached ``InfallPopulation``."""
    _DRAW_CACHE.clear()


class SubhaloPopulation:
    """
    A pyHalo population under one specific dark matter model, restricted to the
    bound-mass window impacts are simulated for.

    Supplies two things to the rest of the pipeline: the survival ``amplitude`` that
    scales the rate, and the pool that impactors are resampled from. Because both
    come from the same object, the impactor structure and the number of impacts
    cannot describe different universes.

    Parameters
    ----------
    model : DarkMatterModel
    log10M_bound_range : (float, float)
        Bound-mass window [log10 Msun]. The range you actually simulate impacts for.
    log10M_infall_range, N, z_eval, logM_host, chost, rmax, seed :
        Passed to the underlying draw. Populations are cached on these plus
        ``model.slope``, so changing only ``c_scale`` reuses the draw.
    """

    def __init__(self,
                 model=CDM,
                 log10M_bound_range=(5.0, 8.3),
                 log10M_infall_range=(4.0, 11.0),
                 N=150_000,
                 z_eval=0.0,
                 logM_host=12.0,
                 chost=9.0,
                 rmax=30.0,
                 seed=0):

        self.model = model
        self.log10M_bound_range = tuple(log10M_bound_range)
        self.draw = infall_population(
            slope=model.slope, log10M_infall_range=log10M_infall_range, N=N,
            z_eval=z_eval, logM_host=logM_host, chost=chost, rmax=rmax, seed=seed,
        )

        lo, hi = self.log10M_bound_range
        if lo < self.draw.log10M_infall_range[0]:
            raise ValueError(
                f"bound-mass window starts at 1e{lo} Msun but the infall draw starts "
                f"at 1e{self.draw.log10M_infall_range[0]} Msun. Halos ending just "
                "below the window's floor can fall in from below it; extend "
                "log10M_infall_range downward."
            )

        f_bound = self.draw.f_bound(model.c_scale)
        m_bound = self.draw.m_infall * f_bound
        keep = (m_bound >= 10.0 ** lo) & (m_bound <= 10.0 ** hi)
        if not keep.any():
            raise ValueError(
                f"No population members in the bound-mass window "
                f"{self.log10M_bound_range}."
            )

        self.m_infall = self.draw.m_infall[keep]
        self.z_infall = self.draw.z_infall[keep]
        self.t_since_infall = self.draw.t_since_infall[keep]
        self.c_infall = model.c_scale * self.draw.c_cdm[keep]
        self.f_bound = f_bound[keep]
        self.m_bound = m_bound[keep]
        self.weights = self.draw.weights[keep]

        self._init_kwargs = dict(
            log10M_bound_range=self.log10M_bound_range,
            log10M_infall_range=log10M_infall_range, N=N, z_eval=z_eval,
            logM_host=logM_host, chost=chost, rmax=rmax, seed=seed,
        )

    def __len__(self):
        return len(self.m_bound)

    def with_model(self, model):
        """
        The same population settings under a different dark matter model. Reuses the
        cached draw when the slope is unchanged.
        """
        return SubhaloPopulation(model, **self._init_kwargs)

    def amplitude(self):
        """
        Survival amplitude for this model. Exact (from the population) unless the
        model supplies an ``amplitude_model``.
        """
        if self.model.amplitude_model is not None:
            return float(self.model.amplitude_model(self.model.c_scale))
        return self.draw.amplitude(self.model.c_scale, self.log10M_bound_range)

    def resample(self, n, weight_fn=None, seed=0):
        """
        Draw ``n`` halos from the population, optionally reweighted by mass.

        Parameters
        ----------
        n : int
            Number of halos to draw (with replacement).
        weight_fn : callable, optional
            Extra weight as a function of bound mass [Msun], on top of the SHMF
            importance weights. For impactors this is the cross-section
            ``b_max(M)`` -- see ``ImpactRates.sample_impactors``. None means a fair
            draw from the population's own mass function.
        seed : int

        Returns
        -------
        dict of np.ndarray with the drawn halos' infall properties, bound masses,
        weights and TNFW parameters, plus ``idx`` into the population.
        """
        from streamsculptor.tnfw import _tidally_evolved_nfw_params

        n = int(n)
        w = np.asarray(self.weights, dtype=float)
        if weight_fn is not None:
            w = w * np.asarray(weight_fn(self.m_bound), dtype=float)
        p = w / w.sum()

        rng = np.random.default_rng(seed)
        idx = rng.choice(len(p), size=n, replace=True, p=p)

        m_inf, c_inf = self.m_infall[idx], self.c_infall[idx]
        z_inf, f_b = self.z_infall[idx], self.f_bound[idx]
        t_inf = self.t_since_infall[idx]

        rhos, rs, ft, rt = _tidally_evolved_nfw_params(
            jnp.asarray(m_inf), jnp.asarray(c_inf), jnp.asarray(z_inf), jnp.asarray(f_b)
        )

        return dict(
            idx=idx, m_bound=self.m_bound[idx], m_infall=m_inf, c_infall=c_inf,
            z_infall=z_inf, t_since_infall=t_inf, f_bound=f_b,
            rhos=np.asarray(rhos), rs=np.asarray(rs),
            ft=np.asarray(ft), rt=np.asarray(rt),
        )


# =============================================================================
# Cross-sections
# =============================================================================

class CrossSection(eqx.Module):
    """
    Maximum impact parameter as a function of subhalo bound mass.

    The only thing that distinguishes these rates from a bare ``RateCalculator``.
    Subclasses implement ``b_max(m_bound)`` in kpc for masses in Msun; ``__call__``
    takes log10 mass, which is the form the rate engine wants.
    """

    def b_max(self, m_bound):
        raise NotImplementedError

    @eqx.filter_jit
    def __call__(self, log10M):
        return self.b_max(10.0 ** log10M)


class KickCrossSection(CrossSection):
    """
    ``b_max`` set by the faintest velocity kick worth simulating, in the point-mass
    impulse limit:

        b_max(M) = min( 2 G M / (v_char * dV_min),  b_ceiling )

    Concentration-independent by construction, since at ``b >~ r_t`` a subhalo acts as
    a point mass. Tying ``b_max`` to ``r_s`` instead shrinks the sampling volume for
    compact subhalos at the same moment the survival amplitude says there are more of
    them -- two concentration effects with opposite signs, partly cancelling for no
    physical reason. Here concentration moves the amplitude and nothing else.

    Holding ``v_char`` fixed, rather than integrating ``b_max`` over the relative
    velocity distribution, is what keeps the ``sqrt(2 pi) sigma`` prefactor of
    Erkal+2016 valid unchanged: a velocity-dependent b_max would move inside the
    velocity integral, and the flux (perpendicular component) and the kick (total
    speed) would no longer combine into a constant. Choose ``v_char`` low in the
    relative-speed distribution so b_max errs generous -- over-included encounters are
    weak ones that cost compute, not accuracy. 100-150 km/s is the 2nd-7th percentile
    for a typical stream.

    ``b_ceiling`` is not optional. Because ``b_max ~ M`` while
    ``dN/dlog10M ~ M^-0.9``, the encounter count runs as ``M^0.1`` -- flat per decade.
    Uncapped, a 2e8 Msun subhalo at ``v_char = 50 km/s`` gets ``b_max = 344 kpc``,
    which is a global tidal field, not a flyby, and every assumption behind the rate
    formula has failed by then.

    Parameters
    ----------
    v_char, dV_min : float
        Characteristic relative speed and minimum kick [km/s]. The rate scales as
        ``1/dV_min``; scan it and confirm your summary statistic plateaus.
    b_ceiling : float
        Hard cap [kpc].
    """

    v_char: jnp.ndarray
    dV_min: jnp.ndarray
    b_ceiling: jnp.ndarray

    def __init__(self, v_char=100.0, dV_min=0.1, b_ceiling=5.0):
        self.v_char = jnp.asarray(v_char * _KMS)
        self.dV_min = jnp.asarray(dV_min * _KMS)
        self.b_ceiling = jnp.asarray(b_ceiling)

    @eqx.filter_jit
    def b_max(self, m_bound):
        b = 2.0 * _G * jnp.asarray(m_bound) / (self.v_char * self.dV_min)
        return jnp.minimum(b, self.b_ceiling)

    @property
    def m_ceiling(self):
        """Bound mass above which ``b_max`` is capped [Msun]."""
        return float(self.b_ceiling * self.v_char * self.dV_min / (2.0 * _G))


class ScaleRadiusCrossSection(CrossSection):
    """
    ``b_max = b_max_fac * r_s(M)`` with ``r_s = r_s_ref sqrt(M / m_ref)``.

    The convention in ``subhalostatistics.RateCalculator`` and in
    ``perturbedstream``. Reproduces a bare ``RateCalculator`` exactly when
    ``b_max_fac`` matches. Note this makes ``dN_enc/dlog10M ~ M^-0.4``, falling, where
    ``KickCrossSection`` gives ``M^+0.1`` with a break at its ceiling -- the two are
    not small perturbations on each other.
    """

    b_max_fac: jnp.ndarray
    r_s_ref: jnp.ndarray
    m_ref: jnp.ndarray
    b_ceiling: Optional[jnp.ndarray]

    def __init__(self, b_max_fac=5.0, r_s_ref=1.05, m_ref=1e8, b_ceiling=None):
        self.b_max_fac = jnp.asarray(b_max_fac)
        self.r_s_ref = jnp.asarray(r_s_ref)
        self.m_ref = jnp.asarray(m_ref)
        self.b_ceiling = None if b_ceiling is None else jnp.asarray(b_ceiling)

    @eqx.filter_jit
    def r_s(self, m_bound):
        return self.r_s_ref * jnp.sqrt(jnp.asarray(m_bound) / self.m_ref)

    @eqx.filter_jit
    def b_max(self, m_bound):
        b = self.b_max_fac * self.r_s(m_bound)
        return b if self.b_ceiling is None else jnp.minimum(b, self.b_ceiling)


# =============================================================================
# Impactor catalog
# =============================================================================

@dataclass
class ImpactorCatalog:
    """
    The impactors of one realization: bound masses and the TNFW structure of the same
    halos, plus each one's ``b_max``.

    Dict-style access (``cat["m_bound"]``) works alongside attributes. Knows its two
    handoffs -- ``b_bounds`` for ``ImpactGenerator`` and ``to_subhalo_potential`` for
    the perturbing potential -- so neither has to be wired by hand.
    """

    m_bound: np.ndarray
    m_infall: np.ndarray
    c_infall: np.ndarray
    z_infall: np.ndarray
    t_since_infall: np.ndarray
    f_bound: np.ndarray
    rhos: np.ndarray
    rs: np.ndarray
    ft: np.ndarray
    rt: np.ndarray
    b_max: np.ndarray
    model: Optional[DarkMatterModel] = None
    log10M_bound_range: Optional[Tuple[float, float]] = None
    rate: Optional[float] = None

    _FIELDS = ("m_bound", "m_infall", "c_infall", "z_infall", "t_since_infall",
               "f_bound",
               "rhos", "rs", "ft", "rt", "b_max")

    def __len__(self):
        return len(self.m_bound)

    def __getitem__(self, key):
        return getattr(self, key)

    def keys(self):
        return self._FIELDS

    def as_dict(self):
        return {k: getattr(self, k) for k in self._FIELDS}

    @classmethod
    def from_padded(cls, batch, i=None, drop_padding=True, model=None,
                    log10M_bound_range=None):
        """
        Pull one realization out of a batched, zero-padded draw.

        Parameters
        ----------
        batch : dict
            Output of ``ImpactRates.impactor_sampler``, either a single realization
            (arrays of shape ``(max_num_impacts,)``) or a vmapped batch (shape
            ``(n_keys, max_num_impacts)``).
        i : int, optional
            Which realization, if ``batch`` is a vmapped batch.
        drop_padding : bool
            Drop the padded slots, giving the variable-length catalog the rest of the
            pipeline expects. Set False to keep the fixed-size arrays.
        """
        d = {k: np.asarray(batch[k]) for k in cls._FIELDS}
        valid = np.asarray(batch["valid"])
        if i is not None:
            d = {k: v[i] for k, v in d.items()}
            valid = valid[i]
        if valid.ndim > 1:
            raise ValueError(
                "batch holds several realizations; pass i= to select one."
            )
        if drop_padding:
            d = {k: v[valid] for k, v in d.items()}
        rate = batch.get("rate")
        return cls(**d, model=model, log10M_bound_range=log10M_bound_range,
                   rate=None if rate is None else float(np.asarray(rate).ravel()[0]))

    @property
    def b_bounds(self):
        """
        Impact-parameter bounds for ``ImpactGenerator``, shape ``(2, N)``: row 0 the
        lower bound per impactor, row 1 its own ``b_max``. This orientation is what
        ``ImpactGenerator`` indexes; the ``(N, 2)`` built in
        ``perturbedstream.gen_perturbed_stream`` is a different (analytic-Hernquist)
        path and is not interchangeable.
        """
        b_high = jnp.asarray(self.b_max)
        return jnp.stack([jnp.zeros_like(b_high), b_high])

    def to_subhalo_potential(self, cart, tImpact, t_window=150.0, n_r=128):
        """
        Build the perturbing potential from this catalog and an ``ImpactGenerator``
        result.

        Parameters
        ----------
        cart : array (N, 6)
            ``ImpactGenerator.get_subhalo_ImpactParams()["CartesianImpactParams"]``.
        tImpact : array (N,)
            ``...["ImpactFrameParams"]["tImpact"]``.
        t_window : float
            Time window [Myr] each subhalo is active for.
        """
        from streamsculptor.tnfw import TNFWSubhaloLinePotential

        cart = jnp.asarray(cart)
        if cart.shape[0] != len(self):
            raise ValueError(
                f"cart has {cart.shape[0]} rows but the catalog holds {len(self)} "
                "impactors."
            )
        return TNFWSubhaloLinePotential(
            rhos=self.rhos, rs=self.rs, ft=self.ft, rt=self.rt,
            subhalo_x0=cart[:, :3], subhalo_v=cart[:, 3:],
            subhalo_t0=jnp.asarray(tImpact), t_window=t_window, n_r=n_r,
        )

    def summary(self):
        """One-line description, for tables."""
        return (f"N={len(self)}  median M_bound={np.median(self.m_bound):.2e}  "
                f"f(>1e7)={np.mean(self.m_bound > 1e7):.3f}  "
                f"median r_s={np.median(self.rs):.3f} kpc")


# =============================================================================
# Rates
# =============================================================================

class ImpactRates:
    """
    Expected number of subhalo impacts on a stream, and the impactors themselves.

    Thin layer over ``subhalostatistics.RateCalculator``: the Erkal+2016 time
    integral is the engine's, the cross-section is ours, and the survival amplitude
    enters through the engine's ``normalization`` argument. Nothing about the rate
    math is reimplemented here.

    Does **not** own the stream. It gives you a number of impacts and their
    properties; placing them (``ImpactGenerator``) is a separate step, so the rate
    machinery stays independent of the stream representation.

    Parameters
    ----------
    population : SubhaloPopulation
        Supplies the dark matter model, the survival amplitude and the resampling
        pool.
    cross_section : CrossSection
        Sets ``b_max(M)``.
    orbit : diffrax solution
        Progenitor orbit. ``orbit.ys[:, 0:3]`` gives the radius track and
        ``orbit.ts`` the times.
    t_age : float
        Stream age [Myr].
    l_obs : float
        Present-day stream length [kpc].
    prog_today, first_strip_lead, first_strip_trail, pot :
        Supply all four to get stream-length oscillations, and hence
        ``linear_growth=False``. Without them only linear growth is available.
    linear_growth : bool, optional
        Default None: use the oscillations if they could be computed, else linear
        growth. Passing True forces the Erkal+2016 linear assumption.
    sigma_kms : float
        Subhalo velocity dispersion [km/s].
    a0, c0, m0, alpha, r_minus2, disk_factor :
        Erkal+2016 SHMF and radial-profile parameters. ``disk_factor`` is the
        suppression by the disk (Erkal+2016 use 1/3).
    """

    def __init__(self,
                 population,
                 cross_section,
                 orbit,
                 t_age,
                 l_obs,
                 prog_today=None,
                 first_strip_lead=None,
                 first_strip_trail=None,
                 pot=None,
                 linear_growth=None,
                 sigma_kms=180.0,
                 a0=1.77e-5,
                 c0=2.02e-13,
                 m0=2.52e7,
                 alpha=0.678,
                 r_minus2=162.4,
                 disk_factor=1.0):

        self.population = population
        self.cross_section = cross_section
        self.orbit_ts_signed = jnp.asarray(orbit.ts)

        self.engine = RateCalculator(
            orbit=orbit,
            t_age=t_age,
            b_max_fac=1.0,   # unused: the cross-section is passed in explicitly
            l_obs=l_obs,
            sigma=sigma_kms * _KMS,
            a0=a0, c0=c0, m0=m0, alpha=alpha, r_minus2=r_minus2,
            disk_factor=disk_factor,
            prog_today=prog_today,
            first_strip_lead=first_strip_lead,
            first_strip_trail=first_strip_trail,
            pot=pot,
        )

        have_osc = self.engine.length_osc is not None
        if linear_growth is None:
            linear_growth = not have_osc
        elif not linear_growth and not have_osc:
            raise ValueError(
                "linear_growth=False needs the stream-length oscillations: pass "
                "prog_today, first_strip_lead, first_strip_trail and pot."
            )
        self.linear_growth = bool(linear_growth)

        if not have_osc:
            # The engine dereferences length_osc even when linear_growth=True, so it
            # can never be None. Fill it with the exact linear ramp: the two growth
            # models then agree identically instead of one of them being a trap.
            n = len(self.engine.orbit_ts)
            osc = dict(ts=jnp.linspace(-t_age, 0.0, n),
                       length_func=jnp.asarray(l_obs) * jnp.linspace(0.0, 1.0, n))
            # eqx.tree_at cannot address a None field on this class, and the engine is
            # ours and not yet shared, so set it directly.
            object.__setattr__(self.engine, "length_osc", osc)

    # ---- the model -------------------------------------------------------

    @property
    def model(self):
        return self.population.model

    @property
    def log10M_bound_range(self):
        return self.population.log10M_bound_range

    @property
    def amplitude(self):
        """Survival amplitude for this model."""
        return self.population.amplitude()

    @property
    def normalization(self):
        """
        What the engine sees: the model's normalization times the survival amplitude.
        The amplitude is a multiplier on the subhalo number density, so it needs no
        separate machinery.
        """
        return float(self.model.normalization) * float(self.amplitude)

    def fsub(self, M_vir=1e12, r_vir=300.0, log10M_min=None, log10M_max=None):
        """
        Fraction of ``M_vir`` bound in subhalos inside ``r_vir`` under this dark matter
        model.

        Uses ``self.normalization``, so the pyHalo survival amplitude is included: if
        scaling the concentrations makes twice as many subhalos survive, this reports
        twice the mass. The model's ``slope`` and WDM parameters enter the integral too.

        Defaults to the population's ``log10M_bound_range`` -- the window impacts are
        actually simulated for -- so this answers "what fraction of the halo is in the
        subhalos I am modeling". That is a smaller number than a cosmologist's fsub,
        which would run to 1e10 Msun and beyond. Pass ``log10M_min``/``log10M_max``
        explicitly to widen it, remembering the integrand is flat per dex and so the
        result is dominated by whatever upper limit you choose.

        Caveat worth knowing: the Galacticus tidal calibration behind the survival
        amplitude is only valid inside ``rmax`` (30 kpc by default), so integrating out
        to a 300 kpc virial radius extrapolates that amplitude well past where it was
        calibrated. The Erkal+2016 radial profile itself is fine out there.
        """
        lo, hi = self.log10M_bound_range
        m = self.model
        return float(self.engine.fsub(
            normalization=self.normalization, M_vir=M_vir, r_vir=r_vir,
            log10M_min=lo if log10M_min is None else log10M_min,
            log10M_max=hi if log10M_max is None else log10M_max,
            slope=m.slope, gamma=m.gamma, M_hm=m.M_hm, beta=m.beta,
        ))

    def _engine_kwargs(self):
        m = self.model
        return dict(normalization=self.normalization, slope=m.slope,
                    gamma=m.gamma, M_hm=m.M_hm, beta=m.beta,
                    linear_growth=self.linear_growth)

    # ---- rates -----------------------------------------------------------

    def dN_dlog10M(self, log10M):
        """
        Expected encounters per dex of bound mass. Accepts a scalar or an array.
        """
        log10M = jnp.asarray(log10M)
        f = lambda m: self.engine.dN_encounter_dlog10M_general(
            log10M=m, b_max_func=self.cross_section, **self._engine_kwargs()
        )
        return f(log10M) if log10M.ndim == 0 else jax.vmap(f)(log10M)

    def expected_impacts(self, log10M_min=None, log10M_max=None):
        """Poisson rate over the bound-mass window."""
        lo, hi = self.log10M_bound_range
        lo = lo if log10M_min is None else log10M_min
        hi = hi if log10M_max is None else log10M_max
        return float(self.engine.N_encounter_general(
            log10M_min=lo, log10M_max=hi, b_max_func=self.cross_section,
            **self._engine_kwargs()
        ))

    def sample_n_impacts(self, key, log10M_min=None, log10M_max=None):
        """Poisson draw of the number of impacts."""
        rate = self.expected_impacts(log10M_min, log10M_max)
        return int(jax.random.poisson(key, rate))

    def nsub_along_orbit(self):
        """
        ``(times, n_sub)`` along the progenitor orbit, for ``ImpactGenerator``'s
        ``nsub_times`` / ``nsub_vals``. Impact times should be sampled from
        ``p(t) ~ l(t) n_sub(r(t))``, not from the stream length alone.
        """
        return self.orbit_ts_signed, self.engine.nsub(self.engine.orbital_r)

    # ---- impactors -------------------------------------------------------

    def sample_impactors(self, key, seed=None, n_impacts=None):
        """
        Poisson-draw the number of impacts, then resample the population for their
        properties.

        Impactors are **not** a fair sample of the population. The encounter rate
        weights by cross-section, and since the radial profile factorises out of
        ``dn/dlog10M``, the only mass-dependent weight is ``b_max(M)``:

            p(impactor = halo i)  ~  w_i * b_max(m_bound_i)

        A uniform draw badly under-represents massive impactors: with a typical
        ``KickCrossSection``, ~1 percent of a fair draw lies above 1e7 Msun against
        ~11 percent of the correctly weighted draw, and the median impactor mass
        shifts by nearly a dex.

        Parameters
        ----------
        key : PRNGKey
            For the Poisson draw, and for the resampling seed unless ``seed`` is given.
        seed : int, optional
            numpy seed for the resampling.
        n_impacts : int, optional
            Skip the Poisson draw and take exactly this many. For diagnostics that
            need a large sample of the impactor distribution.

        Returns
        -------
        ImpactorCatalog
        """
        key_n, key_seed = jax.random.split(key)
        rate = self.expected_impacts()
        if n_impacts is None:
            n_impacts = int(jax.random.poisson(key_n, rate))
        if seed is None:
            seed = int(jax.random.randint(key_seed, (), 0, 2**31 - 1))

        drawn = self.population.resample(
            n_impacts, weight_fn=self.cross_section.b_max, seed=seed
        )
        b_max = np.asarray(self.cross_section.b_max(drawn["m_bound"]), dtype=float)

        return ImpactorCatalog(
            m_bound=drawn["m_bound"], m_infall=drawn["m_infall"],
            c_infall=drawn["c_infall"], z_infall=drawn["z_infall"],
            t_since_infall=drawn["t_since_infall"],
            f_bound=drawn["f_bound"], rhos=drawn["rhos"], rs=drawn["rs"],
            ft=drawn["ft"], rt=drawn["rt"], b_max=b_max,
            model=self.model, log10M_bound_range=self.log10M_bound_range, rate=rate,
        )

    # ---- batched sampling ------------------------------------------------

    def suggest_max_num_impacts(self, n_sigma=5.0):
        """
        A fixed array length that a Poisson draw will overflow only rarely:
        ``rate + n_sigma sqrt(rate)``, rounded up. At ``n_sigma = 5`` the overflow
        probability is ~1e-6 per realization, so check ``n_impacts`` anyway if you
        draw millions.
        """
        rate = self.expected_impacts()
        return int(np.ceil(rate + n_sigma * np.sqrt(rate)))

    def impactor_sampler(self, max_num_impacts, rate=None):
        """
        A jittable, vmappable ``key -> dict`` sampler with fixed-size output.

        Everything key-dependent is pure JAX; the pyHalo population and the survival
        amplitude are baked in as constants at build time. So::

            sampler = rates.impactor_sampler(max_num_impacts=80)
            batch = jax.vmap(sampler)(jax.random.split(key, 512))
            # every array has shape (512, 80)

        Padding: slots beyond the drawn ``n_impacts`` get **zero mass and zero
        density** (``m_bound``, ``m_infall``, ``rhos``, ``b_max`` = 0) but **unit
        radii** (``rs``, ``rt``, ``ft`` = 1). Padding radii with zeros instead would
        put 0/0 into every downstream profile evaluation; with ``rhos = 0`` a padded
        subhalo contributes exactly nothing to the potential, which is the point.
        ``valid`` is the boolean mask, and ``n_impacts`` the drawn count.

        A realization overflows if ``n_impacts > max_num_impacts``. That cannot raise
        inside a trace, so ``n_impacts`` is returned unclipped -- check
        ``(batch["n_impacts"] > max_num_impacts).any()`` afterwards, and size the
        array with ``suggest_max_num_impacts``.

        Parameters
        ----------
        max_num_impacts : int
            Fixed output length. Static: changing it retraces.
        rate : float, optional
            Override the Poisson rate. Default is ``expected_impacts()``, computed
            once here rather than per key.

        Returns
        -------
        callable
            ``key -> dict`` of arrays of length ``max_num_impacts``, plus scalar
            ``n_impacts`` and ``rate``.
        """
        from streamsculptor.tnfw import _tidally_evolved_nfw_params

        max_num_impacts = int(max_num_impacts)
        rate = float(self.expected_impacts() if rate is None else rate)

        pop = self.population
        b_max_pop = np.asarray(self.cross_section.b_max(pop.m_bound), dtype=float)

        # p(halo i) ~ w_i b_max(M_i). Key-independent, so the CDF is built once and
        # the draw inside the trace is a searchsorted rather than a full choice().
        w = np.asarray(pop.weights, dtype=float) * b_max_pop
        cdf = jnp.asarray(np.cumsum(w) / w.sum())

        m_infall = jnp.asarray(pop.m_infall)
        c_infall = jnp.asarray(pop.c_infall)
        z_infall = jnp.asarray(pop.z_infall)
        t_since_infall = jnp.asarray(pop.t_since_infall)
        f_bound = jnp.asarray(pop.f_bound)
        m_bound = jnp.asarray(pop.m_bound)
        b_max_arr = jnp.asarray(b_max_pop)
        n_pop = len(pop)

        @eqx.filter_jit
        def sample(key):
            key_n, key_idx = jax.random.split(key)
            n_impacts = jax.random.poisson(key_n, rate)

            u = jax.random.uniform(key_idx, (max_num_impacts,))
            idx = jnp.clip(jnp.searchsorted(cdf, u), 0, n_pop - 1)
            valid = jnp.arange(max_num_impacts) < n_impacts

            # Gather real halos everywhere -- padded slots borrow halo idx[0] -- so the
            # tidal track never sees a zero mass. The masking happens afterwards.
            m_inf = m_infall[idx]
            c_inf = c_infall[idx]
            z_inf = z_infall[idx]
            f_b = f_bound[idx]

            rhos, rs, ft, rt = _tidally_evolved_nfw_params(m_inf, c_inf, z_inf, f_b)

            keep = lambda x, pad: jnp.where(valid, x, pad)
            return dict(
                m_bound=keep(m_bound[idx], 0.0),
                m_infall=keep(m_inf, 0.0),
                c_infall=keep(c_inf, 0.0),
                z_infall=keep(z_inf, 0.0),
                t_since_infall=keep(t_since_infall[idx], 0.0),
                f_bound=keep(f_b, 0.0),
                rhos=keep(rhos, 0.0),
                rs=keep(rs, 1.0),
                ft=keep(ft, 1.0),
                rt=keep(rt, 1.0),
                b_max=keep(b_max_arr[idx], 0.0),
                valid=valid,
                n_impacts=n_impacts,
                rate=jnp.asarray(rate),
            )

        return sample

    def with_model(self, model):
        """
        The same stream and cross-section under a different dark matter model. Reuses
        the cached pyHalo draw when the slope is unchanged.
        """
        out = object.__new__(ImpactRates)
        out.population = self.population.with_model(model)
        out.cross_section = self.cross_section
        out.orbit_ts_signed = self.orbit_ts_signed
        out.engine = self.engine
        out.linear_growth = self.linear_growth
        return out
