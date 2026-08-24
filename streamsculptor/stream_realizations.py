"""
Fast SHMF realizations of a linearly-perturbed stream.

Given a *pre-computed* library of stream response derivatives -- the output of
``generate_derivs.get_derivs`` (Chen+25 base stream) -- this module draws many cheap
Monte-Carlo realizations of the perturbed stream under different draws from the subhalo
mass function (SHMF), without ever re-integrating.

The expensive step (integrating the linear-response fields for a library of ``N_lib``
candidate impacts) is done *once* by ``get_derivs``. Each library slot is a specific
subhalo with its own geometry AND its own fiducial mass/radius baked in: the mass
derivative is taken at unit mass and the structural (radius) derivative at that slot's
own root radius ``r_s_root`` (the mass--concentration value for its build mass). This is
why ``get_derivs`` saves ``r_s_root`` per slot.

A realization then:

  1. draws impact masses from the true SHMF with ``RateCalculator.sample_masses``
     (Poisson count from the orbit-weighted encounter rate; masses from the SHMF),
  2. **greedily matches** each drawn mass to the pool slot whose ``r_s_root`` is closest
     to that mass's (``r_s_fac``-rescaled) scale radius, **without replacement** (each slot
     used at most once),
  3. assigns the drawn mass to the matched slot, with
     ``delta_r = r_s_fac * r_s(m) - r_s_root`` -- small, because the match is close in
     radius (the match tracks ``r_s_fac``), and
  4. sums the linear responses.

Why match instead of reweight? The ``get_derivs`` pool oversamples high masses (proposal
``dN/dM ~ M^-0.3``), so a *uniform* subset would over-represent massive subhalos. Drawing
the masses from the true SHMF and matching to the nearest-radius slot gives a correct SHMF
realization while keeping the response in the linear regime (``delta_r ~ 0``, and the
assigned mass ~ the slot's fiducial mass, so the geometry / impact parameter b -- which
``get_derivs`` scales with mass -- also stays appropriate). It reproduces the standard
"draw SHMF, match to nearest precomputed impact, remove it, repeat" recipe.

Because every slot has a fixed shape and inactive slots carry mass 0, a realization has
static array shapes -> the whole thing ``vmap``s (all realizations at once, GPU-friendly)
or ``scan``s (memory-lean).

Conventions
-----------
* ``stream_derivs`` has shape ``(N_particle, N_lib, 12)``: columns ``0:6`` are
  ``d(stream)/d(mass)`` (at unit mass), columns ``6:12`` are ``d(stream)/d(r_s)`` (the
  structural derivative, at ``r_s = r_s_root``).
* Linear model (the notebook's ``compute_pert_stream``, einsum form):
  ``stream = base + Σ_i [ deriv_mass_i * m_i + deriv_struct_i * m_i * Δr_i ]``.
* ``r_s(m) = r_s_coeff * sqrt(m / 1e8)`` with ``r_s_coeff = 1.05`` to MATCH the
  mass--concentration relation ``get_derivs`` uses to build the pool (1.05). This must
  agree with the pool's relation, otherwise matched ``delta_r`` picks up a constant offset.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_enable_x64", True)

R_S_COEFF = 1.05  # mass--concentration coefficient used by generate_derivs.get_derivs


# -----------------------------------------------------------------------------
# Functional core (stateless helpers; all jittable / vmappable)
# -----------------------------------------------------------------------------

def r_s_from_mass(mass, r_s_coeff=R_S_COEFF):
    """Scale radius [kpc] from mass [Msun]. Must match the pool's relation (get_derivs: 1.05)."""
    return r_s_coeff * jnp.sqrt(mass / 1e8)


def compute_pert_stream(stream_base, stream_derivs, mass_subhalos, delta_rs):
    """
    Linear-response perturbed stream (the notebook's compute_pert_stream, einsum form).

    stream_base    : (N, 6)
    stream_derivs  : (N, N_lib, 12)   cols 0:6 = d/dm, cols 6:12 = d/dr_s
    mass_subhalos  : (N_lib,)         mass of each library subhalo (0 => inactive)
    delta_rs       : (N_lib,)         r_s(mass) - r_s_root for each subhalo

    returns stream_pert : (N, 6)
    """
    dm = stream_derivs[:, :, :6]      # (N, N_lib, 6)
    dr = stream_derivs[:, :, 6:]      # (N, N_lib, 6)
    mass_response = jnp.einsum('nlj,l->nj', dm, mass_subhalos)
    radius_response = jnp.einsum('nlj,l->nj', dr, mass_subhalos * delta_rs)
    return stream_base + mass_response + radius_response


def draw_realization_params(
    RateCalculator,
    key,
    r_s_root,
    log10M_min,
    log10M_max,
    M_hm=0.0,
    normalization=1.0,
    slope=-1.9,
    mass_fac=1.0,
    r_s_fac=1.0,
    r_s_coeff=R_S_COEFF,
):
    """
    Draw one realization's per-slot (mass, delta_r) vectors by SHMF-draw + nearest-radius
    matching (without replacement).

    Steps: RateCalculator.sample_masses gives a Poisson number of SHMF masses (rest padded
    to 0). Each drawn mass is matched greedily to the available pool slot whose r_s_root is
    closest to its RESCALED radius r_s_fac * r_s(mass); that slot is removed from the pool.
    The slot receives the drawn mass and delta_r = r_s_fac * r_s(mass) - r_s_root[slot].

    r_s_fac rescales each subhalo's radius (compact < 1 < diffuse) and drives BOTH the match
    and delta_r: matching to the nearest r_s_root of the rescaled radius reselects a better
    slot as r_s_fac moves away from 1, so delta_r stays small (linear regime) instead of
    growing like (r_s_fac - 1) * r_s(mass).

    Returns a dict with:
      mass       : (n_lib,)  assigned masses [Msun] (0 for unused slots)
      delta_r    : (n_lib,)  r_s(mass) - r_s_root [kpc] (0 for unused slots)
      r_s_root   : (n_lib,)  the pool's per-slot fiducial radii (passthrough)
      active     : (n_lib,)  bool mask of used slots
      n_impact   : ()        int, number of impacts drawn (clipped to n_lib)
    """
    r_s_root = jnp.asarray(r_s_root)
    n_lib = r_s_root.shape[0]

    sample_dict = RateCalculator.sample_masses(
        log10M_min=log10M_min, log10M_max=log10M_max, key=key,
        M_hm=M_hm, normalization=normalization, slope=slope, array_length=n_lib)

    log10_mass = sample_dict['log10_mass']                # (n_lib,), padding == 0
    tgt_active = log10_mass > 0
    drawn_mass = jnp.where(tgt_active, 10.0 ** log10_mass, 0.0)                  # physical SHMF draw
    tgt_rs_nom = jnp.where(tgt_active, r_s_from_mass(drawn_mass, r_s_coeff), 0.0)  # nominal radius r_s(mass)
    tgt_rs = tgt_rs_nom * r_s_fac                                               # rescaled radius; drives match + delta_r
    assign_mass = drawn_mass * mass_fac                                         # mass used in the response

    # r_s_fac drives the radius matching: each mass matches to the slot nearest its RESCALED radius
    # tgt_rs = r_s_fac * r_s(mass), so a compact/diffuse population reselects better-radius slots and
    # delta_r stays small. mass_fac still scales ONLY the assigned mass (-> linear response), never the
    # matching (which is on the physical drawn mass's radius) -- so mass_fac=10 gives ~10x the
    # perturbation instead of re-matching to weaker-coupling slots.

    # Greedy nearest-r_s matching, without replacement (scan over the drawn targets).
    def match_step(carry, t):
        used, mass_out, dr_out = carry
        m_assign, rs_match, rs_assign, act = assign_mass[t], tgt_rs[t], tgt_rs[t], tgt_active[t]
        dist = jnp.where(used, jnp.inf, jnp.abs(r_s_root - rs_match))
        j = jnp.argmin(dist)                              # nearest available slot (by rescaled radius r_s_fac*r_s(m))
        do = act & jnp.logical_not(used[j])               # skip padded targets / exhausted pool
        mass_out = mass_out.at[j].set(jnp.where(do, m_assign, mass_out[j]))
        dr_out = dr_out.at[j].set(jnp.where(do, rs_assign - r_s_root[j], dr_out[j]))
        used = used.at[j].set(jnp.logical_or(used[j], do))
        return (used, mass_out, dr_out), None

    init = (jnp.zeros(n_lib, dtype=bool), jnp.zeros(n_lib), jnp.zeros(n_lib))
    (used, mass_out, dr_out), _ = jax.lax.scan(match_step, init, jnp.arange(n_lib))

    return dict(mass=mass_out, delta_r=dr_out, r_s_root=r_s_root,
                active=used, n_impact=jnp.minimum(sample_dict['N_encounter'], n_lib))


# -----------------------------------------------------------------------------
# StreamRealizationGenerator: build once from a deriv library, then generate fast
# -----------------------------------------------------------------------------

class StreamRealizationGenerator(eqx.Module):
    """
    Generate fast vmapped/scanned SHMF realizations of a linearly-perturbed stream from a
    pre-computed response-derivative library (generate_derivs.get_derivs) + a RateCalculator.

    Build the library once, then:

        from streamsculptor.generate_derivs import get_derivs
        batches = get_derivs(..., save=False)          # list of per-batch dicts
        gen = StreamRealizationGenerator.from_get_derivs(
                  batches, RateCalculator=RC, log10M_min=6.0, log10M_max=8.0)
        out = gen.generate(jax.random.PRNGKey(0), n_realizations=512, method='vmap')
        streams = out                                  # (n_realizations, N_particle, 6)

    Parameters
    ----------
    stream_base   : (N, 6) base (unperturbed) stream.
    stream_derivs : (N, N_lib, 12) response derivatives (cols 0:6 = d/dm, 6:12 = d/dr_s).
    r_s_root      : (N_lib,) each slot's fiducial scale radius (get_derivs `r_s_root`).
    RateCalculator: a subhalostatistics.RateCalculator; supplies the impact count and SHMF.
    log10M_min/max: SHMF mass window [log10 Msun].
    M_hm, normalization, slope : SHMF knobs passed to RateCalculator.sample_masses.
    mass_fac      : multiplicative knob on the drawn masses.
    r_s_fac       : rescales each subhalo's radius (compact < 1 < diffuse); drives the
                    nearest-r_s_root match (so it reselects slots, not just inflates delta_r).
                    The default for generate/realize, overridable per call.
    r_s_coeff     : mass--concentration coefficient; must match the pool's (get_derivs: 1.05).
    """
    stream_base: jnp.ndarray
    stream_derivs: jnp.ndarray
    r_s_root: jnp.ndarray
    RateCalculator: eqx.Module
    n_lib: int = eqx.field(static=True)

    log10M_min: float
    log10M_max: float
    M_hm: float
    normalization: float
    slope: float
    mass_fac: float
    r_s_fac: float
    r_s_coeff: float

    def __init__(self, stream_base, stream_derivs, r_s_root, RateCalculator,
                 log10M_min=6.0, log10M_max=8.0, M_hm=0.0, normalization=1.0,
                 slope=-1.9, mass_fac=1.0, r_s_fac=1.0, r_s_coeff=R_S_COEFF):
        self.stream_base = jnp.asarray(stream_base)
        self.stream_derivs = jnp.asarray(stream_derivs)
        self.r_s_root = jnp.asarray(r_s_root)
        self.RateCalculator = RateCalculator
        self.n_lib = int(self.stream_derivs.shape[1])
        self.log10M_min = float(log10M_min)
        self.log10M_max = float(log10M_max)
        self.M_hm = float(M_hm)
        self.normalization = float(normalization)
        self.slope = float(slope)
        self.mass_fac = float(mass_fac)
        self.r_s_fac = float(r_s_fac)
        self.r_s_coeff = float(r_s_coeff)

    @classmethod
    def from_get_derivs(cls, batches, RateCalculator, **kwargs):
        """
        Build from generate_derivs.get_derivs(save=False) output (a list of per-batch dicts,
        or a single dict). Each dict has pert_out = [stream_base (N,6), stream_derivs
        (N, n_sub, 12)] and r_s_root (n_sub,). Batches share the same base stream; their
        subhalo pools are concatenated along the slot axis.
        """
        if isinstance(batches, dict):
            batches = [batches]
        base = jnp.asarray(batches[0]['pert_out'][0])
        derivs = jnp.concatenate([jnp.asarray(b['pert_out'][1]) for b in batches], axis=1)
        r_s_root = jnp.concatenate([jnp.asarray(b['r_s_root']) for b in batches], axis=0)
        return cls(base, derivs, r_s_root, RateCalculator, **kwargs)

    @eqx.filter_jit
    def draw_params(self, key, r_s_fac=None):
        """Draw one realization's (mass, delta_r, ...). See draw_realization_params.
        r_s_fac overrides the stored default (compact < 1 < diffuse)."""
        r_s_fac = self.r_s_fac if r_s_fac is None else r_s_fac
        return draw_realization_params(
            self.RateCalculator, key, self.r_s_root, self.log10M_min, self.log10M_max,
            M_hm=self.M_hm, normalization=self.normalization, slope=self.slope,
            mass_fac=self.mass_fac, r_s_fac=r_s_fac, r_s_coeff=self.r_s_coeff)

    @eqx.filter_jit
    def realize(self, key, return_params=False, r_s_fac=None):
        """
        One realization. Returns the perturbed stream (N, 6), or a dict including the
        drawn parameters if return_params=True. r_s_fac overrides the stored radius scaling.
        """
        p = self.draw_params(key, r_s_fac=r_s_fac)
        stream = compute_pert_stream(self.stream_base, self.stream_derivs,
                                     p['mass'], p['delta_r'])
        if return_params:
            return dict(stream=stream, **p)
        return stream

    @eqx.filter_jit
    def generate(self, key, n_realizations, method='vmap', return_params=False, r_s_fac=None):
        """
        Generate n_realizations perturbed streams.

        method : 'vmap' (all realizations at once; GPU-friendly, higher memory) or
                 'scan' (sequential; memory-lean).
        return_params : also return per-realization (mass, delta_r, r_s_root, active,
                 n_impact) draws.
        r_s_fac : override the stored radius scaling (compact < 1 < diffuse); pass a float to
                 sweep it (recompiles per distinct value) or jnp.asarray(val) to avoid recompiles.

        Returns (n_realizations, N, 6), or a dict of stacked arrays if return_params=True.
        """
        keys = jax.random.split(key, n_realizations)
        one = lambda k: self.realize(k, return_params=return_params, r_s_fac=r_s_fac)

        if method == 'vmap':
            return jax.vmap(one)(keys)
        elif method == 'scan':
            def step(carry, k):
                return carry, one(k)
            _, out = jax.lax.scan(step, None, keys)
            return out
        else:
            raise ValueError(f"method must be 'vmap' or 'scan', got {method!r}")
