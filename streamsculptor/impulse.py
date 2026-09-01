"""
Analytic impulsive subhalo kicks for stellar streams.

This module provides the *impulse approximation* to a subhalo flyby: instead of
integrating a live, time-windowed subhalo potential through the encounter (see
``SubhaloLinePotential``), each star receives an instantaneous velocity kick equal
to the closed-form time-integral of the subhalo's acceleration along the
unperturbed (straight-line) encounter. For a Plummer subhalo this is the standard
result of Erkal & Belokurov (2015) and Sanders, Bovy & Erkal (2016).

Three layers:
  1. Kernels  -- pure, coordinate-free velocity-kick expressions
     (``plummer_impulse``, ``plummer_impulse_from_A``, ``plummer_kick_perp``).
  2. Geometry -- ``impact_frame`` builds the (T,N,B) frame + subhalo location at
     the impact time from a stream "patch" phase-space point (the same frame
     convention as ``GenerateImpactParams.ImpactGenerator``).
  3. Application -- ``apply_impulse_kicks`` flows spray particles through a set of
     kicks (applied chronologically), and ``gen_stream_impulse_Chen25`` wraps it
     into a Chen+25 stream generator that returns present-day (lead, trail).

Everything is JAX/diffrax-native and differentiable in the kick parameters
(masses, scale radii, impact geometry) and in the progenitor IC. As with the
other ``gen_stream_*`` paths, use FORWARD-mode autodiff (``jax.jacfwd``, or an
optimizer like optimistix / a Gauss-Newton loop) -- the adaptive integrator's
``while_loop`` is not reverse-mode (``jax.grad``) differentiable.
"""

import jax
import jax.numpy as jnp
import diffrax

from streamsculptor.main import usys
from streamsculptor.streamhelpers import gen_stream_ics_Chen25

jax.config.update("jax_enable_x64", True)

__all__ = [
    "plummer_impulse",
    "plummer_impulse_from_A",
    "plummer_kick_perp",
    "impact_frame",
    "apply_impulse_kicks",
    "perp_kick_stream",
    "perp_kick_stream_curved",
    "gen_stream_impulse_Chen25",
]


# ---------------------------------------------------------------------------
# 1. Kernels (pure, vmappable over the star axis)
# ---------------------------------------------------------------------------
def plummer_impulse(r, subhalo_x, w_vec, M, r_s, G=usys.G):
    """Velocity kick from a straight-line Plummer flyby (impulse approximation).

    Coordinate-free and valid for *any* encounter angle -- the star's separation
    is projected onto the plane perpendicular to the subhalo's line of motion, so
    no stream frame is assumed:

        dv = (2 G M / |w|) * r_perp / (|r_perp|^2 + r_s^2)
        r_perp = (r - subhalo_x) - ((r - subhalo_x) . w_hat) w_hat

    Parameters
    ----------
    r : (3,) array
        Star position at the impact time [kpc].
    subhalo_x : (3,) array
        Any point on the subhalo's trajectory at the impact time [kpc]
        (the parallel component is projected out, so the exact point is irrelevant).
    w_vec : (3,) array
        Relative velocity of the subhalo w.r.t. the star [kpc/Myr].
    M : float
        Subhalo mass [Msun].
    r_s : float
        Plummer scale radius [kpc].
    G : float
        Gravitational constant (default: galactic units, kpc^3 / (Msun Myr^2)).

    Returns
    -------
    dv : (3,) array
        Velocity kick [kpc/Myr].
    """
    w_norm = jnp.linalg.norm(w_vec)
    w_hat = w_vec / w_norm
    dr = r - subhalo_x
    r_perp = dr - jnp.dot(dr, w_hat) * w_hat
    return (2.0 * G * M / w_norm) * r_perp / (jnp.dot(r_perp, r_perp) + r_s**2)


def plummer_impulse_from_A(r, subhalo_x, w_hat, A, r_s):
    """Same kick as ``plummer_impulse`` but parametrized by the amplitude A = 2GM/|w|.

    Only ``A`` (i.e. M/|w|) and ``r_s`` are identifiable from a stream's morphology,
    so this is the natural form for inference: fit ``A`` directly and convert to a
    mass afterwards via M = A |w| / (2 G).

    Parameters
    ----------
    r, subhalo_x : (3,) arrays
        Star and subhalo positions at the impact time [kpc].
    w_hat : (3,) array
        Unit vector along the subhalo's direction of motion.
    A : float
        Kick amplitude 2 G M / |w| [kpc/Myr].
    r_s : float
        Plummer scale radius [kpc].
    """
    dr = r - subhalo_x
    r_perp = dr - jnp.dot(dr, w_hat) * w_hat
    return A * r_perp / (jnp.dot(r_perp, r_perp) + r_s**2)


def plummer_kick_perp(r, x0, T, N, B, A, s0, bN, bB, r_s):
    """Perpendicular-crossing Plummer impulse expressed in a stream (T,N,B) frame.

    The special case in which the subhalo velocity is perpendicular to the stream,
    written with the star's along-track coordinate ``s`` measured along the stream
    tangent T rather than along the subhalo path:

        s  = (r - x0) . T
        p  = (s - s0) T + bN N + bB B
        dv = A p / [ (s - s0)^2 + bN^2 + bB^2 + r_s^2 ]

    This is the free-form kick used for spur/gap fitting: ``A`` sets the strength
    (= 2GM/|w|), ``s0`` slides the kick along the track, and ``bN, bB`` are the
    perpendicular impact offsets. Fit ``(A, s0, bN, bB, r_s)`` per kick.

    Parameters
    ----------
    r : (3,) array
        Star position at the impact time [kpc].
    x0, T, N, B : (3,) arrays
        Impact-point position and orthonormal (tangent, normal, binormal) frame,
        e.g. from :func:`impact_frame`.
    A : float
        Kick amplitude 2 G M / |w| [kpc/Myr].
    s0, bN, bB, r_s : float
        Along-track center, perpendicular impact offsets, and scale radius [kpc].
    """
    s = jnp.dot(r - x0, T)
    p = (s - s0) * T + bN * N + bB * B
    D = (s - s0) ** 2 + bN**2 + bB**2 + r_s**2
    return A * p / D


# ---------------------------------------------------------------------------
# 2. Geometry
# ---------------------------------------------------------------------------
def impact_frame(pot, w_patch, t_impact, t0=0.0,
                 solver=diffrax.Dopri8(), rtol=1e-7, atol=1e-7, dtmin=0.1, max_steps=10_000):
    """Build the (T,N,B) impact frame at ``t_impact`` from a stream patch.

    Integrates the patch phase-space point ``w_patch`` (given at time ``t0``, e.g.
    the mean phase-space coordinate of a phi1 band today) back to ``t_impact`` and
    returns the position there plus the orthonormal frame

        T = v_hat                 (stream tangent / direction of motion)
        B = (x cross v) hat       (orbital angular-momentum direction)
        N = B cross T             (in-plane normal)

    matching the convention in ``GenerateImpactParams.ImpactGenerator``.

    Returns
    -------
    dict with keys ``x`` (3,), ``v`` (3,), ``T`` (3,), ``N`` (3,), ``B`` (3,).
    """
    W0 = pot.integrate_orbit(w0=w_patch, ts=jnp.array([t_impact]), t0=t0, t1=t_impact,
                             solver=solver, rtol=rtol, atol=atol, dtmin=dtmin,
                             max_steps=max_steps).ys[0]
    T = W0[3:] / jnp.linalg.norm(W0[3:])
    B = jnp.cross(W0[:3], W0[3:]); B = B / jnp.linalg.norm(B)
    N = jnp.cross(B, T)
    return {"x": W0[:3], "v": W0[3:], "T": T, "N": N, "B": B}


# ---------------------------------------------------------------------------
# 3. Application to a spray of particles
# ---------------------------------------------------------------------------
def apply_impulse_kicks(pot, Y0, T0, t_final, t_impact, kick_fn,
                        solver=diffrax.Dopri8(), rtol=1e-7, atol=1e-7, dtmin=0.3, max_steps=10_000):
    """Flow spray particles through a set of impulsive kicks, then to ``t_final``.

    Kicks are applied in chronological order. A particle released after a given
    kick time is skipped for that kick (the intervening flow is a zero-length no-op
    and the kick is masked to zero), so heterogeneous release times are handled
    correctly.

    Parameters
    ----------
    pot : Potential
        Base potential the particles move in between kicks.
    Y0 : (Npart, 6) array
        Particle phase-space ICs [kpc, kpc/Myr].
    T0 : (Npart,) array
        Release time of each particle [Myr].
    t_final : float
        Observation time [Myr].
    t_impact : (Nkick,) array
        Impact time of each kick [Myr].
    kick_fn : callable
        ``kick_fn(r, k) -> dv`` giving the velocity kick for a star at position
        ``r`` (3,) from kick index ``k`` (an integer into the ORIGINAL, unsorted
        ``t_impact`` / parameter arrays). Build it from one of the kernels above,
        e.g. ``lambda r, k: plummer_impulse(r, x0[k], w_vec[k], M[k], r_s[k], G)``.
    solver, rtol, atol, dtmin, max_steps :
        Passed to ``pot.integrate_orbit`` for the between-kick flow.

    Returns
    -------
    (Npart, 6) array of present-day phase-space states.
    """
    order = jnp.argsort(t_impact)      # apply chronologically
    t_sorted = t_impact[order]
    Nk = t_impact.shape[0]

    def _flow(w, ta, tb):
        return pot.integrate_orbit(w0=w, ts=jnp.array([tb]), t0=ta, t1=tb, solver=solver,
                                   rtol=rtol, atol=atol, dtmin=dtmin, max_steps=max_steps).ys[-1]

    def one(w0, t_rel):
        w, t = w0, t_rel
        for j in range(Nk):
            k = order[j]                                 # original (unsorted) kick index
            t_next = jnp.maximum(t_sorted[j], t_rel)     # no-op flow if star not yet stripped
            w = _flow(w, t, t_next)
            dv = kick_fn(w[:3], k)
            w = w.at[3:].add(dv * (t_sorted[j] > t_rel)) # mask kick if star not yet stripped
            t = t_next
        return _flow(w, t, t_final)

    return jax.vmap(one)(Y0, T0)


def perp_kick_stream(pot, Y0, T0, t_final, params, t_impacts, w_anchors,
                     frame_solver=diffrax.Dopri8(), frame_kw=None, **flow_kw):
    """Apply a sum of perpendicular-crossing Plummer kicks parametrized in stream frames.

    The inference-friendly counterpart to :func:`gen_stream_impulse_Chen25`: instead
    of physical ``(M, w_vec)`` per subhalo, each kick is parametrized directly in the
    ``(T, N, B)`` frame at its impact point (see :func:`plummer_kick_perp`), which is
    the well-conditioned form for gradient-fitting spurs/gaps.

    Parameters
    ----------
    pot : Potential
        Base potential the particles move in between kicks.
    Y0 : (Npart, 6) array
        Particle phase-space ICs [kpc, kpc/Myr].
    T0 : (Npart,) array
        Release time of each particle [Myr].
    t_final : float
        Observation time [Myr].
    params : (Nkick, 5) array
        Per-kick ``[A, s0, b_N, b_B, r_s]``: amplitude A = 2GM/|w| [kpc/Myr], along-track
        center, perpendicular impact offsets, and Plummer scale radius [kpc].
    t_impacts : (Nkick,) array
        Impact times [Myr].
    w_anchors : (Nkick, 6) array
        Phase-space point (at ``t0=0``) anchoring each kick's frame -- e.g. the mean
        phase-space coordinate of the stream at the crossing point. Integrated back to
        the impact time by :func:`impact_frame`.
    frame_solver, frame_kw :
        Solver / kwargs for the frame integration (``frame_kw`` defaults to
        ``dict(atol=1e-7, rtol=1e-7, dtmin=0.1)``).
    **flow_kw :
        Passed to :func:`apply_impulse_kicks` for the between-kick flow
        (``solver``, ``rtol``, ``atol``, ``dtmin``, ``max_steps``).

    Returns
    -------
    (Npart, 6) array of present-day phase-space states.
    """
    if frame_kw is None:
        frame_kw = dict(atol=1e-7, rtol=1e-7, dtmin=0.1)
    frames = [impact_frame(pot, w_anchors[k], t_impacts[k], t0=0.0, solver=frame_solver, **frame_kw)
              for k in range(t_impacts.shape[0])]
    x0 = jnp.stack([f["x"] for f in frames]); T = jnp.stack([f["T"] for f in frames])
    N  = jnp.stack([f["N"] for f in frames]); B = jnp.stack([f["B"] for f in frames])

    def kick_fn(r, k):                                   # k = original (unsorted) kick index
        A, s0, bN, bB, rs = params[k]
        return plummer_kick_perp(r, x0[k], T[k], N[k], B[k], A, s0, bN, bB, rs)

    return apply_impulse_kicks(pot, Y0, T0, t_final, t_impacts, kick_fn, **flow_kw)


def perp_kick_stream_curved(pot, Y0, T0, t_final, params, t_impacts, w_anchors,
                            frame_solver=diffrax.Dopri8(), frame_kw=None, **flow_kw):
    """Curvature-aware counterpart of :func:`perp_kick_stream`.

    Identical API and per-kick parametrization ``[A, s0, b_N, b_B, r_s]``, but the kick
    uses the exact projected Plummer kernel (:func:`plummer_impulse_from_A`) evaluated at
    each star's *true* position, rather than :func:`plummer_kick_perp`'s straight-line
    reconstruction from the along-track coordinate alone. The stream frame is used only to
    place the subhalo's trajectory from the fit parameters; the star offset that enters the
    kick is the real 3-D perpendicular separation, so the stream's curvature and finite
    width are retained.

    Given the impact frame ``(x0, T, N, B)`` at each crossing, the fit parameters map to a
    physical fly-by as

        x_sub = x0 + s0 * T - b_N * N - b_B * B      # closest-approach point on the path
        w_hat = (-b_B * N + b_N * B) / b,   b = sqrt(b_N^2 + b_B^2)   # perpendicular crossing

    For a star lying exactly on the ``T`` line this reduces to :func:`perp_kick_stream`
    term-for-term; off the line the two differ by exactly the curvature/width effect.

    Parameters
    ----------
    Identical to :func:`perp_kick_stream` (see there). ``params`` is ``(Nkick, 5)`` of
    ``[A, s0, b_N, b_B, r_s]``, ``w_anchors`` is ``(Nkick, 6)``.

    Returns
    -------
    (Npart, 6) array of present-day phase-space states.
    """
    if frame_kw is None:
        frame_kw = dict(atol=1e-7, rtol=1e-7, dtmin=0.1)
    frames = [impact_frame(pot, w_anchors[k], t_impacts[k], t0=0.0, solver=frame_solver, **frame_kw)
              for k in range(t_impacts.shape[0])]
    x0 = jnp.stack([f["x"] for f in frames]); T = jnp.stack([f["T"] for f in frames])
    N  = jnp.stack([f["N"] for f in frames]); B = jnp.stack([f["B"] for f in frames])

    def kick_fn(r, k):                                   # k = original (unsorted) kick index
        A, s0, bN, bB, rs = params[k]
        b         = jnp.sqrt(bN**2 + bB**2) + 1e-12
        subhalo_x = x0[k] + s0 * T[k] - bN * N[k] - bB * B[k]   # a point on the subhalo path
        w_hat     = (-bB * N[k] + bN * B[k]) / b                # perpendicular crossing (N-B plane)
        return plummer_impulse_from_A(r, subhalo_x, w_hat, A, rs)

    return apply_impulse_kicks(pot, Y0, T0, t_final, t_impacts, kick_fn, **flow_kw)


def gen_stream_impulse_Chen25(pot_base, ts, prog_w0, Msat, key,
                              subhalo_x, w_vec, M, r_s, t_impact,
                              solver=diffrax.Dopri8(), rtol=1e-7, atol=1e-7, dtmin=0.3,
                              max_steps=10_000):
    """Chen+25 spray stream perturbed by analytic impulsive Plummer kicks.

    Generates the unperturbed spray ICs, then applies one ``plummer_impulse`` per
    subhalo at its impact time and integrates to the present. Forward-mode
    differentiable in the kick parameters (``M``, ``r_s``, ``subhalo_x``, ``w_vec``)
    and in ``prog_w0`` (use ``jax.jacfwd`` / optimistix, not ``jax.grad``).

    Parameters
    ----------
    pot_base : Potential
        Host potential.
    ts : (Nrelease,) array
        Stripping times; ``ts[-1]`` is the observation time (Chen+25 convention).
    prog_w0 : (6,) array
        Progenitor phase-space IC at ``ts[0]``.
    Msat : float
        Progenitor mass used to size the spray [Msun].
    key : PRNGKey
        Spray RNG key.
    subhalo_x : (Nkick, 3) array
        Subhalo position at each impact time [kpc].
    w_vec : (Nkick, 3) array
        Subhalo velocity relative to the stream at each impact [kpc/Myr].
    M : (Nkick,) array
        Subhalo masses [Msun].
    r_s : (Nkick,) array
        Plummer scale radii [kpc].
    t_impact : (Nkick,) array
        Impact times [Myr].

    Returns
    -------
    (lead, trail) : each (Npart, 6), matching ``gen_stream_vmapped_Chen25``.
    """
    stream_ics, _ = gen_stream_ics_Chen25(pot_base=pot_base, ts=ts, prog_w0=prog_w0,
                                          Msat=Msat, key=key, solver=solver, rtol=rtol,
                                          atol=atol, dtmin=dtmin, max_steps=max_steps)
    pos_close, pos_far, vel_close, vel_far = stream_ics
    n = pos_close.shape[0] - 1
    Y0 = jnp.concatenate([jnp.hstack([pos_close[:-1], vel_close[:-1]]),
                          jnp.hstack([pos_far[:-1],  vel_far[:-1]])], 0)
    T0 = jnp.concatenate([ts[:-1], ts[:-1]])

    G = pot_base.units.G
    def kick_fn(r, k):
        return plummer_impulse(r, subhalo_x[k], w_vec[k], M[k], r_s[k], G)

    out = apply_impulse_kicks(pot_base, Y0, T0, ts[-1], t_impact, kick_fn,
                              solver=solver, rtol=rtol, atol=atol, dtmin=dtmin, max_steps=max_steps)
    return out[:n], out[n:]
