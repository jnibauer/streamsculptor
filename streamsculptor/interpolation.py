# streamsculptor/interpolation.py
import jax
import jax.numpy as jnp
from functools import partial


# ---------------------------------------------------------------------------
# Shared: JIT-compiled coefficient computation for C² spline
# ---------------------------------------------------------------------------
# Defined at module level so @jax.jit compiles it once and caches globally.
# For a uniform grid the tridiagonal system has constant structure:
#   diag = 4, lower = upper = 1  (everywhere)
# so we bake those into the scan body — no array indexing needed.

@jax.jit
def _compute_c2_coeffs(y, dt):
    """
    Natural cubic spline coefficients via Thomas algorithm.

    Uses jax.lax.scan for both the forward sweep and back-substitution,
    compiling each into a single XLA while-loop rather than N Python dispatches.

    Parameters
    ----------
    y  : (N, D)  function values
    dt : scalar  uniform spacing

    Returns
    -------
    a, b, c, d : each (N-1, D)  cubic polynomial coefficients per segment
    """
    N, D = y.shape
    h = dt
    n = N - 2  # number of interior unknowns

    # Right-hand side (vectorised, no loop)
    r = (6.0 / (h * h)) * (y[:-2] - 2.0 * y[1:-1] + y[2:])  # (n, D)

    # ------------------------------------------------------------------
    # Forward sweep  (serial dependency: each row needs the previous)
    # ------------------------------------------------------------------
    # For uniform grid: diag=4, lower=upper=1 everywhere.
    # Row 0 is handled as the initial carry so the scan starts at row 1.
    c0 = 1.0 / 4.0                # scalar, same for every row
    d0 = r[0] / 4.0               # (D,)

    def _fwd_step(carry, r_i):
        # carry = (c_prev,  d_prev)   shapes: (scalar, (D,))
        c_prev, d_prev = carry
        denom  = 4.0 - c_prev       # lower[i-1]*c_prev = 1*c_prev
        c_i    = 1.0 / denom
        d_i    = (r_i - d_prev) / denom   # lower[i-1]*d_prev = 1*d_prev
        return (c_i, d_i), (c_i, d_i)

    # Scan over rows 1 .. n-1  (n-1 steps)
    _, (c_rest, d_rest) = jax.lax.scan(_fwd_step, (c0, d0), r[1:])

    # Assemble full arrays
    # c_prime has n-1 entries (used only in back-sub); last row has no c_prime
    c_prime = jnp.concatenate([jnp.array([c0]), c_rest[:-1]])  # (n-1,)
    d_prime = jnp.concatenate([d0[None], d_rest], axis=0)      # (n, D)

    # ------------------------------------------------------------------
    # Back-substitution  (serial, reversed)
    # ------------------------------------------------------------------
    # x[n-1] = d_prime[n-1]   (no c_prime entry for the last row)
    # x[i]   = d_prime[i] - c_prime[i] * x[i+1]   for i = n-2 .. 0

    def _back_step(x_next, inputs):
        c_i, d_i = inputs          # c_i: scalar, d_i: (D,)
        x_i = d_i - c_i * x_next   # (D,)
        return x_i, x_i

    x_last = d_prime[-1]           # (D,)  — the known last value

    # Feed reversed (c_prime, d_prime[:-1]) so scan goes from n-2 down to 0
    _, x_body_rev = jax.lax.scan(
        _back_step,
        x_last,
        (c_prime[::-1], d_prime[:-1][::-1])   # both length n-1
    )
    x_body = x_body_rev[::-1]     # un-reverse  → (n-1, D)

    # Full solution vector including boundary zeros
    x = jnp.concatenate([x_body, x_last[None]], axis=0)  # (n, D)
    M = jnp.concatenate([
        jnp.zeros((1, D)),
        x,
        jnp.zeros((1, D))
    ], axis=0)                                            # (N, D)

    # ------------------------------------------------------------------
    # Assemble polynomial coefficients per segment
    # ------------------------------------------------------------------
    a = y[:-1]
    b = (y[1:] - y[:-1]) / h - h * (2.0 * M[:-1] + M[1:]) / 6.0
    c = M[:-1] / 2.0
    d = (M[1:] - M[:-1]) / (6.0 * h)

    return a, b, c, d


# ---------------------------------------------------------------------------
# Shared: vectorised eval kernels (no vmap — native array ops)
# ---------------------------------------------------------------------------
# Written as standalone jitted functions so they compile once and are
# reusable. They operate on *batches* of query points natively.
# Each interpolation coordinate type (linear / log) has its own kernel
# so the branch is resolved at __call__ time, not inside a traced function.

# ---- Catmull-Rom kernels ------------------------------------------------

@jax.jit
def _eval_catmull_rom(t_queries, t_min, dt, n_points, y):
    """
    Evaluate Catmull-Rom (C¹) interpolator at an array of query times on a linear-uniform grid.

    All operations vectorise over t_queries with no explicit loop or vmap.
    """
    idx_f = (t_queries - t_min) / dt
    idx0  = jnp.floor(idx_f).astype(jnp.int32)
    idx0  = jnp.clip(idx0, 1, n_points - 3)   # need idx0-1 .. idx0+2

    s = idx_f - idx0                           # fractional part, (Q,)

    # Gather 4 neighbours — fancy indexing broadcasts over the batch dim
    y_m1 = y[idx0 - 1]                         # (Q, D)
    y0   = y[idx0]
    y1   = y[idx0 + 1]
    y2   = y[idx0 + 2]

    # Catmull-Rom tangents
    m0 = (y1 - y_m1) * 0.5
    m1 = (y2 - y0)   * 0.5

    # Hermite basis  — broadcast s to (Q, 1) for the (Q, D) multiply
    s  = s[:, None]
    s2 = s * s
    s3 = s2 * s

    h00 =  2.0 * s3 - 3.0 * s2 + 1.0
    h10 =        s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 =        s3 -       s2

    return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1   # (Q, D)

@jax.jit
def _eval_catmull_rom_log(t_queries, log_t_min, d_log_t, n_points, y):
    """Catmull-Rom eval on a log-uniform grid.  Same math, log index lookup."""
    idx_f = (jnp.log(t_queries) - log_t_min) / d_log_t
    idx0  = jnp.floor(idx_f).astype(jnp.int32)
    idx0  = jnp.clip(idx0, 1, n_points - 3)

    s = idx_f - idx0

    y_m1 = y[idx0 - 1]
    y0   = y[idx0]
    y1   = y[idx0 + 1]
    y2   = y[idx0 + 2]

    m0 = (y1 - y_m1) * 0.5
    m1 = (y2 - y0)   * 0.5

    s  = s[:, None]
    s2 = s * s
    s3 = s2 * s

    h00 =  2.0 * s3 - 3.0 * s2 + 1.0
    h10 =        s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 =        s3 -       s2

    return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1

# ---- C² spline kernels --------------------------------------------------

@jax.jit
def _eval_c2_spline(t_queries, t_min, dt, a, b, c, d):
    """
    Evaluate precomputed C² cubic spline at an array of query times.

    All operations vectorise over t_queries with no explicit loop or vmap.
    """
    idx_f  = (t_queries - t_min) / dt
    idx0   = jnp.floor(idx_f).astype(jnp.int32)
    idx0   = jnp.clip(idx0, 0, a.shape[0] - 1)

    delta  = t_queries - (t_min + idx0 * dt)   # (Q,)

    # Gather coefficients for each query's segment
    ai = a[idx0]                                # (Q, D)
    bi = b[idx0]
    ci = c[idx0]
    di = d[idx0]

    # Horner's method: a + t*(b + t*(c + t*d))  — fewer multiplies
    delta = delta[:, None]                      # (Q, 1) for broadcast
    return ai + delta * (bi + delta * (ci + delta * di))   # (Q, D)

@jax.jit
def _eval_c2_spline_log(t_queries, log_t_min, d_log_t, a, b, c, d):
    """C² spline eval on a log-uniform grid.  Same math, log index lookup."""
    log_t  = jnp.log(t_queries)
    idx_f  = (log_t - log_t_min) / d_log_t
    idx0   = jnp.floor(idx_f).astype(jnp.int32)
    idx0   = jnp.clip(idx0, 0, a.shape[0] - 1)

    # delta is in log-space — that is the coordinate the spline lives in
    delta  = log_t - (log_t_min + idx0 * d_log_t)

    ai = a[idx0]
    bi = b[idx0]
    ci = c[idx0]
    di = d[idx0]

    delta = delta[:, None]
    return ai + delta * (bi + delta * (ci + delta * di))

# ---------------------------------------------------------------------------
# Shared helper: uniform-spacing validation
# ---------------------------------------------------------------------------

def _check_uniform_spacing(t_transformed):
    dt = jnp.diff(t_transformed)
    
    # Use the expected dt based on endpoints to avoid jnp.median (which is slow in JIT)
    expected_dt = (t_transformed[-1] - t_transformed[0]) / (t_transformed.shape[0] - 1)
    
    # Calculate the max deviation
    max_dev = jnp.max(jnp.abs(dt - expected_dt)) / expected_dt
    is_bad = max_dev > 0.05

    # jax.debug.print works inside JIT and will print to your terminal/notebook 
    # only when the condition is met during execution.
    jax.lax.cond(
        is_bad,
        lambda _: jax.debug.print(
            "!!! WARNING: Non-uniform grid detected (max deviation {dev}%). "
            "Interpolation results may be inaccurate.",
            dev=max_dev * 100
        ),
        lambda _: None,  # Do nothing if the grid is fine
        operand=None
    )
    
    # Optional: return a boolean to allow the caller to handle it (e.g., inject NaNs)
    return is_bad

# ===========================================================================
# Public classes
# ===========================================================================

class UniformCubicInterpolator:
    """
    Fast O(1) cubic Hermite interpolator for uniformly-spaced data.

    Uses Catmull-Rom splines (C¹ continuous) — smooth first derivatives,
    perfect for dynamics and orbit integration.  No binary search needed
    on uniform grids (linear or log).

    Recommended for most uses.  Use ``UniformSplineC2Interpolator`` only if
    you specifically need C² continuity for higher-order autodiff.

    Supports CPU and GPU transparently: arrays are pinned to the current
    JAX default device at construction time.

    Parameters
    ----------
    t : array_like, shape (N,)
        Time points.  Must be uniformly spaced in linear space when
        ``log_spacing=False`` (default), or uniformly spaced in
        ``log(t)`` when ``log_spacing=True``.  All values must be
        positive when ``log_spacing=True``.
    y : array_like, shape (N,) or (N, D)
        Function values at each time point.  All values must be positive
        when ``log_values=True``.
    check_uniform : bool, optional
        If ``True``, verify that ``t`` is uniformly spaced (in the
        relevant coordinate) and raise ``ValueError`` if not.  Set to
        ``False`` to skip the check when you know the grid is uniform
        (e.g. it came from ``jnp.linspace`` or ``jnp.logspace``).
        Default is ``False``.
    log_spacing : bool, optional
        If ``True``, the grid is uniform in ``log(t)`` rather than in
        ``t``.  The O(1) index lookup becomes
        ``idx = (log(t_query) - log(t_min)) / d_log_t``.  Use this when
        your time array was generated with ``jnp.logspace`` or equivalent.
        Default is ``False``.
    log_values : bool, optional
        If ``True``, interpolation is performed on ``log(y)`` and the
        result is exponentiated before returning.  Useful when ``y``
        spans several orders of magnitude and you want the interpolation
        error to be relative rather than absolute.  Requires all values
        in ``y`` to be strictly positive.  Default is ``False``.

    Examples
    --------
    Linear grid, 1-D values:

    >>> t = jnp.linspace(0, 10, 1000)
    >>> y = jnp.sin(t)
    >>> interp = UniformCubicInterpolator(t, y)
    >>> interp(5.5)                              # scalar → scalar

    Linear grid, multi-dimensional values:

    >>> y2d = jnp.stack([jnp.sin(t), jnp.cos(t)], axis=1)  # (1000, 2)
    >>> interp = UniformCubicInterpolator(t, y2d)
    >>> interp(jnp.array([1.0, 2.0, 3.0]))       # (3,) → (3, 2)

    Log-spaced grid (e.g. radial coordinate spanning decades):

    >>> r = jnp.logspace(-2, 2, 1000)            # 0.01 … 100
    >>> rho = 1.0 / r**2                         # density profile
    >>> interp = UniformCubicInterpolator(r, rho, log_spacing=True)
    >>> interp(jnp.array([0.5, 1.0, 5.0]))       # queries in original r

    Log-spaced grid *and* log-valued (both span orders of magnitude):

    >>> interp = UniformCubicInterpolator(r, rho,
    ...                                   log_spacing=True, log_values=True)
    >>> interp(jnp.array([0.5, 1.0, 5.0]))       # error is now relative
    """

    def __init__(self, t, y, check_uniform=False,
                 log_spacing=False, log_values=False):
        t = jnp.asarray(t)
        y = jnp.asarray(y)

        if y.ndim == 1:
            y = y[:, None]

        n = len(t)
        if n < 4:
            raise ValueError("Cubic Hermite interpolation requires at least 4 points")

        self._log_spacing    = log_spacing
        self._log_values     = log_values
        self._squeeze_output = (y.shape[1] == 1)

        # ------------------------------------------------------------------
        # Coordinate transform: work in the space where the grid is uniform
        # ------------------------------------------------------------------
        if log_spacing:
            t_coord = jnp.log(t)                  # uniform in log(t)
        else:
            t_coord = t                           # uniform in t

        coord_min = t_coord[0] # float(t_coord[0])
        coord_max = t_coord[-1] # float(t_coord[-1])
        d_coord   = (coord_max - coord_min) / (n - 1)

        if check_uniform:
            _check_uniform_spacing(t_coord)

        # Store the coordinate-space quantities the eval kernels need
        self._coord_min = coord_min   # t_min  or log(t_min)
        self._d_coord   = d_coord     # dt     or d_log_t
        self.n_points   = n

        # ------------------------------------------------------------------
        # Value transform: interpolate log(y) if requested
        # ------------------------------------------------------------------
        if log_values:
            y = jnp.log(y)

        # Pin to current device (CPU or GPU)
        self.y = jax.device_put(y)

    # ------------------------------------------------------------------
    # Single-point fast path (used inside integration loops)
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def _eval_scalar(self, t):
        """Scalar eval — linear or log spacing, selected at trace time."""
        if self._log_spacing:
            idx_f = (jnp.log(t) - self._coord_min) / self._d_coord
        else:
            idx_f = (t - self._coord_min) / self._d_coord

        idx0 = jnp.floor(idx_f).astype(jnp.int32)
        idx0 = jnp.clip(idx0, 1, self.n_points - 3)
        s    = idx_f - idx0

        y_m1 = self.y[idx0 - 1]
        y0   = self.y[idx0]
        y1   = self.y[idx0 + 1]
        y2   = self.y[idx0 + 2]

        m0 = (y1 - y_m1) * 0.5
        m1 = (y2 - y0)   * 0.5

        s2 = s * s;  s3 = s2 * s

        h00 =  2.0*s3 - 3.0*s2 + 1.0
        h10 =        s3 - 2.0*s2 + s
        h01 = -2.0*s3 + 3.0*s2
        h11 =        s3 -       s2

        result = h00*y0 + h10*m0 + h01*y1 + h11*m1

        if self._log_values:
            result = jnp.exp(result)

        return result

    def __call__(self, t):
        """
        Evaluate interpolator at time(s) t.

        Parameters
        ----------
        t : float or array_like
            Query time(s) in the *original* coordinate (not log-transformed
            even when ``log_spacing=True``; the transform is internal).

        Returns
        -------
        array_like
            Interpolated values.  Shape rules:

            =============  ============  ==============
            ``t`` shape    ``y`` shape   output shape
            =============  ============  ==============
            scalar         (N,)          scalar
            scalar         (N, D)        (D,)
            (Q,)           (N,)          (Q,)
            (Q,)           (N, D)        (Q, D)
            =============  ============  ==============
        """
        t = jnp.asarray(t)

        if t.ndim == 0:
            result = self._eval_scalar(t)
        else:
            if self._log_spacing:
                result = _eval_catmull_rom_log(
                    t, self._coord_min, self._d_coord,
                    self.n_points, self.y
                )
            else:
                result = _eval_catmull_rom(
                    t, self._coord_min, self._d_coord,
                    self.n_points, self.y
                )
            if self._log_values:
                result = jnp.exp(result)

        if self._squeeze_output:
            result = result.squeeze(axis=-1)
        return result

class UniformSplineC2Interpolator:
    """
    Fast O(1) natural cubic spline interpolator (C² continuous).

    Only use this if you specifically need:

    - Smooth second derivatives
    - Higher-order autodiff (``jax.grad(jax.grad(...))``)
    - Computing tidal tensors or curvature

    For standard orbit integration, use ``UniformCubicInterpolator``
    instead (faster init, similar eval performance).

    Supports CPU and GPU transparently: coefficient computation is fully
    JIT-compiled (single XLA graph via ``lax.scan``), and all arrays are
    pinned to the current device at construction time.

    Parameters
    ----------
    t : array_like, shape (N,)
        Time points.  Must be uniformly spaced in linear space when
        ``log_spacing=False`` (default), or uniformly spaced in
        ``log(t)`` when ``log_spacing=True``.  All values must be
        positive when ``log_spacing=True``.
    y : array_like, shape (N,) or (N, D)
        Function values at each time point.  All values must be positive
        when ``log_values=True``.
    check_uniform : bool, optional
        If ``True``, verify that ``t`` is uniformly spaced (in the
        relevant coordinate) and raise ``ValueError`` if not.  Set to
        ``False`` to skip the check when you know the grid is uniform.
        Default is ``False``.
    log_spacing : bool, optional
        If ``True``, the grid is uniform in ``log(t)`` rather than in
        ``t``.  The O(1) index lookup becomes
        ``idx = (log(t_query) - log(t_min)) / d_log_t``.  The spline
        coefficients are computed in log-space so the polynomial segments
        are in the log coordinate.  Default is ``False``.
    log_values : bool, optional
        If ``True``, interpolation is performed on ``log(y)`` and the
        result is exponentiated before returning.  Useful when ``y``
        spans several orders of magnitude.  Requires all values in ``y``
        to be strictly positive.  Default is ``False``.

    Examples
    --------
    Linear grid, 1-D values:

    >>> t = jnp.linspace(0, 10, 1000)
    >>> y = jnp.sin(t) + 2.0                     # shifted positive for log demo
    >>> interp = UniformSplineC2Interpolator(t, y)
    >>> interp(5.5)                              # scalar → scalar

    Linear grid, multi-dimensional values:

    >>> y2d = jnp.stack([jnp.sin(t) + 2, jnp.cos(t) + 2], axis=1)
    >>> interp = UniformSplineC2Interpolator(t, y2d)
    >>> interp(jnp.array([1.0, 2.0, 3.0]))       # (3,) → (3, 2)

    Log-spaced grid:

    >>> r = jnp.logspace(-2, 2, 1000)            # 0.01 … 100
    >>> rho = 1.0 / r**2
    >>> interp = UniformSplineC2Interpolator(r, rho, log_spacing=True)
    >>> interp(jnp.array([0.5, 1.0, 5.0]))

    Log-spaced grid and log-valued:

    >>> interp = UniformSplineC2Interpolator(r, rho,
    ...                                      log_spacing=True, log_values=True)
    >>> interp(jnp.array([0.5, 1.0, 5.0]))       # relative-error interpolation

    Higher-order autodiff (the whole point of C²):

    >>> interp_1d = UniformSplineC2Interpolator(t, jnp.sin(t) + 2.0)
    >>> jax.grad(lambda x: interp_1d(x))(5.0)              # first derivative
    >>> jax.grad(jax.grad(lambda x: interp_1d(x)))(5.0)    # second derivative
    """

    def __init__(self, t, y, check_uniform=False,
                 log_spacing=False, log_values=False):
        t = jnp.asarray(t)
        y = jnp.asarray(y)

        if y.ndim == 1:
            y = y[:, None]

        n = len(t)
        if n < 3:
            raise ValueError("Spline interpolation requires at least 3 points")

        self._log_spacing    = log_spacing
        self._log_values     = log_values
        self._squeeze_output = (y.shape[1] == 1)

        # ------------------------------------------------------------------
        # Coordinate transform
        # ------------------------------------------------------------------
        if log_spacing:
            t_coord = jnp.log(t)
        else:
            t_coord = t

        coord_min = t_coord[0] # float(t_coord[0])
        coord_max = t_coord[-1] # float(t_coord[-1])
    
        # coord_min = float(t_coord[0])
        # coord_max = float(t_coord[-1])
        d_coord   = (coord_max - coord_min) / (n - 1)

        if check_uniform:
            _check_uniform_spacing(t_coord)

        self._coord_min = coord_min
        self._d_coord   = d_coord
        self.n_points   = n

        # ------------------------------------------------------------------
        # Value transform
        # ------------------------------------------------------------------
        if log_values:
            y = jnp.log(y)

        # Pin input to device before jitted coeff computation
        y = jax.device_put(y)

        # One-time cost: fully JIT-compiled Thomas algorithm (lax.scan).
        # d_coord is the spacing in the interpolation coordinate (log or linear).
        a, b, c, d = _compute_c2_coeffs(y, d_coord)

        # Pin coefficients to device
        self.a = jax.device_put(a)
        self.b = jax.device_put(b)
        self.c = jax.device_put(c)
        self.d = jax.device_put(d)

    # ------------------------------------------------------------------
    # Single-point fast path
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def _eval_scalar(self, t):
        """Scalar eval using Horner's method — linear or log spacing."""
        if self._log_spacing:
            coord = jnp.log(t)
        else:
            coord = t

        idx_f  = (coord - self._coord_min) / self._d_coord
        idx0   = jnp.floor(idx_f).astype(jnp.int32)
        idx0   = jnp.clip(idx0, 0, self.a.shape[0] - 1)
        delta  = coord - (self._coord_min + idx0 * self._d_coord)

        ai = self.a[idx0]
        bi = self.b[idx0]
        ci = self.c[idx0]
        di = self.d[idx0]

        result = ai + delta * (bi + delta * (ci + delta * di))

        if self._log_values:
            result = jnp.exp(result)

        return result

    def __call__(self, t):
        """
        Evaluate interpolator at time(s) t.

        Parameters
        ----------
        t : float or array_like
            Query time(s) in the *original* coordinate (not log-transformed
            even when ``log_spacing=True``; the transform is internal).

        Returns
        -------
        array_like
            Interpolated values.  Shape rules:

            =============  ============  ==============
            ``t`` shape    ``y`` shape   output shape
            =============  ============  ==============
            scalar         (N,)          scalar
            scalar         (N, D)        (D,)
            (Q,)           (N,)          (Q,)
            (Q,)           (N, D)        (Q, D)
            =============  ============  ==============
        """
        t = jnp.asarray(t)

        if t.ndim == 0:
            result = self._eval_scalar(t)
        else:
            if self._log_spacing:
                result = _eval_c2_spline_log(
                    t, self._coord_min, self._d_coord,
                    self.a, self.b, self.c, self.d
                )
            else:
                result = _eval_c2_spline(
                    t, self._coord_min, self._d_coord,
                    self.a, self.b, self.c, self.d
                )
            if self._log_values:
                result = jnp.exp(result)

        if self._squeeze_output:
            result = result.squeeze(axis=-1)
        return result
# ===========================================================================