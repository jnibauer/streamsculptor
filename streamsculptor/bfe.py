"""
bfe — streamsculptor (eqx_everything branch) wrapper for bfeax.

Drop this file into streamsculptor/ and import as:

    from streamsculptor.bfeax_potential import BFEAxPotential

Requirements
------------
    pip install git+https://github.com/jnibauer/bfeax.git

Usage
-----
    from streamsculptor.bfeax_potential import BFEAxPotential
    from streamsculptor.main import usys   # default GalacticUnitSystem

    # From Agama-style spheroid parameters (rho0 in Msun/kpc^3, a in kpc)
    pot = BFEAxPotential.from_spheroid(
        rho0=1e7, alpha=1.0, beta=3.0, gamma=1.0, a=20.0,
        p=0.8, q=0.5,
        r_min=0.2, r_max=600., n_r=128, l_max=8,
        symmetry="triaxial",
    )

    # From any JAX-traceable density function
    pot = BFEAxPotential.from_density(
        rho, r_min=0.2, r_max=600., n_r=128, l_max=8
    )

    # From a pre-built MultipoleExpansion
    from bfeax import MultipoleExpansion
    exp = MultipoleExpansion.from_density(rho, r_min=0.2, r_max=600., n_r=128, l_max=8)
    pot = BFEAxPotential(exp)

    # Standard streamsculptor interface
    phi = pot.potential(xyz, t)       # xyz: jnp.array([x, y, z])
    acc = pot.acceleration(xyz, t)    # -grad Phi
    rho = pot.density(xyz, t)         # Laplacian(Phi) / (4 pi G)

Notes on units
--------------
bfeax solves Poisson's equation with G=1 internally, so the raw expansion
output has units of [rho0 * length^2].  Multiplying by self.units.G converts
this to a true potential in (kpc/Myr)^2, consistent with the rest of
streamsculptor.  All parameters (rho0, a, r_min, r_max) should be passed as
plain floats in kpc / Msun / Myr units to match the default GalacticUnitSystem.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from functools import partial
try:
    from streamsculptor.main import Potential, usys
except ImportError as e:
    raise ImportError(
        "streamsculptor (eqx_everything branch) is required. "
        "Install it from https://github.com/jnibauer/streamsculptor"
    ) from e

try:
    from bfeax import MultipoleExpansion
except ImportError as e:
    raise ImportError(
        "bfeax is required. "
        "Install it with: pip install git+https://github.com/jnibauer/bfeax.git"
    ) from e


class BFEPotential(Potential):
    """
    Wrap a bfeax.MultipoleExpansion as a streamsculptor Potential.

    The expansion is static (time-independent). The `t` argument accepted
    by all methods is ignored and exists only to satisfy the streamsculptor
    interface.

    gradient, acceleration, and density are all inherited from the
    streamsculptor Potential base class and computed via JAX autodiff
    through potential().

    Parameters
    ----------
    expansion : bfeax.MultipoleExpansion
        A pre-built expansion.  Parameters (rho0, a, etc.) should be in
        kpc / Msun / Myr to match the default GalacticUnitSystem.
    units : GalacticUnitSystem, optional
        Defaults to the streamsculptor default unit system (usys).
    """

    _exp: MultipoleExpansion# = eqx.field(static=True) # used to have just  _exp: MultipoleExpansion

    def __init__(self, expansion: MultipoleExpansion, units=usys):
        super().__init__(units)
        self._exp = expansion

    
    def potential(self, xyz, t):
        """
        Evaluate Phi(x, y, z).

        bfeax solves Poisson with G=1, so we multiply by self.units.G to
        recover the physical potential in (kpc/Myr)^2.

        Parameters
        ----------
        xyz : jnp.ndarray, shape (3,)
            Position vector [x, y, z] in kpc.
        t : float
            Time in Myr (unused -- expansion is static).

        Returns
        -------
        float
            Gravitational potential Phi at xyz, in (kpc/Myr)^2.
        """
        return self.units.G * self._exp(xyz[0], xyz[1], xyz[2])

    
    def gradient(self,xyz,t):
        """
        Evaluate the gradient of the potential at xyz.

        Parameters
        ----------
        xyz : jnp.ndarray, shape (3,)
            Position vector [x, y, z] in kpc.
        t : float
            Time in Myr (unused -- expansion is static).

        Returns
        -------
        jnp.ndarray, shape (3,)
            Gradient of the potential at xyz, in (kpc/Myr)^2 / kpc.
        """
        return -self.units.G * jnp.hstack(self._exp.force(xyz[0], xyz[1], xyz[2]))
        

    # gradient(xyz, t)     = jax.grad(self.potential)(xyz, t)   [inherited]
    # acceleration(xyz, t) = -gradient(xyz, t)                   [inherited]
    # density(xyz, t)      = Laplacian(potential) / (4 pi G)     [inherited]

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_density(
        cls,
        rho,
        r_min: float,
        r_max: float,
        n_r: int,
        l_max: int,
        n_theta: int | None = None,
        n_phi: int | None = None,
        symmetry: str | None = None,
        prune_modes: bool = False,
        units=usys,
    ) -> "BFEAxPotential":
        """
        Build from any JAX-traceable density function rho(x, y, z).

        rho should return density in Msun/kpc^3 as a plain float.
        The potential() output is then scaled by self.units.G automatically.

        Parameters
        ----------
        rho      : callable -- rho(x, y, z) -> scalar in Msun/kpc^3
        r_min    : inner radius of the radial grid in kpc
        r_max    : outer radius of the radial grid in kpc
        n_r      : number of radial grid points (128 recommended)
        l_max    : maximum spherical harmonic degree (8 is a good default)
        n_theta  : Gauss-Legendre nodes in cos(theta) (default: l_max + 2)
        n_phi    : uniform phi points (default: 2*l_max + 1)
        symmetry : "spherical", "axisymmetric", "triaxial", or None
        units    : GalacticUnitSystem (default: streamsculptor usys)
        """
        exp = MultipoleExpansion.from_density(
            rho, r_min=r_min, r_max=r_max, n_r=n_r, l_max=l_max,
            n_theta=n_theta, n_phi=n_phi, symmetry=symmetry, prune_modes=prune_modes
        )
        return cls(exp, units=units)

    @classmethod
    def from_spheroid(
        cls,
        rho0: float,
        alpha: float,
        beta: float,
        gamma: float,
        a: float,
        *,
        p: float = 1.0,
        q: float = 1.0,
        r_cut: float | None = None,
        xi: float = 0.0,
        r_min: float,
        r_max: float,
        n_r: int = 128,
        l_max: int = 8,
        n_theta: int | None = None,
        n_phi: int | None = None,
        symmetry: str | None = None,
        units=usys,
    ) -> "BFEAxPotential":
        """
        Build from Agama-style spheroid parameters.

        The density profile is:
            rho(r) = rho0 * (r/a)^{-gamma} * [1 + (r/a)^alpha]^{(gamma-beta)/alpha}
                     * exp[-(r/r_cut)^xi]

        where r = sqrt(x^2 + (y/p)^2 + (z/q)^2) is the spheroidal radius.
        All length parameters should be in kpc; rho0 in Msun/kpc^3.

        Common profiles
        ---------------
        NFW       : alpha=1, beta=3, gamma=1
        Hernquist : alpha=1, beta=4, gamma=1
        Plummer   : alpha=2, beta=5, gamma=0
        Jaffe     : alpha=1, beta=4, gamma=2

        Parameters
        ----------
        rho0     : density normalisation in Msun/kpc^3
        alpha, beta, gamma : profile shape exponents (dimensionless)
        a        : scale radius in kpc
        p, q     : y and z axis ratios (dimensionless, default 1.0 = spherical)
        r_cut    : outer exponential cutoff radius in kpc (None = no cutoff)
        xi       : cutoff strength (dimensionless, 0 = no cutoff)
        r_min    : inner radius of the expansion grid in kpc
        r_max    : outer radius in kpc (use >= 30*a for NFW)
        n_r      : radial grid points (128 recommended)
        l_max    : max spherical harmonic degree (8 recommended)
        symmetry : "spherical", "axisymmetric", "triaxial", or None
        units    : GalacticUnitSystem (default: streamsculptor usys)
        """
        exp = MultipoleExpansion.from_spheroid(
            rho0=rho0, alpha=alpha, beta=beta, gamma=gamma, a=a,
            p=p, q=q, r_cut=r_cut, xi=xi,
            r_min=r_min, r_max=r_max, n_r=n_r, l_max=l_max,
            n_theta=n_theta, n_phi=n_phi, symmetry=symmetry,
        )
        return cls(exp, units=units)
