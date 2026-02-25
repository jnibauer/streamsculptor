import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from jax.scipy import special
import equinox as eqx
from quadax import quadgk
import interpax
from jax.scipy.interpolate import RegularGridInterpolator
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
import os
from typing import Callable, Any

# Assuming main.py is in the same directory or accessible via import
from streamsculptor.main import Potential, GalacticUnitSystem, usys

jax.config.update("jax_enable_x64", True)

def _require_agama():
    try:
        from .InterpAGAMA import AGAMA_Spheroid
    except Exception as e:
        raise ImportError(
            "AGAMA-based spheroidal potential is optional and requires the 'agama' extra.\n"
            "Install with: 'streamsculptor[agama]'"
        ) from e
    return AGAMA_Spheroid

# =============================================================================
# Standard Potentials
# =============================================================================

class MiyamotoNagaiDisk(Potential):
    m: float
    a: float
    b: float

    def __init__(self, m, a, b, units=usys):
        super().__init__(units)
        self.m = m
        self.a = a
        self.b = b

    def potential(self, xyz, t):
        R2 = xyz[0]**2 + xyz[1]**2
        return -self.units.G * self.m / jnp.sqrt(R2 + jnp.square(jnp.sqrt(xyz[2]**2 + self.b**2) + self.a))

class NFWPotential(Potential):
    m: float
    r_s: float

    def __init__(self, m, r_s, units=usys):
        super().__init__(units)
        self.m = m
        self.r_s = r_s

    def potential(self, xyz, t):
        v_h2 = -self.units.G * self.m / self.r_s
        m_val = jnp.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2) / self.r_s
        return v_h2 * jnp.log(1.0 + m_val) / m_val

class TriaxialNFWPotential(Potential):
    m: float
    r_s: float
    q1: float
    q2: float
    q3: float

    def __init__(self, m, r_s, q1=1.0, q2=1.0, q3=1.0, units=usys):
        super().__init__(units)
        self.m = m
        self.r_s = r_s
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3

    def potential(self, xyz, t):
        xyz_scaled = jnp.array([xyz[0] / self.q1, xyz[1] / self.q2, xyz[2] / self.q3])
        v_h2 = -self.units.G * self.m / self.r_s
        m_val = jnp.sqrt(xyz_scaled[0]**2 + xyz_scaled[1]**2 + xyz_scaled[2]**2) / self.r_s
        return v_h2 * jnp.log(1.0 + m_val) / m_val

class Isochrone(Potential):
    m: float
    a: float

    def __init__(self, m, a, units=usys):
        super().__init__(units)
        self.m = m
        self.a = a

    def potential(self, xyz, t):
        r = jnp.linalg.norm(xyz, axis=0)
        return -self.units.G * self.m / (self.a + jnp.sqrt(r**2 + self.a**2))
    
class PlummerPotential(Potential):
    m: float
    r_s: float

    def __init__(self, m, r_s, units=usys):
        super().__init__(units)
        self.m = m
        self.r_s = r_s

    def potential(self, xyz, t):
        r_squared = xyz[0]**2 + xyz[1]**2 + xyz[2]**2
        return -self.units.G * self.m / jnp.sqrt(r_squared + self.r_s**2)

class HernquistPotential(Potential):
    m: float
    r_s: float
    soft: float

    def __init__(self, m, r_s, soft=0.0, units=usys):
        super().__init__(units)
        self.m = m
        self.r_s = r_s
        self.soft = soft

    def potential(self, xyz, t):
        r = jnp.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2 + self.soft)
        return -self.units.G * self.m / (r + self.r_s) 

# =============================================================================
# Time-Dependent & Progenitor Potentials
# =============================================================================

class LMCPotential(Potential):
    LMC_internal: dict = eqx.field(static=True) 
    spl_x: Any
    spl_y: Any
    spl_z: Any

    def __init__(self, LMC_internal, LMC_orbit, units=usys):
        super().__init__(units)
        self.LMC_internal = LMC_internal
        self.spl_x = InterpolatedUnivariateSpline(LMC_orbit['t'], LMC_orbit['x'], k=3)
        self.spl_y = InterpolatedUnivariateSpline(LMC_orbit['t'], LMC_orbit['y'], k=3)
        self.spl_z = InterpolatedUnivariateSpline(LMC_orbit['t'], LMC_orbit['z'], k=3)

    def potential(self, xyz, t):
        LMC_pos = jnp.array([self.spl_x(t), self.spl_y(t), self.spl_z(t)])
        xyz_adjust = xyz - LMC_pos
        potential_lmc = NFWPotential(m=self.LMC_internal['m_NFW'], r_s=self.LMC_internal['r_s_NFW'], units=self.units)
        return potential_lmc.potential(xyz_adjust, t)

class ProgenitorPotential(Potential):
    m: float
    r_s: float
    interp_func: Any
    prog_pot_cls: type = eqx.field(static=True)

    def __init__(self, m, r_s, interp_func, prog_pot, units=usys):
        super().__init__(units)
        self.m = m
        self.r_s = r_s
        self.interp_func = interp_func
        self.prog_pot_cls = prog_pot

    def potential(self, xyz, t):
        eval_pt = xyz - self.interp_func.evaluate(t)[:3]
        prog_pot = self.prog_pot_cls(m=self.m, r_s=self.r_s, units=self.units)
        return prog_pot.potential(eval_pt, t)

class TimeDepProgenitorPotential(Potential):
    mass_spl: Any
    r_s_spl: Any
    interp_func: Any
    prog_pot_cls: type = eqx.field(static=True)

    def __init__(self, mass_spl, r_s_spl, interp_func, prog_pot, units=usys):
        super().__init__(units)
        self.mass_spl = mass_spl
        self.r_s_spl = r_s_spl
        self.interp_func = interp_func
        self.prog_pot_cls = prog_pot

    def potential(self, xyz, t):
        eval_pt = xyz - self.interp_func.evaluate(t)[:3]
        mass_curr = self.mass_spl(t)
        r_s_curr = self.r_s_spl(t)
        pot_curr = self.prog_pot_cls(m=mass_curr, r_s=r_s_curr, units=self.units)
        return pot_curr.potential(eval_pt, t)

class TimeDepTranslatingPotential(Potential):
    pot: Potential
    center_spl: Any

    def __init__(self, pot, center_spl, units=usys):
        super().__init__(units)
        self.pot = pot
        self.center_spl = center_spl

    def potential(self, xyz, t):
        center = self.center_spl(t)
        xyz_adjust = xyz - center
        return self.pot.potential(xyz_adjust, t)

class GrowingPotential(Potential):
    pot: Potential
    growth_func: Any

    def __init__(self, pot, growth_func, units=usys):
        super().__init__(units)
        self.pot = pot
        self.growth_func = growth_func
    
    def potential(self, xyz, t):
        growth_factor = self.growth_func(t)
        return self.pot.potential(xyz, t) * growth_factor

# =============================================================================
# Bars & Exponential Disks
# =============================================================================

class BarPotential(Potential):
    m: float
    a: float
    b: float
    c: float
    Omega: float

    def __init__(self, m, a, b, c, Omega, units=usys):
        super().__init__(units)
        self.m = m
        self.a = a
        self.b = b
        self.c = c
        self.Omega = Omega

    def potential(self, xyz, t):
        ang = -self.Omega * t
        Rot_mat = jnp.array([[jnp.cos(ang), -jnp.sin(ang), 0], [jnp.sin(ang), jnp.cos(ang), 0.], [0.0, 0.0, 1.0]])
        xyz_corot = jnp.matmul(Rot_mat, xyz)
        
        T_plus = jnp.sqrt((self.a + xyz_corot[0])**2 + xyz_corot[1]**2 + (self.b + jnp.sqrt(self.c**2 + xyz_corot[2]**2))**2)
        T_minus = jnp.sqrt((self.a - xyz_corot[0])**2 + xyz_corot[1]**2 + (self.b + jnp.sqrt(self.c**2 + xyz_corot[2]**2))**2)
        
        return (self.units.G * self.m / (2.0 * self.a)) * jnp.log((xyz_corot[0] - self.a + T_minus) / (xyz_corot[0] + self.a + T_plus))

class DehnenBarPotential(Potential):
    alpha: float
    v0: float
    R0: float
    Rb: float
    phib: float
    Omega: float

    def __init__(self, alpha, v0, R0, Rb, phib, Omega, units=usys):
        super().__init__(units)
        self.alpha = alpha
        self.v0 = v0
        self.R0 = R0
        self.Rb = Rb
        self.phib = phib
        self.Omega = Omega

    def potential(self, xyz, t):
        phi = jnp.arctan2(xyz[1], xyz[0])
        R = jnp.sqrt(xyz[0]**2 + xyz[1]**2)
        r = jnp.sqrt(jnp.sum(xyz**2))

        def U_func(r_val):
            def gtr_func():
                return -(r_val / self.Rb)**(-3)
            def less_func():
                return (r_val / self.Rb)**3 - 2.0
            bool_eval = r_val >= self.Rb
            return jax.lax.cond(bool_eval, gtr_func, less_func)
        
        U_eval = U_func(r)
        prefacs = self.alpha * ((self.v0**2) / 3) * ((self.R0 / self.Rb)**3)
        return prefacs * ((R**2 / r**2)) * U_eval * jnp.cos(2 * (phi - self.phib - self.Omega * t))

class MN3ExponentialDiskPotential(Potential):
    m: float
    h_R: float
    h_z: float
    positive_density: bool = eqx.field(static=True)
    sech2_z: bool = eqx.field(static=True)
    _ms: jnp.ndarray
    _as: jnp.ndarray
    _b: float

    def __init__(self, m, h_R, h_z, units=usys, positive_density=True, sech2_z=True):
        super().__init__(units)
        self.m = m
        self.h_R = h_R
        self.h_z = h_z
        self.positive_density = positive_density
        self.sech2_z = sech2_z

        _K_pos_dens = jnp.array([
            [0.0036, -0.0330, 0.1117, -0.1335, 0.1749],
            [-0.0131, 0.1090, -0.3035, 0.2921, -5.7976],
            [-0.0048, 0.0454, -0.1425, 0.1012, 6.7120],
            [-0.0158, 0.0993, -0.2070, -0.7089, 0.6445],
            [-0.0319, 0.1514, -0.1279, -0.9325, 2.6836],
            [-0.0326, 0.1816, -0.2943, -0.6329, 2.3193],
        ])
        _K_neg_dens = jnp.array([
            [-0.0090, 0.0640, -0.1653, 0.1164, 1.9487],
            [0.0173, -0.0903, 0.0877, 0.2029, -1.3077],
            [-0.0051, 0.0287, -0.0361, -0.0544, 0.2242],
            [-0.0358, 0.2610, -0.6987, -0.1193, 2.0074],
            [-0.0830, 0.4992, -0.7967, -1.2966, 4.4441],
            [-0.0247, 0.1718, -0.4124, -0.5944, 0.7333],
        ])

        K = _K_pos_dens if positive_density else _K_neg_dens
        hzR = h_z / h_R

        if sech2_z:
            b_hR = -0.033 * hzR**3 + 0.262 * hzR**2 + 0.659 * hzR
        else:
            b_hR = -0.269 * hzR**3 + 1.08 * hzR**2 + 1.092 * hzR

        x = jnp.array([b_hR**4, b_hR**3, b_hR**2, b_hR, 1.0])
        param_vec = jnp.dot(K, x)

        self._ms = param_vec[:3] * m
        self._as = param_vec[3:] * h_R
        self._b = b_hR * h_R

    def potential(self, xyz, t):
        pot = 0.0
        for i in range(3):
            pot += MiyamotoNagaiDisk(m=self._ms[i], a=self._as[i], b=self._b, units=self.units).potential(xyz, t)
        return pot

# =============================================================================
# Composite & Advanced Potentials
# =============================================================================

class Potential_Combine(Potential):
    potential_list: list

    def __init__(self, potential_list, units=usys):
        super().__init__(units)
        self.potential_list = potential_list

    def potential(self, xyz, t):
        return jnp.sum(jnp.array([pot.potential(xyz, t) for pot in self.potential_list]))

    def gradient(self, xyz, t):
        # Initialize a zero vector of shape (3,)
        total_grad = jnp.zeros((3,))
        for pot in self.potential_list:
            # Add each gradient one by one to avoid jnp.array() shape-matching strictness
            total_grad += pot.gradient(xyz, t).reshape(3)
        return total_grad

class GalaMilkyWayPotential(Potential):
    pot: Potential_Combine

    def __init__(self, units=usys):
        super().__init__(units)
        pot_disk = MiyamotoNagaiDisk(m=6.80e10, a=3.0, b=0.28, units=units)
        pot_bulge = HernquistPotential(m=5e9, r_s=1.0, units=units)
        pot_nucleus = HernquistPotential(m=1.71e9, r_s=0.07, units=units)
        pot_halo = NFWPotential(m=5.4e11, r_s=15.62, units=units)
        self.pot = Potential_Combine([pot_disk, pot_bulge, pot_nucleus, pot_halo], units=units)

    def potential(self, xyz, t):
        return self.pot.potential(xyz, t)

class BovyMWPotential2014(Potential):
    pot: Potential_Combine

    def __init__(self, units=usys):
        super().__init__(units)
        pot_disk = MiyamotoNagaiDisk(m=6.82e10, a=3.0, b=0.28, units=units)
        pot_bulge = PowerLawCutoffPotential(m=4.50e9, alpha=1.80, r_c=1.90, units=units)
        pot_halo = NFWPotential(m=4.37e11, r_s=16.0, units=units)
        self.pot = Potential_Combine([pot_disk, pot_bulge, pot_halo], units=units)

    def potential(self, xyz, t):
        return self.pot.potential(xyz, t)

# =============================================================================
# Power Law & Zhao
# =============================================================================

class PowerLawCutoffPotential(Potential):
    m: float
    alpha: float
    r_c: float

    def __init__(self, m, alpha, r_c, units=usys):
        super().__init__(units)
        self.m = m
        self.alpha = alpha
        self.r_c = r_c

    def potential(self, xyz, t):
        r = jnp.sqrt(jnp.sum(xyz**2))
        tmp_0 = (1/2.) * self.alpha
        tmp_1 = -tmp_0
        tmp_2 = tmp_1 + 1.5
        tmp_3 = r**2
        tmp_4 = tmp_3 / self.r_c**2
        tmp_5 = self.units.G * self.m
        tmp_6 = tmp_5 * special.gammainc(tmp_2, tmp_4) * special.gamma(tmp_2) / (jnp.sqrt(tmp_3) * special.gamma(tmp_1 + 2.5))
        return tmp_0 * tmp_6 - 3./2.0 * tmp_6 + tmp_5 * special.gammainc(tmp_1 + 1, tmp_4) * special.gamma(tmp_1 + 1) / (self.r_c * special.gamma(tmp_2))
    
    def gradient(self, xyz, t):
        r = jnp.sqrt(jnp.sum(xyz**2))
        dPhi_dr = (self.units.G * self.m / (r**2) * special.gammainc(0.5 * (3 - self.alpha), r*r / (self.r_c * self.r_c)))
        return dPhi_dr * xyz / r

class ZhaoPotential(Potential):
    m: float
    r_s: float
    alpha: float
    beta: float
    gamma: float

    def __init__(self, m, r_s, alpha, beta, gamma, units=usys):
        super().__init__(units)
        self.m = m
        self.r_s = r_s
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _rho0(self, m, alpha, beta, gamma):
        denom = alpha * jsp.beta(alpha * (3.0 - gamma), alpha * (beta - 3.0))
        return m / (4.0 * jnp.pi * denom)

    def _potential(self, m, r_s, alpha, beta, gamma, G, r):
        x = r / r_s
        z = x ** (1.0 / alpha) / (1.0 + x ** (1.0 / alpha))
        rho0 = self._rho0(m, alpha, beta, gamma)
        p0 = alpha * (2.0 - gamma)
        q0 = alpha * (beta - 3.0)
        c0 = alpha * (beta - gamma)
        term_l = jsp.beta(c0 - q0, q0) * jsp.betainc(c0 - q0, q0, z)
        eps = jnp.sqrt(jnp.finfo(r.dtype).eps)
        p0_safe = jnp.where(p0 <= 0, eps, p0)
        logB = jsp.betaln(p0_safe, c0 - p0)
        log1mI = jnp.log1p(-jsp.betainc(p0_safe, c0 - p0, z))
        term_r = jnp.exp(logB + log1mI)
        return -4.0 * jnp.pi * G * rho0 * alpha * (term_l / r + term_r / r_s)

    def potential(self, xyz, t):
        r = jnp.sqrt(jnp.sum(xyz**2))
        return self._potential(self.m, self.r_s, self.alpha, self.beta, self.gamma, self.units.G, r)

    def gradient(self, xyz, t):
        r = jnp.sqrt(jnp.sum(xyz**2))
        dr = 1e-5 * r + 1e-8
        phi_r = self._potential(self.m, self.r_s, self.alpha, self.beta, self.gamma, self.units.G, r)
        phi_r_dr = self._potential(self.m, self.r_s, self.alpha, self.beta, self.gamma, self.units.G, r + dr)
        dphi_dr = (phi_r_dr - phi_r) / dr
        return dphi_dr * xyz / r

# =============================================================================
# Custom & Density-based Potentials
# =============================================================================

class UniformAcceleration(Potential):
    velocity_func: Any 

    def __init__(self, velocity_func, units=usys):
        super().__init__(units)
        self.velocity_func = velocity_func

    def potential(self, xyz, t):
        raise NotImplementedError("Uniform acceleration has no defined scalar potential.")
    
    def gradient(self, xyz, t):
        # We need the derivative of v(t) with respect to t.
        # This function MUST return a (3,) array for JAX to see the correct Jacobian.
        def get_v(time):
            # Ensure time is a 1D array for the interpolator, then index out the 3-vector
            v = self.velocity_func(jnp.atleast_1d(time))
            return v.reshape(3) # Forces the output to be a 3-vector

        # Use jacfwd on our wrapper. Since get_v: R -> R^3, 
        # jacfwd(get_v)(t) will be shape (3,)
        accel = jax.jacfwd(get_v)(t)
        
        # Final safety check to ensure we return (3,)
        return accel.reshape(3)
    
    def acceleration(self, xyz, t):
        return -self.gradient(xyz, t)

class get_potential_from_density(Potential):
    density_func: Any
    r_grid: jnp.ndarray
    phi_grid: jnp.ndarray

    def __init__(self, density_func, r_grid=None, units=usys):
        super().__init__(units)
        self.density_func = density_func
        self.r_grid = jnp.logspace(-3, 3, 128) if r_grid is None else r_grid
        self.phi_grid = jax.vmap(self.compute_potential)(self.r_grid)

    def compute_potential(self, r):
        def inner_integrand(rp):
            return self.density_func(rp) * rp**2
        def outer_integrand(rp):
            return self.density_func(rp) * rp
        
        inner = quadgk(fun=inner_integrand, interval=jnp.array([0., r]))[0]
        outer = quadgk(fun=outer_integrand, interval=jnp.array([r, jnp.inf]))[0]
        return -4 * jnp.pi * self.units.G * (inner / r + outer)

    def potential(self, xyz, t):
        func = interpax.Interpolator1D(x=jnp.log(self.r_grid), f=self.phi_grid, method='cubic')
        return func(jnp.log(jnp.linalg.norm(xyz)))

class CustomPotential(Potential):
    potential_func: Any

    def __init__(self, potential_func=None, units=usys):
        super().__init__(units)
        self.potential_func = potential_func
        
    def potential(self, xyz, t):
        return self.potential_func(xyz, t)

# =============================================================================
# AGAMA Interfacing & Milky Way / LMC System
# =============================================================================


class MW_LMC_Potential(Potential):
    """
    MW-LMC potential in the non-inertial frame of the Milky Way,
    using per-component cubic spline (interpax.Interpolator1D) interpolators
    for LMC position and MW velocity so velocities/accelerations are smooth
    and JAX-friendly.
    """
    # static interpolator objects (Equinox should not try to differentiate through them)
    LMC_x: Any = eqx.field(static=True)
    LMC_y: Any = eqx.field(static=True)
    LMC_z: Any = eqx.field(static=True)

    vel_x: Any = eqx.field(static=True)
    vel_y: Any = eqx.field(static=True)
    vel_z: Any = eqx.field(static=True)

    pot_MW: Potential_Combine
    pot_LMC: NFWPotential
    translating_LMC_pot: TimeDepTranslatingPotential
    unif_acc: UniformAcceleration
    total_pot: Potential_Combine

    def __init__(self, units=usys):
        super().__init__(units)

        # 1. Load Data (same arrays you saved earlier)
        data_path_MW = os.path.join(os.path.dirname(__file__), 'data/LMC_MW_potential', 'MW_motion_dict.npy')
        data_path_LMC = os.path.join(os.path.dirname(__file__), 'data/LMC_MW_potential', 'LMC_motion_dict.npy')

        MW_motion_dict = jnp.load(data_path_MW, allow_pickle=True).item()
        LMC_motion_dict = jnp.load(data_path_LMC, allow_pickle=True).item()

        flip_tsave = LMC_motion_dict['flip_tsave']  # should be same time grid used earlier
        flip_trajLMC = LMC_motion_dict['flip_trajLMC']  # shape (nt, 3+?)
        flip_traj_MW = MW_motion_dict['flip_traj']   # shape (nt, 6)

        # 2. Make per-component cubic interpolators (Interpax is JAX-friendly)
        self.LMC_x = interpax.Interpolator1D(x=flip_tsave, f=flip_trajLMC[:, 0], method='cubic2')
        self.LMC_y = interpax.Interpolator1D(x=flip_tsave, f=flip_trajLMC[:, 1], method='cubic2')
        self.LMC_z = interpax.Interpolator1D(x=flip_tsave, f=flip_trajLMC[:, 2], method='cubic2')

        self.vel_x = interpax.Interpolator1D(x=flip_tsave, f=flip_traj_MW[:, 3], method='cubic2')
        self.vel_y = interpax.Interpolator1D(x=flip_tsave, f=flip_traj_MW[:, 4], method='cubic2')
        self.vel_z = interpax.Interpolator1D(x=flip_tsave, f=flip_traj_MW[:, 5], method='cubic2')

        # 3. Component Potentials
        pot_bulge = HernquistPotential(m=5e9, r_s=1.0, units=units)
        pot_disk = MiyamotoNagaiDisk(m=5.0e10, a=3.0, b=0.3, units=units)
        pot_halo = NFWPotential(m=5.4e11, r_s=15.62, units=units)
        self.pot_MW = Potential_Combine([pot_bulge, pot_disk, pot_halo], units=units)

        massLMC = 0.85e11
        radiusLMC = (massLMC/1e11)**0.6 * 8.5
        self.pot_LMC = NFWPotential(m=massLMC, r_s=radiusLMC, units=units)

        # 4. Build small JAX-friendly pure functions that use the static interpolators:
        # bind local names to avoid capturing `self` in a way that re-introduces recursion
        LMC_x_local, LMC_y_local, LMC_z_local = self.LMC_x, self.LMC_y, self.LMC_z
        vel_x_local, vel_y_local, vel_z_local = self.vel_x, self.vel_y, self.vel_z

        # these functions are pure wrt time and return jnp arrays (and are JIT-able)
        @jax.jit
        def lmc_center_fn(t):
            return jnp.array([LMC_x_local(t), LMC_y_local(t), LMC_z_local(t)])

        @jax.jit
        def mw_velocity_fn(t):
            return jnp.array([vel_x_local(t), vel_y_local(t), vel_z_local(t)])

        self.translating_LMC_pot = TimeDepTranslatingPotential(
            pot=self.pot_LMC, center_spl=lmc_center_fn, units=units
        )
        self.unif_acc = UniformAcceleration(
            velocity_func=mw_velocity_fn, units=units
        )

        self.total_pot = Potential_Combine(
            [self.pot_MW, self.translating_LMC_pot, self.unif_acc],
            units=units
        )

    # JIT-able accessors that return jnp arrays (for diagnostics / external use)
    @eqx.filter_jit
    def LMC_center_spline(self, t):
        return jnp.array([self.LMC_x(t), self.LMC_y(t), self.LMC_z(t)])

    @eqx.filter_jit
    def MW_velocity_func(self, t):
        return jnp.array([self.vel_x(t), self.vel_y(t), self.vel_z(t)])

    @eqx.filter_jit
    def gradient(self, xyz, t):
        return self.total_pot.gradient(xyz, t)

    @eqx.filter_jit
    def acceleration(self, xyz, t):
        return self.total_pot.acceleration(xyz, t)

    def potential(self, xyz, t):
        raise NotImplementedError("Potential not implemented, force is non-conservative")
        
class AGAMA_MW_LMC_Potential(Potential):
    LMC_x: Any
    LMC_y: Any
    LMC_z: Any
    velocity_func_x: Any
    velocity_func_y: Any
    velocity_func_z: Any
    pot_MW: Potential_Combine
    pot_LMC: Any
    translating_LMC_pot: TimeDepTranslatingPotential
    unif_acc: UniformAcceleration
    total_pot: Potential_Combine

    def __init__(self, units=usys):
        AGAMA_Spheroid = _require_agama()
        super().__init__(units)
        
        data_path_MW = os.path.join(os.path.dirname(__file__), 'data/LMC_MW_potential', 'MW_motion_dict.npy')
        data_path_LMC = os.path.join(os.path.dirname(__file__), 'data/LMC_MW_potential', 'LMC_motion_dict.npy')
        MW_motion_dict = jnp.load(data_path_MW, allow_pickle=True).item()
        LMC_motion_dict = jnp.load(data_path_LMC, allow_pickle=True).item()
        
        self.LMC_x = interpax.Interpolator1D(x=LMC_motion_dict['flip_tsave'], f=LMC_motion_dict['flip_trajLMC'][:,0], method='cubic2')
        self.LMC_y = interpax.Interpolator1D(x=LMC_motion_dict['flip_tsave'], f=LMC_motion_dict['flip_trajLMC'][:,1], method='cubic2')
        self.LMC_z = interpax.Interpolator1D(x=LMC_motion_dict['flip_tsave'], f=LMC_motion_dict['flip_trajLMC'][:,2], method='cubic2')

        self.velocity_func_x = interpax.Interpolator1D(x=MW_motion_dict['flip_tsave'], f=MW_motion_dict['flip_traj'][:,3], method='cubic2')
        self.velocity_func_y = interpax.Interpolator1D(x=MW_motion_dict['flip_tsave'], f=MW_motion_dict['flip_traj'][:,4], method='cubic2')
        self.velocity_func_z = interpax.Interpolator1D(x=MW_motion_dict['flip_tsave'], f=MW_motion_dict['flip_traj'][:,5], method='cubic2')

        paramBulge = dict(type='Spheroid', mass=1.2e10, scaleRadius=0.2, outerCutoffRadius=1.8, gamma=0.0, beta=1.8)
        paramDisk = dict(type='MiyamotoNagai', mass=5.0e10, scaleRadius=3.0, scaleHeight=0.3)
        paramHalo = dict(type='Spheroid', densityNorm=1.35e7, scaleRadius=14, outerCutoffRadius=300, cutoffStrength=4, gamma=1, beta=3)

        massLMC = 1.5e11
        radiusLMC = (massLMC/1e11)**0.6 * 8.5
        paramLMC = dict(type='spheroid', mass=massLMC, scaleRadius=radiusLMC, outerCutoffRadius=radiusLMC*10, gamma=1, beta=3)
            
        pot_bulge = AGAMA_Spheroid(**paramBulge)
        pot_disk = MiyamotoNagaiDisk(m=paramDisk['mass'], a=paramDisk['scaleRadius'], b=paramDisk['scaleHeight'], units=units)
        pot_halo = AGAMA_Spheroid(**paramHalo)
        self.pot_MW = Potential_Combine([pot_bulge, pot_disk, pot_halo], units=units)
        
        self.pot_LMC = AGAMA_Spheroid(**paramLMC)
        self.translating_LMC_pot = TimeDepTranslatingPotential(pot=self.pot_LMC, center_spl=self.LMC_center_spline, units=units)
        self.unif_acc = UniformAcceleration(velocity_func=self.MW_velocity_func, units=units)
        self.total_pot = Potential_Combine([self.pot_MW, self.translating_LMC_pot, self.unif_acc], units=units)

    def LMC_center_spline(self, t):
        return jnp.array([self.LMC_x(t), self.LMC_y(t), self.LMC_z(t)])
        
    def MW_velocity_func(self, t):
        return jnp.array([self.velocity_func_x(t), self.velocity_func_y(t), self.velocity_func_z(t)])

    def gradient(self, xyz, t):
        return self.total_pot.gradient(xyz, t)
    
    def acceleration(self, xyz, t):
        return self.total_pot.acceleration(xyz, t)

    def potential(self, xyz, t):
        raise NotImplementedError("Potential not implemented, force is non-conservative")

# =============================================================================
# Subhalo Potentials
# =============================================================================

class SubhaloLinePotential(Potential):
    m: float
    a: float
    subhalo_x0: jnp.ndarray
    subhalo_v: jnp.ndarray
    subhalo_t0: float
    t_window: float

    def __init__(self, m, a, subhalo_x0, subhalo_v, subhalo_t0, t_window, units=usys):
        super().__init__(units)
        self.m = m
        self.a = a
        self.subhalo_x0 = jnp.asarray(subhalo_x0)
        self.subhalo_v = jnp.asarray(subhalo_v)
        self.subhalo_t0 = subhalo_t0
        self.t_window = t_window

    def single_subhalo_potential(self, xyz, m, a, t):
        return PlummerPotential(m=m, r_s=a, units=self.units).potential(xyz, t)

    def potential(self, xyz, t):
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v * (t - subhalo_t0))
            return self.single_subhalo_potential(relative_position, m, a, t)
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window 
        vmapped_cond = jax.vmap(jax.lax.cond, in_axes=((0, None, None, 0, 0, 0, 0, 0, None)))
        pot_per_subhalo = vmapped_cond(pred, true_func, false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, self.m, self.a, t)
        return jnp.sum(pot_per_subhalo)

    def potential_per_SH(self, xyz, t):
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v * (t - subhalo_t0))
            return self.single_subhalo_potential(relative_position, m, a, t)
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window 
        vmapped_cond = jax.vmap(jax.lax.cond, in_axes=((0, None, None, 0, 0, 0, 0, 0, None)))
        return vmapped_cond(pred, true_func, false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, self.m, self.a, t)

class SubhaloLinePotential_dRadius(Potential):
    m: float
    a: float
    subhalo_x0: jnp.ndarray
    subhalo_v: jnp.ndarray
    subhalo_t0: float
    t_window: float

    def __init__(self, m, a, subhalo_x0, subhalo_v, subhalo_t0, t_window, units=usys):
        super().__init__(units)
        self.m = m
        self.a = a
        self.subhalo_x0 = jnp.asarray(subhalo_x0)
        self.subhalo_v = jnp.asarray(subhalo_v)
        self.subhalo_t0 = subhalo_t0
        self.t_window = t_window

    def single_subhalo_potential(self, xyz, m, a, t):
        func = lambda m_val, r_s: PlummerPotential(m=m_val, r_s=r_s, units=self.units).potential(xyz, t)
        return jax.grad(func, argnums=(1))(m, a) 

    def potential(self, xyz, t):
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v * (t - subhalo_t0))
            return self.single_subhalo_potential(relative_position, m, a, t)
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window 
        vmapped_cond = jax.vmap(jax.lax.cond, in_axes=((0, None, None, 0, 0, 0, 0, 0, None)))
        pot_per_subhalo = vmapped_cond(pred, true_func, false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, self.m, self.a, t)
        return jnp.sum(pot_per_subhalo)

    def potential_per_SH(self, xyz, t):
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v * (t - subhalo_t0))
            return self.single_subhalo_potential(relative_position, m, a, t)
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window 
        vmapped_cond = jax.vmap(jax.lax.cond, in_axes=((0, None, None, 0, 0, 0, 0, 0, None)))
        return vmapped_cond(pred, true_func, false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, self.m, self.a, t)

class SubhaloLinePotential_Custom(Potential):
    pot: Potential
    subhalo_x0: jnp.ndarray
    subhalo_v: jnp.ndarray
    subhalo_t0: float
    t_window: float

    def __init__(self, pot, subhalo_x0, subhalo_v, subhalo_t0, t_window, units=usys):
        super().__init__(units)
        self.pot = pot
        self.subhalo_x0 = jnp.asarray(subhalo_x0)
        self.subhalo_v = jnp.asarray(subhalo_v)
        self.subhalo_t0 = subhalo_t0
        self.t_window = t_window

    def single_subhalo_potential(self, xyz, t):
        return self.pot.potential(xyz, t) 

    def potential(self, xyz, t):
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v * (t - subhalo_t0))
            return self.single_subhalo_potential(relative_position, t)
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window 
        vmapped_cond = jax.vmap(jax.lax.cond, in_axes=((0, None, None, 0, 0, 0, None)))
        pot_per_subhalo = vmapped_cond(pred, true_func, false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, t)
        return jnp.sum(pot_per_subhalo)

    def potential_per_SH(self, xyz, t):
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v * (t - subhalo_t0))
            return self.single_subhalo_potential(relative_position, t)
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window 
        vmapped_cond = jax.vmap(jax.lax.cond, in_axes=((0, None, None, 0, 0, 0, None)))
        return vmapped_cond(pred, true_func, false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, t)

class SubhaloLinePotential_dRadius_Custom(Potential):
    dpot_dRadius: Any
    subhalo_x0: jnp.ndarray
    subhalo_v: jnp.ndarray
    subhalo_t0: float
    t_window: float

    def __init__(self, dpot_dRadius, subhalo_x0, subhalo_v, subhalo_t0, t_window, units=usys):
        super().__init__(units)
        self.dpot_dRadius = dpot_dRadius
        self.subhalo_x0 = jnp.asarray(subhalo_x0)
        self.subhalo_v = jnp.asarray(subhalo_v)
        self.subhalo_t0 = subhalo_t0
        self.t_window = t_window

    def single_subhalo_potential(self, xyz, t):
        return self.dpot_dRadius(xyz, t) 

    def potential(self, xyz, t):
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v * (t - subhalo_t0))
            return self.single_subhalo_potential(relative_position, t)
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window 
        vmapped_cond = jax.vmap(jax.lax.cond, in_axes=((0, None, None, 0, 0, 0, None)))
        pot_per_subhalo = vmapped_cond(pred, true_func, false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, t)
        return jnp.sum(pot_per_subhalo)

    def potential_per_SH(self, xyz, t):
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v * (t - subhalo_t0))
            return self.single_subhalo_potential(relative_position, t)
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window 
        vmapped_cond = jax.vmap(jax.lax.cond, in_axes=((0, None, None, 0, 0, 0, None)))
        return vmapped_cond(pred, true_func, false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, t)

class SubhaloLinePotentialCustom_fromFunc(Potential):
    func: type = eqx.field(static=True)
    m: float
    r_s: float
    subhalo_x0: jnp.ndarray
    subhalo_v: jnp.ndarray
    subhalo_t0: float
    t_window: float

    def __init__(self, func, m, r_s, subhalo_x0, subhalo_v, subhalo_t0, t_window, units=usys):
        super().__init__(units)
        self.func = func
        self.m = m
        self.r_s = r_s
        self.subhalo_x0 = jnp.asarray(subhalo_x0)
        self.subhalo_v = jnp.asarray(subhalo_v)
        self.subhalo_t0 = subhalo_t0
        self.t_window = t_window

    def single_subhalo_potential(self, xyz, m, r_s, t):
        return self.func(m=m, r_s=r_s, units=self.units).potential(xyz, t)

    def potential(self, xyz, t):
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, m, r_s, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v * (t - subhalo_t0))
            return self.single_subhalo_potential(relative_position, m, r_s, t)
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, m, r_s, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window 
        vmapped_cond = jax.vmap(jax.lax.cond, in_axes=((0, None, None, 0, 0, 0, 0, 0, None)))
        pot_per_subhalo = vmapped_cond(pred, true_func, false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, self.m, self.r_s, t)
        return jnp.sum(pot_per_subhalo)

    def potential_per_SH(self, xyz, t):
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, m, r_s, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v * (t - subhalo_t0))
            return self.single_subhalo_potential(relative_position, m, r_s, t)
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, m, r_s, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window 
        vmapped_cond = jax.vmap(jax.lax.cond, in_axes=((0, None, None, 0, 0, 0, 0, 0, None)))
        return vmapped_cond(pred, true_func, false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, self.m, self.r_s, t)

class SubhaloLinePotentialCustom_dRadius_fromFunc(Potential):
    func: type = eqx.field(static=True)
    m: float
    r_s: float
    subhalo_x0: jnp.ndarray
    subhalo_v: jnp.ndarray
    subhalo_t0: float
    t_window: float

    def __init__(self, func, m, r_s, subhalo_x0, subhalo_v, subhalo_t0, t_window, units=usys):
        super().__init__(units)
        self.func = func
        self.m = m
        self.r_s = r_s
        self.subhalo_x0 = jnp.asarray(subhalo_x0)
        self.subhalo_v = jnp.asarray(subhalo_v)
        self.subhalo_t0 = subhalo_t0
        self.t_window = t_window

    def single_subhalo_potential(self, xyz, m, r_s, t):
        func_eval = lambda m_val, r_s_val: self.func(m=m_val, r_s=r_s_val, units=self.units).potential(xyz, t)
        return jax.grad(func_eval, argnums=(1))(m, r_s) 

    def potential(self, xyz, t):
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, m, r_s, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v * (t - subhalo_t0))
            return self.single_subhalo_potential(relative_position, m, r_s, t)
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, m, r_s, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window 
        vmapped_cond = jax.vmap(jax.lax.cond, in_axes=((0, None, None, 0, 0, 0, 0, 0, None)))
        pot_per_subhalo = vmapped_cond(pred, true_func, false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, self.m, self.r_s, t)
        return jnp.sum(pot_per_subhalo)

    def potential_per_SH(self, xyz, t):
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, m, r_s, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v * (t - subhalo_t0))
            return self.single_subhalo_potential(relative_position, m, r_s, t)
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, m, r_s, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window 
        vmapped_cond = jax.vmap(jax.lax.cond, in_axes=((0, None, None, 0, 0, 0, 0, 0, None)))
        return vmapped_cond(pred, true_func, false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, self.m, self.r_s, t)

# =============================================================================
# Helper Functions
# =============================================================================

@eqx.filter_jit
def interp_func(t, ind, stream_func):
    arr, narr = eqx.partition(stream_func, eqx.is_array)
    arr = jax.tree_util.tree_map(lambda x: x[ind], arr)
    interp = eqx.combine(arr, narr)
    return interp.evaluate(t)