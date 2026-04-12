"""
Tests for TNFWSubhaloLinePotential (tnfw_new.py).

Verifies that the precomputed-coefficient implementation agrees with the
reference tNFWPotential (potential.py) to floating-point precision.

Run with:
    python tests/test_tnfw_subhalo_line_potential.py
"""
import traceback

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from streamsculptor.potential import tNFWPotential
from streamsculptor.tnfw_new import TNFWSubhaloLinePotential
from streamsculptor.main import usys


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Three subhalos with distinct, physically reasonable parameters
RHOS = jnp.array([1e7,  5e6,  2e7])   # Msun/kpc^3
R_S  = jnp.array([0.5,  1.0,  0.3])   # kpc
R_T  = jnp.array([2.0,  3.5,  1.2])   # kpc  (rt > rs required)
F_T  = jnp.array([0.8,  0.5,  0.95])  # dimensionless

SUBHALO_X0 = jnp.array([[10.0, 0.0, 0.0],
                          [0.0, 8.0, 2.0],
                          [-5.0, 3.0, -1.0]])   # kpc
SUBHALO_V  = jnp.array([[0.0,  0.1, 0.0],
                          [0.05, 0.0, 0.02],
                          [0.0, -0.05, 0.1]])    # kpc/Myr
SUBHALO_T0 = jnp.array([0.0, -50.0, 100.0])     # Myr

T_WINDOW = 200.0   # Myr — wide enough that all subhalos are active at t=0

# Test positions: a spread of radii to exercise the log/arctan terms
TEST_POINTS = jnp.array([
    [0.1,  0.0,  0.0],
    [1.0,  0.0,  0.0],
    [5.0,  0.0,  0.0],
    [0.3,  0.3,  0.3],
    [10.0, 2.0, -1.0],
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reference_phi(xyz, rhos, r_s, r_t, f_t, t):
    """Evaluate tNFWPotential directly — the unoptimized reference."""
    pot = tNFWPotential(rhos=rhos, r_s=r_s, r_t=r_t, f_t=f_t, units=usys)
    return pot.potential(xyz, t)


def build_line_potential():
    return TNFWSubhaloLinePotential(
        rhos=RHOS, r_s=R_S, r_t=R_T, f_t=F_T,
        subhalo_x0=SUBHALO_X0,
        subhalo_v=SUBHALO_V,
        subhalo_t0=SUBHALO_T0,
        t_window=T_WINDOW,
        units=usys,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSingleSubhaloPotential:
    """
    For each subhalo, compare TNFWSubhaloLinePotential.single_subhalo_potential
    against tNFWPotential at several radii.
    """

    def test_single_subhalo_matches_reference_all_subhalos_all_points(self):
        line_pot = build_line_potential()
        t = 0.0

        for i in range(len(RHOS)):
            c0 = line_pot._c0[i]
            a1 = line_pot._a1[i]; b1 = line_pot._b1[i]
            a2 = line_pot._a2[i]; b2 = line_pot._b2[i]
            a3 = line_pot._a3[i]; b3 = line_pot._b3[i]
            r_s = line_pot._r_s[i]; r_t = line_pot._r_t[i]

            for xyz in TEST_POINTS:
                fast = line_pot.single_subhalo_potential(
                    xyz, c0, a1, b1, a2, b2, a3, b3, r_s, r_t, t
                )
                ref = reference_phi(xyz, RHOS[i], R_S[i], R_T[i], F_T[i], t)
                assert jnp.allclose(fast, ref, rtol=1e-10, atol=0.0), (
                    f"Subhalo {i}, xyz={xyz}: fast={fast:.6e}, ref={ref:.6e}, "
                    f"rel_err={abs(fast-ref)/abs(ref):.2e}"
                )


class TestPotentialPerSH:
    """
    potential_per_SH should return the reference phi for active subhalos
    and exactly 0.0 for inactive ones.
    """

    def test_active_subhalos_match_reference(self):
        """All subhalos are active (|t - t0| < t_window). Check each."""
        line_pot = build_line_potential()
        t = 0.0
        phi_per_sh = line_pot.potential_per_SH(jnp.array([1.0, 0.5, -0.5]), t)

        for i in range(len(RHOS)):
            # position relative to subhalo i at t=0
            rel = jnp.array([1.0, 0.5, -0.5]) - (SUBHALO_X0[i] + SUBHALO_V[i] * (t - SUBHALO_T0[i]))
            ref = reference_phi(rel, RHOS[i], R_S[i], R_T[i], F_T[i], t)
            assert jnp.allclose(phi_per_sh[i], ref, rtol=1e-10, atol=0.0), (
                f"Subhalo {i}: fast={phi_per_sh[i]:.6e}, ref={ref:.6e}"
            )

    def test_inactive_subhalos_return_zero(self):
        """A subhalo outside its time window must contribute exactly 0."""
        line_pot = TNFWSubhaloLinePotential(
            rhos=RHOS[:1], r_s=R_S[:1], r_t=R_T[:1], f_t=F_T[:1],
            subhalo_x0=SUBHALO_X0[:1],
            subhalo_v=SUBHALO_V[:1],
            subhalo_t0=jnp.array([0.0]),
            t_window=10.0,   # narrow window
            units=usys,
        )
        # Evaluate well outside the window
        phi = line_pot.potential_per_SH(jnp.array([1.0, 0.0, 0.0]), t=500.0)
        assert phi[0] == 0.0, f"Expected 0.0 for inactive subhalo, got {phi[0]}"


class TestPotentialSum:
    """
    potential() is the sum over all subhalos; verify it equals the
    sum of individual reference evaluations.
    """

    def test_potential_sum_matches_reference_sum(self):
        line_pot = build_line_potential()
        xyz = jnp.array([2.0, -1.0, 0.5])
        t = 0.0

        fast_total = line_pot.potential(xyz, t)

        ref_total = sum(
            float(reference_phi(
                xyz - (SUBHALO_X0[i] + SUBHALO_V[i] * (t - SUBHALO_T0[i])),
                RHOS[i], R_S[i], R_T[i], F_T[i], t
            ))
            for i in range(len(RHOS))
        )

        assert jnp.allclose(fast_total, ref_total, rtol=1e-10, atol=0.0), (
            f"Sum: fast={fast_total:.6e}, ref={ref_total:.6e}, "
            f"rel_err={abs(fast_total - ref_total) / abs(ref_total):.2e}"
        )

    def test_potential_sum_at_multiple_times(self):
        """Check agreement across a range of evaluation times."""
        line_pot = build_line_potential()
        xyz = jnp.array([1.0, 1.0, 1.0])

        for t in [-100.0, -50.0, 0.0, 50.0, 100.0]:
            fast = float(line_pot.potential(xyz, t))

            ref = 0.0
            for i in range(len(RHOS)):
                dt = t - float(SUBHALO_T0[i])
                if abs(dt) < T_WINDOW:
                    rel = xyz - (SUBHALO_X0[i] + SUBHALO_V[i] * dt)
                    ref += float(reference_phi(rel, RHOS[i], R_S[i], R_T[i], F_T[i], t))

            assert abs(fast - ref) / (abs(ref) + 1e-30) < 1e-10, (
                f"t={t}: fast={fast:.6e}, ref={ref:.6e}"
            )


class TestFromInfall:
    """
    from_infall uses the Du+2024 tidal track. Verify that the resulting
    precomputed coefficients are consistent with tNFWPotential built from
    the same derived parameters.
    """

    def test_from_infall_matches_reference(self):
        from streamsculptor.tnfw_new import _tidally_evolved_nfw_params

        m_infall = jnp.array([1e8, 5e7])
        c_infall = jnp.array([15.0, 10.0])
        z_infall = jnp.array([1.0, 0.5])
        f_bound  = jnp.array([0.3, 0.7])

        x0 = jnp.zeros((2, 3))
        v  = jnp.zeros((2, 3))
        t0 = jnp.array([0.0, 0.0])

        line_pot = TNFWSubhaloLinePotential.from_infall(
            m_infall, c_infall, z_infall, f_bound,
            x0, v, t0, t_window=1000.0,
        )

        rhos, rs, ft, rt = _tidally_evolved_nfw_params(m_infall, c_infall, z_infall, f_bound)
        xyz = jnp.array([1.0, 0.0, 0.0])

        for i in range(2):
            fast = line_pot.potential_per_SH(xyz, t=0.0)[i]
            ref  = reference_phi(xyz, rhos[i], rs[i], rt[i], ft[i], t=0.0)
            assert jnp.allclose(fast, ref, rtol=1e-10, atol=0.0), (
                f"from_infall subhalo {i}: fast={fast:.6e}, ref={ref:.6e}"
            )


if __name__ == "__main__":
    test_classes = [
        TestSingleSubhaloPotential,
        TestPotentialPerSH,
        TestPotentialSum,
        TestFromInfall,
    ]
    passed = failed = 0
    for cls in test_classes:
        obj = cls()
        for name in [m for m in dir(cls) if m.startswith("test_")]:
            method = getattr(obj, name)
            try:
                method()
                print(f"  PASS  {cls.__name__}.{name}")
                passed += 1
            except Exception:
                print(f"  FAIL  {cls.__name__}.{name}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
