@jax.jit
def tnfw_enclosed_mass(R, rhos, rs, ft, rt):
    tau = rt / rs
    u = R / rs
    t2 = tau**2

    term1 = (t2 - 1) / (t2 + 1)**2 * jnp.log(1 + u)
    term2 = 1 / (t2 + 1) * (1/(1 + u) - 1)
    term3 = -(t2 - 1) / (2*(t2 + 1)**2) * jnp.log(1 + u**2/t2)
    term4 = 2*tau / (t2 + 1)**2 * jnp.arctan(u / tau)
    return 4 * jnp.pi * rhos * ft * rs**3 * t2 * (term1 + term2 + term3 + term4)

@jax.jit
def nfw_params_from_infall(m_infall, c_infall, z_infall,
                           H0=67.4, Omega_m=0.315, Omega_L=0.685):
    """
    compute rhos, rs from infall params
    Defaults to Planck 2018 cosmology
    """
    G = 4.498e-12  # kpc^3 / (Msun * Myr^2)
    H0_myr = H0 * 1e3 / 3.0856e22 * 3.15576e13  # km/s/Mpc -> Myr^-1
    Ez = jnp.sqrt(Omega_m * (1 + z_infall) ** 3 + Omega_L)
    H_z_myr = H0_myr * Ez
    rho_crit = 3 * H_z_myr ** 2 / (8 * jnp.pi * G)
    R200 = (3 * m_infall / (4 * jnp.pi * 200 * rho_crit)) ** (1 / 3)
    rs = R200 / c_infall
    rhos = m_infall / (4 * jnp.pi * rs ** 3 * (jnp.log(1 + c_infall) - c_infall / (1 + c_infall)))
    return rhos, rs, R200
@jax.jit
def tidally_evolved_nfw_params_from_infall(m_infall, c_infall, z_infall, f_bound):
    """
    Compute the density profile a tidally evolved (along tidal tracks) TNFW profile, given infall properties and
    bound mass fraction
    """
    rhos, rs, r200 = nfw_params_from_infall(m_infall, c_infall, z_infall)
    # Du et al. (2024) tidal track parameters, NFW (alpha=1, beta=3, gamma=1, delta=2)
    A, B, C = 0.68492777, 0.66438857, 2.07766512
    D, E = 0.75826635, 0.23376409
    ft = jnp.minimum((1 + D) * f_bound ** E / (1 + D * f_bound ** (2 * E)), 1.0)
    rt = (1 + A) * f_bound ** B / (1 + A * f_bound ** (2 * B)) / jnp.exp(C * (1 - f_bound)) * r200
    return rhos, rs, ft, rt

# Cannot compile this block alone, it returns an object
def make_tnfw_potential_from_infall(m_infall, c_infall, z_infall, f_bound):
    """
    Create a TNFW potential from infall properties
    """
    rhos, rs, ft, rt = tidally_evolved_nfw_params_from_infall(m_infall, c_infall, z_infall, f_bound)
    return make_tnfw_potential_from_density_profile(rhos, rs, ft, rt)

# Cannot compile this block alone, it returns an object
def make_tnfw_potential_from_density_profile(rhos, rs, ft, rt, n_grid=256):
    """
    Like get_potential_from_density but for a TNFW profile given
    rhos, rs, ft, rt directly. Returns the object with r_grid and phi_grid
    accessible for precomputation.
    """
    def tnfw_density(r):
        x = r / rs
        return rhos * ft / (x * (1 + x)**2 * (1 + (r / rt)**2))

    r_grid = jnp.logspace(jnp.log10(1e-4 * rs), jnp.log10(1e3 * rs), n_grid)
    return get_potential_from_density(density_func=tnfw_density, r_grid=r_grid, units=usys)



class TNFWSubhaloLinePotential(Potential):
    def __init__(self, m_infall, c_infall, z_infall, f_bound,
                 subhalo_x0, subhalo_v, subhalo_t0, t_window, units=None):
        """
        Initialize TNFW subhalo potentials from infall mass, concentration, infall redshift, and bound mass fraction
        The bound mass fraction is definied as the bound subhalo mass when it hits the stream
        Mass definition is m200 with respect to rho_crit(z_infall)
        """

        # calculate density profile parameters
        rhos, rs, ft, rt = tidally_evolved_nfw_params_from_infall(m_infall, c_infall, z_infall, f_bound)

        # 1. Define a function that computes the grids for a SINGLE subhalo
        def compute_single_grid(rho_s_i, r_s_i, f_t_i, r_t_i):
            pot_i = make_tnfw_potential_from_density_profile(
                rho_s_i, r_s_i, f_t_i, r_t_i, n_grid=256
            )
            return jnp.log(pot_i.r_grid), pot_i.phi_grid

        # 2. Vectorize the function across the 0th axis of all four input arrays
        vmap_compute = jax.vmap(compute_single_grid, in_axes=(0, 0, 0, 0))

        # 3. Execute in parallel (JAX automatically stacks the results into arrays of shape (n_subhalos, 256))
        log_r_grids, phi_grids = vmap_compute(rhos, rs, ft, rt)



        # # precompute potential grids for each subhalo at init time
        # n_subhalos = len(m_infall)
        # n_grid = 256
        # log_r_grids = jnp.zeros((n_subhalos, n_grid))
        # phi_grids = jnp.zeros((n_subhalos, n_grid))
        # for i in range(n_subhalos):
        #     pot_i = make_tnfw_potential_from_density_profile(
        #         rhos[i], rs[i], ft[i], rt[i], n_grid=n_grid
        #     )
        #     log_r_grids = log_r_grids.at[i].set(jnp.log(pot_i.r_grid))
        #     phi_grids = phi_grids.at[i].set(pot_i.phi_grid)

        super().__init__(units, {
            'subhalo_x0': subhalo_x0, 'subhalo_v': subhalo_v,
            'subhalo_t0': subhalo_t0, 't_window': t_window,
            'log_r_grids': log_r_grids, 'phi_grids': phi_grids,
        })

    @partial(jax.jit, static_argnums=(0,))
    def single_subhalo_potential(self, xyz, log_r_grid, phi_grid, t):
        r = jnp.sqrt(jnp.sum(xyz**2))
        func = interpax.Interpolator1D(x=log_r_grid, f=phi_grid, method='cubic')
        return func(jnp.log(r))

    @partial(jax.jit, static_argnums=(0,))
    def potential(self, xyz, t):
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, log_r_grid, phi_grid, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v * (t - subhalo_t0))
            return self.single_subhalo_potential(relative_position, log_r_grid, phi_grid, t)

        def false_func(subhalo_x0, subhalo_v, subhalo_t0, log_r_grid, phi_grid, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window

        vmapped_cond = jax.vmap(jax.lax.cond, in_axes=(0, None, None, 0, 0, 0, 0, 0, None))
        pot_per_subhalo = vmapped_cond(
            pred, true_func, false_func,
            self.subhalo_x0, self.subhalo_v, self.subhalo_t0,
            self.log_r_grids, self.phi_grids, t
        )
        return jnp.sum(pot_per_subhalo)

    @partial(jax.jit, static_argnums=(0,))
    def potential_per_SH(self, xyz, t):
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, log_r_grid, phi_grid, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v * (t - subhalo_t0))
            return self.single_subhalo_potential(relative_position, log_r_grid, phi_grid, t)

        def false_func(subhalo_x0, subhalo_v, subhalo_t0, log_r_grid, phi_grid, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window

        vmapped_cond = jax.vmap(jax.lax.cond, in_axes=(0, None, None, 0, 0, 0, 0, 0, None))
        return vmapped_cond(
            pred, true_func, false_func,
            self.subhalo_x0, self.subhalo_v, self.subhalo_t0,
            self.log_r_grids, self.phi_grids, t
        )