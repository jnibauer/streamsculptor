import jax
import jax.numpy as jnp
from astropy import units as u
import diffrax
import tqdm

# Ensure we import the refactored modules
from streamsculptor.main import usys
from streamsculptor import potential
import streamsculptor as ssc
from streamsculptor import perturbative as pert
from streamsculptor.GenerateImpactParams import ImpactGenerator

jax.config.update("jax_enable_x64", True)

def get_derivs(
                prog_wtoday: jnp.ndarray,
                t_age: float,
                t_dissolve: float,
                log10_min_mass: float,
                log10_max_mass: float,
                phi1_bounds: list,
                phi1_exclude: list,
                stream_seednum: int,
                key: jax.random.PRNGKey,
                Msat: float,
                target_num: int,
                phi1_function: callable, # takes N x 6 --> phi1 (length N)
                pot: callable,
                path: str = None,
                save: bool = True,
                r_s: float = 0.2,
                N_batch: int = 500,
                atol: float = 1e-11,
                rtol: float = 1e-11,
                bmax_fac: float = 5.0,
                phi1window: float = 0.5,
                N_arm: int = 5_000,
                save_iter_start: int = 0,
                self_grav: bool = True,
                ):

    """
    Generate perturbation derivatives to a stellar stream using subhalo encounters and save the results to disk.

    Parameters
    ----------
    prog_wtoday : jnp.ndarray
        Present-day phase-space coordinates (6D) of the progenitor.
    t_age : float
        Total age of the stream in Gyr (used to integrate backward). Must be positive.
    t_dissolve : float
        Time since dissolution began (Gyr). Must be negative.
    log10_min_mass : float
        Log10 of the minimum subhalo mass.
    log10_max_mass : float
        Log10 of the maximum subhalo mass.
    phi1_bounds : list
        List defining the angular bounds in phi1 coordinates for impact consideration.
    phi1_exclude : list
        List of phi1 regions to exclude from impacts (to avoid contaminated or unreliable areas).
    stream_seednum : int
        Random seed for generating the stream.
    key : jax.random.PRNGKey
        JAX PRNGKey used for subhalo sampling and random number generation.
    Msat : float
        Satellite (progenitor) mass.
    r_s : float
        Scale radius of the progenitor Plummer potential.
    target_num : int
        Total number of subhalos to simulate impacts with.
    phi1_function : callable
        Function that maps a stream (N x 6 array) to phi1 values (length N).
    pot : callable
        Potential object used for the base stream model. Must be compatible with streamsculptor.
    path : str, optional
        Directory path where output `.npy` files are saved. Required when ``save=True``.
    save : bool, optional
        If True (default), each batch is written to ``{path}/{i}.npy`` and the function
        returns None. If False, nothing is written to disk and the function returns a list
        of the per-batch dicts that would have been saved (keys: ``pert_out``,
        ``r_s_root``, ``ImpactFrameParams``).
    N_batch : int, optional
        Number of subhalos to process in each batch (default is 500).
    atol : float, optional
        Absolute tolerance for orbit integrator (default is 1e-11).
    rtol : float, optional
        Relative tolerance for orbit integrator (default is 1e-11).
    bmax_fac : float, optional
        Maximum impact parameter factor for computing impact bounds (default is 10.0).
    phi1window : float, optional
        Angular window (in phi1) around subhalos for considering impacts (default is 0.5).
    N_arm : int, optional
        Number of time samples for stream arm generation (default is 5000).
    save_iter_start : int, optional
        Starting index for saving output files (default is 0).
    self_grav : bool, optional
        If True (default), the progenitor feels its own self-gravity via a
        ``PlummerPotential(m=Msat, r_s=r_s)`` translating with the progenitor orbit -- both when
        generating the base stream and when building the base model. If False, the progenitor is
        treated as massless (no Plummer term), so the base stream and derivatives are integrated
        in ``pot`` alone. Use False to match a notebook/stream built without progenitor self-gravity.

    Returns
    -------
    None
        This function saves the perturbation outputs to disk and does not return any values.

    Notes
    -----
    - Perturbations are computed using an on-the-fly (OTF) method for performance and accuracy.
    - The output files contain subhalo parameters and the resulting stream perturbation data.
    - All computations use `jax` for automatic differentiation and parallel computation.
    """

    print('backend –– ' + str(jax.devices()[0]))

    if save and path is None:
        raise ValueError("path must be provided when save=True")
    results = [] if not save else None

    IC = pot.integrate_orbit(w0=prog_wtoday, t0=0.0, t1=-t_age, ts=jnp.array([-t_age])).ys[0]
    ts = jnp.hstack([jnp.linspace(-t_age, t_dissolve, N_arm), jnp.array([0.0])])

    # Progenitor self-gravity: a massive Plummer (self_grav=True) or a massless one (self_grav=False,
    # i.e. no self-gravity -> base stream and derivatives integrate in `pot` alone).
    prog_pot = potential.PlummerPotential(m=Msat if self_grav else 0.0, r_s=r_s, units=usys)

    print('generating base stream')
    l, t = ssc.gen_stream_vmapped_Chen25(pot_base=pot,
                                         prog_w0=IC,
                                         ts=ts,
                                         key=jax.random.PRNGKey(stream_seednum),
                                         Msat=Msat,
                                         atol=1e-7,
                                         rtol=1e-7,
                                         solver=diffrax.Dopri8(),
                                         prog_pot=prog_pot)

    stream = jnp.vstack([l, t])
    phi1_model = phi1_function(stream)

    mass_lin = 10**jnp.linspace(log10_min_mass, log10_max_mass, 50_000)
    prob = mass_lin**(-0.3)
    
    # How many iterations do we need to get to the target number of stars?
    N_iter = int(jnp.ceil(target_num / N_batch))

    # Define base model
    assert ts[-1] == 0.0 # Make sure the last time is 0.0
    print('setting base model')
    BaseModel = pert.BaseStreamModelChen25(
                            pot_base=pot,
                            ts=ts,
                            prog_w0=IC,
                            Msat=Msat,
                            key=jax.random.PRNGKey(stream_seednum),
                            units=usys,
                            prog_pot=prog_pot,
                            rtol=1e-7,
                            atol=1e-7,
                            solver=diffrax.Dopri8())

    keys = jax.random.split(key, N_iter)
    print('running perturbation loop for ' + str(N_iter) + ' iterations')
    
    for i in tqdm.tqdm(range(N_iter)):
        mass_samps = ssc.sample_from_1D_pdf(x=mass_lin, y=prob, key=keys[i], num_samples=N_batch)
        rs_samps = 1.05 * jnp.sqrt(mass_samps / 1e8)
        
        # Generate random int
        keynew = jax.random.split(keys[i], 1)[0]
        rand_int = int(jax.random.randint(key=keynew, shape=(1,), minval=0, maxval=10_000_000)[0])
        
        ImpactGen = ImpactGenerator(
                                pot=pot, 
                                tobs=0.0, 
                                stream=stream, 
                                stream_phi1=phi1_model, 
                                phi1_bounds=phi1_bounds,
                                tImpactBounds=[-t_age, 0.],
                                phi1window=phi1window,
                                NumImpacts=len(mass_samps),
                                bImpact_bounds=[jnp.zeros_like(mass_samps), 1.05 * jnp.sqrt(mass_samps / 1e8) * bmax_fac],
                                stripping_times=jnp.hstack([ts[:-1], ts[:-1]]),
                                phi1_exclude=phi1_exclude,
                                prog_today=prog_wtoday,
                                seednum=rand_int)
        
        ImpactDict = ImpactGen.get_subhalo_ImpactParams()
        assert jnp.isnan(ImpactDict['CartesianImpactParams'].sum()) == False # check for nans

        sub_pot = ssc.potential.SubhaloLinePotentialCustom_fromFunc(
                    func=potential.HernquistPotential, 
                    m=jnp.ones(mass_samps.shape[0]), 
                    r_s=rs_samps,
                    subhalo_x0=ImpactDict['CartesianImpactParams'][:, :3],
                    subhalo_v=ImpactDict['CartesianImpactParams'][:, 3:],
                    subhalo_t0=ImpactDict['ImpactFrameParams']['tImpact'],
                    t_window=150.0,
                    units=usys)

        pertgen = pert.GenerateMassRadiusPerturbation_Chen25(
                                           potential_base=pot,
                                           potential_perturbation=sub_pot,
                                           BaseStreamModel=BaseModel,
                                           units=usys)
                                           
        pert_out = pertgen.compute_perturbation_OTF(
                                            solver=diffrax.Dopri8(),
                                            rtol=rtol,
                                            atol=atol,
                                            dtmin=0.01, 
                                            cpu=False)

        dict_save = dict(pert_out=pert_out,
                        r_s_root=sub_pot.r_s,
                        ImpactFrameParams=ImpactDict['ImpactFrameParams'])

        if save:
            jnp.save(f"{path}/{i + save_iter_start}.npy", dict_save)
        else:
            results.append(dict_save)  # list keeps the references alive past the del below

        # Clean up memory (names only; the returned list still references dict_save/pert_out)
        del mass_samps, rs_samps, ImpactGen, ImpactDict, sub_pot, pertgen, pert_out, dict_save

    return results