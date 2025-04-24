from streamsculptor import potential
from functools import partial
from astropy.constants import G
import astropy.coordinates as coord
import astropy.units as u
# gala
from gala.units import dimensionless, UnitSystem

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from diffrax import diffeqsolve, ODETerm, Dopri5,SaveAt,PIDController,DiscreteTerminatingEvent, DirectAdjoint, RecursiveCheckpointAdjoint, ConstantStepSize, Euler, StepTo
import diffrax
import equinox as eqx
import interpax
import streamsculptor as ssc

from streamsculptor.GenerateImpactParams import ImpactGenerator


@partial(jax.jit,static_argnames=('BaseStreamModel','RateCalculator','Max_Num_Impacts'))
def gen_perturbed_stream(BaseStreamModel: object,
                         RateCalculator: object,
                         unperturbed_stream: jnp.array,
                         phi1_unperturbed: jnp.array,
                         log10M_min: float,
                         log10M_max: float,
                         key: jnp.array,
                         phi1_bounds: list,
                         tImpactBounds: list,
                         phi1_exclude = [1.,1.],
                         phi1window = 1.,
                         M_hm= 0.0, 
                         normalization = 1.0, 
                         slope = -1.9,
                         Max_Num_Impacts = 10,
                         rtol= 1e-7,
                         atol= 1e-7,
                         mass_fac = 1.0,
                         r_s_fac = 1.0,
        ):

    ts = BaseStreamModel.ts
    prog_w0 = BaseStreamModel.prog_w0
    prog_pot = BaseStreamModel.prog_pot
    pot_base = BaseStreamModel.pot_base
    Msat = BaseStreamModel.Msat
       
    sample_dict = RateCalculator.sample_masses(log10M_min=log10M_min,log10M_max=log10M_max,key=key, M_hm=M_hm, normalization=normalization,slope=slope, array_length=Max_Num_Impacts)


    mass_vals = (10**sample_dict['log10_mass'])*mass_fac
    mass_vals = jnp.where(sample_dict['log10_mass']>0, mass_vals, 0.0)
    r_s_vals = 1.05 * jnp.sqrt(mass_vals / 1e8) * r_s_fac
    r_s_vals  = jnp.where(r_s_vals==0.0, 1.0, r_s_vals)
    b_high = RateCalculator.b_max_fac * r_s_vals
    b_low = b_high * 0 
    b_bounds = jnp.vstack([b_low,b_high]).T
    
    key1, key2 = jax.random.split(key, 2)
    seednum = jax.random.randint(key=key1,shape=(1,),minval=0,maxval=10_000)[0]
    tstrip_stack = jnp.hstack([ts[:-1],ts[:-1]])
    ImpactGen = ImpactGenerator(pot=pot_base, 
                                tobs=ts.max(), 
                                stream=unperturbed_stream, 
                                stream_phi1=phi1_unperturbed, 
                                phi1_bounds=phi1_bounds,
                                tImpactBounds=tImpactBounds,
                                phi1window=phi1window,
                                NumImpacts=len(mass_vals),
                                bImpact_bounds=b_bounds,
                                stripping_times=tstrip_stack,
                                phi1_exclude=phi1_exclude,
                                seednum=seednum)
    
    ImpactParams = ImpactGen.get_subhalo_ImpactParams()
    
    out = dict(cartImpact=ImpactParams['CartesianImpactParams'],
                tImpact=ImpactParams['ImpactFrameParams']['tImpact'],
                mass=mass_vals,
                r_s=r_s_vals)
    
    Hernquist = potential.HernquistPotential
    pot_subhalos = potential.SubhaloLinePotentialCustom_fromFunc(
                        func=Hernquist, 
                        m=out['mass'], 
                        r_s=out['r_s'],
                        subhalo_x0=out['cartImpact'][:,:3], 
                        subhalo_v=out['cartImpact'][:,3:],
                        subhalo_t0=out['tImpact'], 
                        t_window=100.0,
                        units=pot_base.units)
    
    l_pert, t_pert = ssc.gen_stream_vmapped_with_pert_Chen25_fixed_prog(  
                                                pot_base=pot_base, 
                                                pot_pert=pot_subhalos,
                                                prog_pot=prog_pot,
                                                prog_w0=prog_w0,
                                                ts=ts,
                                                Msat=Msat,
                                                key=key2,
                                                solver=diffrax.Dopri8(),
                                                atol=atol,
                                                rtol=rtol,
                                                dtmin=0.05
                                                )    

    return dict(leading=l_pert, trailing=t_pert, impactors=out)
