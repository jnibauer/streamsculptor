import jax.numpy as jnp
import jax

@jax.jit
def simcart_to_icrs(r_gc):
    alpha_gc = jnp.deg2rad(266.4051)
    delta_gc = jnp.deg2rad(-28.936175)
    d_gc = 8.122
    z_sun = 0.0208

    eta = jnp.deg2rad(58.5986320306)
    R1 = jnp.array([[jnp.cos(delta_gc), 0, jnp.sin(delta_gc)],
                    [0,1.0,0],
                    [-jnp.sin(delta_gc), 0, jnp.cos(delta_gc)]])
    R2 = jnp.array([[jnp.cos(alpha_gc), jnp.sin(alpha_gc), 0.0],
                    [-jnp.sin(alpha_gc), jnp.cos(alpha_gc), 0.0],
                    [0,0,1.]])
    R3 = jnp.array([[1.,0,0],
                   [0, jnp.cos(eta), jnp.sin(eta)],
                   [0,-jnp.sin(eta), jnp.cos(eta)]])

    R = jnp.matmul(R3, jnp.matmul(R1,R2))
    x_gc = jnp.array([1.0,0,0])


    theta = jnp.arcsin(z_sun/d_gc)
    H = jnp.array([[jnp.cos(theta), 0, jnp.sin(theta)],
                   [0,1,0],
                   [-jnp.sin(theta), 0, jnp.cos(theta)]])



    H_inv = jnp.linalg.inv(H)
    R_inv = jnp.linalg.inv(R)
    q_icrs = jnp.matmul(R_inv, jnp.matmul(H_inv,r_gc) + d_gc*x_gc)
    
    d = jnp.sqrt(jnp.sum(q_icrs**2))
    
    delta = jnp.arcsin(q_icrs[2]/d)
    
    alpha = jnp.arctan2(q_icrs[1],q_icrs[0])
    alpha = jnp.where(alpha>0, alpha, 2*jnp.pi + alpha)
    
  
    return jnp.rad2deg(alpha), jnp.rad2deg(delta), d

@jax.jit
def r_icrs_to_simcart(r_icrs):
    alpha_gc = jnp.deg2rad(266.4051)
    delta_gc = jnp.deg2rad(-28.936175)
    d_gc = 8.122
    z_sun = 0.0208

    eta = jnp.deg2rad(58.5986320306)
    R1 = jnp.array([[jnp.cos(delta_gc), 0, jnp.sin(delta_gc)],
                    [0,1.0,0],
                    [-jnp.sin(delta_gc), 0, jnp.cos(delta_gc)]])
    R2 = jnp.array([[jnp.cos(alpha_gc), jnp.sin(alpha_gc), 0.0],
                    [-jnp.sin(alpha_gc), jnp.cos(alpha_gc), 0.0],
                    [0,0,1.]])
    R3 = jnp.array([[1.,0,0],
                   [0, jnp.cos(eta), jnp.sin(eta)],
                   [0,-jnp.sin(eta), jnp.cos(eta)]])

    R = jnp.matmul(R3, jnp.matmul(R1,R2))
    x_gc = jnp.array([1.0,0,0])


    theta = jnp.arcsin(z_sun/d_gc)
    H = jnp.array([[jnp.cos(theta), 0, jnp.sin(theta)],
                   [0,1,0],
                   [-jnp.sin(theta), 0, jnp.cos(theta)]])

    
    return jnp.matmul(H, jnp.matmul(R,r_icrs) - d_gc*x_gc)


@jax.jit
def alpha_delta_to_simcart(alpha_delta_d):
    alpha, delta, d = alpha_delta_d
    r_icrs = jnp.array([d*jnp.cos(alpha)*jnp.cos(delta),
                       d*jnp.sin(alpha)*jnp.cos(delta),
                       d*jnp.sin(delta)])
    
    return r_icrs_to_simcart(r_icrs)

@jax.jit
def simcart_to_q_icrs(r_gc):
    alpha_gc = jnp.deg2rad(266.4051)
    delta_gc = jnp.deg2rad(-28.936175)
    d_gc = 8.122
    z_sun = 0.0208

    eta = jnp.deg2rad(58.5986320306)
    R1 = jnp.array([[jnp.cos(delta_gc), 0, jnp.sin(delta_gc)],
                    [0,1.0,0],
                    [-jnp.sin(delta_gc), 0, jnp.cos(delta_gc)]])
    R2 = jnp.array([[jnp.cos(alpha_gc), jnp.sin(alpha_gc), 0.0],
                    [-jnp.sin(alpha_gc), jnp.cos(alpha_gc), 0.0],
                    [0,0,1.]])
    R3 = jnp.array([[1.,0,0],
                   [0, jnp.cos(eta), jnp.sin(eta)],
                   [0,-jnp.sin(eta), jnp.cos(eta)]])

    R = jnp.matmul(R3, jnp.matmul(R1,R2))
    x_gc = jnp.array([1.0,0,0])


    theta = jnp.arcsin(z_sun/d_gc)
    H = jnp.array([[jnp.cos(theta), 0, jnp.sin(theta)],
                   [0,1,0],
                   [-jnp.sin(theta), 0, jnp.cos(theta)]])



    H_inv = jnp.linalg.inv(H)
    R_inv = jnp.linalg.inv(R)
    q_icrs = jnp.matmul(R_inv, jnp.matmul(H_inv,r_gc) + d_gc*x_gc)
    

    return q_icrs
    
@jax.jit
def ICRS_to_simcart(alpha, delta, dist, pm_ra_cosdec, pm_dec, rv):
    """
    Inverse of simvel_to_ICRS. Takes us from icrs to reflex corrected galactocentric frame
    inputs:
        alpha, delta [rad]
        dist [heliocentric, kpc]
        pm_ra_cosdec [NOT refelx corrected, mas/yr]
        pm_dec [NOT reflex corrected, mas/yr]
        rv [NOT reflex corrected, kpc/Myr]
    outputs:
        sim_q (position in kpc), sim_qdot (velocity in kpc/Myr, reflex corrected)
    """
    rad_Myr_to_mas_yr = 206.26480624709637
    Sun_reflex = jnp.array([0.01319299, 0.25117811, 0.0079567 ])  # kpc/Myr
    
    icrs_vec = jnp.array([alpha, delta, dist])
    X_galcen = jc.alpha_delta_to_simcart(icrs_vec)
    
    deriv = jax.jacfwd(jc.alpha_delta_to_simcart)(icrs_vec)
    alpha_hat = deriv[:,0]/jnp.linalg.norm(deriv[:,0])
    dec_hat = deriv[:,1]/jnp.linalg.norm(deriv[:,1])
    d_hat = deriv[:,2]/jnp.linalg.norm(deriv[:,2])
    
    

    V_galcen = dist*(pm_ra_cosdec / rad_Myr_to_mas_yr)*alpha_hat + dist*(pm_dec / rad_Myr_to_mas_yr)*dec_hat + rv*d_hat + Sun_reflex
    

    return X_galcen, V_galcen

@jax.jit
def simvel_to_ICRS(sim_q, sim_qdot):
    """
    Returns pmracosdec, pmdec, rv with simulation frame inputs.
    inputs in kpc, kpc/Myr. outputs in mas/yr and kpc/Myr.
    """
    rad_Myr_to_mas_yr = 206.26480624709637
    Sun_reflex = jnp.array([0.01319299, 0.25117811, 0.0079567 ]) # kpc/Myr
    alpha, delta, dist = simcart_to_icrs(sim_q)
    deriv = jax.jacfwd(alpha_delta_to_simcart)(jnp.deg2rad(jnp.array([alpha,delta,dist])))
    
    alpha_hat = deriv[:,0]/jnp.linalg.norm(deriv[:,0])
    dec_hat = deriv[:,1]/jnp.linalg.norm(deriv[:,1])
    d_hat = deriv[:,2]/jnp.linalg.norm(deriv[:,2])

    reflex_added = sim_qdot - Sun_reflex

    pm_ra_cosdec = jnp.sum((reflex_added*alpha_hat/dist))*rad_Myr_to_mas_yr
    pm_dec = jnp.sum((reflex_added*dec_hat/dist))*rad_Myr_to_mas_yr
    rv = jnp.sum(reflex_added*d_hat)
    
    return pm_ra_cosdec, pm_dec, rv
    
    