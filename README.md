![streamsculptor](docs/streamsculptor_png.png) 

StreamSculptor: perturbative and non-perturbative stream models implemented in Jax.

Supports CPU and GPU architectures, automatic differentiation, custom time-dependent potentials.

Auxillary ODEs can be computed along trajectories using the fields module.



Based on Nibauer et al. 2024: https://arxiv.org/abs/2410.21174


### Requirements
Listed in requirements.txt 

Required installations:
- Jax (for backends)
    - Note: `pip install jax` will assume cpu usage. Please see the Jax documentation for the appropriate install if you would like to run streamsculptor on gpu.
- Diffrax [version $\geq$ 0.6.2] (for differentiable numerical integration)
- Equinox (class structure)
- Interpax (spline interpolation)
- Gala (for unitsystem module)
- quadax (Gaussian quadrature)

Optional installations:
- Jaxopt and Optax (for optimization of restricted N-body)
- Agama (for interpolation of spheroidal density and potential)

### Installation
After installing the prerequisites:
- `git clone https://github.com/jnibauer/streamsculptor.git`
- `cd streamsculptor`
- `python setup.py install`


------
### Common install issues
- Problem: `ForwardMode not found`
    - Solution: ensure you have installed Jax with a minimum version of 0.4.38, and diffrax with a minimum version of 0.6.2. Attempt imports again.