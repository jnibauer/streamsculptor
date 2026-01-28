![streamsculptor](docs/streamsculptor_png.png) 

`streamsculptor`: perturbative and non-perturbative stream models implemented in Jax.

Supports CPU and GPU architectures, automatic differentiation, custom time-dependent potentials.

Auxillary ODEs can be computed along trajectories using the fields module.



Based on Nibauer et al. 2025: https://arxiv.org/abs/2410.21174


### Installation
- `git clone https://github.com/jnibauer/streamsculptor.git`
- `cd streamsculptor`
- Install Jax using pip. 
    - For CPU usage: `pip install "jax==0.4.38"`
    - For GPU usage: streamsculptor sucessfully runs on cudatoolkit 12.0 with the Jax install `pip install --upgrade "jax[cuda12]==0.4.38"`
- Now install streamsculptor: `pip install .`
- For editable/development mode: `pip install -e .`





Optional installations:
- Agama (for interpolation of spheroidal density and potential)

------
### Common install issues
- Problem: `ForwardMode not found`
    - Solution: ensure you have installed Jax with a minimum version of 0.4.38, and diffrax with a minimum version of 0.6.2. Attempt imports again.