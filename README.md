![streamsculptor](docs/streamsculptor_png.png) 

`streamsculptor`: perturbative and non-perturbative stream models implemented in Jax.

Supports CPU and GPU architectures, automatic differentiation, custom time-dependent potentials.

Auxillary ODEs can be computed along trajectories using the fields module.

**Example notebooks included in [streamsculptor/examples](https://github.com/jnibauer/streamsculptor/tree/main/streamsculptor/examples)**

Based on Nibauer et al. 2025: https://arxiv.org/abs/2410.21174 (see below for bibtex)


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

------
### Attribution
If you make use of this code, please cite the paper:

    @ARTICLE{2025ApJ...983...68N,
       author = {{Nibauer}, Jacob and {Bonaca}, Ana and {Spergel}, David N. and {Price-Whelan}, Adrian M. and {Greene}, Jenny E. and {Starkman}, Nathaniel and {Johnston}, Kathryn V.},
        title = "{StreamSculptor: Hamiltonian Perturbation Theory for Stellar Streams in Flexible Potentials with Differentiable Simulations}",
      journal = {\apj},
     keywords = {Stellar streams, Dark matter, Milky Way Galaxy, 2166, 353, 1054, Astrophysics - Astrophysics of Galaxies, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2025,
        month = apr,
       volume = {983},
       number = {1},
          eid = {68},
        pages = {68},
          doi = {10.3847/1538-4357/adb8e8},
    archivePrefix = {arXiv},
        eprint = {2410.21174},
    primaryClass = {astro-ph.GA},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2025ApJ...983...68N},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }


### License

Copyright (c) 2024–2026 Jacob Nibauer

`streamsculptor` is free software made available under the MIT License. For details, see the
[LICENSE](https://github.com/jnibauer/streamsculptor/blob/main/LICENSE) file.