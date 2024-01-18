<img src="docs/src/images/Magrittetorch_logo_no_background.png" alt="logo">

Welcome to Magrittetorch, a modern software library for simulating radiation transport, using pytorch as backend for GPU computations.
This is a port of [Magritte](https://github.com/Magritte-code/Magritte) to python, to allow for a redesign of the parallelization scheme.
Magrittetorch is currently mainly used for post-processing hydrodynamical simulations of astrophysical models by creating synthetic observations.

# Features
Magrittetorch can be used for 3D (N)LTE line radiative transfer and imaging. The software library supports both CPU and GPU parallelization, using pytorch as backend. Model input can be done using astropy Quantities to allow for effortless unit conversions.

# Documentation
The documentation of Magrittetorch can be found at TODO add readthedocs url, but is still a work in progress.
As Magrittetorch is a port, one of its goals is to reuse the significant parts of the API of [Magritte](https://github.com/Magritte-code/Magritte), to allow for seamless transition to this library. Take a look at the [documentation](https://magritte.readthedocs.io) of Magritte instead for now.

# Performance comparison
TODO: compare CPU performance vs Magritte, compare GPU performance, scaling behavior

# Issues & Contact
Please report any issues with Magritte-torch [here](https://github.com/Magritte-code/Magritte-torch/issues). If you need any further help, please contact [Thomas Ceulemans](https://thomasceulemans.github.io/).

# Other sections
TODO: contributers, acknowledgements, citation, installation (this library will be written entirely in python, so pip/conda install should be possible). These sections will be added when magritte-torch is sufficiently developed.
