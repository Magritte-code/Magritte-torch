<img src="docs/src/images/Magrittetorch_logo_no_background.png" alt="logo">

Welcome to Magrittetorch, a modern software library for simulating radiation transport, using pytorch as backend for GPU computations.
This is a port of [Magritte](https://github.com/Magritte-code/Magritte) to python, to allow for a redesign of the parallelization scheme.
Magrittetorch is currently mainly used for post-processing hydrodynamical simulations of astrophysical models by creating synthetic observations.

## Features
Magrittetorch can be used for 3D (N)LTE line radiative transfer and imaging. The software library supports both CPU and GPU parallelization, using pytorch as backend. Model input can be done using astropy Quantities to allow for effortless unit conversions.


## Installation
Magrittetorch is entirely written in python, and therefore pip installable. 
```console
pip install git+https://github.com/Magritte-code/Magritte-torch.git
```
Find the complete installation instructions in the [documentation](https://magritte-torch.readthedocs.io/en/latest/0_getting_started/index.html)

## Documentation
The documentation of Magrittetorch can be found [here](https://magritte-torch.readthedocs.io).
See the [examples](https://magritte-torch.readthedocs.io/en/latest/1_examples/index.html) in the
documentation to learn how to use Magrittetorch to create synthetic observations.

## Performance comparison
TODO: compare CPU performance vs Magritte, compare GPU performance, scaling behavior

## Issues & Contact
Please report any issues with Magritte-torch [here](https://github.com/Magritte-code/Magritte-torch/issues). If you need any further help, please contact [Thomas Ceulemans](https://thomasceulemans.github.io/).

## Cite
Please contact the authors of the papers referenced above if you want to use
Magrittetorch in your research. We are currently working on documentation and
examples to facilitate its independent use. Until then, please
[contact us](https://thomasceulemans.github.io/).

## Developers & Contributors
**Developers**
* Thomas Ceulemans

**Scientific & Technical advisors**
* Frederik De Ceuster
* Leen Decin

**Contributors**
* Silke Maes
* Jolien Malfait
* Mats Esseldeurs
* Arnout Coenegrachts
* Owen Vermeulen

# Other sections
TODO: installation (this library will be written entirely in python, so pip/conda install should be possible). These sections will be added when magritte-torch is sufficiently developed.
