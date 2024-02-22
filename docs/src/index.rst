Magrittetorch documentation
###########################

Welcome to the Magrittetorch documentation! Magrittetorch is a PyTorch port of `Magritte <https://github.com/Magritte-code/Magritte>`_, an open-source software
library for simulating radiation transport. Magrittetorch has been developed at `KU Leuven
<https://www.kuleuven.be/english/>`_ (Belgium).

Magrittetorch is currently mainly used for post-processing hydrodynamical simulations by
creating synthetic observations, but allows for uncertainty quantification by utilizing the automatic differentiation capabilities of PyTorch.
Magrittetorch uses a deterministic ray-tracer with a formal solver that currently focusses on
line radiative transfer (see
`De Ceuster et al. 2019 <https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.1812D/abstract>`_, `De Ceuster et al. 2020 <https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.5194D/abstract>`_ and Ceulemans et al. (in prep.). 
for more details). 

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   0_getting_started/index
   1_examples/index
   2_benchmarks/index
   3_python_api_documentation/index
..  4_extra_options/index


Papers about Magritte(torch)
****************************

The following list of papers might provide further insights in the inner workings of
Magritte(torch):

* Magritte: **Adaptive ray-tracing, mesh construction and reduction**,
  *F. De Ceuster, J. Bolte, W. Homan, S. Maes, J. Malfait, L. Decin, J. Yates, P. Boyle, J. Hetherington*, 2020
  (`arXiv <https://arxiv.org/abs/2011.14998>`_,
  `MNRAS <https://doi.org/10.1093/mnras/staa3199>`_);

* Magritte: **Non-LTE atomic and molecular line modelling**,
  *F. De Ceuster, W. Homan, J. Yates, L. Decin, P. Boyle, J. Hetherington*, 2019
  (`arXiv <https://arxiv.org/abs/1912.08445>`_,
  `MNRAS <https://doi.org/10.1093/mnras/stz3557>`_);
  
* **3D Line Radiative Transfer & Synthetic Observations with Magritte**,
  *F. De Ceuster, T. Ceulemans, A. Srivastava, W. Homan, J. Bolte, J. Yates, L. Decin, P. Boyle, J., Hetherington*
  (`JOSS <https://doi.org/10.21105/joss.03905>`_).

Please note that some features presented in the Magritte papers are not implemented in Magrittetorch.


Issues & Contact
****************

Please report any `issues <https://github.com/Magritte-code/Magritte-torch/issues>`_ with
Magrittetorch or its documentation `here <https://github.com/Magritte-code/Magritte-torch/issues>`_.
If you need any further help, please contact `Thomas Ceulemans <https://thomasceulemans.github.io/>`_.


Developers & Contributors
*************************

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

Acknowledgements
****************
TC acknowledges support from the Research Foundation - Flanders (FWO) through the PhD Fellowship 1166722N.
