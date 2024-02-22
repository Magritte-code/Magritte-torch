.. _link-prerequisites:

Prerequisites
#############

Throughout this documentation, we assume a Unix-based operating system (e.g. Linux or MacOS).
For Windows users, we recommend to use the Windows Subsystem Linux (WSL; see e.g. `here <https://learn.microsoft.com/en-us/windows/wsl/install>`_).

Below is a list of the most important python packages required to be able to install Magrittetorch.

* `PyTorch <https://pytorch.org/>`_ (version :literal:`2.0.0` or later): for GPU parallelisation and automatic differentiation;
* `Astropy <https://www.astropy.org/>`_: for `units <https://docs.astropy.org/en/stable/units/>`_ and astronomy related file IO;
* `Anaconda <https://www.anaconda.com/blog/individual-edition-2020-11>`_: for managing the required Python packages;
* `Pip <https://pip.pypa.io/en/stable/>`_: for installing Magrittetorch;

And software required for GPU parallelisation (optional, depending on the GPU vendor):

* `CUDA <https://developer.nvidia.com/cuda-zone>`_: for Nvidia GPU's;
* `ROCm <https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html>`_: for AMD GPU's;

The comprehensive list of required Python packages can be found in the `pyproject.toml file <https://github.com/Magritte-code/Magritte-torch/blob/main/pyproject.toml>`_.

Once Anaconda, Pip and (optionally) CUDA or ROCm are in place, Magrittetorch can be installed following the :ref:`quickstart <link-quickstart>`.

.. note::
    Magrittetorch can also be run on CPU. For this, the GPU related dependencies (CUDA, ROCm) are not required.
    However, computation times will be significantly longer, when compared to compiled C++ code `Magritte <https://github.com/Magritte-code/Magritte>`_.
