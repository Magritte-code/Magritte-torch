.. _link-quickstart:

Quickstart
##########

.. Warning::
    Please ensure that the :ref:`prerequisites <link-prerequisites>` are in place before continuing with the installation.


Setup
*****

Create a new `conda <https://www.anaconda.com/products/individual>`_ environment with:

.. code-block:: shell

    conda create -n magrittetorch

and activate it with:

.. code-block:: shell

    conda activate magrittetorch

This environment has to be active whenever Magrittetorch is installed or used!
Proceed by installing the latest version of Magrittetorch (including its python dependencies) with pip:

.. code-block:: shell

    pip install git+https://github.com/Magritte-code/Magritte-torch.git


Clone (optional)
****************

If you want to make changes to the source code, you can instead clone the repository using:

.. code-block:: shell

    git clone https://github.com/Magritte-code/Magritte-torch.git

Afterwards, go to the root directory of the cloned repository and install the package in editable mode:

.. code-block:: shell

    pip install -e .


Run
***

If all the above worked, go download and experiment with some of our :ref:`examples
<link-examples>`!