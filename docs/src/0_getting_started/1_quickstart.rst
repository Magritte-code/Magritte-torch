.. _link-quickstart:

Quickstart
##########

.. Warning::
    Please ensure that the :ref:`prerequisites <link-prerequisites>` are in place before continuing with the installation.

Clone
*****

For now, Magrittetorch has to be installed from source, but might be available using pip in the future.
Magrittetorch can be installed from its source code, which can be cloned using:

.. code-block:: shell

    git clone https://github.com/Magritte-code/Magritte-torch.git

from our `GitHub <https://github.com/Magritte-code/Magritte-torch>`_ repository.


Setup
*****

Create a new `conda <https://www.anaconda.com/products/individual>`_ environment with:

.. code-block:: shell

    conda create -n magrittetorch

and activate it with:

.. code-block:: shell

    conda activate magrittetorch

This environment has to be active whenever Magrittetorch is installed or used!
Proceed by installing Magrittetorch (including its python dependencies) with pip, by using the following command from the root of the cloned repository:

.. code-block:: shell

    pip install .


Run
***

If all the above worked, go download and experiment with some of our :ref:`examples
<link-examples>`!
