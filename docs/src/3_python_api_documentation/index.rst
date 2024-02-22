Python API documentation
########################

Every module in the MagritteTorch package is documented here. This contains all the classes and functions that are available to the user. 
Model setup can be done using the :class:`.Model` class and setting all necessary variables according to the :ref:`examples <link-examples>`.
The :py:mod:`.solvers` module contains high-level routines for line radiative transfer computations.
The :mod:`.plot` module aids with visualizing the results of the simulations.


Main operations on a model and utilities for model setup:

.. toctree::
    :maxdepth: 3

    magrittetorch.solvers
    magrittetorch.setup
    magrittetorch.mesher
    magrittetorch.plot



.. Note::

   In the documentation of the class variables, you might see the following notation: 
    - dtype: the datatype of the variable
    - dims: the dimensions of the variable, can be a list of (integers, parameter values) or None
    - unit: the astropy unit of the variable

Classes pertaining to the model data:

.. toctree::
    :maxdepth: 3

    magrittetorch.model
    magrittetorch.image
    magrittetorch.io
    magrittetorch.storagetypes
    magrittetorch.parameters
    magrittetorch.geometry
    magrittetorch.points
    magrittetorch.boundary
    magrittetorch.rays
    magrittetorch.chemistry
    magrittetorch.species
    magrittetorch.sources
    magrittetorch.lines
    magrittetorch.lineproducingspecies
    magrittetorch.linedata
    magrittetorch.collisionpartner



And then some utility functions and classes:

.. toctree::
    :maxdepth: 3

    magrittetorch.logger
    magrittetorch.memorymapping
    magrittetorch.timer



.. note::
   This API documentation only includes methods and classes that are part of the public API.
   You are free to use any other method of any class, but be aware that these might change in future releases.

