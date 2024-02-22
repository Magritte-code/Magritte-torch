magrittetorch.model.parameters
==============================
For data consistency, we define some general :class:`.Parameter`'s which denote the shape of the data tensors.
For example, the number of points :attr:`.Parameters.npoints` in the model is fixed, and determines the size of the model positions, velocities, ...

.. autoclass:: magrittetorch.model.parameters.Parameter
    :members: get, set, name, value

.. autoclass:: magrittetorch.model.parameters.EnumParameter
    :members:
    :show-inheritance:

.. autoclass:: magrittetorch.model.parameters.Parameters
    :members:
    :undoc-members: