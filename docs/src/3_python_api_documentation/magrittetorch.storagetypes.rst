magrittetorch.utils.storagetypes
================================

This module contains classes that are used to store data in a way that is consistent and easy to setup.
The main takeaway is that all :class:`.StorageTensor`, :class:`.StorageNdarray`, :class:`.InferredTensor` have fixed dimensions and data types, which are checked on data setting.
In order to simplify the setup of all intermediate inferred data, the :class:`.DataCollection` is used.

.. autoclass:: magrittetorch.utils.storagetypes.DataCollection
    :members: infer_data, is_data_complete

.. autoclass:: magrittetorch.utils.storagetypes.DelayedListOfClassInstances
    :members: get
    :special-members: __getitem__

.. autoclass:: magrittetorch.utils.storagetypes.InferredTensor
    :members: get, get_astropy, get_type, is_complete, set, set_astropy, infer
    :special-members: __init__

.. autoclass:: magrittetorch.utils.storagetypes.InferredTensorPerNLTEIter
    :members: get, get_astropy, get_type, is_complete, set, set_astropy, infer
    :special-members: __init__

.. autoclass:: magrittetorch.utils.storagetypes.StorageNdarray
    :members: get, get_astropy, get_type, is_complete, set, set_astropy
    :special-members: __init__

.. autoclass:: magrittetorch.utils.storagetypes.StorageTensor
    :members: get, get_astropy, get_type, is_complete, set, set_astropy
    :special-members: __init__

.. autoclass:: magrittetorch.utils.storagetypes.Types
    :members:


.. .. automodule:: magrittetorch.utils.storagetypes
..     :members: 