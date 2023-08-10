import torch
from typing import Optional, Any, Generic, TypeVar, Type, List, Union, Tuple, Callable
from magrittetorch.model.parameters import Parameter
import numpy as np


class StorageTensor():
    """Dataset which stores a torch.Tensor
    """
    def __init__(self, dtype : torch.dtype, dims : List[Union[Parameter[int], int, None]], relative_storage_location : str, tensor : Optional[torch.Tensor] = None):
        """Initializes the storageTensor with the given datatype, required dimensions and storage location (for saving/loading). Optionally already sets the data to the given torch.Tensor.

        Args:
            dtype (torch.dtype): The data type of the stored data
            dims (List[Union[Parameter[int], int, None]]): The dimensions to check; None is used for not checking the dimension_
            relative_storage_location (str): The location at which this torch.Tensor is stored (relative to the model name)
            tensor (Optional[torch.Tensor]): _tensor to store and check dimensions of; if None, then dimensions are not checked. The tensor has to be filled in later on_
        """
        if tensor is not None:
            self.check_dims(tensor)
        self.relative_storage_location : str = relative_storage_location
        self.dtype = dtype
        self.tensor : Optional[torch.Tensor] = tensor
        self.dims : List[Union[Parameter[int], int, None]] = dims
        self.isSet : bool = False

    def check_dims(self, new_tensor : torch.Tensor) -> None:
        """Checks the dimensions of the stored tensor. Also checks type of data in tensor.

        Raises:
            AssertionError: If the data has not yet been set or the dimensions of the tensor do not correspond to the dimensions to check
            ValueError: If the dimension size is not equal to the set integer size
            TypeError: Dev error: If the dimensions to check contain wrong types
        """
        if new_tensor is None:
            raise AssertionError("The tensor has not yet been set")
        size : torch.Size = new_tensor.size()
        if len(size) != len(self.dims):
            raise AssertionError("The amount of dimensions of the tensor does not correspond to the amount of dimensions required. new_tensor dims: " + str(size) + " required: " + str(self.dims))
        for i, dim in zip(range(len(size)), self.dims):
            #check every option
            if dim is None:
                continue
            elif type(dim) is int:
                if size[i]!=dim:
                    raise ValueError("Dimension size is not equal tensor size." + str(size) + ", " + str(self.dims))
                continue
            elif type(dim) is Parameter:
                try:
                    dim.set(size[i])
                except ValueError as err:
                    raise AssertionError("Parameter dimension has changed. " + str(err))
            else:
                raise TypeError("Dev Error: storageTensor dims has wrong types. " + str(self.dims))
        if new_tensor.dtype != self.dtype:
            raise TypeError("The type of the stored data does not correspond to required datatype; required: ", self.dtype, " input dtype: ", new_tensor.dtype, " storage_location: ", self.relative_storage_location)
        self.isSet = True
        
    def set(self, tensor : torch.Tensor) -> None:
        """Sets the internal data to the given torch.Tensor if the dimensions are correct.

        Args:
            tensor (torch.Tensor): the new torch.Tensor

        Raises:
            AssertionError: If the data has not yet been set or the dimensions of the tensor do not correspond to the dimensions to check
            ValueError: If the dimension size is not equal to the set integer size
            TypeError: Dev error: If the dimensions to check contain wrong types
        """
        self.check_dims(tensor)
        self.tensor = tensor

    def get(self) -> torch.Tensor:
        """Returns the stored torch.Tensor

        Raises:
            ValueError: If no tensor has yet been stored

        Returns:
            torch.Tensor: the stored torch.Tensor
        """
        if self.tensor is None:
            raise ValueError("The data has not yet been set for " +str(self.relative_storage_location))
        return self.tensor
    
    def get_type(self) -> torch.dtype:
        """Returns the torch.dtype of the dataset

        Returns:
            torch.dtype: The data type
        """
        return self.dtype

    def is_complete(self) -> bool:
        """Returns whether the dataset is complete

        Returns:
            bool: Whether the dataset has already been set
        """
        return self.tensor is not None


#Maybe TODO: consider adding some ordering in inferring (maybe even a simple temporary skip if dependencies have not yet been inferred)
class InferredTensor():
    """Inferred dataset which stores a torch.Tensor. Has an automatic .infer() method to automatically set data after all storageTensors are set.
    """
    def __init__(self, dtype : torch.dtype, dims : List[Union[Parameter[int], int, None]], infer_function : Callable[[], torch.Tensor]):
        """Initializes the inferredTensor with the given datatype, required dimensions and function used for inferring its data.

        Args:
            dtype (torch.dtype): The data type of the stored data
            dims (List[Union[Parameter[int], int, None]]): The dimensions to check; None is used for not checking the dimension_
            infer_function (Callable[[] torch.Tensor]): function which returns the inferred data, if called after the data has been fully set
        """
        self.dtype = dtype
        self.infer_function : Callable[[], torch.Tensor] = infer_function
        self.tensor : Optional[torch.Tensor] = None
        self.dims : List[Union[Parameter[int], int, None]] = dims
        self.isSet : bool = False

    def check_dims(self, new_tensor : torch.Tensor) -> None:
        """Checks the dimensions of the stored tensor. Also checks type of data in tensor.

        Raises:
            AssertionError: If the data has not yet been set or the dimensions of the tensor do not correspond to the dimensions to check
            ValueError: If the dimension size is not equal to the set integer size
            TypeError: Dev error: If the dimensions to check contain wrong types
        """
        if new_tensor is None:
            raise AssertionError("The tensor has not yet been set")
        size : torch.Size = new_tensor.size()
        if len(size) != len(self.dims):
            raise AssertionError("The amount of dimensions of the tensor does not correspond to the amount of dimensions required. new_tensor dims: " + str(size) + " required: " + str(self.dims))
        for i, dim in zip(range(len(size)), self.dims):
            #check every option
            if dim is None:
                continue
            elif type(dim) is int:
                if size[i]!=dim:
                    raise ValueError("Dimension size is not equal tensor size." + str(size) + ", " + str(self.dims))
                continue
            elif type(dim) is Parameter:
                try:
                    dim.set(size[i])
                except ValueError as err:
                    raise AssertionError("Parameter dimension has changed. " + str(err))
            else:
                raise TypeError("Dev Error: storageTensor dims has wrong types. " + str(self.dims))
        if new_tensor.dtype != self.dtype:
            raise TypeError("The type of the stored data does not correspond to required datatype; required: ", self.dtype, " input dtype: ", new_tensor.dtype)
        self.isSet = True
        
    def set(self, tensor : torch.Tensor) -> None:
        """Sets the internal data to the given torch.Tensor if the dimensions are correct.

        Args:
            tensor (torch.Tensor): the new torch.Tensor

        Raises:
            AssertionError: If the data has not yet been set or the dimensions of the tensor do not correspond to the dimensions to check
            ValueError: If the dimension size is not equal to the set integer size
            TypeError: Dev error: If the dimensions to check contain wrong types
        """
        self.check_dims(tensor)
        self.tensor = tensor

    def infer(self) -> None:
        """Infers the data that should be in this structure. Only call after all storageTensors are complete.
        """
        self.set(self.infer_function())

    def get(self) -> torch.Tensor:
        """Returns the stored torch.Tensor

        Raises:
            ValueError: If no tensor has yet been stored

        Returns:
            torch.Tensor: the stored torch.Tensor
        """
        if self.tensor is None:
            raise ValueError("The data has not yet been set for an inferred tensor")
        return self.tensor
    
    def get_type(self) -> torch.dtype:
        """Returns the torch.dtype of the dataset

        Returns:
            torch.dtype: The data type
        """
        return self.dtype

    def is_complete(self) -> bool:
        """Returns whether the dataset is complete

        Returns:
            bool: Whether the dataset has already been set
        """
        return self.tensor is not None

# Some data (strings) cannot be put into pytorch tensors (and do not need to be put there)
class StorageNdarray():
    """Dataset which stores a numpy.ndarray.
    Only use this for data which cannot be stored in pytorch tensors (e.g. strings).
    """
    def __init__(self, type : np.dtype, dims : List[Union[Parameter[int], int, None]], relative_storage_location : str, array : Optional[np.ndarray[Any, Any]] = None):
        """Initializes the storageNdarray with the given datatype, required dimensions and storage location (for saving/loading). Optionally already sets the data to the given numpy.ndarray.

        Args:
            dtype (np.dtype): The data type of the stored data
            dims (List[Union[Parameter[int], int, None]]): The dimensions to check; None is used for not checking the dimension_
            relative_storage_location (str): The location at which this torch.Tensor is stored (relative to the model name)
            array (Optional[np.ndarray[Any, Any]]):  Array to store and check dimensions of; if None, then dimensions are not checked. The array has to be filled in later on_
        """
        if array is not None:
            self.check_dims(array)
        self.relative_storage_location : str = relative_storage_location
        self.type : np.dtype = type
        self.array : Optional[np.ndarray[Any, Any]] = array
        self.dims : List[Union[Parameter[int], int, None]] = dims
        self.isSet : bool = False

    def check_dims(self, new_array : np.ndarray) -> None:
        """Checks the dimensions of the stored array. Also checks type of data in array.

        Raises:
            AssertionError: If the data has not yet been set or the dimensions of the tensor do not correspond to the dimensions to check
            ValueError: If the dimension size is not equal to the set integer size
            TypeError: Dev error: If the dimensions to check contain wrong types
        """
        if new_array is None:
            raise AssertionError("The array has not yet been set")
        size : Tuple[int, ...] = np.shape(new_array)
        if len(size) != len(self.dims):
            raise AssertionError("The amount of dimensions of the tensor does not correspond to the amount of dimensions required. new_tensor dims: " + str(size) + " required: " + str(self.dims))
        for i, dim in zip(range(len(size)), self.dims):
            #check every option
            if dim is None:
                continue
            elif type(dim) is int:
                if size[i]!=dim:
                    raise ValueError("Dimension size is not equal to array size." + str(size) + ", " + str(self.dims))
                continue
            elif type(dim) is Parameter:
                try:
                    dim.set(size[i])
                except ValueError as err:
                    raise AssertionError("Parameter dimension has changed. " + str(err))
            else:
                raise TypeError("Dev Error: storageNdarray dims has wrong types. " + str(self.dims))
        if new_array.dtype != type:
            raise TypeError("The type of the stored array does not correspond to required datatype; required: ", self.type, " input dtype: ", new_array.dtype, " storage_location: ", self.relative_storage_location)
        self.isSet = True
        
    def set(self, array : np.ndarray[Any, Any]) -> None:
        """Sets the internal dataset to the given numpy array

        Args:
            array (np.ndarray[Any, Any]): The new numpy array
            
        Raises:
            AssertionError: If the data has not yet been set or the dimensions of the tensor do not correspond to the dimensions to check
            ValueError: If the dimension size is not equal to the set integer size
            TypeError: Dev error: If the dimensions to check contain wrong types
        """
        self.check_dims(array)
        self.array = array

    def get(self) -> np.ndarray[Any, Any]:
        """Returns the internal numpy array

        Raises:
            ValueError: If no numpy array has yet been stored

        Returns:
            np.ndarray[Any, Any]: The stored numpy array
        """
        if self.array is None:
            raise ValueError("The data has not yet been set for " +str(self.relative_storage_location))
        return self.array
    
    def get_type(self) -> np.dtype:
        """Returns the internal datatype of this dataset

        Returns:
            type: Returns the numpy.dtype of the stored numpy array
        """
        return self.type

    def is_complete(self) -> bool:
        """Returns whether the dataset is complete

        Returns:
            bool: Whether the dataset has already been set
        """
        return self.array is not None


#For convenience, use ducktyping to avoid needing to define superclasses for the datatypes
StorageTypes = Union[StorageTensor, StorageNdarray]
InferredTypes = InferredTensor
class DataCollection():
    """Collection of all data to store for a Magritte model
    """

    def __init__(self) -> None:
        """Initializes the DataCollection with an empty list of storageTypes
        """
        self.storedData : List[StorageTypes] = []
        self.inferredData : List[InferredTypes] = []

    def add_data(self, data : StorageTypes) -> None:
        """Adds a single dataset to the DataCollection

        Args:
            data (storageTypes): Dataset to add
        """
        self.storedData.append(data)
    
    def add_inferred_dataset(self, inferred_dataset : InferredTypes) -> None:
        self.inferredData.append(inferred_dataset)


    def is_data_complete(self) -> bool:
        """Checks whether every dataset in the collection is complete

        Returns:
            bool: Whether all datasets of this collection are complete
        """
        isCompletelist : List[Tuple[bool, str]] = [(data.is_complete(), data.relative_storage_location) for data in self.storedData]
        if (all([bool for bool, __ in isCompletelist])):
            return True
        print("Data is not yet fully set. Data which has not yet been set: ", [name for (cond, name) in isCompletelist if not cond])
        return False
    
    def infer_data(self) -> None:
        #safety check: check once more if the dataset is complete before wrongfully inferring some data
        if not self.is_data_complete():
            raise ValueError("Stored data has not yet been fully set. Therefore we cannot yet infer the other data.")
        #Dev note: inferred data might rely on other inferred data, so ordering of self.inferredData might be important. In general, I recommend to put the inferred datasets last in the constructor (in this way, the data of the subclass will be inferred first)
        for inferred_dataset in self.inferredData:
            inferred_dataset.infer()

    #TODO: maybe add inspection capabilities for figuring out for each parameter which dataset's dimensions are influenced; might be useful for debugging
    
class Types:
    """Contains expected types for tensor data. Summarized together such that tweaking them is more simple
    """
    GeometryInfo = torch.float64 #64 bit float # for positions, velocities, densities, temperatures
    IndexInfo = torch.int64 #64 bit signed int # for index information
    Enum = torch.int64 #64 bit signed int # for enums
    Bool = torch.bool #boolean # for truth values

