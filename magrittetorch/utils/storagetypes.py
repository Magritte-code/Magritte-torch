import torch
from typing import Optional, Any, Generic, TypeVar, Type, List, Union, Tuple, Callable, TypeAlias, Dict, Iterator
from magrittetorch.model.parameters import Parameter, Parameters
import numpy as np
from astropy import units
import astropy


T = TypeVar('T')

class StorageTensor():
    """Dataset which stores a torch.Tensor
    """
    def __init__(self, dtype : torch.dtype, dims : List[Union[Parameter[int], int, None]], unit : units.Unit, relative_storage_location : str, legacy_converter : Optional[Tuple[str, Optional[Callable[[Any], torch.Tensor]]]] = None, tensor : Optional[torch.Tensor] = None):
        """Initializes the storageTensor with the given datatype, required dimensions and storage location (for saving/loading). Optionally already sets the data to the given torch.Tensor.

        Args:
            dtype (torch.dtype): The data type of the stored data
            dims (List[Union[Parameter[int], int, None]]): The dimensions to check; None is used for not checking the dimension_
            relative_storage_location (str): The location at which this torch.Tensor is stored (relative to the model name)
            legacy_converter (Tuple[str, Optional[Callable[[Any], torch.Tensor]]): The location at which this torch.Tensor is stored in C++ magritte (relative to the model name) + optionally a method to convert the data
            tensor (Optional[torch.Tensor]): tensor to store and check dimensions of; if None, then dimensions are not checked. The tensor has then to be filled in later on
        """
        if tensor is not None:
            self.check_dims(tensor)
        self.relative_storage_location : str = relative_storage_location

        #some storage locations/data might be different for C++ magritte and python magritte, thus we need some conversion for legacy functionality
        self.legacy_relative_storage_location : str = self.relative_storage_location
        self.legacy_conversion_function : Optional[Callable[[Any], torch.Tensor]] = None #function to apply after reading the data
        if legacy_converter is not None:
            self.legacy_relative_storage_location = legacy_converter[0]
            self.legacy_conversion_function = legacy_converter[1]

        self.dtype = dtype
        self.unit: units.Unit = unit
        self.tensor: Optional[torch.Tensor] = tensor
        self.tensormap: Dict[torch.device, torch.Tensor] = {}
        self.dims: List[Union[Parameter[int], int, None]] = dims
        self.isSet: bool = False

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
            elif (type(dim) is Parameter):
                try:
                    dim.set(size[i])
                except ValueError as err:
                    raise AssertionError("Parameter dimension has changed. " + str(err))
            else:
                raise TypeError("Dev Error: storageTensor dims has wrong types. " + str(self.dims))
        if new_tensor.dtype != self.dtype:
            raise TypeError("The type of the stored data does not correspond to required datatype; required: ", self.dtype, " input dtype: ", new_tensor.dtype, " storage_location: ", self.relative_storage_location)
        self.isSet = True
        
    def set_astropy(self, astropy_quantity : units.Quantity) -> None:
        """Sets the internal data to the given astropy Quantity if the dimensions are correct. Automatically converts units

        Args:
            astropy_array (units.Quantity): _description_

        Raises:
            astropy.units.core.UnitConversionError: when the units are incompatible
        """
        self.set(torch.from_numpy(np.array(astropy_quantity.to(self.unit))))
        
    def set(self, tensor : torch.Tensor) -> None:
        """Sets the internal data to the given torch.Tensor if the dimensions are correct. Does not automatically convert units.
        Invalidates mapped references to the data, as the internal data has changed.

        Args:
            tensor (torch.Tensor): the new torch.Tensor

        Raises:
            AssertionError: If the data has not yet been set or the dimensions of the tensor do not correspond to the dimensions to check
            ValueError: If the dimension size is not equal to the set integer size
            TypeError: Dev error: If the dimensions to check contain wrong types
        """
        self.check_dims(tensor)
        self.tensor = tensor
        #also invalidate mapped tensors, as the data is no longer correct
        self.tensormap = {}

    def get_astropy(self) -> units.Quantity:
        """Returns the stored astropy Quantity

        Raises:
            ValueError: If no data has yet been stored

        Returns:
            units.Quantity: the stored astropy Quantity
        """
        return self.get().numpy(force=True)*self.unit


    def get(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Returns the stored torch.Tensor. Optionally returns a reference to the tensor on the specified device, without remapping if already mapped to that device.

        Args:
            device (Optional[torch.device]): if None, return the tensor on cpu. Else, return a reference to the tensor on this device.

        Raises:
            ValueError: If no data has yet been stored

        Returns:
            torch.Tensor: the stored torch.Tensor
        """
        if self.tensor is None:
            raise ValueError("The data has not yet been set for " +str(self.relative_storage_location))
        #if device is specified, grab the reference to the already mapped tensor
        if device is not None:
            #map to device if not already present
            if device not in self.tensormap:
                self.tensormap[device] = self.tensor.to(device)
            return self.tensormap[device]
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
    def __init__(self, dtype : torch.dtype, dims : List[Union[Parameter[int], int, None]], unit : units.Unit, infer_function : Callable[[], torch.Tensor], relative_storage_location : Optional[str] = None, legacy_converter : Optional[Tuple[str, Optional[Callable[[Any], torch.Tensor]]]] = None):
        """Initializes the inferredTensor with the given datatype, required dimensions and function used for inferring its data.

        Args:
            dtype (torch.dtype): The data type of the stored data
            dims (List[Union[Parameter[int], int, None]]): The dimensions to check; None is used for not checking the dimension_
            infer_function (Callable[[] torch.Tensor]): function which returns the inferred data, if called after the data has been fully set
            legacy_converter (Optional[Tuple[str, Optional[Callable[[Any], torch.Tensor]]]]): The location at which this torch.Tensor is stored in C++ magritte (relative to the model name) + optionally a method to convert the data
            relative_storage_location (Optional[str]): Location where to store/read the tensor; if None, then the data will not be stored
        """
        self.dtype = dtype
        self.unit : units.Unit = unit
        self.infer_function : Callable[[], torch.Tensor] = infer_function
        self.relative_storage_location : Optional[str] = relative_storage_location
        #some storage locations/data might be different for C++ magritte and python magritte, thus we need some conversion for legacy functionality
        self.legacy_relative_storage_location = self.relative_storage_location
        self.legacy_conversion_function : Optional[Callable[[Any], torch.Tensor]] = None #function to apply after reading the data
        if legacy_converter is not None:
            self.legacy_relative_storage_location = legacy_converter[0]
            self.legacy_conversion_function = legacy_converter[1]

        self.tensor : Optional[torch.Tensor] = None
        self.tensormap: Dict[torch.device, torch.Tensor] = {}
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

    def set_astropy(self, astropy_quantity : units.Quantity) -> None:
        """Sets the internal data to the given astropy Quantity if the dimensions are correct. Automatically converts units

        Args:
            astropy_array (units.Quantity): _description_

        Raises:
            astropy.units.core.UnitConversionError: when the units are incompatible
        """
        self.set(torch.from_numpy(np.array(astropy_quantity.to(self.unit))))
        
    def set(self, tensor : torch.Tensor) -> None:
        """Sets the internal data to the given torch.Tensor if the dimensions are correct. Does not automatically convert units.
        Invalidates mapped references to the data, as the internal data has changed.

        Args:
            tensor (torch.Tensor): the new torch.Tensor

        Raises:
            AssertionError: If the data has not yet been set or the dimensions of the tensor do not correspond to the dimensions to check
            ValueError: If the dimension size is not equal to the set integer size
            TypeError: Dev error: If the dimensions to check contain wrong types
        """
        self.check_dims(tensor)
        self.tensor = tensor
        #also invalidate mapped tensors, as the data is no longer correct
        self.tensormap = {}

    def infer(self) -> None:
        """Infers the data that should be in this structure if not yet set. Only call after all storageTensors are complete.
        """
        if self.tensor is None:
            self.set(self.infer_function())

    def _force_infer(self) -> None:
        """Internal function: sets the data in this structure according to the infer_function. Only call after all storageTensors are complete.
        """
        self.set(self.infer_function())

    def get_astropy(self) -> units.Quantity:
        """Returns the stored astropy Quantity

        Raises:
            ValueError: If no data has yet been stored

        Returns:
            units.Quantity: the stored astropy Quantity
        """
        return self.get().numpy(force=True)*self.unit

    def get(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Returns the stored torch.Tensor. Optionally returns a reference to the tensor on the specified device, without remapping if already mapped to that device.

        Args:
            device (Optional[torch.device]): if None, return the tensor on cpu. Else, return a reference to the tensor on this device.

        Raises:
            ValueError: If no data has yet been stored

        Returns:
            torch.Tensor: the stored torch.Tensor
        """
        if self.tensor is None:
            raise ValueError("The data has not yet been set for " +str(self.relative_storage_location))
        #if device is specified, grab the reference to the already mapped tensor
        if device is not None:
            #map to device if not already present
            if device not in self.tensormap:
                self.tensormap[device] = self.tensor.to(device)
            return self.tensormap[device]
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
    
# Some helper data is constant per iteration and should not be recomputed constantly
class InferredTensorPerNLTEIter(InferredTensor):
    """This class is identical to InferredTensor, but is used to denote data which is constant over a single NLTE iteration. After each NLTE iteration, _force_infer will be called. 
    """
    def __init__(self, dtype: torch.dtype, dims: List[Parameter[int] | int | None], unit: units.Unit, infer_function: Callable[[], torch.Tensor], relative_storage_location: str | None = None, legacy_converter: Tuple[str, Callable[[Any], torch.Tensor] | None] | None = None):
        super().__init__(dtype, dims, unit, infer_function, relative_storage_location, legacy_converter)


# Some data (strings) cannot be put into pytorch tensors (and do not need to be put there)
class StorageNdarray():
    """Dataset which stores a numpy.ndarray.
    Only use this for data which cannot be stored in pytorch tensors (e.g. strings).
    """
    def __init__(self, type : np.dtype, dims : List[Union[Parameter[int], int, None]], unit : units.Unit, relative_storage_location : str, legacy_converter : Optional[Tuple[str, Optional[Callable[[Any], np.ndarray]]]] = None, array : Optional[np.ndarray[Any, Any]] = None):
        """Initializes the storageNdarray with the given datatype, required dimensions and storage location (for saving/loading). Optionally already sets the data to the given numpy.ndarray.

        Args:
            dtype (np.dtype): The data type of the stored data
            dims (List[Union[Parameter[int], int, None]]): The dimensions to check; None is used for not checking the dimension_
            relative_storage_location (str): The location at which this torch.Tensor is stored (relative to the model name)
            legacy_converter (Optional[Tuple[str, Callable[[Any], np.ndarray[any, any]]]]): The location at which this np.ndarray is stored in C++ magritte (relative to the model name) + optionally a method to convert the data
            array (Optional[np.ndarray[Any, Any]]):  Array to store and check dimensions of; if None, then dimensions are not checked. The array has to be filled in later on_
        """
        if array is not None:
            self.check_dims(array)
        self.relative_storage_location : str = relative_storage_location
        #some storage locations/data might be different for C++ magritte and python magritte, thus we need some conversion for legacy functionality
        self.legacy_relative_storage_location = self.relative_storage_location
        self.legacy_conversion_function : Optional[Callable[[Any], np.ndarray]] = None #function to apply after reading the data
        if legacy_converter is not None:
            self.legacy_relative_storage_location = legacy_converter[0]
            self.legacy_conversion_function = legacy_converter[1]

        self.dtype : np.dtype = type
        self.unit : units.Unit = unit
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
        if not np.issubdtype(new_array.dtype, self.dtype):
            raise TypeError("The type of the stored array does not correspond to required datatype; required: ", self.dtype, " input dtype: ", new_array.dtype, " storage_location: ", self.relative_storage_location)
        self.isSet = True

    def set_astropy(self, astropy_quantity : units.Quantity) -> None:
        """Sets the internal data to the given astropy Quantity if the dimensions are correct. Automatically converts units

        Args:
            astropy_array (units.Quantity): _description_

        Raises:
            astropy.units.core.UnitConversionError: when the units are incompatible
        """
        self.set(np.array(astropy_quantity.to(self.unit)))
        
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

    def get_astropy(self) -> units.Quantity:
        """Returns the stored astropy Quantity

        Raises:
            ValueError: If no data has yet been stored

        Returns:
            units.Quantity: the stored astropy Quantity
        """
        return self.get()*self.unit

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
        return self.dtype

    def is_complete(self) -> bool:
        """Returns whether the dataset is complete

        Returns:
            bool: Whether the dataset has already been set
        """
        return self.array is not None
    
class DelayedListOfClassInstances(Generic[T]):
    """Allows for delayed construction of a list of class instances.
    Implements a very limited subset of list functions.

    Args:
        Generic (_type_): Type of the class instance. Must support initialization with extra index in constructor for identifying different 
    """
    def __init__(self, length_param : Parameter[int], class_init_function : Callable[[int], T], instance_name : str) -> None:
        """Constructor for the Delay list of class instances

        Args:
            length_param (Parameter[int]): Parameter which will contain the length of the list. Must be set before calling construct
            class_init_function (Callable[[int], T]): Constructor function for a single class instance; must accept an index for differentiation the storage locations
            instance_name (str): Name of a single class instance
        """
        self.length_param = length_param
        self.class_init_function = class_init_function
        self.list : Optional[List[T]] = None
        self.instance_name = instance_name

    def construct(self) -> None:
        """Construct the list of class instances
        """
        self.list = [self.class_init_function(i) for i in range(self.length_param.get())]

    def __getitem__(self, i : int) -> T:
        """Return the element at the specified index

        Args:
            i (int): index of the data

        Raises:
            AssertionError: If the list has not yet been initialized

        Returns:
            T: the instance at list index i
        """
        if self.list is None:
            raise AssertionError("List of type "+self.instance_name+" has not yet been set.")
        return self.list[i]
    
    def is_complete(self) -> bool:
        return self.list is not None
    
    def __str__(self) -> str:
        return "Delayed list of class instances: " + self.instance_name + " of parameter size: "+str(self.length_param)
    
    def __iter__(self) -> Iterator[T]:
        if self.list is not None:
            for element in self.list:
                yield element

    
    # TODO: automatically call construct when reading from file

    # def get(self) -> List[T]:
    #     if list is None:
    #         raise AssertionError("List of type "+self.instance_name+" has not yet been set.")
    #     return list


#For convenience, use ducktyping to avoid needing to define superclasses for the datatypes
StorageTypes: TypeAlias = Union[StorageTensor, StorageNdarray]
InferredTypes: TypeAlias = InferredTensor
DelayedLists: TypeAlias = DelayedListOfClassInstances[Any]
class DataCollection():
    """Collection of all data to store for a Magritte model
    """

    def __init__(self, parameters : Parameters) -> None:
        """Initializes the DataCollection with an empty list of storageTypes
        """
        self.parameters: Parameters = parameters #list of global parameters for a model
        self.localParameters: List[Parameter[Any]] = [] #list of parameters within class instances; might not make sense to access globally, so no associated dictionary
        self.storedData: List[StorageTypes] = []
        self.storedDataDict: Dict[str, int] = {}
        self.inferredData: List[InferredTypes] = []
        self.inferredDataDict: Dict[str, int] = {}
        self.delayedLists: List[DelayedLists] = []

    def add_data(self, data : StorageTypes, name : str) -> None:
        """Adds a single dataset to the DataCollection. Does not add the dataset if a dataset with the same name is already present.

        Args:
            data (storageTypes): Dataset to add
            name (str): key for retrieving the data
        """
        if name not in self.storedDataDict:
            self.storedDataDict[name] = len(self.storedData)
            self.storedData.append(data)

    def get_data(self, name : str) -> StorageTypes:
        """Returns the data with the given key

        Args:
            name (str): the key

        Returns:
            StorageTypes: The data stored at that key
        """
        return self.storedData[self.storedDataDict[name]]

    
    def add_inferred_dataset(self, inferred_dataset : InferredTypes, name : str) -> None:
        """Adds a single inferred dataset to the DataCollection. Does not add the dataset if a dataset with the same name is already present.

        Args:
            inferred_dataset (InferredTypes): Inferred dataset to add
            name (str): key for retrieving the data
        """
        if name not in self.storedDataDict:
            self.inferredDataDict[name] = len(self.inferredData)
            self.inferredData.append(inferred_dataset)

    def get_inferred_dataset(self, name : str) -> InferredTypes:
        """Returns the inferred data with the given key

        Args:
            name (str): the key

        Returns:
            StorageTypes: The inferred data stored at that key
        """
        return self.inferredData[self.inferredDataDict[name]]


    def add_delayed_list(self, delayed_list : DelayedLists) -> None:
        """Adds a delayed list to the DataCollection.

        Args:
            delayed_list (DelayedLists): The delayed list
        """
        self.delayedLists.append(delayed_list)

    def add_local_parameter(self, local_parameter : Parameter[Any]) -> None:
        """Adds a local parameter to the DataCollection. As the parameter is local, it will not be made searchable by key.

        Args:
            local_parameter (Parameter[Any]): _description_
        """
        self.localParameters.append(local_parameter)


    def is_data_complete(self) -> bool:
        """Checks whether every dataset in the collection is complete

        Returns:
            bool: Whether all datasets of this collection are complete
        """
        isComplete : bool = True
        isCompletelistDelayedLists : List[Tuple[bool, str]] = [(data.is_complete(), data.instance_name) for data in self.delayedLists]
        if (not all([bool for bool, __ in isCompletelistDelayedLists])):
            isComplete = False
            print("Lists are not yet initialized. Lists which has not yet been set: ", [name for (cond, name) in isCompletelistDelayedLists if not cond])
        isCompletelistData : List[Tuple[bool, str]] = [(data.is_complete(), data.relative_storage_location) for data in self.storedData]
        if (not all([bool for bool, __ in isCompletelistData])):
            isComplete = False
            print("Data is not yet fully set. Data which has not yet been set: ", [name for (cond, name) in isCompletelistData if not cond])
        return isComplete
    
    def infer_data(self) -> None:
        """Infers all stored inferred datasets.

        Raises:
            ValueError: if the stored datasets are not yet complete.
        """
        #safety check: check once more if the dataset is complete before wrongfully inferring some data
        if not self.is_data_complete():
            raise ValueError("Stored data has not yet been fully set. Therefore we cannot yet infer the other data.")
        #Dev note: inferred data might rely on other inferred data, so ordering of self.inferredData might be important. In general, I recommend to put the inferred datasets last in the constructor (in this way, the data of the subclass will be inferred first)
        for inferred_dataset in self.inferredData:
            inferred_dataset.infer()

    def reset_NLTE_infer(self) -> None:
        """Reinfers the InferredTensorPerNLTEIter data
        """
        for inferred_dataset in self.inferredData:
            if type(inferred_dataset) is InferredTensorPerNLTEIter:
                inferred_dataset._force_infer()

    #TODO: maybe add inspection capabilities for figuring out for each parameter which dataset's dimensions are influenced; might be useful for debugging
    
class Types:
    """Contains expected types for tensor data. Summarized together such that tweaking them is more simple
    """
    GeometryInfo = torch.float64 #64 bit float # for positions, velocities, densities, temperatures
    IndexInfo = torch.int64 #64 bit signed int # for index information
    LevelPopsInfo = torch.float64 #64 bit float; I would like to have 128 bit instead for more accurate Ng-acceleration computations, but this not supported #for storing level populations
    #TODO: check if more accurate workaround can be added
    FrequencyInfo = torch.float64 #64 bit float; might work with 32 bit floats instead
    Enum = torch.int64 #64 bit signed int # for enums
    Bool = torch.bool #boolean # for truth values
    NpString = np.dtype('S') #use null-terminated objects to store strings in hdf5; unicode is not supported

