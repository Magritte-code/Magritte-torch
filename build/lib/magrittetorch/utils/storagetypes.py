import torch
from typing import Optional, Any, Generic, TypeVar, Type, List, Union, Tuple
from magrittetorch.model.parameters import Parameter



class storageTensor():
    instances : List[Any]
    def __init__(self, dtype : torch.dtype, dims : List[Union[Parameter[int], int, None]], relative_storage_location : str, tensor : Optional[torch.Tensor] = None):
        """_summary_ TODO: think about how to automatically store everything; maybe include the class somehow from which this is called? or just dump everything at base level in hdf5 file (bad idea, might cause duplicates)
        Just add some storage_location thing to every file; maybe generate this on init
        or only on write ?, hmmm, think about this a bit more

        Args:
            dims (List[Union[Parameter[int], int, None]]): The dimensions to check; None is used for not checking the dimension_
            tensor (Optional[torch.Tensor]): _tensor to store and check dimensions of; if None, then dimensions are not checked. The tensor has to be filled in later on_
        """
        self.relative_storage_location : str = relative_storage_location
        self.dtype = dtype
        self.tensor : Optional[torch.Tensor] = tensor
        self.dims : List[Union[Parameter[int], int, None]] = dims
        self.isSet : bool = False
        if tensor is not None:
            self.check_dims()
        DataChecker.add_tensor(self) #for keeping track of all storageTensors

    def check_dims(self) -> None:
        """Checks the dimensions of the stored tensor. Also checks type of data in tensor.

        Raises:
            AssertionError: If the dimensions of the tensor do not correspond to the dimensions to check
            ValueError: If the dimension size is not equal to the set integer size
            TypeError: Dev error: If the dimensions to check contain wrong types
        """
        if self.tensor is None:
            raise AssertionError("The tensor has not yet been set")
        size : torch.Size = self.tensor.size()
        if len(size) != len(self.dims):
            raise AssertionError("The amount of dimensions of the tensor does not correspond to the amount of ")
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
        if self.tensor.dtype != self.dtype:
            raise TypeError("The type of the stored data does not correspond to ")
        self.isSet = True
        
    def set(self, tensor : torch.Tensor) -> None:
        self.tensor = tensor
        self.check_dims()

    def get(self) -> torch.Tensor:
        if self.tensor is None:
            raise ValueError("The data has not yet been set for " +str(self.relative_storage_location))
        return self.tensor
    
    def get_type(self) -> torch.dtype:
        return self.dtype

    def is_complete(self) -> bool:
        return self.tensor is not None
    

    #TODO: add saving/loading capabilities


class DataChecker():
    storedTensors : List[storageTensor] = []
    @staticmethod
    def add_tensor(tensor : storageTensor) -> None:
        DataChecker.storedTensors.append(tensor)

    @staticmethod
    def check_data_complete() -> bool:
        isCompletelist : List[Tuple[bool, str]] = [(data.is_complete(), data.relative_storage_location) for data in DataChecker.storedTensors]
        if (all([bool for bool, __ in isCompletelist])):
            return True
        raise ValueError("Data is not yet fully set. Data which has not yet been set: ", [name for (cond, name) in isCompletelist if not cond])
    
    @staticmethod
    def save_all_tensors() -> None:
        pass

    #TODO: also make sure to store all metadata required for all tensors and some vector of locations, ...
    @staticmethod
    def load_all_tensors() -> None:
        pass

    
class Types:
    """Contains expected types for data. Summarized together such that tweaking them is more simple
    """
    GeometryInfo = torch.float #32 bit float
    IndexInfo = torch.int #32bit signed int


