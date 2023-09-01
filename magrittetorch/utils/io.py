import h5py # type: ignore
import numpy as np
import torch
from magrittetorch.utils.storagetypes import DataCollection, StorageTensor, StorageNdarray
from magrittetorch.model.parameters import Parameter
from typing import Optional, Any, Generic, TypeVar, Type, List, Union, Tuple, Callable, Collection, MutableSet

# TODO: complete docs

class IO:

    def __init__(self, save_location : str) -> None:
        """Initializes the IO instance, with given save location for the model.

        Args:
            save_location (str): Save location (for loading/saving)
        """
        self.save_location = save_location

    def read(self, dataCollection : DataCollection, legacy_mode : bool = True) -> None:
        """Reads all stored values from a given DataCollection (stored in hdf5 format)

        Args:
            dataCollection (DataCollection): The DataCollection storing all values

        Raises:
            TypeError: Dev error: when reading a storagetypes.storageTypes for which reading has not yet been implemented
        """
        file = h5py.File(self.save_location, 'r')
        parameter : Parameter[Any]
        if legacy_mode:#in legacy mode, data can be saved at different locations
            try_read_again_parameters: MutableSet[Parameter[Any]] = set()
            for parameter in dataCollection.parameters:
                self.read_parameter(file, parameter, parameter.legacy_name)
            #read all delayed lists; Dev Note: the storedData list might get appended during reading the delayedLists
            for delayedlist in dataCollection.delayedLists:
                delayedlist.construct()
                #now read all local parameters, as they might have been constructed within a delayed list
                for localParameter in dataCollection.localParameters:
                    #TODO: reading every local parameter again is inefficient; keep track of currently read parameter/not yet read parameters
                    try:
                        print(localParameter, localParameter.legacy_name)
                        self.read_parameter(file, localParameter, localParameter.legacy_name, localParameter.legacy_conversion_function)
                    except Exception as e:
                        print(e)
                        try_read_again_parameters.add(localParameter)
            for datapiece in dataCollection.storedData:
                if type(datapiece) is StorageTensor:
                    legacy_data_torch = self.read_torch(file, datapiece.legacy_relative_storage_location)
                    if datapiece.legacy_conversion_function is not None: legacy_data_torch = datapiece.legacy_conversion_function(legacy_data_torch)
                    datapiece.set(legacy_data_torch)
                elif type(datapiece) is StorageNdarray:
                    legacy_data_numpy = self.read_numpy(file, datapiece.legacy_relative_storage_location)
                    if datapiece.legacy_conversion_function is not None: legacy_data_numpy = datapiece.legacy_conversion_function(legacy_data_numpy)
                    # print(legacy_data_numpy)
                    datapiece.set(legacy_data_numpy)
                else: 
                    raise TypeError("Data reading not yet implemented for " + str(type(datapiece)))
            #some local parameters might need more data in order to be correctly set
            for localParameter in try_read_again_parameters:
                self.read_parameter(file, localParameter, localParameter.legacy_name, localParameter.legacy_conversion_function)
            for inferredDatapiece in dataCollection.inferredData:
                if inferredDatapiece.legacy_relative_storage_location is not None:
                    try:
                        #for now only inferred tensors exist
                        legacy_data_torch = self.read_torch(file, inferredDatapiece.legacy_relative_storage_location)
                        if inferredDatapiece.legacy_conversion_function is not None: legacy_data_torch = inferredDatapiece.legacy_conversion_function(legacy_data_torch)
                        inferredDatapiece.set(legacy_data_torch)
                    except TypeError as e:
                        print(e)#inferred data might not be set

        else:#non-legacy mode reading
            for parameter in dataCollection.parameters:
                self.read_parameter(file, parameter, parameter.name)
            #read all delayed lists; Dev Note: the storedData list might get appended during reading the delayedLists
            for delayedlist in dataCollection.delayedLists:
                delayedlist.construct()
            #now read all local parameters, as they might have been constructed within a delayed list
            for localParameter in dataCollection.localParameters:
                self.read_parameter(file, localParameter, localParameter.name)
            for datapiece in dataCollection.storedData:
                if type(datapiece) is StorageTensor:
                    datapiece.set(self.read_torch(file, datapiece.relative_storage_location))
                elif type(datapiece) is StorageNdarray:
                    datapiece.set(self.read_numpy(file, datapiece.relative_storage_location))
                else: 
                    raise TypeError("Data reading not yet implemented for " + str(type(datapiece)))
            for inferredDatapiece in dataCollection.inferredData:
                if inferredDatapiece.relative_storage_location is not None:
                    try:
                        inferredDatapiece.set(self.read_torch(file, inferredDatapiece.relative_storage_location))
                    except TypeError as e:
                        print(e)#inferred data might not be set

    def write(self, dataCollection : DataCollection) -> None:
        """Write all stored values from a given DataCollection to hdf5 format

        Args:
            dataCollection (DataCollection): The DataCollection storing all values

        Raises:
            ValueError: If not all data is already set.
            TypeError: Dev error: when reading a storagetypes.storageTypes for which reading has not yet been implemented
        """
        if not dataCollection.is_data_complete():
            raise ValueError("Data set is not yet complete. Please fill in the missing data points")
        file = h5py.File(self.save_location, 'w')
        for param in dataCollection.parameters:#type: ignore
            self.write_parameter(file, param, param.name)
        for datapiece in dataCollection.storedData:
            if type(datapiece) is StorageTensor:
                self.write_torch(file, datapiece.relative_storage_location, datapiece.get())
            elif type(datapiece) is StorageNdarray:
                self.write_numpy(file, datapiece.relative_storage_location, datapiece.get())
            else: 
                raise TypeError("Data writing not yet implemented for " + str(type(datapiece)))
        pass

    def read_numpy(self, file : h5py.File, dataset_name : str) -> np.ndarray[Any, Any]:
        """Reads a single numpy dataset from the given hdf5 file
        TODO: raise error if no data found at that location

        Args:
            file (h5py.File): The file to read from
            dataset_name (str): The name of the dataset

        Returns:
            np.ndarray[Any, Any]: The np.ndarray saved at that location
        """
        return np.array(file.get(dataset_name))

    def write_numpy(self, file : h5py.File, dataset_name : str, numpy_data: np.ndarray[Any, Any]) -> None:
        """Write a single numpy dataset to the given hdf5 file
        TODO: raise error if no data found at that location

        Args:
            file (h5py.File): The file to write to
            dataset_name (str): The name of the dataset
            numpy_data (np.ndarray[Any, Any]): The data to write
        """
        #first make sure that all groups exist before writing anything
        group = ''
        for g in dataset_name.split('/')[:-1]:
            group += f'/{g}'
            file.require_group (group)
        file.create_dataset (name=dataset_name, data=numpy_data)


    def read_torch(self, file : h5py.File, dataset_name : str) -> torch.Tensor:
        """Reads a single torch dataset from the given hdf5 file

        Args:
            file (h5py.File): The file to read from
            dataset_name (str): The name of the dataset

        Returns:
            torch.Tensor: The torch.Tensor saved at that location
        """
        return torch.from_numpy(self.read_numpy(file, dataset_name))
    
    def write_torch(self, file : h5py.File, dataset_name : str, torch_data : torch.Tensor) -> None:
        """Writes a single torch dataset to the given hdf5 file

        Args:
            file (h5py.File): The file to write to
            dataset_name (str): The name of the dataset
            torch_data (torch.Tensor): The data to write
        """
        self.write_numpy(file, dataset_name, torch_data.numpy(force=True))



    T = TypeVar('T')
    def read_parameter(self, file : h5py.File, parameter : Parameter[Any], parameter_path : str, legacy_conversion_function: Optional[Callable[[Any], T]] = None) -> None:
        """Reads a single parameter from the given hdf5 file

        Args:
            file (h5py.File): The file to read from
            attrs_name (str): The name of the corresponding attribute
            parameter (Parameter): The parameter to read
            legacy_conversion_function (Optional[Callable[[Any], T]]): Optionally adds a conversion function for the read data to determine the value of the parameter
        """
        if legacy_conversion_function is not None:
            try:
                parameter.set(legacy_conversion_function(file[parameter_path]))#legacy conversion function will act directly on a dataset to infer the parameter value
            except KeyError:
                parameter.set(legacy_conversion_function(file.attrs[parameter_path]))#legacy conversion function acting on an attribute
        else:
            try:
                path : str; attribute : str 
                #mypy doesnt allow for unpacking of strings, thus ignore the warnings
                path, attribute = parameter_path.rsplit("/", 1)#type: ignore
                parameter.set(file[path].attrs[attribute])
            except ValueError:#guard against not enough parameters to unpack
                parameter.set(file.attrs[parameter_path])

    def write_parameter(self, file : h5py.File, parameter : Parameter[Any], storage_path : str) -> None:
        """Writes a single parameter to the given hdf5 file

        Args:
            file (h5py.File): The file to write to
            attrs_name (str): The name of the corresponding attribute
            parameter (Parameter): The parameter to write
        """
        path : str; attribute : str 
        #mypy doesnt allow for unpacking of strings, thus ignore the warnings
        try:
            path, attribute = storage_path.rsplit("/", 1)#type: ignore
            file[path].attrs[attribute] = parameter.get()
        except ValueError:#guard against not enough parameters to unpack
            file.attrs[parameter.name] = parameter.get()


    
        
class LegacyHelper:
    """Collection of helper functions for helping reading legacy C++ magritte models, for filling in gaps of metadata
    """
    @staticmethod
    def read_length_of_group(search: str, h5pygroup: h5py.Group) -> int:
        """Reads the length of a h5py Group

        Args:
            search (str): common name of the group
            h5pygroup (h5py.Group): h5py Group to search within

        Returns:
            int: length of the group
        """
        return len([1 for key in h5pygroup.keys() if search in key])

    @staticmethod
    def read_length_of_dataset(h5pydataset: h5py.Dataset) -> int:
        """Reads the length of a h5py Dataset

        Args:
            h5pydataset (h5py.Dataset): Dataset to read

        Returns:
            int: Length of the h5py.Dataset
        """
        return len(h5pydataset)

