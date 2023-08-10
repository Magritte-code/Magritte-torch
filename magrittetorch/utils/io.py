import h5py # type: ignore
import numpy as np
import torch
from magrittetorch.utils.storagetypes import DataCollection, StorageTensor, StorageNdarray
from typing import Optional, Any, Generic, TypeVar, Type, List, Union, Tuple

class IO:

    def __init__(self, save_location : str) -> None:
        """Initializes the IO instance, with given save location for the model.

        Args:
            save_location (str): Save location (for loading/saving)
        """
        self.save_location = save_location

    def read(self, dataCollection : DataCollection) -> None:
        """Reads all stored values from a given DataCollection (stored in hdf5 format)

        Args:
            dataCollection (DataCollection): The DataCollection storing all values

        Raises:
            TypeError: Dev error: when reading a storagetypes.storageTypes for which reading has not yet been implemented
        """
        #TODO: for all parameters, try to read a model
        file = h5py.File(self.save_location, 'r')
        for datapiece in dataCollection.storedData:
            if type(datapiece) is StorageTensor:
                datapiece.set(self.read_torch(file, datapiece.relative_storage_location))
            elif type(datapiece) is StorageNdarray:
                datapiece.set(self.read_numpy(file, datapiece.relative_storage_location))
            else: 
                raise TypeError("Data reading not yet implemented for " + str(type(datapiece)))
        pass

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

    
        

