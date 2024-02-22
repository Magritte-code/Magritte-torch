from typing import List, Union, Optional
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor, StorageNdarray
from magrittetorch.model.parameters import Parameters
import torch
import numpy as np
from astropy import units

storagedir : str = "chemistry/species/"

class Species:
    def __init__(self, params: Parameters, dataCollection : DataCollection) -> None:
        self.parameters: Parameters = params
        self.dataCollection : DataCollection = dataCollection
        self.abundance: StorageTensor = StorageTensor(Types.GeometryInfo, [self.parameters.npoints, self.parameters.nspecs], units.m**-3, storagedir+"abundance")#: Species abundance; dtype= :attr:`.Types.GeometryInfo`, dims=[:py:attr:`.Parameters.npoints`, :py:attr:`.Parameters.nspecs`], unit = units.m**-3
        self.dataCollection.add_data(self.abundance, "species abundance")
        self.symbol : StorageNdarray = StorageNdarray(Types.NpString, [self.parameters.nspecs], units.dimensionless_unscaled, storagedir+"species", legacy_converter = [storagedir+"species", self.__legacy_get_symbol])#: symbols for the species; dtype= :attr:`.Types.NpString`, dims=[:py:attr:`.Parameters.nspecs`], unit = units.dimensionless_unscaled
        self.dataCollection.add_data(self.symbol, "species symbol")
        #Note: symbol currently commented out, as some older magritte models might not store this unfortunately

    def __legacy_get_symbol(self, symbol: np.ndarray) -> np.ndarray[Types.NpString]:
        #check dtype; if int64, convert to string; as it has not been set correctly
        if symbol.dtype == np.int64:
            symbol = np.array([str(s) for s in symbol], dtype=Types.NpString)
            #the resulting symbols for the species will be meaningless, but at least the model will load
        return symbol
