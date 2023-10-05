from typing import List, Union, Optional
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor, StorageNdarray
from magrittetorch.model.parameters import Parameters
import torch
from astropy import units

storagedir : str = "chemistry/species/"

class Species:
    def __init__(self, params: Parameters, dataCollection : DataCollection) -> None:
        self.parameters: Parameters = params
        self.dataCollection : DataCollection = dataCollection
        self.abundance: StorageTensor = StorageTensor(Types.GeometryInfo, [self.parameters.npoints, self.parameters.nspecs], units.m**-3, storagedir+"abundance"); self.dataCollection.add_data(self.abundance, "species abundance") # abundance; number density per m**3
        self.symbol : StorageNdarray = StorageNdarray(Types.NpString, [self.parameters.nspecs], units.dimensionless_unscaled, storagedir+"species"); self.dataCollection.add_data(self.symbol, "species symbol") #symbols for the species
        #Note: symbol currently commented out, as some older magritte models might not store this unfortunately

