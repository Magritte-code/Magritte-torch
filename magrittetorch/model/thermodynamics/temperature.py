from typing import List, Union, Optional
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor
from magrittetorch.model.parameters import Parameters
import torch
from astropy import units

storagedir : str = "thermodynamics/temperature/"

class Temperature:
    def __init__(self, params: Parameters, dataCollection : DataCollection) -> None:
        self.parameters: Parameters = params
        self.dataCollection : DataCollection = dataCollection
        self.gas: StorageTensor = StorageTensor(Types.GeometryInfo, [self.parameters.npoints], units.K, storagedir+"gas"); self.dataCollection.add_data(self.gas, "gas temperature") # Gas temperature [K]