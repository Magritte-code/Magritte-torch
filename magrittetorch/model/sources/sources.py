from typing import List, Union, Optional
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor
from magrittetorch.model.parameters import Parameters
from magrittetorch.model.sources.lines import Lines
import torch
from astropy import units

storagedir : str = "sources/"

#TODO: implement in seperate file
class Continuum:
    pass

#plan to implement
class Scattering:
    pass

class Sources:
    def __init__(self, params: Parameters, dataCollection : DataCollection) -> None:
        self.parameters: Parameters = params
        self.dataCollection : DataCollection = dataCollection
        self.lines = Lines(self.parameters, dataCollection)
        self.continuum : Continuum = Continuum()