from typing import List, Union, Optional
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor
from magrittetorch.model.parameters import Parameters
from magrittetorch.model.chemistry.species import Species
import torch

storagedir : str = "chemistry/"

class Chemistry:
    #class that was originally intended to do some chemistry calculations, but now just a dummy class
    #might be worth keeping for introducing a dust datastructure

    def __init__(self, params: Parameters, dataCollection : DataCollection) -> None:
        self.parameters: Parameters = params
        self.dataCollection : DataCollection = dataCollection
        self.species: Species = Species(self.parameters, self.dataCollection)#: Species data
