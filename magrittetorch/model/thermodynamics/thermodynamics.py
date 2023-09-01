#contains everything about the thermodynamics of a model
from enum import Enum
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor
from magrittetorch.model.parameters import Parameters
from magrittetorch.model.thermodynamics.temperature import Temperature
from magrittetorch.model.thermodynamics.turbulence import Turbulence
import torch

storagedir : str = "thermodynamics/"

class Thermodynamics:
    
    def __init__(self, params : Parameters, dataCollection : DataCollection) -> None:
        self.parameters: Parameters = params
        self.dataCollection : DataCollection = dataCollection
        self.temperature : Temperature = Temperature(self.parameters, self.dataCollection)
        self.turbulence : Turbulence = Turbulence(self.parameters, self.dataCollection)
