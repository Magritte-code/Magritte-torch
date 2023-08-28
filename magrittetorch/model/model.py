from enum import Enum
from typing import List, Union, TypeVar
from magrittetorch.model.geometry.geometry import Geometry
from magrittetorch.model.chemistry.chemistry import Chemistry
from magrittetorch.model.parameters import Parameters
from magrittetorch.utils.storagetypes import DataCollection
from magrittetorch.model.thermodynamics.thermodynamics import Thermodynamics
from magrittetorch.model.sources.sources import Sources
from magrittetorch.utils.io import IO
import torch



class NgAccelerationType(Enum):
    """
    Type of ng-acceleration to use
    """
    Default = 0
    Adaptive = 1

# Dummy classes; should be refactored into seperate files
class Radiation:
    # Define the Radiation class here
    pass

class Image:
    # Define the Image class here
    pass

class Model:
    def __init__(self, name : str) -> None:
        self.parameters: Parameters = Parameters()
        self.dataCollection = DataCollection(self.parameters)
        self.name : str = name
        self.io = IO(self.name)
        self.geometry: Geometry = Geometry(self.parameters, self.dataCollection)
        self.chemistry: Chemistry = Chemistry(self.parameters, self.dataCollection)
        self.thermodynamics: Thermodynamics = Thermodynamics(self.parameters, self.dataCollection)
        self.sources : Sources = Sources(self.parameters, self.dataCollection)
        self.radiation: Union[Radiation, None] = None
        self.images: List[Image] = []

    def write(self) -> None:
        self.io.write(self.dataCollection)

    def read(self) -> None:
        self.io.read(self.dataCollection)

    
