from enum import Enum
from typing import List, Union, TypeVar
from magrittetorch.model.geometry.geometry import Geometry
from magrittetorch.model.parameters import Parameters
from magrittetorch.utils.storagetypes import DataCollection
from magrittetorch.utils.io import IO
import torch



class NgAccelerationType(Enum):
    """
    Type of ng-acceleration to use
    """
    Default = 0
    Adaptive = 1

# Dummy classes; should be refactored into seperate files
class Chemistry:
    # Define the Chemistry class here
    pass

class Thermodynamics:
    # Define the Thermodynamics class here
    pass

class Lines:
    # Define the Lines class here
    pass

class Radiation:
    # Define the Radiation class here
    pass

class Image:
    # Define the Image class here
    pass

class Model:
    def __init__(self, name : str) -> None:
        self.dataCollection = DataCollection()
        self.name : str = name
        self.io = IO(name)
        self.parameters: Parameters = Parameters()
        self.parameters.name.set(name)
        self.geometry: Geometry = Geometry(self.parameters, self.dataCollection)
        self.chemistry: Union[Chemistry, None] = None
        self.thermodynamics: Union[Thermodynamics, None] = None
        self.lines: Union[Lines, None] = None
        self.radiation: Union[Radiation, None] = None
        self.images: List[Image] = []

    def write(self) -> None:
        self.io.write(self.dataCollection)

    def read(self) -> None:
        self.io.read(self.dataCollection)    

    
