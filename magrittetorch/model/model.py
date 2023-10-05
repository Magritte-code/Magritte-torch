from enum import Enum
from typing import List, Union, TypeVar
from magrittetorch.model.geometry.geometry import Geometry
from magrittetorch.model.chemistry.chemistry import Chemistry
from magrittetorch.model.parameters import Parameters
from magrittetorch.utils.storagetypes import DataCollection
from magrittetorch.model.thermodynamics.thermodynamics import Thermodynamics
from magrittetorch.model.sources.sources import Sources
from magrittetorch.utils.io import IO
from magrittetorch.model.sources.frequencyevalhelper import FrequencyEvalHelper
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
    def __init__(self, save_location: str) -> None:
        self.parameters: Parameters = Parameters()
        self.dataCollection = DataCollection(self.parameters)
        # self.name : str = name
        self.io = IO(save_location)
        self.geometry: Geometry = Geometry(self.parameters, self.dataCollection)
        self.chemistry: Chemistry = Chemistry(self.parameters, self.dataCollection)
        self.thermodynamics: Thermodynamics = Thermodynamics(self.parameters, self.dataCollection)
        self.sources : Sources = Sources(self.parameters, self.dataCollection)
        self.radiation: Union[Radiation, None] = None
        self.images: List[Image] = []

    def write(self) -> None:
        print("writing")
        self.io.write(self.dataCollection)

    def read(self, legacy_mode: bool = False) -> None:
        self.io.read(self.dataCollection, legacy_mode)

    def get_boundary_intensity(self, point_indices: torch.Tensor, freqhelper: FrequencyEvalHelper, device: torch.device) -> torch.Tensor:
        """Computes the boundary intensity for the given boundary points and frequencies

        Args:
            point_indices (torch.Tensor): Indices of the boundary points. Has dimensions [NPOINTS]
            freqhelper (FrequencyEvalHelper): Frequencies to evaluate. Has dimensions [NPOINTS, NFREQS]
            device (torch.device): Device on which to compute and return the result

        Returns:
            torch.Tensor: The boundary intensity. Has dimensions [NPOINTS, NFREQS]
        """
        #I need both frequency info (contained within sources) and boundary info, so this function goes in model
        return self.geometry.boundary.get_boundary_intensity(point_indices, freqhelper.original_frequencies[point_indices,:], device)

    # def solve_long_characteristics_NLTE_to_astropy(self, device: torch.device) -> Quantity[J/units.m**2]:
    #     solve_long_characteristics_NLTE(self, device).to_numpy(force=True)



    
