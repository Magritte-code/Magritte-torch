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
from magrittetorch.model.image import Image, ImageType
import torch


class Model:
    def __init__(self, save_location: str) -> None:
        """Initializes the model, and specifies the location to save/read the model to/from.

        Args:
            save_location (str): Location of the hdf5 file to save/read the model to/from 
        """
        self.parameters: Parameters = Parameters()#: Model parameters
        self.dataCollection: DataCollection = DataCollection(self.parameters)#: DataCollection
        # self.name : str = name
        self.io: IO = IO(save_location)#: IO module
        self.geometry: Geometry = Geometry(self.parameters, self.dataCollection)#: Geometry module
        self.chemistry: Chemistry = Chemistry(self.parameters, self.dataCollection)#: Chemistry module
        self.thermodynamics: Thermodynamics = Thermodynamics(self.parameters, self.dataCollection)#: Thermodynamics module
        self.sources : Sources = Sources(self.parameters, self.dataCollection)#: Sources module
        self.images: list[Image] = []#: List of images

    def write(self) -> None:
        """Writes the model to the save location
        """
        print("Writing model to: ", self.io.save_location)
        self.io.write(self.dataCollection)

    def read(self, legacy_mode: bool = False) -> None:
        """Reads the model from the save location

        Args:
            legacy_mode (bool, optional): If False, reads a Magrittetorch model. If True, reads a Magritte model. Defaults to False.
        """
        print("Reading model from: ", self.io.save_location)
        self.io.read(self.dataCollection, legacy_mode)

    def get_boundary_intensity(self, point_indices: torch.Tensor, freqhelper: FrequencyEvalHelper, doppler_shift: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Computes the boundary intensity for the given boundary points and frequencies

        Args:
            point_indices (torch.Tensor): Indices of the boundary points. Has dimensions [NPOINTS]
            freqhelper (FrequencyEvalHelper): Frequencies to evaluate. Has dimensions [NPOINTS, NFREQS]
            doppler_shift (torch.Tensor): Doppler shift with respect to the boundary points. Has dimensions [NPOINTS]
            device (torch.device): Device on which to compute and return the result

        Returns:
            torch.Tensor: The boundary intensity. Has dimensions [NPOINTS, NFREQS]
        """
        #I need both frequency info (contained within sources) and boundary info, so this function goes in model
        return self.geometry.boundary.get_boundary_intensity(point_indices, freqhelper.original_frequencies[point_indices,:]*doppler_shift[:, None], device)
