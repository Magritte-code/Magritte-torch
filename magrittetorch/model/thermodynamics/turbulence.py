from typing import List, Union, Optional
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor
from magrittetorch.model.parameters import Parameters
from astropy import units
import astropy.constants
import torch

storagedir : str = "thermodynamics/turbulence/"

class Turbulence:
    def __init__(self, params: Parameters, dataCollection : DataCollection) -> None:
        self.parameters: Parameters = params
        self.dataCollection : DataCollection = dataCollection
        self.vturb: StorageTensor = StorageTensor(Types.GeometryInfo, [self.parameters.npoints], units.dimensionless_unscaled, storagedir+"vturb", legacy_converter = (storagedir+"vturb2", self.__legacy_vturb2_to_v)); self.dataCollection.add_data(self.vturb, "vturb") 
        #python : stores vturb; legacy stores: (vturb/c)**2
        #TODO: ask the non squared/c**2 quantity instead; and replace vturb2 by an inferredTensor instead

    def __legacy_vturb2_to_v(self) -> None:
        self.vturb.set(torch.sqrt(self.vturb.get())*astropy.constants.c.value)