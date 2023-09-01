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
        self.vturb: StorageTensor = StorageTensor(Types.GeometryInfo, [self.parameters.npoints], units.m/units.s, storagedir+"vturb", legacy_converter = (storagedir+"vturb2", self.__legacy_vturb2_to_v)); self.dataCollection.add_data(self.vturb, "vturb") 
        self.vturb2_norm: InferredTensor = InferredTensor(Types.GeometryInfo, [self.parameters.npoints], units.dimensionless_unscaled, self.__infer_vturb2_norm); self.dataCollection.add_inferred_dataset(self.vturb2_norm, "vturb2 normalized")

    def __legacy_vturb2_to_v(self, vturb2: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(vturb2)*astropy.constants.c.value#type: ignore #as of writing: astropy has only partially implemented type hints
    
    def __infer_vturb2_norm(self) -> torch.Tensor:
        return torch.pow(self.vturb.get()/astropy.constants.c.value, 2)
