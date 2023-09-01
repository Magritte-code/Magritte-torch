from typing import List, Union, Optional
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor
from magrittetorch.model.parameters import Parameters
import torch
from astropy import units

storagedir : str = "geometry/rays/"

class Rays:
    def __init__(self, params: Parameters, dataCollection : DataCollection) -> None:
        self.parameters: Parameters = params
        self.dataCollection : DataCollection = dataCollection
        self.direction: StorageTensor = StorageTensor(Types.GeometryInfo, [self.parameters.nrays, 3], units.dimensionless_unscaled, storagedir+"direction"); self.dataCollection.add_data(self.direction, "ray direction") # Direction vectors
        self.weight: StorageTensor = StorageTensor(Types.GeometryInfo, [self.parameters.nrays], units.dimensionless_unscaled, storagedir+"weight"); self.dataCollection.add_data(self.weight, "ray weight") # Corresponding weights
        self.antipod: InferredTensor = InferredTensor(Types.IndexInfo, [self.parameters.nrays], units.dimensionless_unscaled, self._infer_antipod); self.dataCollection.add_inferred_dataset(self.antipod, "ray antipod") # Index of antipod

    def _infer_antipod(self) -> torch.Tensor:
        """Infers the antipodal ray indices, assuming that all ray directions have antipods

        Returns:
            torch.Tensor: The antipodal ray indices. Has dimensions [parameters.nrays]
        """
        nrays : int = self.parameters.nrays.get()
        direction_tensor = self.direction.get()
        distances = torch.sum(torch.abs(direction_tensor.reshape(nrays, 3, 1) + direction_tensor.T.reshape(1,3,nrays)), dim=1)#L1 norm for simplicity of implementation
        minidx = torch.argmin(distances, dim=0)#value of dim doesn't matter: distance matrix, and indices are symmetric
        return minidx


