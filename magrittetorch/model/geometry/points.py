from typing import List, Union, Optional, Tuple
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection
from magrittetorch.model.parameters import Parameters
import torch
from astropy import units
from astropy.constants import c#speed of light

storagedir : str = "geometry/points/"

class Points:
    def __init__(self, params: Parameters, dataCollection : DataCollection) -> None:
        self.parameters: Parameters = params
        self.dataCollection : DataCollection = dataCollection
        self.position: StorageTensor = StorageTensor(Types.GeometryInfo, [self.parameters.npoints, 3], units.m, storagedir+"position"); self.dataCollection.add_data(self.position, "position") # Position coordinates
        self.velocity: StorageTensor = StorageTensor(Types.GeometryInfo, [self.parameters.npoints, 3], units.m/units.s, storagedir+"velocity", legacy_converter=[storagedir+"velocity", self._legacy_import_velocity]); self.dataCollection.add_data(self.velocity, "velocity") # Velocity vectors (in m/s)
        self.n_neighbors: StorageTensor = StorageTensor(Types.IndexInfo, [self.parameters.npoints], units.dimensionless_unscaled, storagedir+"n_neighbors"); self.dataCollection.add_data(self.n_neighbors, "n_neighbors") # number of neighbors per point
        self.neighbors: StorageTensor = StorageTensor(Types.IndexInfo, [None], units.dimensionless_unscaled, storagedir+"neighbors"); self.dataCollection.add_data(self.neighbors, "neighbors") # linearized neighbors vector

    def get_cum_n_neighbors(self, device: torch.device) -> torch.Tensor:
        temp: torch.Tensor = torch.cumsum(self.n_neighbors.get(device), dim=0).roll(1)
        temp[0] = 0
        return temp

    def _legacy_import_velocity(self, dimensionless_velocity: torch.Tensor) -> torch.Tensor:
        return c.value*dimensionless_velocity#type: ignore
    
    def get_neighbors_in_direction_3D_geometry(self, raydir: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the subset of the neighbors which lie in the given direction for 3D geometries.

        Args:
            raydir (torch.Tensor): The ray direction. Has dimensions [3]
            device (torch.device): The device on which to compute and store the torch.Tensors

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Linearized neighbors tensor in correct direction, corresponding number of neighbors per point and cumulative number of neighbors.
            The tensors respectively have dimensions [sum(n_neighbors_in_correct_direction)], [parameters.npoints] and [parameters.npoints]
        """
        extension_indices = torch.arange(self.parameters.npoints.get(), dtype=Types.IndexInfo, device=device).repeat_interleave(self.n_neighbors.get(device))
        position_device = self.position.get(device)#dims: [parameters.npoints, 3]
        neighbors_device = self.neighbors.get(device)#dims: [sum(self.n_neighborsj)]
        relative_difference_position = position_device[neighbors_device] - position_device[extension_indices]
        in_correct_direction = torch.sum(relative_difference_position * raydir[None, :], dim=1) > 0.0
        neighbors_in_correct_direction = neighbors_device[in_correct_direction]
        n_neighbors_in_correct_direction = torch.scatter_add(torch.zeros(self.parameters.npoints.get(), dtype=Types.IndexInfo, device=device), 0, extension_indices, in_correct_direction.type(Types.IndexInfo))
        cum_n_neighbors_in_correct_direction = torch.cumsum(n_neighbors_in_correct_direction, dim=0).roll(1)
        cum_n_neighbors_in_correct_direction[0] = 0
        return neighbors_in_correct_direction, n_neighbors_in_correct_direction, cum_n_neighbors_in_correct_direction




