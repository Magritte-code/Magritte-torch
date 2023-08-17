from typing import List, Union, Optional
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection
from magrittetorch.model.parameters import Parameters
import torch
from astropy import units

storagedir : str = "geometry/points/"

class Points:
    def __init__(self, params: Parameters, dataCollection : DataCollection) -> None:
        self.parameters: Parameters = params
        self.dataCollection : DataCollection = dataCollection
        self.position: StorageTensor = StorageTensor(Types.GeometryInfo, [self.parameters.npoints, 3], units.m, storagedir+"position"); self.dataCollection.add_data(self.position, "position") # Position coordinates
        self.velocity: StorageTensor = StorageTensor(Types.GeometryInfo, [self.parameters.npoints, 3], units.dimensionless_unscaled, storagedir+"velocity"); self.dataCollection.add_data(self.velocity, "velocity") # Velocity vectors (in units of speed of light)
        self.n_neighbors: StorageTensor = StorageTensor(Types.IndexInfo, [self.parameters.npoints], units.dimensionless_unscaled, storagedir+"n_neighbors"); self.dataCollection.add_data(self.n_neighbors, "n_neighbors") # number of neighbors per point
        self.neighbors: StorageTensor = StorageTensor(Types.IndexInfo, [None], units.dimensionless_unscaled, storagedir+"neighbors"); self.dataCollection.add_data(self.neighbors, "neighbors") # linearized neighbors vector

    def get_cum_n_neighbors(self) -> torch.Tensor:
        temp : torch.Tensor = torch.cumsum(self.n_neighbors.get(), dim=0).roll(1)
        temp[0] = 0
        return temp