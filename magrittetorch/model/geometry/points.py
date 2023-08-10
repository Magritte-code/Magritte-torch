from typing import List, Union, Optional
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection
from magrittetorch.model.parameters import Parameters
import torch

storagedir : str = "geometry/points/"

class Points:
    def __init__(self, params: Parameters, dataCollection : DataCollection) -> None:
        self.parameters: Parameters = params
        self.dataCollection : DataCollection = dataCollection
        self.position: StorageTensor = StorageTensor(Types.GeometryInfo, [self.parameters.npoints, 3], storagedir+"position"); self.dataCollection.add_data(self.position) # Position coordinates
        self.velocity: StorageTensor = StorageTensor(Types.GeometryInfo, [self.parameters.npoints, 3], storagedir+"velocity"); self.dataCollection.add_data(self.velocity) # Velocity vectors
        self.n_neighbors: StorageTensor = StorageTensor(Types.IndexInfo, [self.parameters.npoints], storagedir+"n_neighbors"); self.dataCollection.add_data(self.n_neighbors) # number of neighbors per point
        self.neighbors: StorageTensor = StorageTensor(Types.IndexInfo, [None], storagedir+"neighbors"); self.dataCollection.add_data(self.neighbors) # linearized neighbors vector

    def get_cum_n_neighbors(self) -> torch.Tensor:
        temp : torch.Tensor = torch.cumsum(self.n_neighbors.get(), dim=0).roll(1)
        temp[0] = 0
        return temp