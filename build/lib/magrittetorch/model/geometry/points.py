from typing import List, Union, Optional
from magrittetorch.utils.storagetypes import storageTensor, Types
from magrittetorch.model.parameters import Parameters
import torch

storagedir : str = "geometry/points/"

class Points:
    def __init__(self, params: Parameters) -> None:
        self.parameters: Parameters = params
        self.position: storageTensor = storageTensor(Types.GeometryInfo, [Parameters.npoints, 3], storagedir+"position") # Position coordinates
        self.velocity: storageTensor = storageTensor(Types.GeometryInfo, [Parameters.npoints, 3], storagedir+"velocity") # Velocity vectors
        self.cum_n_neighbors: storageTensor = storageTensor(Types.IndexInfo, [Parameters.npoints], storagedir+"cum_n_neighbors") # cumulative number of neighbors
        self.n_neighbors: storageTensor = storageTensor(Types.IndexInfo, [Parameters.npoints], storagedir+"n_neighbors") # number of neighbors per point
        self.neighbors: storageTensor = storageTensor(Types.IndexInfo, [None], storagedir+"neighbors") # linearized neighbors vector

