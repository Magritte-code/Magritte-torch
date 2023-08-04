from enum import Enum
from typing import List, Union, Dict
from magrittetorch.model.geometry.points import Points
from magrittetorch.model.parameters import Parameters
import torch

class Frame(Enum):
    CoMoving = 0
    Rest = 1

class Tracer(Enum):
    Defaulttracer = 0
    Imagetracer = 1

class GeometryType(Enum):
    General3D = 0
    SpericallySymmetric1D = 1

class Rays:
    pass  # Define the Rays class here

class Boundary:
    pass  # Define the Boundary class here

class Geometry:
    def __init__(self, params : Parameters) -> None:
        self.parameters: Parameters = params
        self.points: Points = Points(params)
        self.rays: Union[Rays, None] = None
        self.boundary: Union[Boundary, None] = None


    
    def distance_in_direction_3D_geometry(self, origin_coords : List[float], raydirection : torch.Tensor, points_idx : torch.Tensor) -> torch.Tensor[float]:
        return torch.tensordot(self.points.position.get().gather(points_idx)-origin_coords, raydirection, dims=([1], [1]))

