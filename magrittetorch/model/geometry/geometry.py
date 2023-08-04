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


    
    def distance_in_direction_3D_geometry(self, origin_coords : torch.Tensor, raydirection : torch.Tensor, points_idx : torch.Tensor, distance_travelled : torch.Tensor) -> torch.Tensor:
        """Computes the distances to the shells in a 3D setting

        Args:
            origin_coords (torch.Tensor): Coordinates of the origin of the rays
            raydirection (torch.Tensor): Direction of the rays
            points_idx (torch.Tensor): Point indices of the points in the 3D geometry
            distance_travelled (torch.Tensor): NOT USED. Neccesary for having the same argument list as Geometry.distance_in_direction_1D_spherical_symmetry.

        Returns:
            torch.Tensor: Distance of the points on the rays, with respect to the origin_coords and the raydirection.
        """
        pointpositions : torch.Tensor = self.points.position.get().gather(dim=0, index = points_idx)
        return torch.tensordot(pointpositions-origin_coords, raydirection, dims=([1], [1])) # type: ignore
        ## pytorch itself does not report return types for tensordot, so type needs to be ignored
    
    def distance_in_direction_1D_spherical_symmetry(self, origin_coords : torch.Tensor, raydirection : torch.Tensor, points_idx : torch.Tensor, distance_travelled : torch.Tensor) -> torch.Tensor:
        """Computes the distances to the shells in a 1D spherically symmetric setting

        Args:
            origin_coords (torch.Tensor): (3D) Coordinates of the origin of the rays
            raydirection (torch.Tensor): (3D) Direction of the rays
            points_idx (torch.Tensor): Point indices of the 1D shells
            distance_travelled (torch.Tensor): Currently travelled distance along the ray. Used for determining which intersection with a ray to use (going in or out)

        Returns:
            torch.Tensor: Distance of the shells on the rays, with respect to the origin_coords and the raydirection.
        """
        radii : torch.Tensor = self.points.position.get()[:,0] #shell radii are assumed to be stored in the x-coordinate of the position
        vertical_dist_center : torch.Tensor = torch.tensordot(torch.zeros(origin_coords.size(), dtype=torch.float)-origin_coords, raydirection, dims=([1], [1]))
        horizontal_dist_center : torch.Tensor = torch.sqrt(torch.linalg.vector_norm(origin_coords, dim=1)**2-vertical_dist_center**2)
        mask_ray_inside_shell : torch.Tensor = (radii > horizontal_dist_center).type(torch.float)
        delta_dist = torch.sqrt((radii**2-horizontal_dist_center**2)*mask_ray_inside_shell)
        mask_ray_farther_than_first_shell_encounter = (distance_travelled >= vertical_dist_center-delta_dist).type(torch.float)
        return vertical_dist_center + (2 * mask_ray_farther_than_first_shell_encounter - 1) * delta_dist

