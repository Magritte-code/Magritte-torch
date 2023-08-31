from enum import Enum
from typing import List, Union, Dict, Tuple
from magrittetorch.model.geometry.points import Points
from magrittetorch.model.geometry.boundary import Boundary
from magrittetorch.model.geometry.rays import Rays
from magrittetorch.model.parameters import Parameters
from magrittetorch.utils.storagetypes import DataCollection, Types
from magrittetorch.model.parameters import Parameter
from magrittetorch.algorithms.torch_algorithms import multi_arange
from magrittetorch.utils.constants import min_dist
from astropy.constants import c
import torch

class Frame(Enum):
    CoMoving = 0
    Rest = 1

class Tracer(Enum):
    Defaulttracer = 0
    Imagetracer = 1

class GeometryType(Enum):
    General3D: int = 0 #TODO: add options for the general3D geometry: specify which type of boundary is expected; axisalignedcube or sphere (for easier raytracing; Note: enum inside enum looks ridiculous, so just add extra enums to this list)
    SpericallySymmetric1D: int = 1

class Geometry:
    def __init__(self, params: Parameters, dataCollection: DataCollection) -> None:
        self.parameters: Parameters = params
        self.dataCollection : DataCollection = dataCollection
        self.points: Points = Points(params, self.dataCollection)
        self.boundary: Boundary = Boundary(params, self.dataCollection)
        self.rays: Rays = Rays(params, self.dataCollection)
        self.geometryType: Parameter[GeometryType] = Parameter("geometryType", ("spherical_symmetry", self.__legacy_convert_geometryType)); self.dataCollection.add_local_parameter(self.geometryType)

    def __legacy_convert_geometryType(self, is_spherically_symmetric: bool) -> GeometryType:
        print("is the model spherically symmetric", is_spherically_symmetric, type(is_spherically_symmetric))
        match is_spherically_symmetric:
            case "true":
                return GeometryType.SpericallySymmetric1D
            case "false":
                return GeometryType.General3D
            case _:
                raise KeyError("Something went wrong with setting the model geometry")
    
    # def map_data_to_device(self) -> None:
    #     self.points.map_data_to_device()
    
    def distance_in_direction_3D_geometry(self, origin_coords : torch.Tensor, raydirection : torch.Tensor, points_position : torch.Tensor, distance_travelled : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the distances to the shells in a 3D setting

        Args:
            origin_coords (torch.Tensor): Coordinates of the origin of the rays
            raydirection (torch.Tensor): Direction of the rays
            points_idx (torch.Tensor): Point indices of the points in the 3D geometry
            distance_travelled (torch.Tensor): NOT USED. Neccesary for having the same argument list as Geometry.distance_in_direction_1D_spherical_symmetry.

        Returns:
            torch.Tensor: Distance travelled on the rays to reach the given points, with respect to the origin_coords and the raydirection.
            torch.Tensor: Perpendicular distance of the rays to the points
        """
        distance_on_ray : torch.Tensor = torch.sum((points_position-origin_coords) * raydirection, dim = 1)

        total_distance2 : torch.Tensor = torch.sum((points_position-origin_coords)**2, dim = 1)
        return distance_on_ray, torch.sqrt(total_distance2-distance_on_ray**2)
    
    # def distance_in_direction_1D_spherical_symmetry(self, origin_coords : torch.Tensor, raydirection : torch.Tensor, points_position : torch.Tensor, distance_travelled : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """Computes the distances to the shells in a 1D spherically symmetric setting

    #     Args:
    #         origin_coords (torch.Tensor): (3D) Coordinates of the origin of the rays
    #         raydirection (torch.Tensor): (3D) Direction of the rays
    #         points_idx (torch.Tensor): Point indices of the 1D shells
    #         distance_travelled (torch.Tensor): Currently travelled distance along the ray, compared to origin. Used for determining which intersection with a ray to use (going in or out)

    #     Returns:
    #         torch.Tensor: Distance of the shells on the rays, with respect to the origin_coords and the raydirection.
    #         torch.Tensor: Ordering hint for which shell to prefer. Lower is better.
    #     """
    #     radii : torch.Tensor = points_position[:,0] #shell radii are assumed to be stored in the x-coordinate of the position
    #     vertical_dist_center : torch.Tensor = torch.sum((torch.zeros(origin_coords.size(), dtype=torch.float)-origin_coords) * raydirection, dim=1)
    #     horizontal_dist_center : torch.Tensor = torch.sqrt(torch.linalg.vector_norm(origin_coords, dim=1)**2-vertical_dist_center**2)
    #     mask_ray_intersects_shell : torch.Tensor = (radii > horizontal_dist_center).type(torch.float64)
    #     # mask_ray_farther_than_first_shell_encounter = (distance_travelled >= vertical_dist_center).type(torch.float)
    #     delta_dist = torch.sqrt((radii**2-horizontal_dist_center**2)*mask_ray_intersects_shell)
    #     mask_ray_farther_than_first_shell_encounter = (distance_travelled >= vertical_dist_center - delta_dist).type(torch.float)
    #     distance_on_ray : torch.Tensor = vertical_dist_center + (2 * mask_ray_farther_than_first_shell_encounter - 1) * delta_dist
    #     # return distance_on_ray, torch.ones_like(distance_on_ray) 
    #     return distance_on_ray, torch.ones_like(distance_on_ray) - mask_ray_intersects_shell
    #     #slightly deprioritize the second shell encounter
    #     # return distance_on_ray, -distance_on_ray

    # def distance_in_direction(self, origin_coords : torch.Tensor, raydirection : torch.Tensor, points_position : torch.Tensor, distance_travelled : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """Computes the distances to the points/shells in the current geometry

    #     Args:
    #         origin_coords (torch.Tensor): (3D) Coordinates of the origin of the rays
    #         raydirection (torch.Tensor): (3D) Direction of the rays
    #         points_idx (torch.Tensor): Point indices of the points/shells
    #         distance_travelled (torch.Tensor): Currently travelled distance along the ray. Used in 1D spherical symmetry 

    #     Returns:
    #         torch.Tensor: Distance of the shells on the rays, with respect to the origin_coords and the raydirection.
    #         torch.Tensor: Information for ordering which points to prefer for the next point on the ray. Lower is better.
    #     """
    #     match self.geometryType:
    #         case geometryType.General3D:
    #             return self.distance_in_direction_3D_geometry(origin_coords, raydirection, points_position, distance_travelled)
    #         case geometryType.SpericallySymmetric1D:
    #             return self.distance_in_direction_1D_spherical_symmetry(origin_coords, raydirection, points_position, distance_travelled)
    #         case _:
    #             raise TypeError("Geometry type not yet supported: ", self.geometryType)

    def get_next(self, origin_coords : torch.Tensor, raydirection : torch.Tensor, curr_points_index : torch.Tensor, 
                 distance_travelled : torch. Tensor, device : torch.device, positions_device :torch.Tensor, neighbors_device : torch.Tensor, n_neighbors_device : torch.Tensor, cum_n_neighbors_device : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the next set of points on the ray, given the current points and ray direction.

        Args:
            origin_coords (torch.Tensor): Coordinates of the origin points. Has dimensions [NPOINTS, 3]
            raydirection (torch.Tensor): Ray direction. Has dimensions [3]
            curr_points_index (torch.Tensor): Current point indices. Has dimensions [NPOINTS]
            distance_travelled (torch.Tensor): Current travelled distance. Has dimensions [NPOINTS]
            device (torch.device): Device on which to compute.
            positions_device (torch.Tensor): Positions of all points in the model. TODO: get in this function. Has dimensions [parameters.npoints]
            neighbors_device (torch.Tensor): Linearize neighbors of all points in the given direction. TODO: actually implement different neighbors per direction. Has dimensions [sum(n_neighbors_device)]
            n_neighbors_device (torch.Tensor): Number of neighbors per point in the given direction. SAME TODO. Has dimensions [parameters.npoints]
            cum_n_neighbors_device (torch.Tensor): Cumulative number of neighbors per point (starts at 0). SAME TODO. Has dimensions [parameters.npoints]

        Raises:
            TypeError: _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Next point indices, accumulated distance to the next points. Both torch.Tensors have dimensions [NPOINTS]
        """
        match self.geometryType.get():
            case GeometryType.General3D:
                return self.get_next_3D_geometry(origin_coords, raydirection, curr_points_index, distance_travelled, device, positions_device, neighbors_device, n_neighbors_device, cum_n_neighbors_device)
            case GeometryType.SpericallySymmetric1D:
                return self.get_next_1D_spherical_symmetry(origin_coords, raydirection, curr_points_index, distance_travelled, device, positions_device)
            case _:
                raise TypeError("Geometry type not yet supported: ", self.geometryType)


    def get_next_3D_geometry(self, origin_coords : torch.Tensor, raydirection : torch.Tensor, curr_points_index : torch.Tensor, distance_travelled: torch.Tensor, device : torch.device, positions_device :torch.Tensor, neighbors_device : torch.Tensor, n_neighbors_device : torch.Tensor, cum_n_neighbors_device : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # positions_device = self.points.position.get().to(device=device)
        # neighbors_device = self.points.neighbors.get().to(device=device)
        # n_neighbors_device = self.points.n_neighbors.get().to(device=device)
        # cum_n_neighbors_device = self.points.get_cum_n_neighbors().to(device=device)
        masked_n_neighbors = n_neighbors_device[curr_points_index]
        input_size =origin_coords.size(dim=0)

        # indices_to_check = multi_arange(torch.gather(cum_n_neighbors_device, 0, curr_points_index), torch.gather(n_neighbors_device, 0, curr_points_index), device)
        indices_to_check = multi_arange(cum_n_neighbors_device[curr_points_index], n_neighbors_device[curr_points_index], device)
        # lengthened_curr_ids = torch.repeat_interleave(curr_points_index, masked_n_neighbors)
        # lengthened_origin_coords = torch.repeat_interleave(origin_coords, masked_n_neighbors, dim=0)
        # lengthened_distance_travelled = torch.repeat_interleave(distance_travelled, masked_n_neighbors)
        
        lengthening_indices = torch.arange(masked_n_neighbors.size(dim=0), device=device).repeat_interleave(masked_n_neighbors)
        lengthened_curr_ids = curr_points_index[lengthening_indices]
        lengthened_origin_coords = origin_coords[lengthening_indices, :]
        lengthened_distance_travelled = distance_travelled[lengthening_indices]

        neighbors_to_check = torch.gather(neighbors_device, 0, indices_to_check)

        #Get distances and compare with distance of current point; note: some computation might be wasted
        xdistance, ydistance = self.distance_in_direction_3D_geometry(lengthened_origin_coords, raydirection, positions_device[neighbors_to_check], lengthened_distance_travelled)
        currdistance, _ = self.distance_in_direction_3D_geometry(lengthened_origin_coords, raydirection, positions_device[lengthened_curr_ids], lengthened_distance_travelled)

        wrongdirection = xdistance <= currdistance
        #add penalty to points in wrong direction, making sure that it is never the correct point;
        #note: penalty 1 plus max value to make sure that these penalized values are higher than any original value
        # ydistance += wrongdirection.type(Types.GeometryInfo) * (1.0 + torch.max(ydistance, 0)[0])
        ydistance += wrongdirection * (1.0 + torch.max(ydistance, 0)[0])

        scatter_ids = torch.repeat_interleave(torch.arange(input_size, device=device).type(torch.int64), masked_n_neighbors)

        tempstuff = torch.zeros(input_size, device=device, dtype=Types.GeometryInfo)
        minydists_per_point = tempstuff.scatter_reduce(0, scatter_ids, ydistance, reduce="amin", include_self=False)#TODO: technically, we can reuse x dist at this point
        #broadcast these minimal distances once more, using gather
        minydists = minydists_per_point.gather(0, scatter_ids)
        minindices = torch.nonzero(minydists == ydistance).flatten()#torch.nonzero likes to transpose the matrix for some reason
        corresp_scatter_ids = torch.gather(scatter_ids, 0, minindices)

        #If equal distances would arise, the resulting dimension would be wrong. Thus I only use the first of each scatter (corresponding to the curr_points_indices)
        first_result_of_each_scatter = torch.searchsorted(corresp_scatter_ids, torch.arange(input_size, device=device).type(torch.int64))
        next_idx_of_neighbors_to_check = minindices[first_result_of_each_scatter]

        next_index = neighbors_to_check[next_idx_of_neighbors_to_check]
        next_dist_travelled = xdistance[next_idx_of_neighbors_to_check]

        return next_index, next_dist_travelled

    def get_next_1D_spherical_symmetry(self, origin_coords : torch.Tensor, raydirection : torch.Tensor, curr_points_index : torch.Tensor, distance_travelled: torch.Tensor, device : torch.device, positions_device : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #In 1D, use the geometry to your advantage for determining the ray tracing procedure
        #dev note: add a variant for the 3D raytracing procedure, but not for this (headaches will ensue)
        #TODO: bench several minor variants of this algorithm (check whether precomputing the outerneighbors is wasted effort)

        #1D algorithm:
        #if previous radius intersects:
        # check if we are past halfway:
        #  if yes: just go outwards
        #  if no: go inwards
        #otherwise:
        # check if we are past halfway:
        #  if yes: go outwards
        #  if no: stay on same index, but return post halfway distance
        #
        #thus restructured
        #
        #if prev intersect and before half -> go inwards
        #if post-halfway -> go outwards
        #distance returning: only first half if go inwards; otherwise always second half


        #first determine which point to be the next point, using the only two options available; also make sure that we do not go out of bounds with these indices
        innerneighbors : torch.Tensor = torch.maximum(0*torch.ones_like(curr_points_index), curr_points_index-1)
        outerneighbors : torch.Tensor = torch.minimum((self.parameters.npoints.get()-1)*torch.ones_like(curr_points_index), curr_points_index+1)
        next_index = curr_points_index
        radii_device = positions_device[:,0]
        radii_in = radii_device[innerneighbors]
        Rcos : torch.Tensor = torch.sum(-origin_coords * raydirection, dim=1)
        Rsin : torch.Tensor = torch.sqrt(torch.sum(origin_coords**2, dim=1) - Rcos**2 + min_dist)#sqrt of 0 is not differentiable
        # print("RSIN", Rsin)
        ray_intersects_inner : torch.Tensor = radii_in > Rsin
        ray_travelled_past_half: torch.Tensor = distance_travelled >= Rcos

        in_criterion = torch.logical_and(ray_intersects_inner, torch.logical_not(ray_travelled_past_half))
        out_criterion = ray_travelled_past_half

        next_index[in_criterion] = innerneighbors[in_criterion]
        next_index[out_criterion] = outerneighbors[out_criterion]

        #finally compute the distances for the new indices
        radii_next = torch.gather(radii_device, 0, next_index)
        mask_ray_intersects_shell : torch.Tensor = (radii_next > Rsin).type(Types.GeometryInfo)
        delta_dist = torch.sqrt((radii_next**2-Rsin**2)*mask_ray_intersects_shell + min_dist)#distance between intersections
        #sqrt of 0 is not differentiable, so +min_dist required to allow gradient computation
        # print("delta dist", delta_dist)
        next_distance = Rcos + (1-2*in_criterion) * delta_dist

        return next_index, next_distance
    
    def get_doppler_shift(self, curr_point: torch.Tensor, origin_position:torch.Tensor, origin_velocity: torch.Tensor, raydir: torch.Tensor, distance_travelled: torch.Tensor, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Computes the doppler shift for arbitrary geometries by assuming non-relativistic velocities. 

        Args:
            curr_point (torch.Tensor): Indices of the points for which we want to compute the doppler shift. Has dimensions [NPOINTS]
            origin_position (torch.Tensor): 3D position of the origin of the ray. Is only used in spherically symmetric geometries. Has dimensions [NPOINTS, 3]
            origin_velocity (torch.Tensor): 3D velocity of the origin of the ray. Has dimensions [NPOINTS, 3]
            raydir (torch.Tensor): Ray directions. Has dimensions [NPOINTS, 3]
            distance_travelled (torch.Tensor): Distance travelled by the ray. Is only used in spherically symmetric geometries. Has dimensions [NPOINTS]
            device (torch.device, optional): Device on which to compute. Defaults to torch.device("cpu").

        Raises:
            TypeError: If the geometry type has not yet been implemented.

        Returns:
            torch.Tensor: The doppler shift factor for shifting from the origin frame to the comoving frame. Multiply with the origin frequency to compare versus line comoving frequencies. Has dimensions [NPOINTS]
        """
        curr_velocity_device = self.points.velocity.get(device)[curr_point]
        match self.geometryType.get():
            case GeometryType.General3D:
                return self.get_doppler_shift_3D_geometry(curr_velocity_device, origin_velocity, raydir)
            case GeometryType.SpericallySymmetric1D:
                curr_radius_device = self.points.position.get(device)[curr_point,0]
                return self.get_doppler_shift_1D_spherical_symmetry(curr_velocity_device[:,0], origin_position, origin_velocity, raydir, curr_radius_device, distance_travelled)
            case _:
                raise TypeError("Geometry type not yet supported: ", self.geometryType)
            
    def get_doppler_shift_3D_geometry(self, current_velocity: torch.Tensor, origin_velocity: torch.Tensor, raydir: torch.Tensor) -> torch.Tensor:
        """Returns the doppler shift factor for a 3D geometry by assuming non-relativistic velocities

        Args:
            current_velocity (torch.Tensor): Velocity at the point for which we want the doppler shift. Has dimensions [NPOINTS, 3]
            origin_velocity (torch.Tensor): Velocity at the origin of the ray. Has dimensions [NPOINTS, 3]
            raydir (torch.Tensor): Ray directions. Has dimensions [NPOINTS, 3]

        Returns:
            torch.Tensor: The doppler shift. Has dimension [NPOINTS]
        """
        return 1.0 + torch.sum((current_velocity-origin_velocity)*raydir, dim=1)/c.value#type: ignore
        
    def get_doppler_shift_1D_spherical_symmetry(self, current_velocity: torch.Tensor, origin_position: torch.Tensor, origin_velocity: torch.Tensor, raydir: torch.Tensor, curr_radius: torch.Tensor, distance_travelled: torch.Tensor) -> torch.Tensor:
        """Returns the doppler shift factor for a 1D spherically symmetric geometry by assuming non-relativistic velocities

        Args:
            current_velocity (torch.Tensor): 1D spherically symmetric velocity at the point for which we want the doppler shift. Has dimensions [NPOINTS]
            origin_position (torch.Tensor): 3D position of the origin of the ray. Has dimensions [NPOINTS, 3]
            origin_velocity (torch.Tensor): 3D velocity at the origin of the ray. Has dimensions [NPOINTS, 3]
            raydir (torch.Tensor): Ray directions. Has dimensions [NPOINTS, 3]
            curr_radius (torch.Tensor): 1D radius of the point for which we want the doppler shift. Has dimensions [NPOINTS]
            distance_travelled (torch.Tensor): Currently travelled distance on the ray. Has dimensions [NPOINTS]

        Returns:
            torch.Tensor: The doppler shift. Has dimensions [NPOINTS]
        """
        distance_to_origin = torch.sum(-origin_position*raydir, dim=1) #dims: [NPOINTS]
        distance_diff = distance_travelled - distance_to_origin #dims: [NPOINTS]
        # print("fraction should be below 1", distance_diff/curr_radius, distance_diff, distance_to_origin)#debug
        # Note: projected velocity of the shell onto the ray is = distance_diff/radius * current_velocity.
        # print("curr part", current_velocity * distance_diff / curr_radius, "origin part", torch.sum(origin_velocity * raydir, dim = 1), origin_velocity)
        return 1.0 + (current_velocity * distance_diff / curr_radius - torch.sum(origin_velocity * raydir, dim=1))/c.value#type: ignore
            