from enum import Enum
from typing import List, Union, Dict, Tuple
from magrittetorch.model.geometry.points import Points
from magrittetorch.model.geometry.boundary import Boundary, BoundaryType 
from magrittetorch.model.geometry.rays import Rays
from magrittetorch.model.parameters import Parameters
from magrittetorch.utils.storagetypes import DataCollection, Types
from magrittetorch.model.parameters import Parameter, EnumParameter
from magrittetorch.algorithms.torch_algorithms import multi_arange
from magrittetorch.utils.constants import min_dist
from astropy.constants import c
import torch

#err, do we need these two enums?
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
        self.boundary: Boundary = Boundary(params, self.dataCollection)
        self.points: Points = Points(params, self.dataCollection)
        self.rays: Rays = Rays(params, self.dataCollection)
        self.geometryType: EnumParameter[GeometryType, type[GeometryType]] = EnumParameter(GeometryType, "geometryType", ("spherical_symmetry", self.__legacy_convert_geometryType)); self.dataCollection.add_local_parameter(self.geometryType)

    def __legacy_convert_geometryType(self, is_spherically_symmetric: bool) -> GeometryType:
        print("is the model spherically symmetric", is_spherically_symmetric, type(is_spherically_symmetric))
        match is_spherically_symmetric:
            case "true":
                return GeometryType.SpericallySymmetric1D
            case "false":
                return GeometryType.General3D
            case _:
                raise KeyError("Something went wrong with setting the model geometry")
    
    def distance_in_direction_3D_geometry(self, origin_coords : torch.Tensor, raydirection : torch.Tensor, points_position : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the distances to the points in a 3D setting

        Args:
            origin_coords (torch.Tensor): Coordinates of the origin of the rays
            raydirection (torch.Tensor): Direction of the rays
            points_idx (torch.Tensor): Point indices of the points in the 3D geometry

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
            origin_coords (torch.Tensor): 3D Coordinates of the origin points. Has dimensions [NPOINTS, 3]
            raydirection (torch.Tensor): Ray direction. Has dimensions [3]
            curr_points_index (torch.Tensor): Current point indices. Has dimensions [NPOINTS]
            distance_travelled (torch.Tensor): Current travelled distance. Has dimensions [NPOINTS]
            device (torch.device): Device on which to compute.
            positions_device (torch.Tensor): Positions of all points in the model. Has dimensions [parameters.npoints]. TODO: get in this function, reduce number of function arguments.
            neighbors_device (torch.Tensor): Linearized neighbors of all points in the given direction. Has dimensions [sum(n_neighbors_device)]
            n_neighbors_device (torch.Tensor): Number of neighbors per point in the given direction. Has dimensions [parameters.npoints]
            cum_n_neighbors_device (torch.Tensor): Cumulative number of neighbors per point in the given direction (starts at 0). Has dimensions [parameters.npoints]

        Raises:
            NotImplementedError: If get_next has not yet been implemented for the geometryType.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Next point indices, accumulated distance to the next points. Both torch.Tensors have dimensions [NPOINTS]
        """
        match self.geometryType.get():
            case GeometryType.General3D:
                return self.get_next_3D_geometry(origin_coords, raydirection, curr_points_index, device, positions_device, neighbors_device, n_neighbors_device, cum_n_neighbors_device)
            case GeometryType.SpericallySymmetric1D:
                return self.get_next_1D_spherical_symmetry(origin_coords, raydirection, curr_points_index, distance_travelled, device, positions_device)
            case _:
                raise NotImplementedError("Geometry type not yet supported: ", self.geometryType)


    def get_next_3D_geometry(self, origin_coords : torch.Tensor, raydirection : torch.Tensor, curr_points_index : torch.Tensor, device : torch.device, positions_device :torch.Tensor, neighbors_device : torch.Tensor, n_neighbors_device : torch.Tensor, cum_n_neighbors_device : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the next point on the ray in a general 3D geometry, given the raydirection.

        Args:
            origin_coords (torch.Tensor): 3D Coordinates of the origin points. Has dimensions [NPOINTS, 3]
            raydirection (torch.Tensor): Ray direction. Has dimensions [3]
            curr_points_index (torch.Tensor): Current point indices. Has dimensions [NPOINTS]
            device (torch.device): Device on which to compute.
            positions_device (torch.Tensor): Positions of all points in the model. Has dimensions [parameters.npoints]. TODO: get in this function, reduce number of function arguments.
            neighbors_device (torch.Tensor): Linearized neighbors of all points in the given direction. Has dimensions [sum(n_neighbors_device)]
            n_neighbors_device (torch.Tensor): Number of neighbors per point in the given direction. Has dimensions [parameters.npoints]
            cum_n_neighbors_device (torch.Tensor): Cumulative number of neighbors per point in the given direction (starts at 0). Has dimensions [parameters.npoints]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Next point indices, accumulated distance to the next points. Both torch.Tensors have dimensions [NPOINTS]
        """
        masked_n_neighbors = n_neighbors_device[curr_points_index]# dims: [NPOINTS]
        input_size = origin_coords.size(dim=0)

        #obtain which (indices of) neighbors to check in the linearized neighbors list 
        indices_to_check = multi_arange(cum_n_neighbors_device[curr_points_index], n_neighbors_device[curr_points_index], device) #dims: [sum(n_neighbors_device[curr_points_index])]
        #duplicate index of curr_points_index correspondingly
        lengthening_indices = torch.arange(masked_n_neighbors.size(dim=0), device=device).repeat_interleave(masked_n_neighbors) #dims: [sum(n_neighbors_device[curr_points_index])]
        lengthened_origin_coords = origin_coords[lengthening_indices, :]# dims: [sum(n_neighbors_device[curr_points_index]), 3]

        neighbors_to_check = torch.gather(neighbors_device, 0, indices_to_check) # dims: [sum(n_neighbors_device[curr_points_index])]

        #Get distances and compare with distance of current point; TODO: maybe use squared distance instead for ydist
        xdistance, ydistance = self.distance_in_direction_3D_geometry(lengthened_origin_coords, raydirection, positions_device[neighbors_to_check]) # dims: [sum(n_neighbors_device[curr_points_index])] for both

        #Figure out the indices of the closest points
        tempstuff = torch.zeros(input_size, device=device, dtype=Types.GeometryInfo) #dims: [NPOINTS]
        minydists_per_point = tempstuff.scatter_reduce(0, lengthening_indices, ydistance, reduce="amin", include_self=False) #dims: [NPOINTS]
        #broadcast these minimal distances once more, using gather
        minydists = minydists_per_point.gather(0, lengthening_indices) #dims: [sum(n_neighbors_device[curr_points_index])]
        #Ties can arise with the computed ydistance's, so the dimension can be larger than NPOINTS
        minindices = torch.nonzero(minydists == ydistance).flatten() #dims: [>=NPOINTS]
        #torch.nonzero likes to transpose the matrix for some reason
        corresp_scatter_ids = torch.gather(lengthening_indices, 0, minindices) #dims: [>=NPOINTS]

        #If equal distances would arise, the resulting dimension would be wrong. Thus I only use the first of each scatter (corresponding to the curr_points_indices)
        #TODO: figure out cheaper way to do this
        first_result_of_each_scatter = torch.searchsorted(corresp_scatter_ids, torch.arange(input_size, device=device).type(torch.int64)) #dims: [NPOINTS]
        next_idx_of_neighbors_to_check = minindices[first_result_of_each_scatter] #dims:[NPOINTS]

        next_index = neighbors_to_check[next_idx_of_neighbors_to_check] #dims: [NPOINTS]
        next_dist_travelled = xdistance[next_idx_of_neighbors_to_check] #dims: [NPOINTS]

        return next_index, next_dist_travelled

    def get_next_1D_spherical_symmetry(self, origin_coords : torch.Tensor, raydirection : torch.Tensor, curr_points_index : torch.Tensor, distance_travelled: torch.Tensor, device : torch.device, positions_device : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the next shell on the ray given the raydirection and the currently travelled distance.

        Args:
            origin_coords (torch.Tensor): 3D Coordinates of the origin points. Has dimensions [NPOINTS, 3]
            raydirection (torch.Tensor): Ray direction. Has dimensions [3]
            curr_points_index (torch.Tensor): Current point indices. Has dimensions [NPOINTS]
            device (torch.device): Device on which to compute.
            positions_device (torch.Tensor): Positions of all points in the model. Has dimensions [parameters.npoints]. TODO: get in this function, reduce number of function arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Next point indices, accumulated distance to the next points. Both torch.Tensors have dimensions [NPOINTS]
        """
        #In 1D, use the geometry to your advantage for determining the ray tracing procedure
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
        #thus restructured for less computations
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
        Rsin : torch.Tensor = torch.sqrt(torch.sum(origin_coords**2, dim=1) - Rcos**2 + min_dist)
        #sqrt of 0 is not differentiable, so +min_dist required to allow gradient computation
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
        next_distance = Rcos + (1-2*in_criterion) * delta_dist

        return next_index, next_distance
    
    def get_doppler_shift(self, curr_point: torch.Tensor, origin_position:torch.Tensor, origin_velocity: torch.Tensor, raydir: torch.Tensor, distance_travelled: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Computes the doppler shift for arbitrary geometries by assuming non-relativistic velocities. 

        Args:
            curr_point (torch.Tensor): Indices of the points for which we want to compute the doppler shift. Has dimensions [NPOINTS]
            origin_position (torch.Tensor): 3D position of the origin of the ray. Is only used in spherically symmetric geometries. Has dimensions [NPOINTS, 3]
            origin_velocity (torch.Tensor): 3D velocity of the origin of the ray. Has dimensions [NPOINTS, 3]
            raydir (torch.Tensor): Ray directions. Has dimensions [NPOINTS, 3]
            distance_travelled (torch.Tensor): Distance travelled by the ray. Is only used in spherically symmetric geometries. Has dimensions [NPOINTS]
            device (torch.device, optional): Device on which to compute.

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
        projected_distance_curr_to_origin = torch.sum(origin_position * raydir, dim=1) + distance_travelled
        return 1.0 + (current_velocity * projected_distance_curr_to_origin / curr_radius - torch.sum(origin_velocity * raydir, dim=1))/c.value#type: ignore
            

    def get_neighbors_in_direction(self, raydir: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the subset of the neighbors which lie in the given direction. For 3D geometries, it excludes the neighbors in the wrong direction.
        For 1D geometries, this returns the full neighbors, as no shells can be excluded only based on ray direction.

        Args:
            raydir (torch.Tensor): The ray direction. Has dimensions [3]
            device (torch.device): The device on which to compute and store the torch.Tensors

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Linearized neighbors tensor in correct direction, corresponding number of neighbors per point and cumulative number of neighbors.
            The tensors respectively have dimensions [sum(n_neighbors_in_correct_direction)], [parameters.npoints] and [parameters.npoints]
        """
        match self.geometryType.get():
            case GeometryType.General3D:
                return self.points.get_neighbors_in_direction_3D_geometry(raydir, device)
            case GeometryType.SpericallySymmetric1D:
                return self.points.neighbors.get(device), self.points.n_neighbors.get(device), self.points.get_cum_n_neighbors(device)
            case _:
                raise TypeError("Geometry type not yet supported: ", self.geometryType)
            
    def get_starting_distance_3D_geometry(self, raydir: torch.Tensor, starting_indices: torch.Tensor, starting_positions: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Computes the distance between the starting positions and the location of the starting indices, given the ray direction, for a `General3D` geometry.

        Args:
            raydir (torch.Tensor): The ray direction. Has dimensions [3].
            starting_indices (torch.Tensor): The starting indices for the rays. Has dimensions [NPOINTS].
            starting_positions (torch.Tensor): The starting positions for the rays. Has dimensions [NPOINTS, 3].
            device (torch.device): The device on which to compute.

        Returns:
            torch.Tensor: The starting distances. Has dimensions [NPOINTS].
        """
        distance_on_ray : torch.Tensor = torch.sum((self.points.position.get(device)[starting_indices]-starting_positions) * raydir, dim = 1)
        return distance_on_ray
    
    def get_starting_distance_1D_spherical_symmetry(self, raydir: torch.Tensor, starting_indices: torch.Tensor, starting_positions: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Computes the distance between the starting positions and the location of the starting indices, given the ray direction, for a `SpericallySymmetric1D` geometry.
        Warning: Assumes the starting_positions to be either at the positions of the corresponding starting_indices, or outside the model.
        
        Args:
            raydir (torch.Tensor): The ray direction. Has dimensions [3].
            starting_indices (torch.Tensor): The starting indices for the rays. Has dimensions [NPOINTS].
            starting_positions (torch.Tensor): The starting positions for the rays. Has dimensions [NPOINTS, 3].
            device (torch.device): The device on which to compute.

        Returns:
            torch.Tensor: The starting distances. Has dimensions [NPOINTS].
        """
        distance = torch.zeros_like(starting_indices, dtype=Types.GeometryInfo, device=device) #dims: [NPOINTS]
        maxdist = self.points.position.get(device)[:,0].max() #dims: [1]
        # Only compute the distance for the rays that start outside the model
        # For rays which start inside, but are not at the corresponding position, we could use the same algorithm as in get_next_1D_spherical_symmetry, 
        # but we currently have not use for this functionality
        starting_outside = torch.sum(starting_positions**2, dim=1) > maxdist**2 #dims: [NPOINTS]
        projected_positions_outside = starting_positions[starting_outside] - raydir[None, :] * torch.sum(starting_positions[starting_outside] * raydir, dim=1, keepdim=True) #dims: [NPOINTS, 3]
        rsquared = maxdist**2 - torch.sum(projected_positions_outside**2, dim=1) #dims: [NPOINTS]
        # Negative values of rsquared are invalid to sqrt, so we set them to the minimal allowed distance
        rsquared[rsquared<=min_dist**2] = min_dist**2 #dims: [NPOINTS]
        distance[starting_outside] = maxdist - torch.sqrt(rsquared) #dims: [NPOINTS]
        return distance


    
    def get_starting_distance(self, raydir: torch.Tensor, starting_indices: torch.Tensor, starting_positions: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Computes the distance between the starting positions and the location of the starting indices, given the ray direction.

        Args:
            raydir (torch.Tensor): The ray direction. Has dimensions [3]
            starting_indices (torch.Tensor): The starting indices for the rays. Has dimensions [NPOINTS]
            starting_positions (torch.Tensor): The starting positions for the rays. Has dimensions [NPOINTS, 3]
            device (torch.device): The device on which to compute

        Returns:
            torch.Tensor: The starting distances. Has dimensions [NPOINTS]
        """
        match self.geometryType.get():
            case GeometryType.General3D:
                return self.get_starting_distance_3D_geometry(raydir, starting_indices, starting_positions, device)
            case GeometryType.SpericallySymmetric1D:
                return self.get_starting_distance_1D_spherical_symmetry(raydir, starting_indices, starting_positions, device)
            case _:
                raise TypeError("Geometry type not yet supported: ", self.geometryType)
