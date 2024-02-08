# TODO: will contain everything needed for tracing rays on gpu
# TODO: think about memory management when implementing
# TODO: deprecate this file, as it is not used anymore; see geometry.py for the new implementation

from magrittetorch.model.model import Model
from magrittetorch.utils.memorymapping import MemoryManager
from magrittetorch.model.model import Model
from magrittetorch.algorithms.torch_algorithms import multi_arange
from magrittetorch.utils.storagetypes import Types
from typing import Tuple, List
import torch


# def format_paths(index_matrix : torch.Tensor, distance_matrix : torch.Tensor, device : torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """Formats traced rays into a sparser format

#     Args:
#         index_matrix (torch.Tensor): Indices of the traced points. Has dimensions [MAX_RAYLENGTH, NPOINTS]
#         distance_matrix (torch.Tensor): Distances corresponding to the traced rays. Has dimensions [MAX_RAYLENGTH, NPOINTS]
#         device (torch.device): Device on which to compute this. Defaults to torch.device("cpu")

#     Returns:
#         Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Linearized tensors of the indices, of the corresponding distances and one with unique indices per ray. All torch.Tensors have dimensions [N_LINEAR_INDICES <= MAX_RAYLENGTH * NPOINTS]
#     """

#     #both have the same size
#     flatmatidx = index_matrix.flatten()
#     flatmatdist = distance_matrix.flatten()
#     length = index_matrix.size(dim=0)
#     width = index_matrix.size(dim=1)
#     # linear = torch.empty((length*width), device=device)#will contain the linearized values

#     # For making sure that successive rays have different distances for unique_consecutive
#     shift = (flatmatdist.max()+1) * (torch.arange(length*width, device=device) // width % 2)
#     temp = flatmatdist + shift

#     __, counts = torch.unique_consecutive(temp, return_counts = True)

#     indices = torch.cumsum(counts, dim=0).roll(1)#contains all indices of non-duplicate values
#     indices[0] = 0

#     linearidx = flatmatidx[indices]
#     lineardist = flatmatdist[indices]

#     reduce_ind = torch.arange(length, dtype=Types.IndexInfo, device=device).repeat_interleave(width)[indices]
    
#     return linearidx, lineardist, reduce_ind

# def trace_rays_sparse(model: Model, raydir : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """Ray tracing algorithm for unstructured grids, available on gpus. TODO: use generator approach instead, as this wastes significant time appending stuff to a single large array

#     Args:
#         model (Model): The model for which to apply the ray-tracing procedure
#         raydir (torch.Tensor): The direction in which to trace the rays. Has dimensions [3]

#     Returns:
#         Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Linearized tensors of the indices, of the corresponding distances with respect to the ray origin and one with unique indices per ray. All torch.Tensors have dimensions [N_LINEAR_INDICES]
#     """
#     #Sparse version should behave better when having imbalanced amounts of neighbors
#     #TODO: improve memory usage; 
#     # option 1: directly 'sparsify' rays when possible; reducing the memory footprint of every remaining
#     # option 2: apply memory management in loop outside this function; in this way, this function will be a limited get_next
#     #  then the generated data will directly be consumed by the solvers
#     #TODO: also implement version in which the origin positions can be specified


#     memhelper : MemoryManager = MemoryManager()
#     # memhelper.use_gpu = False#temporarily disable gpu computation
#     # maxn_neigbors : int = int(torch.max(model.geometry.points.n_neighbors.get()))
#     # expected_memory_usage : int = memhelper.compute_data_size_torch_tensor(model.parameters.npoints.get()*maxn_neigbors, Types.GeometryInfo)#Note: incomplete memory usage. Will crash for large models 
#     expected_memory_usage : int = memhelper.compute_data_size_torch_tensor(model.parameters.npoints.get(), Types.GeometryInfo)#Note: incomplete memory usage. Will crash for large models 
#     #TODO: figure out better estimate of memory usage
#     splits = memhelper.split_data(model.parameters.npoints.get(), expected_memory_usage)

#     raytrace = torch.empty((0), dtype=Types.IndexInfo) #very empty tensor for storing the result; TODO: might need to add the computed distances too
#     raydist = torch.empty((0), dtype=Types.GeometryInfo) #for storing the corresponding distances on the rays
#     rayscatter = torch.empty((0), dtype=Types.IndexInfo) #for storing the corresponding scatter indices of the rays
#     tuples : List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
#     for split, device, i in splits:
#         #unavoidable full mapping; costs O(Npoints*Nneighbors) memory
#         positions_device = model.geometry.points.position.get(device)
#         neighbors_device = model.geometry.points.neighbors.get(device)
#         #minor extra memory use
#         n_neighbors_device = model.geometry.points.n_neighbors.get(device)
#         cum_n_neighbors_device = model.geometry.points.get_cum_n_neighbors(device)
#         is_boundary_point_device = model.geometry.boundary.is_boundary_point.get(device)

#         #do computation per split
#         Nrays : int = split.size(dim=0)
#         raydir_device = raydir.to(device).reshape((1, 3))#we will only allow a single direction at a time to be traced, due to boundary point determination only being valid for a single direction at a
#         start_ids = split.to(device=device)

#         log_encountered_points = torch.empty((Nrays, 1), dtype=Types.IndexInfo, device=device)#on average 2x too large
#         log_encountered_points[:,0] = start_ids#on average 2x too large
#         distance_travelled = torch.zeros((Nrays), dtype=Types.GeometryInfo, device=device)
#         log_distances = torch.empty((Nrays, 1), device=device)
#         log_distances[:,0] = distance_travelled

#         curr_ids = start_ids.clone().type(Types.IndexInfo)
#         Nraystotrace = start_ids.size()
#         mask_active_rays = torch.ones(Nraystotrace, dtype=torch.bool, device=device)

#         while (torch.any(mask_active_rays)):
#             masked_curr_ids = curr_ids[mask_active_rays]
#             masked_start_ids = start_ids[mask_active_rays]
#             masked_distance_travelled = distance_travelled[mask_active_rays]
#             masked_origin_positions = positions_device[masked_start_ids]

#             next_ids, distances = model.geometry.get_next(masked_origin_positions, raydir_device, masked_curr_ids, masked_distance_travelled, device, positions_device, neighbors_device, n_neighbors_device, cum_n_neighbors_device)
#             distance_travelled[mask_active_rays] = distances
#             curr_ids[mask_active_rays] = next_ids

#             mask_active_rays = torch.logical_not(is_boundary_point_device[curr_ids])

#             log_encountered_points = torch.cat((log_encountered_points, curr_ids.reshape(-1,1)), dim = 1)
#             log_distances = torch.cat((log_distances, distance_travelled.reshape(-1,1)), dim = 1)

#         #map results back to cpu
#         #linearize results and map back to cpu
#         linearpos, lineardist, scatterind = format_paths(log_encountered_points, log_distances, device)
#         tuples.append((linearpos.to("cpu"), lineardist.to("cpu"), scatterind.to("cpu"), i))

#     sorted_tuples = sorted(tuples, key=lambda tup: tup[-1])
#     scatter_increment : torch.Tensor = torch.zeros((1), dtype=Types.IndexInfo)
#     for tuple in sorted_tuples:
#         raytrace = torch.cat((raytrace, tuple[0]), dim = 0)
#         raydist = torch.cat((raydist, tuple[1]), dim = 0)
#         rayscatter = torch.cat((rayscatter, tuple[2] + scatter_increment), dim=0)#make sure that every scatter value is unique per ray
#         scatter_increment = rayscatter[-1]
#     return raytrace, raydist, rayscatter
#     #TODO: also specify return device... ah well, in the generator approach, it makes more sense to not remap to cpu


def trace_rays_test_single_iteration(model: Model, raydir_device: torch.Tensor, device: torch.device):
    #performance analysis indicated that when most rays have stopped, the other raytracer only spent time on appending stuff. This is wasteful.
    #On another note, it is also impossible (without reverting to this formalism again) for me to accurately compute total optical depths, due to cumulative sums being required, applied to huge tensors.
    #thus we need this approach; TODO: bench vs old approach; DONE: appending is slow

    #unavoidable full mapping; costs O(Npoints*Nneighbors) memory
    positions_device = model.geometry.points.position.get(device)
    # neighbors_device = model.geometry.points.neighbors.get(device)
    # #minor extra memory use
    # n_neighbors_device = model.geometry.points.n_neighbors.get(device)
    # cum_n_neighbors_device = model.geometry.points.get_cum_n_neighbors(device)
    neighbors_device, n_neighbors_device, cum_n_neighbors_device = model.geometry.get_neighbors_in_direction(raydir_device, device)
    is_boundary_point_device = model.geometry.boundary.is_boundary_point.get(device)

    # expected_memory_usage : int = memhelper.compute_data_size_torch_tensor(model.parameters.npoints.get(), Types.GeometryInfo)#Note: incomplete memory usage. Will crash for large models 
    # #TODO: figure out better estimate of memory usage
    # splits = memhelper.split_data(model.parameters.npoints.get(), expected_memory_usage)



    # log_encountered_points = torch.empty((Nrays, 1), dtype=Types.IndexInfo, device=device)#on average 2x too large
    # log_encountered_points[:,0] = start_ids#on average 2x too large
    distance_travelled = torch.zeros((model.parameters.npoints.get()), dtype=Types.GeometryInfo, device=device)

    start_ids = torch.arange(model.parameters.npoints.get(), device=device)

    curr_ids = start_ids.clone().type(Types.IndexInfo)
    Nraystotrace = start_ids.size()
    mask_active_rays = torch.ones(Nraystotrace, dtype=torch.bool, device=device)
    # i:int = 0
    total_raytrace: int = 0
    while (torch.any(mask_active_rays)):
        total_raytrace+=mask_active_rays.sum(dim=0)
        #bad masking method, always starting from the full array
        masked_curr_ids = curr_ids[mask_active_rays]
        masked_start_ids = start_ids[mask_active_rays]
        masked_distance_travelled = distance_travelled[mask_active_rays]
        masked_origin_positions = positions_device[masked_start_ids]

        next_ids, distances = model.geometry.get_next(masked_origin_positions, raydir_device, masked_curr_ids, masked_distance_travelled, device, positions_device, neighbors_device, n_neighbors_device, cum_n_neighbors_device)
        distance_travelled[mask_active_rays] = distances
        curr_ids[mask_active_rays] = next_ids

        mask_active_rays = torch.logical_not(is_boundary_point_device[curr_ids])
        # print(mask_active_rays.size())
    print("total raytrace len", total_raytrace)

def trace_rays_test_single_iteration_less_masking(model: Model, raydir_device: torch.Tensor, device: torch.device):
    #performance analysis indicated that when most rays have stopped, the other raytracer only spent time on appending stuff. This is wasteful.
    #On another note, it is also impossible (without reverting to this formalism again) for me to accurately compute total optical depths, due to cumulative sums being required, applied to huge tensors.
    #thus we need this approach; TODO: bench vs old approach; somewhat comparable vs full masking, but slightly faster

    #unavoidable full mapping; costs O(Npoints*Nneighbors) memory
    positions_device = model.geometry.points.position.get(device)
    is_boundary_point_device = model.geometry.boundary.is_boundary_point.get(device)
    neighbors_device, n_neighbors_device, cum_n_neighbors_device = model.geometry.get_neighbors_in_direction(raydir_device, device)

    distance_travelled = torch.zeros((model.parameters.npoints.get()), dtype=Types.GeometryInfo, device=device)

    start_ids = torch.arange(model.parameters.npoints.get(), device=device)

    curr_ids = start_ids.clone().type(Types.IndexInfo)
    Nraystotrace = start_ids.size()
    mask_active_rays = torch.ones(Nraystotrace, dtype=torch.bool, device=device)

    while (torch.any(mask_active_rays)):
        #bad masking method, always starting from the full array
        curr_ids = curr_ids.masked_select(mask_active_rays)
        start_ids = start_ids.masked_select(mask_active_rays)
        distance_travelled = distance_travelled.masked_select(mask_active_rays)
        origin_positions = positions_device[start_ids]

        next_ids, distances = model.geometry.get_next(origin_positions, raydir_device, curr_ids, distance_travelled, device, positions_device, neighbors_device, n_neighbors_device, cum_n_neighbors_device)
        distance_travelled = distances
        curr_ids = next_ids

        mask_active_rays = torch.logical_not(is_boundary_point_device[curr_ids])


#DEPRECATED: even though manual masking is annoying, I will probably only implement a single solver anyway; so this class is redundant
class RaytracerGenerator:
    def __init__(self, model: Model, raydir: torch.Tensor, device: torch.device) -> None:
        self.device = device
        self.raydir = raydir
        self.model = model

    def __iter__(self):
        #unavoidable full mapping; costs O(Npoints*Nneighbors) memory
        #TODO: compute subset of neighbors in correct direction; results in a speedup of 2 times; do this at the same time as the directional boundary computation
        positions_device = self.model.geometry.points.position.get(self.device)
        # neighbors_device = self.model.geometry.points.neighbors.get(self.device)
        # #minor extra memory use
        # n_neighbors_device = self.model.geometry.points.n_neighbors.get(self.device)
        # cum_n_neighbors_device = self.model.geometry.points.get_cum_n_neighbors(self.device)
        is_boundary_point_device = self.model.geometry.boundary.is_boundary_point.get(self.device)
        neighbors_device, n_neighbors_device, cum_n_neighbors_device = self.model.geometry.get_neighbors_in_direction(self.raydir, self.device)



        # log_encountered_points = torch.empty((Nrays, 1), dtype=Types.IndexInfo, device=device)#on average 2x too large
        # log_encountered_points[:,0] = start_ids#on average 2x too large
        distance_travelled = torch.zeros((self.model.parameters.npoints.get()), dtype=Types.GeometryInfo, device=self.device)

        start_ids = torch.arange(self.model.parameters.npoints.get(), device=self.device)
        prev_ids = start_ids

        curr_ids = start_ids.clone().type(Types.IndexInfo)
        Nraystotrace = start_ids.size()
        mask_active_rays = torch.ones(Nraystotrace, dtype=torch.bool, device=self.device)

        # i:int = 0
        while (torch.any(mask_active_rays)):
            # print("hier", i)
            # i+=1
            #bad masking method, always starting from the full array
            curr_ids = curr_ids.masked_select(mask_active_rays)
            # print("size currids", curr_ids.size())
            start_ids = start_ids.masked_select(mask_active_rays)
            distance_travelled = distance_travelled.masked_select(mask_active_rays)
            origin_positions = positions_device[start_ids]

            next_ids, distances = self.model.geometry.get_next(origin_positions, self.raydir, curr_ids, distance_travelled, self.device, positions_device, neighbors_device, n_neighbors_device, cum_n_neighbors_device)
            # print(next_ids)
            distance_travelled = distances
            prev_ids = curr_ids
            curr_ids = next_ids

            mask_active_rays = torch.logical_not(is_boundary_point_device[curr_ids])

            yield curr_ids, prev_ids, distance_travelled, start_ids