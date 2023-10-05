from magrittetorch.algorithms.raytracer import RaytracerGenerator
from magrittetorch.model.model import Model
from magrittetorch.model.sources.frequencyevalhelper import FrequencyEvalHelper
from magrittetorch.utils.storagetypes import Types
from magrittetorch.utils.constants import min_opacity, min_optical_depth
import torch
import time

#DEPRECATED: appends waste a lot of time
# def solve_long_characteristics(model: Model, direction: torch.Tensor, device: torch.device = torch.device("cpu")) -> None:
#     start =time.time()
#     point_ind, distances, scatter_ind = trace_rays_sparse(model, direction) #dims: [N_POINTS_TO_EVAL]
#     end = time.time()
#     direction_device = direction.to(device)[None, :]
#     print(point_ind.get_device(), distances.get_device(), scatter_ind.get_device())
#     print("ray trace time", end-start)
#     NLTE_freqs = model.sources.lines.get_all_line_frequencies(device=device) #dims: [parameters.npoints, NFREQS]#TODO: move to function arguments
#     model_velocities = model.geometry.points.velocity.get(device) #dims: [parameters.npoints, 3]
#     model_positions = model.geometry.points.position.get(device) #dims: [parameters.npoints, 3]
#     origin_velocities = model_velocities[scatter_ind, :]#err, assumes behavior that might change in the future# dims: [N_POINTS_TO_EVAL]
#     origin_positions = model_positions[scatter_ind, :]#err, assumes behavior that might change in the future # dims: [N_POINTS_TO_EVAL]
#     # point_velocities = model_velocities[point_ind, :] #dims: [parameters.npoints, 3]
#     print("computing doppler")
#     doppler_shift = model.geometry.get_doppler_shift(point_ind.to(device), origin_positions, origin_velocities, direction_device, distances.to(device), device)
#     print("shift", doppler_shift)
#     freqs_to_evaluate = NLTE_freqs[point_ind, :]*doppler_shift[:, None]#TODO: add doppler shift here
#     print("evaluating opacities")
#     # print()
#     opacities, emissivities = model.sources.get_total_opacity_emissivity(point_ind, freqs_to_evaluate, device)
#     print(opacities, emissivities)


#DEPRECATED: generator will be removed
def solve_long_characteristics_with_generator(model:Model, direction: torch.Tensor, device: torch.device) -> None:
    #note: this is just a test function for now
    #I might change the api completely
    #also: no memory management yet, should be implemented in some function above this one; providing which points we need
    NLTE_freqs = model.sources.lines.get_all_line_frequencies(device=device) #dims: [parameters.npoints, NFREQS]#TODO: move to function arguments
    model_velocities = model.geometry.points.velocity.get(device) #dims: [parameters.npoints, 3]
    model_positions = model.geometry.points.position.get(device) #dims: [parameters.npoints, 3]
    origin_velocities = model_velocities[:, :]#err, assumes behavior that might change in the future# dims: [N_POINTS_TO_EVAL]
    origin_positions = model_positions[:, :]#err, assumes behavior that might change in the future # dims: [N_POINTS_TO_EVAL]
    
    for (next_point, prev_point, travelled_distance, original_index) in RaytracerGenerator(model, direction, device):

        doppler_shift = model.geometry.get_doppler_shift(next_point, origin_positions[original_index], origin_velocities[original_index], direction, travelled_distance, device)
        freqs_to_evaluate = NLTE_freqs[next_point, :]*doppler_shift[:, None]#TODO: add doppler shift here
        opacities, emissivities = model.sources.get_total_opacity_emissivity(next_point, freqs_to_evaluate, device)
        optical_depths = model.sources.get_total_optical_depth(next_point, freqs_to_evaluate, freqs_to_evaluate, travelled_distance, doppler_shift, doppler_shift, opacities, emissivities, device)
        pass

#DEPRECATED: generator will be removed
def solve_long_characteristics_restructured_freqs(model:Model, direction: torch.Tensor, device: torch.device) -> None:
    #note: this is just a test function for now
    #I might change the api completely
    #also: no memory management yet, should be implemented in some function above this one; providing which points we need
    NLTE_freqs = model.sources.lines.get_all_line_frequencies(device=device) #dims: [parameters.npoints, NFREQS]#TODO: move to function arguments
    model_velocities = model.geometry.points.velocity.get(device) #dims: [parameters.npoints, 3]
    model_positions = model.geometry.points.position.get(device) #dims: [parameters.npoints, 3]
    origin_velocities = model_velocities[:, :]#err, assumes behavior that might change in the future# dims: [N_POINTS_TO_EVAL]
    origin_positions = model_positions[:, :]#err, assumes behavior that might change in the future # dims: [N_POINTS_TO_EVAL]
    freqhelper = FrequencyEvalHelper(NLTE_freqs, model.sources.lines.lineProducingSpecies.list, model_velocities, device)#TODO: should be result of called get_all_line_freqs
    sum_optical_depths = torch.zeros_like(NLTE_freqs, dtype=Types.FrequencyInfo, device=device)
    computed_intensity = torch.zeros_like(NLTE_freqs, dtype=Types.FrequencyInfo, device=device)
    #TODO: add boundary intensity add end point

    for (next_point, prev_point, travelled_distance, original_index) in RaytracerGenerator(model, direction, device):
        # print(next_point, prev_point)
        doppler_shift = model.geometry.get_doppler_shift(next_point, origin_positions[original_index], origin_velocities[original_index], direction, travelled_distance, device)
        opacities, emissivities = model.sources.get_total_opacity_emissivity_freqhelper(next_point, doppler_shift, freqhelper, device)
        optical_depths = model.sources.get_total_optical_depth_freqhelper(next_point, prev_point, original_index, freqhelper, travelled_distance, doppler_shift, doppler_shift, opacities, emissivities, device)
        source_function = emissivities/opacities
        #TODO: better implementation, using prev and current source function, instead of constant source
        computed_intensity[original_index] += source_function * (1.0-torch.exp(-optical_depths)) * torch.exp(-sum_optical_depths[original_index])
        sum_optical_depths[original_index] += optical_depths
        # print(computed_intensity)


def solve_long_characteristics_single_direction(model: Model, raydir: torch.Tensor, start_positions: torch.Tensor, start_velocities: torch.Tensor, start_indices: torch.Tensor, freqhelper: FrequencyEvalHelper, device: torch.device) -> torch.Tensor:
    """Solves the radiative transfer equation using the long-characteristics method in a single direction. TODO: implement non-zero starting distances (for imaging)

    Args:
        model (Model): The model on which to compute
        raydir (torch.Tensor): Direction of the ray. Has dimensions [3]
        start_positions (torch.Tensor): Positions of the origins of the rays in 3D space. Has dimensions [NPOINTS, 3]
        start_velocities (torch.Tensor): Velocities of the origins of the rays in 3D space. Has dimensions [NPOINTS, 3]
        start_indices (torch.Tensor): Point indices to start tracing the rays. Has dimensions [NPOINTS]. TODO? automatically determine?
        freqhelper (FrequencyEvalHelper): Frequency evaluation helper object for narrow spectral lines.
        device (torch.device): Device on which to compute and return the result.

    Returns:
        torch.Tensor: The computed intensities [W/m**2/Hz/rad**2]. Has dimensions [NPOINTS, NFREQS]
    """
    #no generator approach
    #no work distribution, as this should happen in a higher function

    #unavoidable full mapping; costs O(Npoints*Nneighbors) memory
    positions_device = model.geometry.points.position.get(device)
    npoints = start_positions.size(dim=0)#number of points to trace

    #get neighbors per direction; saves 50% time for raytracing in 3D geometries
    neighbors_device, n_neighbors_device, cum_n_neighbors_device = model.geometry.get_neighbors_in_direction(raydir, device)


    # is_boundary_point_device = model.geometry.boundary.is_boundary_point.get(device)
    is_boundary_point_device = model.geometry.boundary.get_boundary_in_direction(raydir, device)
    sum_optical_depths = torch.zeros_like(freqhelper.original_frequencies, dtype=Types.FrequencyInfo, device=device)
    computed_intensity = torch.zeros_like(freqhelper.original_frequencies, dtype=Types.FrequencyInfo, device=device)
    distance_travelled = torch.zeros(npoints, dtype=Types.GeometryInfo, device=device)

    start_ids = start_indices.clone()#will be masked, so needs to be clone in order to preserve the original indices (for tracing other directions)
    prev_ids = start_ids

    curr_ids = start_ids.clone()
    mask_active_rays = torch.logical_not(is_boundary_point_device[curr_ids])
    boundary_indices = curr_ids[is_boundary_point_device[curr_ids]]
    #TODO: refactor source function computation (emissivty/opacity), as some safety constant min_opacity is included
    doppler_shift = torch.ones(npoints, dtype = Types.FrequencyInfo, device=device)
    prev_opacities, starting_emissivities = model.sources.get_total_opacity_emissivity_freqhelper(curr_ids, doppler_shift, freqhelper, device)
    prev_source_function = starting_emissivities/(prev_opacities+min_opacity)

    #already mask previous values
    prev_opacities = prev_opacities[mask_active_rays]
    prev_source_function = prev_source_function[mask_active_rays]

    #when encountering boundary points, add their intensity contribution
    #evaluate boundary intensity
    #technically boundary_indices
    computed_intensity[boundary_indices] = model.get_boundary_intensity(boundary_indices, freqhelper, device)

    while (torch.any(mask_active_rays)):
        #continuously subset the tensors, reducing time needed to access the relevant subsets
        #TODO? also try subsetting computed_intensity, sum_optical_depth
        curr_ids = curr_ids.masked_select(mask_active_rays)
        start_ids = start_ids.masked_select(mask_active_rays)
        distance_travelled = distance_travelled.masked_select(mask_active_rays)
        start_positions = start_positions[mask_active_rays,:]
        start_velocities = start_velocities[mask_active_rays,:]

        next_ids, distances = model.geometry.get_next(start_positions, raydir, curr_ids, distance_travelled, device, positions_device, neighbors_device, n_neighbors_device, cum_n_neighbors_device)
        distance_increment = distances - distance_travelled

        distance_travelled = distances
        prev_ids = curr_ids
        curr_ids = next_ids

        mask_active_rays = torch.logical_not(is_boundary_point_device[curr_ids])

        #After raytracing, we now compute everything required
        doppler_shift = model.geometry.get_doppler_shift(curr_ids, start_positions, start_velocities, raydir, distances, device)
        curr_opacities, curr_emissivities = model.sources.get_total_opacity_emissivity_freqhelper(curr_ids, doppler_shift, freqhelper, device)
        optical_depths = model.sources.get_total_optical_depth_freqhelper(curr_ids, prev_ids, start_ids, freqhelper, distance_increment, doppler_shift, doppler_shift, curr_opacities, prev_opacities, device)
        curr_source_function = curr_emissivities/(curr_opacities+min_opacity)

        #To compute the intensity contribution, we need to multiply the source function by the factors below
        close_factor = 1+torch.expm1(-optical_depths-min_optical_depth)/(optical_depths+min_optical_depth)
        far_factor = -close_factor-torch.expm1(-optical_depths)
        computed_intensity[start_ids] += (far_factor * curr_source_function + close_factor * prev_source_function) * torch.exp(-sum_optical_depths[start_ids])
        sum_optical_depths[start_ids] += optical_depths

        #already mask the current data
        prev_source_function = curr_source_function[mask_active_rays]
        prev_opacities = curr_opacities[mask_active_rays]

        #and add boundary intensity to the rays which have ended, taking into account the current extincition factor e^-tau
        ended_rays = torch.logical_not(mask_active_rays)
        computed_intensity[start_ids[ended_rays]] += model.get_boundary_intensity(curr_ids[ended_rays], freqhelper, device) * torch.exp(-sum_optical_depths[start_ids[ended_rays]])

    return computed_intensity




def solve_long_characteristics_NLTE(model: Model, device: torch.device) -> torch.Tensor:
    NLTE_freqs = model.sources.lines.get_all_line_frequencies(device=device) #dims: [parameters.npoints, NFREQS]#TODO: move to function arguments; err only for imaging, this should be an argument
    model_velocities = model.geometry.points.velocity.get(device) #dims: [parameters.npoints, 3]
    model_positions = model.geometry.points.position.get(device) #dims: [parameters.npoints, 3]
    freqhelper = FrequencyEvalHelper(NLTE_freqs, model.sources.lines.lineProducingSpecies.get(), model_velocities, device)#TODO: should be result of called get_all_line_freqs
    raydirs = model.geometry.rays.direction.get(device)
    weights = model.geometry.rays.weight.get(device)
    total_intensity = torch.zeros_like(NLTE_freqs) #dims: [parameters.npoints, NFREQS]
    total_integrated_line_intensity = torch.zeros((model.parameters.npoints.get(), model.sources.lines.get_total_number_lines()), dtype=Types.FrequencyInfo)
    #add everything, with correct contribution
    for raydir_index in range(raydirs.shape[0]):
        print("rr:", raydir_index)
        #Adding the results immediately, as we will otherwise run out of memory
        total_intensity += weights[raydir_index] * solve_long_characteristics_single_direction(model, raydirs[raydir_index,:], model_positions, model_velocities, torch.arange(model.parameters.npoints.get(), device=device), freqhelper, device)

    encountered_freqs: int = 0
    # now numerically integrate the intensities of each different line# TODO: check if better option exists, without for loop
    for lineidx in range(model.sources.lines.lineProducingSpecies.length_param.get()):
        lspec = model.sources.lines.lineProducingSpecies[lineidx]
        quad_weights = lspec.linequadrature.weights.get(device)
        nfreqs: int = lspec.get_n_lines_freqs()
        total_integrated_line_intensity[:, lineidx] = torch.sum(quad_weights[None, :] * total_intensity[:,encountered_freqs:encountered_freqs+nfreqs], dim=1)

        encountered_freqs += nfreqs

    return total_integrated_line_intensity
    #For initial testing, only solve for a single direction. This will be changed afterwards
    #TODO list: compute mean line intensities, ALI factors (somehow), and feed tghis into statistical equilibrium equationsn, add ng-acceleration
    # return solve_long_characteristics_single_direction(model, raydirs[0,:], model_positions, model_velocities, torch.arange(model.parameters.npoints.get(), device=device), freqhelper, device)
