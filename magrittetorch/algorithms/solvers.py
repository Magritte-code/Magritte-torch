from magrittetorch.algorithms.raytracer import RaytracerGenerator
from magrittetorch.algorithms.torch_algorithms import interpolate2D_linear
from magrittetorch.model.model import Model
from magrittetorch.model.geometry.geometry import GeometryType
from magrittetorch.model.sources.frequencyevalhelper import FrequencyEvalHelper
from magrittetorch.utils.storagetypes import Types
from magrittetorch.utils.constants import min_opacity, min_optical_depth, convergence_fraction, min_rel_pop_for_convergence
from magrittetorch.model.image import Image, ImageType
from magrittetorch.tools.radiativetransferutils import relative_error

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
        doppler_shift = model.geometry.get_doppler_shift(next_point, origin_positions[original_index], origin_velocities[original_index], direction, travelled_distance, device)
        opacities, emissivities = model.sources.get_total_opacity_emissivity_freqhelper(original_index, next_point, doppler_shift, freqhelper, device)
        optical_depths = model.sources.get_total_optical_depth_freqhelper(next_point, prev_point, original_index, freqhelper, travelled_distance, doppler_shift, doppler_shift, opacities, emissivities, device)
        source_function = emissivities/opacities
        #TODO: better implementation, using prev and current source function, instead of constant source
        computed_intensity[original_index] += source_function * (1.0-torch.exp(-optical_depths)) * torch.exp(-sum_optical_depths[original_index])
        sum_optical_depths[original_index] += optical_depths


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
    positions_device = model.geometry.points.position.get(device) #dims: [parameters.npoints, 3]
    npoints = start_positions.size(dim=0)#number of points to trace NPOINTS

    #get neighbors per direction; saves 50% time for raytracing in 3D geometries
    neighbors_device, n_neighbors_device, cum_n_neighbors_device = model.geometry.get_neighbors_in_direction(raydir, device) 
    #dims: [sum(n_neighbors_in_correct_direction)], [parameters.npoints], [parameters.npoints]


    # is_boundary_point_device = model.geometry.boundary.is_boundary_point.get(device)
    is_boundary_point_device = model.geometry.boundary.get_boundary_in_direction(raydir, device) #dims: [parameters.npoints]
    sum_optical_depths = torch.zeros((npoints, freqhelper.original_frequencies.size(dim=1)), dtype=Types.FrequencyInfo, device=device) #dims: [NPOINTS, NFREQS]
    computed_intensity = torch.zeros((npoints, freqhelper.original_frequencies.size(dim=1)), dtype=Types.FrequencyInfo, device=device) #dims: [NPOINTS, NFREQS]

    # The starting indices do not necessarily correspond with the starting indices, as imaging starts tracing outside the model
    # Therefore the starting distance can be nonzero
    distance_travelled = model.geometry.get_starting_distance(raydir, start_indices, start_positions, device) #dims: [NPOINTS]

    start_ids = start_indices.clone()#will be masked, so needs to be clone in order to preserve the original indices (for tracing other directions)
    prev_ids = start_ids #dims: [NPOINTS]

    curr_ids = start_ids.clone() #dims: [NPOINTS]
    boundary_mask = is_boundary_point_device[curr_ids] #dims: [NPOINTS]
    #for 1D spherical symmetry, we need to allow the rays to start if they are going inside the model
    if (model.geometry.geometryType.get() == GeometryType.SpericallySymmetric1D):
        boundary_mask = torch.sum((start_positions - raydir[None, :] * torch.matmul(start_positions, raydir)[:, None])**2, dim=1) >= torch.max(model.geometry.points.position.get(device)[:, 0])**2 #dims: [NPOINTS]

    mask_active_rays = torch.logical_not(boundary_mask) #dims: [N_ACTIVE_RAYS<=NPOINTS] Dimension will change, as not all rays end at the same time
    data_indices = torch.arange(npoints, device=device) #dims: [N_ACTIVE_RAYS<=NPOINTS]
    boundary_indices = curr_ids[boundary_mask] #dims: [N_ACTIVE_RAYS<=NPOINTS]
    #TODO: refactor source function computation (emissivty/opacity), as some safety constant min_opacity is included
    prev_shift = model.geometry.get_doppler_shift(curr_ids, start_positions, start_velocities, raydir, distance_travelled, device) #dims: [NPOINTS]
    prev_opacities, starting_emissivities = model.sources.get_total_opacity_emissivity_freqhelper(start_ids, curr_ids, prev_shift, freqhelper, device) #dims: [NPOINTS, NFREQS]
    prev_source_function = starting_emissivities/(prev_opacities+min_opacity) #dims: [NPOINTS, NFREQS]

    #when encountering boundary points, add their intensity contribution
    #evaluate boundary intensity
    #technically boundary_indices
    computed_intensity[boundary_mask] = model.get_boundary_intensity(boundary_indices, freqhelper, prev_shift[boundary_mask], device) #dims: [NPOINTS, NFREQS]

    #already mask previous values
    prev_opacities = prev_opacities[mask_active_rays] #dims: [N_ACTIVE_RAYS<=NPOINTS, NFREQS]
    prev_source_function = prev_source_function[mask_active_rays] #dims: [N_ACTIVE_RAYS<=NPOINTS, NFREQS]
    prev_shift = prev_shift[mask_active_rays] #dims: [N_ACTIVE_RAYS<=NPOINTS]

    while (torch.any(mask_active_rays)):
        #continuously subset the tensors, reducing time needed to access the relevant subsets
        #TODO? also try subsetting computed_intensity, sum_optical_depth
        curr_ids = curr_ids.masked_select(mask_active_rays)
        start_ids = start_ids.masked_select(mask_active_rays)
        data_indices = data_indices.masked_select(mask_active_rays) #dims: [N_ACTIVE_RAYS<=NPOINTS]
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
        curr_opacities, curr_emissivities = model.sources.get_total_opacity_emissivity_freqhelper(start_ids, curr_ids, doppler_shift, freqhelper, device)
        optical_depths = model.sources.get_total_optical_depth_freqhelper(curr_ids, prev_ids, start_ids, freqhelper, distance_increment, doppler_shift, prev_shift, curr_opacities, prev_opacities, device)
        curr_source_function = curr_emissivities/(curr_opacities+min_opacity)

        #To compute the intensity contribution, we need to multiply the source function by the factors below
        close_factor = 1+torch.expm1(-optical_depths-min_optical_depth)/(optical_depths+min_optical_depth)
        far_factor = -close_factor-torch.expm1(-optical_depths)
        computed_intensity[data_indices] += (far_factor * curr_source_function + close_factor * prev_source_function) * torch.exp(-sum_optical_depths[data_indices])
        sum_optical_depths[data_indices] += optical_depths

        #already mask the current data
        prev_source_function = curr_source_function[mask_active_rays]
        prev_opacities = curr_opacities[mask_active_rays]
        prev_shift = doppler_shift[mask_active_rays]

        #and add boundary intensity to the rays which have ended, taking into account the current extincition factor e^-tau
        ended_rays = torch.logical_not(mask_active_rays)
        computed_intensity[data_indices[ended_rays]] += model.get_boundary_intensity(curr_ids[ended_rays], freqhelper, doppler_shift[ended_rays], device) * torch.exp(-sum_optical_depths[data_indices[ended_rays]])

    return computed_intensity


def get_total_optical_depth_single_direction(model: Model, raydir: torch.Tensor, start_positions: torch.Tensor, start_velocities: torch.Tensor, start_indices: torch.Tensor, freqhelper: FrequencyEvalHelper, device: torch.device) -> torch.Tensor:
    """Computes the total optical depth along a single direction, for all given start positions.
    Based on solve_long_characteristics_single_direction, but only returns the optical depth, and does not compute the intensity.

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
    #no work distribution, as this should happen in a higher function

    #unavoidable full mapping; costs O(Npoints*Nneighbors) memory
    positions_device = model.geometry.points.position.get(device) #dims: [parameters.npoints, 3]
    npoints = start_positions.size(dim=0)#number of points to trace NPOINTS

    #get neighbors per direction; saves 50% time for raytracing in 3D geometries
    neighbors_device, n_neighbors_device, cum_n_neighbors_device = model.geometry.get_neighbors_in_direction(raydir, device) 
    #dims: [sum(n_neighbors_in_correct_direction)], [parameters.npoints], [parameters.npoints]

    is_boundary_point_device = model.geometry.boundary.get_boundary_in_direction(raydir, device) #dims: [parameters.npoints]
    sum_optical_depths = torch.zeros((npoints, freqhelper.original_frequencies.size(dim=1)), dtype=Types.FrequencyInfo, device=device) #dims: [NPOINTS, NFREQS]

    # The starting indices do not necessarily correspond with the starting indices, as imaging starts tracing outside the model
    # Therefore the starting distance can be nonzero
    distance_travelled = model.geometry.get_starting_distance(raydir, start_indices, start_positions, device) #dims: [NPOINTS]

    start_ids = start_indices.clone()#will be masked, so needs to be clone in order to preserve the original indices (for tracing other directions)
    prev_ids = start_ids #dims: [NPOINTS]

    curr_ids = start_ids.clone() #dims: [NPOINTS]
    boundary_mask = is_boundary_point_device[curr_ids] #dims: [NPOINTS]
    #for 1D spherical symmetry, we need to allow the rays to start if they are going inside the model
    if (model.geometry.geometryType.get() == GeometryType.SpericallySymmetric1D):
        boundary_mask = torch.sum((start_positions - raydir[None, :] * torch.matmul(start_positions, raydir)[:, None])**2, dim=1) >= torch.max(model.geometry.points.position.get(device)[:, 0])**2 #dims: [NPOINTS]

    mask_active_rays = torch.logical_not(boundary_mask) #dims: [N_ACTIVE_RAYS<=NPOINTS] Dimension will change, as not all rays end at the same time
    data_indices = torch.arange(npoints, device=device) #dims: [N_ACTIVE_RAYS<=NPOINTS]
    #TODO: refactor source function computation (emissivty/opacity), as some safety constant min_opacity is included
    prev_shift = model.geometry.get_doppler_shift(curr_ids, start_positions, start_velocities, raydir, distance_travelled, device) #dims: [NPOINTS]
    prev_opacities, _ = model.sources.get_total_opacity_emissivity_freqhelper(start_ids, curr_ids, prev_shift, freqhelper, device) #dims: [NPOINTS, NFREQS]

    #already mask previous values
    prev_opacities = prev_opacities[mask_active_rays] #dims: [N_ACTIVE_RAYS<=NPOINTS, NFREQS]
    prev_shift = prev_shift[mask_active_rays] #dims: [N_ACTIVE_RAYS<=NPOINTS]

    while (torch.any(mask_active_rays)):
        #continuously subset the tensors, reducing time needed to access the relevant subsets
        #TODO? also try subsetting computed_intensity, sum_optical_depth
        curr_ids = curr_ids.masked_select(mask_active_rays)
        start_ids = start_ids.masked_select(mask_active_rays)
        data_indices = data_indices.masked_select(mask_active_rays) #dims: [N_ACTIVE_RAYS<=NPOINTS]
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
        curr_opacities, _ = model.sources.get_total_opacity_emissivity_freqhelper(start_ids, curr_ids, doppler_shift, freqhelper, device)
        optical_depths = model.sources.get_total_optical_depth_freqhelper(curr_ids, prev_ids, start_ids, freqhelper, distance_increment, doppler_shift, prev_shift, curr_opacities, prev_opacities, device)

        #Simply sum up the optical depths
        sum_optical_depths[data_indices] += optical_depths

        #already mask the current data
        prev_opacities = curr_opacities[mask_active_rays]
        prev_shift = doppler_shift[mask_active_rays]

    # return computed_optical depths
    return sum_optical_depths


def solve_long_characteristics_single_direction_all_NLTE_freqs(model: Model, raydir: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Computes the intensities in a given direction at all required frequencies for NLTE computations.
    This is a helper function for benchmarking purposes.

    Args:
        model (Model): The model
        raydir (torch.Tensor): The ray direction. Has dimensions [3]
        device (torch.device): The device on which to compute

    Returns:
        torch.Tensor: The computed intensities [W/m**2/Hz/rad**2]. Has dimensions [NPOINTS, NFREQS]
    """
    NLTE_freqs = model.sources.lines.get_all_line_frequencies(device=device) #dims: [parameters.npoints, NFREQS]#TODO: move to function arguments; err only for imaging, this should be an argument
    model_velocities = model.geometry.points.velocity.get(device) #dims: [parameters.npoints, 3]
    model_positions = model.geometry.points.position.get(device) #dims: [parameters.npoints, 3]
    freqhelper = FrequencyEvalHelper(NLTE_freqs, model.sources.lines.lineProducingSpecies.get(), model_velocities, device)#TODO: should be result of called get_all_line_freqs

    return solve_long_characteristics_single_direction(model, raydir, model_positions, model_velocities, torch.arange(model.parameters.npoints.get()), freqhelper, device)


# @torch.compile
def solve_long_characteristics_NLTE(model: Model, device: torch.device) -> torch.Tensor:
    """Computes the mean line intensity J_ij at every point in the model

    Args:
        model (Model): The model
        device (torch.device): Device on which to compute

    Returns:
        torch.Tensor: The mean line intensity. Has dimensions [parameters.npoints, model.sources.lines.get_total_number_lines()]
    """
    NLTE_freqs = model.sources.lines.get_all_line_frequencies(device=device) #dims: [parameters.npoints, NFREQS]#TODO: move to function arguments; err only for imaging, this should be an argument
    model_velocities = model.geometry.points.velocity.get(device) #dims: [parameters.npoints, 3]
    model_positions = model.geometry.points.position.get(device) #dims: [parameters.npoints, 3]
    freqhelper = FrequencyEvalHelper(NLTE_freqs, model.sources.lines.lineProducingSpecies.get(), model_velocities, device)#TODO: should be result of called get_all_line_freqs
    raydirs = model.geometry.rays.direction.get(device)
    weights = model.geometry.rays.weight.get(device)
    total_intensity = torch.zeros_like(NLTE_freqs) #dims: [parameters.npoints, NFREQS]
    total_integrated_line_intensity = torch.zeros((model.parameters.npoints.get(), model.sources.lines.get_total_number_lines()), dtype=Types.FrequencyInfo)
    #add everything, with correct contribution
    starttime = time.time()
    for raydir_index in range(raydirs.shape[0]):
        print("rr:", raydir_index)
        print(time.time()-starttime)
        #Adding the results immediately, as we will otherwise run out of memory
        total_intensity += weights[raydir_index] * solve_long_characteristics_single_direction(model, raydirs[raydir_index,:], model_positions, model_velocities, torch.arange(model.parameters.npoints.get(), device=device), freqhelper, device)

    encountered_freqs: int = 0
    # now numerically integrate the intensities of each different line# TODO: check if better option exists, without for loop
    for lineidx in range(model.sources.lines.lineProducingSpecies.length_param.get()):
        lspec = model.sources.lines.lineProducingSpecies[lineidx]
        quad_weights = lspec.linequadrature.weights.get(device)# dims: [NQUADS]
        nfreqs: int = lspec.get_n_lines_freqs()
        expanded_quad_weights = quad_weights.repeat(lspec.linedata.nrad.get())#needs to be expanded to match the number of frequencies
        total_integrated_line_intensity[:, lineidx] = torch.sum(expanded_quad_weights[None, :] * total_intensity[:,encountered_freqs:encountered_freqs+nfreqs], dim=1)

        encountered_freqs += nfreqs

    return total_integrated_line_intensity
    #For initial testing, only solve for a single direction. This will be changed afterwards
    #TODO list: compute mean line intensities, ALI factors (somehow), and feed tghis into statistical equilibrium equationsn, add ng-acceleration
    # return solve_long_characteristics_single_direction(model, raydirs[0,:], model_positions, model_velocities, torch.arange(model.parameters.npoints.get(), device=device), freqhelper, device)


def compute_level_populations_statistical_equilibrium(model: Model, J: torch.Tensor, device: torch.device) -> list[torch.Tensor]:
    """Computes the level populations using the statistical equilibrium equations, given the mean line intensities J
    TODO: also include accelerated lambda iteration, just add (optional) extra functon argument

    Args:
        model (Model): The model
        J (torch.Tensor): The (effective) mean line intensity. Has dims [parameters.npoints, parameters.nlines?]
        device (torch.device): Device on which to compute

    Returns:
        The list of updated level populations. Has dims [[parameters.npoints, parameters.nlev] for every line species]
    """


    total_encountered_lines: int = 0
    computed_level_pops: list[torch.Tensor] = []
    for linespecidx in range(model.parameters.nlspecs.get()):#every species is assumed to be independent
        lspec = model.sources.lines.lineProducingSpecies[linespecidx]
        linedata = lspec.linedata
        nlines: int = linedata.nrad.get()
        nlev: int = linedata.nlev.get()
        npoints: int = model.parameters.npoints.get()
        einsteinA = linedata.A.get(device)
        einsteinBa = linedata.Ba.get(device)
        einsteinBs = linedata.Bs.get(device)
        Jrelevant = J[:, total_encountered_lines:total_encountered_lines+nlines]
        upperidx = linedata.irad.get(device) #dims: [nlines]; all values (indices) in [0, nlev-1]
        loweridx = linedata.jrad.get(device) #dims: [nlines]; all values (indices) in [0, nlev-1]
        rate_upper_to_lower = torch.zeros_like(Jrelevant[0, :])
        rate_lower_to_upper = torch.zeros_like(rate_upper_to_lower)
        # Lambdarelevant = ... [:, also the same]
        matrix = torch.zeros((npoints, nlev, nlev), dtype=Types.LevelPopsInfo, device=device)

        rate_upper_to_lower = einsteinA[None, :] + einsteinBs[None, :] * Jrelevant#dims: [npoints, nlines]
        rate_lower_to_upper = einsteinBa[None, :] * Jrelevant
        full_upperidx = upperidx.repeat(npoints, 1)#dims: [npoints, nlines] values: [0, nlev-1]
        full_loweridx = loweridx.repeat(npoints, 1)#dims: [npoints, nlines] values: [0, nlev-1]

        pointidxrange = torch.arange(npoints, dtype=Types.IndexInfo, device=device)#dims: [npoints]

        # Correctly adds all the contributions to the transition matrix (accumulates rates at same indices)
        matrix.index_put_((pointidxrange[:, None], full_loweridx, full_upperidx), rate_upper_to_lower, accumulate = True)
        matrix.index_put_((pointidxrange[:, None], full_upperidx, full_upperidx), -rate_upper_to_lower, accumulate = True)
        matrix.index_put_((pointidxrange[:, None], full_upperidx, full_loweridx), rate_lower_to_upper, accumulate = True)
        matrix.index_put_((pointidxrange[:, None], full_loweridx, full_loweridx), -rate_lower_to_upper, accumulate = True)


        temperature = model.thermodynamics.temperature.gas.get(device)#dims: [npoints]
        full_abundance = model.chemistry.species.abundance.get(device)[:, :]#dims: [npoints, nspecs]
        #for all collision partners, add their contributions to the transition matrix
        for colpar in lspec.linedata.colpar:
            #Adjusting the abundance, depending on whether we have ortho or para H2
            colpar_abundance = colpar.adjust_abundace_for_ortho_para_h2(temperature, full_abundance[:, colpar.num_col_partner.get()])#dims: [npoints]
            
            #and do exactly the same as for the normal level transitions
            #dims tmp:[ntmp], Cd/Ce: [ntmp, ncol], temperature: [npoints] -> interpolated -> dims: [npoints, ncol]
            rate_upper_to_lower = interpolate2D_linear(colpar.tmp.get(device), colpar.Cd.get(device), temperature) * colpar_abundance[:, None]#dims: [npoints, ncol]
            rate_lower_to_upper = interpolate2D_linear(colpar.tmp.get(device), colpar.Ce.get(device), temperature) * colpar_abundance[:, None]
            upperidx = colpar.icol.get(device)#dims:[ncol]
            loweridx = colpar.jcol.get(device)
            full_upperidx = upperidx.repeat(npoints, 1)#dims: [npoints, ncol] values: [0, nlev-1]
            full_loweridx = loweridx.repeat(npoints, 1)#dims: [npoints, ncol] values: [0, nlev-1]
            pointidxrange = torch.arange(npoints, dtype=Types.IndexInfo, device=device)#dims: [npoints]

            # Correctly adds all the contributions to the transition matrix (accumulates rates at same indices)
            matrix.index_put_((pointidxrange[:, None], full_loweridx, full_upperidx), rate_upper_to_lower, accumulate = True)
            matrix.index_put_((pointidxrange[:, None], full_upperidx, full_upperidx), -rate_upper_to_lower, accumulate = True)
            matrix.index_put_((pointidxrange[:, None], full_upperidx, full_loweridx), rate_lower_to_upper, accumulate = True)
            matrix.index_put_((pointidxrange[:, None], full_loweridx, full_loweridx), -rate_lower_to_upper, accumulate = True)

        #and make sure that the sum of all levelpops is equal to 1
        matrix[:, nlev-1, :] = torch.ones((npoints, nlev), dtype=Types.LevelPopsInfo, device=device)

        vector = torch.zeros(npoints, nlev, dtype=Types.LevelPopsInfo, device=device)
        vector[:, nlev-1] = full_abundance[:, lspec.linedata.num.get()]

        levelpops: torch.Tensor = torch.linalg.solve(matrix, vector)
        computed_level_pops.append(levelpops)
        total_encountered_lines+=nlines

    return computed_level_pops


def level_pops_converged(previous_level_pops: torch.Tensor, current_level_pops: torch.Tensor, relative_diff_threshold: float = min_rel_pop_for_convergence, convergence_fraction: float = convergence_fraction) -> bool:
    print("relative error:", torch.mean(relative_error(previous_level_pops, current_level_pops)).item())
    print("relative diff threshold:", relative_diff_threshold)
    print("convergence fraction:", convergence_fraction)
    print("mean relative error:", torch.mean((relative_error(previous_level_pops, current_level_pops) < relative_diff_threshold).type(Types.LevelPopsInfo)))
    print("converged:", torch.mean((relative_error(previous_level_pops, current_level_pops) < relative_diff_threshold).type(Types.LevelPopsInfo)).item() > convergence_fraction)
    return (torch.mean((relative_error(previous_level_pops, current_level_pops) < relative_diff_threshold).type(Types.LevelPopsInfo)).item() > convergence_fraction)


def compute_level_populations(model: Model, device: torch.device, max_n_iterations: int = 50, use_ng_acceleration: bool = True, max_its_between_ng_accel: int = 8) -> None:
    """
    Computes the level populations of the given model using an iterative approach,
    utilizing the mean line intensities to solve the statistical equilibrium equations.
    TODO: update this docstring with ng-acceleration
    TODO: think about implementing ALI

    Args:
        model (Model): The model for which to compute the level populations.
        device (torch.device): The device on which to compute.
        max_n_iterations (int, optional): The maximum number of iterations to perform. Defaults to 50.

    Returns:
        None
    """
    #TODO: check if level pops have converged
    #to make sure that everything is up to date, we infer
    model.dataCollection.infer_data()
    if use_ng_acceleration:
        relative_diff_default_its: float
        relative_diff_ng_accel: float

        previous_level_pops: list[torch.Tensor] = [] #dims: [[NUMBER_ITS, parameters.npoints, parameters.nlev] for every line species]
        last_ng_accelerated_pops: list[torch.Tensor] = [] #dims: [[parameters.npoints, parameters.nlev] for every line species]
        for linespecidx in range(model.parameters.nlspecs.get()):#initialize for every species
            previous_level_pops.append(model.sources.lines.lineProducingSpecies[linespecidx].population.get(device).unsqueeze(0))
            last_ng_accelerated_pops.append(torch.zeros((model.parameters.npoints.get(), model.sources.lines.lineProducingSpecies[linespecidx].linedata.nlev.get()), dtype=Types.LevelPopsInfo, device=device))
        n_its_in_ng_accel: int = 1
            
        for i in range(max_n_iterations):
            print("it:", i)

            all_species_converged = True
            relative_diff_default_its = 0.0

            mean_line_intensities = solve_long_characteristics_NLTE(model, device).to(device=device)
            computed_level_pops: list[torch.Tensor] = compute_level_populations_statistical_equilibrium(model, mean_line_intensities, device)
            for lspecidx in range(model.parameters.nlspecs.get()):
                lspec = model.sources.lines.lineProducingSpecies[lspecidx]
                if not level_pops_converged(lspec.population.get(device), computed_level_pops[lspecidx]):
                    all_species_converged = False
                relative_diff_default_its += torch.mean(relative_error(computed_level_pops[lspecidx], lspec.population.get(device))).item()
                lspec.population.set(computed_level_pops[lspec.linedata.num.get()].to("cpu"))
                #Add current level pops to previous level pops
                previous_level_pops[lspec.linedata.num.get()] = torch.cat((previous_level_pops[lspec.linedata.num.get()], computed_level_pops[lspec.linedata.num.get()].unsqueeze(0)), dim=0)
            n_its_in_ng_accel += 1

            model.dataCollection.infer_data()

            if all_species_converged:
                break

            relative_diff_ng_accel = 0.0
            #only start using the acceleration procedure after accumulating 3 iterations
            if n_its_in_ng_accel >= 3:#TODO: replace with len(previous_level_pops[0])

                for lspecidx in range(model.parameters.nlspecs.get()):
                    lspec = model.sources.lines.lineProducingSpecies[lspecidx]
                    ng_accel_level_pops = lspec.compute_ng_accelerated_level_pops(previous_level_pops[lspec.linedata.num.get()], device)
                    relative_diff_ng_accel += torch.mean(relative_error(ng_accel_level_pops, last_ng_accelerated_pops[lspec.linedata.num.get()])).item()
                    last_ng_accelerated_pops[lspec.linedata.num.get()] = ng_accel_level_pops
                
                #Use the ng-accelerated populations, if the difference between ng-accelerated iterations is smaller than the difference between the regular iterations
                if relative_diff_ng_accel < relative_diff_default_its or n_its_in_ng_accel > max_its_between_ng_accel:
                    print("Using ng-acceleration; using " + str(n_its_in_ng_accel-1) + " iterations")
                    #redo convergence checking
                    all_species_converged = True
                    for lspecidx in range(model.parameters.nlspecs.get()):
                        lspec = model.sources.lines.lineProducingSpecies[lspecidx]
                        if not level_pops_converged(lspec.population.get(device), last_ng_accelerated_pops[lspecidx]):
                            all_species_converged = False
                        lspec.population.set(last_ng_accelerated_pops[lspec.linedata.num.get()].to("cpu"))
                    previous_level_pops = [last_ng_accelerated_pops[lspecidx].unsqueeze(0) for lspecidx in range(model.parameters.nlspecs.get())]
                    n_its_in_ng_accel = 1

                    model.dataCollection.infer_data()
            
                    if all_species_converged:
                        break

    else:
        for i in range(max_n_iterations):
            print("it:", i)
            all_species_converged = True
            mean_line_intensities = solve_long_characteristics_NLTE(model, device).to(device=device)
            computed_level_pops: list[torch.Tensor] = compute_level_populations_statistical_equilibrium(model, mean_line_intensities, device)#type: ignore
            for lspecidx in range(model.parameters.nlspecs.get()):
                lspec = model.sources.lines.lineProducingSpecies[lspecidx]
                if not level_pops_converged(lspec.population.get(device), computed_level_pops[lspecidx]):
                    all_species_converged = False

                lspec.population.set(computed_level_pops[lspecidx].to("cpu"))
            model.dataCollection.infer_data()
            if all_species_converged:
                break
            


def image_model(model: Model, ray_direction: torch.Tensor, freqs: torch.Tensor, device: torch.device, Nxpix: int = 256, Nypix: int = 256, imageType: ImageType = ImageType.Intensity) -> None:
    """
    Computes an image of the model for a given ray direction at the specified indices.
    Appends the resulting image to model.images

    Args:
    - model (Model): The model to use for computing the image.
    - ray_direction (torch.Tensor): The direction of the ray. Has dimensions [3].
    - freqs (torch.Tensor): The frequency range to use for computing the image. Has dimensions [NFREQS].
    - device (torch.device): The device on which to compute.
    - Nxpix (int, optional): The number of pixels in the x direction. Defaults to 256.
    - Nypix (int, optional): The number of pixels in the y direction. Defaults to 256.
    - imageType (ImageType, optional): The type of image to compute. Defaults to ImageType.Intensity. TODO: not implemented for other types

    Returns:
    - None
    """
    image_index: int = len(model.images)
    image = Image(model.parameters, model.dataCollection, imageType, ray_direction, freqs, image_index)
    image.setup_image(model.geometry, Nxpix, Nypix, imageType)
    image.imageType.set(imageType)
    freqhelper = FrequencyEvalHelper(freqs.unsqueeze(0).repeat(model.parameters.npoints.get(), 1), model.sources.lines.lineProducingSpecies.list, model.geometry.points.velocity.get(device), device)#type: ignore

    if model.geometry.geometryType.get() == GeometryType.General3D:
        start_positions, start_indices = image.transform_pixel_coordinates_to_3D_starting_coordinates(model.geometry, device)
        start_velocities = torch.zeros_like(start_positions)
        match imageType:
            case ImageType.Intensity:
                intensities = solve_long_characteristics_single_direction(model, ray_direction, start_positions, start_velocities, start_indices, freqhelper, device)
                image.I.set(intensities)
            case ImageType.OpticalDepth:
                optical_depths = get_total_optical_depth_single_direction(model, ray_direction, start_positions, start_velocities, start_indices, freqhelper, device)
                image.I.set(optical_depths)

    elif model.geometry.geometryType.get() == GeometryType.SpericallySymmetric1D:
        start_positions, start_indices = image.transform_pixel_coordinates_to_1D_starting_coordinates(model.geometry, device)
        start_velocities = torch.zeros_like(start_positions)
        match imageType:
            case ImageType.Intensity:
                intensities = solve_long_characteristics_single_direction(model, ray_direction, start_positions, start_velocities, start_indices, freqhelper, device)
                image.I.set(intensities)
            case ImageType.OpticalDepth:
                optical_depths = get_total_optical_depth_single_direction(model, ray_direction, start_positions, start_velocities, start_indices, freqhelper, device)
                image.I.set(optical_depths)

    model.images.append(image)
