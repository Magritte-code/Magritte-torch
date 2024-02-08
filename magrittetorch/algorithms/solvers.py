from magrittetorch.algorithms.torch_algorithms import interpolate2D_linear
from magrittetorch.model.model import Model
from magrittetorch.model.geometry.geometry import GeometryType
from magrittetorch.model.sources.frequencyevalhelper import FrequencyEvalHelper, ALIFreqEvalHelper
from magrittetorch.utils.storagetypes import Types
from magrittetorch.utils.constants import min_opacity, min_optical_depth, convergence_fraction, min_rel_pop_for_convergence, min_level_pop
from magrittetorch.model.image import Image, ImageType
from magrittetorch.tools.radiativetransferutils import relative_error

import torch
import time


def solve_long_characteristics_ALI_diag_single_direction(model: Model, raydir: torch.Tensor, start_positions: torch.Tensor, start_velocities: torch.Tensor, start_indices: torch.Tensor, freqhelper: FrequencyEvalHelper, ALIfreqhelper: ALIFreqEvalHelper, device: torch.device) -> torch.Tensor:
    """Computes the ALI diagonal for a single direction, for all given start positions.
    Note: all data must be consistent with the normal solvers.solve_long_characteristics_single_direction function, as this function is a direct extension of that function.

    Args:
        model (Model): The model on which to compute
        raydir (torch.Tensor): Direction of the ray. Has dimensions [3]
        start_positions (torch.Tensor): Positions of all points in 3D space. Has dimensions [parameters.npoints, 3]
        start_velocities (torch.Tensor): Corresponding velocities all points in 3D space. Has dimensions [parameters.npoints, 3]
        start_indices (torch.Tensor): Point indices to start tracing the rays. Has dimensions [parameters.npoints].
        freqhelper (FrequencyEvalHelper): Frequency evaluation helper object for narrow spectral lines.
        ALIfreqhelper (ALIFreqEvalHelper): Frequency evaluation helper object for ALI calculations.
        device (torch.device): Device on which to compute and return the result.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple containing the ALI diagonal fraction and the fraction times source function. Both have dimensions [parameters.npoints, model.sources.lines.get_total_number_lines]
    """
    #Note: contains some duplicated code from solve_long_characteristics_single_direction
    positions_device = model.geometry.points.position.get(device) #dims: [parameters.npoints, 3]
    npoints = start_positions.size(dim=0)#number of points to trace NPOINTS

    #get neighbors per direction; saves 50% time for raytracing in 3D geometries
    neighbors_device, n_neighbors_device, cum_n_neighbors_device = model.geometry.get_neighbors_in_direction(raydir, device) 
    #dims: [sum(n_neighbors_in_correct_direction)], [parameters.npoints], [parameters.npoints]


    is_boundary_point_device = model.geometry.boundary.get_boundary_in_direction(raydir, device) #dims: [parameters.npoints]
    # computed_intensity = torch.zeros((npoints, freqhelper.original_frequencies.size(dim=1)), dtype=Types.FrequencyInfo, device=device) #dims: [NPOINTS, NFREQS]

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
    n_active_rays = torch.sum(mask_active_rays) #number of active rays, dims: []
    data_indices = torch.arange(npoints, device=device) #dims: [N_ACTIVE_RAYS<=NPOINTS]
    # boundary_indices = curr_ids[boundary_mask] #dims: [N_ACTIVE_RAYS<=NPOINTS]
    prev_shift = model.geometry.get_doppler_shift(curr_ids, start_positions, start_velocities, raydir, distance_travelled, device) #dims: [NPOINTS]
    prev_opacities, starting_emissivities = model.sources.get_total_opacity_emissivity_freqhelper(start_ids, curr_ids, prev_shift, freqhelper, device) #dims: [NPOINTS, NFREQS]
    prev_source_function = starting_emissivities/(prev_opacities+min_opacity) #dims: [NPOINTS, NFREQS]
    # We need to compute the source function at the current point for the ALI contribution
    single_line_opacities, prev_single_line_emissivities = model.sources.get_total_opacity_emissivity_freqhelper(start_ids, curr_ids, prev_shift, ALIfreqhelper, device) #dims: [NPOINTS, NFREQS]
    single_line_source_function = prev_single_line_emissivities/(single_line_opacities+min_opacity) #dims: [NPOINTS, NFREQS]

    #already mask previous values
    prev_opacities = prev_opacities[mask_active_rays] #dims: [N_ACTIVE_RAYS<=NPOINTS, NFREQS]
    prev_single_line_opacities = single_line_opacities[mask_active_rays] #dims: [N_ACTIVE_RAYS<=NPOINTS, NFREQS]
    prev_source_function = prev_source_function[mask_active_rays] #dims: [N_ACTIVE_RAYS<=NPOINTS, NFREQS]
    prev_shift = prev_shift[mask_active_rays] #dims: [N_ACTIVE_RAYS<=NPOINTS]
    single_line_source_function = single_line_source_function[mask_active_rays] #dims: [N_ACTIVE_RAYS<=NPOINTS, NFREQS]

    ALI_diag_single_dir_fraction = torch.zeros((npoints, model.sources.lines.get_total_number_lines()), dtype=Types.FrequencyInfo, device=device) #dims: [NPOINTS, NFREQS]
    ALI_diag_single_dir_Jdiff = torch.zeros((npoints, model.sources.lines.get_total_number_lines()), dtype=Types.FrequencyInfo, device=device) #dims: [NPOINTS, NFREQS]

    #only compute ALI stuff for non-boundary points (per direction); For zero distances, the contribution will be zero anyway
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


    #After raytracing, we now compute everything required
    doppler_shift = model.geometry.get_doppler_shift(curr_ids, start_positions, start_velocities, raydir, distances, device)
    curr_opacities, _ = model.sources.get_total_opacity_emissivity_freqhelper(start_ids, curr_ids, doppler_shift, freqhelper, device)
    optical_depths = model.sources.get_total_optical_depth_freqhelper(curr_ids, prev_ids, start_ids, freqhelper, distance_increment, doppler_shift, prev_shift, curr_opacities, prev_opacities, device)
    single_line_opacities, _ = model.sources.get_total_opacity_emissivity_freqhelper(start_ids, curr_ids, doppler_shift, ALIfreqhelper, device)
    single_line_optical_depths = model.sources.get_total_optical_depth_freqhelper(curr_ids, prev_ids, start_ids, ALIfreqhelper, distance_increment, doppler_shift, prev_shift, single_line_opacities, prev_single_line_opacities, device)

    #To compute the intensity contribution, we need to multiply the source function by the factors below
    # This is exactly the ALI diagonal (minus averaging over direction and line profile, and ignoring the fact that continuum/other lines might also contribute)
    close_factor = (1+torch.expm1(-optical_depths-min_optical_depth)/(optical_depths+min_optical_depth))

    encountered_freqs: int = 0
    encountered_lines: int = 0
    #And average already over the line profile
    for specidx in range(model.sources.lines.lineProducingSpecies.length_param.get()):
        lspec = model.sources.lines.lineProducingSpecies[specidx]
        nfreqs: int = lspec.get_n_lines_freqs()
        nlines: int = lspec.linedata.nrad.get()
        npoints: int = model.parameters.npoints.get()
        expanded_quad_weights = lspec.get_line_weights(device) #dims: [N_LINE_FREQS]
        expanded_line_index = lspec.get_line_indices_NTLE(device) #dims: [lspec.linequadrature.nquads*lspec.linedata.nlines=N_LINE_FREQS]

        #The ALI diagonal is defined by the intensity generated locally because of the line; without any contributions from other lines
        # Practically, this means we need the 'close factor', multiplied by the contribution fraction of the optical depth;
        # This all times the source function of the line, and then integrated over the line quadrature (frequency space) and averaged over the direction
        # dims: [N_ACTIVE_RAYS<=NPOINTS, N_LINE_FREQS]
        single_line_optical_depths = model.sources.get_total_optical_depth_freqhelper(curr_ids, prev_ids, start_ids, freqhelper, distance_increment, doppler_shift, doppler_shift, curr_opacities[:, encountered_freqs:encountered_freqs+nfreqs], prev_opacities[:, encountered_freqs:encountered_freqs+nfreqs], device)
        # dims: [N_ACTIVE_RAYS<=NPOINTS, NFREQS]
        ALI_fraction = expanded_quad_weights[None, :] * close_factor[:,encountered_freqs:encountered_freqs+nfreqs] * single_line_optical_depths[:, encountered_freqs:encountered_freqs+nfreqs]/(optical_depths[:, encountered_freqs:encountered_freqs+nfreqs]+min_optical_depth)
        # dims: [N_ACTIVE_RAYS<=NPOINTS, NFREQS]
        ALI_Jdiff = ALI_fraction * single_line_source_function[:, encountered_freqs:encountered_freqs+nfreqs]

        #Dummy data for scatter_add, with correct dimensions
        ALI_diag_dummy = torch.zeros((n_active_rays, nlines), dtype=Types.FrequencyInfo, device=device) #dims: [NPOINTS, NFREQS]

        #dims: [N_ACTIVE_RAYS<=NPOINTS, lspec.linedata.nrad]
        ALI_diag_single_dir_fraction[mask_active_rays, encountered_lines:encountered_lines+nlines] = torch.scatter_add(ALI_diag_dummy, 1, expanded_line_index[None, :].expand(n_active_rays, -1), ALI_fraction)
        #dims: [N_ACTIVE_RAYS<=NPOINTS, lspec.linedata.nrad]
        ALI_diag_single_dir_Jdiff[mask_active_rays, encountered_lines:encountered_lines+nlines] = torch.scatter_add(ALI_diag_dummy, 1, expanded_line_index[None, :].expand(n_active_rays, -1), ALI_Jdiff)

        encountered_freqs += nfreqs
        encountered_lines += nlines

    return ALI_diag_single_dir_fraction, ALI_diag_single_dir_Jdiff #NOT YET AVERAGED IN ANGULAR DIRECTION


def solve_long_characteristics_ALI_diag(model: Model, device: torch.device) -> torch.Tensor:
    """Computes the ALI diagonal for all given start positions.
    Note: all data must be consistent with the normal solvers.solve_long_characteristics_single_direction function, as this function is based on a direct extension of that function.

    Args:
        model (Model): The model on which to compute
        start_positions (torch.Tensor): Positions of all points in 3D space. Has dimensions [parameters.npoints, 3]
        start_velocities (torch.Tensor): Corresponding velocities all points in 3D space. Has dimensions [parameters.npoints, 3]
        start_indices (torch.Tensor): Point indices to start tracing the rays. Has dimensions [parameters.npoints].
        freqhelper (FrequencyEvalHelper): Frequency evaluation helper object for narrow spectral lines.
        ALIfreqhelper (ALIFreqEvalHelper): Frequency evaluation helper object for ALI calculations.
        device (torch.device): Device on which to compute and return the result.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple containing the ALI diagonal fraction and the fraction times source function. Both have dimensions [parameters.npoints, model.sources.lines.get_total_number_lines]
    """
    start_positions = model.geometry.points.position.get(device) #dims: [parameters.npoints, 3]
    start_velocities = model.geometry.points.velocity.get(device) #dims: [parameters.npoints, 3]
    start_indices = torch.arange(model.parameters.npoints.get(), device=device) #dims: [parameters.npoints]
    freqhelper = FrequencyEvalHelper(model.sources.lines.get_all_line_frequencies(device=device), model.sources.lines.lineProducingSpecies.get(), start_velocities, device)
    ALIfreqhelper = ALIFreqEvalHelper(model.sources.lines.get_all_line_frequencies(device=device), model.sources.lines.lineProducingSpecies.get(), device)
    #Integrate over all directions
    ALI_diag_fraction = torch.zeros((model.parameters.npoints.get(), model.sources.lines.get_total_number_lines()), dtype=Types.FrequencyInfo, device=device) #dims: [NPOINTS, NFREQS]
    ALI_diag_Jdiff = torch.zeros((model.parameters.npoints.get(), model.sources.lines.get_total_number_lines()), dtype=Types.FrequencyInfo, device=device) #dims: [NPOINTS, NFREQS]
    raydirs = model.geometry.rays.direction.get(device)
    weights = model.geometry.rays.weight.get(device)
    rr = 0
    for raydir_index in range(raydirs.shape[0]):
        #Adding the results immediately, as we will otherwise run out of memory
        ALI_diag_single_dir_fraction, ALI_diag_single_dir_Jdiff = solve_long_characteristics_ALI_diag_single_direction(model, raydirs[raydir_index], start_positions, start_velocities, start_indices, freqhelper, ALIfreqhelper, device)
        ALI_diag_fraction += ALI_diag_single_dir_fraction * weights[raydir_index]
        ALI_diag_Jdiff += ALI_diag_single_dir_Jdiff * weights[raydir_index]
        rr += 1

    return ALI_diag_fraction, ALI_diag_Jdiff

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
        compute_ALI_diag (bool, optional): Whether to compute the ALI diagonal. Defaults to False.

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
        #Adding the results immediately, as we will otherwise run out of memory
        total_intensity += weights[raydir_index] * solve_long_characteristics_single_direction(model, raydirs[raydir_index,:], model_positions, model_velocities, torch.arange(model.parameters.npoints.get(), device=device), freqhelper, device)
    print("time for this iteration: ", time.time()-starttime, "s")


    encountered_freqs: int = 0
    encountered_lines: int = 0
    # now numerically integrate the intensities of each different line# TODO: check if better option exists, without for loop
    for specidx in range(model.sources.lines.lineProducingSpecies.length_param.get()):
        lspec = model.sources.lines.lineProducingSpecies[specidx]
        nfreqs: int = lspec.get_n_lines_freqs()
        nlines: int = lspec.linedata.nrad.get()
        npoints: int = model.parameters.npoints.get()
        expanded_quad_weights = lspec.get_line_weights(device)#needs to be expanded to match the number of frequencies
        expanded_line_index = lspec.get_line_indices_NTLE(device) #dims: [nquads*nlines=NFREQS]

        total_integrated_line_intensity[:, encountered_lines:encountered_lines+nlines].scatter_add_(1, expanded_line_index[None, :].expand(npoints, -1), expanded_quad_weights[None, :] * total_intensity[:,encountered_freqs:encountered_freqs+nfreqs])

        encountered_freqs += nfreqs
        encountered_lines += nlines

    return total_integrated_line_intensity


def compute_level_populations_statistical_equilibrium(model: Model, J: torch.Tensor, device: torch.device, ALI_diag: tuple[torch.Tensor, torch.Tensor] = None) -> list[torch.Tensor]:
    """Computes the level populations using the statistical equilibrium equations, given the mean line intensities J
    TODO: also include accelerated lambda iteration, just add (optional) extra functon arguments

    Args:
        model (Model): The model
        J (torch.Tensor): The (effective) mean line intensity. Has dims [parameters.npoints, parameters.nlines?]
        device (torch.device): Device on which to compute
        ALI_diag (tuple[torch.Tensor, torch.Tensor], optional): The ALI diagonal fraction and the fraction times the source function. Defaults to None.
        Both have dimensions [parameters.npoints, model.sources.lines.get_total_number_lines]

    Returns:
        The list of updated level populations. Has dims [[parameters.npoints, parameters.nlev] for every line species]
    """

    if (ALI_diag is None):
        ALI_diag_fraction = torch.zeros((model.parameters.npoints.get(), model.sources.lines.get_total_number_lines()), dtype=Types.FrequencyInfo, device=device) #dims: [NPOINTS, NLINES]
        ALI_diag_Jdiff = torch.zeros((model.parameters.npoints.get(), model.sources.lines.get_total_number_lines()), dtype=Types.FrequencyInfo, device=device) #dims: [NPOINTS, NLINES]
    else:
        ALI_diag_fraction, ALI_diag_Jdiff = ALI_diag #dims: [NPOINTS, NLINES]


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

        #Correct J for ALI
        Jrelevant = (Jrelevant - ALI_diag_Jdiff[:, total_encountered_lines:total_encountered_lines+nlines])

        rate_upper_to_lower = einsteinA[None, :] * (1 - ALI_diag_fraction[:, total_encountered_lines: total_encountered_lines+nlines]) \
            + einsteinBs[None, :] * Jrelevant#dims: [npoints, nlines]
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
        # Put negative level populations to zero and renormalize
        levelpops[levelpops < 0.0] = 0.0
        levelpops = levelpops * lspec.population_tot.get(device)[:, None]/torch.sum(levelpops, dim=1)[:, None]
        computed_level_pops.append(levelpops)
        total_encountered_lines+=nlines

    return computed_level_pops


def level_pops_converged(previous_level_pops: torch.Tensor, current_level_pops: torch.Tensor, relative_diff_threshold: float = min_rel_pop_for_convergence, convergence_fraction: float = convergence_fraction) -> bool:
    print("mean relative error:", torch.mean(relative_error(previous_level_pops+min_level_pop, current_level_pops+min_level_pop)).item())
    # print("relative diff threshold:", relative_diff_threshold)
    # print("convergence fraction:", convergence_fraction)
    print("fraction converged:", torch.mean((relative_error(previous_level_pops+min_level_pop, current_level_pops+min_level_pop) < relative_diff_threshold).type(Types.LevelPopsInfo)))
    print("converged:", torch.mean((relative_error(previous_level_pops+min_level_pop, current_level_pops+min_level_pop) < relative_diff_threshold).type(Types.LevelPopsInfo)).item() > convergence_fraction)
    return (torch.mean((relative_error(previous_level_pops+min_level_pop, current_level_pops+min_level_pop) < relative_diff_threshold).type(Types.LevelPopsInfo)).item() > convergence_fraction)


def compute_level_populations(model: Model, device: torch.device, max_n_iterations: int = 50, use_ng_acceleration: bool = True, use_ALI: bool = True, max_its_between_ng_accel: int = 8) -> None:
    """
    Computes the level populations of the given model using an iterative approach,
    utilizing the mean line intensities to solve the statistical equilibrium equations.
    Optionally also uses Ng-acceleration and local ALI to speed up the convergence.

    Args:
        model (Model): The model for which to compute the level populations.
        device (torch.device): The device on which to compute.
        max_n_iterations (int, optional): The maximum number of iterations to perform. Defaults to 50.
        use_ng_acceleration (bool, optional): Whether to use Ng-acceleration. Defaults to True.
        use_ALI (bool, optional): Whether to use ALI. Defaults to True.
        max_its_between_ng_accel (int, optional): The maximum number of iterations between Ng-acceleration. Defaults to 8.

    Returns:
        None
    """
    #TODO: check if level pops have converged
    #to make sure that everything is up to date, we infer
    model.dataCollection.infer_data()
    starttime = time.time()
    if use_ng_acceleration:
        relative_diff_default_its: float
        relative_diff_ng_accel: float

        previous_level_pops: list[torch.Tensor] = [] #dims: [[NUMBER_ITS, parameters.npoints, parameters.nlev] for every line species]
        last_ng_accelerated_pops: list[torch.Tensor] = [] #dims: [[parameters.npoints, parameters.nlev] for every line species]
        for linespecidx in range(model.parameters.nlspecs.get()):#initialize for every species
            previous_level_pops.append(model.sources.lines.lineProducingSpecies[linespecidx].population.get(device).unsqueeze(0))
            last_ng_accelerated_pops.append(torch.zeros((model.parameters.npoints.get(), model.sources.lines.lineProducingSpecies[linespecidx].linedata.nlev.get()), dtype=Types.LevelPopsInfo, device=device))
        # n_its_in_ng_accel: int = 1
            
        for i in range(max_n_iterations):
            print("it:", i)

            all_species_converged = True
            relative_diff_default_its = 0.0

            mean_line_intensities = solve_long_characteristics_NLTE(model, device)
            print("total time elapsed:", time.time()-starttime, "s")

            if use_ALI:
                ALI_diag = solve_long_characteristics_ALI_diag(model, device)
                computed_level_pops: list[torch.Tensor] = compute_level_populations_statistical_equilibrium(model, mean_line_intensities, device, ALI_diag)
            else:
                computed_level_pops: list[torch.Tensor] = compute_level_populations_statistical_equilibrium(model, mean_line_intensities, device)
            for lspecidx in range(model.parameters.nlspecs.get()):
                lspec = model.sources.lines.lineProducingSpecies[lspecidx]
                if not level_pops_converged(lspec.population.get(device), computed_level_pops[lspecidx]):
                    all_species_converged = False
                relative_diff_default_its += torch.mean(relative_error(computed_level_pops[lspecidx], lspec.population.get(device))).item()
                lspec.population.set(computed_level_pops[lspec.linedata.num.get()].to("cpu"))
                #Add current level pops to previous level pops
                previous_level_pops[lspec.linedata.num.get()] = torch.cat((previous_level_pops[lspec.linedata.num.get()], computed_level_pops[lspec.linedata.num.get()].unsqueeze(0)), dim=0)

            model.dataCollection.infer_data()

            if all_species_converged:
                break

            relative_diff_ng_accel = 0.0
            #only start using the acceleration procedure after accumulating 3 iterations
            if len(previous_level_pops[0]) > 3:#TODO: replace with len(previous_level_pops[0])

                for lspecidx in range(model.parameters.nlspecs.get()):
                    lspec = model.sources.lines.lineProducingSpecies[lspecidx]
                    ng_accel_level_pops = lspec.compute_ng_accelerated_level_pops(previous_level_pops[lspec.linedata.num.get()][1:], device)
                    relative_diff_ng_accel += torch.mean(relative_error(ng_accel_level_pops, last_ng_accelerated_pops[lspec.linedata.num.get()])).item()
                    last_ng_accelerated_pops[lspec.linedata.num.get()] = ng_accel_level_pops
                
                #Use the ng-accelerated populations, if the difference between ng-accelerated iterations is smaller than the difference between the regular iterations
                if relative_diff_ng_accel < relative_diff_default_its or len(previous_level_pops[0]) > max_its_between_ng_accel:
                    print("Using ng-acceleration; using " + str(len(previous_level_pops[0])-1) + " iterations")
                    #redo convergence checking
                    all_species_converged = True
                    for lspecidx in range(model.parameters.nlspecs.get()):
                        lspec = model.sources.lines.lineProducingSpecies[lspecidx]
                        if not level_pops_converged(lspec.population.get(device), last_ng_accelerated_pops[lspecidx]):
                            all_species_converged = False
                        lspec.population.set(last_ng_accelerated_pops[lspec.linedata.num.get()].to("cpu"))

                    previous_level_pops = [last_ng_accelerated_pops[lspecidx].unsqueeze(0) for lspecidx in range(model.parameters.nlspecs.get())]

                    model.dataCollection.infer_data()
            
                    if all_species_converged:
                        break

    else:
        for i in range(max_n_iterations):
            print("it:", i)
            all_species_converged = True
            mean_line_intensities = solve_long_characteristics_NLTE(model, device).to(device=device)
            print("total time elapsed:", time.time()-starttime, "s")

            if use_ALI:
                print("using ALI")
                ALI_diag = solve_long_characteristics_ALI_diag(model, device)
                computed_level_pops: list[torch.Tensor] = compute_level_populations_statistical_equilibrium(model, mean_line_intensities, device, ALI_diag)
            else:
                computed_level_pops: list[torch.Tensor] = compute_level_populations_statistical_equilibrium(model, mean_line_intensities, device)
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
