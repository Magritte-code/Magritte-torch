from magrittetorch.algorithms.raytracer import trace_rays_sparse, RaytracerGenerator
from magrittetorch.model.model import Model
import torch
import time


def solve_long_characteristics(model: Model, direction: torch.Tensor, device: torch.device = torch.device("cpu")) -> None:
    start =time.time()
    point_ind, distances, scatter_ind = trace_rays_sparse(model, direction) #dims: [N_POINTS_TO_EVAL]
    end = time.time()
    direction_device = direction.to(device)[None, :]
    print(point_ind.get_device(), distances.get_device(), scatter_ind.get_device())
    print("ray trace time", end-start)
    NLTE_freqs = model.sources.lines.get_all_line_frequencies(device=device) #dims: [parameters.npoints, NFREQS]#TODO: move to function arguments
    model_velocities = model.geometry.points.velocity.get().to(device) #dims: [parameters.npoints, 3]
    model_positions = model.geometry.points.position.get().to(device) #dims: [parameters.npoints, 3]
    origin_velocities = model_velocities[scatter_ind, :]#err, assumes behavior that might change in the future# dims: [N_POINTS_TO_EVAL]
    origin_positions = model_positions[scatter_ind, :]#err, assumes behavior that might change in the future # dims: [N_POINTS_TO_EVAL]
    # point_velocities = model_velocities[point_ind, :] #dims: [parameters.npoints, 3]
    print("computing doppler")
    doppler_shift = model.geometry.get_doppler_shift(point_ind.to(device), origin_positions, origin_velocities, direction_device, distances.to(device), device)
    print("shift", doppler_shift)
    freqs_to_evaluate = NLTE_freqs[point_ind, :]*doppler_shift[:, None]#TODO: add doppler shift here
    print("evaluating opacities")
    # print()
    opacities, emissivities = model.sources.get_total_opacity_emissivity(point_ind, freqs_to_evaluate, device)
    print(opacities, emissivities)


def solve_long_characteristics_with_generator(model:Model, direction: torch.Tensor, device: torch.device) -> None:
    #note: this is just a test function for now
    #I might change the api completely
    #also: no memory management yet, should be implemented in some function above this one; providing which points we need
    NLTE_freqs = model.sources.lines.get_all_line_frequencies(device=device) #dims: [parameters.npoints, NFREQS]#TODO: move to function arguments
    model_velocities = model.geometry.points.velocity.get().to(device) #dims: [parameters.npoints, 3]
    model_positions = model.geometry.points.position.get().to(device) #dims: [parameters.npoints, 3]
    origin_velocities = model_velocities[:, :]#err, assumes behavior that might change in the future# dims: [N_POINTS_TO_EVAL]
    origin_positions = model_positions[:, :]#err, assumes behavior that might change in the future # dims: [N_POINTS_TO_EVAL]
    
    for (next_point, travelled_distance, original_index) in RaytracerGenerator(model, direction, device):

        doppler_shift = model.geometry.get_doppler_shift(next_point, origin_positions[original_index], origin_velocities[original_index], direction, travelled_distance, device)
        print("shift", doppler_shift, doppler_shift.size())
        freqs_to_evaluate = NLTE_freqs[next_point, :]*doppler_shift[:, None]#TODO: add doppler shift here
        # print("evaluating opacities")
        # print()
        opacities, emissivities = model.sources.get_total_opacity_emissivity(next_point, freqs_to_evaluate, device)
        # print(a, b, ids)
        #TODO: fix function arguments; freqs, dist
        print("nexpoitn", next_point.size(), "freqs", freqs_to_evaluate.size(), "dist", travelled_distance.size(), "shift", doppler_shift.size(), "op/em", opacities.size())
        optical_depths = model.sources.get_total_optical_depth(next_point, freqs_to_evaluate, freqs_to_evaluate, travelled_distance, doppler_shift, doppler_shift, opacities, emissivities, device)
        pass


