import torch
from magrittetorch.utils.storagetypes import Types


#adapted from https://stackoverflow.com/questions/64004559/is-there-multi-arange-in-numpy
#in this a[:,0] is the start, a[:,1] is the stop, a[:,2] is the step size (not necessary for me)
# def multi_arange(a):
#     steps = a[:,2]
#     lens = ((a[:,1]-a[:,0]) + steps-np.sign(steps))//steps
#     b = np.repeat(steps, lens)
#     ends = (lens-1)*steps + a[:,0]
#     b[0] = a[0,0]
#     b[lens[:-1].cumsum()] = a[1:,0] - ends[:-1]
#     return b.cumsum()
    
def multi_arange(start : torch.Tensor, delta : torch.Tensor, device : torch.device) -> torch.Tensor:
    """Variant of torch.arange which handles tensor input and aranges the entire tensor at once

    Args:
        start (torch.Tensor|int): 1D torch.Tensor of starting indices for each arange 
        delta (torch.Tensor|int): 1D torch.Tensor of a mount of indices per arange 
        device (torch.device): Device on which the tensor must be created 

    Returns:
        torch.Tensor: The resulting 1D tensor of arange'd pieces put after eachother 
    """
    #delta is 1D tensor, so sum will be a 0D tensor
    keep_indices: torch.Tensor = delta!=0#zero deltas cause nonsense, so prune them
    keep_delta = delta[keep_indices]
    keep_start = start[keep_indices]
    #increment will contain the increments required
    increment : torch.Tensor = torch.ones(torch.sum(keep_delta, dim=0), device=device, dtype=Types.IndexInfo) #type: ignore
    keep_end = keep_delta-1 + keep_start
    increment[0] = keep_start[0]
    increment[keep_delta[:-1].cumsum(dim = 0, dtype=Types.IndexInfo)] = keep_start[1:]-keep_end[:-1]
    
    return increment.cumsum(dim = 0, dtype=Types.IndexInfo)



def interpolate1D(interpolation_position: torch.Tensor, interpolation_value: torch.Tensor, evaluation_points: torch.Tensor) -> torch.Tensor:
    """Interpolates a discrete set of (position, value)'s at the given evaluation_points.

    Args:
        interpolation_position (torch.Tensor): Sorted 1D positions. dims:[N_INTERP_VALUES]
        interpolation_value (torch.Tensor): Corresponding value to interpolate. dims:[N_INTERP_VALUES]
        evaluation_points (torch.Tensor): Positions at which to interpolate. dims: [arbitrary]

    Returns:
        torch.Tensor: The interpolated  dims: same dimensions as evaluation_points
    """
    n_interp_values: int = interpolation_position.shape[0]
    indices = torch.searchsorted(interpolation_position, evaluation_points)#dims: [arbitrary]
    minval, maxval = torch.min(interpolation_value), torch.max(interpolation_value)#dims: [1]
    result = torch.zeros_like(evaluation_points, dtype=interpolation_value.dtype)
    inboundind = torch.logical_and(indices > 0, indices < n_interp_values)
    #computing linearly interpolated 1D values
    posdiff = interpolation_position[indices[inboundind]] - evaluation_points[inboundind]
    result[inboundind] = interpolation_value[indices[inboundind]] - posdiff * \
        (interpolation_value[indices[inboundind]]-interpolation_value[indices[inboundind]-1])/ \
        (interpolation_position[indices[inboundind]]-interpolation_position[indices[inboundind]-1])

    zeroind = indices == 0
    maxind = indices == n_interp_values
    #bounding out of bounds indices/positions correctly
    result[zeroind] = minval
    result[maxind] = maxval

    return result

def interpolate1D_extended(interpolation_position: torch.Tensor, interpolation_value: torch.Tensor, evaluation_points: torch.Tensor) -> torch.Tensor:
    """Interpolates a discrete set of (position, value)'s at the given evaluation_points.

    Args:
        interpolation_position (torch.Tensor): Sorted 1D positions. dims:[N_INTERP_VALUES]
        interpolation_value (torch.Tensor): Corresponding value to interpolate. dims:[N_INTERP_VALUES, OTHERDIMS]
        evaluation_points (torch.Tensor): 2D Positions at which to interpolate. dims: [Any1, OTHERDIMS]

    Returns:
        torch.Tensor: The interpolated  dims: [Any1, OTHERDIMS]
    """
    n_interp_values: int = interpolation_position.shape[0]
    evaldim_size: int = evaluation_points.shape[0]
    indices = torch.searchsorted(interpolation_position, evaluation_points)#dims: [Any1]
    minval = torch.zeros_like(evaluation_points)
    eval_size_repeat = list(evaluation_points.size())
    eval_size_repeat[0] = 1

    minval = torch.min(interpolation_value, dim=0).values
    maxval = torch.max(interpolation_value, dim=0).values#dims: [OTHERDIMS]
    minval = minval[None, :]
    maxval = maxval[None, :]
    minval = minval.repeat(evaldim_size, 1)#dims: [Any1, OTHERDIMS]
    maxval = maxval.repeat(evaldim_size, 1)
    result = torch.zeros_like(evaluation_points, dtype=interpolation_value.dtype)
    #hmm, I'll just ignore some wrong stuff and just use the interpolation formula on all points
    #I make sure that no indices overflow because of being out of bounds, so use modulo operator
    modified_indices = indices%n_interp_values#dims: [Any1]
    modified_indicesmin1 = (indices+n_interp_values-1)%n_interp_values
    index_select = torch.arange(evaldim_size).unsqueeze(1)
    posdiff = interpolation_position[modified_indices] - evaluation_points
    result = interpolation_value[modified_indices, index_select] - posdiff * \
        (interpolation_value[modified_indices, index_select]-interpolation_value[modified_indicesmin1, index_select])/ \
        (interpolation_position[modified_indices]-interpolation_position[modified_indicesmin1])
    
    zeroind = indices == 0#dims: [Any1, otherdims]
    maxind = indices == n_interp_values
    #bounding out of bounds indices/positions correctly
    result[zeroind] = minval[zeroind]
    result[maxind] = maxval[maxind]

    return result


def interpolate2D_linear(interpolation_position: torch.Tensor, interpolation_value: torch.Tensor, evaluation_points: torch.Tensor) -> torch.Tensor:
    """Interpolates a discrete set of (position, (vector of) value)'s at the given evaluation_points.

    Args:
        interpolation_position (torch.Tensor): Sorted 1D positions dims: [N_INTERP_VALUES]
        interpolation_value (torch.Tensor): Corresponding matrix containing vectors of values to interpolate. dims: [N_INTERP_VALUES, OTHERDIMS]
        evaluation_points (torch.Tensor): 1D Tensor of positions to interpolate for. dims: [Any1]

    Returns:
        torch.Tensor: The interpolated matrix dims: [Any1, OTHERDIMS]
    """
    n_interp_values: int = interpolation_position.shape[0]
    otherdims_size: int = interpolation_value.shape[1]
    evaldim_size: int = evaluation_points.shape[0]
    indices = torch.searchsorted(interpolation_position, evaluation_points).repeat(otherdims_size, 1).T#dims: [Any1, OTHERDIMS]
    minval = torch.zeros_like(evaluation_points)
    eval_size_repeat = list(evaluation_points.size())
    eval_size_repeat[0] = 1

    minval = torch.min(interpolation_value, dim=0).values
    maxval = torch.max(interpolation_value, dim=0).values#dims: [OTHERDIMS]
    minval = minval.repeat(evaldim_size, 1)#dims: [Any1, OTHERDIMS]
    maxval = maxval.repeat(evaldim_size, 1)
    result = torch.zeros_like(evaluation_points, dtype=interpolation_value.dtype)
    #hmm, I'll just ignore some wrong stuff and just use the interpolation formula on all points
    #I make sure that no indices overflow because of being out of bounds, so use modulo operator
    modified_indices = indices%n_interp_values#dims: [Any1, OTHERDIMS]
    modified_indicesmin1 = (indices+n_interp_values-1)%n_interp_values#dims: [Any1, OTHERDIMS]
    #Define this index_select in order to properly index the 2d tensors
    #It just needs to contain the correct index value at the relevant dimension
    index_select = torch.arange(otherdims_size).unsqueeze(0)#dims: [1, OTHERDIMS]

    posdiff = interpolation_position[modified_indices] - evaluation_points[:, None]
    result = interpolation_value[modified_indices, index_select] - posdiff * \
        (interpolation_value[modified_indices, index_select]-interpolation_value[modified_indicesmin1, index_select])/ \
        (interpolation_position[modified_indices]-interpolation_position[modified_indicesmin1])
    
    zeroind = indices == 0#dims: [Any1, OTHERDIMS]
    maxind = indices == n_interp_values#dims: [Any1, OTHERDIMS]
    #bounding out of bounds indices/positions correctly
    result[zeroind] = minval[zeroind]
    result[maxind] = maxval[maxind]

    return result