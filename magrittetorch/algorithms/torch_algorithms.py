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
