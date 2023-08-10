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
    #increment will contain the increments required
    #delta is 1D tensor, so sum will be a 0D tensor
    increment : torch.Tensor = torch.ones(torch.sum(delta, dim=0), device=device).type(Types.IndexInfo) #type: ignore
    end = delta-1 + start
    increment[0] = start[0]
    increment[delta[:-1].cumsum(dim = 0)] = start[1:]-end[:-1]
    return increment.cumsum(dim = 0)
