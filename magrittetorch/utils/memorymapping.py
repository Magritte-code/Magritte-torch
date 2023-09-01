#The computed data can take up too much memory, thus we need some utilities for facilitating the memory management.
#TODO: for now (7/09/2023), I have only tested it using a single GPU; extend this if needed.
#TODO2: for now (7/09/2023), does not take into account current memory usage

from typing import Optional, Any, Generic, TypeVar, Type, List, Union, Tuple
from magrittetorch.utils.storagetypes import DataCollection
import torch
import psutil

class MemoryManager:
    """Helper object for memory management on GPU's. Assumes for now that all (if any) gpu's are identical
    """

    #TODO: add safety factor for memory usage

    def __init__(self) -> None:
        self.use_gpu : bool = torch.cuda.device_count() > 0 
        self.available_gpus : List[torch.device]= [torch.device("cuda:"+str(i)) for i in range(torch.cuda.device_count())]

    def split_data(self, max_Nindices : int, expected_memory_usage : int) -> List[Tuple[torch.Tensor, torch.device, int]]:
        #returns tensor of indices to use, device on which to map, and also an integer for assembling the result in the correct order
        #mmh, do we just temporarily store all results in a list of things; probably, might be more efficient this way, as we do not necessarily want to wait for all gpu's to be ready with a given iteration
        if not self.use_gpu:
            #get cpu memory
            cpu_mem : int = psutil.virtual_memory().available
            splits = self.generate_splits(max_Nindices, cpu_mem, expected_memory_usage)  
            return [(split, torch.device("cpu"), i) for split, i in zip(splits, range(len(splits)))]

        #for simplicity, I will assume all gpus to be the same. I will also assume all gpu memory to be available for gpu computing
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        ngpus = len(self.available_gpus)
        combined_splits = self.generate_splits(max_Nindices, gpu_mem*ngpus, expected_memory_usage)
        return [(split, gpu, gpuid + ngpus * splitid) for combined_split, splitid in zip(combined_splits, range(len(combined_splits))) for split, gpu, gpuid in zip(torch.chunk(combined_split, ngpus), self.available_gpus, range(ngpus))]


    def generate_splits(self, max_Nindices : int, max_memory: int, expected_memory_usage : int) -> List[torch.Tensor]:
        """Generates splits based on expected and allowed memory usage

        Args:
            max_Nindices (int): Number of indices to distribute
            max_memory (int): Maximum allowed memory usage
            expected_memory_usage (int): Expected memory usage

        Returns:
            List[torch.Tensor]: List of torch.Tensors containing the allowed indices per split
        """
        Nsplits : int = (expected_memory_usage + max_memory - 1) // max_memory
        max_indices_per_split : int = (max_Nindices + Nsplits - 1) // Nsplits
        return torch.split(torch.arange(max_Nindices), [max_indices_per_split])
    
    @staticmethod
    def compute_data_size_torch_tensor(expected_data_size : int, dtype : torch.dtype) -> int:
        """Helper method for finding the size of a torch tensor

        Args:
            expected_data_size (int): Total size (#elements) of the torch tensor (multiplication of size of its dimensions)
            dtype (torch.dtype): type of the torch tensor

        Raises:
            TypeError: Dev error: if the torch.dtype has not yet been implemented

        Returns:
            int: Total size (in bytes) of the resulting tensor
        """
        unit_byte_size : int = 0
        match dtype:#python 3.10 or above needed for pattern matching
            #see pytorch docs for possible dtypes https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype
            case torch.float:
                unit_byte_size = 4
            case torch.float64:
                unit_byte_size = 8
            case torch.complex64:
                unit_byte_size = 8
            case torch.complex128:
                unit_byte_size = 16
            case torch.float16:
                unit_byte_size = 2
            case torch.bfloat16:
                unit_byte_size = 2
            case torch.uint8:
                unit_byte_size = 1
            case torch.int8:
                unit_byte_size = 1
            case torch.int16:
                unit_byte_size = 2
            case torch.int32:
                unit_byte_size = 4
            case torch.int64:
                unit_byte_size = 8
            case torch.bool:
                unit_byte_size = 1 #probably overestimates the size of a bool tensor by a lot
            case _:
                raise TypeError("Dev error: Pytorch dtype not yet supported for ")
        return unit_byte_size*expected_data_size





        


    