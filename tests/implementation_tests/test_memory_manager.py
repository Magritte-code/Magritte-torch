#just some simple tests to check whether the automatic memory manager works correctly
from magrittetorch.utils.storagetypes import DataCollection
import torch
import psutil
from magrittetorch.utils.memorymapping import MemoryManager
cpu_mem : int = psutil.virtual_memory().total
cuda_mem : int = torch.cuda.get_device_properties(0).total_memory

print(cpu_mem)

test_mem = int(15.5 * cpu_mem)
test_mem = int(10.1 * cuda_mem)

test_manager = MemoryManager()

print(test_manager.generate_splits(20, cpu_mem, test_mem))
print(test_manager.split_data(200, test_mem))