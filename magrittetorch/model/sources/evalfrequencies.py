from typing import List, Union, Optional, Any
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor, DelayedListOfClassInstances, StorageNdarray
from magrittetorch.model.parameters import Parameters, Parameter
from magrittetorch.utils.io import LegacyHelper
from magrittetorch.model.sources.lines import Lines
import torch
from astropy import units

class EvalFrequencies:
    #Helper class for evaluating of frequencies, keeping track of which lines need to be evaluated.
    def __init__(self, frequencies: torch.Tensor, lines: Lines) -> None:
        self.original_frequencies: torch.Tensor = frequencies#err, we do not actually need to keep the original frequencies

        self.corresponding_lines: torch.Tensor = TODO

        pass