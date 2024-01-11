from typing import List, Union, Optional, Any
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor, DelayedListOfClassInstances, StorageNdarray
from magrittetorch.model.parameters import Parameters, Parameter
from magrittetorch.utils.io import LegacyHelper
from magrittetorch.model.sources.lineproducingspecies import LineProducingSpecies
from magrittetorch.algorithms.torch_algorithms import multi_arange
import torch
from astropy import units
from astropy.constants import c

class FrequencyEvalHelper:
    """Lines only act on narrow frequency ranges. This structure helps figuring out which lines are needed per frequency, using a heuristic approach.
    """

    #Helper class for evaluating of frequencies, keeping track of which lines need to be evaluated.
    def __init__(self, frequencies: torch.Tensor, listlspec: List[LineProducingSpecies], model_velocities: torch.Tensor, device: torch.device) -> None:
        """Initialize the FrequencyEvalHelper

        Args:
            frequencies (torch.Tensor): Non-shifted frequencies to use for all positions. Has dimensions [parameters.npoints, NFREQS]
            listlspec (List[LineProducingSpecies]): List of LineProducingSpecies to use. Has dimensions [NLSPECS]
            model_velocities (torch.Tensor): Velocities of the model, used to infer the maximal doppler shift. Has dimensions [parameters.npoints, 3]
            device (torch.device): Device on which to compute.
        """
        self.lineProducingSpecies = listlspec
        self.original_frequencies = frequencies
        self.duplicated_frequencies: List[torch.Tensor] = [] #frequencies to use per LineProducingSpecies #dims for each: [parameters.npoints, probably>=NFREQS]
        self.original_frequency_index: List[torch.Tensor] = [] # per LineProducingSpecies #dims for each: [probably>=NFREQS]
        self.corresponding_lines: List[torch.Tensor] = [] # per LineProducingSpecies #dims for each: [probably>=NFREQS]
        self.deduce_close_lines(frequencies, model_velocities, device)
        self.device = device

    def deduce_close_lines(self, base_frequencies: torch.Tensor, model_velocities: torch.Tensor, device: torch.device) -> None:
        """Infers which lines lie close enough to each frequency and sets the internal data accordingly.

        Args:
            base_frequencies (torch.Tensor): Non-shifted frequencies to use for all positions. Has dimensions [NPOINTS, NFREQS]
            model_velocities (torch.Tensor): Velocities of the model, used to infer the maximal doppler shift. Has dimensions [parameters.npoints, 3]
            device (torch.device): Device on which to compute this.
        """ 
        nfreqs = base_frequencies.size(dim=1)
        max_shift = torch.sqrt(torch.max(torch.sum(torch.pow(model_velocities, 2.0), dim=1)))/c.value
        for lspec in self.lineProducingSpecies:
            minids, maxids = lspec.get_global_bound_relevant_line_indices(base_frequencies[0, :], max_shift, device)
            self.duplicated_frequencies.append(base_frequencies.repeat_interleave(maxids-minids, dim=1))
            self.original_frequency_index.append(torch.arange(nfreqs, device=device).repeat_interleave(maxids-minids))
            # If no lines are found, the multi_arange will crash. As that snippet of code should be optimized (so no adding if-clauses), it is better to create a workaround here.
            # We just manually set the corresponding lines to empty in that case.
            if torch.all(minids == maxids):
                self.corresponding_lines.append(torch.empty((0), device=device, dtype=Types.IndexInfo))
            else:
                self.corresponding_lines.append(multi_arange(minids, maxids-minids, device))

        