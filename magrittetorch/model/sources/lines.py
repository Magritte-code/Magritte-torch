from typing import List, Union, Optional, Any, Tuple
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor, DelayedListOfClassInstances
from magrittetorch.model.parameters import Parameters
from magrittetorch.model.sources.lineproducingspecies import LineProducingSpecies
import torch
from astropy import units

storagedir : str = "sources/"


class Lines:
    def __init__(self, params: Parameters, dataCollection : DataCollection) -> None:
        self.parameters: Parameters = params
        self.dataCollection : DataCollection = dataCollection
        self.lineProducingSpecies : DelayedListOfClassInstances[LineProducingSpecies] = DelayedListOfClassInstances(self.parameters.nlspecs, lambda i: LineProducingSpecies(self.parameters, self.dataCollection, i), "lineProducingSpecies"); dataCollection.add_delayed_list(self.lineProducingSpecies)

    def get_total_number_lines(self) -> int:
        return sum([lspec.linedata.nrad.get() for lspec in self.lineProducingSpecies])
    
    def get_total_number_line_frequencies(self) -> int:
        return sum([lspec.get_n_lines_freqs() for lspec in self.lineProducingSpecies])

    def get_sum_line_opacities_emissivities(self, point_indices: torch.Tensor, frequencies: torch.Tensor, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the total opacity and emissivity contributions of all lines.

        Args:
            point_indices (torch.Tensor): Point indices at which to evaluate the total line opacity/emissivity. Has dimensions [NPOINTS, NFREQS]
            frequencies (torch.Tensor): Frequencies at which to evaluate the line opacity/emissivity in comoving frame at the point_indices. Has dimensions [NPOINTS, NFREQS]
            device (torch.device, optional): Device on which to compute and return the result. Defaults to torch.device("cpu").

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Computed line opacities and emissivities. Both torch.Tensors have dimensions [NPOINTS, NFREQS]
        """
        #TODO?: just take tensor and add to those tensors?; err, only useful if multiple types of sources exist
        sum_opacities, sum_emissivities = torch.zeros_like(frequencies, dtype=Types.FrequencyInfo, device=device), torch.zeros_like(frequencies, dtype=Types.FrequencyInfo, device=device)
        #maybe TODO?: vectorize computation over species; then less search overhead
        for lspec in self.lineProducingSpecies:
            #TODO: add memory management; maybe just add in higher level, as the frequencies vector can easily be split
            opacities = lspec.line_opacities.get(device)
            emissivities = lspec.line_emissivities.get(device)
            sorted_linefreqs = lspec.sorted_linefreqs.get(device)
            sorted_linewidths = lspec.sorted_linewidths.get(device)
            if lspec.linedata.nrad.get()<5:#for very low amounts of lines, the heuristic method is more expensive
                line_opacities, line_emissivities = lspec.evaluate_line_opacities_emissivites_sum_every_line(frequencies, opacities[point_indices], emissivities[point_indices], sorted_linefreqs, sorted_linewidths[point_indices, :], device)
                sum_opacities += line_opacities
                sum_emissivities += line_emissivities
            else:
                # eval_op, eval_em = old_model.sources.lines.lineProducingSpecies[0].evaluate_line_opacities_emissivites_sum_every_line(freqs, opacities, emissivites, linefreqs, linewidths)
                min_index, max_index = lspec.get_relevant_line_indices(frequencies, sorted_linewidths[point_indices, :], sorted_linefreqs)
                line_opacities, line_emissivities = lspec.evaluate_line_opacities_emissivities(frequencies, min_index, max_index, opacities[point_indices], emissivities[point_indices], sorted_linefreqs, sorted_linewidths[point_indices, :], device=device)
                sum_opacities += line_opacities
                sum_emissivities += line_emissivities
        return sum_opacities, sum_emissivities
    
    def get_sum_line_optical_depths(self, point_indices: torch.Tensor, curr_frequencies: torch.Tensor, prev_frequencies: torch.Tensor, distance_increments: torch.Tensor, curr_shift, prev_shift, curr_opacity: torch.Tensor, prev_opacity: torch.Tensor, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        sum_optical_depths = torch.zeros_like(curr_frequencies, dtype=Types.FrequencyInfo, device=device)
        # dummy = torch.zeros_like(curr_frequencies, dtype=Types.FrequencyInfo, device=device)
        # nfreqs = sum_optical_depths.size(dim=1)
        for lspec in self.lineProducingSpecies:
            #check doppler shift; if too large, go compute with large variant
            #otherwise, just compute using already computed opacities
            small_shift = torch.abs(curr_shift-prev_shift)<0.1*lspec.relative_line_width.get(device)[point_indices]
            # simple timings test: evidently adding random for loops lead to terrible performance
            # for quad in range(nfreqs):
            #     sum_optical_depths[small_shift, quad] += ((curr_opacity+prev_opacity)[:, quad]*distance_increments*0.5)[small_shift]
            sum_optical_depths[small_shift, :] += ((curr_opacity+prev_opacity)*distance_increments[:, None]*0.5)[small_shift, :]
            # dummy.scatter_add(1, torch.arange(nfreqs, dtype=torch.int64, device=device)[None,:], sum_optical_depths)
            #TODO: either already implement harder to compute optical depth here; or wait until heuristic 
            #TODO: or only implement sum all version
            # if lspec.linedata.nrad.get()<5:
            #     sum_optical_depths[large shift, :] += 
        print(sum_optical_depths)
        return sum_optical_depths

    
    def get_all_line_frequencies(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Return all frequencies (in comoving frame) required for NLTE radiative transfer

        Args:
            device (torch.device, optional): Device on which to compute and return the line frequencies. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: All NLTE line frequencies. Has dimensions [parameters.npoints, self.get_total_number_line_frequencies()]
        """
        line_frequencies = torch.empty((self.parameters.npoints.get(), self.get_total_number_line_frequencies()), dtype=Types.FrequencyInfo, device=device)
        start_freq_idx = 0
        for lspec in self.lineProducingSpecies:
            nfreqs: int = lspec.get_n_lines_freqs()
            line_frequencies[:, start_freq_idx:start_freq_idx+nfreqs] = lspec.get_line_frequencies(device=device)
            start_freq_idx+=nfreqs
        return line_frequencies