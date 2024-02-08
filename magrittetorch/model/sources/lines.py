from typing import List, Union, Optional, Any, Tuple
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor, DelayedListOfClassInstances
from magrittetorch.model.parameters import Parameters
from magrittetorch.model.sources.lineproducingspecies import LineProducingSpecies
from magrittetorch.model.sources.frequencyevalhelper import FrequencyEvalHelper
from magrittetorch.utils.constants import min_freq_difference
import torch

storagedir : str = "sources/"
MAX_VELOCITY_DIFFERENCE = 0.35#TODO: define param somewhere else

class Lines:
    def __init__(self, params: Parameters, dataCollection : DataCollection) -> None:
        self.parameters: Parameters = params
        self.dataCollection : DataCollection = dataCollection
        self.lineProducingSpecies : DelayedListOfClassInstances[LineProducingSpecies] = DelayedListOfClassInstances(self.parameters.nlspecs, lambda i: LineProducingSpecies(self.parameters, self.dataCollection, i), "lineProducingSpecies"); dataCollection.add_delayed_list(self.lineProducingSpecies)

    def get_total_number_lines(self) -> int:
        """Return the total number of lines in the model

        Returns:
            int: Total number of lines
        """
        return sum([lspec.linedata.nrad.get() for lspec in self.lineProducingSpecies])
    
    def get_total_number_line_frequencies(self) -> int:
        """Return the total number of line frequencies in the model for NLTE radiative transfer

        Returns:
            int: Total number of line frequencies
        """
        return sum([lspec.get_n_lines_freqs() for lspec in self.lineProducingSpecies])

    #DERPRECATED: use pre-computed lines instead to sum over
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
    
    #DEPRECATED: not fully implemented optical depth
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
        # print("sum optical depth", sum_optical_depths)
        return sum_optical_depths
    
    def get_sum_total_optical_depth_using_freqhelper(self, curr_point_indices: torch.Tensor, prev_point_indices: torch.Tensor, original_point_indices: torch.Tensor, freqhelper: FrequencyEvalHelper, curr_shift: torch.Tensor, prev_shift:torch.Tensor, curr_opacity: torch.Tensor, prev_opacity: torch.Tensor, distance_increments: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Computes the total line opacity and emissivity for a segment using a FrequencyEvalHelper for efficiently determining which lines need to be included in the calculation.

        Args:
            curr_point_indices (torch.Tensor): Indices of the current points in the model. Has dimensions [NPOINTS]
            prev_point_indices (torch.Tensor): Indices of the previous points in the model. Has dimensions [NPOINTS]
            original_point_indices (torch.Tensor): Indicies of the starting points in the model. Has dimensions [NPOINTS]
            freqhelper (FrequencyEvalHelper): FrequencyEvalHelper object
            curr_shift (torch.Tensor): Doppler shift at the current point. Has dimensions [NPOINTS]
            prev_shift (torch.Tensor): Doppler shift at the next point. Has dimensions [NPOINTS]
            curr_opacity (torch.Tensor): Opacity at the current point. Has dimensions [NPOINTS, NFREQS]
            prev_opacity (torch.Tensor): Opacity at the previous point. Has dimensions [NPOINTS, NFREQS]
            distance_increments (torch.Tensor): Distance increment. Has dimensions [NPOINTS]
            device (torch.device): Device on which to compute

        Returns:
            torch.Tensor: Total optical depth for the segment. Has dimensions [NPOINTS, NFREQS]
        """
        nfreqs = curr_opacity.size(dim=1)
        npoints = curr_point_indices.size(dim=0)
        sum_optical_depths = torch.zeros((npoints, nfreqs), dtype=Types.FrequencyInfo, device=device)
        for lspec, i in zip(freqhelper.lineProducingSpecies, range(len(freqhelper.lineProducingSpecies))):
            #check doppler shift; if too large, go compute with large variant
            #otherwise, just compute using already computed opacities
            #TODO: define param somewhere else
            small_shift = torch.abs(curr_shift-prev_shift)<MAX_VELOCITY_DIFFERENCE*lspec.relative_line_width.get(device)[curr_point_indices]
            large_shift = torch.logical_not(small_shift)
            #For a small shift, we can just multiply average opacity with distance to obtain the optical depth
            line_optical_depth = ((curr_opacity+prev_opacity)[:, freqhelper.original_frequency_index[i]]*distance_increments[:, None]*0.5)#dims: [NPOINTS, n_eval_freqs]
            #For a large shift, we analytically integrate the line profile function of an averaged line width
            non_shifted_freqs = freqhelper.duplicated_frequencies[i][original_point_indices,:]#dims: [NPOINTS, n_eval_freqs]
            corresponding_lines = freqhelper.corresponding_lines[i]#dims: [n_eval_freqs]
            #err, forgot to add factor sqrt2 to linewidths
            mean_linewidth = 0.5*(lspec.sorted_linewidths.get(device)[curr_point_indices[large_shift], :]+lspec.sorted_linewidths.get(device)[prev_point_indices[large_shift], :])#dims: [sum(large_shift), NLINES]
            dimensionless_curr_freq = (curr_shift[large_shift, None] * non_shifted_freqs[large_shift, :] - lspec.sorted_linefreqs.get(device)[None, corresponding_lines])/mean_linewidth[:, corresponding_lines]#dims: [sum(large_shift), n_eval_freqs]
            dimensionless_prev_freq = (prev_shift[large_shift, None] * non_shifted_freqs[large_shift, :] - lspec.sorted_linefreqs.get(device)[None, corresponding_lines])/mean_linewidth[:, corresponding_lines]
            dimensionless_diff_freq = dimensionless_curr_freq - dimensionless_prev_freq #+ min_freq_difference is not required, as the shift is large
            line_opacities = lspec.line_opacities.get(device)[:, freqhelper.corresponding_lines[i]]#dims: [NPOINTS, n_eval_freqs]
 
            average_line_opacity = 0.5 * non_shifted_freqs[large_shift, :] * (line_opacities[curr_point_indices[large_shift], :] + line_opacities[prev_point_indices[large_shift], :])
            erfterm = 0.5 * average_line_opacity / dimensionless_diff_freq * (torch.erf(dimensionless_curr_freq) - torch.erf(dimensionless_prev_freq))
            line_optical_depth[large_shift, :] = distance_increments[large_shift, None] / mean_linewidth[:, corresponding_lines] * erfterm

            n_eval_freqs = freqhelper.corresponding_lines[i].size(dim=0)
            expanded_freq_index = freqhelper.original_frequency_index[i].expand(npoints, n_eval_freqs)
            sum_optical_depths += sum_optical_depths.scatter_add(1, expanded_freq_index, line_optical_depth)

        return sum_optical_depths
        

    
    def get_all_line_frequencies(self, device: torch.device) -> torch.Tensor:
        """Return all frequencies (in comoving frame) required for NLTE radiative transfer

        Args:
            device (torch.device, optional): Device on which to compute and return the line frequencies. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: All NLTE line frequencies, ordered per line. Has dimensions [parameters.npoints, self.get_total_number_line_frequencies()]
        """
        line_frequencies = torch.empty((self.parameters.npoints.get(), self.get_total_number_line_frequencies()), dtype=Types.FrequencyInfo, device=device)
        start_freq_idx = 0
        for lspec in self.lineProducingSpecies:
            nfreqs: int = lspec.get_n_lines_freqs()
            line_frequencies[:, start_freq_idx:start_freq_idx+nfreqs] = lspec.get_line_frequencies(device=device)
            start_freq_idx+=nfreqs
        return line_frequencies


    # @torch.compile
    def get_sum_line_opacities_emissivities_using_freqevalhelper(self, origin_point_indices: torch.Tensor, curr_point_indices: torch.Tensor, doppler_shift: torch.Tensor, freqEvalHelper: FrequencyEvalHelper) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the total opacity and emissivity contributions of all lines.

        Args:
            origin_point_indices (torch.Tensor): Point indices for the freqEvalHelper at which to evaluate the comoving frame frequency. Has dimensions [NPOINTS]
            curr_point_indices (torch.Tensor): Point indices at which to evaluate the total line opacity/emissivity. Has dimensions [NPOINTS]
            doppler_shift (torch.Tensor): Doppler shift at the current point. Has dimensions [NPOINTS]
            freqEvalHelper (FrequencyEvalHelper): FrequencyEvalHelper object

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Computed line opacities and emissivities. Both torch.Tensors have dimensions [NPOINTS, NFREQS]
        """
        #TODO?: just take tensor and add to those tensors?; err, only useful if multiple types of sources exist
        device = freqEvalHelper.device
        npoints = origin_point_indices.size(dim=0)
        nfreqs = freqEvalHelper.original_frequencies.size(dim=1)
        sum_opacities, sum_emissivities = torch.zeros((npoints, nfreqs), dtype=Types.FrequencyInfo, device=device), torch.zeros((npoints, nfreqs), dtype=Types.FrequencyInfo, device=device)
        #maybe TODO?: vectorize computation over species; then less search overhead
        for lspec, i in zip(freqEvalHelper.lineProducingSpecies, range(len(freqEvalHelper.lineProducingSpecies))):
            #TODO: add memory management; maybe just add in higher level, as the frequencies vector can easily be split
            opacities = lspec.line_opacities.get(device)
            emissivities = lspec.line_emissivities.get(device)
            sorted_linefreqs = lspec.sorted_linefreqs.get(device)
            sorted_linewidths = lspec.sorted_linewidths.get(device)
            frequencies_to_eval = freqEvalHelper.duplicated_frequencies[i][origin_point_indices, :] * doppler_shift[:, None]
            line_opacities, line_emissivities = lspec.evaluate_line_opacities_emissivites_single_line(frequencies_to_eval, freqEvalHelper.corresponding_lines[i], opacities[curr_point_indices], emissivities[curr_point_indices], sorted_linefreqs, sorted_linewidths[curr_point_indices, :], device)
            n_eval_freqs = freqEvalHelper.corresponding_lines[i].size(dim=0)
            expanded_freq_index = freqEvalHelper.original_frequency_index[i].expand(npoints, n_eval_freqs)
            sum_opacities = sum_opacities.scatter_add(1, expanded_freq_index, line_opacities)
            sum_emissivities = sum_emissivities.scatter_add(1, expanded_freq_index, line_emissivities)

        return sum_opacities, sum_emissivities