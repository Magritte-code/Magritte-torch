from typing import List, Union, Optional, Tuple
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor
from magrittetorch.model.parameters import Parameters
from magrittetorch.model.sources.lines import Lines
from magrittetorch.model.sources.frequencyevalhelper import FrequencyEvalHelper
import torch
from astropy import units

storagedir : str = "sources/"

#TODO: implement in seperate file
class Continuum:
    pass

#plan to implement
class Scattering:
    pass

class Sources:
    def __init__(self, params: Parameters, dataCollection : DataCollection) -> None:
        self.parameters: Parameters = params
        self.dataCollection : DataCollection = dataCollection
        self.lines = Lines(self.parameters, dataCollection)
        self.continuum : Continuum = Continuum()

    #Deprecated: use get_total_opacity_emissivity_freqhelper instead
    def get_total_opacity_emissivity(self, point_indices: torch.Tensor, frequencies: torch.Tensor, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deprecated
        Computes the total opacity and emissivity contributions of all sources. Currently, only line contributions are implemented.

        Args:
            point_indices (torch.Tensor): Point indices at which to evaluate the total opacity/emissivity. Has dimensions [NPOINTS]
            frequencies (torch.Tensor): Frequencies at which to evaluate the total opacity/emissivity. Has dimensions [NPOINTS, NFREQS]
            device (torch.device, optional): Device on which to compute and return the result. Defaults to torch.device("cpu").

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Computed line opacities and emissivities. Both torch.Tensors have dimensions [NPOINTS, NFREQS]
        """
        #TODO: when adding continuum or scattering, add them here
        return self.lines.get_sum_line_opacities_emissivities(point_indices, frequencies, device)

    def get_total_opacity_emissivity_freqhelper(self, origin_point_indices: torch.Tensor, curr_point_indices: torch.Tensor, shift: torch.Tensor, freqhelper: FrequencyEvalHelper, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the total opacity and emissivity contributions of all sources. Currently, only line contributions are implemented.
        Dev note: to add other contributions, create a subclass of Sources and override this method.

        Args:
            origin_point_indices (torch.Tensor): Point indices for the freqEvalHelper at which to evaluate the comoving frame frequency. Has dimensions [NPOINTS]
            curr_point_indices (torch.Tensor): Point indices at which to evaluate the total opacity/emissivity. Has dimensions [NPOINTS]
            shift (torch.Tensor): Doppler shift at the current point. Has dimensions [NPOINTS]
            freqhelper (FrequencyEvalHelper): FrequencyEvalHelper object
            device (torch.device): device on which to compute

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Computed line opacities and emissivities. Both torch.Tensors have dimensions [NPOINTS, NFREQS]
        """
        return self.lines.get_sum_line_opacities_emissivities_using_freqevalhelper(origin_point_indices, curr_point_indices, shift, freqhelper)

    #Deprecated: use get_total_optical_depth_freqhelper instead
    def get_total_optical_depth(self, point_indices: torch.Tensor, frequencies: torch.Tensor, prev_frequencies: torch.Tensor, distances: torch.Tensor, curr_shift: torch.Tensor, prev_shift: torch.Tensor, curr_opacity: torch.Tensor, prev_opacity: torch.Tensor, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Deprecated
        Computes the total optical depth in an interval by adding all sources. Currently, only line contributions are implemented.

        Args:
            point_indices (torch.Tensor): Indices of the points in the model. Has dimensions [NPOINTS]
            frequencies (torch.Tensor): _description_
            prev_frequencies (torch.Tensor): _description_
            distances (torch.Tensor): _description_
            curr_shift (torch.Tensor): _description_
            prev_shift (torch.Tensor): _description_
            curr_opacity (torch.Tensor): _description_
            prev_opacity (torch.Tensor): _description_
            device (torch.device, optional): _description_. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: _description_
        """
        #TODO: when adding continumm or scattering, add them here
        return self.lines.get_sum_line_optical_depths(point_indices, frequencies, prev_frequencies, distances, curr_shift, prev_shift, curr_opacity, prev_opacity, device)
    
    def get_total_optical_depth_freqhelper(self, point_indices: torch.Tensor, prev_point_indices: torch.Tensor, original_point_indices: torch.Tensor, freqhelper: FrequencyEvalHelper, distances: torch.Tensor, curr_shift: torch.Tensor, prev_shift: torch.Tensor, curr_opacity: torch.Tensor, prev_opacity: torch.Tensor, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Computes the total optical depth in an interval by adding all sources. Currently, only line contributions are implemented.
        Dev note: to add other contributions, create a subclass of Sources and override this method.

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
        #TODO: when adding continumm or scattering, add them here

        return self.lines.get_sum_total_optical_depth_using_freqhelper(point_indices, prev_point_indices, original_point_indices, freqhelper, curr_shift, prev_shift, curr_opacity, prev_opacity, distances, device)
        