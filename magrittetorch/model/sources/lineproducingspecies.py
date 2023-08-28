from typing import List, Union, Optional, Any, Tuple
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor, DelayedListOfClassInstances, InferredTensorPerNLTEIter
from magrittetorch.model.parameters import Parameters
from magrittetorch.model.sources.linequadrature import LineQuadrature
from magrittetorch.algorithms.torch_algorithms import multi_arange
import torch
from astropy import units
import astropy.constants
from magrittetorch.model.sources.linedata import Linedata
import numpy as np


#TODO: implement in seperate file

#plan to implement
class ApproxLambda:
    pass

class LineProducingSpecies:
    MAX_DISTANCE_OPACITY_CONTRIBUTION : float = 10.0#TODO: put somewhere else
    def __init__(self, params: Parameters, dataCollection : DataCollection, lineproducingspeciesindex: int) -> None:
        storagedir : str = "lines/lineProducingSpecies_"+str(lineproducingspeciesindex)+"/"#TODO: is legacy io for now; figure out how to switch to new structure
        self.parameters: Parameters = params
        self.dataCollection : DataCollection = dataCollection
        self.linedata : Linedata = Linedata(self.parameters, self.dataCollection, lineproducingspeciesindex)
        self.linequadrature : LineQuadrature = LineQuadrature(params, dataCollection, lineproducingspeciesindex)
        self.approxlambda : ApproxLambda = ApproxLambda()
        self.population_tot : InferredTensor = InferredTensor(Types.LevelPopsInfo, [self.parameters.npoints], units.m**-3, self._infer_population_tot, storagedir+"population_tot"); dataCollection.add_inferred_dataset(self.population_tot, "population_tot_"+str(lineproducingspeciesindex))
        self.population : InferredTensor = InferredTensor(Types.LevelPopsInfo, [self.parameters.npoints, self.linedata.nlev], units.m**-3, self._infer_population, relative_storage_location = storagedir+"population"); dataCollection.add_inferred_dataset(self.population, "population_"+str(lineproducingspeciesindex))
        # self.prev_populations : InferredTensor TODO: err, list/tensor with more dims should suffice
        self.sorted_linewidths: InferredTensorPerNLTEIter = InferredTensorPerNLTEIter(Types.FrequencyInfo, [self.parameters.npoints, self.linedata.nrad], units.Hz, self._infer_sorted_linewidths); self.dataCollection.add_inferred_dataset(self.sorted_linewidths, "sorted linewidths_"+str(lineproducingspeciesindex))
        self.sorted_linefreqs: InferredTensor = InferredTensor(Types.FrequencyInfo, [self.linedata.nrad], units.Hz, self._infer_sorted_linefreqs); self.dataCollection.add_inferred_dataset(self.sorted_linefreqs, "sorted linefreqs_"+str(lineproducingspeciesindex))
        self.relative_line_width: InferredTensor = InferredTensor(Types.FrequencyInfo, [self.parameters.npoints], units.dimensionless_unscaled, self._infer_relative_line_width); self.dataCollection.add_inferred_dataset(self.relative_line_width, "relative line width_"+str(lineproducingspeciesindex))
        # TODO: fix units for line opacities and emissivities
        self.line_opacities: InferredTensorPerNLTEIter = InferredTensorPerNLTEIter(Types.FrequencyInfo, [self.parameters.npoints, self.linedata.nrad], units.radian**-2, self._infer_line_opacities); self.dataCollection.add_inferred_dataset(self.line_opacities, "line opacities_"+str(lineproducingspeciesindex))
        self.line_emissivities: InferredTensorPerNLTEIter = InferredTensorPerNLTEIter(Types.FrequencyInfo, [self.parameters.npoints, self.linedata.nrad], units.radian**-2, self._infer_line_emissivities); self.dataCollection.add_inferred_dataset(self.line_emissivities, "line emissivities_"+str(lineproducingspeciesindex))

    def _infer_sorted_linewidths(self) -> torch.Tensor:
        sorted_data: torch.Tensor
        sorted_data, _ = torch.sort(self.get_line_widths(), dim=1)
        return sorted_data
    
    def _infer_sorted_linefreqs(self) -> torch.Tensor:
        sorted_data: torch.Tensor
        sorted_data, _ = torch.sort(self.linedata.frequency.get())
        return sorted_data
    
    def _infer_relative_line_width(self) -> torch.Tensor:
        return self.get_relative_line_width()

    def _infer_population(self) -> torch.Tensor:
        non_normalized_pops = torch.reshape(self.linedata.weight.get(), (1,-1)) * torch.exp(-torch.reshape(self.linedata.energy.get(), (1,-1)) / (astropy.constants.k_B.value * torch.reshape(self.dataCollection.get_data("gas temperature").get(), (-1,1))))#type: ignore
        return torch.reshape(self.population_tot.get(), (-1,1)) * non_normalized_pops / torch.sum(non_normalized_pops, dim = 1).reshape(-1,1)

    def _infer_population_tot(self) -> torch.Tensor:
        return self.dataCollection.get_data("species abundance").get()[:,self.linedata.num.get()]#type: ignore
    
    def _infer_line_opacities(self) -> torch.Tensor:
        device_pops = self.population.get()
        device_einstein_Ba = self.linedata.Ba.get()
        device_einstein_Bs = self.linedata.Bs.get()
        device_irad = self.linedata.irad.get()
        device_jrad = self.linedata.jrad.get()
        return  astropy.constants.h.value / (4.0 * np.pi) * (device_pops[:, device_jrad] * device_einstein_Ba.reshape(1,-1) - device_pops[:, device_irad] * device_einstein_Bs.reshape(1,-1))#type: ignore

    def _infer_line_emissivities(self) -> torch.Tensor:
        device_pops = self.population.get()
        device_einstein_A = self.linedata.A.get()
        device_irad = self.linedata.irad.get()
        return  astropy.constants.h.value / (4.0 * np.pi) * device_pops[:, device_irad] * device_einstein_A.reshape(1,-1)#type: ignore
        
    
    def get_line_opacities_emissivities(self, device : torch.device=torch.device("cpu")) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the line opacities and emissivities for this species

        Args:
            device (torch.device, optional): Device on which to compute. Defaults to torch.device("cpu").

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of computed opacities and emissivities. Both tensors have dimensions [parameters.npoints, linedata.nrad]
        """
        device_pops = self.population.get(device)
        device_einstein_A = self.linedata.A.get(device)
        device_einstein_Ba = self.linedata.Ba.get(device)
        device_einstein_Bs = self.linedata.Bs.get(device)
        device_irad = self.linedata.irad.get(device)
        device_jrad = self.linedata.jrad.get(device)
        return  (astropy.constants.h.value / (4.0 * np.pi) * (device_pops[:, device_jrad] * device_einstein_Ba.reshape(1,-1) - device_pops[:, device_irad] * device_einstein_Bs.reshape(1,-1)),
                 astropy.constants.h.value / (4.0 * np.pi) * device_pops[:, device_irad] * device_einstein_A.reshape(1,-1))
        #output format: ([parameters.npoints, linedata.nrad])

    
    # def get_line_quadrature_frequencies(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    #     return self.linedata.frequency.get().to(device)

    def get_relative_line_width(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Computes the relative line width per point, using thermodynamics data

        Args:
            device (torch.device, optional): Device on which to compute. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: Relative line widths (unitless). Tensor has dimensions [parameters.npoints]
        """
        return torch.sqrt(self.linedata.inverse_mass.get() 
                          * (astropy.constants.k_B*2.0/astropy.constants.c**2/astropy.constants.u).value*self.dataCollection.get_data("gas temperature").get(device)#type: ignore
                            + self.dataCollection.get_inferred_dataset("vturb2 normalized").get(device))
    
    def get_line_widths(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Computes line widths, using line and thermodynamics data

        Args:
            device (torch.device, optional): Device on which to compute. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: computed line widths (in Hz). Tensor has dimensions [parameters.npoints, linedata.nrad]
        """
        return self.linedata.frequency.get(device)[None, :]*self.get_relative_line_width(device)[:, None]

        # return self.linedata.frequency.get(device).reshape(1,-1)*torch.sqrt(self.linedata.inverse_mass.get()
        #        * (astropy.constants.k_B*2.0/astropy.constants.c**2/astropy.constants.u).value*self.dataCollection.get_data("gas temperature").get(device).reshape(-1,1)#type: ignore
        #          + self.dataCollection.get_inferred_dataset("vturb2 normalized").get(device).reshape(-1,1))
        #output format ([parameters.npoints, linedata.nrad])
        
    def get_line_frequencies(self, device: torch.device = torch.device("cpu")) ->torch.Tensor:
        """Computes all frequencies necessary to compute NLTE radiative transfer

        Args:
            device (torch.device, optional): Device on which to compute and return the result. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: 2D torch.Tensor contain all relevant frequencies per point. Tensor has dimensions [parameters.npoints, linedata.rad*linequadrature.nquads]
        """
        linewidths_device = self.get_line_widths(device)
        #point, line, quadidx
        temp = self.linedata.frequency.get(device)[None, :, None] + self.linequadrature.roots.get(device)[None, None, :] * linewidths_device[:,:, None]
        return temp.flatten(start_dim=1)

    def get_n_lines_freqs(self) -> int:
        """Returns number of line frequencies of this species used for NLTE computations

        Returns:
            int: Amount of line frequencies returned by self.get_line_frequencies
        """
        return self.linequadrature.nquads.get()*self.linedata.nrad.get()
    
    def get_relevant_line_indices(self, opacity_compute_frequencies: torch.Tensor, sorted_linewidths_device: torch.Tensor, sorted_linefreqs_device: torch.Tensor, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes which line should be evaluated when evaluating line opacities/emissivites at given frequencies. TODO: can be made more efficient by inspecting line width

        Args:
            opacity_compute_frequencies (torch.Tensor): Given frequencies for each point. Has dimensions [NPOINTS, NFREQS]
            sorted_linewidths_device (torch.Tensor): Sorted line widths of the lines (sorted for each point). Has dimensions [NPOINTS, self.linedata.nrad]
            sorted_linefreqs_device (torch.Tensor): Sorted line frequencies of the devices (sorted for each point). Has dimensions [self.linedata.nrad]
            device (torch.device, optional): Device on which to compute and return the result. Defaults to torch.device("cpu").

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tensors of respectively the minimum and maximum relevant lines for evaluating opacities/emissivities. Both torch.Tensors have dimensions [parameters.npoints, NFREQS]
        """
        # linefreqs_device = self.linedata.frequency.get().to(device)
        # linewidths_device = self.get_line_widths(device) also subset to actual point indices [corresponding_point_indices, :]
        min_bound_linefreqs = sorted_linefreqs_device[None, :] - self.MAX_DISTANCE_OPACITY_CONTRIBUTION * sorted_linewidths_device
        max_bound_linefreqs = sorted_linefreqs_device[None, :] + self.MAX_DISTANCE_OPACITY_CONTRIBUTION * sorted_linewidths_device

        lower_indices = torch.searchsorted(max_bound_linefreqs, opacity_compute_frequencies, right=False)
        upper_indices = torch.searchsorted(min_bound_linefreqs, opacity_compute_frequencies)
        # print(lower_indices, upper_indices)
        return lower_indices, upper_indices
    
    def evaluate_line_opacities_emissivities(self, frequencies: torch.Tensor, min_line_indices: torch.Tensor, max_line_indices: torch.Tensor, line_opacities: torch.Tensor, line_emissivities: torch.Tensor, sorted_linefreqs_device: torch.Tensor, sorted_linewidths_device: torch.Tensor, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates the line opacities and emissivities using a heuristic method to determine which lines are relevant to the computation.

        Args:
            frequencies (torch.Tensor): Frequencies at which to evaluate the line opacity/emissivity. Has dimensions [NPOINTS, NFREQS]
            min_line_indices (torch.Tensor): minimum index to use of the sorted lines. Has dimensions [NPOINTS, NFREQS]
            max_line_indices (torch.Tensor): maximum index to use of the sorted lines. Has dimensions [NPOINTS, NFREQS]
            line_opacities (torch.Tensor): Integrated line opacities. Has dimensions [NPOINTS, NFREQS]
            line_emissivities (torch.Tensor): Integrated line emissivities. Has dimensions [NPOINTS, NFREQS]
            sorted_linefreqs_device (torch.Tensor): Sorted line frequencies. Has dimensions [self.linedata.nrad]
            sorted_linewidths_device (torch.Tensor): Sorted line widths. Has dimensions [NPOINTS, self.linedata.nrad]
            device (torch.device, optional): Device on which to compute and return the result. Defaults to torch.device("cpu").

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Computed line opacities and emissivities. Both torch.Tensors have dimensions [NPOINTS, NFREQS]
        """
        #due to frequencies having irregular line requirements, quite some reshaping will need to be done.
        #First, we flatten the freqs and line indices; keeping scatter indices for summing the results
        Npoints, Nfreqs = frequencies.size(dim=0), frequencies.size(dim=1)
        delta_indices_flat = (max_line_indices - min_line_indices).flatten() #dimensions = [N_LINE_EVALS = sum(delta_indices)]
        extended_line_indices = multi_arange(min_line_indices.flatten(), delta_indices_flat, device) #dimensions = [N_LINE_EVALS = sum(delta_indices)]
        # extended_frequencies = torch.repeat_interleave(frequencies.flatten(), delta_indices_flat) #dimensions = [N_LINE_EVALS]
        scatter_indices = torch.arange(Nfreqs*Npoints, device=device).repeat_interleave(delta_indices_flat) #dimensions = [N_LINE_EVALS]
        extended_frequencies = frequencies.flatten()[scatter_indices] #dimensions = [N_LINE_EVALS]
        #Now, we evaluate the line profile function
        extended_point_indices = torch.div(scatter_indices, Nfreqs, rounding_mode="floor").type_as(scatter_indices)
        eval_opacities = line_opacities[extended_point_indices, extended_line_indices] #dimensions = [N_LINE_EVALS]
        eval_emissivities = line_emissivities[extended_point_indices, extended_line_indices] #dimensions = [N_LINE_EVALS]
        extended_inv_line_widths = 1.0/sorted_linewidths_device[extended_point_indices, extended_line_indices] #dimensions = [N_LINE_EVALS]
        line_profile_evaluation = sorted_linefreqs_device[extended_line_indices]*extended_inv_line_widths/torch.sqrt(torch.pi*torch.ones(1, device=device))*torch.exp(-torch.pow((sorted_linefreqs_device[extended_line_indices] - extended_frequencies)*extended_inv_line_widths, 2)) #dimensions = [N_LINE_EVALS]
        #and we sum the computed results to get the total opacities/emissivities
        line_opacities = torch.zeros((Npoints*Nfreqs), dtype=Types.FrequencyInfo, device=device).scatter_add(0, scatter_indices, line_profile_evaluation * eval_opacities).reshape(Npoints, Nfreqs)
        line_emissivities = torch.zeros((Npoints*Nfreqs), dtype=Types.FrequencyInfo, device=device).scatter_add(0, scatter_indices, line_profile_evaluation * eval_emissivities).reshape(Npoints, Nfreqs)
        return line_opacities, line_emissivities
    
    def evaluate_line_opacities_emissivites_sum_every_line(self, frequencies: torch.Tensor, line_opacities: torch.Tensor, line_emissivities: torch.Tensor, sorted_linefreqs_device: torch.Tensor, sorted_linewidths_device: torch.Tensor, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates the line opacities and emissivites by bruteforce summing contributions from all lines.

        Args:
            frequencies (torch.Tensor): Frequencies at which to evaluate the line opacity/emmissivity. Has dimensions [NPOINTS, NFREQS]
            line_opacities (torch.Tensor): Integrated line opacities. Has dimensions [NPOINTS, NFREQS]
            line_emissivities (torch.Tensor): Integrated line emissivities. Has dimensions [NPOINTS, NFREQS]
            sorted_linefreqs_device (torch.Tensor): Sorted line frequencies. Has dimensions [self.linedata.nrad]
            sorted_linewidths_device (torch.Tensor): Sorted line widths. Has dimensions [NPOINTS, self.linedata.nrad]
            device (torch.device, optional): Device on which to compute and return the result. Defaults to torch.device("cpu").

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Computed line opacities and emissivities. Both torch.Tensors have dimensions [NPOINTS, NFREQS]
        """
        #by summing over all lines, we can write a more simple computation. performance seems to favor this method prior to 5 lines and the other past 5 lines. TODO: bench on GPU instead
        #dims: [npoints, nfreqs, self.linedata.nrad]
        inverse_line_widths = 1.0/sorted_linewidths_device
        #very narrow line profile
        line_profile_evaluation = sorted_linefreqs_device[None, None, :]*inverse_line_widths[:, None, :]/torch.sqrt(torch.pi*torch.ones(1, device=device))*torch.exp(-torch.pow(inverse_line_widths[:, None, :]*(sorted_linefreqs_device[None, None, :]-frequencies[:, :, None]), 2))
        line_opacities = torch.sum(line_profile_evaluation*line_opacities[:, None, :], dim=2)
        line_emissivities = torch.sum(line_profile_evaluation*line_emissivities[:, None, :], dim=2)
        return line_opacities, line_emissivities
    




