from typing import List, Union, Optional, Any, Tuple
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor, DelayedListOfClassInstances, InferredTensorPerNLTEIter
from magrittetorch.algorithms.torch_algorithms import interpolate2D_linear
from magrittetorch.model.parameters import Parameters
from magrittetorch.utils.logger import Logger, Level
from magrittetorch.model.sources.linequadrature import LineQuadrature
from magrittetorch.algorithms.torch_algorithms import multi_arange
from magrittetorch.utils.constants import min_level_pop, population_inversion_fraction, min_line_opacity
import torch
from astropy import units
import astropy.constants
from magrittetorch.model.sources.linedata import Linedata
import numpy as np


class LineProducingSpecies:

    def __init__(self, params: Parameters, dataCollection : DataCollection, lineproducingspeciesindex: int) -> None:
        storagedir : str = "lines/lineProducingSpecies_"+str(lineproducingspeciesindex)+"/"#TODO: is legacy io for now; figure out how to switch to new structure
        self.parameters: Parameters = params
        self.dataCollection : DataCollection = dataCollection
        self.linedata : Linedata = Linedata(self.parameters, self.dataCollection, lineproducingspeciesindex)#: Line data for this species
        self.linequadrature : LineQuadrature = LineQuadrature(params, dataCollection, lineproducingspeciesindex)#: Line quadrature for this species
        self.population_tot : InferredTensor = InferredTensor(Types.LevelPopsInfo, [self.parameters.npoints], units.m**-3, self._infer_population_tot, storagedir+"population_tot")#: Number density of this species; dtype = :attr:`.Types.LevelPopsInfo`, dims = [:py:attr:`.Parameters.npoints`], units = units.m**-3
        dataCollection.add_inferred_dataset(self.population_tot, "population_tot_"+str(lineproducingspeciesindex))
        self.population : InferredTensor = InferredTensor(Types.LevelPopsInfo, [self.parameters.npoints, self.linedata.nlev], units.m**-3, self._infer_population, relative_storage_location = storagedir+"population")#: Level populations of this species; dtype = :attr:`.Types.LevelPopsInfo`, dims = [:py:attr:`.Parameters.npoints`, :py:attr:`.Linedata.nlev`], units = units.m**-3
        dataCollection.add_inferred_dataset(self.population, "population_"+str(lineproducingspeciesindex))
        self.sorted_linewidths: InferredTensorPerNLTEIter = InferredTensorPerNLTEIter(Types.FrequencyInfo, [self.parameters.npoints, self.linedata.nrad], units.Hz, self._infer_sorted_linewidths)#: Widths of the lines, sorted in according to the line frequencies; dtype = :attr:`.Types.FrequencyInfo`, dims = [:py:attr:`.Parameters.npoints`, :py:attr:`.Linedata.nrad`], units = units.Hz
        self.dataCollection.add_inferred_dataset(self.sorted_linewidths, "sorted linewidths_"+str(lineproducingspeciesindex))
        self.sorted_linefreqs: InferredTensor = InferredTensor(Types.FrequencyInfo, [self.linedata.nrad], units.Hz, self._infer_sorted_linefreqs)#: Frequencies of the lines, sorted in ascending order; dtype = :attr:`.Types.FrequencyInfo`, dims = [:py:attr:`.Linedata.nrad`], units = units.Hz
        self.dataCollection.add_inferred_dataset(self.sorted_linefreqs, "sorted linefreqs_"+str(lineproducingspeciesindex))
        self.relative_line_width: InferredTensor = InferredTensor(Types.FrequencyInfo, [self.parameters.npoints], units.dimensionless_unscaled, self._infer_relative_line_width)#: Relative line width per point; dtype = :attr:`.Types.FrequencyInfo`, dims = [:py:attr:`.Parameters.npoints`], units = units.dimensionless_unscaled
        self.dataCollection.add_inferred_dataset(self.relative_line_width, "relative line width_"+str(lineproducingspeciesindex))
        self.line_opacities: InferredTensorPerNLTEIter = InferredTensorPerNLTEIter(Types.FrequencyInfo, [self.parameters.npoints, self.linedata.nrad], units.Hz*units.m**-1, self._infer_line_opacities)#: Line opacities for this species, for all positions and all line transitions; dtype = :attr:`.Types.FrequencyInfo`, dims = [:py:attr:`.Parameters.npoints`, :py:attr:`.Linedata.nrad`], units = units.Hz*units.m**-1
        self.dataCollection.add_inferred_dataset(self.line_opacities, "line opacities_"+str(lineproducingspeciesindex))
        self.line_emissivities: InferredTensorPerNLTEIter = InferredTensorPerNLTEIter(Types.FrequencyInfo, [self.parameters.npoints, self.linedata.nrad], units.W*units.m**-3*units.sr**-1, self._infer_line_emissivities)# Line emissivities for this species, for all positions and all line transitions; dtype = :attr:`.Types.FrequencyInfo`, dims = [:py:attr:`.Parameters.npoints`, :py:attr:`.Linedata.nrad`], units = units.W*units.m**-3*units.sr**-1
        self.dataCollection.add_inferred_dataset(self.line_emissivities, "line emissivities_"+str(lineproducingspeciesindex))

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
        non_normalized_pops = min_level_pop + torch.reshape(self.linedata.weight.get(), (1,-1)) * torch.exp(-torch.reshape(self.linedata.energy.get(), (1,-1)) / (astropy.constants.k_B.value * torch.reshape(self.dataCollection.get_data("gas temperature").get(), (-1,1))))#type: ignore
        return torch.reshape(self.population_tot.get(), (-1,1)) * non_normalized_pops / torch.sum(non_normalized_pops, dim = 1).reshape(-1,1)


    def _infer_population_tot(self) -> torch.Tensor:
        return self.dataCollection.get_data("species abundance").get()[:,self.linedata.num.get()]#type: ignore
    
    def _infer_line_opacities(self) -> torch.Tensor:
        device_pops = self.population.get()
        device_einstein_Ba = self.linedata.Ba.get()
        device_einstein_Bs = self.linedata.Bs.get()
        device_irad = self.linedata.irad.get()
        device_jrad = self.linedata.jrad.get()
        return torch.maximum(astropy.constants.h.value / (4.0 * np.pi) * (device_pops[:, device_jrad] * device_einstein_Ba.reshape(1,-1) - device_pops[:, device_irad] * device_einstein_Bs.reshape(1,-1)), min_line_opacity*torch.ones((self.parameters.npoints.get(), self.linedata.nrad.get()))) #type: ignore
        # return  astropy.constants.h.value / (4.0 * np.pi) * (device_pops[:, device_jrad] * device_einstein_Ba.reshape(1,-1) - device_pops[:, device_irad] * device_einstein_Bs.reshape(1,-1))#type: ignore

    def _infer_line_emissivities(self) -> torch.Tensor:
        device_pops = self.population.get()
        device_einstein_A = self.linedata.A.get()
        device_irad = self.linedata.irad.get()
        return  astropy.constants.h.value / (4.0 * np.pi) * device_pops[:, device_irad] * device_einstein_A.reshape(1,-1)#type: ignore
    
    def correct_population(self, new_population: torch.Tensor, device: torch.device) -> torch.Tensor:
        """First checks at which lines population inversions occur, and adjusts the population accordingly, setting the offending level populations to LTE.

        Args:
            new_population (torch.Tensor): New population to set. Has dimensions [parameters.npoints, linedata.nlev]
            device (torch.device): Device on which to compute the correction
        
        Returns:
            torch.Tensor: The corrected population
        """

        return new_population#TEST: not going to correct levelpops, instead changing the line opacity

        # device_pops = new_population
        # device_einstein_Ba = self.linedata.Ba.get(device)
        # device_einstein_Bs = self.linedata.Bs.get(device)
        # device_irad = self.linedata.irad.get(device)
        # device_jrad = self.linedata.jrad.get(device)
        # device_pop_tot = self.population_tot.get(device)
        # npoints = self.parameters.npoints.get()
        # nlev = self.linedata.nlev.get()
        # gas_temperature = self.dataCollection.get_data("gas temperature").get(device).unsqueeze(1).expand(-1, nlev)#type: ignore #dims: [parameters.npoints, linedata.nlev]
        # lev_weight = self.linedata.weight.get(device).expand(npoints, -1)#type: ignore
        # lev_energy = self.linedata.energy.get(device).expand(npoints, -1)#type: ignore
        # masering_levels = torch.zeros((npoints, nlev), dtype=Types.Bool, device=device)
        # prev_n_masering_lines = 0

        # #Attempt 2: put al levels of the masering points to LTE, but fit the temperature to the level populations
        # masering_lines = (device_pops[:, device_jrad] * device_einstein_Ba.reshape(1,-1) - population_inversion_fraction * device_pops[:, device_irad] * device_einstein_Bs.reshape(1,-1)) <= 0.0#dims: [parameters.npoints, linedata.nrad]
        # masering_levels = masering_levels.scatter_add(1, device_irad.unsqueeze(0).expand(npoints, -1), masering_lines)
        # masering_levels = masering_levels.scatter_add(1, device_jrad.unsqueeze(0).expand(npoints, -1), masering_lines)
        # masering_positions = torch.any(masering_lines, dim=1)#dims: [parameters.npoints]
        # print("number of masering positions", torch.sum(masering_positions))
        # # print(device_pops[masering_positions, :].shape, device_pop_tot[masering_positions].shape)
        # # print(lev_weight[masering_positions, :].shape, lev_energy[masering_positions].shape, gas_temperature[masering_positions].shape)

        # #Now go fit the following values linearly: ln(rel_pop/w) = -E/k * 1/T + C
        # yvals = torch.log(device_pops[masering_positions, :]/device_pop_tot[masering_positions, None]/self.linedata.weight.get(device)[None, :])#dims: [nmasering_positions, linedata.nlev]
        # #And assume that the energy of the base level is 0, such that we do not need to fit the constant C
        # #FIXME: either check that the base level is the first level, or do a least squares fit with two unknowns
        # # yvals = yvals - yvals[:, 0].unsqueeze(1).expand(-1, nlev)#type: ignore
        # # xvals = -lev_energy[masering_positions, :]/(astropy.constants.k_B.value)#dims: [nmasering_positions, linedata.nlev]
        # xvals = -(self.linedata.energy.get(device)/astropy.constants.k_B.value)[None, :]#dims: [1, linedata.nlev]
        # #extend xvals with a column of ones to fit the constant C
        # xvals_extended = torch.cat((xvals, torch.ones_like(xvals)), dim=0).T#dims: [nmasering_positions, 2]
        # # print(xvals_extended, xvals_extended.shape)
        # #fit the temperature
        # inv_temp = torch.linalg.solve(xvals_extended.T @ xvals_extended, xvals_extended.T @ yvals.T)[0, :]#dims: [nmasering_positions]
        # # inv_temp = torch.sum(yvals*xvals, dim=1)/torch.sum(xvals*xvals, dim=1)#dims: [nmasering_positions]
        # # print("inv_temp", inv_temp)

        # #now create new populations from another blackbody distribution
        # new_population[masering_positions, :] = lev_weight[masering_positions, :] * torch.exp(-lev_energy[masering_positions, :] / astropy.constants.k_B.value * inv_temp.unsqueeze(1).expand(-1, nlev)) + min_level_pop#type: ignore
        # new_population[masering_positions, :] = new_population[masering_positions, :] / torch.sum(new_population[masering_positions, :], dim=1)[:, None] * device_pop_tot[masering_positions, None]#type: ignore
        # #or just only apply this to the masering levels; note: no conservation of total population is applied
        # # new_pops = lev_weight[masering_positions, :] * torch.exp(-lev_energy[masering_positions, :] / astropy.constants.k_B.value * inv_temp.unsqueeze(1).expand(-1, nlev)) + min_level_pop#type: ignore
        # # new_pops = new_pops / torch.sum(new_pops, dim=1)[:, None] * device_pop_tot[masering_positions, None]#type: ignore
        # # new_population[masering_levels] = new_pops[masering_levels[masering_positions, :]]

        # if torch.any(masering_positions):
        #     Logger().log("Negative line opacities encountered. Corresponding positions have been set to LTE.", Level.WARNING)


        # # #TODO: it might possible to optimize this further, by only considering the points at which masers occur
        # # while True:
        # #     #determine the masering lines by computing the line opacity prefactors, which are not necessary for the comparison
        # #     #Note: near to masering lines will also be set to LTE
        # #     masering_lines = (device_pops[:, device_jrad] * device_einstein_Ba.reshape(1,-1) - population_inversion_fraction * device_pops[:, device_irad] * device_einstein_Bs.reshape(1,-1)) <= 0.0#dims: [parameters.npoints, linedata.nrad]
        # #     #current iter masering lines
        # #     #Amount of masering levels will only increase
        # #     #TODO: check whether the simultaneous assignment contains errors
        # #     masering_levels = masering_levels.scatter_add(1, device_irad.unsqueeze(0).expand(npoints, -1), masering_lines)
        # #     masering_levels = masering_levels.scatter_add(1, device_jrad.unsqueeze(0).expand(npoints, -1), masering_lines)
        # #     # masering_levels[:, device_irad] = torch.logical_or(masering_levels[:, device_irad], masering_lines)
        # #     # masering_levels[:, device_jrad] = torch.logical_or(masering_levels[:, device_jrad], masering_lines)
        # #     curr_n_masering_levels = torch.sum(masering_levels)

        # #     sum_masering_pop_before_adjust = torch.sum(new_population * masering_levels, dim=1)#dims: [parameters.npoints]
        # #     #for every point, set the masering lines simultaneously to LTE
        # #     new_population[masering_levels] = lev_weight[masering_levels] * torch.exp(-lev_energy[masering_levels] / (astropy.constants.k_B.value * gas_temperature[masering_levels])) + min_level_pop#type: ignore

        # #     sum_masering_pop_after_adjust = torch.sum(new_population * masering_levels, dim=1)#dims: [parameters.npoints]
        # #     #and renormalize them
        # #     new_population[masering_levels] *= (((sum_masering_pop_before_adjust / sum_masering_pop_after_adjust)).unsqueeze(1).repeat(1, nlev))[masering_levels]

        # #     if curr_n_masering_levels == prev_n_masering_lines:
        # #         break
        # #     prev_n_masering_lines = curr_n_masering_levels

        # # if prev_n_masering_lines>0:
        # #     Logger().log("Negative line opacities encountered. Corresponding levels have been set to LTE.", Level.WARNING)
        
        # return new_population

    #TODO: add warning to this function in 
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

    def get_line_frequencies(self, device: torch.device = torch.device("cpu")) ->torch.Tensor:
        """Computes all frequencies necessary to compute NLTE radiative transfer

        Args:
            device (torch.device, optional): Device on which to compute and return the result. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: 2D torch.Tensor contain all relevant frequencies per point. Tensor has dimensions [parameters.npoints, linedata.nrad*linequadrature.nquads]
        """
        linewidths_device = self.get_line_widths(device)
        #dims: [parameters.npoints, linedata.nrad, linequadrature.nquads]
        temp = self.linedata.frequency.get(device)[None, :, None] + self.linequadrature.roots.get(device)[None, None, :] * linewidths_device[:,:, None]
        return temp.flatten(start_dim=1)

    def get_line_weights(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Computes the weights of the line quadrature, corresponding to the line frequencies returned by self.get_line_frequencies()

        Args:
            device (torch.device, optional): Device on which to compute and return the result. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: Weights of the line quadrature, indexed corresponding to the frequencies from self.get_line_frequencies(). Tensor has dimensions [linequadrature.nquads*linedata.nrad]
        """
        return self.linequadrature.weights.get(device).repeat(self.linedata.nrad.get())

    def get_n_lines_freqs(self) -> int:
        """Returns number of line frequencies of this species used for NLTE computations

        Returns:
            int: Amount of line frequencies returned by self.get_line_frequencies
        """
        return self.linequadrature.nquads.get()*self.linedata.nrad.get()

    def get_line_indices_NTLE(self, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Returns the line indices for the NLTE computation

        Args:
            device (torch.device, optional): Device on which to compute and return the result. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: Line indices for the NLTE computation. Frequencies corresponding to the same line have the same index. Tensor has dimensions [self.get_n_lines_freqs()]
        """
        return torch.arange(self.linedata.nrad.get(), device=device, dtype=Types.IndexInfo).repeat_interleave(self.linequadrature.nquads.get())


    def get_relevant_line_indices(self, opacity_compute_frequencies: torch.Tensor, sorted_linewidths_device: torch.Tensor, sorted_linefreqs_device: torch.Tensor, device: torch.device = torch.device("cpu"), MAX_DISTANCE_OPACITY_CONTRIBUTION: float = 10.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes which line should be evaluated when evaluating line opacities/emissivites at given frequencies. TODO: can be made more efficient by inspecting line width

        Args:
            opacity_compute_frequencies (torch.Tensor): Given frequencies for each point. Has dimensions [NPOINTS, NFREQS]
            sorted_linewidths_device (torch.Tensor): Sorted line widths of the lines (sorted for each point). Has dimensions [NPOINTS, self.linedata.nrad]
            sorted_linefreqs_device (torch.Tensor): Sorted line frequencies of the devices (sorted for each point). Has dimensions [self.linedata.nrad]
            device (torch.device, optional): Device on which to compute and return the result. Defaults to torch.device("cpu").

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tensors of respectively the minimum and maximum relevant lines for evaluating opacities/emissivities. Both torch.Tensors have dimensions [parameters.npoints, NFREQS]
        """
        min_bound_linefreqs = sorted_linefreqs_device[None, :] - MAX_DISTANCE_OPACITY_CONTRIBUTION * sorted_linewidths_device
        max_bound_linefreqs = sorted_linefreqs_device[None, :] + MAX_DISTANCE_OPACITY_CONTRIBUTION * sorted_linewidths_device

        lower_indices = torch.searchsorted(max_bound_linefreqs, opacity_compute_frequencies, right=False).to(Types.IndexInfo)
        upper_indices = torch.searchsorted(min_bound_linefreqs, opacity_compute_frequencies).to(Types.IndexInfo)
        return lower_indices, upper_indices
    
    def get_global_bound_relevant_line_indices(self, opacity_compute_frequencies: torch.Tensor, max_shift:torch.Tensor,  device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes a global heuristic bound for which line(s) to use for which frequencies when computing opacities/emissivities.

        Args:
            opacity_compute_frequencies (torch.Tensor): Given frequencies to use for the entire computation. Has dimensions [NFREQS]
            max_shift (torch.Tensor): Maximum shift given by the doppler shift. Lies between 0 and 1 and has dimensions [1]
            device (torch.device): Device on which to compute and return the result.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tensors of respectively the minimum and maximum relevant lines for evaluating opacities/emissivities. Both torch.Tensors have dimensions [NFREQS]
        """
        #opacity_compute_freqs: dims [NFREQS]
        max_relative_linewidth = torch.max(self.get_relative_line_width(device))
        min_bound_linefreqs = self.sorted_linefreqs.get(device) * (1.0 - 10.0 * max_relative_linewidth - 2.0*max_shift)
        max_bound_linefreqs = self.sorted_linefreqs.get(device) * (1.0 + 10.0 * max_relative_linewidth + 2.0*max_shift)

        lower_indices = torch.searchsorted(max_bound_linefreqs, opacity_compute_frequencies, right=False).to(Types.IndexInfo)
        upper_indices = torch.searchsorted(min_bound_linefreqs, opacity_compute_frequencies).to(Types.IndexInfo)
        return lower_indices, upper_indices
    
    #DEPRECATED: use pre-evaluated line instead
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
        inverse_line_widths = 1.0/sorted_linewidths_device
        #very narrow line profile
        line_profile_evaluation = sorted_linefreqs_device[None, None, :]*inverse_line_widths[:, None, :]/torch.sqrt(torch.pi*torch.ones(1, device=device))*torch.exp(-torch.pow(inverse_line_widths[:, None, :]*(sorted_linefreqs_device[None, None, :]-frequencies[:, :, None]), 2))
        line_opacities = torch.sum(line_profile_evaluation*line_opacities[:, None, :], dim=2)
        line_emissivities = torch.sum(line_profile_evaluation*line_emissivities[:, None, :], dim=2)
        return line_opacities, line_emissivities
    
    def evaluate_line_opacities_emissivites_single_line(self, frequencies: torch.Tensor, line_index: torch.Tensor, line_opacities: torch.Tensor, line_emissivities: torch.Tensor, sorted_linefreqs_device: torch.Tensor, sorted_linewidths_device: torch.Tensor, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluates the line opacities and emissivites for a single line at a time.

        Args:
            frequencies (torch.Tensor): Frequencies at which to evaluate the line opacity/emmissivity. Has dimensions [NPOINTS, NEVAL]
            line_index (torch.Tensor): Index of the line to evaluate. Has dimensions [NEVAL]
            line_opacities (torch.Tensor): Integrated line opacities. Has dimensions [NPOINTS, self.linedata.nrad]
            line_emissivities (torch.Tensor): Integrated line emissivities. Has dimensions [NPOINTS, self.linedata.nrad]
            sorted_linefreqs_device (torch.Tensor): Sorted line frequencies. Has dimensions [self.linedata.nrad]
            sorted_linewidths_device (torch.Tensor): Sorted line widths. Has dimensions [NPOINTS, self.linedata.nrad]
            device (torch.device, optional): Device on which to compute and return the result. Defaults to torch.device("cpu").

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Computed line opacities and emissivities. Both torch.Tensors have dimensions [NPOINTS, NFREQS]
        """
        inverse_line_widths = 1.0/sorted_linewidths_device
        #very narrow line profile
        line_profile_evaluation = sorted_linefreqs_device[None, line_index]*inverse_line_widths[:, line_index]/torch.sqrt(torch.pi*torch.ones(1, device=device))*torch.exp(-torch.pow(inverse_line_widths[:, line_index]*(sorted_linefreqs_device[None, line_index]-frequencies), 2))
        line_opacities = line_profile_evaluation*line_opacities[:, line_index]
        line_emissivities = line_profile_evaluation*line_emissivities[:, line_index]
        return line_opacities, line_emissivities

    def compute_ng_accelerated_level_pops(self, previous_level_pops: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Computes the Ng-accelerated level populations based on the previous level populations. (method based on Olson, Buchler, Auer 1986; described in more detail in Ceulemans et al. 2023 (in prep.)))

        Args:
            previous_level_pops (torch.Tensor): The previous level populations. Has dimensions [N_PREV_ITS, parameters.npoints, linedata.nlev].
            device (torch.device): The device on which to compute.

        Returns:
            torch.Tensor: The NG-accelerated level populations. Has dimensions [parameters.npoints, linedata.nlev].
        """
        normalized_level_pops = previous_level_pops / torch.sum(previous_level_pops, dim=2)[:, :, None]
        residual_pops = normalized_level_pops.diff(dim=0) #dimensions = [N_PREV_ITS-1, parameters.npoints, linedata.nlev]
        residual_matrix = torch.tensordot(residual_pops, residual_pops, dims=([1,2],[1,2])) #dims = [N_PREV_ITS-1, N_PREV_ITS-1]
        ng_accel_order = residual_matrix.shape[0]
        ones = torch.ones(ng_accel_order, device=device, dtype=Types.LevelPopsInfo)
        ng_coefficients = torch.linalg.solve(residual_matrix, ones) #dims = [N_PREV_ITS-1]
        ng_coefficients = ng_coefficients / torch.sum(ng_coefficients)#normalize the coefficients
        ng_accelerated_pops = torch.einsum("i,ijk->jk", ng_coefficients, previous_level_pops[1:,:,:]) #dims = [parameters.npoints, linedata.nlev]
        #put negative levelpops to 0 and renormalize
        ng_accelerated_pops[ng_accelerated_pops < 0.0] = 0.0
        ng_accelerated_pops = ng_accelerated_pops * self.population_tot.get(device)[:, None] / torch.sum(ng_accelerated_pops, dim=1)[:, None]
        return ng_accelerated_pops

    def compute_line_cooling(self, current_level_pops: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Computes the line cooling rate for each point, based on the given level populations
            TODO: test whether memory error might occur for NLTE models with many levels; if so, consider adding memory management

        Args:
            current_level_pops (torch.Tensor): The current level populations. Has dimensions [parameters.npoints, linedata.nlev].
            device (torch.device): The device on which to compute.

        Returns:
            torch.Tensor: The line cooling rate. Has dimensions [parameters.npoints] and units W/m^3
        """
        # The cooling rate is given by the net rates of the collisional transitions times the energy difference between the levels
        energy = self.linedata.energy.get(device)#dims = [linedata.nlev]
        temperature = self.dataCollection.get_data("gas temperature").get(device)#dims = [parameters.npoints]
        abundance = self.dataCollection.get_data("species abundance").get(device)#dims: [parameters.npoints, parameters.nspecs]
        cooling_rate = torch.zeros(self.parameters.npoints.get(), device=device, dtype=Types.LevelPopsInfo)
        for colpar in self.linedata.colpar:
            upper_levels = colpar.icol.get(device)
            lower_levels = colpar.jcol.get(device)

            #TODO: Refactor this; reuse of a code snippet code in solvers.py::compute_level_populations_statistical_equilibrium
            #Adjusting the abundance, depending on whether we have ortho or para H2
            colpar_abundance = colpar.adjust_abundace_for_ortho_para_h2(temperature, abundance[:, colpar.num_col_partner.get()])#dims: [parameters.npoints]
            
            collisional_rate_upper_to_lower = interpolate2D_linear(colpar.tmp.get(device), colpar.Cd.get(device), temperature) * colpar_abundance[:, None]#dims: [parameters.npoints, colpar.ncol]
            collisional_rate_lower_to_upper = interpolate2D_linear(colpar.tmp.get(device), colpar.Ce.get(device), temperature) * colpar_abundance[:, None]
            total_rate_upper_to_lower = collisional_rate_upper_to_lower * current_level_pops[:, upper_levels]#dims = [parameters.npoints, colpar.ncol]
            total_rate_lower_to_upper = collisional_rate_lower_to_upper * current_level_pops[:, lower_levels]#dims = [parameters.npoints, colpar.ncol]
            energy_diff = energy[upper_levels] - energy[lower_levels]#dims = [colpar.ncol]
            cooling_rate += torch.sum((total_rate_lower_to_upper - total_rate_upper_to_lower) * energy_diff[None, :], dim=1)#dims = [parameters.npoints]
            
        return cooling_rate


