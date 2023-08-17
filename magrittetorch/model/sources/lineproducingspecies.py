from typing import List, Union, Optional, Any
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor, DelayedListOfClassInstances
from magrittetorch.model.parameters import Parameters
import torch
from astropy import units
import astropy.constants
from magrittetorch.model.sources.linedata import Linedata


#TODO: implement in seperate file
class Quadrature:
    pass

#plan to implement
class ApproxLambda:
    pass

class LineProducingSpecies:
    def __init__(self, params: Parameters, dataCollection : DataCollection, i: int) -> None:
        storagedir : str = "lines/lineProducingSpecies_"+str(i)+"/"#TODO: is legacy io for now; figure out how to switch to new structure
        self.parameters: Parameters = params
        self.dataCollection : DataCollection = dataCollection
        self.linedata : Linedata = Linedata(self.parameters, self.dataCollection, i)
        self.quadrature : Quadrature = Quadrature()
        self.approxlambda : ApproxLambda = ApproxLambda()
        self.population_tot : InferredTensor = InferredTensor(Types.LevelPopsInfo, [self.parameters.npoints], units.m**-3, self._infer_population_tot, storagedir+"population_tot"); dataCollection.add_inferred_dataset(self.population_tot, "population_tot_"+str(i))
        self.population : InferredTensor = InferredTensor(Types.LevelPopsInfo, [self.parameters.npoints, self.linedata.nlev], units.m**-3, self._infer_population, relative_storage_location = storagedir+"population"); dataCollection.add_inferred_dataset(self.population, "population_"+str(i))
        # self.prev_populations : InferredTensor TODO: err, list/tensor with more dims should suffice
                
    def _infer_population(self) -> torch.Tensor:
        non_normalized_pops = torch.reshape(self.linedata.weight.get(), (1,-1)) * torch.exp(-torch.reshape(self.linedata.energy.get(), (1,-1)) / (astropy.constants.k_B.value * torch.reshape(self.dataCollection.get_data("gas temperature").get(), (-1,1))))#type: ignore
        return torch.reshape(self.population_tot.get(), (-1,1)) * non_normalized_pops / torch.sum(non_normalized_pops, dim = 1).reshape(-1,1)

    def _infer_population_tot(self) -> torch.Tensor:
        return self.dataCollection.get_data("species abundance").get()[:,self.linedata.num.get()]#type: ignore