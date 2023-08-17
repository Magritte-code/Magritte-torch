from typing import List, Union, Optional, Any
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