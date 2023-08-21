from typing import List, Union, Optional, Any
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor, DelayedListOfClassInstances, StorageNdarray
from magrittetorch.model.parameters import Parameters, Parameter
from magrittetorch.utils.io import LegacyHelper
import torch
from astropy import units

class LineQuadrature:
    def __init__(self, params: Parameters, dataCollection : DataCollection, lineproducingspeciesidx: int) -> None:
        storagedir : str = "lines/lineProducingSpecies_"+str(lineproducingspeciesidx)+"/quadrature/"#TODO: is legacy io for now; figure out how to switch to new structure
        self.parameters: Parameters = params
        self.dataCollection: DataCollection = dataCollection
        self.nquads: Parameter[int] = Parameter[int](storagedir+"nquads", legacy_converter=(storagedir+"weights", LegacyHelper.read_length_of_dataset)); self.dataCollection.add_local_parameter(self.nquads)
        self.weights: StorageTensor = StorageTensor(Types.FrequencyInfo, [self.nquads], units.dimensionless_unscaled, storagedir+"weights"); self.dataCollection.add_data(self.weights, "quadrature weights_"+str(lineproducingspeciesidx))
        self.roots: StorageTensor = StorageTensor(Types.FrequencyInfo, [self.nquads], units.dimensionless_unscaled, storagedir+"roots"); self.dataCollection.add_data(self.roots, "quadrature roots_"+str(lineproducingspeciesidx))
