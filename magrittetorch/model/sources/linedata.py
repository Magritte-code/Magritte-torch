from typing import List, Union, Optional, Any
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor, DelayedListOfClassInstances, StorageNdarray
from magrittetorch.model.parameters import Parameters, Parameter
import torch
from astropy import units


#TODO: implement in seperate file
class Quadrature:
    pass

#plan to implement
class ApproxLambda:
    pass

class Linedata:
    def __init__(self, params: Parameters, dataCollection : DataCollection, i: int) -> None:
        storagedir : str = "lines/lineProducingSpecies_"+str(i)+"/linedata/"#TODO: is legacy io for now; figure out how to switch to new structure
        self.parameters: Parameters = params
        self.dataCollection : DataCollection = dataCollection
        # self.num : StorageNdarray(Types.IndexInfo, [1], units.dimensionless_unscaled, storagedir+".num") = i
        #set local parameters
        self.num : Parameter[int] = Parameter[int](storagedir+"num"); self.dataCollection.add_local_parameter(self.num)
        self.sym : Parameter[str] = Parameter[str](storagedir+"sym"); self.dataCollection.add_local_parameter(self.sym)
        self.nlev : Parameter[int] = Parameter[int](storagedir+"nlev"); self.dataCollection.add_local_parameter(self.nlev)
        print(self.nlev, self.num, self.sym)
        self.energy : StorageTensor = StorageTensor(Types.LevelPopsInfo, [self.nlev], units.J, storagedir+"energy"); self.dataCollection.add_data(self.energy, "line energy_"+str(i))#energy of level
        self.weight : StorageTensor = StorageTensor(Types.LevelPopsInfo, [self.nlev], units.dimensionless_unscaled, storagedir+"weight"); self.dataCollection.add_data(self.weight, "line weight_"+str(i)) #statistical weight
        
                