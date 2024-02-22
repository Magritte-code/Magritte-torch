from typing import List, Union, Optional, Any
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor, DelayedListOfClassInstances, StorageNdarray
from magrittetorch.model.parameters import Parameters, Parameter
from magrittetorch.model.sources.collisionpartner import CollisionPartner
from magrittetorch.utils.io import LegacyHelper
from astropy import units


#TODO: implement in seperate file
class Quadrature:
    pass

#plan to implement
class ApproxLambda:
    pass

class Linedata:
    def __init__(self, params: Parameters, dataCollection : DataCollection, lineproducingspeciesidx: int) -> None:
        storagedir : str = "lines/lineProducingSpecies_"+str(lineproducingspeciesidx)+"/linedata/"#TODO: is legacy io for now; figure out how to switch to new structure
        self.parameters: Parameters = params
        self.dataCollection : DataCollection = dataCollection
        #set local parameters
        self.num: Parameter[int] = Parameter[int](storagedir+"num")#: Index of species in list
        self.dataCollection.add_local_parameter(self.num)
        self.sym: Parameter[str] = Parameter[str](storagedir+"sym")#: Symbol of chemical species
        self.dataCollection.add_local_parameter(self.sym)
        self.nlev: Parameter[int] = Parameter[int](storagedir+"nlev")#: Number of levels
        self.dataCollection.add_local_parameter(self.nlev) 
        self.nrad: Parameter[int] = Parameter[int](storagedir+"nrad")#: Number of radiative transitions
        self.dataCollection.add_local_parameter(self.nlev)
        self.ncolpar: Parameter[int] = Parameter[int](storagedir+"ncolpar", legacy_converter=(storagedir, lambda x: LegacyHelper.read_length_of_group("collisionPartner_", x)))#: Number of collision partners
        self.dataCollection.add_local_parameter(self.ncolpar)
        self.inverse_mass: Parameter[float] = Parameter[float](storagedir+"inverse_mass") #: Inverse mass of species (in 1/atomic mass units)
        self.dataCollection.add_local_parameter(self.inverse_mass) # inverse mass of species (in 1/atomic mass units)
        #hmm, not sure where inverse mass belongs to; either parameter (stored in an attribute) or in storagetensor (as it is data we actually use; but dimension/unit checking will not be used)
        #set delayed list
        self.colpar: DelayedListOfClassInstances[CollisionPartner] = DelayedListOfClassInstances[CollisionPartner](self.ncolpar, lambda j: CollisionPartner(self.parameters, self.dataCollection, lineproducingspeciesidx, j), "collisionPartner")#: Collision partner data
        self.dataCollection.add_delayed_list(self.colpar)
        #all stored data
        self.energy: StorageTensor = StorageTensor(Types.LevelPopsInfo, [self.nlev], units.J, storagedir+"energy")#: Energy of each level; dtype = :py:attr:`.Types.LevelPopsInfo`, dims = [:py:attr:`nlev`], units = units.J
        self.dataCollection.add_data(self.energy, "line energy_"+str(lineproducingspeciesidx))
        self.weight: StorageTensor = StorageTensor(Types.LevelPopsInfo, [self.nlev], units.dimensionless_unscaled, storagedir+"weight")#: Statistical weight of each level; dtype = :py:attr:`.Types.LevelPopsInfo`, dims = [:py:attr:`nlev`], units = units.dimensionless_unscaled
        self.dataCollection.add_data(self.weight, "line weight_"+str(lineproducingspeciesidx))
        self.frequency: StorageTensor = StorageTensor(Types.FrequencyInfo, [self.nrad], units.Hz, storagedir+"frequency")#: Frequency of each radiative transition; dtype = :py:attr:`.Types.FrequencyInfo`, dims = [:py:attr:`nrad`], units = units.Hz
        self.dataCollection.add_data(self.frequency, "line frequency_"+str(lineproducingspeciesidx))
        self.A: StorageTensor = StorageTensor(Types.LevelPopsInfo, [self.nrad], units.s**-1, storagedir+"A") #: Einstein A coefficients (spontaneous emission); dtype = :py:attr:`.Types.LevelPopsInfo`, dims = [:py:attr:`nrad`], units = units.s**-1
        self.dataCollection.add_data(self.A, "Einstein A_"+str(lineproducingspeciesidx))
        self.Ba: StorageTensor = StorageTensor(Types.LevelPopsInfo, [self.nrad], units.m**2*units.J**-1*units.s**-1, storagedir+"Ba")#: Einstein Ba coefficients (absorption); dtype = :py:attr:`.Types.LevelPopsInfo`, dims = [:py:attr:`nrad`], units = units.m**2*units.J**-1*units.s**-1
        self.dataCollection.add_data(self.Ba, "Einstein Ba_"+str(lineproducingspeciesidx))
        self.Bs: StorageTensor = StorageTensor(Types.LevelPopsInfo, [self.nrad], units.m**2*units.J**-1*units.s**-1, storagedir+"Bs")#: Einstein Bs coefficients (stimulated emission); dtype = :py:attr:`.Types.LevelPopsInfo`, dims = [:py:attr:`nrad`], units = units.m**2*units.J**-1*units.s**-1
        self.dataCollection.add_data(self.Bs, "Einstein Bs_"+str(lineproducingspeciesidx))
        self.irad: StorageTensor = StorageTensor(Types.IndexInfo, [self.nrad], units.dimensionless_unscaled, storagedir+"irad") #: Upper level of each line transition; dtype = :py:attr:`.Types.IndexInfo`, dims = [:py:attr:`nrad`], units = units.dimensionless_unscaled
        self.dataCollection.add_data(self.irad, "line upper indices_"+str(lineproducingspeciesidx))
        self.jrad: StorageTensor = StorageTensor(Types.IndexInfo, [self.nrad], units.dimensionless_unscaled, storagedir+"jrad") #: Lower level of each line transition; dtype = :py:attr:`.Types.IndexInfo`, dims = [:py:attr:`nrad`], units = units.dimensionless_unscaled
        self.dataCollection.add_data(self.jrad, "line lower indices_"+str(lineproducingspeciesidx))
        