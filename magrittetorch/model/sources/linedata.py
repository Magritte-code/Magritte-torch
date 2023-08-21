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
        # self.num : StorageNdarray(Types.IndexInfo, [1], units.dimensionless_unscaled, storagedir+".num") = i
        #set local parameters
        self.num: Parameter[int] = Parameter[int](storagedir+"num"); self.dataCollection.add_local_parameter(self.num) #index of species in list
        self.sym: Parameter[str] = Parameter[str](storagedir+"sym"); self.dataCollection.add_local_parameter(self.sym) #symbol of species
        self.nlev: Parameter[int] = Parameter[int](storagedir+"nlev"); self.dataCollection.add_local_parameter(self.nlev) #number of levels
        self.nrad: Parameter[int] = Parameter[int](storagedir+"nrad"); self.dataCollection.add_local_parameter(self.nlev) #number of radiative transitions
        self.ncolpar: Parameter[int] = Parameter[int](storagedir+"ncolpar", legacy_converter=(storagedir, lambda x: LegacyHelper.read_length_of_group("collisionPartner_", x))); self.dataCollection.add_local_parameter(self.ncolpar)
        self.inverse_mass: Parameter[float] = Parameter[float](storagedir+"inverse_mass"); self.dataCollection.add_local_parameter(self.inverse_mass) # inverse mass of species (in 1/atomic mass units)
        #hmm, not sure where inverse mass belongs to; either parameter (stored in an attribute) or in storagetensor (as it is data we actually use; but dimension/unit checking will not be used)
        # self.inverse_mass: StorageTensor = StorageTensor(Types.GeometryInfo, [1], units.dimensionless_unscaled, storagedir+"inverse_mass", read_from_attribute = True); self.dataCollection.add_data(self.inverse_mass, "inverse mass_"+str(lineproducingspeciesidx)) #1/species mass (in atomic mass unit)
        #set delayed list
        self.colpar: DelayedListOfClassInstances[CollisionPartner] = DelayedListOfClassInstances[CollisionPartner](self.ncolpar, lambda j: CollisionPartner(self.parameters, self.dataCollection, lineproducingspeciesidx, j), "collisionPartner"); self.dataCollection.add_delayed_list(self.colpar)
        #all stored data
        self.energy: StorageTensor = StorageTensor(Types.LevelPopsInfo, [self.nlev], units.J, storagedir+"energy"); self.dataCollection.add_data(self.energy, "line energy_"+str(lineproducingspeciesidx))#energy of level
        self.weight: StorageTensor = StorageTensor(Types.LevelPopsInfo, [self.nlev], units.dimensionless_unscaled, storagedir+"weight"); self.dataCollection.add_data(self.weight, "line weight_"+str(lineproducingspeciesidx)) #statistical weight
        self.frequency: StorageTensor = StorageTensor(Types.FrequencyInfo, [self.nrad], units.Hz, storagedir+"frequency"); self.dataCollection.add_data(self.frequency, "line frequency_"+str(lineproducingspeciesidx)) #line frequency
        self.A: StorageTensor = StorageTensor(Types.LevelPopsInfo, [self.nrad], units.s**-1, storagedir+"A"); self.dataCollection.add_data(self.A, "Einstein A_"+str(lineproducingspeciesidx)) #einstein A coefficients (spontaneous emission)
        self.Ba: StorageTensor = StorageTensor(Types.LevelPopsInfo, [self.nrad], units.m**3*units.J**-1*units.s**-2, storagedir+"Ba"); self.dataCollection.add_data(self.Ba, "Einstein Ba_"+str(lineproducingspeciesidx))#einstein Ba coefficients (absorption)
        self.Bs: StorageTensor = StorageTensor(Types.LevelPopsInfo, [self.nrad], units.m**3*units.J**-1*units.s**-2, storagedir+"Bs"); self.dataCollection.add_data(self.Bs, "Einstein Bs_"+str(lineproducingspeciesidx))#einstein Bs coefficients (stimulated emission)
        self.irad: StorageTensor = StorageTensor(Types.IndexInfo, [self.nrad], units.dimensionless_unscaled, storagedir+"irad"); self.dataCollection.add_data(self.irad, "line upper indices_"+str(lineproducingspeciesidx)) #upper index of line transition
        self.jrad: StorageTensor = StorageTensor(Types.IndexInfo, [self.nrad], units.dimensionless_unscaled, storagedir+"jrad"); self.dataCollection.add_data(self.jrad, "line lower indices_"+str(lineproducingspeciesidx)) #lower index of line transition
        