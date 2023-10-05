from typing import List, Union, Optional, Any
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor, DelayedListOfClassInstances, StorageNdarray
from magrittetorch.model.parameters import Parameters, Parameter
from magrittetorch.utils.io import LegacyHelper
import torch
from astropy import units

class CollisionPartner:
    def __init__(self, params: Parameters, dataCollection : DataCollection, lineproducingspeciesidx: int, colparidx: int) -> None:
        storagedir : str = "lines/lineProducingSpecies_"+str(lineproducingspeciesidx)+"/linedata/collisionPartner_"+str(colparidx)+"/"#TODO: is legacy io for now; figure out how to switch to new structure
        self.parameters: Parameters = params
        self.dataCollection: DataCollection = dataCollection
        self.ntmp: Parameter[int] = Parameter[int](storagedir+"ntmp", legacy_converter=(storagedir+"tmp", LegacyHelper.read_length_of_dataset)); self.dataCollection.add_local_parameter(self.ntmp)
        self.ncol: Parameter[int] = Parameter[int](storagedir+"ncol", legacy_converter=(storagedir+"icol", LegacyHelper.read_length_of_dataset)); self.dataCollection.add_local_parameter(self.ncol)
        self.icol: StorageTensor = StorageTensor(Types.IndexInfo, [self.ncol], units.dimensionless_unscaled, storagedir+"icol"); self.dataCollection.add_data(self.icol, "collisional upper level_"+str(lineproducingspeciesidx)+"_"+str(colparidx))
        self.jcol: StorageTensor = StorageTensor(Types.IndexInfo, [self.ncol], units.dimensionless_unscaled, storagedir+"jcol"); self.dataCollection.add_data(self.jcol, "collisional lower level_"+str(lineproducingspeciesidx)+"_"+str(colparidx))
        self.tmp: StorageTensor = StorageTensor(Types.GeometryInfo, [self.ntmp], units.K, storagedir+"tmp"); self.dataCollection.add_data(self.tmp, "collision temperature_"+str(lineproducingspeciesidx)+"_"+str(colparidx))
        self.Ce: StorageTensor = StorageTensor(Types.LevelPopsInfo, [self.ntmp, self.ncol], units.s**-1, storagedir+"Ce"); self.dataCollection.add_data(self.Ce, "Collisional excitation rates_"+str(lineproducingspeciesidx)+"_"+str(colparidx))
        self.Cd: StorageTensor = StorageTensor(Types.LevelPopsInfo, [self.ntmp, self.ncol], units.s**-1, storagedir+"Cd"); self.dataCollection.add_data(self.Cd, "Collisional de-excitation rates_"+str(lineproducingspeciesidx)+"_"+str(colparidx))
        self.num_col_partner: Parameter[int] = Parameter[int](storagedir+"num_col_partner"); self.dataCollection.add_local_parameter(self.num_col_partner)
        self.orth_or_para_H2: Parameter[str] = Parameter[str](storagedir+"orth_or_para_H2"); self.dataCollection.add_local_parameter(self.orth_or_para_H2)

        