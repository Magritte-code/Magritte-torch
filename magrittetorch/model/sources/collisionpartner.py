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


    #TODO: implement this in preprocessing, maybe just infer some more complete set of species data
    def adjust_abundace_for_ortho_para_h2(self, temperature: torch.Tensor, abundance: torch.Tensor) -> torch.Tensor:
        """Adjusts the abundance of H2 for ortho-para H2

        Args:
            temperature (torch.Tensor): Temperature at which to evaluate the collisional rates
            abundance (torch.Tensor): The abundance of H2 (or any other species)

        Returns:
            torch.Tensor: The abundance of H2 adjusted for ortho-para H2
            The abundance is unchanged for other species
        """
        if self.orth_or_para_H2.get() != "n":
            #See erratum on Flower 1984, https://ui.adsabs.harvard.edu/abs/1985MNRAS.213..991F/abstract
            #There exists both nuclear spin degeneracy and rotational angular momentum degeneracy, both accounting for a factor of 3
            frac_H2_para = 1.0/(1.0 + 9.0 * torch.exp(-170.5/temperature))
            if self.orth_or_para_H2.get() == "o":
                return abundance * (1.0 - frac_H2_para)
            elif self.orth_or_para_H2.get() == "p":
                return abundance * frac_H2_para
        return abundance
