#contains everything about the boundary of a model
from enum import Enum
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor
from magrittetorch.model.parameters import Parameters, Parameter, EnumParameter
from magrittetorch.utils.constants import astropy_const
from typing import Union, Tuple
import torch
from astropy import units, constants

storagedir : str = "geometry/boundary/"

class BoundaryCondition(Enum):
    """
    Type of ng-acceleration to use
    """
    Zero : int = 0
    Thermal : int = 1
    CMB : int = 2

class BoundaryType(Enum):
    Sphere1D: int = 0
    Sphere3D: int = 1
    AxisAlignedCube: int = 2

#TODO: fully implement cube boundary
class Boundary:
    
    def __init__(self, params : Parameters, dataCollection : DataCollection) -> None:
        self.parameters: Parameters = params
        self.dataCollection: DataCollection = dataCollection
        self.boundary2point: StorageTensor = StorageTensor(Types.IndexInfo, [self.parameters.nboundary], units.dimensionless_unscaled, storagedir+"boundary2point"); self.dataCollection.add_data(self.boundary2point, "boundary2point") # maps boundary index \in[0, nboundary-1] to point index \in[0, npoints-1]
        self.boundary_condition: StorageTensor = StorageTensor(Types.Enum, [self.parameters.nboundary], units.dimensionless_unscaled, storagedir+"boundary_condition"); self.dataCollection.add_data(self.boundary_condition, "boundary_condition") #contains the boundary conditions for each boundary point
        self.boundary_temperature: StorageTensor = StorageTensor(Types.GeometryInfo, [self.parameters.nboundary], units.K, storagedir+"boundary_temperature"); self.dataCollection.add_data(self.boundary_temperature, "boundary_temperature") # contains the CMB temperature corresponding to incoming photons
        self.point2boundary: InferredTensor = InferredTensor(Types.IndexInfo, [self.parameters.npoints], units.dimensionless_unscaled, self._infer_point2boundary); self.dataCollection.add_inferred_dataset(self.point2boundary, "point2boundary") # maps point index \in[0, npoints-1] to boundary index \in[0, nboundary-1]
        self.is_boundary_point: InferredTensor = InferredTensor(Types.Bool, [self.parameters.npoints], units.dimensionless_unscaled, self._infer_is_boundary_points); self.dataCollection.add_inferred_dataset(self.is_boundary_point, "is_boundary_point") # contains whether or not the given point is a boundary point
        self.boundaryType: EnumParameter[BoundaryType, type[BoundaryType]] = EnumParameter[BoundaryType, type[BoundaryType]](BoundaryType, "boundaryType", ("spherical_symmetry", self.__legacy_convert_boundaryType)); self.dataCollection.add_local_parameter(self.boundaryType)
        self.center: InferredTensor = InferredTensor(Types.GeometryInfo, [3], units.m, self._infer_center); self.dataCollection.add_inferred_dataset(self.center, "model center") #contains the center of the model, used for computing which part of the boundary is necessary

    def _infer_point2boundary(self) -> torch.Tensor:
        """Infers the tensor which maps points to boundary points.

        Returns:
            torch.Tensor: Contains the boundary point index for every boundary point. Contain parameters.npoints for all other points. Has dimensions [parameters.npoints]
        """
        data : torch.Tensor = self.parameters.npoints.get()*torch.ones(self.parameters.npoints.get()).type(Types.IndexInfo)
        data[self.boundary2point.get()] = torch.arange(self.parameters.nboundary.get()).type(Types.IndexInfo)
        return data

    def _infer_is_boundary_points(self) -> torch.Tensor:
        """Infers which points are boundary points

        Returns:
            torch.Tensor: Boolean tensor. Has dimensions [parameters.npoints]
        """
        data : torch.Tensor = (self.point2boundary.get()-self.parameters.npoints.get()).type(Types.Bool)
        return data
    
    def __legacy_convert_boundaryType(self, is_spherically_symmetric: bool) -> BoundaryType:
        match is_spherically_symmetric:#for some reason the "bool" is saved as a string. I have no clue why
            case "true":
                return BoundaryType.Sphere1D
            case "false":
                if self.__check_if_cube3D():
                    return BoundaryType.AxisAlignedCube
                else:
                    return BoundaryType.Sphere3D
            case _:
                raise KeyError("Something went wrong with setting the model geometry")
            
    def __check_if_cube3D(self) -> bool:
        #very simple check whether the model could be a cube, by checking if a single corner exists
        #This is just for differentiating between cubes and spheres when reading old magritte models
        positions_device: torch.Tensor
        positions_device = self.dataCollection.get_data("position").get()#type: ignore
        max_vals, _ = torch.max(positions_device, dim=0)
        is_max = torch.eq(positions_device, max_vals[None, :])
        return torch.any(torch.prod(is_max, dim=1))#type: ignore
    
    def _infer_center(self) -> torch.Tensor:
        #makes assumption that all old models are (almost) symmetrical in all major axes
        positions: torch.Tensor = self.dataCollection.get_data("position").get()#type: ignore
        min_vals, _ = torch.min(positions, dim=0)
        max_vals, _ = torch.max(positions, dim=0)
        return 0.5*(min_vals+max_vals)#type: ignore
        
    def get_boundary_in_direction(self, direction: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Returns the boundary points which are valid (lie at the end of rays) in the given direction.

        Args:
            direction (torch.Tensor): The direction the rays will be traveling in. Has dimensions [3]
            device (torch.device): The device on which to compute and return the result

        Raises:
            NotImplementedError: If self.boundaryType has not yet been implemented. TODO: implement BoundaryType.AxisAlignedCube

        Returns:
            torch.Tensor: The boolean tensor of boundary points lying at the end of rays in the given direction. Has dimensions [parameters.npoints]
        """
        match self.boundaryType.get():
            case BoundaryType.Sphere1D:
                #1D models do not need any changes
                return self.is_boundary_point.get(device)
            case BoundaryType.Sphere3D:
                #For a 3D sphere, we just cut the sphere in half
                positions_device: torch.Tensor
                positions_device = self.dataCollection.get_data("position").get(device)#type: ignore
                is_boundary_copy = self.is_boundary_point.get(device).clone()
                full_boundary_positions = positions_device[is_boundary_copy,:]
                incorrect_boundary = torch.matmul(full_boundary_positions - self.center.get(device)[None, :], direction)<=0.0
                is_boundary_copy[self.boundary2point.get(device)[incorrect_boundary]] = False
                return is_boundary_copy
            case BoundaryType.AxisAlignedCube:
                #For a 3D cube, we get min and max boundary points
                positions_device = self.dataCollection.get_data("position").get(device)#type: ignore
                is_boundary_copy = self.is_boundary_point.get(device).clone()
                full_boundary_positions = positions_device[is_boundary_copy,:]
                minpos, maxpos = torch.min(full_boundary_positions, dim=0).values, torch.max(full_boundary_positions, dim=0).values

                #there might exist some better way to find the indices of each of the sides, but for now, this suffices
                xbounds = torch.logical_or(full_boundary_positions[:,0] == minpos[0], full_boundary_positions[:,0] == maxpos[0])
                ybounds = torch.logical_or(full_boundary_positions[:,1] == minpos[1], full_boundary_positions[:,1] == maxpos[1])
                zbounds = torch.logical_or(full_boundary_positions[:,2] == minpos[2], full_boundary_positions[:,2] == maxpos[2])

                # For every axis direction, we check which points may still be on the boundary     
                # For lower dimensional models, we do not care about the directions in which the model width is 0 
                correctfullbounds = torch.zeros(self.parameters.nboundary.get(), dtype=Types.Bool, device=device)
                if (maxpos[0] - minpos[0] != 0.0):
                    correctfullbounds = torch.logical_or(correctfullbounds, torch.logical_and(xbounds, (full_boundary_positions[:, 0]-self.center.get(device)[None, 0])*direction[0] >= 0.0))
                if (maxpos[1] - minpos[1] != 0.0):
                    correctfullbounds = torch.logical_or(correctfullbounds, torch.logical_and(ybounds, (full_boundary_positions[:, 1]-self.center.get(device)[None, 1])*direction[1] >= 0.0))
                if (maxpos[2] - minpos[2] != 0.0):
                    correctfullbounds = torch.logical_or(correctfullbounds, torch.logical_and(zbounds, (full_boundary_positions[:, 2]-self.center.get(device)[None, 2])*direction[2] >= 0.0))

                incorrect_boundary = torch.logical_not(correctfullbounds)

                is_boundary_copy[self.boundary2point.get(device)[incorrect_boundary]] = False

                return is_boundary_copy
            case _:
                raise NotImplementedError("Not yet implemented for BoundaryType", self.boundaryType.get())

    def get_boundary_intensity(self, point_indices: torch.Tensor, frequencies: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Computes the boundary intensity, given the boundary point and (comoving frame) frequencies at which to compute it.

        Args:
            point_indices (torch.Tensor): Indices of the points at which to compute the boundary intensities. Has dimensions [NPOINTS]
            frequencies (torch.Tensor): (comoving frame) frequencies at which to compute the boundary intensities. Has dimensions [NPOINTS, NFREQS]
            device (torch.device): Device on which to compute and return the result

        Returns:
            torch.Tensor: The boundary intensity. Has dimensions [NPOINTS, NFREQS]
        """
        nboundeval = point_indices.size(dim=0)
        nfreqeval = frequencies.size(dim=1)
        boundary_indices = self.point2boundary.get(device)[point_indices]#dims: [NBOUNDEVAL]
        boundary_condition = self.boundary_condition.get(device)[boundary_indices]#dims: [NBOUNDEVAL]
        boundary_intensity = torch.zeros(nboundeval, nfreqeval, dtype=Types.FrequencyInfo, device=device)
        # zero_boundary = boundary_condition == BoundaryCondition.Zero #err, zero boundary doesn't do anything, so can be ignored
        CMB_boundary_indices = torch.eq(boundary_condition, BoundaryCondition.CMB.value)
        Thermal_boundary_indices = torch.eq(boundary_condition, BoundaryCondition.Thermal.value)
        boundary_intensity[CMB_boundary_indices, :]=self.blackbody_intensity(frequencies[CMB_boundary_indices, :], astropy_const.Tcmb.value*torch.ones(CMB_boundary_indices.size(dim=0), dtype=Types.FrequencyInfo, device=device)) 
        boundary_intensity[Thermal_boundary_indices, :]=self.blackbody_intensity(frequencies[Thermal_boundary_indices, :], self.boundary_temperature.get(device=device)[boundary_indices[Thermal_boundary_indices]])
        return boundary_intensity
    
    def blackbody_intensity(self, frequency: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        """Computes the blackbody (planckian) intensity, given frequency and temperature.

        Args:
            frequency (torch.Tensor): The frequency [in Hz]. Has dimensions [NPOINTS, NFREQS]
            temperature (torch.Tensor): The temperature [in K]. Has dimensions [NPOINTS]

        Returns:
            _type_: The blackbody intensity. Has dimensions [NPOINTS, NFREQS]
        """
        return 2.0 * constants.h.value / constants.c.value**2 * frequency**3 / torch.expm1(constants.h.value / constants.k_B.value * frequency / temperature[:, None])#type: ignore
    