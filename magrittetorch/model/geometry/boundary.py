#contains everything about the boundary of a model
from enum import Enum
from magrittetorch.utils.storagetypes import StorageTensor, Types, DataCollection, InferredTensor
from model.parameters import Parameters
import torch

storagedir : str = "geometry/boundary/"

class BoundaryCondition(Enum):
    """
    Type of ng-acceleration to use
    """
    Zero : int = 0
    Thermal : int = 1
    CMB : int = 2

#TODO: add functionality for reduced boundary to class (for better ray tracing); just return some mask on the boundary and a reduced point2boundary
class Boundary:
    
    def __init__(self, params : Parameters, dataCollection : DataCollection) -> None:
        self.parameters: Parameters = params
        self.dataCollection : DataCollection = dataCollection
        self.boundary2point : StorageTensor = StorageTensor(Types.IndexInfo, [self.parameters.nboundary], storagedir+"boundary2point"); self.dataCollection.add_data(self.boundary2point) # maps boundary index \in[0, nboundary-1] to point index \in[0, npoints-1]
        self.boundary_condition : StorageTensor = StorageTensor(Types.Enum, [self.parameters.nboundary], storagedir+"boundary_condition"); self.dataCollection.add_data(self.boundary_condition) #contains the boundary conditions for each boundary point
        self.boundary_temperature: StorageTensor = StorageTensor(Types.GeometryInfo, [self.parameters.nboundary], storagedir+"boundary_temperature"); self.dataCollection.add_data(self.boundary_temperature) # contains the CMB temperature corresponding to incoming photons
        self.point2boundary : InferredTensor = InferredTensor(Types.IndexInfo, [self.parameters.npoints], self._infer_point2boundary); self.dataCollection.add_inferred_dataset(self.point2boundary) # maps point index \in[0, npoints-1] to boundary index \in[0, nboundary-1]
        self.is_boundary_point: InferredTensor = InferredTensor(Types.Bool, [self.parameters.npoints], self._infer_is_boundary_points); self.dataCollection.add_inferred_dataset(self.is_boundary_point) # contains whether or not the given point is a boundary point

    def _infer_point2boundary(self) -> torch.Tensor:
        data : torch.Tensor = self.parameters.npoints.get()*torch.ones(self.parameters.npoints.get()).type(Types.IndexInfo)
        data[self.boundary2point.get()] = torch.arange(self.parameters.nboundary.get()).type(Types.IndexInfo)
        return data

    def _infer_is_boundary_points(self) -> torch.Tensor:
        data : torch.Tensor = (self.point2boundary.get()-self.parameters.npoints.get()).type(Types.Bool)
        return data