from magrittetorch.model.model import Model
from magrittetorch.utils.storagetypes import DataCollection
import torch

torch.set_default_device('cuda')

testmodel = Model("models/testmodel.hdf5")

Pointsperside = 3
posx, posy, posz = torch.meshgrid(torch.arange(Pointsperside), torch.arange(Pointsperside), torch.arange(Pointsperside))
positions = torch.cat((posx.flatten(), posy.flatten(), posz.flatten()), dim=0).reshape(3,Pointsperside**3).T.type(torch.float64)
print(positions)

testmodel.geometry.points.position.set(positions)
testmodel.geometry.points.velocity.set(positions)
zeros = torch.zeros((Pointsperside**3)).type(torch.int64)
testmodel.geometry.points.n_neighbors.set(zeros)
empty = torch.zeros((0)).type(torch.int64)
testmodel.geometry.points.neighbors.set(empty)

Nrays = 5

origin_coords = torch.zeros((Nrays, 3))
raydirs = torch.ones((Nrays, 3))/torch.sqrt(3*torch.ones(1, dtype=torch.float64))
print("raydirs", raydirs)

distances = testmodel.geometry.distance_in_direction_3D_geometry(origin_coords, raydirs, torch.arange(Nrays), None)
print(distances)#seems to work, (note: ray dirs were not normalized, s)
# print(DataChecker.check_data_complete())
# print(testmodel.dataCollection.is_data_complete())

testmodel.write()

testReadmodel = Model("models/testmodel.hdf5")
testReadmodel.read()
print(testReadmodel.geometry.points.position.get())