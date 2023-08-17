from magrittetorch.model.model import Model 
from magrittetorch.model.geometry.geometry import GeometryType
from magrittetorch.algorithms import raytracer
import torch
import numpy as np
import time

old_model = Model("./models/vanZadelhoff_1a_1D.hdf5")
# old_model = Model("./models/vanZadelhoff_1a_1D_reduced.hdf5")

old_model.read()
try:
    print(old_model.geometry.boundary.is_boundary_point.get())
except ValueError as e:
    print("Caught error: ", e)
old_model.dataCollection.infer_data()
print(old_model.geometry.boundary.is_boundary_point.get())
old_model.geometry.geometrytype = GeometryType.SpericallySymmetric1D #spherical symmetry
#to get the angle: tan(\alpha) = vertical component/horizontal component
#thus \alpha = arctan(sqrt(y**2+z**2)/x)
raydir = torch.Tensor([-1,0.1,0])
normed_raydir = raydir/torch.linalg.vector_norm(raydir)
# print(normed_raydir)
# result, dist = None, None
# with torch.autograd.profiler.profile(with_stack=True, profile_memory=True, use_cuda=True) as prof:
#     # for i in range(10):
#         result, dist, scatter = raytracer.trace_rays_sparse(old_model,normed_raydir.type(torch.float64))
# print(prof.key_averages(group_by_stack_n=10).table(sort_by='self_cuda_time_total', row_limit=10))
# # print(prof.key_averages)
# print(result.size(), result)
# pos = old_model.geometry.points.position.get()
# print(torch.min(torch.nonzero(pos[:,0]>=np.sqrt(2)*pos[0,0])))
# print(torch.min(torch.nonzero(pos[:,0]>=2*pos[0,0])))
# print(torch.min(pos[:,0]), torch.max(pos))
# closest_factor = raydir[1]/torch.sqrt(raydir[0]**2+raydir[1]**2)
# print("factor: ", closest_factor, "applied to max: ", closest_factor*pos[-1,0])
# print("thus all which corresponds:", (closest_factor*pos[-1,0]<pos[:,0]).nonzero())
# print(pos[:,0])
# # print(pos)

# start = time.time()
# result, dist, scatter = raytracer.trace_rays_sparse(old_model,normed_raydir.type(torch.float64))
# end =time.time()
# print("total time: ", end-start)


print(old_model.parameters.nlspecs.get())
print(old_model.sources.lines.lineProducingSpecies[0].linedata)
print(old_model.sources.lines.lineProducingSpecies[0].population.get())
