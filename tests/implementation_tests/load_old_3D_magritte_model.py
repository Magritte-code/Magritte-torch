from magrittetorch.model.model import Model 
from magrittetorch.model.geometry.geometry import GeometryType
from magrittetorch.algorithms import raytracer
from torch.profiler import ProfilerActivity
import torch
import numpy as np
import time

# old_model = Model("./models/all_constant_single_ray.hdf5")
# old_model = Model("./models/vanZadelhoff_1a_3D_mesher.hdf5")
old_model = Model("./models/model_Phantom_3D.hdf5")
# torch.set_num_threads(10)
device = "cuda"

# old_model = Model("./models/vanZadelhoff_1a_1D_reduced.hdf5")

old_model.read()
print("npoints:", old_model.parameters.npoints.get())
try:
    print(old_model.geometry.boundary.is_boundary_point.get())
except ValueError as e:
    print("Caught error: ", e)
old_model.dataCollection.infer_data()
print(old_model.geometry.boundary.is_boundary_point.get())
old_model.geometry.geometrytype = GeometryType.General3D #spherical symmetry
result, dist = None, None
raydir = torch.ones((3), device=device)
with torch.profiler.profile(with_stack=True, profile_memory=True, activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # for i in range(4):
        # raydir[1] = i
        normed_raydir = raydir/torch.linalg.vector_norm(raydir)
        result, dist, scatter = raytracer.trace_rays_sparse(old_model,normed_raydir.type(torch.float64))
prof.export_chrome_trace("raytrace_3D")
print(prof.key_averages(group_by_stack_n=10).table(sort_by='self_cuda_time_total', row_limit=10))
print("results: ", result.size(), result.type(torch.int))
pos = old_model.geometry.points.position.get()
print(pos[:,0])
print(pos)

start = time.time()
result, dist, scatter = raytracer.trace_rays_sparse(old_model,normed_raydir.type(torch.float64))
end =time.time()
print("total time: ", end-start)
print("result size", result.size())