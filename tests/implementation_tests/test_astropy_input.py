#based on simple old magritte model generation; all_constant_single_ray
#Note: imcomplete model for now


import os
import sys

curdir = os.path.dirname(os.path.realpath(__file__))
datdir = f'{curdir}/../../data/'
moddir = f'{curdir}/../../models/'
resdir = f'{curdir}/../../results/'

import numpy             as np
import matplotlib.pyplot as plt
from magrittetorch.model.model import Model
from astropy import constants
from astropy import units


dimension = 1
npoints   = 50
nrays     = 2
nspecs    = 3
nlspecs   = 1
nquads    = 1

nH2  = 1.0E+12*units.m**-3
nTT  = 1.0E+03*units.m**-3
temp = 4.5E+00*units.K
turb = 0.0E+00*units.m/units.s
dx   = 1.0E+12 *units.m
dv   = 0.0E+00 *units.dimensionless_unscaled


modelName = f'all_constant_single_ray'
modelFile = f'{moddir}{modelName}.hdf5'
lamdaFile = f'{datdir}test.txt'


model = Model(modelName)
model.geometry.points.position.set_astropy([[i*dx/units.m, 0, 0] for i in range(npoints)]*units.m)
model.geometry.points.velocity.set_astropy([[i*dv, 0, 0] for i in range(npoints)]*units.dimensionless_unscaled)

model.chemistry.species.abundance.set_astropy([[nTT*units.m**3, nH2*units.m**3, 0.0] for _ in range(npoints)]*units.m**-3)
model.chemistry.species.symbol.set(np.array(['test', 'H2', 'e-'], dtype=np.dtype('S')))

model.thermodynamics.temperature.gas  .set_astropy( temp                 * np.ones(npoints))
model.thermodynamics.turbulence.vturb2.set_astropy((turb/constants.c)**2 * np.ones(npoints))

#TODO: still implement useful setup functions for a lot of input

# model = setup.set_Delaunay_neighbor_lists (model)
# model = setup.set_Delaunay_boundary       (model)
# model = setup.set_boundary_condition_CMB  (model)
# model = setup.set_uniform_rays            (model)
# model = setup.set_linedata_from_LAMDA_file(model, lamdaFile)
# model = setup.set_quadrature              (model)

# model.write()


