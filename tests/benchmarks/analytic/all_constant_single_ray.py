import os
import sys

curdir = os.path.dirname(os.path.realpath(__file__))
datdir = f'{curdir}/../../data/'
moddir = f'{curdir}/../../models/'
resdir = f'{curdir}/../../results/'

import numpy             as np
import torch
import matplotlib.pyplot as plt
import magrittetorch.model.model as magritte
import astropy.constants as constants
import astropy.units as units
from astropy.units import Quantity
import magrittetorch.tools.setup as setup
from magrittetorch.model.geometry.geometry import GeometryType
from magrittetorch.model.geometry.boundary import BoundaryType
import magrittetorch.tools.timer as timer
import magrittetorch.tools.radiativetransferutils as rtutils
from magrittetorch.algorithms.solvers import solve_long_characteristics_NLTE#TODO: add single iteration version that return mean (line) intensity
# import magritte.tools    as tools
# import magritte.setup    as setup
# import magritte.core     as magritte


dimension = 1
npoints   = 50
nrays     = 2
nspecs    = 3
nlspecs   = 1
nquads    = 1

nH2  = 1.0E+12 * units.m **-3                 # [m^-3]
nTT  = 1.0E+03 * units.m **-3                 # [m^-3]
temp = 4.5E+00 * units.K                # [K]
turb = 0.0E+00 * units.m / units.s                # [m/s]
print("turb", turb)
dx   = 1.0E+12 * units.m                # [m]
dv   = 0.0E+00 * units.m / units.s


def create_model ():
    """
    Create a model file for the all_constant benchmark, single ray.
    """

    modelName = f'all_constant_single_ray'
    modelFile = f'{moddir}{modelName}.hdf5'
    lamdaFile = f'{datdir}test.txt'


    model = magritte.Model(modelFile)
    model.geometry.geometryType.set(GeometryType.General3D)
    model.geometry.boundary.boundaryType.set(BoundaryType.AxisAlignedCube)

    model.geometry.points.position.set_astropy([[i, 0, 0] for i in range(npoints)]*dx)
    model.geometry.points.velocity.set_astropy([[i, 0, 0] for i in range(npoints)]*dv)
    model.chemistry.species.abundance.set_astropy([[nTT*units.m**3, nH2*units.m**3, 0.0] for _ in range(npoints)]*units.m**-3)
    model.chemistry.species.symbol.set(np.array(['test', 'H2', 'e-'], dtype='S'))

    model.thermodynamics.temperature.gas.set_astropy(temp * np.ones(npoints))
    model.thermodynamics.turbulence.vturb.set_astropy(turb * np.ones(npoints))

    model = setup.set_Delaunay_neighbor_lists (model)
    model = setup.set_Delaunay_boundary       (model)
    model = setup.set_boundary_condition_CMB  (model)
    model = setup.set_uniform_rays            (model, nrays)
    model = setup.set_linedata_from_LAMDA_file(model, lamdaFile)
    model = setup.set_quadrature              (model, nquads)

    model.write()

    return 


def run_model (nosave=False):

    modelName = f'all_constant_single_ray'
    modelFile = f'{moddir}{modelName}.hdf5'
    timestamp = timer.timestamp()

    timer1 = timer.Timer('reading model')
    timer1.start()
    model = magritte.Model(modelFile)
    model.read()
    timer1.stop()

    # print(model.geometry.boundary.boundary_condition.get())

    timer2 = timer.Timer('setting model')
    timer2.start()
    model.dataCollection.infer_data()
    timer2.stop()

    timer3 = timer.Timer('shortchar 0  ')
    timer3.start()
    J_0s = solve_long_characteristics_NLTE(model, device=torch.device("cpu")).numpy(force=True) * units.J/units.m**2
    timer3.stop()

    x = model.geometry.points.position.get_astropy()[:,0]
    nu = model.sources.lines.lineProducingSpecies[0].linedata.frequency.get_astropy()[0]

    ld = model.sources.lines.lineProducingSpecies[0].linedata

    k = 0

    frq = ld.frequency.get_astropy()[k]
    pop = rtutils.LTEpop(ld, temp) * nTT
    phi = rtutils.profile(ld, k, temp, turb, frq)
    eta = rtutils.lineEmissivity(ld, pop)[k] * phi
    chi = rtutils.lineOpacity(ld, pop)[k] * phi
    src = rtutils.lineSource(ld, pop)[k]
    bdy = rtutils.I_CMB(frq)

    def I_0 (x: Quantity[units.m]) -> Quantity[units.J/units.m**2]:
        return src + (bdy-src)*np.exp(-chi*x)

    def I_1 (x: Quantity[units.m]) -> Quantity[units.J/units.m**2]:
        return src + (bdy-src)*np.exp(-chi*(x[-1]-x))

    def u_ (x: Quantity[units.m]) -> Quantity[units.J/units.m**2]:
        return 0.5 * (I_0(x) + I_1(x))
    
    #note: for the case of a single frequency: J==u

    error_u_0s = np.abs(rtutils.relative_error (u_(x), J_0s[:,0]))

    result  = f'--- Benchmark name ----------------------------\n'
    result += f'{modelName                                    }\n'
    result += f'--- Parameters --------------------------------\n'
    result += f'dimension = 1\n'
    result += f'npoints   = {model.parameters.npoints.get()     }\n'
    result += f'nrays     = {model.parameters.nrays.get()     }\n'
    result += f'nquads    = {model.sources.lines.lineProducingSpecies[0].linequadrature.nquads.get()}\n'
    result += f'--- Accuracy ----------------------------------\n'
    result += f'max error in shortchar 0 = {np.max(error_u_0s)}\n'
    result += f'--- Timers ------------------------------------\n'
    result += f'{timer1.print()                               }\n'
    result += f'{timer2.print()                               }\n'
    result += f'{timer3.print()                               }\n'
    result += f'-----------------------------------------------\n'

    print(result)

    if not nosave:
        with open(f'{resdir}{modelName}-{timestamp}.log' ,'w') as log:
            log.write(result)

        plt.figure(dpi=150)
        plt.title(modelName)
        plt.scatter(x, J_0s[:,0], s=0.5, label='0s', zorder=1)
        plt.plot(x, u_(x), c='lightgray', zorder=0)
        plt.legend()
        plt.xscale('log')
        plt.xlabel('r [m]')
        plt.ylabel('Mean intensity [W/m$^{2}$]')
        plt.savefig(f'{resdir}{modelName}-{timestamp}.png', dpi=150)


    #error bounds are chosen somewhat arbitrarily, based on previously obtained results; this should prevent serious regressions.
    FIRSTORDER_AS_EXPECTED=(np.max(error_u_0s)<5.5e-7)#within order of magnitude as precision of 32bit float

    if not FIRSTORDER_AS_EXPECTED:
        print("First order solver max error too large: ", np.max(error_u_0s))

    return FIRSTORDER_AS_EXPECTED


def run_test (nosave=False):

    create_model ()
    run_model    (nosave)

    return


if __name__ == '__main__':

    nosave = (len(sys.argv) > 1) and (sys.argv[1] == 'nosave')

    run_test (nosave)
