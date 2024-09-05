import os
import sys

curdir = os.path.dirname(os.path.realpath(__file__))
datdir = f'{curdir}/../../data/'
moddir = f'{curdir}/../../models/'
resdir = f'{curdir}/../../results/'

import numpy             as np
import torch
import matplotlib.pyplot as plt
import scipy as sp
import magrittetorch.model.model as magritte
import ipywidgets        as widgets
import astropy.constants as constants
import astropy.units as units
from astropy.units import Quantity
import magrittetorch.tools.setup as setup
from magrittetorch.model.geometry.geometry import GeometryType
from magrittetorch.model.geometry.boundary import BoundaryType
import magrittetorch.tools.timer as timer
import magrittetorch.tools.radiativetransferutils as rtutils
from magrittetorch.algorithms.solvers import solve_long_characteristics_single_direction_all_NLTE_freqs#TODO: add single iteration version that return mean (line) intensity
import magrittetorch.utils.constants as magritte_constants

magritte_constants.min_line_opacity = 1e-13


dimension = 1
npoints   = 200
nrays     = 2
nspecs    = 3
nlspecs   = 1
nquads    = 1

r_in   = 1.0E13 * units.m   # [m]
r_out  = 7.8E16 * units.m   # [m]
nH2_in: Quantity = 2.0E13 * units.m**-3  # [m^-3]
temp   = 2.0E+01 * units.K # [K]
turb   = 1.5E+02 * units.m/units.s # [m/s]

get_X_mol = {
    'a' : 1.0E-8,
    'b' : 1.0E-6
}

rs = np.logspace (np.log10(r_in/units.m).value, np.log10(r_out/units.m).value, npoints, endpoint=True) * units.m

device = torch.device("cpu")#1D models run best on cpu

def create_and_run_model (a_or_b: str) -> None:
    """
    Create a model file for the density distribution benchmark, single ray.
    """

    modelName = f'density_distribution_VZ{a_or_b}_single_ray'
    modelFile = f'{moddir}{modelName}.hdf5'
    lamdaFile = f'{datdir}test.txt'

    X_mol = get_X_mol[a_or_b]

    def nH2 (r: Quantity) -> Quantity:
        return nH2_in * np.power(r_in/r, 2.0)

    def nTT (r: Quantity) -> Quantity:
        return X_mol  * nH2(r)

    model = magritte.Model(modelFile)
    model.geometry.geometryType.set(GeometryType.General3D)
    model.geometry.boundary.boundaryType.set(BoundaryType.AxisAlignedCube)

    model.geometry.points.position.set_astropy([[r/units.m, 0, 0] for r in rs]*units.m)
    model.geometry.points.velocity.set_astropy([[0, 0, 0] for i in range(npoints)]*units.m/units.s)

    model.chemistry.species.abundance.set_astropy([[nTT(r)*units.m**3, nH2(r)*units.m**3, 0.0] for r in rs]/units.m**3)
    model.chemistry.species.symbol.set(np.array(['test', 'H2', 'e-'], dtype='S'))

    model.thermodynamics.temperature.gas.set_astropy(temp * np.ones(npoints))
    model.thermodynamics.turbulence.vturb.set_astropy(turb * np.ones(npoints))

    model = setup.set_Delaunay_neighbor_lists (model)
    model = setup.set_Delaunay_boundary       (model)
    model = setup.set_boundary_condition_CMB  (model)
    model = setup.set_uniform_rays            (model, nrays)
    model = setup.set_linedata_from_LAMDA_file(model, lamdaFile)
    model = setup.set_quadrature              (model, nquads)

#     model.write()

#     return magritte.Model (modelFile)


# def run_model (a_or_b, nosave=False):

    modelName = f'density_distribution_VZ{a_or_b}_single_ray'
    modelFile = f'{moddir}{modelName}.hdf5'
    timestamp = timer.timestamp()

    X_mol = get_X_mol[a_or_b]

    # timer1 = timer.Timer('reading model')
    # timer1.start()
    # model = magritte.Model (modelFile)
    # timer1.stop()

    timer2 = timer.Timer('setting model')
    timer2.start()
    model.dataCollection.infer_data()
    timer2.stop()

    timer3 = timer.Timer('computing shortchar')
    timer3.start()
    I_computed = torch.zeros((npoints, nlspecs*nquads, nrays))
    for raydiridx in range(model.parameters.nrays.get()):
        raydir = model.geometry.rays.direction.get()[raydiridx, :]
        I_computed[:,:,raydiridx] = solve_long_characteristics_single_direction_all_NLTE_freqs(model, raydir, device)
    timer3.stop()
    u_computed = 0.5 * torch.sum(I_computed, dim=2).numpy(force=True) * units.J/units.m**2


    x  = model.geometry.points.position.get_astropy()[:,0]
    nu = model.sources.lines.lineProducingSpecies[0].get_line_frequencies()[0,:] * units.Hz

    ld = model.sources.lines.lineProducingSpecies[0].linedata

    k = 0

    frq = ld.frequency.get_astropy()[k]
    pop = rtutils.LTEpop         (ld, temp) * X_mol * nH2_in
    phi = rtutils.profile        (ld, k, temp, turb, frq)
    eta = rtutils.lineEmissivity (ld, pop)[k] * phi
    chi = rtutils.lineOpacity    (ld, pop)[k] * phi
    src = rtutils.lineSource     (ld, pop)[k]
    bdy = rtutils.I_CMB          (frq)

    def I_0 (x: Quantity[units.m]) -> Quantity[units.J/units.m**2]:
        return src + (bdy-src)*np.exp(-chi*r_in*(1.0    - r_in/x    ))

    def I_1 (x: Quantity[units.m]) -> Quantity[units.J/units.m**2]:
        return src + (bdy-src)*np.exp(-chi*r_in*(r_in/x - r_in/r_out))

    def u_ (x: Quantity[units.m]) -> Quantity[units.J/units.m**2]:
        return 0.5 * (I_0(x) + I_1(x))

    print(u_computed, u_(x), u_computed.shape, u_(x).shape)
    error_u_0s = np.abs(rtutils.relative_error (u_(x), u_computed[:,0]))
    # error_u_2f = np.abs(rtutils.relative_error (u_(x), u_2f[0,:,0]))

    result  = f'--- Benchmark name ----------------------------\n'
    result += f'{modelName                                    }\n'
    result += f'--- Parameters --------------------------------\n'
    # result += f'dimension = {model.parameters.dimension()     }\n'
    result += f'npoints   = {model.parameters.npoints.get()     }\n'
    result += f'nrays     = {model.parameters.nrays.get()     }\n'
    # result += f'nquads    = {model.parameters.nquads   ()     }\n'
    result += f'--- Accuracy ----------------------------------\n'
    result += f'max error in shortchar 0 = {np.max(error_u_0s)}\n'
    result += f'mean error in shortchar 0 = {np.mean(error_u_0s)}\n'
    # result += f'max error in feautrier 2 = {np.max(error_u_2f)}\n'
    # result += f'mean error in feautrier 2 = {np.mean(error_u_2f)}\n'
    result += f'--- Timers ------------------------------------\n'
    # result += f'{timer1.print()                               }\n'
    result += f'{timer2.print()                               }\n'
    result += f'{timer3.print()                               }\n'
    # result += f'{timer4.print()                               }\n'
    result += f'-----------------------------------------------\n'

    print(result)

    if not nosave:
        with open(f'{resdir}{modelName}-{timestamp}.log' ,'w') as log:
            log.write(result)

        plt.figure(dpi=150)
        plt.title(modelName)
        plt.scatter(x, u_computed[:,0], s=0.5, label='0s', zorder=1)
        # plt.scatter(x, u_2f[0,:,0], s=0.5, label='2f', zorder=1)
        plt.plot(x, u_(x), c='lightgray', zorder=0)
        plt.legend()
        plt.xscale('log')
        plt.xlabel('r [m]')
        plt.ylabel('Mean intensity [W/m$^{2}$]')
        plt.savefig(f'{resdir}{modelName}-{timestamp}.png', dpi=150)

    return


def run_test (nosave=False) -> None:

    create_and_run_model ('a')
    # run_model    ('a', nosave)

    create_and_run_model ('b')
    # run_model    ('b', nosave)


if __name__ == '__main__':

    nosave = (len(sys.argv) > 1) and (sys.argv[1] == 'nosave')

    run_test (nosave)
