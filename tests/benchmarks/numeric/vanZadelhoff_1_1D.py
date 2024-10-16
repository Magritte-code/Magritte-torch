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
import magrittetorch.utils.constants as magritte_constants
import magrittetorch.tools.setup as setup
from magrittetorch.model.geometry.geometry import GeometryType
from magrittetorch.model.geometry.boundary import BoundaryType
import magrittetorch.tools.timer as timer
import magrittetorch.tools.radiativetransferutils as rtutils
from magrittetorch.algorithms.solvers import compute_level_populations
from scipy.interpolate import interp1d

#Note: compared to C++ magritte, we use a more sparse discretization for these benchmarks to speed up the tests
# this does slightly affect the accuracy, but the benchmarks can still run manually with more resolution to be compared to the C++ results 

magritte_constants.min_line_opacity = 1e-13

dimension = 1
npoints   = 100
nrays     = 40
nspecs    = 3
nlspecs   = 1
nquads    = 11

r_in   = 1.0E13 * units.m   # [m]
r_out  = 7.8E16 * units.m   # [m]
nH2_in = 2.0E13 / units.m**3   # [m^-3]
temp   =  20.00 * units.K   # [K]
turb   = 150.00 * units.m/units.s # [m/s]

get_X_mol = {
    'a' : 1.0E-8,
    'b' : 1.0E-6
}

rs = np.logspace (np.log10(r_in/units.m).value, np.log10(r_out/units.m).value, npoints, endpoint=True) * units.m


def create_and_run_model (a_or_b, nosave = False):
    """
    Create a model file for the van Zadelhoff 1 benchmark in 1D.
    """

    modelName = f'vanZadelhoff_1{a_or_b}_1D'
    modelFile = f'{moddir}{modelName}.hdf5'
    lamdaFile = f'{datdir}test.txt'

    X_mol = get_X_mol[a_or_b]

    def nH2 (r: Quantity[units.m]) -> Quantity[units.m**-3]:
        return nH2_in * np.power(r_in/r, 2.0)

    def nTT (r: Quantity[units.m]) -> Quantity[units.m**-3]:
        return X_mol  * nH2(r)
    
    model = magritte.Model (modelFile)
    model.geometry.geometryType.set(GeometryType.SpericallySymmetric1D)
    model.geometry.boundary.boundaryType.set(BoundaryType.Sphere1D)

    model.geometry.points.position.set_astropy([[r/units.m, 0, 0] for r in rs]*units.m)
    model.geometry.points.velocity.set_astropy([[0, 0, 0] for i in range(npoints)]*units.m/units.s)

    model.chemistry.species.abundance.set_astropy([[nTT(r)*units.m**3, nH2(r)*units.m**3,  0.0] for r in rs]/units.m**3)
    model.chemistry.species.symbol.set(np.array(['test', 'H2', 'e-'], dtype='S'))

    model.thermodynamics.temperature.gas.set_astropy(temp * np.ones(npoints))
    model.thermodynamics.turbulence.vturb.set_astropy(turb * np.ones(npoints))

    model = setup.set_Delaunay_neighbor_lists (model)
    model = setup.set_Delaunay_boundary       (model)
    model = setup.set_boundary_condition_CMB  (model)
    model = setup.set_rays_spherical_symmetry (model, nrays)
    model = setup.set_linedata_from_LAMDA_file(model, lamdaFile)
    model = setup.set_quadrature              (model, nquads)    

    timestamp = timer.timestamp()

    timer1 = timer.Timer('setting model')
    timer1.start()
    model.dataCollection.infer_data()
    timer1.stop()

    device = torch.device("cpu")#1D models run best on cpu

    timer2 = timer.Timer('running model')
    timer2.start()
    # compute_level_populations(model, device, 10, use_ng_acceleration=True, use_ALI=True, max_its_between_ng_accel=8)
    compute_level_populations(model, device, 50, use_ng_acceleration=True, use_ALI=True, max_its_between_ng_accel=8)
    timer2.stop()

    pops = model.sources.lines.lineProducingSpecies[0].population.get_astropy()
    abun = model.chemistry.species.abundance.get_astropy()[:,0]

    (i,ra,rb,nh,tk,nm,vr,db,td,lp0,lp1) = np.loadtxt (f'{curdir}/Ratran_results/vanZadelhoff_1{a_or_b}.out', skiprows=14, unpack=True)

    interp_0 = interp1d(0.5*(ra+rb), lp0, fill_value='extrapolate')
    interp_1 = interp1d(0.5*(ra+rb), lp1, fill_value='extrapolate')

    error_0 = np.abs(rtutils.relative_error(pops[:,0]/abun, interp_0(rs)))
    error_1 = np.abs(rtutils.relative_error(pops[:,1]/abun, interp_1(rs)))

    result  = f'--- Benchmark name -----------------------\n'
    result += f'{modelName                               }\n'
    result += f'--- Parameters ---------------------------\n'
    result += f'dimension = {dimension}\n'
    result += f'npoints   = {model.parameters.npoints.get()}\n'
    result += f'nrays     = {model.parameters.nrays.get()}\n'
    result += f'nquads    = {nquads}\n'
    result += f'--- Accuracy -----------------------------\n'
    result += f'max error in (0) = {np.max(error_0[1:])  }\n'
    result += f'max error in (1) = {np.max(error_1[1:])  }\n'
    result += f'--- Timers -------------------------------\n'
    result += f'{timer1.print()                          }\n'
    result += f'{timer2.print()                          }\n'
    result += f'------------------------------------------\n'

    print(result)

    if not nosave:
        with open(f'{resdir}{modelName}-{timestamp}.log' ,'w') as log:
            log.write(result)

        plt.figure(dpi=150)
        plt.title(modelName)
        plt.scatter(rs, pops[:,0]/abun, s=0.5, label='i=0', zorder=1)
        plt.scatter(rs, pops[:,1]/abun, s=0.5, label='i=1', zorder=1)
        plt.plot(ra, lp0, c='lightgray', zorder=0)
        plt.plot(ra, lp1, c='lightgray', zorder=0)
        plt.legend()
        plt.xscale('log')
        plt.xlabel('r [m]')
        plt.ylabel('fractional level populations [.]')
        plt.savefig(f'{resdir}{modelName}-{timestamp}.png', dpi=150)

    #maximal error lies mainly in the regime change
    #bound valid for for only benchmark a; b converges a bit slower
    if a_or_b == 'a':
        FEAUTRIER_AS_EXPECTED=((np.max(error_0[1:])<0.073)&(np.max(error_1[1:])<0.143))

        if not FEAUTRIER_AS_EXPECTED:
            print("Feautrier solver max error too large; [0]:", np.max(error_0[1:]), " [1]:", np.max(error_1[1:]))

        return (FEAUTRIER_AS_EXPECTED)

def run_test (nosave=False):

    create_and_run_model ('a')
    create_and_run_model ('b')

    return


if __name__ == '__main__':

    nosave = (len(sys.argv) > 1) and (sys.argv[1] == 'nosave')

    run_test (nosave)
