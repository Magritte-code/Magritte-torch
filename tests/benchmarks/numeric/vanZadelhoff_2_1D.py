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
from magrittetorch.algorithms.solvers import compute_level_populations
from scipy.interpolate import interp1d

def create_and_run_model (a_or_b, nosave = False):
    """
    Create a model file for the van Zadelhoff 2 benchmark in 1D.
    """

    modelName = f'vanZadelhoff_2{a_or_b}_1D'
    modelFile = f'{moddir}{modelName}.hdf5'
    lamdaFile = f'{datdir}hco+.txt'

    # Read input file
    rs, nH2, X_mol, temp, vs, turb = np.loadtxt(f'vanZadelhoff_2{a_or_b}.in', skiprows=7, unpack=True)

    # Parameters
    dimension = 1
    npoints   = len(rs)
    nrays     = 20#200
    nspecs    = 3
    nlspecs   = 1
    nquads    = 11

    # Convert units to SI
    rs   = rs   * 1.0E-2 * units.m
    nH2  = nH2  * 1.0E+6 / units.m**3
    vs   = vs   * 1.0E+3 * units.m/units.s
    turb = turb * 1.0E+3 * units.m/units.s
    temp = temp * units.K

    # Put radii in ascending order
    rs    = np.flip (rs,    axis=0)
    nH2   = np.flip (nH2,   axis=0)
    X_mol = np.flip (X_mol, axis=0)
    temp  = np.flip (temp,  axis=0)
    vs    = np.flip (vs,    axis=0)
    turb  = np.flip (turb,  axis=0)

    model = magritte.Model (modelFile)
    model.geometry.geometryType.set(GeometryType.SpericallySymmetric1D)
    model.geometry.boundary.boundaryType.set(BoundaryType.Sphere1D)

    model.geometry.points.position.set_astropy([[r/units.m, 0, 0] for r in rs]*units.m)
    model.geometry.points.velocity.set_astropy([[v/units.m*units.s, 0, 0] for v in vs]*units.m/units.s)

    model.chemistry.species.abundance.set_astropy([[x*n*units.m**3, n*units.m**3,  0.0] for (x,n) in zip(X_mol, nH2)]/units.m**3)
    model.chemistry.species.symbol.set(np.array(['HCO+', 'H2', 'e-'], dtype='S'))

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

    device = torch.device('cpu')#1D models run best on cpu

    timer2 = timer.Timer('running model')
    timer2.start()
    compute_level_populations(model, device, 50)
    timer2.stop()

    npoints = model.parameters.npoints.get()
    nlev    = model.sources.lines.lineProducingSpecies[0].linedata.nlev.get()

    pops = model.sources.lines.lineProducingSpecies[0].population.get_astropy().reshape((npoints, nlev))
    abun = model.chemistry.species.abundance.get_astropy()[:,0]
    rs   = np.linalg.norm(model.geometry.points.position.get_astropy(), axis=1)

    (i,ra,rb,nh,tk,nm,vr,db,td,lp0,lp1,lp2,lp3,lp4) = np.loadtxt (f'Ratran_results/vanZadelhoff_2{a_or_b}.out', skiprows=14, unpack=True, usecols=range(14))

    interp_0 = interp1d(0.5*(ra+rb), lp0, fill_value='extrapolate')
    interp_1 = interp1d(0.5*(ra+rb), lp1, fill_value='extrapolate')
    interp_2 = interp1d(0.5*(ra+rb), lp2, fill_value='extrapolate')
    interp_3 = interp1d(0.5*(ra+rb), lp3, fill_value='extrapolate')
    interp_4 = interp1d(0.5*(ra+rb), lp4, fill_value='extrapolate')

    error_0 = np.abs(rtutils.relative_error(pops[:,0]/abun, interp_0(rs)))
    error_1 = np.abs(rtutils.relative_error(pops[:,1]/abun, interp_1(rs)))
    error_2 = np.abs(rtutils.relative_error(pops[:,2]/abun, interp_2(rs)))
    error_3 = np.abs(rtutils.relative_error(pops[:,3]/abun, interp_3(rs)))
    error_4 = np.abs(rtutils.relative_error(pops[:,4]/abun, interp_4(rs)))

    result  = f'--- Benchmark name -----------------------\n'
    result += f'{modelName                               }\n'
    result += f'--- Parameters ---------------------------\n'
    result += f'dimension = {dimension}\n'
    result += f'npoints   = {model.parameters.npoints.get()}\n'
    result += f'nrays     = {nrays}\n'
    result += f'nquads    = {nquads}\n'
    result += f'--- Accuracy -----------------------------\n'
    result += f'max error in (0) = {np.max(error_0[1:])  }\n'
    result += f'max error in (1) = {np.max(error_1[1:])  }\n'
    result += f'max error in (2) = {np.max(error_2[1:])  }\n'
    result += f'max error in (3) = {np.max(error_3[1:])  }\n'
    result += f'max error in (4) = {np.max(error_4[1:])  }\n'
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
        plt.scatter(rs, pops[:,2]/abun, s=0.5, label='i=2', zorder=1)
        plt.scatter(rs, pops[:,3]/abun, s=0.5, label='i=3', zorder=1)
        plt.scatter(rs, pops[:,4]/abun, s=0.5, label='i=4', zorder=1)
        plt.plot(ra, lp0, c='lightgray', zorder=0)
        plt.plot(ra, lp1, c='lightgray', zorder=0)
        plt.plot(ra, lp2, c='lightgray', zorder=0)
        plt.plot(ra, lp3, c='lightgray', zorder=0)
        plt.plot(ra, lp4, c='lightgray', zorder=0)
        plt.legend()
        plt.xscale('log')
        plt.xlabel('r [m]')
        plt.ylabel('fractional level populations [.]')
        plt.savefig(f'{resdir}{modelName}-{timestamp}.png', dpi=150)

    return


def run_test (nosave=False):

    create_and_run_model ('a')
    create_and_run_model ('b')

    return


if __name__ == '__main__':

    nosave = (len(sys.argv) > 1) and (sys.argv[1] == 'nosave')

    run_test (nosave)
