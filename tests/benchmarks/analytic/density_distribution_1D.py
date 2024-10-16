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
npoints   = 200#200
nrays     = 10#50
nspecs    = 3
nlspecs   = 1
nquads    = 50#50

r_in   = 1.0E13 * units.m # [m]
r_out  = 7.8E16 * units.m # [m]
nH2_in = 2.0E13 * units.m**-3  # [m^-3]
temp   = 2.0E+01 * units.K # [K]
turb   = 1.5E+02 * units.m/units.s # [m/s]

get_X_mol = {
    'a' : 1.0E-8,
    'b' : 1.0E-6
}

rs = np.logspace (np.log10(r_in/units.m).value, np.log10(r_out/units.m).value, npoints, endpoint=True) * units.m

device = torch.device("cpu")#1D models run best on cpu

def create_and_run_model (a_or_b: str, nosave = False, use_widgets: bool = True) -> None:
    """
    Create a model file for the density distribution benchmark 1D.
    """

    modelName = f'density_distribution_VZ{a_or_b}_1D'
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

    modelName = f'density_distribution_VZ{a_or_b}_1D'
    modelFile = f'{moddir}{modelName}.hdf5'
    timestamp = timer.timestamp()

    X_mol = get_X_mol[a_or_b]


    timer2 = timer.Timer('setting model')
    timer2.start()
    model.dataCollection.infer_data()
    timer2.stop()

    hnrays  = nrays//2

    timer3 = timer.Timer('computing shortchar')
    timer3.start()
    I_computed = torch.zeros((npoints, nlspecs*nquads, nrays))
    for raydiridx in range(model.parameters.nrays.get()):
        raydir = model.geometry.rays.direction.get()[raydiridx, :]
        I_computed[:,:,raydiridx] = solve_long_characteristics_single_direction_all_NLTE_freqs(model, raydir, device)
    timer3.stop()
    u_computed = np.transpose(0.5 * (I_computed[:,:,:hnrays] + I_computed[:,:,hnrays:]).numpy(force=True) * units.J/units.m**2, (2, 0, 1))

    # rs = model.geometry.points.position.get_astropy()[:,0]
    nu = model.sources.lines.lineProducingSpecies[0].get_line_frequencies()[0,:] * units.Hz

    ld = model.sources.lines.lineProducingSpecies[0].linedata

    k = 0

    frq = ld.frequency.get_astropy()[k]
    pop = rtutils.LTEpop         (ld, temp) * X_mol * nH2_in
    phi = rtutils.profile        (ld, k, temp, turb, frq)
    eta = rtutils.lineEmissivity (ld, pop)[k]
    chi = rtutils.lineOpacity    (ld, pop)[k]
    src = rtutils.lineSource     (ld, pop)[k]
    bdy = rtutils.I_CMB          (frq)

    def phi (nu):
        return rtutils.profile (ld, k, temp, turb, nu)

    def bdy (nu):
        return rtutils.I_CMB (nu)

    def tau (nu: Quantity[units.Hz], r: Quantity[units.m], theta) -> Quantity[units.dimensionless_unscaled]:
        pref = chi * phi(nu) * r_in**2
        if (theta == 0.0):
            return pref * (1.0/r_in - 1.0/r)
        if (theta == np.pi):
            return pref * (1.0/r - 1.0/r_out)
        #if intersecting with the inner core
        if (theta < np.arcsin(r_in/r).value):
            closest = r * np.sin(theta)
            in_vertical = np.sqrt(r_in**2 - closest**2)
            out_vertical = np.sqrt(r**2 - closest**2)
            return pref * (np.arctan(out_vertical/closest).value - np.arctan(in_vertical/closest).value) / closest
        else:
            #Other derivation: results in exactly the same error, so should be equivalent
            # closest = r * np.sin(theta)
            # out_vertical = np.sqrt(r_out**2 - closest**2)
            # start_vertical = np.sqrt(r**2 - closest**2)
            # if np.abs(theta)<np.pi/2:
            #     return pref * (np.arctan(out_vertical/closest) + np.arctan(start_vertical/closest)) / closest
            # else:
            #     return pref * (np.arctan(out_vertical/closest) - np.arctan(start_vertical/closest)) / closest

            #Old derivation
            factor = np.arccos(r*np.sin(theta)/r_out).value + 0.5*np.pi - theta
            return pref / (r*np.sin(theta)) * factor

    def I_ (nu, r, theta):
        return src + (bdy(nu)-src)*np.exp(-tau(nu, r, theta))

    def u_ (nu, r, theta):
        return 0.5 * (I_(nu, r, theta.value) + I_(nu, r, np.pi+theta.value))

    rx, ry, rz = model.geometry.rays.direction.get_astropy().T
    angles     = np.arctan2(ry,rx)
    angles     = angles[:hnrays]

    us = np.array([[[u_(f,r,a)/units.J*units.m**2 for f in nu] for r in rs] for a in angles])*units.J/units.m**2

    fs = (nu-frq)/frq*constants.c
    #hmm, do we need this widget if we are not plotting it?
    def plot (r, p):
        plt.figure(dpi=150)
        plt.plot(fs, us  [r,p,:], marker='.')
        plt.plot(fs, u_computed[r,p,:])
        # plt.plot(fs, u_0s[r,p,:])

    #during automated testing, the widgets only consume time to create
    if use_widgets:
        widgets.interact(plot, r=(0,hnrays-1,1), p=(0,npoints-1,1))


    error_u_computed = np.abs(rtutils.relative_error(us, u_computed)[:-1, :-1, :])
    # error_u_2f = np.abs(rtutils.relative_error(us, u_2f)[:-1, :-1, :])

    log_err_min = np.log10(np.min(error_u_computed))
    log_err_max = np.log10(np.max(error_u_computed))

    bins = np.logspace(log_err_min.value, log_err_max.value, 100)

    result  = f'--- Benchmark name ------------------------------\n'
    result += f'{modelName                                      }\n'
    result += f'--- Parameters ----------------------------------\n'
    # result += f'dimension = {model.parameters.dimension()       }\n'
    result += f'npoints   = {model.parameters.npoints.get()       }\n'
    result += f'nrays     = {model.parameters.nrays.get()       }\n'
    # result += f'nquads    = {model.parameters.nquads   ()       }\n'
    result += f'--- Accuracy ------------------------------------\n'
    result += f'mean error in shortchar 0 = {np.mean(error_u_computed)}\n'
    # result += f'mean error in feautrier 2 = {np.mean(error_u_2f)}\n'
    result += f'--- Timers --------------------------------------\n'
    # result += f'{timer1.print()                                 }\n'
    result += f'{timer2.print()                                 }\n'
    result += f'{timer3.print()                                 }\n'
    # result += f'{timer4.print()                                 }\n'
    result += f'-------------------------------------------------\n'

    print(result)

    if not nosave:
        with open(f'{resdir}{modelName}-{timestamp}.log' ,'w') as log:
            log.write(result)

        plt.figure()
        plt.title(modelName)
        plt.hist(error_u_computed.ravel(), bins=bins, histtype='step', label='0s')
        # plt.hist(error_u_2f.ravel(), bins=bins, histtype='step', label='2f')
        plt.xscale('log')
        plt.legend()
        plt.savefig(f'{resdir}{modelName}_hist-{timestamp}.png', dpi=150)

        plt.figure()
        plt.hist(error_u_computed.ravel(), bins=bins, histtype='step', label='0s', cumulative=True)
        # plt.hist(error_u_2f.ravel(), bins=bins, histtype='step', label='2f', cumulative=True)
        plt.xscale('log')
        plt.legend()
        plt.savefig(f'{resdir}{modelName}_cumu-{timestamp}.png', dpi=150)

    #setting 'b' not yet used for testing
    if a_or_b == 'a':
        # error bounds are chosen somewhat arbitrarily, based on previously obtained results; this should prevent serious regressions.
        FIRSTORDER_AS_EXPECTED=(np.mean(error_u_computed)<4.4e-4)

        if not FIRSTORDER_AS_EXPECTED:
            print("First order solver mean error too large: ", np.mean(error_u_computed))

        return FIRSTORDER_AS_EXPECTED

    return


def run_test (nosave=False, use_widgets=True):

    create_and_run_model ('a', nosave, use_widgets)

    # create_and_run_model ('b', nosave, use_widgets)


if __name__ == '__main__':

    nosave = (len(sys.argv) > 1) and (sys.argv[1] == 'nosave')

    run_test (nosave)
