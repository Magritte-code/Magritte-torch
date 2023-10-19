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

#limited parameter settings for benchmarking, compared to C++ Magritte, as 1D is excruciatingly slow
dimension = 1
npoints   = 20#100
nrays     = 10#100
nspecs    = 3
nlspecs   = 1
nquads    = 50#100

nH2  = 1.0E+12 * units.m **-3
nTT  = 1.0E+08 * units.m **-3
temp = 4.5E+01 * units.K
turb = 0.0E+00 * units.m/units.s
dx   = 1.0E+04 * units.m
dv   = 2.5E+02 * units.m/units.s

L    = dx*npoints
vmax = dv*npoints


def create_and_run_model (nosave = False, use_widgets = True):
    """
    Create a model file for the constant velocity gradient benchmark 1D.
    """

    modelName = f'constant_velocity_gradient_1D'
    modelFile = f'{moddir}{modelName}.hdf5'
    lamdaFile = f'{datdir}test.txt'

    model = magritte.Model (modelFile)
    model.geometry.geometryType.set(GeometryType.SpericallySymmetric1D)
    model.geometry.boundary.boundaryType.set(BoundaryType.Sphere1D)

    model.geometry.points.position.set_astropy([[(i+1), 0, 0] for i in range(npoints)]*dx)
    model.geometry.points.velocity.set_astropy([[(i+1), 0, 0] for i in range(npoints)]*dv)


    model.chemistry.species.abundance.set_astropy([[nTT*units.m**3, nH2*units.m**3] for _ in range(npoints)]*units.m**-3)
    model.chemistry.species.symbol.set(np.array(['test', 'H2'], dtype='S'))

    model.thermodynamics.temperature.gas.set_astropy(temp * np.ones(npoints))
    model.thermodynamics.turbulence.vturb.set_astropy(turb * np.ones(npoints))

    model = setup.set_Delaunay_neighbor_lists (model)
    model = setup.set_Delaunay_boundary       (model)
    model = setup.set_boundary_condition_CMB  (model)
    model = setup.set_rays_spherical_symmetry (model, nrays)
    model = setup.set_linedata_from_LAMDA_file(model, lamdaFile)
    model = setup.set_quadrature              (model, nquads)


    modelName = f'constant_velocity_gradient_1D'
    modelFile = f'{moddir}{modelName}.hdf5'
    timestamp = timer.timestamp()

    #for controlling accuracy of optical depth computation during benchmark runs
    #TODO: create once again a parameter one can set for this; currently hardcoded in lines.py
    # if (benchindex==1):
    #     model.parameters.max_width_fraction=0.5;#fast, but slightly inaccurate
    # elif (benchindex==2):
    #     model.parameters.max_width_fraction=0.35;#more reasonably accurate, but minor performance hit
    # elif (benchindex==3):
    #     model.parameters.max_width_fraction=0.10;#major performance hit, very accurate


    timer2 = timer.Timer('setting model')
    timer2.start()
    model.dataCollection.infer_data()
    timer2.stop()

    device = torch.device("cpu")#1D models run best on cpu

    timer3 = timer.Timer('computing shortchar')
    timer3.start()
    I_computed = torch.zeros((npoints, nlspecs*nquads, nrays))
    for raydiridx in range(model.parameters.nrays.get()):
        raydir = model.geometry.rays.direction.get()[raydiridx, :]
        I_computed[:,:,raydiridx] = solve_long_characteristics_single_direction_all_NLTE_freqs(model, raydir, device)
    timer3.stop()
    I_computed = I_computed.numpy(force = True) * units.J/units.m**2
    u_computed = np.transpose((I_computed[:,:,:nrays//2] + I_computed[:,:,nrays//2:])/2, (2,0,1))


    rs = model.geometry.points.position.get_astropy()[:,0]
    nu = model.sources.lines.lineProducingSpecies[0].get_line_frequencies()[0,:] * units.Hz
    #same frequency for all points

    print("nu", nu)

    ld = model.sources.lines.lineProducingSpecies[0].linedata

    k = 0

    frq = ld.frequency.get_astropy()[k]
    pop = rtutils.LTEpop         (ld, temp) * nTT
    eta = rtutils.lineEmissivity (ld, pop)[k]
    chi = rtutils.lineOpacity    (ld, pop)[k]
    src = rtutils.lineSource     (ld, pop)[k]
    dnu = rtutils.dnu            (ld, k, temp, turb)

    r_in  = rs[ 0]
    r_out = rs[-1]

    def bdy (nu: Quantity, r: Quantity, theta) -> Quantity:
        # print(theta, np.arcsin(r_in/r))
        if (theta < np.arcsin(r_in/r).value):#inside bdy condition
            # shift at position r is quite simple (just cos(θ)*v)
            r_shift=np.cos(theta)*v_(r)
            #for computing the shift at the edge, one need to compute first the closest distance to the center (of the ray); this is L=r*sin(θ)
            #then one realizes that the sin of the angle α one needs corresponds with L/r_out (so cos(α)=√(1-sin^2(α))
            #finally one evidently needs to multiply with the velocity at the boundary
            r_in_shift=np.sqrt(1- ( (r/r_in)*np.abs(np.sin(theta)) )**2)*v_(r_in)
            nu_shifted=nu* (1.0+(r_in_shift - r_shift)/constants.c)
            return rtutils.I_CMB (nu_shifted)
        else:#outside bdy condition
            # shift at position r is quite simple (just cos(θ)*v)
            r_shift=np.cos(theta)*v_(r)
            #for computing the shift at the edge, one need to compute first the closest distance to the center (of the ray); this is L=r*sin(θ)
            #then one realizes that the sin of the angle α one needs corresponds with L/r_out (so cos(α)=√(1-sin^2(α))
            #finally one evidently needs to multiply with the velocity at the boundary
            r_out_shift=np.sqrt(1- ( (r/r_out)*np.abs(np.sin(theta)) )**2)*v_(r_out)
            nu_shifted=nu* (1.0+(r_out_shift - r_shift)/constants.c)
            return rtutils.I_CMB (nu_shifted)

    def z_max(r: Quantity, theta) -> Quantity:
        # print(theta, np.arcsin(r_in/r), theta < np.arcsin(r_in/r))
        if (theta < np.arcsin(r_in/r).value):
            # print(r_in**2 - (r*np.sin(theta))**2)
            return r * np.cos(theta) - np.sqrt(r_in**2 - (r*np.sin(theta))**2)
        else:
            # print(L**2 - (r*np.sin(theta))**2)
            return r * np.cos(theta) + np.sqrt(L   **2 - (r*np.sin(theta))**2)

    def tau(nu: Quantity, r: Quantity, theta) -> Quantity:
        l   = z_max(r, theta)
        arg = (nu - frq) / dnu
        fct = vmax * nu / dnu / constants.c
        return chi*L / (fct*dnu) * 0.5 * (sp.special.erf(-arg) + sp.special.erf(fct*l/L+arg))

    def I_ (nu: Quantity, r: Quantity, theta) -> Quantity:
        return src + (bdy(nu, r, theta)-src)*np.exp(-tau(nu, r, theta))

    def v_(r: Quantity) -> Quantity:
        return dv*(1+(r-r_in)/dx)

    def u_ (nu: Quantity, r: Quantity, theta) -> Quantity:
        return 0.5 * (I_(nu, r, theta) + I_(nu, r, np.pi+theta))
    rx, ry, rz = np.array(model.geometry.rays.direction.get_astropy()).T
    angles     = np.arctan2(ry,rx)#*units.rad
    angles     = angles[:nrays//2]
    fullangles = np.arctan2(ry,rx)
    fullangles[fullangles<0] += 2*np.pi
    tempangles = np.copy(fullangles)
    fullangles[nrays//2:] = tempangles[:nrays//2]
    fullangles[:nrays//2] = tempangles[nrays//2:]

    I_analytic = np.array([[[I_(f,r,a)/units.J*units.m**2 for f in nu] for r in rs] for a in fullangles])*units.J/units.m**2
    u_analytic = np.array([[[u_(f,r,a)/units.J*units.m**2 for f in nu] for r in rs] for a in angles])*units.J/units.m**2
    fs = (nu-frq)/frq*constants.c#times c?

    #hmm, do we need this widget if we are not plotting it?
    def plot (r, p):
        plt.figure(dpi=150)
        plt.plot(fs, u_analytic[r,p,:], marker='.')
        plt.plot(fs, u_computed[r,p,:])

    def plot2(r, p):
        plt.figure(dpi=150)
        plt.plot(fs, I_analytic[r,p,:], marker='.')
        plt.plot(fs, I_computed[p,:,r])


    #during automated testing, the widgets only consume time to create
    if use_widgets:
        widgets.interact(plot, r=(0,nrays//2-1,1), p=(0,npoints-1,1))
        widgets.interact(plot2, r=(0,nrays-1,1), p=(0,npoints-1,1))

    #I am aware that errors exist on the first and last point on the rays, but this is due to how the boundary is handled
    error_u = np.abs(rtutils.relative_error(u_analytic, u_computed))
    # error_u_2f = np.abs(rtutils.relative_error(us, u_2f))

    log_err_min: Quantity = np.log10(np.min(error_u))
    log_err_max: Quantity = np.log10(np.max(error_u))

    #logspace does not work with astropy Quantity's
    bins = np.logspace(log_err_min.value, log_err_max.value, 100)

    result  = f'--- Benchmark name ------------------------------\n'
    result += f'{modelName                                      }\n'
    result += f'--- Parameters ----------------------------------\n'
    # result += f'dimension = {model.parameters.dimension()       }\n'
    result += f'npoints   = {model.parameters.npoints.get()       }\n'
    result += f'nrays     = {model.parameters.nrays.get()       }\n'
    # result += f'nquads    = {model.parameters.nquads   ()       }\n'
    result += f'--- Accuracy ------------------------------------\n'
    result += f'mean error in shortchar 0 = {np.mean(error_u)}\n'
    # result += f'mean error in feautrier 2 = {np.mean(error_u_2f)}\n'
    result += f'--- Timers --------------------------------------\n'
    # result += f'{timer1.print()                                 }\n'
    result += f'{timer2.print()                                 }\n'
    result += f'{timer3.print()                                 }\n'
    # result += f'{timer4.print()                                }\n'
    result += f'-------------------------------------------------\n'

    print(result)

    if not nosave:
        with open(f'{resdir}{modelName}-{timestamp}.log' ,'w') as log:
            log.write(result)

        plt.figure()
        plt.title(modelName)
        plt.hist(error_u.ravel(), bins=bins, histtype='step', label='formal solution')
        # plt.hist(error_u_2f.ravel(), bins=bins, histtype='step', label='2f')
        plt.xscale('log')
        plt.legend()
        plt.savefig(f'{resdir}{modelName}_hist-{timestamp}.png', dpi=150)

        plt.figure()
        plt.hist(error_u.ravel(), bins=bins, histtype='step', label='formal solution', cumulative=True)
        # plt.hist(error_u_2f.ravel(), bins=bins, histtype='step', label='2f', cumulative=True)
        plt.xscale('log')
        plt.legend()
        plt.show()
        plt.savefig(f'{resdir}{modelName}_cumu-{timestamp}.png', dpi=150)


    # FEAUTRIER_AS_EXPECTED=True
    # FIRSTORDER_AS_EXPECTED=True
    # print(np.mean())
    #if we are actually doing benchmarks, the accuracy will depend on how accurately we compute the optical depth
    # if (benchindex==1):
    #     FEAUTRIER_AS_EXPECTED=(np.mean(error_u_2f)<1e-6)
    #     FIRSTORDER_AS_EXPECTED=(np.mean(error_u_0s)<1e-6)
    # elif(benchindex==2):
    #     FEAUTRIER_AS_EXPECTED=(np.mean(error_u_2f)<5e-7)
    #     FIRSTORDER_AS_EXPECTED=(np.mean(error_u_0s)<5e-7)
    # elif(benchindex==3):
    #     FEAUTRIER_AS_EXPECTED=(np.mean(error_u_2f)<5.31e-8)
    #     FIRSTORDER_AS_EXPECTED=(np.mean(error_u_0s)<5.42e-8)

    # if not FIRSTORDER_AS_EXPECTED:
    #     print("First order solver mean error too large: ", np.mean(error_u_0s), "bench nr: ", benchindex)
    # if not FEAUTRIER_AS_EXPECTED:
    #     print("Feautrier solver mean error too large: ", np.mean(error_u_2f), "bench nr: ", benchindex)

    # return (FEAUTRIER_AS_EXPECTED&FIRSTORDER_AS_EXPECTED)


def run_test (nosave=False):

    create_and_run_model(nosave)

    return


if __name__ == '__main__':

    nosave = (len(sys.argv) > 1) and (sys.argv[1] == 'nosave')

    run_test (nosave)
