from magrittetorch.model.model import Model
from magrittetorch.model.sources.linedata import Linedata
import torch
import astropy.constants as constants
from magrittetorch.utils.constants import astropy_const
from astropy.units import Quantity
from astropy import units
import numpy as np
from typing import TypeVar, Any, Callable

#Note: Even though we define some fixed units (in SI units) in these quantities, it does not mean that you cannot use another unit system (e.g. cgs)
# Astropy can automatically convert units if necessary

def LTEpop(linedata: Linedata, temperature: Quantity[units.K]) -> Quantity[units.dimensionless_unscaled]:
    """Computes the LTE level populations given the temperature

    Args:
        linedata (Linedata): Magritte-torch linedata object
        temperature (Quantity[units.K]): Temperature for which to evaluate the LTE level populations

    Returns:
        Quantity[units.dimensionless_unscaled]: The LTE relative level populations. dims: [linedata.nlev]
    """
    pop = np.zeros(linedata.nlev.get())
    # Calculate the LTE populations
    for i in range(linedata.nlev.get()):
        pop[i] = linedata.weight.get_astropy()[i] * np.exp(-linedata.energy.get_astropy()[i]/(constants.k_B*temperature))
    # Normalize to relative populations
    pop = pop / sum(pop)
    # Done
    return pop


def lineEmissivity(linedata: Linedata, pop: Quantity[units.m**-3]) -> Quantity[units.Hz * units.J / units.m**3]:
    """Computes the total line emissivity for each radiative transition

    Args:
        linedata (Linedata): Magritte-torch linedata object
        pop (Quantity[units.m**-3]): Populations of the levels. dims: [linedata.nlev]

    Returns:
        Quantity[units.Hz * units.J / units.m**3]: Line emissivity for each radiative transition. dims: [linedata.nrad]
    """
    eta = np.zeros(linedata.nrad.get()) * units.Hz * units.J / units.m**3
    for k in range(linedata.nrad.get()):
        i = linedata.irad.get()[k]
        j = linedata.jrad.get()[k]
        eta[k] = constants.h*linedata.frequency.get_astropy()[k]/(4.0*np.pi) * linedata.A.get_astropy()[k]*pop[i]
    # Done
    return eta


def lineOpacity(linedata: Linedata, pop: Quantity[units.m**-3]) -> Quantity[units.Hz/units.m]:
    """Computes the total line opacity for each radiative transition

    Args:
        linedata (Linedata): Magritte-torch linedata object
        pop (Quantity[units.m**-3]): Populations of the levels. dims: [linedata.nlev]

    Returns:
        Quantity[units.Hz/units.m]: Line opacity for each radiative transition. dims: [llnedata.nrad]
    """
    chi = np.zeros(linedata.nrad.get()) * units.Hz / units.m
    for k in range(linedata.nrad.get()):
        i = linedata.irad.get()[k]
        j = linedata.jrad.get()[k]
        print(linedata.Ba.get_astropy()[k] *  constants.h * pop[j])
        chi[k] = constants.h*linedata.frequency.get_astropy()[k]/(4.0*np.pi) * (linedata.Ba.get_astropy()[k]*pop[j] - linedata.Bs.get_astropy()[k]*pop[i])
    # Done
    return chi


def lineSource(linedata: Linedata, pop: Quantity[units.m**-3]) -> Quantity[units.J/units.m**2]:
    """Computes the line source function for each radiative transition

    Args:
        linedata (Linedata): Magritte-torch linedata object
        pop (Quantity[units.m**-3]): Populations of the levels. dims: [linedata.nlev]
    
    Returns:
        Quantity[units.J/units.m**2]: Line source function for each radiative transitio. dims: [linedata.nlev]
    """
    S = lineEmissivity (linedata, pop) / lineOpacity (linedata, pop)
    # Done
    return S


def planck(temperature: Quantity[units.K], frequency: Quantity[units.Hz]) -> Quantity[units.J/units.m**2]:
    """Computes the intensity according to the Planck function, at the given temperature and frequency

    Args:
        temperature (Quantity[units.K]): The temperature
        frequency (Quantity[units.Hz]): The frequency

    Returns:
        Quantity[units.J/units.m**2]: The intensity according to the Planck function.
    """
    return 2.0*constants.h/constants.c**2 * np.power(frequency,3) / np.expm1(constants.h*frequency/(constants.k_B*temperature))


def I_CMB (frequency: Quantity[units.Hz]) -> Quantity[units.J/units.m**2]:
    """Intensity of the cosmic microwave background

    Args:
        frequency (Quantity[units.Hz]): The frequency at which to evaluate the Planck function

    Returns:
        Quantity[units.J/units.m**2]: The intensity
    """
    return planck (astropy_const.Tcmb, frequency)


def dnu (linedata: Linedata, k: int, temp: Quantity[units.K], vturb: Quantity[units.m/units.s]) -> Quantity[units.Hz**-1]:
    """Computes the spectral line width of the given transition, at the specified temperature, taking into account the turbulence

    Args:
        k (int): Index of the line transition
        temp (Quantity[units.K]): The temperature
        vturb (Quantity[units.m/units.s]): The turbulence

    Returns:
        Quantity[units.Hz**-1]: The line width of the profile function
    """
    return linedata.frequency.get_astropy()[k] * np.sqrt(2.0*constants.k_B*temp/(constants.u*constants.c**2)*linedata.inverse_mass.get() + vturb**2/constants.c**2)


def profile (linedata: Linedata, k: int, temp: Quantity[units.K], vturb: Quantity[units.m/units.s], nu: Quantity[units.Hz]) -> Quantity[units.Hz**-1]:
    """Evaluates the Gaussian line profile function of the specied spectral line at a given frequency, given the local temperature and turbulence

    Args: 
        linedata (Linedata): Magritte-torch Linedata object
        k (int): Index of the line transition
        vturb (Quantity[units.m/units.s]): the turbulent velocity
        nu (Quantity[units.Hz]): The frequency at which to evaluate the profile function

    Returns:
        Quantity[units.Hz**-1]: The gaussian profile function evaluated at frequency nu
    """ 
    x = (nu - linedata.frequency.get_astropy()[k]) / dnu(linedata, k, temp, vturb)
    return np.exp(-x**2) / (np.sqrt(np.pi) * dnu(linedata, k, temp, vturb))

T = TypeVar("T")
def relative_error (a: T,b: T) -> T:
    """Computes the relative error between a and b

    Args:
        a (T): quantity a
        b (T): quantity b

    Returns:
        T: The relative error
    """

    return 2.0*(a-b)/(a+b)#type: ignore
