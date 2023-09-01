import astropy.units as u
import astropy
from astropy.units import Quantity
#file containing many constants that will be used throughout magritte-torch
#includes both arbitrary choices of constants for numerical stability and more
min_opacity = 1e-100 #for bounding source function from below; assumes 64 bit float for opacity
min_freq_difference = 1e-100 # assumes 64 bit float for frequency
min_dist = 1e-100 # assumes 64 bit float for distances
min_optical_depth = 1e-100 # assumes 64 bit float for optical depth


class astropy_const:
    #Defining some constants with proper units
    Tcmb: Quantity[u.K] = 2.72548 * u.K #CMB temperature for background intensity