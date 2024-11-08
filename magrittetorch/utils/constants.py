import astropy.units as u
import astropy
from astropy.units import Quantity
#file containing many constants that will be used throughout magritte-torch
#includes both arbitrary choices of constants for numerical stability and more
min_line_opacity = 1e-10 #Minimum value for the line opacity; is able to suppress numerical artifacts from masering regimes
#NOTE: this value might need to be adjusted for different model sizes; larger models need lower values
min_opacity = 1e-45 #for bounding source function from below; assumes 64 bit float for opacity
min_freq_difference = 1e-45 # assumes 64 bit float for frequency
min_dist = 1e-45 # assumes 64 bit float for distances
min_optical_depth = 1e-45 # assumes 64 bit float for optical depth
min_rel_pop_for_convergence = 1.0e-6 #minimum relative population change for convergence
min_level_pop = 1e-45 #minimum level population for convergence computations
convergence_fraction = 0.995 #threshold for converged populations
population_inversion_fraction = 1.00#1.01 #threshold for population inversion

class astropy_const:
    """This class contains some physical constants defined in Magritte, with proper astropy units
    """
    Tcmb: Quantity[u.K] = 2.72548 * u.K #CMB temperature for background intensity