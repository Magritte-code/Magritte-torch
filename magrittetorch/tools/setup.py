#this file contains many useful scripts for creating models
#note: this is a simple port of the equivalent file in Magritte

from typing import List, Union
import numpy as np
import scipy as sp
import healpy
import torch
import re
import astropy
import astroquery.lamda as lamda

from magrittetorch.model.model import Model 
from magrittetorch.model.chemistry.species import Species
from magrittetorch.model.geometry.geometry import GeometryType
from magrittetorch.model.geometry.boundary import BoundaryCondition, BoundaryType
from magrittetorch.utils.constants import astropy_const
from magrittetorch.utils.storagetypes import Types
# LineProducingSpecies, vLineProducingSpecies,            \
                        #   CollisionPartner, vCollisionPartner, CC, HH, KB, T_CMB, \
                        #   BoundaryCondition


def check_if_1D(model: Model) -> bool:
    """Checks whether the model is 1D

    Args:
        model (Model): The model to check

    Returns:
        bool: _description_
    """
    pos = model.geometry.points.position.get()
    minpos = torch.min(pos, dim=0)
    maxpos = torch.max(pos, dim=0)
    if (torch.any(maxpos.values[1:]-minpos.values[1:]>0)):
        return False
    return True


def check_if_ordered(arr: torch.Tensor) -> None:
    """
    Check if an array is ordered, raise a ValueError if not.

    Parameters
    ----------
    arr (torch.Tensor): 1D Tensor to check for ordering.
    """
    if sorted(arr):
        pass
    else:
        raise ValueError('Not 1D (x-coordinates are not ordered).')
    return


def set_Delaunay_neighbor_lists (model: Model) -> Model:
    """
    Setter for the neighbor lists for each point, assuming they are the cell centers of a Voronoi tesselation.

    Parameters
    ----------
    model (Model): Magritte-torch model object to set.

    Returns
    -------
    out : Magritte model object
        Updated Magritte object.
    """
    nbs: list[int]
    n_nbs: list[int]
    pos = model.geometry.points.position.get()
    ndims: torch.Tensor = torch.count_nonzero(torch.max(pos, dim=0).values-torch.min(pos, dim=0).values)
    if (ndims == 1):
        check_if_ordered(model.geometry.points.position.get()[:,0])
        # For the first point
        nbs   = [1]
        n_nbs = [1]
        # For the middle points
        for p in range(1, model.parameters.npoints.get()-1):
            nbs  .extend([p-1, p+1])
            n_nbs.append(2)
        # For the last point
        nbs  .append(model.parameters.npoints.get()-2)
        n_nbs.append(1)
    elif (ndims ==3):
        # Make a Delaulay triangulation
        delaunay = sp.spatial.Delaunay(model.geometry.points.position.get().numpy())
        # Extract Delaunay vertices (= Voronoi neighbors)
        # (indptr, indices) = delaunay.vertex_neighbor_vertices
        # nbs = [indices[indptr[k]:indptr[k+1]] for k in range(model.parameters.npoints())]
        # n_nbs = [len (nb) for nb in nbs]
        (indptr, nbs) = delaunay.vertex_neighbor_vertices
        # Extract the number of neighbors for each point
        n_nbs = [indptr[k+1]-indptr[k] for k in range(model.parameters.npoints.get())]
    else:
        raise ValueError ('Dimension should be 1 or 3.')

    # Cast to numpy arrays of appropriate type
    model.geometry.points.  neighbors.set(torch.Tensor(nbs).type(Types.IndexInfo))
    model.geometry.points.n_neighbors.set(torch.Tensor(n_nbs).type(Types.IndexInfo))
    # Done
    return model


def create_rotation_matrix(n: np.ndarray) -> np.ndarray:
    '''
    Helper function to create rotation matrices.
    Returns a 3x3 orthonormal matrix around n.
    '''

    nx = np.array([1.0, 0.0, 0.0])
    ny = np.array([0.0, 1.0, 0.0])

    n1  = n
    n1 /= np.linalg.norm(n1)

    if np.linalg.norm(n1-nx) < 1.0e-6:
        n2 = np.cross(ny, n1)
    else:
        n2 = np.cross(nx, n1)
    n2 /= np.linalg.norm(n2)

    n3  = np.cross(n2, n)
    n3 /= np.linalg.norm(n3)

    return np.array([n1, n2, n3]).T


def load_balance_directions(directions: np.ndarray, comm_size: int):
    """
    Basic attempt to imporved load balancing between different MPI
    processes by reordering the rays such that each process gets a
    similar set of direcitons.
    """
    # Initialize
    Rs = directions

    # Precompute distances between all directions
    distances = np.arccos(np.matmul(Rs, Rs.T))

    # Get antipode for each direction
    antipodes = np.argmax(distances, axis=0)

    # Restrict to one hemisphere (antipodes will follow)
    Rs = Rs[:len(Rs)//2]

    # Precompute distances between all directions
    distances = np.arccos(np.matmul(Rs, Rs.T))

    # Initialize index lists
    inds   = list(range(len(Rs)))
    inds_o = [[] for _ in range(comm_size)]#type: ignore

    while len(inds) >= comm_size:
        # Take the next index
        id0 = inds[0]
        inds     .remove(id0)
        inds_o[0].append(id0)
        # Append indices of close directions to other processes
        for i in range(1, comm_size):
            idi = inds[np.argmin(distances[id0][inds])]
            inds     .remove(idi)
            inds_o[i].append(idi)

    # Append final indices
    for i, idx in enumerate(inds):
        inds_o[i].append(idx)

    # Unravel
    inds_o = [j for sl in inds_o for j in sl]

    # Add antipodes
    inds_o.extend(antipodes[inds_o].tolist())

    return directions[inds_o]


def set_uniform_rays(model: Model, nrays: int, randomize: bool =False, first_ray: np.ndarray = np.array([1.0, 0.0, 0.0])) -> Model:
    """
    Setter for rays to uniformly distributed directions.

    Args:
        model (Model): Magritte model object to set.
        nrays (int): Number of rays to use
        randomized (bool): Whether or not to randomize the directions of the rays.
        first_ray (np.ndarray): Direction vector of the first ray in the ray list. Has dimensions [3]

    Returns
    -------
    out : Magritte model object
        Updated Magritte object.
    """
    #get number dims:
    pos = model.geometry.points.position.get()
    ndims: torch.Tensor = torch.count_nonzero(torch.max(pos, dim=0).values-torch.min(pos, dim=0).values)
    if (ndims == 1):
        if model.geometry.geometryType.get() == GeometryType.SpericallySymmetric1D:
            return set_rays_spherical_symmetry(model, uniform=True)
        else:
            if (nrays != 2):
                raise ValueError('In 1D without spherical symmetry, nrays should be 2.')
            direction = [[-1, 0, 0], [1, 0, 0]]
    elif (ndims == 3):
        direction = healpy.pixelfunc.pix2vec(healpy.npix2nside(nrays), range(nrays))
        direction = np.array(direction).transpose()
        if randomize:
            direction = sp.spatial.transform.Rotation.random().apply(direction)

        # Rotate such that the first ray is in the given direction
        R1 = create_rotation_matrix(direction[0])
        R2 = create_rotation_matrix(first_ray)
        R  = R2 @ np.linalg.inv(R1)

        direction = direction @ R.T
    else:
        raise ValueError ('Dimension of model should be 1 or 3.')
    # Cast to numpy arrays of appropriate type and shape
    model.geometry.rays.direction.set(torch.Tensor(direction).type(Types.GeometryInfo))
    model.geometry.rays.weight.set(torch.from_numpy((1.0/nrays) * np.ones(nrays)))
    # Done
    return model


def set_rays_spherical_symmetry(model: Model, nrays: int, uniform: bool = True, nextra: int = 0, step: int = 1) -> Model:
    """
    Setter for rays in a 1D spherically symmetric model.

    Parameters
    ----------
    model : Magritte model object
        Magritte model object to set.
    nrays: int
        Number of rays to use
    uniform : bool
        Whether or not to use uniformly distributed rays.
    nextra : int
        Number of extra rays to add.
    step : int
        Step size used to step through the points when shooting rays (step=1 uses all points).

    Returns
    -------
    out : Magritte model object
        Updated Magritte object.
    """
    rs: list[int]|np.ndarray
    if uniform:
        # Treat every ray as "extra", since "extra" rays are added uniformly anyway
        print("nrays", nrays)
        nextra = nrays//2-1
        rs = []
    else:
        rs    = np.array(model.geometry.points.position)[:,0]
        # Below we assume that rs is sorted and the last element is the largest, hence sort
        rs = np.sort(rs)
        R  = rs[-1]
    
    # Add the first ray, right through the centre, i.e. along the x-axis
    Rx = [1.0]
    Ry = [0.0]
    Rz = [0.0]
    
    # Add the other rays, such that for the outer shell each ray touches another shell
    for ri in rs[0:-1:step]:
        ry = ri / R
        rx = np.sqrt(1.0 - ry**2)
        Rx.append(rx)
        Ry.append(ry)
        Rz.append(0.0)
    
    # Determine up to which angle we already have rays
    angle_max   = np.arctan(Ry[-1] / Rx[-1])
    angle_extra = (0.5*np.pi - angle_max) / nextra
    
    # Fill the remaining angular space uniformly
    for k in range(1, nextra):
        Rx.append(np.cos(angle_max + k*angle_extra))
        Ry.append(np.sin(angle_max + k*angle_extra))
        Rz.append(0.0)
    
    # Add the last ray, orthogonal to the radial direction, i.e. along the y-axis
    Rx.append(0.0)
    Ry.append(1.0)
    Rz.append(0.0)
    
    # Determine the number of rays
    assert len(Rx) == len(Ry)
    assert len(Rx) == len(Rz)
    nrays = 2*len(Rx)
    
    # Compute the weights for each ray
    # \int_{half previous}^{half next} sin \theta d\theta
    # = cos(half previous angle) - cos(half next angle)
    Wt = []
    for n in range(nrays//2):
        if   (n == 0):
            upper_x, upper_y = 0.5*(Rx[n]+Rx[n+1]), 0.5*(Ry[n]+Ry[n+1])
            lower_x, lower_y = Rx[ 0], Ry[ 0]
        elif (n == nrays//2-1):
            upper_x, upper_y = Rx[-1], Ry[-1]
            lower_x, lower_y = 0.5*(Rx[n]+Rx[n-1]), 0.5*(Ry[n]+Ry[n-1])
        else:
            upper_x, upper_y = 0.5*(Rx[n]+Rx[n+1]), 0.5*(Ry[n]+Ry[n+1])
            lower_x, lower_y = 0.5*(Rx[n]+Rx[n-1]), 0.5*(Ry[n]+Ry[n-1])
    
        Wt.append(  lower_x / np.sqrt(lower_x**2 + lower_y**2)
                  - upper_x / np.sqrt(upper_x**2 + upper_y**2) )
    
    # Append the antipodal rays
    for n in range(nrays//2):
        Rx.append(-Rx[n])
        Ry.append(-Ry[n])
        Rz.append(-Rz[n])
        Wt.append( Wt[n])
    
    # Create numpy arrays
    direction = np.array((Rx,Ry,Rz)).transpose()
    weight    = np.array(Wt)
    
    # Normalize the weights
    weight /= np.sum(weight)

    print("direction", direction)
    
    # Set the direction and the weights in the Magritte model
    model.geometry.rays.direction.set(torch.from_numpy(direction))
    model.geometry.rays.weight   .set(torch.from_numpy(weight))
    
    # Set nrays in the model
    try:
        model.parameters.nrays.set(nrays)
    except:
        raise RuntimeError(
            f"The specified number of rays in the model (nrays={model.parameters.nrays.get()}) does not match\n"
            f"with the specified nextra={nextra} and step={step}. Either don't specify the number of rays\n"
            f"for the model (i.e. don't invoke model.parameters.set_nrays) so this function can\n"
            f"set nrays, or specify the right number, which is {nrays} in this particular case."
        )    
    
    # Done
    return model


def set_Delaunay_boundary (model: Model) -> Model:
    """
    Setter for the boundary, assuming all points are the cell centers of a Voronoi tesselation.
    Only supported for 1D.

    Parameters
    ----------
    model : Magritte model object
        Magritte model object to set.

    Returns
    -------
    out : Magritte model object
        Updated Magritte object.
    """
    pos = model.geometry.points.position.get()
    ndims: torch.Tensor = torch.count_nonzero(torch.max(pos, dim=0).values-torch.min(pos, dim=0).values)
    if (ndims == 1):
        if not check_if_1D(model): raise ValueError
        check_if_ordered(model.geometry.points.position.get()[:,0])
        model.parameters.nboundary.set(2)
        model.geometry.boundary.boundary2point.set(torch.Tensor([0, model.parameters.npoints.get()-1]).type(Types.IndexInfo))
    else:
        raise ValueError ('Dimension should be 1.')
    # Done
    return model


def set_boundary_condition_zero (model: Model) -> Model:
    """
    Setter for incoming zero boundary condition at each boundary point.

    Parameters
    ----------
    model : Magritte model object
        Magritte model object to set.

    Returns
    -------
    out : Magritte model object
        Updated Magritte object.
    """
    model.geometry.boundary.boundary_condition.set(torch.full([model.parameters.nboundary.get()], BoundaryCondition.Zero.value, dtype=Types.Enum))

    return model


def set_boundary_condition_CMB (model: Model) -> Model:
    """
    Setter for incoming CMB boundary condition at each boundary point.

    Parameters
    ----------
    model : Magritte model object
        Magritte model object to set.

    Returns
    -------
    out : Magritte model object
        Updated Magritte object.
    """
    model.geometry.boundary.boundary_condition.set(torch.full([model.parameters.nboundary.get()], BoundaryCondition.CMB.value, dtype=Types.Enum))
    model.geometry.boundary.boundary_temperature.set_astropy(astropy_const.Tcmb * [1.0 for _ in range(model.parameters.nboundary.get())])
    # Done
    return model


def set_boundary_condition_1D (model: Model, T_in : astropy.units.Quantity = astropy_const.Tcmb, T_out : astropy.units.Quantity =astropy_const.Tcmb):
    """
    Setter for incoming black body radiation boundary condition at each boundary point.

    Parameters
    ----------
    model : Magritte model object
        Magritte model object to set.
    T_in : float
        Boundary temperature at the inner boundary.
    T_out : float
        Boundary temperature at the outer boundary.

    Returns
    -------
    out : Magritte model object
        Updated Magritte object.
    """
    if not (model.geometry.geometryType.get() == GeometryType.SpericallySymmetric1D):
        raise ValueError ('These boundary conditions only work for a 1D model.')
    if not (model.parameters.nboundary.get() == 2):
        raise ValueError ('A 1D model must have exactly 2 boundary points.')
    else:
        # Set all boundary conditions to Thermal
        model.geometry.boundary.boundary_condition.set(torch.full([model.parameters.nboundary.get()], BoundaryCondition.Thermal.value, dtype=Types.Enum))
        # Set inner and outer temperature
        model.geometry.boundary.boundary_temperature.set_astropy([T_in, T_out])
    # Done
    return model


def set_quadrature(model: Model, nquads: int) -> Model:
    """Setter for the quadrature roots and weights for the Gauss-Hermite
    quadrature, used for integrating over (Gaussian) line profiles.

    Args:
        model (Model): Magritte model object
        nquads (int): Number of frequency quadrature points per line

    Returns:
        Model: Updated Magritte object
    """

    for l in range(model.parameters.nlspecs.get()):
        # Get (Gauss-Hermite) quadrature roots and weights
        (roots, weights) = np.polynomial.hermite.hermgauss(nquads)#type: ignore
        # Normalize weights
        weights = weights / np.sum(weights)
        # Set roots and weights
        model.sources.lines.lineProducingSpecies[l].linequadrature.roots  .set(torch.from_numpy(roots))
        model.sources.lines.lineProducingSpecies[l].linequadrature.weights.set(torch.from_numpy(weights))
    return model


def getProperName(name: str) -> str:
    '''
    Return the standard name for the species.
    '''
    if name in ['e']:
        return 'e-'
    if name in ['pH2', 'oH2', 'p-H2', 'o-H2', 'PH2', 'OH2']:
        return 'H2'
    # If none of the above special cases, it should be fine
    return name

def getSpeciesNumber (species: Species, name: str) -> int:
    '''
    Returns number of species given by 'name'

    Parameters
    ----------
    species : Magritte species object
        Contains the information about the chemical species in the model
    name : str
        name of the chemical species

    Raises
    ------
    KeyError : if the species name is not found in the list of chemical species
    '''
    print(species.symbol.get())
    for i in range (len (species.symbol.get())):
        if (species.symbol.get()[i].decode() == str(getProperName(name))):#Internally, we save a byte string, for hdf5 writing purposes
            #We might get issues with this in the future
            return i
    raise KeyError("Could not find species '" + str(getProperName(name)) + "' in the list of chemical species.")

def getSpeciesNumberList (species: Species, name: list[str]) -> list[int]:
    '''
    Returns number of species given by 'name'

    Parameters
    ----------
    species (Species): Magritte species object, contains the information about the chemical species in the model
    name (list[str]): names of the chemical species

    Raises
    ------
    KeyError : if the species name is not found in the list of chemical species
    '''
    return [getSpeciesNumber (species,elem) for elem in name]

# def extractCollisionPartner (fileName: str, line: int, species, elem: str):
#     '''
#     Returns collision partner and whether it is ortho or para (for H2)
#     '''
#     with open (fileName) as dataFile:
#         data = dataFile.readlines ()
#     partner   = re.findall (elem.replace ('+','\+')+'\s*[\+\-]?\s*([\w\+\-]+)\s*', data[line])[0]
#     excess    = re.findall ('[op]\-?', partner)
#     if (len (excess) > 0):
#         orthoPara = re.findall ('[op]', partner)[0]
#         partner   = partner.replace (excess[0],'')
#     else:
#         orthoPara = 'n'
#     return [getSpeciesNumber (species, partner), orthoPara]


#Astroquery actually reads the number in front of the line to determine which collider is used.
# Thus the names for H2 are limited to PH2, OH2.
def extractCollisionPartnerAstroquery (partner: str, species: Species) -> tuple[int, str]:
    orthopara = 'n'
    if partner == 'PH2':
        orthopara = 'p'
    elif partner == 'OH2':
        orthopara = 'o'

    try:
        return (getSpeciesNumber (species, partner), orthopara)
    except KeyError as err:
        print("Warning:", err, "Using dummy values for the density of this collision partner instead.")
        return (0,orthopara)


#Astroquery names the temperature colums for the collision rates in the following manner: "C_ij(T=temp)"
def convertTempToColumnNameAstroquery (temp: str) -> str:
    return "C_ij(T="+str(temp)+")"



class LamdaFileReader ():
    """
    Reader for LAMDA line data files.
    """

    def __init__ (self ,fileName):
        """
        Set the file name of the LAMDA file.

        Parameters
        ----------
        fileName : str
            Name of the LAMDA line data file.
        """
        self.fileName = fileName

    def readColumn (self, start, nElem, columnNr, type):
        '''
        Returns a column of data as a list
        '''
        with open (self.fileName) as dataFile:
            lineNr = 0
            column = []
            for line in dataFile:
                if (lineNr >= start) and (lineNr < start+nElem):
                    if type == 'float':
                        column.append (float(line.split()[columnNr]))
                    if type == 'int':
                        column.append (int  (line.split()[columnNr]))
                    if type == 'str':
                        column.append (str  (line.split()[columnNr]))
                lineNr += 1
        return column

    def extractCollisionPartner (self, line, species, elem):
        '''
        Returns collision partner and whether it is ortho or para (for H2)
        '''
        with open (self.fileName) as dataFile:
            data = dataFile.readlines ()
        partner   = re.findall (elem.replace ('+','\+')+'\s*[\+\-]?\s*([\w\+\-]+)\s*', data[line])[0]
        excess    = re.findall ('[op]\-?', partner)
        if (len (excess) > 0):
            orthoPara = re.findall ('[op]', partner)[0]
            partner   = partner.replace (excess[0],'')
        else:
            orthoPara = 'n'
        return [getSpeciesNumber (species, partner), orthoPara]



def set_linedata_from_LAMDA_file (model: Model, fileNames: Union[list[str],str], config: dict[str, list[int]] = {}):
    """
    Set line data by reading it from a data file in LAMDA format.

    Parameters
    ----------
    model : Magritte model object
        Magritte model object to set.
    fileNames : list of strings
        List of file names for the LAMDA line data files.
    config : dict
        Optionally specify specific radiative transitions to consider.

    Returns
    -------
    out : Magritte model object
        Updated Magritte object.

    Note
    ----
    Do not use the Magritte objects linedata etc. this will kill performance. Hence, the copies.
    """
    # Make sure fileNames is a list
    fileNameList: list[str]
    if type(fileNames) is str:
        fileNameList = [fileNames]
    else:
        fileNameList = fileNames #type: ignore
    model.parameters.nlspecs.set(len(fileNameList))
    # Make sure a file is provides for each species
    # if len(fileNameList) != model.parameters.nlspecs.get():
    #     raise ValueError('Number of provided LAMDA files != nlspecs')
    # Create lineProducingSpecies objects    
    model.sources.lines.lineProducingSpecies.construct()
    # Convenient name
    species = model.chemistry.species
    # Add data for each LAMDA file
    for lspec, fileName in enumerate(fileNameList):
        #new astroquery file read
        collrates, radtransitions, enlevels = lamda.parse_lamda_datafile(fileName)
        sym = enlevels.meta['molecule']# is given after ! MOLECULE
        num          = getSpeciesNumber(species, sym)
        mass = enlevels.meta['molwt']
        inverse_mass = float (1.0/mass)
        nlev = enlevels.meta['nenergylevels']
        energy = enlevels['Energy']
        weight = enlevels['Weight']
        nrad = radtransitions.meta['radtrans']
        irad = radtransitions['Upper'] #upper transition
        jrad = radtransitions['Lower'] #lower transition
        A = radtransitions['EinsteinA']

        # Change index range from [1, nlev] to [0, nlev-1]
        for k in range(nrad):
            irad[k] += -1
            jrad[k] += -1

        # Convert to SI units
        for i in range(nlev):
            # Energy from [cm^-1] to [J]
            energy[i] *= 1.0E+2*astropy.constants.h.value*astropy.constants.c.value

        # Set data
        model.sources.lines.lineProducingSpecies[lspec].linedata.sym.set(sym)
        model.sources.lines.lineProducingSpecies[lspec].linedata.num.set(num)
        model.sources.lines.lineProducingSpecies[lspec].linedata.inverse_mass.set(inverse_mass)
        model.sources.lines.lineProducingSpecies[lspec].linedata.nlev.set(nlev)
        model.sources.lines.lineProducingSpecies[lspec].linedata.energy.set(torch.Tensor(energy).type(Types.LevelPopsInfo))
        model.sources.lines.lineProducingSpecies[lspec].linedata.weight.set(torch.Tensor(weight).type(Types.LevelPopsInfo))

        ncolpar = len(collrates)

        # Set number of collision partners
        model.sources.lines.lineProducingSpecies[lspec].linedata.ncolpar.set(ncolpar)
        # Create list of CollisionPartners
        model.sources.lines.lineProducingSpecies[lspec].linedata.colpar.construct()

        # Loop over the collision partners
        for partner, c in zip(collrates.keys(), range(ncolpar)):
            num_col_partner, orth_or_para_H2 = extractCollisionPartnerAstroquery (partner, species)
            ncol = collrates[partner].meta['ntrans']
            ntmp = collrates[partner].meta['ntemp']
            icol = collrates[partner]['Upper']
            jcol = collrates[partner]['Lower']

            # Change index range from [1, nlev] to [0, nlev-1]
            for k in range(ncol):
                icol[k] += -1
                jcol[k] += -1

            tmp = collrates[partner].meta['temperatures']
            Cd  = []

            for temp in tmp:
                Cd.append(collrates[partner][convertTempToColumnNameAstroquery(temp)])

            # Convert to SI units
            for t in range(ntmp):
                for k in range(ncol):
                    # Cd from [cm^3] to [m^3]
                    Cd[t][k] *= 1.0E-6
            # Derive excitation coefficients
            Ce = [[0.0 for _ in range(ncol)] for _ in range(ntmp)]
            for t in range(ntmp):
                for k in range(ncol):
                    i = icol[k]
                    j = jcol[k]
                    Ce[t][k] = Cd[t][k] * weight[i]/weight[j] * np.exp( -(energy[i]-energy[j]) / (astropy.constants.k_B.value*tmp[t]) )
            # Set data
            model.sources.lines.lineProducingSpecies[lspec].linedata.colpar[c].num_col_partner.set(num_col_partner)
            model.sources.lines.lineProducingSpecies[lspec].linedata.colpar[c].orth_or_para_H2.set(orth_or_para_H2)
            model.sources.lines.lineProducingSpecies[lspec].linedata.colpar[c].ncol.set(ncol)
            model.sources.lines.lineProducingSpecies[lspec].linedata.colpar[c].ntmp.set(ntmp)
            model.sources.lines.lineProducingSpecies[lspec].linedata.colpar[c].icol.set(torch.Tensor(icol).type(Types.IndexInfo))
            model.sources.lines.lineProducingSpecies[lspec].linedata.colpar[c].jcol.set(torch.Tensor(jcol).type(Types.IndexInfo))
            model.sources.lines.lineProducingSpecies[lspec].linedata.colpar[c].tmp.set(torch.Tensor(tmp).type(Types.LevelPopsInfo))
            model.sources.lines.lineProducingSpecies[lspec].linedata.colpar[c].Cd.set(torch.from_numpy(np.array(Cd)).type(Types.LevelPopsInfo))
            model.sources.lines.lineProducingSpecies[lspec].linedata.colpar[c].Ce.set(torch.from_numpy(np.array(Ce)).type(Types.LevelPopsInfo))

        # Limit to the specified lines if required
        if ('considered transitions' in config) and (config['considered transitions'] is not None):
            if not isinstance(config['considered transitions'], list):
                config['considered transitions'] = [config['considered transitions']]
            if (len(config['considered transitions']) > 0):
                print('Not considering all radiative transitions on the data file but only the specified ones!')
                nrad = len (config['considered transitions'])
                irad = [irad[k] for k in config['considered transitions']]
                jrad = [jrad[k] for k in config['considered transitions']]
                A    = [   A[k] for k in config['considered transitions']]

        # Set derived quantities
        Bs        = [0.0 for _ in range(nrad)]
        Ba        = [0.0 for _ in range(nrad)]
        frequency = [0.0 for _ in range(nrad)]
        for k in range(nrad):
            i = irad[k]
            j = jrad[k]
            frequency[k] = (energy[i]-energy[j]) / astropy.constants.h.value
            Bs[k]        = A[k] * astropy.constants.c.value**2 / (2.0*astropy.constants.h.value*(frequency[k])**3)
            Ba[k]        = weight[i]/weight[j] * Bs[k]

        # Set data
        model.sources.lines.lineProducingSpecies[lspec].linedata.nrad.set(nrad)
        model.sources.lines.lineProducingSpecies[lspec].linedata.irad.set(torch.Tensor(irad).type(Types.IndexInfo))
        model.sources.lines.lineProducingSpecies[lspec].linedata.jrad.set(torch.Tensor(jrad).type(Types.IndexInfo))
        model.sources.lines.lineProducingSpecies[lspec].linedata.A.set(torch.Tensor(A).type(Types.LevelPopsInfo))
        model.sources.lines.lineProducingSpecies[lspec].linedata.Bs.set(torch.Tensor(Bs).type(Types.LevelPopsInfo))
        model.sources.lines.lineProducingSpecies[lspec].linedata.Ba.set(torch.Tensor(Ba).type(Types.LevelPopsInfo))
        model.sources.lines.lineProducingSpecies[lspec].linedata.frequency.set(torch.Tensor(frequency).type(Types.FrequencyInfo))

    # Done
    return model

