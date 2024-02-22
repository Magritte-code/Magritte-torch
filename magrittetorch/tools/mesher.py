import numpy as np
import healpy
from astropy import units
from astropy.units import Quantity


def remesh_point_cloud(positions, data, max_depth=9, threshold= 5e-2, hullorder = 3):
    '''
    Remeshing method by comparing the maximal variation of the data against the threshold.
    Uses a recursive method with maximal depth max_depth.
    The hullorder specifies the density of the generated uniform boundary.

    Parameters
    ----------
    positions : numpy array of float OR Quantity
        3D positions of the points (dimension of array N by 3)
    data : numpy array of float OR Quantity
        Corresponding data for the points, in order to determine the variation.
        Must be non-zero
    max_depth : int
        Maximal recursion depth. Determines the miminal scale in the remeshed point cloud.
        Must be positive
    threshold : float
        The threshold to use when deciding the variation to be small enough in order to approximate a region with a single point.
        Must be positive
    hullorder : int
        Determines the amount of boundary points generated in the hull around the remeshed point cloud.
        Increasing this by 1 results in a 4 times increase in number of boundary points.
        Must be positive

    Returns
    -------
    remeshed_positions : numpy array of float
        The positions of the points in the remeshed point cloud. The boundary points lie in front.
    number of boundary points : unsigned int
        The number of boundary points in the remeshed point cloud.
    '''
    # units are not necessary for the remeshing procedure, and will be ignored for the internal calculations
    posunit = None
    if isinstance(positions, Quantity):
        posunit = units.m
        positions = positions.si.value
    if isinstance(data, Quantity):
        data = data.si.value

    # positions = positions.si.value

    new_positions = np.zeros((len(positions), 3))#should be large enough to contain the new positions
    #in the worst case, the recursive remeshing procedure will return a point cloud with size of the old points
    remesh_nb_points = 0#helper index for where to put the generated new points

    xyz_min = np.min(positions, axis=0)
    xyz_max = np.max(positions, axis=0)

    delta_xyz = xyz_max-xyz_min

    #Recursive re-mesh procedure puts the new points into the new_positions vector (with remesh_nb_points now containing the size of the re-meshed point cloud)
    remesh_nb_points = get_recursive_remesh(positions, data, 0, max_depth, threshold, new_positions, remesh_nb_points)
    print("new interior points: ", remesh_nb_points)
    #shrink positions vector to the actual ones
    new_positions.resize((remesh_nb_points,3))

    #boundary should be seperated from the actual points
    eps = 1e-3
    bxyz_min = xyz_min - eps*delta_xyz
    bxyz_max = xyz_max + eps*delta_xyz

    hull = create_cubic_uniform_hull(bxyz_min, bxyz_max, order=hullorder)
    nb_boundary = hull.shape[0]
    print("number boundary points: ", nb_boundary)
    new_positions = np.concatenate((hull, new_positions), axis = 0)

    # The boundary hull is located at the first nb_boundary positions indices of the new_positions vector
    if posunit is not None:
        new_positions = new_positions*posunit
    return (new_positions, nb_boundary)

# @numba.njit(cache=True)
def create_cubic_uniform_hull(xyz_min, xyz_max, order=3):
    nx, ny, nz = (2**order+1, 2**order+1, 2**order+1)
    x_vector = np.linspace(xyz_min[0], xyz_max[0], nx)
    y_vector = np.linspace(xyz_min[1], xyz_max[1], ny)
    z_vector = np.linspace(xyz_min[2], xyz_max[2], nz)

    #x plane does not yet intersect with other planes
    xmin_plane = grid3D([xyz_min[0]], y_vector, z_vector)
    xmax_plane = grid3D([xyz_max[0]], y_vector, z_vector)

    #y plane intersects with x plane, so using reduced vectors for x coordinate
    ymin_plane = grid3D(x_vector[1:nx-1], [xyz_min[1]], z_vector)
    ymax_plane = grid3D(x_vector[1:nx-1], [xyz_max[1]], z_vector)

    #z plane also intersects with x plane
    zmin_plane = grid3D(x_vector[1:nx-1], y_vector[1:ny-1], [xyz_min[2]])
    zmax_plane = grid3D(x_vector[1:nx-1], y_vector[1:ny-1], [xyz_max[2]])

    #At the edges, the hull will contain duplicate points. These need to be removed= 0)
    hull = np.concatenate((xmin_plane, xmax_plane, ymin_plane, ymax_plane, zmin_plane, zmax_plane), axis = 0)

    return hull

#Simple function for the outer product of 1D vectors
#Allows us to create surface point cloud by inserting 2 vectors (and a single value for the last coordinate)
# @numba.njit(cache=True)
def grid3D(x, y, z):
    #if x has a unit, the result will also have the same unit
    if isinstance(x, Quantity):
        unit = x.unit
        xyz = np.empty((len(x)*len(y)*len(z), 3)) * unit
    elif isinstance(y, Quantity):
        unit = y.unit
        xyz = np.empty((len(x)*len(y)*len(z), 3)) * unit
    else:
        xyz = np.empty((len(x)*len(y)*len(z), 3))
    idx = 0
    for k in range(len(x)):
        for j in range(len(y)):
            for i in range(len(z)):
                xyz[idx] = [x[k], y[j], z[i]]
                idx+=1
    return xyz


# Recursive function to generate a hierarchical remeshed grid, based on the total variation in data (compared to the threshold)
# Returns the index 'remesh_nb_points', counting how much points have already been added to the remeshed grid
#Does not use jit, as this is a recursive function
def get_recursive_remesh(positions: np.array, data: np.array, depth: int, max_depth: int, threshold: float, remesh_points: np.array, remesh_nb_points: int):
    '''
    Helper function for the point cloud remeshing method.
    Uses recursion to remesh a given point cloud (uses all data to determine whether to recurse on a smaller scale),
    by evaluating the total variation in the data (compared against the threshold).

    Parameters
    ----------
    positions : numpy array of float
        3D positions of the points (dimension of array N by 3)
    data : numpy array of float
        Corresponding data for the points, in order to determine the variation.
        Must be non-zero
    depth : int
        Current recursion depth
    max_depth : int
        Maximal recursion depth. Determines the miminal scale in the remeshed point cloud.
        Must be positive
    threshold : float
        The threshold to use when deciding the variation to be small enough in order to approximate a region with a single point.
        Must be positive
    remesh_points : numpy array (in/out)
        Array holding the remeshed point positions. Must be large enough to contain all remeshed points,  Must have at least dimensions N by 3.
    remesh_nb_points : int (in/out)
        The number of point in the remeshed point cloud.
        Must be initialized to 0
    '''
    # print("start")

    #If no data is left, no point should be added
    if len(data)==0:
        return remesh_nb_points

    min_coord = np.min(positions, axis=0)
    max_coord = np.max(positions, axis=0)

    #If the subdivision has been going on for too long (only a single point remaining or too deep recursion), we replace this box with a point
    if len(data)==1 or depth==max_depth:
        #add this point to the list
        remesh_points[remesh_nb_points, :] = (min_coord + max_coord) / 2.0
        remesh_nb_points+=1
        return remesh_nb_points

    minval, maxval = np.min(data), np.max(data)

    #If the total variation in this box lies within the defined bounds, we can stop and approximate this box by a single point
    if (maxval-minval)<threshold*(minval+maxval):
        #add this point to the list
        remesh_points[remesh_nb_points, :] = (min_coord + max_coord) / 2.0
        remesh_nb_points+=1
        return remesh_nb_points

    else:
        #go and do some more recursive investigation
        middle = (max_coord + min_coord) / 2.0

        uuu_idx = (positions[:,0] >= middle[0]) & (positions[:,1] >= middle[1]) & (positions[:,2] >= middle[2])
        uuu_data = data[uuu_idx]
        uuu_positions = positions[uuu_idx, :]

        remesh_nb_points = get_recursive_remesh(uuu_positions, uuu_data, depth+1, max_depth, threshold, remesh_points, remesh_nb_points)

        uul_idx = (positions[:,0] >= middle[0]) & (positions[:,1] >= middle[1]) & (positions[:,2] <  middle[2])
        uul_data = data[uul_idx]
        uul_positions = positions[uul_idx, :]

        remesh_nb_points = get_recursive_remesh(uul_positions, uul_data, depth+1, max_depth, threshold, remesh_points, remesh_nb_points)

        ulu_idx = (positions[:,0] >= middle[0]) & (positions[:,1] <  middle[1]) & (positions[:,2] >= middle[2])
        ulu_data = data[ulu_idx]
        ulu_positions = positions[ulu_idx, :]

        remesh_nb_points = get_recursive_remesh(ulu_positions, ulu_data, depth+1, max_depth, threshold, remesh_points, remesh_nb_points)

        ull_idx = (positions[:,0] >= middle[0]) & (positions[:,1] <  middle[1]) & (positions[:,2] <  middle[2])
        ull_data = data[ull_idx]
        ull_positions = positions[ull_idx, :]

        remesh_nb_points = get_recursive_remesh(ull_positions, ull_data, depth+1, max_depth, threshold, remesh_points, remesh_nb_points)

        luu_idx = (positions[:,0] <  middle[0]) & (positions[:,1] >= middle[1]) & (positions[:,2] >= middle[2])
        luu_data = data[luu_idx]
        luu_positions = positions[luu_idx, :]

        remesh_nb_points = get_recursive_remesh(luu_positions, luu_data, depth+1, max_depth, threshold, remesh_points, remesh_nb_points)

        lul_idx = (positions[:,0] <  middle[0]) & (positions[:,1] >= middle[1]) & (positions[:,2] <  middle[2])
        lul_data = data[lul_idx]
        lul_positions = positions[lul_idx, :]

        remesh_nb_points = get_recursive_remesh(lul_positions, lul_data, depth+1, max_depth, threshold, remesh_points, remesh_nb_points)

        llu_idx = (positions[:,0] <  middle[0]) & (positions[:,1] <  middle[1]) & (positions[:,2] >= middle[2])
        llu_data = data[llu_idx]
        llu_positions = positions[llu_idx, :]

        remesh_nb_points = get_recursive_remesh(llu_positions, llu_data, depth+1, max_depth, threshold, remesh_points, remesh_nb_points)

        lll_idx = (positions[:,0] <  middle[0]) & (positions[:,1] <  middle[1]) & (positions[:,2] <  middle[2])
        lll_data = data[lll_idx]
        lll_positions = positions[lll_idx, :]

        remesh_nb_points = get_recursive_remesh(lll_positions, lll_data, depth+1, max_depth, threshold, remesh_points, remesh_nb_points)

    return remesh_nb_points


#Clear for internal use
def point_cloud_clear_inner_boundary_generic(remeshed_positions, nb_boundary, numpy_friendly_function, threshold):
    '''
    General function for consistently clearing an inner boundary region in a remeshed point cloud.

    Parameters
    ----------
    remeshed_positions : numpy array of float
        Positions of the points in the point cloud. Assumes the boundary points to lie in front.
    nb_boundary : unsigned int
        The number of boundary points in the point cloud.
    numpy_friendly_function : lambda function which operates on numpy array
        Function which acts on location for determining the shape of the inner boundary condition region
    threshold : float
        Cutoff value associate to numpy_friendly_function in order to constrain the shape of the inner boundary region

    Returns
    -------
    remeshed_positions : numpy array of float
        The positions of the points in the remeshed point cloud. The boundary points lie in front.
    number of boundary points : unsigned int
        The number of boundary points in the remeshed point cloud.
    '''
    boundary_points = remeshed_positions[:nb_boundary, :].copy()#extract original boundary points

    keep_pos = numpy_friendly_function(remeshed_positions)>threshold
    new_remeshed_positions = remeshed_positions[keep_pos]
    remove_bdy = numpy_friendly_function(boundary_points)<=threshold
    nb_bounds_reduced = np.sum(remove_bdy)
    new_nb_boundary = nb_boundary-nb_bounds_reduced

    return new_remeshed_positions, new_nb_boundary


#Clear for internal use
def point_cloud_clear_outer_boundary_generic(remeshed_positions, nb_boundary, numpy_friendly_function, threshold):
    '''
    General function for consistently clearing all points outside a given region in a remeshed point cloud.

    Parameters
    ----------
    remeshed_positions : numpy array of float
        Positions of the points in the point cloud. Assumes the boundary points to lie in front.
    nb_boundary : unsigned int
        The number of boundary points in the point cloud.
    numpy_friendly_function : lambda function which operates on numpy array
        Function which acts on location for determining the shape of the inner boundary condition region
    threshold : float
        Cutoff value associate to numpy_friendly_function in order to constrain the shape of the inner boundary region

    Returns
    -------
    remeshed_positions : numpy array of float
        The positions of the points in the remeshed point cloud. The boundary points lie in front.
    number of boundary points : unsigned int
        The number of boundary points in the remeshed point cloud.
    '''
    boundary_points = remeshed_positions[:nb_boundary, :].copy()#extract original boundary points

    keep_pos = numpy_friendly_function(remeshed_positions)<threshold
    new_remeshed_positions = remeshed_positions[keep_pos]
    remove_bdy = numpy_friendly_function(boundary_points)>=threshold
    nb_bounds_reduced = np.sum(remove_bdy)
    new_nb_boundary = nb_boundary-nb_bounds_reduced

    return new_remeshed_positions, new_nb_boundary


def point_cloud_add_spherical_inner_boundary(remeshed_positions, nb_boundary, radius, healpy_order = 5, origin = np.array([0.0,0.0,0.0]).T):
    '''
    Function for specifying a spherical inner boundary in a remeshed point cloud.
    First clears the inner region of points and then constructs a spherical boundary using healpy.

    Parameters
    ----------
    remeshed_positions : numpy array of float
        Positions of the points in the point cloud. Assumes the boundary points to lie in front.
    nb_boundary : unsigned int
        The number of boundary points in the point cloud.
    radius : positive float
        Radius of the spherical inner boundary
    healpy_order : unsigned int
        The number of points on the inner boundary is defined as: 12*(healpy_order**2), as the healpy discretization of the sphere is used.
    origin : 1 by 3 numpy vector of float
        Contains x,y,z coordinates of center point of the sphere (by default [0.0,0.0,0.0]^T)

    Returns
    -------
    remeshed_positions : numpy array of float
        The positions of the points in the remeshed point cloud. The boundary points lie in front.
    number of boundary points : unsigned int
        The number of boundary points in the remeshed point cloud.
    '''

    radii2 = lambda r : np.sum(np.power(r-origin[np.newaxis, :], 2),axis=1)#function for computing the square of the radius
    #clear points within inner boundary
    positions_reduced, nb_boundary = point_cloud_clear_inner_boundary_generic(remeshed_positions, nb_boundary, radii2, radius**2)
    #use healpy with 12*5**2 directions to define inner sphere
    N_inner_bdy = 12*healpy_order**2
    #healpix always requires 12*x**2 as number of points on the sphere
    direction  = healpy.pixelfunc.pix2vec(healpy.npix2nside(N_inner_bdy), range(N_inner_bdy))
    direction  = np.array(direction).transpose()
    #then multiply with the desired radius of the inner boundary
    inner_bdy = origin + radius * direction
    positions_reduced = np.concatenate((inner_bdy,positions_reduced))
    nb_boundary = nb_boundary + N_inner_bdy

    return positions_reduced, nb_boundary


def point_cloud_add_spherical_outer_boundary(remeshed_positions, nb_boundary, radius, healpy_order = 10, origin = np.array([0.0,0.0,0.0]).T):
    '''
    Function for specifying a spherical outer boundary in a remeshed point cloud.
    First clears the region outside the sphere and then constructs a spherical boundary using healpy.

    Parameters
    ----------
    remeshed_positions : numpy array of float
        Positions of the points in the point cloud. Assumes the boundary points to lie in front.
    nb_boundary : unsigned int
        The number of boundary points in the point cloud.
    radius : positive float
        Radius of the spherical outer boundary
    healpy_order : unsigned int
        The number of points on the outer boundary is defined as: 12*(healpy_order**2), as the healpy discretization of the sphere is used.
    origin : 1 by 3 numpy vector of float
        Contains x,y,z coordinates of center point of the sphere (by default [0.0,0.0,0.0]^T)

    Returns
    -------
    remeshed_positions : numpy array of float
        The positions of the points in the remeshed point cloud. The boundary points lie in front.
    number of boundary points : unsigned int
        The number of boundary points in the remeshed point cloud.
    '''

    radii2 = lambda r : np.sum(np.power(r-origin[np.newaxis, :], 2),axis=1)#function for computing the square of the radius
    #clear points within inner boundary
    #safety factor due to boundary hull being not a perfect sphere
    max_angle = healpy.pixelfunc.max_pixrad(healpy_order)#max angle between two healpy directions
    safety_factor = np.cos(max_angle)
    positions_reduced, nb_boundary = point_cloud_clear_outer_boundary_generic(remeshed_positions, nb_boundary, radii2, (radius*safety_factor)**2)
    #use healpy with 12*5**2 directions to define inner sphere
    N_outer_bdy = 12*healpy_order**2
    #healpix always requires 12*x**2 as number of points on the sphere
    direction  = healpy.pixelfunc.pix2vec(healpy.npix2nside(N_outer_bdy), range(N_outer_bdy))
    direction  = np.array(direction).transpose()
    #then multiply with the desired radius of the inner boundary
    inner_bdy = origin + radius * direction
    positions_reduced = np.concatenate((inner_bdy,positions_reduced))
    nb_boundary = nb_boundary + N_outer_bdy

    return positions_reduced, nb_boundary
