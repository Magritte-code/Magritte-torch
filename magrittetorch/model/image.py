from magrittetorch.model.parameters import Parameters, Parameter
from magrittetorch.utils.storagetypes import DataCollection, StorageTensor, Types
from magrittetorch.model.geometry.geometry import Geometry, GeometryType
from magrittetorch.utils.constants import min_dist
from enum import Enum
from astropy import units
import torch
from scipy.interpolate import NearestNDInterpolator #err, did not find a pytorch solution for nearest neighbour interpolation
import numpy as np
#TODO: if pytorch solution is found, replace usage of scipy function


class ImageType(Enum):
    Intensity = 0
    OpticalDepth = 1#err, for now only intensity is implemented

#for now, I am disconnecting the image class from any storage capabilities, as it will inevitably lead to storage issues
class Image:
    def __init__(self, params: Parameters, dataCollection : DataCollection, imageType: ImageType, ray_direction: torch.Tensor, freqs: torch.Tensor, image_index: int) -> None:
        self.storage_dir : str = "images/image_"+str(image_index)+"/"
        self.parameters: Parameters = params
        self.dataCollection: DataCollection = dataCollection
        self.imageType: Parameter[ImageType] = Parameter[ImageType]("imageType")#; self.dataCollection.add_local_parameter(self.imageType)
        self.imageType.set(imageType)
        self.ray_direction: StorageTensor = StorageTensor(Types.GeometryInfo, [3], units.dimensionless_unscaled, self.storage_dir+"ray_direction", tensor=ray_direction/torch.norm(ray_direction)); #self.dataCollection.add_data(self.ray_direction, "ray direction")
        self.npix: Parameter[int] = Parameter[int](self.storage_dir+"npix")#TODO: maybe add constraints between parameters?
        # self.nxpix: Parameter[int] = Parameter[int](self.storage_dir+"nxpix")
        # self.nypix: Parameter[int] = Parameter[int](self.storage_dir+"nypix")
        self.nfreqs: Parameter[int] = Parameter[int](self.storage_dir+"nfreqs")
        self.freqs: StorageTensor = StorageTensor(Types.FrequencyInfo, [self.nfreqs], units.Hz, self.storage_dir+"freqs", tensor=freqs)#; self.dataCollection.add_data(self.freqs, "frequencies")

        # The images themselves define some coordinate system, this is copied into these StorageTensors
        self.image_direction_x: StorageTensor = StorageTensor(Types.GeometryInfo, [3], units.dimensionless_unscaled, self.storage_dir+"image_direction_x"); #self.dataCollection.add_data(self.image_direction_x, "image direction x")
        self.image_direction_y: StorageTensor = StorageTensor(Types.GeometryInfo, [3], units.dimensionless_unscaled, self.storage_dir+"image_direction_y"); #self.dataCollection.add_data(self.image_direction_y, "image direction y")
        self.image_direction_z: StorageTensor = StorageTensor(Types.GeometryInfo, [3], units.dimensionless_unscaled, self.storage_dir+"image_direction_z"); #self.dataCollection.add_data(self.image_direction_z, "image direction z")

        self.imX: StorageTensor = StorageTensor(Types.GeometryInfo, [self.npix], units.m, self.storage_dir+"imX")
        self.imY: StorageTensor = StorageTensor(Types.GeometryInfo, [self.npix], units.m, self.storage_dir+"imY")

        # self.pixX: StorageTensor = StorageTensor(Types.GeometryInfo, [self.nxpix], units.m, self.storage_dir+"pixX")
        # self.pixY: StorageTensor = StorageTensor(Types.GeometryInfo, [self.nypix], units.m, self.storage_dir+"pixY")

        # Note: I saves the image data; for most images, this is the intensity, but it can also be the optical depth
        # Thus the units might need be changed manually when setting the image type...
        # Note2: As images are currently not saved in a hdf5 file, we do not yet need to store the unit of the image data
        self.I: StorageTensor = StorageTensor(Types.FrequencyInfo, [self.npix, self.nfreqs], units.W*units.m**-2*units.Hz**-1*units.sr**-1, self.storage_dir+"I")#; self.dataCollection.add_data(self.I, "intensity")

    #TODO: add option for imaging limited region
    def setup_image(self, geometry: Geometry, Nxpix: int, Nypix: int, image_type: ImageType = ImageType.Intensity) -> None:
        """
        Sets up the image coordinate system based on the ray direction and the geometry of the model

        Args:
            geometry (Geometry): The geometry of the model.
            Nxpix (int): The number of pixels in the x-direction.
            Nypix (int): The number of pixels in the y-direction.
            image_type (ImageType, optional): The type of image to compute. Determines the unit of the image data. Defaults to ImageType.Intensity.

        Returns:
            None
        """
        # Set the image data unit correctly
        match image_type:
            case ImageType.Intensity:
                self.I.unit = units.W*units.m**-2*units.Hz**-1*units.sr**-1
            case ImageType.OpticalDepth:
                self.I.unit = units.dimensionless_unscaled
            case _:
                raise NotImplementedError("Image type not implemented")

        raydir: torch.Tensor = self.ray_direction.get() #dims: [3]
        # Compute the rotation angles alpha and beta
        alpha = torch.atan2(raydir[0], raydir[1])
        beta = torch.atan2(-raydir[2], torch.sqrt(raydir[0]**2 + raydir[1]**2))

        # Compute the rotation matrices ()
        rot_x = torch.tensor([[torch.cos(alpha), torch.sin(alpha), 0.0],
                            [-torch.sin(alpha), torch.cos(alpha), 0.0],
                            [0.0, 0.0, 1.0]], dtype=Types.GeometryInfo)
        
        rot_y = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, torch.cos(beta), torch.sin(beta)],
                            [0.0, -torch.sin(beta), torch.cos(beta)]], dtype=Types.GeometryInfo)

        rot = torch.matmul(rot_x, rot_y)    

        # Compute the rotation matrices
        # rot_x = torch.tensor([[torch.cos(alpha), 0.0, torch.sin(alpha)],
        #                     [0.0, 1.0, 0.0],
        #                     [-torch.sin(alpha), 0.0, torch.cos(alpha)]], dtype=Types.GeometryInfo)
        # rot_y = torch.tensor([[1.0, 0.0, 0.0],
        #                     [0.0, torch.cos(beta), torch.sin(beta)],
        #                     [0.0, -torch.sin(beta), torch.cos(beta)]], dtype=Types.GeometryInfo)

        # rot = torch.matmul(rot_x, rot_y)

        # Compute the image directions #Note: this set of rotations is not unique, but it is the one used in the original magritte 
        self.image_direction_x.set(torch.matmul(rot, torch.tensor([1.0, 0.0, 0.0], dtype=Types.GeometryInfo)))
        self.image_direction_y.set(torch.matmul(rot, torch.tensor([0.0, 0.0, -1.0], dtype=Types.GeometryInfo)))
        self.image_direction_z.set(torch.matmul(rot, torch.tensor([0.0, 1.0, 0.0], dtype=Types.GeometryInfo)))

        if (geometry.geometryType.get() == GeometryType.General3D):
            coords = geometry.points.position.get()
            proj_coords_x = torch.matmul(coords, self.image_direction_x.get())
            proj_coords_y = torch.matmul(coords, self.image_direction_y.get())

            xmin = torch.min(proj_coords_x)
            xmax = torch.max(proj_coords_x)
            ymin = torch.min(proj_coords_y)
            ymax = torch.max(proj_coords_y)

        elif (geometry.geometryType.get() == GeometryType.SpericallySymmetric1D):
            max_coords1D = torch.max(geometry.points.position.get()[:,0]) #dims:[1]
            xmin = -max_coords1D
            xmax = max_coords1D
            ymin = -max_coords1D
            ymax = max_coords1D

        # Compute the image pixel coordinates (in 2D)
        x: torch.Tensor = torch.linspace(xmin, xmax, Nxpix, dtype=Types.GeometryInfo)
        y: torch.Tensor = torch.linspace(ymin, ymax, Nypix, dtype=Types.GeometryInfo)
        # self.pixX.set(x)
        # self.pixY.set(y)
        self.imX.set(x.view(-1, 1).repeat(1, Nypix).flatten())
        self.imY.set(y.view(1, -1).repeat(Nxpix, 1).flatten())

        return
    
    def transform_pixel_coordinates_to_3D_starting_coordinates(self, geometry: Geometry, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the 3D starting coordinates and boundary point indices for a 3D geometry
        Warning: no gradients are propagated for the indices computed in this function, as scipy is used for unstructured grid interpolation

        Args:
            geometry (Geometry): Geometry of the model
            device (torch.device): Device on which to compute
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: The corresponding boundary positions and point indices of the boundary points. These have dimensions [self.npix, 3], [self.npix] respectively
            The boundary indices belong to ∈ [0, parameters.npoints.get()-1]

        Raises:
            AssertionError: when called with a geometry with GeometryType other than General3D
        """
        if (geometry.geometryType.get() != GeometryType.General3D):
            raise AssertionError("This method can only be called for general 3D geometries")
        raydir: torch.Tensor = self.ray_direction.get(device) #dims: [3]
        positions_device = geometry.points.position.get(device) #dims: [parameters.npoints, 3]
        #start at the reverse direction boundary
        boundary_boolean = geometry.boundary.get_boundary_in_direction(-raydir, device) #dims: [parameters.npoints]
        #get the boundary positions and indices from the boolean tensor
        boundary_indices = boundary_boolean.nonzero().squeeze(1) #dims: [NBOUNDARY]
        boundary_positions = positions_device[boundary_boolean, :] #dims: [NBOUNDARY, 3]

        # project the boundary positions onto the image plane
        boundary_positions_xpix_coords = torch.matmul(boundary_positions, self.image_direction_x.get(device)) #dims: [NBOUNDARY]
        boundary_positions_ypix_coords = torch.matmul(boundary_positions, self.image_direction_y.get(device)) #dims: [NBOUNDARY]
        boundary_positions_min_z_coord = torch.min(torch.matmul(boundary_positions, self.image_direction_z.get(device))) #dims: [1]

        #Warning: scipy functions used here; take care to properly convert everything to numpy arrays
        # This also means that no gradients are propagated through this part
        NDinterpolator = NearestNDInterpolator(torch.cat((boundary_positions_xpix_coords.unsqueeze(1), boundary_positions_ypix_coords.unsqueeze(1)), dim=1).numpy(force=True), boundary_indices.numpy(force=True))
        scipy_indices = NDinterpolator(self.imX.get(device).numpy(force=True), self.imY.get(device).numpy(force=True)) #dims: [self.npix]
        corresponding_bdy_indices = torch.tensor(scipy_indices, dtype=Types.IndexInfo, device=device) #dims: [self.npix]
        #End of scipy usage

        start_pos = self.imX.get(device)[:, None] * self.image_direction_x.get(device)[None, :] + self.imY.get(device)[:, None] * self.image_direction_y.get(device)[None, :] + boundary_positions_min_z_coord * self.image_direction_z.get(device)[None, :] #dims: [self.npix, 3]

        return (start_pos, corresponding_bdy_indices)

    
    def transform_pixel_coordinates_to_1D_starting_coordinates(self, geometry: Geometry, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Transforms the pixel coordinates to 3D starting coordinates for a 1D spherically symmetric model.

        Args:
            geometry (Geometry): The geometry of the model
            device (torch.device): The device on which to compute

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The corresponding boundary positions and point indices of the boundary points. These have dimensions [self.npix, 3], [self.npix] respectively

        Raises:
            AssertionError: when called with a geometry with GeometryType other than General3D
        """
        if (geometry.geometryType.get() != GeometryType.SpericallySymmetric1D):
            raise AssertionError("This method can only be called for spherically symmetric 1D geometries")       
        raydir: torch.Tensor = self.ray_direction.get() #dims: [3] 
        #I will put all points at the maximal radius away from the center
        maxdist = torch.max(geometry.points.position.get(device)[:,0]) #dims: [1]
        projected_x_pixel_coordinates = self.imX.get(device)[:, None] * self.image_direction_x.get(device)[None, :] #dims: [NPOINTS, 3]
        projected_y_pixel_coordinates = self.imY.get(device)[:, None] * self.image_direction_y.get(device)[None, :] #dims: [NPOINTS, 3]
        projected_total_pixel_coordinates = projected_x_pixel_coordinates + projected_y_pixel_coordinates #dims: [NPOINTS, 3]
        #And move the starting positions just outside the boundary in the negative ray direction
        starting_positions = projected_total_pixel_coordinates - maxdist * raydir[None, :] #dims: [NPOINTS, 3]

        return starting_positions, (geometry.parameters.npoints.get()-1) * torch.ones(self.npix.get(), dtype=Types.IndexInfo, device=device) #We start at the outermost shell
    
