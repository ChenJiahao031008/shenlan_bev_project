import numpy as np

def get_intrinsic_matrix(field_of_view_deg, image_width, image_height):
    """
    Returns intrinsic matrix K.
    """
    # For our Carla camera alpha_u = alpha_v = alpha
    # alpha can be computed given the cameras field of view via
    field_of_view_rad = field_of_view_deg * np.pi/180
    alpha = (image_width / 2.0) / np.tan(field_of_view_rad / 2.)
    Cu = image_width / 2.0
    Cv = image_height / 2.0
    return np.array([[alpha, 0, Cu],
                     [0, alpha, Cv],
                     [0, 0, 1.0]])

def project_polyline(polyline_world, trafo_world_to_cam, K):
    """
    Returns array uv which contains the pixel coordinates of the polyline.

    Parameters
    ----------
    polyline_world : array_like, shape (M,3)
        Each row of this array is a vertex (x,y,z) of the polyline.
    trafo_world_to_cam : array_like, shape (4,4)
        Transformation matrix, that maps vectors (x_world, y_world, z_world, 1) 
        to vectors (x_cam, y_cam, z_cam, 1).
    K: array_like, shape (3,3)
        Intrinsic matrix of  the camera.   
    
    Returns:
    --------
    uv : ndarray, shape (M,2)
        Pixel coordinates of the projected polyline
        First column is u, second column is v
    """
    x,y,z = polyline_world[:,0], polyline_world[:,1], polyline_world[:,2]
    homvec = np.stack((x,y,z,np.ones_like(x)))
    proj_mat = K @ trafo_world_to_cam[:3,:]
    pl_uv_cam = (proj_mat @ homvec).T
    u = pl_uv_cam[:,0] / pl_uv_cam[:,2]
    v = pl_uv_cam[:,1] / pl_uv_cam[:,2]
    return np.stack((u,v)).T


class CameraGeometry(object):
    def __init__(self, height=1.3, yaw_deg=0, pitch_deg=-5, roll_deg=0, image_width=1024, image_height=512, field_of_view_deg=45):
        # scalar constants
        self.height = height
        self.pitch_deg = pitch_deg
        self.roll_deg = roll_deg
        self.yaw_deg = yaw_deg
        self.image_width = image_width
        self.image_height = image_height
        self.field_of_view_deg = field_of_view_deg
        # camera intriniscs and extrinsics
        self.intrinsic_matrix = get_intrinsic_matrix(field_of_view_deg, image_width, image_height)
        self.inverse_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)
        ## Note that "rotation_cam_to_road" has the math symbol R_{rc} in the book
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        roll = np.deg2rad(roll_deg)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        rotation_road_to_cam = np.array([[cr*cy+sp*sr+sy, cr*sp*sy-cy*sr, -cp*sy],
                                            [cp*sr, cp*cr, sp],
                                            [cr*sy-cy*sp*sr, -cr*cy*sp -sr*sy, cp*cy]])
        self.rotation_cam_to_road = rotation_road_to_cam.T # for rotation matrices, taking the transpose is the same as inversion

        # TODO: replace the 'None' values in the following code with correct expressions
        
        self.translation_cam_to_road = None
        self.trafo_cam_to_road = None
        # compute vector nc. Note that R_{rc}^T = R_{cr}
        self.road_normal_camframe = None


    def camframe_to_roadframe(self,vec_in_cam_frame):
        """
        Transform coordinates from camera reference frame to road reference frame.

        Parameters
        ----------
        vec_in_cam_frame: array_like, shape(3,)
            Three dimensional point in the camera reference frame that lies on the road
        
        Returns:
        --------
        XYZ: array_like, shape(3,)
            Three dimensional point in the road reference frame that lies on the road
        """
        # TODO: Write this function
        raise NotImplementedError

    def uv_to_roadXYZ_camframe(self,u,v):
        """
        Inverse perspective mapping from pixel coordinates to 3d coordinates.
        
        Parameters
        ----------
        u,v: Both float
            Pixel coordinates of some part of the road.
        
        Returns:
        --------
        XYZ: array_like, shape(3,)
            Three dimensional point in the camera reference frame that lies on the road
            and was mapped by the camera to pixel coordinates u,v
        """
        # TODO: Write this function
        raise NotImplementedError
    

    def uv_to_roadXYZ_roadframe_iso8855(self,u,v):
        """
        Inverse perspective mapping from pixel coordinates to 3d coordinates in road fram iso8855.
        
        Parameters
        ----------
        u,v: Both float
            Pixel coordinates of some part of the road.
        
        Returns:
        --------
        XYZ: array_like, shape(3,)
            Three dimensional point in the road reference frame iso8855 that lies on the road
            and was mapped by the camera to pixel coordinates u,v
        """
        # TODO: Write this function
        raise NotImplementedError
