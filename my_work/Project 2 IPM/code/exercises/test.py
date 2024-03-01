import sys
from pathlib import Path
sys.path.append(str(Path('../')))

from exercises import camera_geometry
# from solutions import camera_geometry

import numpy as np
import matplotlib.pyplot as plt
import cv2

# obtain raw image
image_fn = str(Path("../../data/sample.png").absolute())
image = cv2.imread(image_fn)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# obtain gt world coordinates of the lane boundaries
boundary_fn = image_fn.replace(".png", "_boundary.txt")
boundary_gt = np.loadtxt(boundary_fn)

# obtain transformation matrix from world coordinates to camera reference frame
trafo_fn = image_fn.replace(".png", "_trafo.txt")
trafo_world_to_cam = np.loadtxt(trafo_fn)

cg = camera_geometry.CameraGeometry()
K = cg.intrinsic_matrix

# obtain left lane boundary gt coordinates
left_boundary_3d_gt_world = boundary_gt[:,0:3]

# transform world coordinates into the camera coordinate system, and then project them to image coordinates (u,v)
uv = camera_geometry.project_polyline(left_boundary_3d_gt_world, trafo_world_to_cam, K)
u,v = uv[:,0], uv[:,1]

# # visualization
# plt.plot(u,v)
# plt.imshow(image)
# plt.savefig("raw_image.jpg")


# Reconstruct the left boundary starting from the known u,v
reconstructed_lb_3d_cam = []
for u,v in uv:
    xyz = cg.uv_to_roadXYZ_camframe(u,v)
    reconstructed_lb_3d_cam.append(xyz)
reconstructed_lb_3d_cam = np.array(reconstructed_lb_3d_cam)

# Map reconstructed left boundary into world reference frame
def map_between_frames(points, trafo_matrix):
    x,y,z = points[:,0], points[:,1], points[:,2]
    homvec = np.stack((x,y,z,np.ones_like(x)))
    return (trafo_matrix @ homvec).T

trafo_cam_to_world = np.linalg.inv(trafo_world_to_cam)
reconstructed_lb_3d_world = map_between_frames(reconstructed_lb_3d_cam, trafo_cam_to_world)

# # plot both ground truth and reconstructed left boundary 3d in X-Y-plane
# plt.plot(left_boundary_3d_gt_world[:,0], left_boundary_3d_gt_world[:,1], label="ground truth")
# plt.plot(reconstructed_lb_3d_world[:,0], reconstructed_lb_3d_world[:,1], ls = "--", label="reconstructed")
# plt.axis("equal")
# plt.legend()
# plt.savefig("boundary_world_frame.jpg")

# compare ground truth and reconstructed boundary in road frame
trafo_world_to_road = cg.trafo_cam_to_road @ trafo_world_to_cam
left_boundary_3d_gt_road = map_between_frames(left_boundary_3d_gt_world, trafo_world_to_road)
reconstructed_lb_3d_road = map_between_frames(reconstructed_lb_3d_cam, cg.trafo_cam_to_road)

# # plot both ground truth and reconstructed left boundary 3d in Z-(-X)-plane (which is X-Y in road iso 8855)
# plt.plot(left_boundary_3d_gt_road[:,2], -left_boundary_3d_gt_road[:,0], label="ground truth")
# plt.plot(reconstructed_lb_3d_road[:,2], -reconstructed_lb_3d_road[:,0], ls = "--", label="reconstructed")
# plt.axis("equal")
# plt.legend()
# plt.savefig("boundary_road_frame_1.jpg")

# Reconstruct the left boundary starting from the known u,v
reconstructed_lb_3d_road_iso = []
for u,v in uv:
    xyz = cg.uv_to_roadXYZ_roadframe_iso8855(u,v)
    reconstructed_lb_3d_road_iso.append(xyz)
reconstructed_lb_3d_road_iso = np.array(reconstructed_lb_3d_road_iso)


plt.plot(left_boundary_3d_gt_road[:,2], -left_boundary_3d_gt_road[:,0], label="ground truth")
plt.plot(reconstructed_lb_3d_road_iso[:,0], reconstructed_lb_3d_road_iso[:,1], ls = "--", label="reconstructed")
plt.axis("equal")
plt.legend()
plt.savefig("boundary_road_frame_2.jpg")
