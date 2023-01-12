import numpy as np
import pinocchio as pin
import cv2

# Compute pose from tag corners
# tag_corners_3d: tag corners in 3d, in the plane z=0 (4x3)
# detected_corners: tag corners in the image (4x2)
# caemra_matrix: intrinsic matrix (3x3)
# distortion vector: according to OpenCV specs
# T: translation vector (3x1)
# w: rotation vector (3x1)
# R: rotation matrix R = exp3(w)
# so that p_camera = T + R * p_tag
def poseFromCorners(tag_corners_3d, detected_corners, camera_matrix, distortion_vector):
    (_, rotation_vector, translation_vector) = cv2.solvePnP(tag_corners_3d, detected_corners, camera_matrix, distortion_vector, flags = cv2.SOLVEPNP_IPPE_SQUARE)
    T = translation_vector
    w = rotation_vector
    R = pin.exp3(w)
    return T, w, R

# Project the 4 corners of a tag onto a camera K at pos T and ori R
# R is rot matrix of tag wrt camera (3x3)
# T is translation vector of tag wrt camera (3x1)
# K is camera matrix (3x3)
# tag_corners_3d are the 4 corners of tag in 3d (4x3)
# result is projected tag corners (4x2)
def projectTagCorners(R, T, K, tag_corners_3d):
    projected_corners_h = (K @ (R @ tag_corners_3d.T + np.hstack([T,T,T,T]))).T
    projected_corners = np.zeros([4,2])
    for row in range(projected_corners_h.shape[0]):
        projected_corners[row,:] = projected_corners_h[row,0:2] / projected_corners_h[row,2]
    return projected_corners

