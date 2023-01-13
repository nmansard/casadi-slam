import numpy as np
import pinocchio as pin
import cv2

# Compute pose from tag corners
# tag_corners_3d: tag corners in 3d, in the plane z=0 (4x3)
# detected_corners: tag corners in the image (4x2)
# caemra_matrix: intrinsic matrix (3x3)
# distortion vector: according to OpenCV specs
# Pose: this is the pose of the tag with respect to the camera. It consists of:
#   T_c_t: translation vector (3x1)
#   w_c_t: rotation vector (3x1)
#   R_c_t: rotation matrix R_c_t = exp3(w_c_t)
# so that p_camera = T_c_t + R_c_t * p_tag
def poseFromCorners(tag_corners_3d, detected_corners, camera_matrix, distortion_vector):
    (_, rotation_vector, translation_vector) = cv2.solvePnP(tag_corners_3d, detected_corners, camera_matrix, distortion_vector, flags = cv2.SOLVEPNP_IPPE_SQUARE)
    T_c_t = (translation_vector.T)[0]
    w_c_t = (rotation_vector.T)[0]
    R_c_t = pin.exp3(w_c_t)
    return T_c_t, R_c_t, w_c_t

# Project the 4 corners of a tag onto a camera K at pos T and ori R
# R is rot matrix of camera wrt tag (3x3)
# T is translation vector of camera wrt tag (3x1)
# K is camera matrix (3x3)
# tag_corners_3d are the 4 corners of tag in 3d (4x3)
# result is projected tag corners (4x2)
def projectTagCorners(T, R, K, tag_corners_3d):
    projected_corners_h = (K @ (R @ tag_corners_3d.T + np.vstack([T,T,T,T]).T)).T
    projected_corners = np.zeros([4,2])
    for row in range(projected_corners_h.shape[0]):
        projected_corners[row,:] = projected_corners_h[row,0:2] / projected_corners_h[row,2]
    return projected_corners

# invert a 3d pose (non casadi)
def invertPose(T, R):
    Ri = R.T
    Ti = - Ri @ T
    wi = pin.log3(Ri)
    return Ti, Ri, wi
    
# Draw all lobjects
def drawAll(opti, keyframes, landmarks, factors, viz):
    '''
    position et orientation tags: faire un petit carreau et bien le positionner en 3d
    position et orientation camera: faire un petit prisma 3d pour chaque keyframe et bien les positionner
    lmk factors: faire une ligne de chaque KF a chaque LMK. Prendre keyframe.position et landmark.position comme extremes, coulour rouge
    motion factors: pareil avec des KFs consecutifs, prendre couleur bleu
    '''
    for landmark in landmarks:
        lmk_p = opti.value(landmark.position)
        lmk_w = opti.value(landmark.anglevector)
        lmk_M = pin.SE3(pin.exp3(lmk_w),lmk_p)
    
        lid = f"lmk_{landmark.id:4}"
        viz.addBox(lid, [0.2, 0.2, 0.005], [0.9, 0.9, 0.9, 0.8])
        viz.applyConfiguration(lid,lmk_M)
    
    for keyframe in keyframes:
        kf_p = opti.value(keyframe.position)
        kf_w = opti.value(keyframe.anglevector)
        kf_M = pin.SE3(pin.exp3(kf_w),kf_p)

        kid = f"kf_{keyframe.id:4}"
        viz.addBox(kid, [0.05, 0.05, 0.1], [0.1, 0.1, 0.8, 0.3])
        viz.applyConfiguration(kid,kf_M)
