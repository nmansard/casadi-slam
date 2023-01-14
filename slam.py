

import numpy as np
import time
import pinocchio as pin
import casadi
from pinocchio import casadi as cpin
from utils.meshcat_viewer_wrapper import MeshcatVisualizer
import apriltag
import cv2
from gslam_types import *
from gslam_april_tools import *



#-----------------------------------------------------------------------------------
# COST FUNCTIONS
#-----------------------------------------------------------------------------------

# motion -- constant position
def cost_constant_position(sqrt_info, keyframe_i, keyframe_j):
    ppred = keyframe_i.R.T @ (keyframe_j.position - keyframe_i.position)
    wpred = log3(exp3(-keyframe_i.anglevector) @ exp3(keyframe_j.anglevector)) # log ( Ri.T * Rj )
    pred  = casadi.vertcat( ppred, wpred )

    cost = casadi.sumsqr(sqrt_info @ pred) 
    return cost

# landmark observations
def cost_landmark(meas, sqrt_info, keyframe_i, landmark_j):
    # compose: landmark pose wrt KF pose
    ppred = keyframe_i.R.T @ (landmark_j.position - keyframe_i.position)
    Rpred = keyframe_i.R.T @ landmark_j.R

    # error: use manifold tools for the orientation part
    perr = meas[0:3] - ppred
    Rerr = Rpred.T @ exp3(meas[3:6])
    werr = log3(Rerr)

    err = casadi.vertcat(perr, werr)
 
    # apply weight and compute squared norm
    cost = casadi.sumsqr(sqrt_info @ err) 
    return cost

# Prior on first KF
def cost_keyframe_prior(meas, sqrt_info, keyframe_i):
    perr = meas[0:3] - keyframe_i.position
    werr = log3(keyframe_i.R.T @ exp3(meas[3:6]))
    err  = casadi.vertcat( perr, werr )

    cost = casadi.sumsqr(sqrt_info @ err) 
    return cost

# Prior on one landmark
def cost_landmark_prior(meas, sqrt_info, landmark):
    perr = meas[0:3] - landmark.position
    werr = log3(landmark.R.T @ exp3(meas[3:6]))
    err  = casadi.vertcat( perr, werr )

    cost = casadi.sumsqr(sqrt_info @ err) 
    return cost


def computeTotalCost(factors, keyframes, landmarks):
    totalcost = 0
    for fid in factors:
        factor = factors[fid]
        i = factor.i
        j = factor.j
        measurement = factor.meas
        sqrt_info   = factor.sqrt_info
        if factor.type == "motion":
            totalcost += cost_constant_position (sqrt_info, keyframes[i], keyframes[j])
        elif factor.type == "landmark":
            totalcost += cost_landmark (measurement, sqrt_info, keyframes[i], landmarks[j])
        elif factor.type == "prior":
            totalcost += cost_keyframe_prior (measurement, sqrt_info, keyframes[i])
        else:
            print('Error in the factor type: type not known')
    return totalcost


#-----------------------------------------------------------------------------------
# PROBLEM DATA
#-----------------------------------------------------------------------------------

# camera and image files
K           = np.array([[   419.53, 0.0,    427.88  ], 
                        [   0.0,    419.53, 241.32  ], 
                        [   0.0,    0.0,    1.0     ]])
file_tmp    = './data/visual_odom_laas_corridor/short2/frame{num:04d}.jpg'

# AprilTag specifications
tag_family  = 'tag36h11'
tag_size    = 0.168
tag_corners = tag_size / 2 * np.array([[-1,1,0],[1,1,0],[1,-1,0],[-1,-1,0]])

# initial conditions
initial_position    = np.array([0,0,0])
initial_orientation = np.array([0,0,0])

#-----------------------------------------------------------------------------------
# INIT THE PROBLEM
#-----------------------------------------------------------------------------------

# AprilTag detector
detector    = apriltag.Detector()

# Create the casadi optimization problem
opti = casadi.Opti()
opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
opti.solver("ipopt", opts)

# Display
viz = MeshcatVisualizer()

# time and ID counters
t       = 0
kf_id   = 0
fac_id  = 0

# object dictionaries
keyframes = dict()
landmarks = dict()
factors   = dict()

#-----------------------------------------------------------------------------------
# TEMPORAL LOOP
#-----------------------------------------------------------------------------------

# process all images in the sequence
first_time = True
while(t <= 25):

    # read one image
    filename = file_tmp.format(num=t)
    print('reading image file:', filename)
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) 
    if image is None:
        break

    cv2.imshow("Image view", image)
    cv2.waitKey(2) # waits until a key is pressed

    
    # make KF for new image
    if first_time: # first time, make new KF, set initial pose and prior factor
        first_time = False
        # new KF
        keyframe = OptiVariablePose3(opti, kf_id, initial_position, initial_orientation)
        keyframes[kf_id] = keyframe
        # prior factor
        measurement = casadi.vertcat(initial_position, initial_orientation)
        sqrt_info   = 1e4 * np.eye(6)
        factors[fac_id] = Factor("prior", fac_id, kf_id, 0, measurement, sqrt_info)

        # store KF values for later
        kf_p = initial_position
        kf_w = initial_orientation
        kf_R = pin.exp3(kf_w)

    else: # other times, make new KF, add factor to last KF
        # we implement a constant-position motion model
        # so we make new KF at same position than last KF

        # recover last KF
        kf_last_id = kf_id
        kf_last_pos = opti.value(keyframe.position)
        kf_last_ori = opti.value(keyframe.anglevector)

        # make new KF
        kf_id += 1
        keyframe = OptiVariablePose3(opti, kf_id, kf_last_pos, kf_last_ori)
        keyframes[kf_id] = keyframe

        # add a constant_position factor between both KF
        fac_id += 1
        measurement = np.zeros([6,1]) # no motion
        sqrt_info   = np.eye(6) / 1e3 # 1000m, 1000rad, uncertainty of motion
        factor      = Factor('motion', fac_id, kf_last_id, kf_id, measurement, sqrt_info)
        factors[fac_id] = factor

        # store KF values for later
        kf_p = kf_last_pos
        kf_w = kf_last_ori
        kf_R = pin.exp3(kf_w)

    # process image detections
    detections   = detector.detect(image)
    for detection in detections:
        lmk_id           = detection.tag_id
        detected_corners = detection.corners

        print('Tag     #', lmk_id, 'detected in image')
        
        # compute pose of tag wrt camera
        T_c_t, R_c_t, w_c_t = poseFromCorners(tag_corners, detected_corners, K, np.array([]))

        # build measurement vector and sqrt info matrix
        measurement = casadi.vertcat(T_c_t, w_c_t)
        sqrt_info   = np.eye(6) / 1e-2 # noise of 1 cm ; 0.01 rad

        # print some quality metrics
        print('detection hamming  = ', detection.hamming)
        print('detection goodness = ', detection.goodness)
        print('decision margin    = ', detection.decision_margin)
        projected_corners = cv2.projectPoints(tag_corners, R_c_t, T_c_t, K, np.array([]))
        projected_corners   = np.reshape(projected_corners[0],[4,2])  # fix weird format from opencv
        print('reproj. error      = ', reprojectionError(projected_corners, detected_corners))
    
        if lmk_id in landmarks: # found: known landmark: only add factor
            print('Landmark #', lmk_id, 'found in graph')
            fac_id += 1
            factors[fac_id] = Factor('landmark', fac_id, kf_id, lmk_id, measurement, sqrt_info)
            print('Factor   #', fac_id, 'appended to graph')

        else: # not found: new landmark
            print('Landmark #', lmk_id, 'not found in graph')
            # lmk pose in world coordinates
            lmk_p = kf_p + kf_R @ T_c_t
            lmk_R = kf_R @ R_c_t
            lmk_w = pin.log3(lmk_R)

            # construct and append new lmk
            landmark = OptiVariablePose3(opti, lmk_id, lmk_p, lmk_w)
            landmarks[lmk_id] = landmark
            print('Landmark #', lmk_id, 'appended to graph')

            # construct and append new factor
            fac_id += 1
            factors[fac_id] = Factor('landmark', fac_id, kf_id, lmk_id, measurement, sqrt_info)
            print('Factor   #', fac_id, 'appended to graph')


    # optimize!
    totalcost = computeTotalCost(factors, keyframes, landmarks)
    opti.minimize(totalcost)
    sol = opti.solve()

    # display
    drawAll(opti, keyframes, landmarks, factors, viz)
    time.sleep(1e-2)
    print('-----------------------------------')

    # advance time
    t += 1

###############################################################################################
    
# print results
print()
for lid in landmarks:
    landmark = landmarks[lid]
    lmk_p = opti.value(landmark.position)
    lmk_w = opti.value(landmark.anglevector)
    print('lmk id: ', landmark.id, '\tpos: ', lmk_p, '\tori: ', lmk_w)
for kid in keyframes:
    keyframe = keyframes[kid]
    kf_p = opti.value(keyframe.position)
    kf_w = opti.value(keyframe.anglevector)
    print('kf  id: ', keyframe.id, '\tpos: ', kf_p, '\tori: ', kf_w)



cv2.destroyAllWindows() # destroys the window showing image
