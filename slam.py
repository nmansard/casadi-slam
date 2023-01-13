

import casadi
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.utils import rotate
import numpy as np
from utils.meshcat_viewer_wrapper import MeshcatVisualizer
import apriltag
import cv2
# from april import *
from gslam_april_tools import *
import time


#-----------------------------------------------------------------------------------
# BASIC HELPER FUNCTIONS
#-----------------------------------------------------------------------------------


# S is a skew-symmetric matrix
# s is the 3-vector extracted from S
def wedge(S):
    s = casadi.vertcat(S[2,1], S[0,2], S[1,0])
    return s
    
# R is a rotation matrix not far from the identity
# w is the approximated rotation vector equivalent to R
def log3_approx(R):
    w = wedge(R - R.T) / 2
    return w

# Find the index in the list according to object id
# if not found, return -1
def find_index(list, id):
    ids = [item.id for item in list]
    try:
        idx = ids.index(id)
    except ValueError:
        idx = -1
    return idx

#-----------------------------------------------------------------------------------
# CASADI HELPER FUNCTIONS
#-----------------------------------------------------------------------------------

cw = casadi.SX.sym("w", 3, 1)
cR = casadi.SX.sym("R", 3, 3)

exp3 = casadi.Function("exp3", [cw], [cpin.exp3(cw)])
#log3 = casadi.Function("logp3", [cR], [cpin.log3(cR)])
log3 = casadi.Function("log3", [cR], [log3_approx(cR)])

#-----------------------------------------------------------------------------------
# Help on data types
#-----------------------------------------------------------------------------------

# keyframe:
# - position p
# - orientation w

# landmarks:
# - position p
# - orientation w

# factors:
# - index first state i
# - index second state j
# - type "motion" "landmark" "prior"
# - measurement: 6-vector, translation and rotation
# - sqrt_info: 6x6 matrix


#-----------------------------------------------------------------------------------
# METHODS
#-----------------------------------------------------------------------------------

# define cost functions

# motion -- constant position
def cost_constant_position(sqrt_info, keyframe_i, keyframe_j):
    ppred = keyframe_j.position - keyframe_i.position
    wpred = log3(exp3(-keyframe_i.anglevector) @ exp3(keyframe_j.anglevector)) # log ( Ri.T * Rj )
    pred  = casadi.vertcat( ppred, wpred )

    cost = casadi.sumsqr(sqrt_info @ pred) 
    return cost

# landmark observations
def cost_landmark(meas, sqrt_info, keyframe_i, landmark_j):
    ppred = landmark_j.position - keyframe_i.position
    Rpred = keyframe_i.R.T @ landmark_j.R

    perr = meas[0:3] - ppred
    Rerr = Rpred.T @ exp3(meas[3:6])
    werr = log3(Rerr)

    err = casadi.vertcat(perr, werr)
 
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


# class for keyframes and landmarks
class OptiVariablePose3:
    def __init__(self, opti, id, position, anglevector):
        self.id             = id
        self.position       = opti.variable(3)
        self.anglevector    = opti.variable(3)
        self.R              = exp3(self.anglevector)
        opti.set_initial(self.position, position)
        opti.set_initial(self.anglevector, anglevector)

# class for factors
class Factor:
    def __init__(self, type, id, i, j, meas, sqrt_info):
        self.type = type
        self.id = id
        self.i = i
        self.j = j
        self.meas = meas
        self.sqrt_info = sqrt_info

def computeTotalCost(factors, keyframes, landmarks):
    totalcost = 0
    for factor in factors:
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

# camera, detector, and image files
K           = np.array([[   419.53, 0.0,    427.88  ], 
                        [   0.0,    419.53, 241.32  ], 
                        [   0.0,    0.0,    1.0     ]])
detector    = apriltag.Detector()
file_tmp    = './data/visual_odom_laas_corridor/short2/frame{num:04d}.jpg'

# AprilTag definitions
tag_family  = 'tag36h11'
tag_size    = 0.168
tag_corners = tag_size / 2 * np.array([[-1,1,0],[1,1,0],[1,-1,0],[-1,-1,0]])

# initial conditions
initial_position    = np.array([0,0,0])
initial_orientation = np.array([0,0,0])

# Create the casadi optimization problem
opti = casadi.Opti()
opti.solver("ipopt")

# Display
viz = MeshcatVisualizer()


#-----------------------------------------------------------------------------------
# INIT THE PROBLEM
#-----------------------------------------------------------------------------------

# time and ID counters
t       = 0
kf_id   = 0
fac_id  = 0

# init object lists
factors   = list()
keyframes = list()
landmarks = list()

#-----------------------------------------------------------------------------------
# TEMPORAL LOOP
#-----------------------------------------------------------------------------------

# process all images in the sequence
first_time = True
while(t <= 250):
    # read one image
    filename = file_tmp.format(num=t)
    print('reading image file:', filename)
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) 
    if image is None:
        break
    
    # make KF for new image
    if first_time: # first time, make new KF, set initial pose and prior factor
        first_time = False
        keyframe = OptiVariablePose3(opti, kf_id, initial_position, initial_orientation)
        keyframes.append(keyframe)
        factors.append(Factor("prior", fac_id, kf_id, 0, np.array([0,0,0, 0,0,0]), 1e4 * np.eye(6)))

    else: # other times, make new KF, add factor to last KF
        # we implement a constant-position motion model
        # so we make new KF at same position than last KF

        # recover last KF
        kf_last_idx = find_index(keyframes, kf_id)
        kf_last_pos = opti.value(keyframe.position)
        kf_last_ori = opti.value(keyframe.anglevector)

        # make new KF
        kf_id += 1
        keyframe = OptiVariablePose3(opti, kf_id, kf_last_pos, kf_last_ori)
        keyframes.append(keyframe)
        kf_new_idx = kf_last_idx + 1

        # add a constant_position factor between both KF
        fac_id += 1
        factor = Factor('motion', fac_id, kf_last_idx, kf_new_idx, np.zeros([6,1]), 1e-2 * np.eye(6))
        factors.append(factor)

    # optimize!
    # solve so that opti.values(...) work
    totalcost = computeTotalCost(factors, keyframes, landmarks)
    opti.minimize(totalcost)
    sol = opti.solve()

    kf_idx = find_index(keyframes, kf_id)
    kf_p = opti.value(keyframe.position)
    kf_w = opti.value(keyframe.anglevector)
    kf_R = pin.exp3(kf_w)

    # process image detections
    detections   = detector.detect(image)
    for detection in detections:
        lmk_id           = detection.tag_id
        print('Tag     #', lmk_id, 'detected in image')

        detected_corners = detection.corners
        
        # compute pose of tag wrt camera
        T_c_t, R_c_t, w_c_t = poseFromCorners(tag_corners, detected_corners, K, np.array([]))

        projected_corners = projectTagCorners(T_c_t, R_c_t, K, tag_corners)

        print(detected_corners)
        print(projected_corners)
    
        measurement     = casadi.vertcat(T_c_t, w_c_t)
        sqrt_info       = np.eye(6) / 1e-2 # noise of 1 cm ; 0.01 rad

        lmk_idx = find_index(landmarks, lmk_id)
        if lmk_idx >= 0: # found: known landmark: only add factor
            print('Landmark #', lmk_id, 'found in map')
            fac_id += 1
            factors.append(Factor('landmark', fac_id, kf_idx, lmk_idx, measurement, sqrt_info))
            print('Factor   #', fac_id, 'appended to map')

        else: # not found: new landmark
            print('Landmark #', lmk_id, 'not found in map')
            # lmk pose in world coordinates
            lmk_p = kf_p + kf_R @ T_c_t
            lmk_R = kf_R @ R_c_t
            lmk_w = pin.log3(lmk_R)

            # construct and append new lmk
            landmark = OptiVariablePose3(opti, lmk_id, lmk_p, lmk_w)
            landmarks.append(landmark)
            print('Landmark #', lmk_id, 'appended to map')
            lmk_idx = find_index(landmarks, lmk_id)
            # construct and append new factor
            fac_id += 1
            factors.append(Factor('landmark', fac_id, kf_idx, lmk_idx, measurement, sqrt_info))
            print('Factor   #', fac_id, 'appended to map')


    # optimize!
    totalcost = computeTotalCost(factors, keyframes, landmarks)
    opti.minimize(totalcost)
    sol = opti.solve()

    # display
    drawAll(opti, keyframes, landmarks, factors, viz)
    time.sleep(1e-2)
    print('finished t=',t)

    # advance time
    t += 5

###############################################################################################
    
# print results
print()
for landmark in landmarks:
    lmk_p = opti.value(landmark.position)
    lmk_w = opti.value(landmark.anglevector)
    print('lmk id: ', landmark.id, '\tpos: ', lmk_p, '\tori: ', lmk_w)
for keyframe in keyframes:
    kf_p = opti.value(keyframe.position)
    kf_w = opti.value(keyframe.anglevector)
    print('kf  id: ', keyframe.id, '\tpos: ', kf_p, '\tori: ', kf_w)








