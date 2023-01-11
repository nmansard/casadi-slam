

import casadi
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.utils import rotate
import numpy as np
import time
from utils.meshcat_viewer_wrapper import MeshcatVisualizer
import apriltag
import cv2

# BASIC HELPER FUNCTIONS


# s is a 3-vector
# S is a skew-symmetric matrix built from s
def hat(s):
    S = np.array([[0, -s[2], s[1]], [s[2], 0, -s[0]], [-s[1], s[0], 0]])
    return S

# S is a skew-symmetric matrix
# s is the 3-vector extracted from S
def wedge(S):
    s = np.array([S[2,1], S[0,2], S[1,0]])
    return s
    
# R is a rotation matrix not far from the identity
# w is the approximated rotation vector equivalent to R
def log3_approx(R):
    w = wedge(R - R.T) / 2
    return w


def find_index(list, id):
    ids = [item.id for item in list]
    idx = ids.index(id)
    return idx

# ## CASADI HELPER FUNCTIONS
cw = casadi.SX.sym("w", 3, 1)
cR = casadi.SX.sym("R", 3, 3)

exp3 = casadi.Function("exp3", [cw], [cpin.exp3(cw)])
#log3 = casadi.Function("logp3", [cR], [cpin.log3(cR)])
log3 = casadi.Function("log3", [cR], [log3_approx(cR)])

# data types
#-----------

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

# ## PROBLEM
# Create the casadi optimization problem
opti = casadi.Opti()

# define cost functions

# motion -- constant position
def cost_constant_position(sqrt_info, keyframe_i, keyframe_j):
    ppred = keyframe_j.position - keyframe_i.position
    wpred = log3(exp3(-keyframe_i.anglevector) @ exp3(keyframe_j.anglevector)) # log ( Ri.T * Rj )
    pred  = casadi.vertcat([ ppred, wpred ])

    cost = casadi.sumsqr(sqrt_info @ pred) 
    return cost

# landmark observations
def cost_landmark(meas, sqrt_info, keyframe_i, landmark_j):
    ppred = landmark_j.position - keyframe_i.position
    Rpred = keyframe_i.R.T @ landmark_j.R

    perr = meas[0:3] - ppred
    Rerr = Rpred.T @ exp3(meas[3:6])
    werr = log3(Rerr)

    err = casadi.vertcat([perr, werr])
 
    cost = casadi.sumsqr(sqrt_info @ err) 
    return cost

# Prior on first KF
def cost_prior(meas, sqrt_info, keyframe_i):
    perr = meas[0:3] - keyframe_i.position
    werr = log3(keyframe_i.R.T @ exp3(meas[3:6]))
    err  = casadi.vertcat([ perr, werr ])

    cost = casadi.sumsqr(sqrt_info @ err) 
    return cost


# class for keyframes and landmarks
class OptiVariablePose3:
    def __init__(self, opti, id, position, anglevector):
        self.id = id
        self.position      = opti.variable(3)
        self.anglevector   = opti.variable(3)
        self.R             = exp3(self.anglevector)
        opti.set_initial(self.position, position)
        opti.set_initial(self.anglevector, anglevector)

# class for factors
class Factor:
    def __init__(self, type, i, j, meas, sqrt_info):
        self.type = type
        self.i = i
        self.j = j
        self.meas = meas
        self.sqrt_info = sqrt_info

# init the problem

factors   = list()
keyframes = list()
landmarks = list()


# fill the problem from incoming data:
# begin
t       = 0
kf_id   = 0
lmk_id  = 0
fac_id  = 0


# prior
keyframe_origin = OptiVariablePose3(opti, kf_id, np.array([0,0,0], np.array([0,0,0])))
factor_prior    = Factor("prior", 0, 0, np.array([0,0,0, 0,0,0]), np.eye(6))
keyframes.append(keyframe_origin)
factors.append(factor_prior)
kf_id += 1

# at each lmk observation
image           = images(t)
tag_detection   = april.detect(image)
tag_id          = tag_detection.id
tag_homog       = tag_detection.homog
tag_pose        = homg_to_pose(tag_homog)
measurement     = tag_pose.to_vector()
sqrt_info       = # put a const value by now

lmk_idx = find_index(landmarks, tag_id)
if tag_id is known:
    factors.append(keyframe_index, lmk_idx, measurement, sqrt_info)
else:
    landmarks.append(new landmark)
    lmk_idx = find_index(landmarks, tag_id)
    factors.append(keyframe_index, lmk_idx, measurement, sqrt_info)

# at each new motion
motion = new motion, 6-vector (dp, dw)
new_kf = compose(keyframes[-1], motion)
keyframes.append(new_kf)
factors.append(old_kf_idx, new_kf_idx, motion, sqrt_info)

# advance time
t += 1
    

# compute total cost

totalcost = 0
for factor in factors
    i = factor.i
    j = factor.j
    measurement = factor.measurement
    sqrt_info   = factor.sqrt_info
    if factor.type == "motion":
        totalcost += cost_motion (measurement, sqrt_info, keyframes[i], keyframes[j])
    elif factor.type == "landmark":
        totalcost += cost_landmark (measurement, sqrt_info, keyframes[i], landmarks[j])
    elif factor.type == "prior":
        totalcost += cost_prior (measurement, sqrt_info, keyframes[i])
    else:
        print('Error in the factor type: type not known')
