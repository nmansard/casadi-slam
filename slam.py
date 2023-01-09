

import casadi
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.utils import rotate
import numpy as np
import time
from utils.meshcat_viewer_wrapper import MeshcatVisualizer

# ## CASADI HELPER FUNCTIONS
cw = casadi.SX.sym("w", 3, 1)
cR = casadi.SX.sym("R", 3, 3)

exp3 = casadi.Function("exp3", [cw], [cpin.exp3(cw)])
log3 = casadi.Function("logp3", [cR], [cpin.log3(cR)])

def wedge(S):
    s = np.array([S(2,1), S(0,2), S(1,0)])
    return s
def log3(R)
    w = wedge(R - R.T) / 2
    return w


def find_index(list, id):
    ids = [item.id for item in list]
    idx = ids.index(id)
    return idx


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
def cost_motion(meas, sqrt_info, keyframe_i, keyframe_j):
    ppred = keyframe_j.p - keyframe_i.p
    wpred = log3(exp3(-keyframe_i.w) @ exp3(keyframe_j.w)) # log ( Ri.T * Rj )
    pred  = casadi.vertcat([ ppred, wpred ])

    cost = casadi.sumsqr(sqrt_info @ (meas - pred)) 
    return cost

# TODO
def cost_landmark(meas, sqrt_info, keyframe_i, landmark_j):
    ppred = landmark_j.p - keyframe_i.p
    wpred = log3(exp3(-keyframe_i.w) @ exp3(landmark_j.w)) # log ( Ri.T * Rj )
    pred  = casadi.vertcat([ ppred, wpred ])

    cost = casadi.sumsqr(sqrt_info @ (meas - pred)) 
    return cost

# TODO
def cost_prior(meas, sqrt_info, keyframe_i):
    ppred = landmark_j.p - keyframe_i.p
    wpred = log3(exp3(-keyframe_i.w) @ exp3(landmark_j.w)) # log ( Ri.T * Rj )
    pred  = casadi.vertcat([ ppred, wpred ])

    cost = casadi.sumsqr(sqrt_info @ (meas - pred)) 
    return cost


# class for keyframes and landmarks
class OptiVariablePose3:
    def __init__(self, opti, position, anglevector):
         self.position      = opti.variable(3)
         self.anglevector   = opti.variable(3)
         self.R             = exp3(self.anglevector)
         opti.set_initial(self.position, position)
         opti.set_initial(self.anglevector, anglevector)

# class for factors
class Factor:
    def __init__(self,type,i,j,mes,sqrt):
        self.type = type
        self.i = i
        self.j = j
        self.measurement = mes
        self.sqrt_info = sqrt

# init the problem

factors   = []
keyframes = []
landmarks = []


# fill the problem from incoming data:
# begin
keyframe_origin = OptiVariablePose3(opti, np.array([0,0,0], np.array([0,0,0])))
factor_prior    = Factor("prior", 0, 0, np.array([0,0,0, 0,0,0]), np.eye(6))

keyframes.append(keyframe_origin)
factors.append(factor_prior)

# time
t = 0

# at each lmk observation
image           = images(t)
tag_detection   = april.detect(image)
tag_id          = tag_detection.id
tag_homog       = tag_detection.homog
tag_pose        = homg_to_pose(tag_homog)
measurement     = tag_pose.to_vector()
sqrt_info       = # put a const value by now

lmk_idx = find_index(landmarks, tag_id)
if tag.id is known
    factors.append(keyframe_index, lmk_idx, measurement, sqrt_info)
else
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
