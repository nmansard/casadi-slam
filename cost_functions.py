import casadi
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
def cost_keyframe_prior(meas, sqrt_info, keyframe):
    perr = meas[0:3] - keyframe.position
    werr = log3(keyframe.R.T @ exp3(meas[3:6]))
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
