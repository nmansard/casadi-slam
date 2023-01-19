import numpy as np
import pinocchio as pin
import casadi
from gslam_types import *
from gslam_april_tools import *
from cost_functions import *

def betweenPoses6D(pose1, pose2):
    pose1_pos = pose1[0:3]
    pose1_ori = pin.exp(pose1[3:6])
    pose2_pos = pose2[0:3]
    pose2_ori = pin.exp(pose2[3:6])
    T, R, w = betweenPoses(pose1_pos, pose1_ori, pose2_pos, pose2_ori)
    return np.hstack((T, w))

def testCostFunctions():

    # define an optimizer
    opti = casadi.Opti()
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    opti.solver("ipopt", opts)

    # true keyframes and landmark values
    kf1_true = np.random.random(6)
    kf2_true = np.random.random(6)
    lmk_true = np.random.random(6)

    # exact measurements of true values
    kf1_prior_meas = kf1_true
    lmk_prior_meas = lmk_true    
    motion_meas = betweenPoses6D(kf1_true, kf2_true)
    landmark_meas = betweenPoses6D(kf1_true, lmk_true)

    # warm start keyframes and landmark at slightly different values
    kf1_warm = kf1_true + np.random.random(6)/10
    kf2_warm = kf2_true + np.random.random(6)/10
    lmk_warm = lmk_true + np.random.random(6)/10

    # build optimization objects
    keyframe1 = OptiVariablePose3(opti, 1, kf1_warm[:3], kf1_warm[3:6])
    keyframe2 = OptiVariablePose3(opti, 1, kf2_warm[:3], kf2_warm[3:6])
    landmark1 = OptiVariablePose3(opti, 1, lmk_warm[:3], lmk_warm[3:6])

    # information matrix and precision for tests
    sqrt_info = np.identity(6)
    eps       = 1e-4

    ## TEST EACH COST FUNCTION

    # cost_keyframe_prior ----------------------------------------------------------
    #
    #  Factor graph to solve ( * = factor; O = state ):
    #
    #   *----------O   
    #
    #   prior     KF1
    #
    # expected after solve : KF1 = kf1_true
    #
    totalcost = cost_keyframe_prior(kf1_prior_meas, sqrt_info, keyframe1)
    opti.minimize(totalcost)
    opti.solve()

    assert  np.linalg.norm(opti.value(keyframe1.position) - kf1_true[:3]) < eps    , 'cost_keyframe_prior failed to estimate position'
    assert  np.linalg.norm(opti.value(keyframe1.anglevector) - kf1_true[3:6]) < eps, 'cost_keyframe_prior failed to estimate orientation'

    print('cost_keyframe_prior    appears to work correctly!')

    # cost_constant_position ---------------------------------------------
    #
    #  Factor graph to solve:
    #
    #   *----------O-------*-------O   
    #
    #   prior     KF1  cst.pos.  KF2
    #
    # expected after solve : KF1 = fk1_true ; KF2 = kf1_true
    #
    totalcost = cost_keyframe_prior(kf1_prior_meas, sqrt_info, keyframe1) + cost_constant_position(sqrt_info, keyframe1, keyframe2)
    opti.minimize(totalcost)
    opti.solve()

    assert  np.linalg.norm(opti.value(keyframe1.position) - kf1_true[:3]) < eps    , 'cost_keyframe_prior failed to estimate position'
    assert  np.linalg.norm(opti.value(keyframe1.anglevector) - kf1_true[3:6]) < eps, 'cost_keyframe_prior failed to estimate orientation'
    assert  np.linalg.norm(opti.value(keyframe2.position) - kf1_true[:3]) < eps    , 'cost_constant_position failed to estimate translation'
    assert  np.linalg.norm(opti.value(keyframe2.anglevector) - kf1_true[3:6]) < eps, 'cost_constant_position failed to estimate rotation'

    print('cost_constant_position appears to work correctly!')

    # cost_landmark ------------------------------------------------
    #
    #  Factor graph to solve:
    #
    #   *----------O-------*-------O   
    #
    #   prior     KF1  landmark  LMK
    #
    # expected after solve : KF1 = kf1_true ; Lmk = lmk_true
    #
    totalcost = cost_keyframe_prior(kf1_prior_meas, sqrt_info, keyframe1) + cost_landmark(landmark_meas, sqrt_info, keyframe1, landmark1)
    opti.minimize(totalcost)
    opti.solve()

    assert  np.linalg.norm(opti.value(keyframe1.position) - kf1_true[:3]) < eps    , 'cost_keyframe_prior failed to estimate position'
    assert  np.linalg.norm(opti.value(keyframe1.anglevector) - kf1_true[3:6]) < eps, 'cost_keyframe_prior failed to estimate orientation'
    assert  np.linalg.norm(opti.value(landmark1.position) - lmk_true[:3]) < eps    , 'cost_landmark failed to estimate position'
    assert  np.linalg.norm(opti.value(landmark1.anglevector) - lmk_true[3:6]) < eps, 'cost_landmark failed to estimate orientation'

    print('cost_landmark          appears to work correctly!')

    # cost_landmark_prior -----------------------------------------------------
    #
    #  Factor graph to solve:
    #
    #   *----------O   
    #
    #   prior     LMK
    #
    # expected after solve : Lmk = lmk_true
    #
    totalcost = cost_landmark_prior(lmk_prior_meas, sqrt_info, landmark1)
    opti.minimize(totalcost)
    opti.solve()

    assert  np.linalg.norm(opti.value(landmark1.position) - lmk_true[:3]) < eps    , 'cost_landmark_prior failed to estimate position'
    assert  np.linalg.norm(opti.value(landmark1.anglevector) - lmk_true[3:6]) < eps, 'cost_landmark_prior failed to estimate orientation'

    print('cost_landmark_prior    appears to work correctly!')





# Do perform all tests !

testCostFunctions()
