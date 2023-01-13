import numpy as np
import pinocchio as pin
# import casadi
from pinocchio import casadi as cpin
from gslam_april_tools import *

#-----------------------------------------------------------------------------------
# DATA TYPES
#-----------------------------------------------------------------------------------

# keyframe:
# - id
# - position p
# - orientation w

# landmarks:
# - id
# - position p
# - orientation w

# factors:
# - id of first state i
# - id of second state j
# - type "motion" "landmark" "prior"
# - measurement: 6-vector, translation and rotation
# - sqrt_info: 6x6 matrix


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
