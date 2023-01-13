import casadi
from pinocchio import casadi as cpin

# ## CASADI HELPER FUNCTIONS
cw = casadi.SX.sym("w", 3, 1)
cR = casadi.SX.sym("R", 3, 3)

exp3 = casadi.Function("exp3", [cw], [cpin.exp3(cw)])
log3 = casadi.Function("log3", [cR], [cpin.log3(cR)])

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

# Replace log3 disfonctional in Pinocchio by the approximation
log3 = casadi.Function("log3_approx", [cR], [log3_approx(cR)])

