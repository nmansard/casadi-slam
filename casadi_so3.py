import casadi
from pinocchio import casadi as cpin

# ## CASADI HELPER FUNCTIONS
cw = casadi.SX.sym("w", 3, 1)
cR = casadi.SX.sym("R", 3, 3)

exp3 = casadi.Function("exp3", [cw], [cpin.exp3(cw)])
log3 = casadi.Function("logp3", [cR], [cpin.log3(cR)])

