"""
Solve a very simple toy problem involving a rotation

min_w   || R p - p* ||**2   with R:= exp(w)
where p and p* are two known 3d vectors.
Display the result in Meshcat.
"""

import casadi
import pinocchio as pin
from pinocchio import casadi as cpin
import numpy as np
from utils.meshcat_viewer_wrapper import MeshcatVisualizer
from utils.meshcat_viewer_wrapper import translation2d  # noqa: F401
from utils.meshcat_viewer_wrapper import planar  # noqa: F401

p = np.array([0.1, 0.2, 0.3])
pdes = np.array([-0.1, 0.2, -0.3])

# ## CASADI HELPER FUNCTIONS
cw = casadi.SX.sym("w", 3, 1)

exp3 = casadi.Function("exp3", [cw], [cpin.exp3(cw)])


# ## PROBLEM
# Create the casadi optimization problem
opti = casadi.Opti()

# The optimization variable is the angle-vector w and the associated rotation R=exp(w)
w = opti.variable(3)
R = exp3(w)

# Beware: casadi matrix product is @ like numpy array product
totalcost = 0.5 * casadi.sumsqr(R @ p - pdes)

# ## SOLVE
opti.minimize(totalcost)
opti.solver("ipopt")

# Initial guess with optimal solution for debug
R0=pin.Quaternion.FromTwoVectors(p,pdes).matrix()
w0=pin.log3(R0)
# Uncomment if you want the warm start 
opti.set_initial(w,w0)

sol = opti.solve()

# The value of the decision variable at the optimum are stored in 2 arrays.
w_sol = opti.value(w)
R_sol = opti.value(R)
totalcost_sol = opti.value(totalcost)

# ## Sanity check
assert np.allclose(R_sol @ R_sol.T, np.eye(3))
# assert( np.allclose(R_sol@p,pdes) )


# ## Display

viz = MeshcatVisualizer()
pointID = "world/point"
viz.addSphere(pointID, 0.1, [1, 0, 0, 1])
pointdesID = "world/pointdes"
viz.addSphere(pointdesID, 0.1, [0, 1, 0, 1])
boxID = "world/box"
viz.addBox(boxID, (p * 2).tolist(), [1, 1, 0, 1])

viz.applyConfiguration(pointdesID, pdes.tolist() + [0, 0, 0, 1])
viz.applyConfiguration(pointID, (R_sol @ p).tolist() + [0, 0, 0, 1])
viz.applyConfiguration(boxID, [0, 0, 0] + pin.Quaternion(R_sol).coeffs().tolist())
