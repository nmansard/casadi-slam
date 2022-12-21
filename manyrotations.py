"""
Solve a very simple toy problem involving a rotation

min_w   || R p - p* ||**2   with R:= exp(w)
where p and p* are two known 3d vectors.
Display the result in Meshcat.
"""

import casadi
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.utils import rotate
import numpy as np
import time
from utils.meshcat_viewer_wrapper import MeshcatVisualizer

p = np.array([0.1, 0.2, 0.3])
omega = 2*np.pi*.5
pdes = [ rotate('x',t)@rotate('y',2*t-5)@ ((1+.2*np.sin(t*omega))*p)
         for t in np.arange(0,5,.02) ]
T = len(pdes)
    

# ## CASADI HELPER FUNCTIONS
cw = casadi.SX.sym("w", 3, 1)
cR = casadi.SX.sym("R", 3, 3)

exp3 = casadi.Function("exp3", [cw], [cpin.exp3(cw)])
log3 = casadi.Function("logp3", [cR], [cpin.log3(cR)])


# ## PROBLEM
# Create the casadi optimization problem
opti = casadi.Opti()

# The optimization variable is the angle-vector w and the associated rotation R=exp(w)
ws = [ opti.variable(3) for t in range(T) ]
Rs = [ exp3(w) for w in ws ]

# Beware: casadi matrix product is @ like numpy array product
totalcost = 0

for t in range(T):
    totalcost += 0.5 * casadi.sumsqr(Rs[t] @ p - pdes[t])
    if t>0:
        totalcost += 0.5 * casadi.sumsqr( log3(Rs[t-1].T@Rs[t]) )

# ## SOLVE
opti.minimize(totalcost)
opti.solver("ipopt")

sol = opti.solve()

# The value of the decision variable at the optimum are stored in 2 arrays.
ws_sol = [ opti.value(w) for w in ws ]
Rs_sol = [ opti.value(R) for R in Rs ]

# ## Sanity check
for R_sol in Rs_sol:
    assert np.allclose(R_sol @ R_sol.T, np.eye(3))

# ## Display

viz = MeshcatVisualizer()
pointID = "world/point"
viz.addSphere(pointID, 0.1, [1, 0, 0, 1])
pointdesID = "world/pointdes"
viz.addSphere(pointdesID, 0.1, [0, 1, 0, 1])
boxID = "world/box"
viz.addBox(boxID, (p * 2).tolist(), [1, 1, 0, .1])

def viewtraj():
    for t,[R_sol,pt] in enumerate(zip(Rs_sol,pdes)):
        viz.applyConfiguration(pointdesID, pt.tolist() + [0, 0, 0, 1])
        viz.applyConfiguration(pointID, (R_sol @ p).tolist() + [0, 0, 0, 1])
        viz.applyConfiguration(boxID, [0, 0, 0] + pin.Quaternion(R_sol).coeffs().tolist())
        time.sleep(1e-2)

