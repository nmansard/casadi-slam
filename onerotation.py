'''
Solve a very simple toy problem involving a rotation

min_w   || R p - p* ||**2   with R:= exp(w)
where p and p* are two known 3d vectors.
Display the result in Meshcat.
'''

import casadi
from pinocchio import casadi as cpin
import numpy as np
from utils.meshcat_viewer_wrapper import MeshcatVisualizer,translation2d,planar

p = np.array([.1,.2,.3])
pdes = np.array([-.1,.2,-.3])

### CASADI HELPER FUNCTIONS
cw = casadi.SX.sym("w",3,1)

exp3 = casadi.Function('exp3',[ cw ],[cpin.exp3(cw) ])


### PROBLEM
# Create the casadi optimization problem
opti = casadi.Opti()

# The optimization variable is the angle-vector w and the associated rotation R=exp(w)
w = opti.variable(3)
R = exp3(w)

# Beware: casadi matrix product is *, numpy matrix product is @
totalcost = .5 * casadi.sumsqr(R * p - pdes)

### SOLVE
opti.minimize(totalcost)
opti.solver('ipopt')
sol = opti.solve()

# The value of the decision variable at the optimum are stored in 2 arrays.
w_sol = opti.value(w)
R_sol = opti.value(R)

### Sanity check
assert( np.allclose(R_sol@R_sol.T,np.eye(3)) )
#assert( np.allclose(R_sol@p,pdes) )


### Display

viz = MeshcatVisualizer()
pointID = 'world/point'; viz.addSphere(pointID,.1,[1,0,0,1])
pointdesID = 'world/pointdes'; viz.addSphere(pointdesID,.1,[0,1,0,1])
boxID = 'world/box';   viz.addBox(boxID,(p*2).tolist(),[1,1,0,1])

viz.applyConfiguration(pointdesID,pdes.tolist()+[0,0,0,1])
viz.applyConfiguration(pointID,(R_sol@p).tolist()+[0,0,0,1])
viz.applyConfiguration(boxID,[0,0,0]+pin.Quaternion(R_sol).coeffs().tolist())
