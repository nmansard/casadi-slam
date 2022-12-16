'''
Solve a very simple toy problem involving a rotation

min_w   || w - p ||**2   s.t w[1] = 0
where p is a known 3d vectors.
'''

import casadi
import numpy as np

p = np.array([.1,.2,.3])

### PROBLEM
# Create the casadi optimization problem
opti = casadi.Opti()

# The optimization variable is the angle-vector w and the associated rotation R=exp(w)
w = opti.variable(3)

totalcost = .5 * casadi.sumsqr(w - p)

opti.subject_to(w[1] == 0)

### SOLVE
opti.minimize(totalcost)
opti.solver('ipopt')
sol = opti.solve()

# The value of the decision variable at the optimum are stored in 2 arrays.
w_sol = opti.value(w)
