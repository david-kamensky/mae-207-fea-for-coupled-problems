"""
Comparison of directly specifying the weak
Poisson problem and minimizing a corresponding
functional, leveraging UFL's automatic 
Gateaux differentiation.
"""

from dolfin import *
N = 16
mesh = UnitSquareMesh(N,N)
k = 1
V = FunctionSpace(mesh,"CG",k)
f = Constant(1)

# Solve by directly defining a and L:
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u),grad(v))*dx
L = f*v*dx
bc = DirichletBC(V,Constant(0),"on_boundary")
uh_aL = Function(V)
solve(a==L,uh_aL,bc)

# Solve by differentiating an energy functional:
uh_J = Function(V)
J = (0.5*dot(grad(uh_J),grad(uh_J)) - f*uh_J)*dx
R = derivative(J,uh_J)
DR = derivative(R,uh_J)
solve(DR==-R,uh_J,bc)

# Could also solve as a nonlinear problem, using
# the first derivative as a residual, with
#
#solve(R==0,uh_J,bcs=[bc,])
#
# Only one Newton iteration would be taken, because
# the residual is linear.
    
# Verify that the solutions are the same by assembling
# the $L^2$ norm of the difference:
print(sqrt(assemble(((uh_aL-uh_J)**2)*dx)))

# Plot one of the solutions:
from matplotlib import pyplot as plt
plot(uh_J)
plt.show()

