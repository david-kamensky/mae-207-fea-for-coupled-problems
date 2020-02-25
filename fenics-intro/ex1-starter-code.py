"""
Starter code from Exercise 1 of the FEniCS notes.
"""

from dolfin import *
Nel = 10
mesh = UnitIntervalMesh(Nel)
V = FunctionSpace(mesh,"CG",1)
u = Function(V)
v = TestFunction(V)
x = SpatialCoordinate(mesh)
F = ((u+1)**2)*inner(grad(u),grad(v))*dx + (u*u + u)*v*dx\
    - sin(pi*x[0])*v*dx

# Replace with a bilinear form:
Delta_u = TrialFunction(V)
J = derivative(F,u,Delta_u)

# Replace with a for-loop implementing Newton's method:
solve(F==0,u,J=J)
