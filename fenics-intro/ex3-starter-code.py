"""
Starter code from Exercise 3 of the FEniCS notes.
"""

from dolfin import *
N = 16
T = 1.0
Dt = Constant(T/N)
mesh = UnitIntervalMesh(N)
V = FunctionSpace(mesh,"CG",1)
u = TrialFunction(V)
uh = Function(V)
uh_old = Function(V)
udot = (u-uh_old)/Dt
v = TestFunction(V)
t = Constant(0.0)
tv = variable(t)
x = SpatialCoordinate(mesh)[0]
u_exact = sin(tv)*sin(pi*x)
udot_exact = diff(u_exact,tv)
f = udot_exact - div(grad(u_exact))
F = ((udot-f)*v + dot(grad(u),grad(v)))*dx
bc = DirichletBC(V,Constant(0.0),"on_boundary")
for step in range(0,N):
    t.assign(float(t)+float(Dt))
    solve(lhs(F)==rhs(F),uh,bcs=[bc,])
    uh_old.assign(uh)
print(sqrt(assemble(((uh-u_exact)**2)*dx)))
