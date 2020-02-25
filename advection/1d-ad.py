"""
The prototypical boundary layer problem with 1D advection--diffusion.
"""

from dolfin import *
from ufl import Min

# Setup:
N = 16
k = 1
use_SUPG = True
# (Setting kappa too low eventually produces
# numerical stability issues in the exact
# solution.)
kappa = Constant(5e-3)
a = Constant((1,))

# Formulation:
mesh = UnitIntervalMesh(N)
V = FunctionSpace(mesh,"CG",k)
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)
res_Gal = (kappa*inner(grad(u),grad(v))
           + dot(a,grad(u))*v - f*v)*dx
res_strong = -div(kappa*grad(u)) + dot(a,grad(u)) - f
h = CellDiameter(mesh)
Cinv = Constant(16e0*k*k)
# (See the Brooks & Hughes reference for a nodally-
# exact tau, tuned for this 1D problem.)
tau = Min(h*h/(Cinv*kappa),h/(2*sqrt(dot(a,a))))
res_SUPG = tau*res_strong*dot(a,grad(v))*dx
res = res_Gal
if(use_SUPG):
    res += res_SUPG

# Solve:
uh = Function(V)
solve(lhs(res)==rhs(res),uh,
      [DirichletBC(V,Constant(0),"near(x[0],0)"),
       DirichletBC(V,Constant(1),"near(x[0],1)")])

# Check error:
x = SpatialCoordinate(mesh)[0]
# (Use Expression to be able to interpolate in plotting)
ex_deg = k+3
u_ex = Expression("(exp(a*x[0]/k)-1)/(exp(a/k)-1)",
                  degree=ex_deg,domain=mesh,
                  a=float(a[0]),k=float(kappa))
e = interpolate(uh,FunctionSpace(mesh,"CG",ex_deg))-u_ex
print(sqrt(assemble(dot(grad(e),grad(e))*dx)))

# Plot:
from matplotlib import pyplot as plt
plot(uh)
# (Put exact solution on refined mesh)
plot(interpolate(u_ex,
                 FunctionSpace(UnitIntervalMesh(1000),
                               "CG",1)))
plt.autoscale()
plt.show()
