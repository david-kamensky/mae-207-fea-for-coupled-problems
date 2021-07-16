"""
The prototypical boundary layer problem with 1D advection--diffusion,
illustrating the effects of SUPG stabilization.
"""

from dolfin import *
from ufl import Min

# Setup:
N = 8
k = 1
# Setting to False on coarse meshes shows
# the poor performance of Bubnov--Galerkin.
use_SUPG = True
# Setting kappa too low eventually produces
# numerical stability issues in the exact
# solution.
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
Cinv = Constant(6.0*k*k)
# (See the Brooks & Hughes reference for a nodally-
# exact tau, tuned for this 1D problem; we use
# the more generic result from numerical analysis
# in this script.)
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
print("H1 seminorm error = "
      +str(sqrt(assemble(dot(grad(e),grad(e))*dx))))
print("L2 error = "
      +str(sqrt(assemble(e*e*dx))))

# Plot:
from matplotlib import pyplot as plt
plot(uh)
# (Put exact solution on refined mesh)
mesh_interp = UnitIntervalMesh(1000)
V_interp = FunctionSpace(mesh_interp,"CG",1)
u_ex_interp = interpolate(u_ex,V_interp)
plot(u_ex_interp)
plt.autoscale()
plt.show()

# Solve again, using weak BC enforcement:

# Function satifying BCs:
g = x
# Penalty constant; must be sufficiently
# large for stability.
C_pen = Constant(1e1*k*k)
# Boundary terms of residual:
n = FacetNormal(mesh)
res_weak = (-kappa*dot(grad(u),n)*v
            - kappa*dot(grad(v),n)*(u-g)
            + kappa*(C_pen/h)*(u-g)*v
            - Min(dot(a,n),Constant(0))*(u-g)*v)*ds
res += res_weak

# No strongly-enforced Dirichlet BCs:
uh_weak = Function(V)
solve(lhs(res)==rhs(res),uh_weak)

# Check error:
e = interpolate(uh_weak,FunctionSpace(mesh,"CG",ex_deg))-u_ex
print("H1 seminorm error (weak BCs) = "
      +str(sqrt(assemble(dot(grad(e),grad(e))*dx))))
print("L2 error (weak BCs) = "
      +str(sqrt(assemble(e*e*dx))))

# Plot and compare with exact solution
# and solution with strong BCs:
plot(uh)
plot(u_ex_interp)
plot(uh_weak)
plt.autoscale()
plt.show()
