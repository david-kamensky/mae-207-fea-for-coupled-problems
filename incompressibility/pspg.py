"""
Stokes flow using equal-order interpolation with PSPG/LSIC stabilization, 
verified via a manufactured solution.
"""

from dolfin import *

# Whether or not to include stabilization:
USE_STAB = True

N = 32
k = 1
mu = Constant(1.0)
mesh = UnitSquareMesh(N,N)
x = SpatialCoordinate(mesh)
I = Identity(mesh.geometry().dim())
cell = mesh.ufl_cell()
Ve = VectorElement("CG",cell,k)
Qe = FiniteElement("CG",cell,k)
W = FunctionSpace(mesh,MixedElement([Ve,Qe]))

# Take the curl of a potential for solution
# to manufacture:
u3 = (sin(pi*x[0])*sin(pi*x[1]))**2
u_exact = as_vector([u3.dx(1),-u3.dx(0)])
p_exact = sin(2.0*pi*x[0])*sin(3.0*pi*x[1])
def sigma(u,p):
    return 2.0*mu*sym(grad(u)) - p*I
def strongRes(u,p,f):
    return -div(sigma(u,p)) - f
f = strongRes(u_exact,p_exact,Constant((0,0)))

# u_exact is zero on boundaries.
bc = DirichletBC(W.sub(0),Constant((0,0)),
                 "on_boundary")

# Galerkin formulation:
u,p = TrialFunctions(W)
v,q = TestFunctions(W)
resGalerkin = (inner(sigma(u,p),grad(v)) + div(u)*q
               - dot(f,v))*dx

# Stabilization:   
Cinv = Constant(1e0*k**2)
h = CellDiameter(mesh)
tau_M = h*h/(Cinv*mu)
resPSPG = tau_M*inner(strongRes(u,p,f),grad(q))*dx 
tau_C = h*h/tau_M
resLSIC = tau_C*div(u)*div(v)*dx
if(USE_STAB):
    resStab = resPSPG + resLSIC
else:
    # Necessary to avoid singularity of problem
    # and NaN pressures without PSPG.
    resStab = Constant(DOLFIN_EPS)*p*q*dx

# Formulation:
res = resGalerkin + resStab

up = Function(W)
solve(lhs(res)==rhs(res),up,bc)

# Check H1 error; can verify optimal (k-th) order
# by modifying parameter N at top of script.
u,p = split(up)
grad_e = grad(u-u_exact)
print(sqrt(assemble(inner(grad_e,grad_e)*dx)))

# Visualize pressure; with USE_STAB==False,
# this shows a checkerboard mode.
from matplotlib import pyplot as plt
plot(p)
plt.show()
