"""
Stokes flow with the Taylor--Hood element, verified using a manufactured
solution.
"""

from dolfin import *
N = 64
k = 2
mu = Constant(1.0)
mesh = UnitSquareMesh(N,N)
x = SpatialCoordinate(mesh)
I = Identity(mesh.geometry().dim())
cell = mesh.ufl_cell()
Ve = VectorElement("CG",cell,k)
Qe = FiniteElement("CG",cell,k-1)
W = FunctionSpace(mesh,MixedElement([Ve,Qe]))

# Take the curl of a potential for solution
# to manufacture:
u3 = (sin(pi*x[0])*sin(pi*x[1]))**2
u_exact = as_vector([u3.dx(1),-u3.dx(0)])
p_exact = sin(2.0*pi*x[0])*sin(3.0*pi*x[1])
def sigma(u,p):
    return 2.0*mu*sym(grad(u)) - p*I
f = -div(sigma(u_exact,p_exact))

# u_exact is zero on boundaries.
bc = DirichletBC(W.sub(0),Constant((0,0)),
                 "on_boundary")

# Galerkin formulation:
u,p = TrialFunctions(W)
v,q = TestFunctions(W)
res = (inner(sigma(u,p),grad(v)) + div(u)*q
       - dot(f,v))*dx
up = Function(W)
solve(lhs(res)==rhs(res),up,bc)

# Check H1 error; can verify optimal (k-th) order
# by modifying parameter N at top of script.
u,p = split(up)
grad_e = grad(u-u_exact)
print(sqrt(assemble(inner(grad_e,grad_e)*dx)))
