from dolfin import *

# Parameters:

# Number of elements across domain:
N = 32
# Polynomial degree of FE space:
k = 1
# Scaling factor on penalty; can be zero with gamma = -1:
alpha = Constant(1e1)
# 1 for symmetric, -1 for non-symmetric:
gamma = Constant(1)
# 1 for Nitsche method, 0 for weakly-consistent penalty:
nitsche = Constant(1)

# Mesh and function space setup:
mesh = UnitSquareMesh(N,N)
n = FacetNormal(mesh)
h = CellDiameter(mesh)
x = SpatialCoordinate(mesh)
V = FunctionSpace(mesh,"CG",k)

# Choose exact solution for testing:
g = sin(x[0])*exp(x[1])
f = -div(grad(g))

# Pose formulation and solve:
u = TrialFunction(V)
v = TestFunction(V)
res_interior = (dot(grad(u),grad(v)) - f*v)*dx
res_bdry = (nitsche*(-dot(grad(u),n)*v - gamma*dot(grad(v),n)*(u-g))
            + (alpha/h)*(u-g)*v)*ds
res = res_interior + res_bdry
u = Function(V)
solve(lhs(res)==rhs(res),u)

# Check error:
e = g - u
print("L^2 error = "+str(sqrt(assemble(e*e*dx))))
print("H^1 error = "+str(sqrt(assemble(dot(grad(e),grad(e))*dx))))
