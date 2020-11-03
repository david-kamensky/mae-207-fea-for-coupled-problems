"""
Method of manufactured solutions for linear elasticity, implemented both 
by deriving the variational form from an energy potential and by 
specifying it manually.
"""

from dolfin import *

# Set up meh, function space.
N = 4
mesh = UnitCubeMesh(N,N,N)
k = 1
dx = dx(metadata={"quadrature_degree":2*k})
V = VectorFunctionSpace(mesh,"CG",k)

# Will need to apply problem definition to multiple functions.
def problem(u):
    I = Identity(len(u))
    eps = sym(grad(u))
    K = Constant(1.0e1) # Bulk modulus
    mu = Constant(1.0e1) # Shear modulus
    sigma = K*tr(eps)*I + 2.0*mu*(eps - tr(eps)*I/3.0) # C:eps
    psi = 0.5*inner(eps,sigma) # 0.5*eps:C:eps
    return sigma,psi

# Specify exact solution:
x = SpatialCoordinate(mesh)
u_ex = as_vector(3*[0.1*sin(pi*x[0])
                  *sin(pi*x[1])*sin(pi*x[2]),])

# Define corresponding body force using strong form of the problem:
sigma_ex,psi_ex = problem(u_ex)
f = -div(sigma_ex)

# Solve weak problem obtained by differentiating psi:
u = Function(V)
sigma,psi = problem(u)
v = TestFunction(V)
a = derivative(derivative(psi*dx,u),u)
L = inner(f,v)*dx
bc = DirichletBC(V,Constant((0,0,0)),"on_boundary")
solve(a==L,u,bc)

# Set up LHS without using derivative function:
u_trial = TrialFunction(V)
sigma_trial,_ = problem(u_trial)
a_manual = inner(sigma_trial,grad(v))*dx
u_manual = Function(V)
solve(a_manual==L,u_manual,bc)

# Check errors in H^1 norm:
e_diff = u-u_manual
e = u-u_ex
H1norm = lambda u : sqrt(assemble(inner(grad(u),grad(u))*dx))
print("Difference between u, u_manual:",H1norm(e_diff))
print("Error in u:",H1norm(e))
