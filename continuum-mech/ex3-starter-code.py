"""
Starter code given for Exercise 2 of the section on continuum mechanics.
"""

from dolfin import *
N = 8
mesh = UnitCubeMesh(N,N,N)
k = 1
dx = dx(metadata={"quadrature_degree":2*k})
V = VectorFunctionSpace(mesh,"CG",k)
u = Function(V)
I = Identity(len(u))
def problem(u):
    F = I + grad(u)
    C = F.T*F
    E = 0.5*(C-I)
    K = Constant(1.0e1)
    mu = Constant(1.0e1)
    S = K*tr(E)*I + 2.0*mu*(E - tr(E)*I/3.0)
    psi = 0.5*inner(E,S)
    return F,S,psi
X = SpatialCoordinate(mesh)
u_ex = as_vector(3*[0.1*sin(pi*X[0])
                  *sin(pi*X[1])*sin(pi*X[2]),])

####### Replace #######
f0 = Constant((1,2,3))
#######################

F,S,psi = problem(u)
v = TestFunction(V)
R = derivative(psi*dx,u,v) - inner(f0,v)*dx
J = derivative(R,u)
bc = DirichletBC(V,Constant((0,0,0)),"on_boundary")
solve(R==0,u,[bc,],J=J)
