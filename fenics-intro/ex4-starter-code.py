"""
Starter code from Exercise 4 of the FEniCS notes.
"""

from dolfin import *
N = 16
mesh = UnitCubeMesh(N,N,N)
d = mesh.geometry().dim()
cell = mesh.ufl_cell()
uE = VectorElement("CG",cell,1)
TE = FiniteElement("CG",cell,1)
V = FunctionSpace(mesh,MixedElement([uE,TE]))
def boundary0(x, on_boundary):
    return on_boundary and near(x[0],0)
def boundary1(x, on_boundary):
    return on_boundary and near(x[1],0)
def boundary2(x, on_boundary):
    return on_boundary and near(x[2],0)
bc1 = DirichletBC(V.sub(0).sub(0), Constant(0), boundary0)
bc2 = DirichletBC(V.sub(0).sub(1), Constant(0), boundary1)
bc3 = DirichletBC(V.sub(0).sub(2), Constant(0), boundary2)
bc4 = DirichletBC(V.sub(1), Constant(0), boundary0)
bc5 = DirichletBC(V.sub(1), Constant(1), boundary1)
u,T = TrialFunctions(V)
v,Q = TestFunctions(V)
K = Constant(1.0)
G = Constant(1.0)
alpha = Constant(1.0)
I = Identity(d)
eps = sym(grad(u)) - alpha*I*T
sigma = K*tr(eps)*I + 2*G*(eps-tr(eps)*I/3)
f = Constant(d*(0,))
k = Constant(1)
a = inner(sigma,grad(v))*dx + dot(k*grad(T),grad(Q))*dx
L = inner(f,v)*dx
uTh = Function(V)
solve(a==L,uTh,[bc1,bc2,bc3,bc4,bc5])
u_out,T_out = uTh.split()
u_out.rename("u","u")
T_out.rename("T","T")
File("u.pvd") << u_out
File("T.pvd") << T_out
