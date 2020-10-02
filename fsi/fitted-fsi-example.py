"""
Simple example of body-fitted FSI.  This code largely follows the notational
conventions from Chapter 4 of the course notes.  The basic problem setup 
is a cantilever beam in a channel with flow around it.  The flow has 
no-slip boundary conditions on the top and bottom, and is driven by a 
periodic-in-time Dirichlet boundary condition on the left end.  
The right end is a stable Neumann boundary condition, using the 
Navier--Stokes extension of the stable Neumann boundary condition for 
advection--diffusion discussed in lecture. 

Some notes:

- The DOLFIN mesh object itself is NOT deformed using ALE.move().  Instead,
  the mesh motion is a field of the solution, outputted as "u" in the
  ParaView files written by this script.  Follow the directions from the
  notes on FEniCS to visualize the solution on the deforming mesh, using
  "Append Attributes" and "Warp by Vector" filters.

- This demo uses what is called "quasi-direct" coupling, where the
  fluid--structure problem for the continuum velocity and pressure is solved
  using a mixed formulation, while the mesh-motion problem is solved 
  separately, and the two subproblems are iterated to convergence in each
  time step.  (A taxonomy of FSI solution schemes can be found in the FSI
  textbook of Bazilevs, Takizawa, and Tezduyar.)

- Stable Neumann BCs are sometimes framed as a form of "backflow 
  stabilization"; see https://doi.org/10.1007%2Fs00466-011-0599-0 for
  discussion and applications.  
"""

from dolfin import *
from ufl import indices, Jacobian, Min
from mshr import *

####### Domain and mesh setup #######

# Parameters defining domain geometry:
SOLID_LEFT = 0.45
SOLID_RIGHT = 0.55
SOLID_TOP = 0.5
OMEGA_H = 0.75
OMEGA_W = 1.0
REF_MARGIN = 0.1

# Define the mshr geometrical primitives for this domain:
r_Omega = Rectangle(Point(0,0),Point(OMEGA_W,OMEGA_H))
r_Omega_s = Rectangle(Point(SOLID_LEFT,0),
                      Point(SOLID_RIGHT,SOLID_TOP))

# Enumerate subdomain markers
FLUID_FLAG = 0
SOLID_FLAG = 1
# Zero is the default flag, and does not need to be
# explicitly set for the fluid subdomain.
r_Omega.set_subdomain(SOLID_FLAG,r_Omega_s)

# Parameters defining refinement level:
N = 70

# Generate mesh of Omega, which will have a fitted
# subdomain for Omega_s.
mesh = generate_mesh(r_Omega, N)

# Mesh-derived quantities:
d = mesh.geometry().dim()
n = FacetNormal(mesh)
I = Identity(d)
h = CellDiameter(mesh)

# Define subdomains for use in boundary condition definitions:
class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary 
                and ((x[1] < DOLFIN_EPS) 
                     or (x[1] > (OMEGA_H - DOLFIN_EPS))))
class Inflow(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and (x[0] < DOLFIN_EPS))
class Outflow(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and (x[0] > (OMEGA_W - DOLFIN_EPS)))
class PartialOmega(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and 
                ((x[1] < DOLFIN_EPS) or 
                 (x[1] > (OMEGA_H - DOLFIN_EPS)) or 
                 (x[0] < DOLFIN_EPS) or 
                 (x[0] > (OMEGA_W - DOLFIN_EPS) )))
class SolidDomainClosure(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] > SOLID_LEFT - DOLFIN_EPS
                and x[0] < SOLID_RIGHT + DOLFIN_EPS
                and x[1] < SOLID_TOP + DOLFIN_EPS)
class SolidDomainInterior(SubDomain):
    def inside(self, x, on_boundary):
        # (Keep boundary at bottom of domain)
        return (x[0] > SOLID_LEFT + DOLFIN_EPS
                and x[0] < SOLID_RIGHT - DOLFIN_EPS
                and x[1] < SOLID_TOP - DOLFIN_EPS)
 
# Set up integration measures, with flags to integrate over
# subsets of the domain.
markers = MeshFunction('size_t', mesh, d, mesh.domains())
bdry_markers = MeshFunction('size_t', mesh, d-1, 0)
OUTFLOW_FLAG = 2
Outflow().mark(bdry_markers,OUTFLOW_FLAG)
dx = dx(metadata={'quadrature_degree': 2},
        subdomain_data=markers)
ds = ds(metadata={'quadrature_degree': 2},
        subdomain_data=bdry_markers)

####### Elements and function spaces #######

# Define function spaces (equal order interpolation):
cell = mesh.ufl_cell()
Ve = VectorElement("CG", cell, 1)
Qe = FiniteElement("CG", cell, 1)
VQe = MixedElement((Ve,Qe))
# Mixed function space for velocity and pressure:
W = FunctionSpace(mesh,VQe)
# Function space for mesh displacement field, 
# which will be solved for separately in a 
# quasi-direct scheme:
V = FunctionSpace(mesh,Ve)

####### Set up time integration variables #######

TIME_INTERVAL = 1e2
N_STEPS = 2000
Dt = Constant(TIME_INTERVAL/N_STEPS)

# Mesh motion functions:
uhat = Function(V)
uhat_old = Function(V)
du = TestFunction(V)
vhat = (uhat-uhat_old)/Dt

# Fluid--structure functions:
(dv, dp) = TestFunctions(W)
w = Function(W)
v,p = split(w)
w_old = Function(W)
v_old, p_old = split(w_old)
dv_dr = (v - v_old)/Dt
dv_ds = dv_dr # (Only valid in solid)

# This is the displacement field used in the 
# solid formulation; notice that it is NOT 
# in the space V; it is an algebraic object 
# involving the unknown fluid--structure velocity 
# field v.
u = uhat_old + Dt*v

# This will need to be updated to match u, for 
# purposes of setting the boundary condition 
# on the mesh motion subproblem.
u_func = Function(V)

####### Changes of variable #######

# Follow notation from Bazilevs et al., where y is 
# the coordinate in the reference domain, x is the 
# coordinate in the spatial domain, and X is the 
# coordinate in the material domain.  Note that 
# the built-in differential operators (e.g., grad, 
# div, etc.) and integration measures (e.g., dx, ds, 
# etc.) are w.r.t. the reference configuration, y, 
# which is the mesh that FEniCS sees.  
dX = dx(SOLID_FLAG)
dy = dx
grad_y = grad
grad_X = grad # (Only valid in solid)
y = SpatialCoordinate(mesh)
x = y + uhat
det_dxdy = det(grad_y(x))
def grad_x(f):
    return dot(grad_y(f),inv(grad_y(x)))
def div_x(f): # (For vector-valued f)
    return tr(grad_x(f))
def div_x_tens(f): # (For (rank-2) tensor-valued f)
    i,j = indices(2)
    return as_tensor(grad_x(f)[i,j,j],(i,))

# Note:  Trying to define dx = det_dxdy*dy would 
# result in an object of type Form, which could no 
# longer be used as an integration measure.
# Thus, integration over the spatial configuration 
# is done with det_dxdy*dy directly in the fluid 
# formulation.  

####### Boundary conditions #######

# BCs for the fluid--structure subproblem:
bc0_fs = DirichletBC(W.sub(0), Constant((0.0,0.0)), Walls())
# Note that "x" in this Expression is really y 
# in the kinematics described above, but, because the 
# mesh motion problem has a zero Dirichlet BC on the inflow
# boundary, there happens to be no difference.
v_in = Expression(("2.0*sin(pi*t)*x[1]*(H - x[1])/(H*H)",
                   "0.0"),t=0.0,H=OMEGA_H,degree=2)
bc1_fs = DirichletBC(W.sub(0), v_in, Inflow())
bc2_fs = DirichletBC(W.sub(1), Constant(0), 
                     SolidDomainInterior())
bcs_fs = [bc0_fs, bc1_fs, bc2_fs]

# BCs for the mesh motion subproblem:
bc_m_walls = DirichletBC(V.sub(1), Constant(0), Walls())
bc_m_inflow = DirichletBC(V, Constant(d*(0,)), Inflow())
bc_m_outflow = DirichletBC(V, Constant(d*(0,)), Outflow())
bc_m_struct = DirichletBC(V, u_func, SolidDomainClosure())
bcs_m = [bc_m_struct,bc_m_walls,bc_m_inflow,bc_m_outflow]


####### Formulation of mesh motion subproblem #######

# Residual for mesh, which satisfies a fictitious elastic problem:
F_m = grad_y(uhat) + I
E_m = 0.5*(F_m.T*F_m - I)
m_jac_stiff_pow = Constant(3)
# Artificially stiffen the mesh where it is getting crushed:
K_m = Constant(1)/pow(det(F_m),m_jac_stiff_pow)
mu_m = Constant(1)/pow(det(F_m),m_jac_stiff_pow)
S_m = K_m*tr(E_m)*I + 2.0*mu_m*(E_m - tr(E_m)*I/3.0)
res_m = (inner(F_m*S_m,grad_y(du)))*dy
Dres_m = derivative(res_m, uhat)


####### Formulation of the solid subproblem #######

# Elastic properties
mu_s = Constant(1e4)
K = Constant(1e4)
rho_s0 = Constant(1)

# Kinematics:
F = grad_X(u) + I  
E = 0.5*(F.T*F - I)
S = K*tr(E)*I + 2.0*mu_s*(E - tr(E)*I/3.0)
f_s = Constant(d*(0,))
res_s = rho_s0*inner(dv_ds - f_s,dv)*dX \
        + inner(F*S,grad_X(dv))*dX

####### Formulation of the fluid subproblem #######

# Galerkin terms:
rho_f = Constant(1)
mu_f = Constant(1e-2)
sigma_f = 2.0*sym(grad_x(v)) - p*I
v_adv = v - vhat
DvDt = dv_dr + dot(grad_x(v),v_adv)
resGal_f = (rho_f*dot(DvDt,dv) + inner(sigma_f,grad_x(dv))
            + dp*div_x(v))*det_dxdy*dy(FLUID_FLAG)

# Stabilization:

# Deformed mesh size tensor in the spatial configuration:
dxi_dy = inv(Jacobian(mesh))
dxi_dx = dxi_dy*inv(grad_y(x))
G = (dxi_dx.T)*dxi_dx

# SUPG/PSPG:
resStrong_f = rho_f*DvDt - div_x_tens(sigma_f)
Cinv = Constant(1.0)
tau_M = 1.0/sqrt(((2*rho_f/Dt)**2) 
                 + inner(rho_f*v_adv,G*(rho_f*v_adv))
                 + Cinv*(mu_f**2)*inner(G,G))
resSUPG_f = inner(tau_M*resStrong_f,
                  rho_f*dot(grad_x(dv),v_adv)
                  + grad_x(dp))*det_dxdy*dy(FLUID_FLAG)
# LSIC/grad-div:
tau_C = 1.0/(tr(G)*tau_M)
resLSIC_f = tau_C*div_x(v)*div_x(dv)*det_dxdy*dy(FLUID_FLAG)

# Stable Neumann BC term, assuming advective 
# form of material time derivative term:
v_adv_minus = Min(dot(v_adv,n),Constant(0))
resOutflow_f = -dot(rho_f*v_adv_minus*dv,v)*ds(OUTFLOW_FLAG)

# Note: On a general deforming mesh, the boundary 
# integration measure ds would need to be scaled using Nanson's 
# formula, but, here, the outflow boundary is fixed so we 
# can directly use the ds measure corresponding to the 
# reference domain.

# Full fluid residual
res_f = resGal_f + resSUPG_f + resLSIC_f + resOutflow_f

# Residual of fluid--structure coupled problem:
res_fs = res_f + res_s 
Dres_fs = derivative(res_fs, w)


####### Nonlinear solver setup #######

# Nonlinear solver parameters; set relative tolerances 
# for subproblems tighter than tolerance for coupled problem, 
# to prevent stagnation.
REL_TOL_FSM = 1e-3
REL_TOL_FS = REL_TOL_M = REL_TOL_FSM*1e-1
MAX_ITERS_FSM = 100
MAX_ITERS_M = 100
MAX_ITERS_FS = 100

# Set up nonlinear problem for mesh motion:
problem_m = NonlinearVariationalProblem(res_m, uhat, 
                                        bcs_m, Dres_m)
solver_m = NonlinearVariationalSolver(problem_m)
solver_m.parameters['newton_solver']\
                   ['maximum_iterations'] = MAX_ITERS_M
solver_m.parameters['newton_solver']\
                   ['relative_tolerance'] = REL_TOL_M

# Create variational problem and solver for 
# the fluid--structure problem:
problem_fs = NonlinearVariationalProblem(res_fs, w, 
                                         bcs_fs, Dres_fs)
solver_fs = NonlinearVariationalSolver(problem_fs)
solver_fs.parameters['newton_solver']\
                    ['maximum_iterations'] = MAX_ITERS_FS
solver_fs.parameters['newton_solver']\
                    ['relative_tolerance'] = REL_TOL_FS

####### Time stepping loop #######

# Create files for storing solution:
vfile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")
mfile = File("results/mesh.pvd")

# Initialize time and step counter.
t = float(Dt)
count = 0
# Prevent divide-by-zero in relative residual on first
# iteration of first step.
for bc in bcs_fs:
    bc.apply(w.vector())
while t < TIME_INTERVAL:

    print(80*"=")
    print("  Time step "+str(count+1)+" , t = "+str(t))
    print(80*"=")
    
    # Use the current time in the inflow BC definition.
    v_in.t = t

    # "Quasi-direct" coupling: the fluid and structure 
    # are solved in one system, but the mesh is solved 
    # in a separate block.
    for i in range(0,MAX_ITERS_FSM):

        # Check fluid--structure residual on the moved
        # mesh, and terminate iteration if this residual 
        # is small:
        res_fs_vec = assemble(res_fs)
        for bc in bcs_fs:
            bc.apply(res_fs_vec,w.vector())
        res_norm = norm(res_fs_vec)
        if(i==0):
            res_norm0 = res_norm
        res_rel = res_norm/res_norm0
        print(80*"*")
        print("  Coupling iteration: "+str(i+1)
              +" , Relative residual = "+str(res_rel))
        print(80*"*")
        if(res_rel < REL_TOL_FSM):
            break

        # Solve for fluid/structure velocity and 
        # pressure at current time:
        solver_fs.solve()
        
        # Update Function in V to be used in mesh 
        # motion BC.  (There are workarounds to avoid 
        # this projection (which requires a linear
        # solve), but projection is most concise for 
        # illustration.)
        u_func.assign(project(u,V))

        # Mesh motion problem; updates uhat at current 
        # time level:
        solver_m.solve()
    
    # Extract solutions:
    (v, p) = w.split()

    # Save to file
    v.rename("v","v")
    p.rename("p","p")
    uhat.rename("u","u")
    vfile << v
    pfile << p
    mfile << uhat
    
    # Move to next time step:
    uhat_old.assign(uhat)
    w_old.assign(w)
    count += 1
    t += float(Dt)
