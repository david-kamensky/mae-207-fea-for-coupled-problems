"""
Code snippet to illustrate action of `derivative` function.
"""

from dolfin import *
from ufl import replace
mesh = UnitCubeMesh(1,1,1)
V = FunctionSpace(mesh,"CG",1)
u = Function(V)
u.rename("u","u")
a = (sin(u)**2 + 1)*dot(grad(u),grad(u))
du = TestFunction(V)
deriv_a = derivative(a,u,du)
deriv_a = replace(deriv_a,{du:Function(V,name="du")})
print(deriv_a)
