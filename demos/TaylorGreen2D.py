"""
This demo solves the 2D Taylor--Green vortex problem, illustrating usage
in unsteady problems and demonstrating spatio-temporal convergence under 
quasi-uniform space--time refinement.  
"""
from VarMINT import *

####### Parameters #######

# Arguments are parsed from the command line, with some hard-coded defaults
# here.  See help messages for descriptions of parameters.

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--Nel',dest='Nel',default=16,
                    help='Number of elements in each direction.')
parser.add_argument('--Re',dest='Re',default=100.0,
                    help='Reynolds number.')
parser.add_argument('--k',dest='k',default=1,
                    help='Polynomial degree.')
parser.add_argument('--T',dest='T',default=1.0,
                    help='Length of time interval to consider.')

args = parser.parse_args()
Nel = int(args.Nel)
Re = Constant(float(args.Re))
k = int(args.k)
T = float(args.T)

####### Analysis #######

# Avoid overkill automatically-determined quadrature degree:
QUAD_DEG = 2*k
dx = dx(metadata={"quadrature_degree":QUAD_DEG})

# Domain:
import math
mesh = RectangleMesh(Point(-math.pi,-math.pi),
                     Point(math.pi,math.pi),
                     Nel,Nel)

# Mixed velocity--pressure space:
V = equalOrderSpace(mesh,k=k)

# Solution and test functions:
vp = Function(V)
v,p = split(vp)
dvdp = TestFunction(V)
dv,dp = split(dvdp)

# Midpoint time integration:
N_STEPS = Nel # Space--time quasi-uniformity
Dt = Constant(T/N_STEPS)
vp_old = Function(V)
v_old,_ = split(vp_old)
v_mid = 0.5*(v+v_old)
v_t = (v-v_old)/Dt

# Determine physical parameters from Reynolds number:
rho = Constant(1.0)
mu = Constant(1.0/Re)
nu = mu/rho

# Initial condition for the Taylor--Green vortex
x = SpatialCoordinate(mesh)
v_IC = as_vector((sin(x[0])*cos(x[1]),-cos(x[0])*sin(x[1])))

# Time dependence of exact solution, evaluated at time T:
solnT = exp(-2.0*nu*T)
v_exact = solnT*v_IC

# Weak problem residual; note use of midpoint velocity:
F = interiorResidual(v_mid,p,dv,dp,rho,mu,mesh,
                     v_t=v_t,Dt=Dt,C_I=Constant(6.0*(k**4)),dy=dx)

# Project the initial condition:
vp_old.assign(project(as_vector((v_IC[0],v_IC[1],Constant(0.0))),V))

# Same-velocity predictor:
vp.assign(vp_old)

# Set no-penetration BCs on velocity and pin down pressure in one corner:
corner_str = "near(x[0],-pi) && near(x[1],-pi)"
bcs = [DirichletBC(V.sub(0).sub(0),Constant(0.0),
                   "near(x[0],-pi) || near(x[0],pi)"),
       DirichletBC(V.sub(0).sub(1),Constant(0.0),
                   "near(x[1],-pi) || near(x[1],pi)"),
       DirichletBC(V.sub(1),Constant(0.0),corner_str,"pointwise")]

# Time stepping loop:
for step in range(0,N_STEPS):
    print("======= Time step "+str(step+1)+"/"+str(N_STEPS)+" =======")
    solve(F==0,vp,bcs=bcs)
    vp_old.assign(vp)

# Check error:
def L2Norm(f):
    return math.sqrt(assemble(inner(f,f)*dx))
e_v = v - v_exact
print("Element size = "+str(2.0*math.pi/Nel))
print("H1 seminorm velocity error = "+str(L2Norm(grad(e_v))))
print("L2 norm velocity error = "+str(L2Norm(e_v)))
