"""
This demo solves the regularized lid-driven cavity problem, to verify optimal 
convergence of the VMS formulation using weak Dirichlet boundary conditions.  
The problem is adapted from a benchmark in

https://doi.org/10.1002/fld.1650090206

but uses an artificial pressure field, rather than the one emerging from the
calculations of the linked refernce.  Additionally, an optional Galilean 
transformation can be applied, to fully test Dirichlet data with nonzero 
normal components.
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
parser.add_argument('--Galilean_x',dest='Galilean_x',default=0.0,
                    help='x-component of Galilean transformation.')
parser.add_argument('--Galilean_y',dest='Galilean_y',default=0.0,
                    help='y-component of Galilean transformation.')
parser.add_argument('--nonSymBC',dest='nonSymBC',action='store_true',
                    help='Use symmetric Nitsche method for weak BCs.')
parser.add_argument('--viz',dest='viz',action='store_true',
                    help='Write ParaView files with solution.')
parser.add_argument('--overPenalize',dest='overPenalize',action='store_true',
                    help='Use penalty with non-symmetric Nitsche method.')
parser.add_argument('--C_pen',dest='C_pen',default=1e3,
                    help='Non-dimensional scaling of Nitsche penalty term.')

args = parser.parse_args()
Nel = int(args.Nel)
Re = Constant(float(args.Re))
k = int(args.k)
v_Gal = as_vector((Constant(float(args.Galilean_x)),
                   Constant(float(args.Galilean_y))))
symBC = (not bool(args.nonSymBC))
overPenalize = bool(args.overPenalize)
C_pen = Constant(float(args.C_pen))
viz = bool(args.viz)

####### Analysis #######

# Avoid overkill automatically-determined quadrature degree:
QUAD_DEG = 2*k
dx = dx(metadata={"quadrature_degree":QUAD_DEG})
ds = ds(metadata={"quadrature_degree":QUAD_DEG})

# Domain:
mesh = UnitSquareMesh(Nel,Nel)

# Mixed velocity--pressure space:
V = equalOrderSpace(mesh,k=k)

# Solution and test functions:
vp = Function(V)
v,p = split(vp)
dvdp = TestFunction(V)
dv,dp = split(dvdp)

# Determine physical parameters from Reynolds number:
rho = Constant(1.0)
mu = Constant(1.0/Re)

# Exact solution:
x = SpatialCoordinate(mesh)
v_exact = as_vector((8.0*(pow(x[0],4) - 2.0*pow(x[0],3)
                          + pow(x[0],2))*(4.0*pow(x[1],3) - 2.0*x[1]),
                     -8.0*(4.0*pow(x[0],3) - 6.0*pow(x[0],2)
                           + 2.0*x[0])*(pow(x[1],4) - pow(x[1],2)))) + v_Gal
p_exact = sin(pi*x[0])*sin(pi*x[1])

# Manufacture a solution:
zero = Constant(mesh.geometric_dimension()*(0,))
f,_ = strongResidual(v_exact,p_exact,x,zero,mu,rho,zero,zero)

# Weak problem residual:
F = interiorResidual(v,p,dv,dp,rho,mu,mesh,
                     f=f,C_I=Constant(6.0*(k**4)),dy=dx)
F += weakDirichletBC(v,p,dv,dp,v_exact,
                     rho,mu,mesh,
                     ds=ds,
                     sym=symBC,C_pen=C_pen,
                     overPenalize=overPenalize)

# Pin pressure at one node to remove hydrostatic mode:
bc = DirichletBC(V.sub(1),Constant(0),"x[0]<DOLFIN_EPS && x[1]<DOLFIN_EPS",
                 "pointwise")

# Solve for velocity and pressure:
solve(F==0,vp,bcs=[bc,])

# Output solution, if requested via --viz argument:
if(viz):
    v_out,p_out = vp.split()
    v_out.rename("v","v")
    p_out.rename("p","p")
    File("results/v.pvd") << v_out
    File("results/p.pvd") << p_out

# Check error in velocity:
import math
def L2Norm(f):
    return math.sqrt(assemble(inner(f,f)*dx))
e_v = v-v_exact
print("Element size = "+str(1.0/Nel))
print("H1 seminorm velocity error = "+str(L2Norm(grad(e_v))))
print("L2 norm velocity error = "+str(L2Norm(e_v)))
