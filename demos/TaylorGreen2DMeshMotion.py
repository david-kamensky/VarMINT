"""
This demo solves the 2D Taylor--Green vortex problem. The problem is solved 
on the reference mesh with a semi-arbitrary mesh displacement, ''uhat''.
"""

from VarMINT import *
import math
import argparse

###############################################################
#### Parameters ###############################################
###############################################################

# Arguments are parsed from the command line, with some hard-coded defaults
# here.  See help messages for descriptions of parameters.

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--Nel',dest='Nel',default=16,
                    help='Number of elements in each direction.')
parser.add_argument('--Re',dest='Re',default=100.0,
                    help='Reynolds number.')
parser.add_argument('--k',dest='k',default=1,
                    help='Polynomial degree.')
parser.add_argument('--T',dest='T',default=1.0,
                    help='Length of time interval to consider.')
parser.add_argument('--viz',dest='viz',default=False,action='store_true',
                    help='Store XDMF File for visualization.')

args = parser.parse_args()
Nel = int(args.Nel)
Re = Constant(float(args.Re))
k = int(args.k)
T = float(args.T)
viz = bool(args.viz)

###############################################################
#### Parameters ###############################################
###############################################################

# Avoid overkill automatically-determined quadrature degree:
dy = dx(metadata={"quadrature_degree":2*k})

# Domain:
mesh = RectangleMesh(Point(-pi,-pi),
                     Point(pi,pi),
                     Nel,Nel)

# Mixed velocity--pressure space:
V = equalOrderSpace(mesh,k=k)
v_space = VectorFunctionSpace(mesh,"CG",k)
p_space = FunctionSpace(mesh,"CG",k)

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
t = Constant(0.0)

# Determine physical parameters from Reynolds number:
rho = Constant(1.0)
mu = Constant(1.0/Re)
nu = mu/rho

# Mesh displacement for the Taylor--Green vortex 
# NOTE: The mesh velocity must be zero at the boundaries with the current
# boundary conditions.
y = SpatialCoordinate(mesh)
A = Constant(0.2)
freq = Constant(1)
term = A*sin(y[0])*sin(y[1])*sin(2*freq*pi*t/T)
uhat = as_vector((term,term))
vhat = diff(uhat,t)

# Midpoint mesh displacements and velocities
uhat_old = Function(v_space)
vhat_old = Function(v_space)
uhat_old.assign(project(uhat,v_space))
vhat_old.assign(project(vhat,v_space))
uhat_mid = 0.5*(uhat+uhat_old)
vhat_mid = 0.5*(vhat+vhat_old)

# Space and time dependencies of the exact solutions:
x = y + uhat
solnT = exp(-2.0*nu*t)

# Exact velocity solution:
v_exact = solnT*as_vector((cos(x[0])*sin(x[1]),-sin(x[0])*cos(x[1])))
v_exact_func = project(v_exact,v_space)
v_exact_func.rename("v_exact","v_exact")

# Exact pressure solution:
p_exact = -(solnT**2)*rho/4*(cos(2*x[0])+cos(2*x[1]))
p_exact_func = project(p_exact,p_space)
p_exact_func.rename("p_exact","p_exact")

# Weak problem residual; note use of midpoint velocity:
res = interiorResidual(v_mid,p,dv,dp,rho,mu,mesh,
                       uhat=uhat_mid,
                       vhat=vhat_mid,
                       v_t=v_t,
                       Dt=Dt,
                       dy=dy)
Dres = derivative(res,vp)

# Project the initial condition:
vp_old.assign(project(as_vector((v_exact[0],v_exact[1],p_exact)),V))

# Same-velocity predictor:
vp.assign(vp_old)

# Set BCs on velocity to match exact solution
corner_str = "near(x[0],-pi) && near(x[1],-pi)"
bcs = [DirichletBC(V.sub(0),v_exact_func,"on_boundary"),
        DirichletBC(V.sub(1),p_exact_func,corner_str,"pointwise"),]

# Set up output file for visualization
# NOTE: To correctly visualize the solution in the current configuration in
# Paraview, use a "Warp by Vector" filter with uhat as the warping function to
# move from the reference to the current configutation. 
results = XDMFFile("results/v-p-uhat-results.xdmf")
results.parameters["flush_output"] = True
results.parameters["functions_share_mesh"] = True
results.parameters["rewrite_function_mesh"] = False

# function for the L2Norm of a function (accepts Expressions):
def L2Norm(f):
    return math.sqrt(assemble(inner(f,f)*dy))


# Time stepping loop:
for step in range(0,N_STEPS):
    print("========== Time step "+str(step+1)+"/"+str(N_STEPS)+" ==========")
    
    # move old solution, time, and exact solutions to the next time step
    vp_old.assign(vp)
    uhat_old.assign(project(uhat,v_space))
    vhat_old.assign(project(vhat,v_space))
    t.assign(t+Dt)
    p_exact_func.assign(project(p_exact,p_space))
    v_exact_func.assign(project(v_exact,v_space))

    # solve the current time step
    solve(res==0,vp,bcs=bcs,J=Dres)

    # save results for visualization
    if viz:
        (v,p) = vp.split()
        v.rename("v","v")
        p.rename("p","p")
        uhat_func = project(uhat,v_space)
        uhat_func.rename("uhat","uhat")
        results.write(v,float(t))
        results.write(p,float(t))
        results.write(uhat_func,float(t))
        results.write(v_exact_func,float(t))
        results.write(p_exact_func,float(t))

    # Check error at the current time step:
    e_v = v - v_exact
    print("Error analysis at the current time step:")
    print("  Element size =               " + str(2.0*math.pi/Nel))
    print("  H1 seminorm velocity error = " + str(L2Norm(gradx(e_v,x))))
    print("  L2 norm velocity error =     " + str(L2Norm(e_v)))
