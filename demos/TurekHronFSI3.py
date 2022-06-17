"""
Turek & Hron's FSI3 benchmark from https://doi.org/10.1007/3-540-34596-5_15. 

This demo requires GMSH (https://gmsh.info/) for meshing and meshio
(https://pypi.org/project/meshio/) for mesh conversion.

To generate the mesh for this problem, use the attached TurekHronGMSH.geo file
(contains geometry and mesh refinement information) and GMSH to generate the
mesh. Then convert it with the convertmesh.py utility provided with this
repository (it splits out the proper mesh, subdomain marker, and boundary
marker files). NOTE: Since this is a 2D problem, the z-coordinates need to be
trimmed in the conversion process, the domain element should be "triangle", and
the boundary element should be "line".
"""

import os

from dolfin import *
import VarMINT as alevms
import SalaMANdER as sm

from numpy import infty

# define MPI communicators
COMM = MPI.comm_world
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

# surpress excessive logging in parallel
from dolfin.cpp.log import LogLevel
from dolfin.cpp.log import log
set_log_active(False)
if (RANK==0):
    set_log_active(True)

# build command-line parser and parse arguments
import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-m",dest="MESH_FOLDER",default="./",type=str,
                    help="Folder that contains the mesh, subdomains, \
                    and boundary files.")
parser.add_argument("-r",dest="RESULTS_FOLDER",default="./",
                    type=str,help="Folder to save the results in.")
args = parser.parse_args()

# enumerate mesh flags
FLAG_NONE = 0
FLAG_FLUID = 1
FLAG_ELASTIC = 2
FLAG_INLET = 3
FLAG_OUTLET = 4
FLAG_WALLS = 5
FLAG_INTERFACE_SOLID = 6
FLAG_INTERFACE_ELASTIC = 7
FLAG_ELASTIC_INTERIOR = 9

# Geometry parameters
A = Point((0.6,0.2))

# Mesh import hard-coded defaults
MARKER_STRING = 'markers'
MESH_FOLDER = args.MESH_FOLDER
FILEPATH_MESH = MESH_FOLDER + "TurekHron_mesh.xdmf"
FILEPATH_SUBDOMAINS = MESH_FOLDER + "TurekHron_subdomains.xdmf"
FILEPATH_BOUNDARIES = MESH_FOLDER + "TurekHron_boundaries.xdmf"

# file export details
VIS_SKIP = 100
VIS_SKIP_STEADY_STATE = 10
QOI_SKIP = 1
RESULTS_FOLDER = args.RESULTS_FOLDER
QOI_FILE = RESULTS_FOLDER + "qoi-data.csv"
OUT_FILE = RESULTS_FOLDER + "v-p-uhat-results.xdmf"
if (RANK==0 and not os.path.exists(RESULTS_FOLDER)):
    os.mkdir(RESULTS_FOLDER)

# problem parameters
K = 1   # polynomial degree
TIME_INTERVAL = 16
STEADY_STATE_TIME = 6
STEPS_PER_TIME = 1000
N_STEPS = TIME_INTERVAL*STEPS_PER_TIME
UBAR = Constant(2)

# nonlinear solver parameters
REL_TOL_FSM = 1e-3
REL_TOL_FS = REL_TOL_M = REL_TOL_FSM*1e-1
MAX_ITERS_FSM = 5
MAX_ITERS_M = 15
MAX_ITERS_FS = 15


####### Domain and mesh setup #######

# import mesh
mesh = Mesh()
with XDMFFile(COMM, FILEPATH_MESH) as f:
    f.read(mesh)

# Mesh-derived quantities:
nsd = mesh.geometry().dim()
n = FacetNormal(mesh)
I = Identity(nsd)
h = CellDiameter(mesh)

# import subdomains
mvc = MeshValueCollection('size_t', mesh, nsd)
with XDMFFile(COMM, FILEPATH_SUBDOMAINS) as f:
    f.read(mvc, MARKER_STRING)
subdomains = cpp.mesh.MeshFunctionSizet(mesh, mvc)

# import boundaries
mvc = MeshValueCollection('size_t', mesh, nsd-1)
with XDMFFile(COMM, FILEPATH_BOUNDARIES) as f:
    f.read(mvc, MARKER_STRING)
boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)

# initiate mesh connectivities
mesh.init(nsd, nsd-1)
mesh.init(0, nsd-1)
mesh.init(nsd,0)

# initialize and fill a facet function for the solid region, leaving off 
# anything touching the interface facets
elastic_interior = MeshFunction("size_t", mesh, nsd-1)
elastic_interior.set_all(FLAG_ELASTIC_INTERIOR)
for facet in facets(mesh):
    marker = False
    for cell in cells(facet):
        marker = marker or subdomains[cell]==FLAG_FLUID
    if (not marker):
        for vertex in vertices(facet):
            vertex_facets = vertex.entities(nsd-1)
            for vertex_facet in vertex_facets:
                marker = marker or \
                    (boundaries[vertex_facet]==FLAG_INTERFACE_ELASTIC)
    if marker:
        elastic_interior[facet] = FLAG_NONE

# Set up integration measures, with flags to integrate over
# subsets of the domain.
dx = dx(metadata={'quadrature_degree': 2*K},
        subdomain_data=subdomains,
        domain=mesh)
ds = ds(metadata={'quadrature_degree': 2*K},
        subdomain_data=boundaries,
        domain=mesh)

####### Elements and function spaces #######

# Define function spaces (equal order interpolation):
# Mixed function space for velocity and pressure:
W = alevms.equalOrderSpace(mesh,k=K)
# Function space for mesh displacement field, 
# which will be solved for separately in a 
# quasi-direct scheme:
V = VectorFunctionSpace(mesh,"CG",K)

####### Set up time integration variables #######

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


####### Boundary conditions #######

# inlet boundary condition
v_x = '1.5*ubar*4.0/.1681*x[1]*(.41-x[1])'
v_in = Expression(("(t<2.0) ? ((1-cos(pi/2*t))/2*"+v_x+") : ("+v_x+")","0.0"),
                  degree=4, t=0.0, ubar=float(UBAR))

# shorthand for vector zero
zero = Constant(nsd*(0,))

# BCs for the fluid--structure subproblem:
bcs_fs = [
    DirichletBC(W.sub(0), zero, boundaries, FLAG_WALLS),
    DirichletBC(W.sub(0), v_in, boundaries, FLAG_INLET),
    DirichletBC(W.sub(1), Constant(0), elastic_interior, 
                FLAG_ELASTIC_INTERIOR),
    DirichletBC(W.sub(0), zero, boundaries, FLAG_INTERFACE_SOLID),
]

# BCs for the mesh motion subproblem:
bcs_m = [
    DirichletBC(V,u_func,elastic_interior,FLAG_ELASTIC_INTERIOR),
    DirichletBC(V,u_func,boundaries,FLAG_INTERFACE_ELASTIC),
    DirichletBC(V,zero,boundaries,FLAG_INTERFACE_SOLID),
    DirichletBC(V,zero,boundaries,FLAG_WALLS),
    DirichletBC(V,zero,boundaries,FLAG_INLET),
    DirichletBC(V,zero,boundaries,FLAG_OUTLET),
]


####### Formulation of mesh motion subproblem #######

# Residual for mesh, which satisfies a fictitious elastic problem:
mesh_model = sm.JacobianStiffening(nsd)
res_m = mesh_model.interiorResidual(uhat,du,dx=dx)
Dres_m = derivative(res_m, uhat)


####### Formulation of the solid subproblem #######

# Elastic properties
nu_s = Constant(0.4)
mu_s = Constant(2e6)
E_s = 2*mu_s*(1+nu_s)
rho_s0 = Constant(1e3)

dX = dx(FLAG_ELASTIC)
solid_model = sm.StVenantKirchoff(nsd, E_s, nu_s, rho_s0)
res_s = solid_model.interiorResidual(u,dv,dx=dX)
res_s += solid_model.accelerationResidual(dv_ds,dv,dx=dX)

####### Formulation of the fluid subproblem #######

rho_f = Constant(1e3)
nu_f = Constant(1e-3)
mu_f = nu_f*rho_f
dy = dx(FLAG_FLUID)
res_f = alevms.interiorResidual(v,p,dv,dp,rho_f,mu_f,mesh,
                     uhat=uhat,
                     vhat=vhat,
                     v_t=dv_dr,
                     Dt=Dt,
                     dy=dy)

# Residual of fluid--structure coupled problem:
res_fs = res_f + res_s 
Dres_fs = derivative(res_fs, w)


####### Nonlinear solver setup #######

# Set up nonlinear problem for mesh motion:
problem_m = NonlinearVariationalProblem(res_m, uhat, 
                                        bcs_m, Dres_m)
solver_m = NonlinearVariationalSolver(problem_m)
solver_m.parameters['newton_solver']\
                   ['maximum_iterations'] = MAX_ITERS_M
solver_m.parameters['newton_solver']\
                   ['relative_tolerance'] = REL_TOL_M
solver_m.parameters['newton_solver']['linear_solver'] = 'mumps'

# Create variational problem and solver for 
# the fluid--structure problem:
problem_fs = NonlinearVariationalProblem(res_fs, w, 
                                         bcs_fs, Dres_fs)
solver_fs = NonlinearVariationalSolver(problem_fs)
solver_fs.parameters['newton_solver']\
                    ['maximum_iterations'] = MAX_ITERS_FS
solver_fs.parameters['newton_solver']\
                    ['relative_tolerance'] = REL_TOL_FS
solver_fs.parameters['newton_solver']['linear_solver'] = 'mumps'

# Create files for storing solution:
outFile = XDMFFile(COMM,OUT_FILE)
outFile.parameters["flush_output"] = True
outFile.parameters["functions_share_mesh"] = True
outFile.parameters["rewrite_function_mesh"] = False
    
# Report the problem size:
log(LogLevel.INFO,"Fluid-solid problem DOFs: " + str(W.dim()))
log(LogLevel.INFO,"Mesh problem DOFs: " + str(V.dim()))



####### Time stepping loop #######

# Initialize time and step counter.
t = float(Dt)
step = 0

# Prevent divide-by-zero in relative residual on first
# iteration of first step.
for bc in bcs_fs:
    bc.apply(w.vector())

while step < N_STEPS:

    if (t>STEADY_STATE_TIME):
        VIS_SKIP = VIS_SKIP_STEADY_STATE

    if (RANK==0):
        log(LogLevel.INFO,100*"=")
        log(LogLevel.INFO,"  Time step "+str(step+1)+" , t = "+str(t))
        log(LogLevel.INFO,100*"=")
    
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
        res_rel = 0
        if (abs(res_norm0) > DOLFIN_EPS):
            res_rel = res_norm/res_norm0

        # log the current coupling iteration
        log(LogLevel.INFO,65*"*")
        log(LogLevel.INFO,"  Coupling iteration: "+str(i+1)
            +" , Relative residual = "+str(res_rel))
        log(LogLevel.INFO,65*"*")

        # break coupling iteration converged
        if (res_rel < REL_TOL_FSM):
            break
        if (i==MAX_ITERS_FSM-1):
            warning("Coupling iteration not converged.")

        # Solve for fluid/structure velocity and pressure at current time
        solver_fs.solve()
        
        # Update Function in V to be used in mesh 
        # motion BC.  (There are workarounds to avoid 
        # this projection (which requires a linear
        # solve), but projection is most concise for 
        # illustration.)
        u_func.assign(project(u,V,solver_type='gmres'))

        # Mesh motion problem; updates uhat at current 
        # time level:
        solver_m.solve()
    
    # Extract solutions:
    (v, p) = w.split()

    # Save visualization to file
    if (step%VIS_SKIP==0):
        v.rename("v","v")
        p.rename("p","p")
        uhat.rename("u","u")
        outFile.write(v,t)
        outFile.write(p,t)
        outFile.write(uhat,t)

    # compute benchmark checks
    if (step%QOI_SKIP==0):

        # tip displacement, accounted for different processors
        d_x = -infty
        d_y = -infty
        bbt = mesh.bounding_box_tree()
        if bbt.compute_first_entity_collision(A) < mesh.num_cells():
            d_x = uhat(A)[0]
            d_y = uhat(A)[1]
        d_x = MPI.max(COMM, d_x)
        d_y = MPI.max(COMM, d_y)

        # lift and drag coefficients
        interfaces = [FLAG_INTERFACE_ELASTIC,FLAG_INTERFACE_SOLID]
        traction = []

        for direction in range(nsd):
            test_function = Function(W)
            boundary_value = nsd*[0,]
            boundary_value[direction] = 1
            boundary_value = Constant(tuple(boundary_value))

            for interface in interfaces:
                bc = DirichletBC(W.sub(0),boundary_value,boundaries,interface)
                bc.apply(test_function.vector())
            
            dv_traction,dp_traction = split(test_function)
            traction.append(assemble(alevms.interiorResidual(v,p,
                dv_traction,dp_traction,rho_f,mu_f,mesh,
                uhat=uhat,vhat=vhat,v_t=dv_dr,Dt=Dt,dy=dy)))


        # save QOI data
        if (RANK==0):
            if (step==0):
                with open(QOI_FILE,'w') as f:
                    f.write("step, time, A_x, A_y, drag, lift \n")
            else:
                with open(QOI_FILE,'a') as f:
                    f.write(str(step)+","+str(t)+","+\
                            str(d_x)+","+str(d_y)+","+\
                            str(traction[0])+","+str(traction[1])+" \n")
    
    # Move to next time step:
    uhat_old.assign(uhat)
    w_old.assign(w)
    step += 1
    t += float(Dt)