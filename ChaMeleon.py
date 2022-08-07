'''
ChaMeleon: Convert Mesh
---------------------------------------------------------------------

Convert a mesh from GMSH (.msh) to XDMF (.xdmf + .h5) that is compatible with
FEniCS with independent mesh, subdomain, and boundary files.

Dependencies:
    meshio - see https://pypi.org/project/meshio/

Written by:
    Grant Neighbor
    Mechanical Engineering
    Iowa State University
    Ames, Iowa, United States

Date written:
    February 22, 2022

Liscense:
    This script and any of it's components may be copied and reused in any 
    form. The author provides no guarantee of the program's performance, 
    outcome, or results, including any effects or changes on the host system.
'''

###############################################################
#### Housekeeping and Imports #################################
###############################################################

# initial imports
import argparse
import os
import meshio

# build command-line parser and parse arguments
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s',dest='MARKER',default='markers',
                    help="String to mark subdomains and boundaries.")
parser.add_argument('-m',dest='MESH',default='./mesh.msh',
                    help="Path to GMSH mesh input and XDMF output.")
parser.add_argument('-d',dest='DOMAIN_ELEMENT',default='tetra',
                    help="Domain element type in GMSH file.")
parser.add_argument('-b',dest='BOUNDARY_ELEMENT',default='triangle',
                    help="Boundary element type in GMSH file.")
parser.add_argument('-z',dest='TRIM_Z',default=False,action='store_true',
                    help="Trim off the z-coordinates for a 2D mesh.")

# parse command-line arguments
args = parser.parse_args()

# mesh inmport parameters
FILEPATH_GMSH_MESH = args.MESH

# mesh export parameters
# save location in the 
BASE = os.path.splitext(FILEPATH_GMSH_MESH)[0]
FILEPATH_MESH = BASE + "_mesh.xdmf"
FILEPATH_SUBDOMAINS = BASE + "_subdomains.xdmf"
FILEPATH_BOUNDARIES = BASE + "_boundaries.xdmf"
MARKER = args.MARKER
DOMAIN_ELEMENT = args.DOMAIN_ELEMENT
BOUNDARY_ELEMENT = args.BOUNDARY_ELEMENT
TRIM_Z_COORDINATE = args.TRIM_Z


###############################################################
#### Import GMSH Information ##################################
###############################################################
print('Reading gmsh file...')
msh = meshio.read(FILEPATH_GMSH_MESH)
points = msh.points
if TRIM_Z_COORDINATE:
    points = msh.points[:,0:2]
cells = {DOMAIN_ELEMENT: msh.cells_dict[DOMAIN_ELEMENT]}
boundaries = {BOUNDARY_ELEMENT: msh.cells_dict[BOUNDARY_ELEMENT]}


###############################################################
#### Export XDMF Information ##################################
###############################################################

# write background mesh
print('Writing background mesh...')
meshio.write(FILEPATH_MESH, 
            meshio.Mesh(points = points, 
                        cells = cells))

# write subdomains mesh
print('Writing subomain data...')
meshio.write(FILEPATH_SUBDOMAINS, 
            meshio.Mesh(points = points, 
                        cells = cells,
                        cell_data = {MARKER: 
                                    [msh.cell_data_dict["gmsh:physical"]
                                    [DOMAIN_ELEMENT]]}))
# write boundary mesh
print('Writing boundary data...')
meshio.write(FILEPATH_BOUNDARIES, 
            meshio.Mesh(points = points, 
                        cells = boundaries,
                        cell_data = {MARKER: 
                                    [msh.cell_data_dict["gmsh:physical"]
                                    [BOUNDARY_ELEMENT]]}))


###############################################################
#### Display Mesh Groupings ###################################
###############################################################

# peek at the next iterator item without popping it
def peek(f):
    pos = f.tell()
    line = f.readline()
    f.seek(pos)
    return line

# display data labels to user
print('Mesh Groupings: ')
print("dimension | ID | name")
with open(FILEPATH_GMSH_MESH,'r') as f:
    line = f.readline()
    while (line != '$PhysicalNames'):
        line = f.readline()[:-1]
    while (peek(f)[:-1] != '$EndPhysicalNames'):
        line = f.readline()[:-1]
        print(line)

print('Done. Goodbye!')