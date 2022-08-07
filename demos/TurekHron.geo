// Turek-Hron FSI Benchmark Mesh
// Created with GMSH version 4.7.1
// Grant Neighbor, Iowa State University
// Mar 30, 2022

////////////////////////////////////////////////
// Geometry Creation
////////////////////////////////////////////////

SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 2.5, .41, 0};
Circle(5) = {.2, .2, 0, .05, 0, 2*Pi};
Rectangle(2) = {.6, .19, 0, -.4, .02, 0};
Curve Loop(3) = {3, 4, 1, 2};
Curve Loop(4) = {8, 9, 6, 7};
Curve Loop(5) = {5};

Curve Loop(6) = {5};
Plane Surface(3) = {6};
BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; Surface{3}; }
BooleanDifference{ Surface{2}; Delete; }{ Surface{3}; Delete;}
Coherence;


////////////////////////////////////////////////
// Physical Groups
////////////////////////////////////////////////
Physical Surface("fluid", 1) = {1};
Physical Surface("elastic", 2) = {2};

Physical Curve("inlet",3) = {11};
Physical Curve("outlet",4) = {12};
Physical Curve("walls",5) = {13, 10};
Physical Curve("interface-solid",6) = {15,17,18};
Physical Curve("interface-elastic",7) = {14, 9, 16};



////////////////////////////////////////////////
// Element Sizing
////////////////////////////////////////////////

// Element Sizing
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 1;
Mesh.MeshSizeExtendFromBoundary = 0;
//Mesh.CharacteristicLengthFactor = 1;

// Element Sizes
h_ref = .0080;  // Reference: M4=.001, M3=.002, M2=.004, M1=.008
h_max = .0150;
h_min = .0005;

Mesh.MeshSizeMax = h_max;
Mesh.MeshSizeMin = h_min;

// Interface Mesh Refinement
Field[1] = Distance;
Field[1].SurfacesList = {2};
Field[1].CurvesList = {15,17,18};
Field[2] = Threshold;
Field[2].DistMax = .55;
Field[2].DistMin = .025;
Field[2].InField = 1;
Field[2].SizeMin = h_ref;
Field[2].SizeMax = h_max;

// Background Field
Background Field = 2;

