# VarMINT
The **Var**iational **M**ultiscale **I**ncompressible **N**avier--Stokes **T**oolkit:  A small Python module with re-usable functionality for solving the Navier--Stokes equations in [FEniCS](https://fenicsproject.org/), using a variational multiscale (VMS) formulation.

This module was originally written to support the following paper, submitted to a special issue on open-source software for partial differential equations:
```
@article{Kamensky2021,
title = "Open-source immersogeometric analysis of fluid--structure interaction using {FEniCS} and {tIGAr}",
journal = "Computers \& Mathematics with Applications",
volume = "81",
pages = "634--648",
year = "2021",
note = "Development and Application of Open-source Software for Problems with Numerical PDEs",
issn = "0898-1221",
author = "D. Kamensky"
}
```

It has since been updated to include be based on an Arbitrary Lagrangian-Eulerian (ALE) framework to facilitate fluid-structure interaction simulations for the below publication:
```
@article{IN PREPARATION}
```

VarMINT is intentionally light-weight, and mainly intended to avoid needless duplication of UFL code defining the VMS formulation in different applications.  A more comprehensive FEniCS-based flow solver using a similar VMS formulation is described by Zhu and Yan [here](https://doi.org/10.1016/j.camwa.2019.07.034).

The **S**olid **M**echanics **A**nd **N**onlinear **E**lasticity **R**outines module, SalaMANdER, provides a framework for implementing solid material models that avoids code duplication where possible.

The **C**onvert **M**esh utility, ChaMeleon, was written to convert meshes generated with [GMSH](gmsh.info) to FEniCS-compatible XDMF-format meshes using [meshio](https://pypi.org/project/meshio/).

