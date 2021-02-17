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
VarMINT is intentionally light-weight, and mainly intended to avoid needless duplication of UFL code defining the VMS formulation in different applications.  A more comprehensive FEniCS-based flow solver using a similar VMS formulation is described by Zhu and Yan [here](https://doi.org/10.1016/j.camwa.2019.07.034).