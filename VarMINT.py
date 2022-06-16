"""
VarMINT: Variational Multiscale Incompressible Navier--Stokes Toolkit
---------------------------------------------------------------------
This module contains routines to use a Variational Multiscale (VMS)
formulation for finite element solution of the incompressible Navier--Stokes 
equations in FEniCS.  The particular VMS formulation considered is taken from

https://doi.org/10.1007/s00466-008-0315-x
"""

from dolfin import *
import ufl

def equalOrderSpace(mesh,k=1):
    """
    Generate the most common choice of equal-order velocity--pressure 
    function space to use on a given ``mesh``.  The polynomial degree 
    ``k`` defaults to ``1``.  (Use of such equal-order space is not
    required.)
    """
    VE = VectorElement("Lagrange",mesh.ufl_cell(),k)
    QE = FiniteElement("Lagrange",mesh.ufl_cell(),k)
    VQE = MixedElement([VE,QE])
    return FunctionSpace(mesh,VQE)

def meshMetric(mesh):
    """
    Extract mesh size tensor from a given ``mesh``.
    This returns the physical element metric tensor, ``G`` as a 
    UFL object.
    """
    dx_dxiHat = 0.5*ufl.Jacobian(mesh)
    dxiHat_dx = inv(dx_dxiHat)
    G = dxiHat_dx.T*dxiHat_dx
    return G

def stabilizationParameters(v,nu,G,C_I,C_t,Dt=None,scale=Constant(1.0)):
    """
    Compute SUPS and LSIC/grad-div stabilization parameters (returned as a 
    tuple, in that order).  Input parameters are the velocity ``v``, the
    kinematic viscosity ``nu``, the mesh metric ``G``, order-one constants
    ``C_I`` and ``C_t``, a time step ``Dt`` (which may be omitted for
    steady problems), and a scaling factor that defaults to unity.  
    """
    # The additional epsilon is needed for zero-velocity robustness
    # in the inviscid limit.
    denom2 = inner(v,G*v) + C_I*nu*nu*inner(G,G) + DOLFIN_EPS
    if(Dt != None):
        denom2 += C_t/Dt**2
    tau_M = scale/sqrt(denom2)
    tau_C = 1.0/(tau_M*tr(G))
    return tau_M, tau_C

def sigmaVisc(v,mu):
    """
    The viscous part of the Cauchy stress, in terms of velocity ``v`` and
    dynamic viscosity ``mu``.
    """
    return 2.0*mu*sym(grad(v))

def sigma(v,p,mu):
    """
    The fluid Cauchy stress, in terms of velocity ``v``, pressure ``p``, 
    and dynamic viscosity ``mu``.
    """
    return sigmaVisc(v,mu) - p*Identity(ufl.shape(v)[0])

def resolvedDissipationForm(v,dv,mu,dx=dx):
    """
    The weak form of the viscous operator, in terms of velocity ``v``, 
    test function ``dv``, and dynamic viscosity ``mu``.  A volume integration
    measure ``dx`` can optionally be specified.
    """
    return inner(sigmaVisc(v,mu),grad(dv))*dx

def materialTimeDerivative(v,v_t=None,f=None):
    """
    The fluid material time derivative, in terms of the velocity ``v``, 
    the partial time derivative ``v_t`` (which may be omitted for steady
    problems), and body force per unit mass, ``f``.
    """
    DvDt = dot(v,nabla_grad(v))
    if(v_t != None):
        DvDt += v_t
    if(f != None):
        DvDt -= f
    return DvDt

def strongResidual(v,p,mu,rho,v_t=None,f=None):
    """
    The momenum and continuity residuals, as a tuple, of the strong PDE,
    system, in terms of velocity ``v``, pressure ``p``, dynamic viscosity
    ``mu``, mass density ``rho``, and, optionally, the partial time derivative
    of velocity, ``v_t``, and a body force per unit mass, ``f``.  
    """
    DvDt = materialTimeDerivative(v,v_t,f)
    i,j = ufl.indices(2)
    r_M = rho*DvDt - as_tensor(grad(sigma(v,p,mu))[i,j,j],(i,))
    r_C = rho*div(v)
    return r_M, r_C

def interiorResidual(v,p,dv,dp,rho,mu,mesh,G=None,
                     v_t=None,Dt=None,
                     f=None,
                     C_I=Constant(3.0),
                     C_t=Constant(4.0),
                     stabScale=Constant(1.0),
                     dx=dx):
    """
    This function returns the residual of the VMS formulation, minus 
    boundary terms, as a UFL ``Form``, with the assumption that boundary 
    terms will be added on top of it.  As arguments, it takes the ``mesh``, 
    the velocity ``v``, the partial derivative of velocity w.r.t. time ``v_t``, 
    the pressure ``p``, the test function (`dv`,`dp`), the time step ``Dt``, 
    the mass density ``rho``, the dynamic viscosity ``mu``, and a body
    force per unit mass, ``f``.  For steady flows, ``v_t`` and ``Dt`` 
    can be left with their default values of ``None``.  The argument ``G``
    can optionally be given a non-``None`` value, to use an alternate
    mesh size tensor.  If left as ``None``, it will be set to the
    output of ``meshMetric(mesh)``.  Optionally, one can also tune 
    the order-1 dimensionless parameters ``C_I`` and 
    ``C_t`` or a (possibly spatially-varying) scaling factor on the SUPG 
    constant, ``stabScale``.  The default values are robust for normal usage
    with linear finite elements.  ``C_I`` may need to be larger for high-order
    elements, and ``stabScale`` needs to be manipulated for use in 
    immersed-boundary methods.  A custom volume integration measure, 
    ``dx``, may also be specified.
    """

    # Do a quick consistency check, to avoid a difficult-to-diagnose error:
    if((v_t != None) and (Dt==None)):
        print("VarMINT WARNING: Missing time step in unsteady problem.")
    if((Dt != None) and (v_t==None)):
        print("VarMINT WARNING: Passing time step to steady problem.")
    if G == None:
        G = meshMetric(mesh)
    nu = mu/rho
    tau_M, tau_C = stabilizationParameters(v,nu,G,C_I,C_t,Dt,stabScale)
    DvDt = materialTimeDerivative(v,v_t,f)
    r_M, r_C = strongResidual(v,p,mu,rho,v_t,f)

    # Ansatz for fine scale velocity and pressure.  (Note that this is
    # somewhat of an abuse of notation, in that these are dimensionally-
    # inconsistent with velocity and pressure.  Additional factors of
    # density are added elsewhere to compensate.  This abuse is borrowed
    # from the literature and likely stems from the fact that the
    # formulation was originally developed in terms of kinematic viscosity,
    # without a separate density parameter.)
    vPrime = -tau_M*r_M
    pPrime = -tau_C*r_C

    return (rho*inner(DvDt,dv) + inner(sigma(v,p,mu),grad(dv))
            + inner(div(v),dp)
            - inner(dot(v,nabla_grad(dv)) + grad(dp)/rho, vPrime)
            - inner(pPrime,div(dv))
            + inner(dv,dot(vPrime,nabla_grad(v)))
            - inner(grad(dv),outer(vPrime,vPrime))/rho)*dx

def stableNeumannBC(traction,rho,v,dv,n,g=None,ds=ds,gamma=Constant(1.0)):
    """
    This function returns the boundary contribution of a stable Neumann BC
    corresponding to a boundary ``traction`` when the velocity ``v`` (with 
    corresponding test function ``dv``) is flowing out of the domain, 
    as determined by comparison with the outward-pointing normal, ``n``.  
    The optional velocity ``g`` can be used to offset the boundary velocity,
    as when this term is used to obtain a(n inflow-
    stabilized) consistent traction for weak enforcement of Dirichlet BCs.  
    The paramter ``gamma`` can optionally be used to scale the
    inflow term.  The BC is integrated using the optionally-specified 
    boundary measure ``ds``, which defaults to the entire boundary.

    NOTE: The sign convention here assumes that the return value is 
    ADDED to the residual given by ``interiorResidual``.

    NOTE: The boundary traction enforced differs from ``traction`` if 
    ``gamma`` is nonzero.  A pure traction BC is not generally stable,
    which is why the default ``gamma`` is one.  See

    https://www.oden.utexas.edu/media/reports/2004/0431.pdf

    for theory in the advection--diffusion model problem, and 

    https://doi.org/10.1007/s00466-011-0599-0

    for discussion in the context of Navier--Stokes.  
    """
    if(g==None):
        v_minus_g = v
    else:
        v_minus_g = v-g
    return -(inner(traction,dv)
             + gamma*rho*ufl.Min(inner(v,n),Constant(0.0))
             *inner(v_minus_g,dv))*ds

def weakDirichletBC(v,p,dv,dp,g,rho,mu,mesh,ds=ds,G=None,
                    sym=True,C_pen=Constant(1e3),
                    overPenalize=False):
    """
    This returns the variational form corresponding to a weakly-enforced 
    velocity Dirichlet BC, with data ``g``, on the boundary measure
    given by ``ds``, defaulting to the full boundary of the domain given by
    ``mesh``.  It takes as parameters an unknown velocity, ``v``, 
    unknown pressure ``p``, corresponding test functions ``dv`` and ``dp``, 
    mass density ``rho``, and viscosity ``mu``.  Optionally, the 
    non-symmetric variant can be used by overriding ``sym``.  ``C_pen`` is
    a dimensionless scaling factor on the penalty term.  The penalty term
    is omitted if ``not sym``, unless ``overPenalize`` is 
    optionally set to ``True``.  The argument ``G`` can optionally be given 
    a non-``None`` value, to use an alternate mesh size tensor.  If left 
    as ``None``, it will be set to the output of ``meshMetric(mesh)``.

    NOTE: The sign convention here assumes that the return value is 
    ADDED to the residual given by ``interiorResidual``.

    For additional information on the theory, see

    https://doi.org/10.1016/j.compfluid.2005.07.012
    """
    n = FacetNormal(mesh)
    sgn = 1.0
    if(not sym):
        sgn = -1.0
    if G == None:
        G = meshMetric(mesh) # $\sim h^{-2}$
    traction = sigma(v,p,mu)*n
    consistencyTerm = stableNeumannBC(traction,rho,v,dv,n,g=g,ds=ds)
    # Note sign of ``dp``, negative for stability, regardless of ``sym``.
    adjointConsistency = -sgn*dot(sigma(dv,-sgn*dp,mu)*n,v-g)*ds
    penalty = C_pen*mu*sqrt(dot(n,G*n))*dot((v-g),dv)*ds
    retval = consistencyTerm + adjointConsistency
    if(overPenalize or sym):
        retval += penalty
    return retval
