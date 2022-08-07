"""
VarMINT: Variational Multiscale Incompressible Navier--Stokes Toolkit
---------------------------------------------------------------------
This module contains routines to use a Variational Multiscale (VMS)
formulation for finite element solution of the incompressible Navier--Stokes 
equations in FEniCS.  The particular VMS formulation considered is taken from

https://doi.org/10.1007/s00466-008-0315-x.

Here, the core formulation follows the current configuration, but each gradient
operation and integrand is transformed to be with respect to the reference
configuration coordinate system. This is to simplify the implementation for
mesh-moving cases where the solid and mesh problems are more easily formulated
with a Lagrangian framework.
"""

from dolfin import *
from dolfin import dx as dy
import ufl

def gradx(f,x):
    """
    Returns the gradient of ``f`` with respect to coordinate ``x``.
    """
    return dot(grad(f),inv(grad(x)))

def divx(f,x):
    """
    Returns the divergence of vector ``f`` with respect to coordinate ``x``.
    """
    return tr(gradx(f,x))

def divx_tens(f,x):
    """
    Returns the divergence of tensor ``f`` with respect to coordinate ``x``.
    """
    i,j = indices(2)
    return as_tensor(gradx(f,x)[i,j,j],(i,))

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

def meshMetric(mesh,x):
    """
    Extract mesh size tensor from a given ``mesh`` in the configuration
    specified by spatial coordinate ``x``. This returns the physical element
    metric tensor, ``G`` as a UFL object.
    """
    dydz = 0.5*ufl.Jacobian(mesh)
    y = SpatialCoordinate(mesh)
    dxdy = gradx(x,y)
    dxdz = dxdy*dydz
    dzdx = inv(dxdz)
    G = dzdx.T*dzdx
    return G

def stabilizationParameters(v,vhat,nu,G,C_I,C_t,Dt=None,scale=Constant(1.0)):
    """
    Compute SUPS and LSIC/grad-divx stabilization parameters (returned as a
    tuple, in that order).  Input parameters are the velocity ``v``,  the mesh
    velocity ``vhat``, the kinematic viscosity ``nu``, the mesh metric ``G``,
    order-one constants ``C_I`` and ``C_t``, a time step ``Dt`` (which may be
    omitted for steady problems), and a scaling factor that defaults to unity.  
    """
    # The additional epsilon is needed for zero-velocity robustness
    # in the inviscid limit.
    denom2 = inner(v-vhat,G*(v-vhat)) + C_I*nu*nu*inner(G,G) + DOLFIN_EPS
    if(Dt != None):
        denom2 += C_t/Dt**2
    tau_M = scale/sqrt(denom2)
    tau_C = 1.0/(tau_M*tr(G))
    return tau_M, tau_C

def sigmaVisc(v,x,mu):
    """
    The viscous part of the Cauchy stress, in terms of velocity ``v``,
    coordinate ``x``, and dynamic viscosity ``mu``.
    """
    return 2.0*mu*sym(gradx(v,x))

def sigma(v,p,x,mu):
    """
    The fluid Cauchy stress, in terms of velocity ``v``, pressure ``p``,
    coordinate ``x``, and dynamic viscosity ``mu``.
    """
    return sigmaVisc(v,x,mu) - p*Identity(ufl.shape(x)[0])

def resolvedDissipationForm(v,dv,x,mu,dy=dy):
    """
    The weak form of the viscous operator, in terms of velocity ``v``, test
    function ``dv``, coordinate ``x``, and dynamic viscosity ``mu``.  A volume
    integration measure ``dy`` can optionally be specified.
    """
    return inner(sigmaVisc(v,x,mu),gradx(dv,x))*dy

def materialTimeDerivative(v,vhat,x,v_t,f):
    """
    The fluid material time derivative, in terms of the velocity ``v``, mesh
    velocity ``vhat``, the partial time derivative ``v_t``, and body force per 
    unit mass, ``f``.
    """ 
    return gradx(v,x)*(v-vhat) + v_t - f

def strongResidual(v,p,x,vhat,mu,rho,v_t,f):
    """
    The momentum and continuity residuals, as a tuple, of the strong PDE,
    system, in terms of velocity ``v``, pressure ``p``, coordinate ``x```, mesh
    velocity ``vhat``, dynamic viscosity ``mu``, mass density ``rho``, the
    partial time derivative of velocity ``v_t``, and a body force per unit mass
    ``f``.  
    """
    DvDt = materialTimeDerivative(v,vhat,x,v_t,f)
    i,j = ufl.indices(2)
    r_M = rho*DvDt - as_tensor(gradx(sigma(v,p,x,mu),x)[i,j,j],(i,))
    r_C = rho*divx(v,x)
    return r_M, r_C

def interiorResidual(v,p,dv,dp,rho,mu,mesh,
                     uhat=None,vhat=None,v_t=None,Dt=None,f=None,
                     C_I=Constant(3.0),
                     C_t=Constant(4.0),
                     stabScale=Constant(1.0),
                     dy=dy):
    """
    This function returns the residual of the VMS formulation, minus boundary
    terms, as a UFL ``Form``, with the assumption that boundary terms will be
    added on top of it.  As arguments, it takes the ``mesh``, the velocity
    ``v``, the partial derivative of velocity w.r.t. time ``v_t``, the pressure
    ``p``, the test function (`dv`,`dp`), the time step ``Dt``, the mass
    density ``rho``, the dynamic viscosity ``mu``, a body force per unit mass,
    ``f``, a mesh displacement ``uhat``, and a mesh velocity ``vhat``.  For
    steady flows, ``v_t`` and ``Dt`` can be left with their default values of
    ``None``. For non-moving meshes, ``uhat`` and ``vhat`` can be left as their
    default values of ``None``. Optionally, one can also tune the order-1
    dimensionless parameters ``C_I`` and ``C_t`` or a (possibly
    spatially-varying) scaling factor on the SUPG constant, ``stabScale``. The
    default values are robust for normal usage with linear finite elements.
    ``C_I`` may need to be larger for high-order elements, and ``stabScale``
    needs to be manipulated for use in immersed-boundary methods.  A custom
    volume integration measure, ``dy``, may also be specified. This form should
    be assembled on the reference mesh configuration.
    """

    uhat, vhat, v_t, f = noneToZeroVector(mesh,uhat,vhat,v_t,f)
    
    y = SpatialCoordinate(mesh)
    x = y + uhat
    J = det(gradx(x,y))
    
    G = meshMetric(mesh,x)
    nu = mu/rho
    tau_M, tau_C = stabilizationParameters(v,vhat,nu,G,C_I,C_t,Dt,
                                           scale=stabScale)
    DvDt = materialTimeDerivative(v,vhat,x,v_t,f)
    r_M, r_C = strongResidual(v,p,x,vhat,mu,rho,v_t,f)

    # Ansatz for fine scale velocity and pressure.  (Note that this is
    # somewhat of an abuse of notation, in that these are dimensionally-
    # inconsistent with velocity and pressure.  Additional factors of
    # density are added elsewhere to compensate.  This abuse is borrowed
    # from the literature and likely stems from the fact that the
    # formulation was originally developed in terms of kinematic viscosity,
    # without a separate density parameter.)
    vPrime = -tau_M*r_M
    pPrime = -tau_C*r_C

    return J*(rho*inner(DvDt,dv) + inner(sigma(v,p,x,mu),gradx(dv,x))
              + inner(divx(v,x),dp)
              - inner(gradx(dv,x)*(v-vhat) + gradx(dp,x)/rho, vPrime)
              - inner(pPrime,divx(dv,x))
              + inner(dv,gradx(v,x)*vPrime)
              - inner(gradx(dv,x),outer(vPrime,vPrime))/rho)*dy

def stableNeumannBC(traction,rho,v,dv,mesh,
                    g=None,uhat=None,vhat=None,
                    ds=ds,gamma=Constant(1.0)):
    """
    This function returns the boundary contribution of a stable Neumann BC
    corresponding to a boundary ``traction`` when the velocity ``v`` (with
    corresponding test function ``dv``) is flowing out of the domain, as
    determined by comparison with the outward-pointing normal, ``n``.  
    The optional velocity ``g`` can be used to offset the boundary velocity, as
    when this term is used to obtain a(n inflow- stabilized) consistent
    traction for weak enforcement of Dirichlet BCs. The mesh displacement ``uhat``
    and velocity ``vhat`` can also be specified when applicable.  
    The paramter ``gamma`` can optionally be used to scale the inflow term.
    The BC is integrated using the optionally-specified boundary measure
    ``ds``, which defaults to the entire boundary. 
    
    NOTE: The sign convention here assumes that the return value is ADDED to
    the residual given by ``interiorResidual``. 
    
    NOTE: The boundary traction enforced differs from ``traction`` if ``gamma``
    is nonzero.  A pure traction BC is not generally stable, which is why the
    default ``gamma`` is one.  See

    https://www.oden.utexas.edu/media/reports/2004/0431.pdf  

    for theory in the advection--diffusion model problem, and 

    https://doi.org/10.1007/s00466-011-0599-0

    for discussion in the context of Navier--Stokes.  
    """
    uhat, vhat, g = noneToZeroVector(mesh,uhat,vhat,g)

    n = FacetNormal(mesh)
    y = SpatialCoordinate(mesh)
    x = y + uhat
    F = gradx(x,y)
    J = det(F)

    return -(inner(J*inv(F.T)*traction,dv)
             + gamma*rho*ufl.Min(inner(v-vhat,n),Constant(0.0))
             *inner(J*inv(F.T)*(v-g),dv))*ds

def weakDirichletBC(v,p,dv,dp,g,rho,mu,mesh,uhat=None,vhat=None,ds=ds,
                    sym=True,C_pen=Constant(1e3),
                    overPenalize=False):
    """
    This returns the variational form corresponding to a weakly-enforced
    velocity Dirichlet BC, with data ``g``, on the boundary measure given by
    ``ds``, defaulting to the full boundary of the domain given by ``mesh``.
    It takes as parameters an unknown velocity, ``v``, unknown pressure ``p``,
    corresponding test functions ``dv`` and ``dp``, mass density ``rho``,
    viscosity ``mu``, mesh displacement ``uhat`` and mesh velocity ``vhat``.
    Optionally, the non-symmetric variant can be used by overriding ``sym``.
    ``C_pen`` is a dimensionless scaling factor on the penalty term.  The
    penalty term is omitted if ``not sym``, unless ``overPenalize`` is
    optionally set to ``True``. 

    NOTE: The sign convention here assumes that the return value is ADDED to
    the residual given by ``interiorResidual``. 
    
    For additional information on the theory, see 
    
    https://doi.org/10.1016/j.compfluid.2005.07.012
    """
    uhat, vhat = noneToZeroVector(mesh,uhat,vhat)
    
    y = SpatialCoordinate(mesh)
    x = y + uhat
    F = gradx(x,y)
    J = det(F)
    n = FacetNormal(mesh)

    sgn = 1.0
    if(not sym):
        sgn = -1.0
    G = meshMetric(mesh,x) # $\sim h^{-2}$
    traction = sigma(v,p,x,mu)*n
    consistencyTerm = stableNeumannBC(traction,rho,v,dv,mesh,
                                      uhat=uhat,vhat=vhat,g=g,ds=ds)
    # Note sign of ``dp``, negative for stability, regardless of ``sym``.
    adjointConsistency = -sgn*dot(J*inv(F.T)*(sigma(dv,-sgn*dp,x,mu)*n),v-g)*ds
    penalty = C_pen*mu*sqrt(dot(n,G*n))*dot(J*inv(F.T)*(v-g),dv)*ds
    retval = consistencyTerm + adjointConsistency
    if(overPenalize or sym):
        retval += penalty
    return retval

def noneToZeroVector(mesh,*args):
    """
    Returns a zero UFL vector matching the ``mesh`` geometric dimension for each
    input in ``args`` that is ``None``. This function only returns vectors,
    not scalars.
    """
    nsd = mesh.geometric_dimension()
    zero = Constant(nsd*(0,))
    outputs = []

    for item in args:
        if (item==None):
            item = zero
        outputs.append(item)

    return outputs
