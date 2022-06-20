"""
SalaMANdER: Solid Mechanics And Nonlinear Elasticity Routines
---------------------------------------------------------------------
This module contains routines for nonlinear solid mechanics formulations for 
use in FEniCS (www.fenicsproject.org) scripts.

The intent here is rapid prototyping of new material models through a subclass
of MaterialModel that implements a constructor and the interiorResidual method.
That way, the other items common to nearly all solid formulations do not have
to be duplicated for each new material model.
"""
from abc import ABC, abstractmethod
from dolfin import *


class MaterialModel(ABC):
    """
    The abstract base model for a material that includes the material 
    properties. This abstract class leaves the interiorResidual to each 
    subclass material.
    """
    E = Constant(0,name="E")            # Young's modulus
    nu = Constant(0,name="nu")          # Poisson's ratio
    kappa = Constant(0,name="kappa")    # build modulus
    mu = Constant(0,name="mu")          # shear modulus
    lmbda = Constant(0,name="lmbda")    # Lame parameter

    rho = Constant(0,name="rho")        # material density
    nsd = 0                             # number of spatial dimensions

    def __init__(self,nsd,E,nu,rho):
        self.nsd = nsd 
        self.E.assign(E)
        self.nu.assign(nu)
        self.kappa.assign(E/(3*(1-2*nu)))
        self.mu.assign(E/(2*(1 + nu)))
        self.lmbda.assign(E*nu/((1 + nu)*(1 - 2*nu)))
        self.rho.assign(rho)
        super().__init__()

    @abstractmethod
    def interiorResidual(self,u,v,dx=dx):
        pass

    def accelerationResidual(self,du_dtt,v,dx=dx):
        return self.rho*inner(du_dtt,v)*dx

    def massDampingResidual(self,du_dt,c,v,dx=dx):
        return self.rho*c*inner(du_dt,v)*dx

    def bodyforceResidual(self,f,v,dx=dx):
        return - self.rho*inner(f,v)*dx

    def tractionBCResidual(self,h,v,ds=ds):
        return - inner(v,h)*ds

    def get_basic_tensors(self,u):
        I = Identity(self.nsd)
        F = grad(u) + I
        J = det(F)
        C = F.T*F
        E = 0.5*(C-I)
        return I,F,J,C,E



class StVenantKirchoff(MaterialModel):
    """ A St. Venant-Kirchoff material."""

    def interiorResidual(self,u,v,dx=dx):
        I,F,_,_,E = self.get_basic_tensors(u)
        S = self.kappa*tr(E)*I + 2.0*self.mu*(E - tr(E)*I/3.0)
        return inner(F*S,grad(v))*dx


class NeoHookean(MaterialModel):
    """ A Neo-Hookean material from Wu et al. 2019."""

    def interiorResidual(self,u,v,S0=None,dx=dx):
        I,F,J,C,_ = self.get_basic_tensors(u)
        if (S0==None):
            S0 = Constant(0)*I
        S = self.mu*(J**(-2/3))*(I-tr(C)*inv(C)/3) \
            + self.kappa/2*((J**2)-1)*inv(C)
        return inner(F*(S+S0),grad(v))*dx


class JacobianStiffening(MaterialModel):
    """ Jacobian-based mesh stiffening."""
    def __init__(self,nsd,power=3):
        self.power = Constant(power,name="power")
        zero = Constant(0)
        super().__init__(nsd,zero,zero,zero)

    def interiorResidual(self, u, v, dx=dx):
        I,F,J,_,E = self.get_basic_tensors(u)
        K = Constant(1)/pow(J,self.power)
        mu = Constant(1)/pow(J,self.power)
        S = K*tr(E)*I + 2.0*mu*(E - tr(E)*I/3.0)
        return inner(F*S,grad(v))*dx