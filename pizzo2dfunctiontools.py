#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:16:16 2022

@author: hannahhaider

This script defines the functions needed to run a 2D model of the Solar Wind using
the governing equations in Pizzo (1979) and with the addition of an artifical viscosity, 
mu* partial(q)/partial(phi^2).

It is adapted from the 3D_HD_SW code found on github.com/opaliss

The numerical schemes included are:

1) Euler - upwind (1st order) 2d pizzo 
2) MacCormack - predictor/corrector (2nd order) 2d pizzo
3) Euler - upwind (1st order) 2d pizzo + viscosity 
4) MacCormack - predictor/corrector (2nd order) 2d pizzo + viscosity 
    
please see the functionsfortools file for helper functions. 
"""

#%%
#import needed modules and helper functions  
import numpy as np
import astropy.units as u
from astropy.constants import G
from astropy import constants as const
from functionsfortools import ddx_fwd, ddx_bwd, diffusive
#%% defining needed functions
def cs_func(Pr, rho, gamma = 5/3):
    """this function returns the cs**2 variable, where cs**2 = gamma*Pr/rho"""
    return (gamma*Pr)/rho

def alpha_func(ur, cs):
    """this function returns the alpha**2 variable, where alpha**2 = ur**2 - cs**2"""
    return ur**2 - cs

def u_func(up, r, theta, omega_rot = ((2 * np.pi) / (25.38 * 86400) * (1 / u.s)).value):
    """This function returns the variable u, where u = uphi - omega_rot(rsin(theta)) - time to spatial"""
    return up - omega_rot*r.value*np.sin(theta)

def H_forward_phi(dqdp, q, r, theta = np.pi/2):
    """This function returns the RHS source term for our governing PDE solving the 2D 
    Pizzo (1979) hyperbolic system of equations
    
    -------
    inputs: 
        dqdp = the derivative of the primitive variables (state vector, q) with respect to phi
        
        q = state vector of primitive variables: 
            q(r,phi) = (radial velocity (ur), density (rho), Pressure (Pr), phi velocity (up))
            
        r = radius 
        
        theta = latitude / equatorial slice (90 degrees)
        """
    ur, rho, Pr, up = q
    cs = cs_func(Pr=Pr, rho=rho)
    alpha =  alpha_func(ur = ur, cs = cs)
    u = u_func(up = up, r = r, theta = theta)
    H_coeff = -1/(alpha*r*np.sin(theta))
    H1_dqdp = H_coeff * (u*ur*dqdp[0] - (u/rho)*dqdp[2] -cs*dqdp[3])
    H2_dqdp = H_coeff * (-rho*u*dqdp[0] + alpha*(u/ur)*dqdp[1] + (u/ur)*dqdp[2] + rho*ur*dqdp[3])
    H3_dqdp = H_coeff * (-cs*rho*u*dqdp[0] + u*ur*dqdp[2] + cs*rho*ur*dqdp[3])
    H4_dqdp = H_coeff * ((alpha/(rho*ur))*dqdp[2] + alpha*(u/ur)*dqdp[3])
    return np.array([H1_dqdp, H2_dqdp, H3_dqdp, H4_dqdp])

def c_vec(q, r, G = G.to(u.km**3 / (u.kg * u.s * u.s)).value, M_s = const.M_sun.to(u.kg).value):
    """this function returns the c vector defined in 2D coordinates, to be called for use in the g vector
    function
    -------
    parameters:
        q = state vector
        r = radius 
        G = gravitational constant in km^3/kgs^2
        M_s = solar mass in kg
    """
    ur, rho, Pr, up = q
    c1 = ( 1 / r.value )*(up**2 - G*(M_s / r.value))
    c2 =(1 / r.value)*(-2*rho*ur)
    c3 = np.zeros(c1.shape) # equal to zero,  but must be of same elements as rest of c vector
    c4 = (1 / r.value)*(-ur*up) #unsure if this one is correct
    return np.array([c1,c2,c3,c4])
def G_vec(q, r):
    """this function returns the g vector defined for 2D coordinates, for use in the RHS of our governing PDE"""
    ur, rho, Pr, up = q
    cs = cs_func(Pr = Pr, rho = rho)
    alpha = alpha_func(ur, cs)
    G_coeff = (1/alpha)
    c1, c2, c3, c4 = c_vec(q,r)
    G1 = G_coeff*(ur*c1 - cs*(c2/rho))
    G2 = G_coeff*(ur*c2 - rho*c1)
    G3 = G_coeff*(cs*ur*c2 - cs*rho*c1)
    G4 = G_coeff*(c4/ur)
    return np.array([G1,G2,G3,G4])
def boundary_conditions(q):
    """this function defines right-handside and left-handside boundary conditions for second order schemes
    ------ 
    """
    # primitive variables
    vr, rho, Pr, vp = q
    # left side second order
    vr[-1] = (-vr[-2] + 4 * vr[-3]) / 3
    rho[-1] = (-rho[ -2] + 4 * rho[-3]) / 3
    Pr[-1] = (-Pr[ -2] + 4 * Pr[-3]) / 3
    vp[-1] = (-vp[ -2] + 4 * vp[-3]) / 3
    
    # right side second order
    vr[0] = (-vr[2] + 4 * vr[1]) / 3
    rho[0] = (-rho[ 2] + 4 * rho[1]) / 3
    Pr[ 0] = (-Pr[2] + 4 * Pr[1]) / 3
    vp[ 0] = (-vp[2] + 4 * vp[1]) / 3

    return np.array([vr, rho, Pr, vp])

def Pizzo_2D_Euler_Upwind(q,r,dp,dr):
    """This function computes the upwind or propogated radial step of any primitive variable in the state vector q
    ------
    parameters:
        q = state vector
        r = radius
        dp = phi step index
        dr = radial step index
    """
    ur, rho, Pr, up = q
    #computing derivative w.r.t. phi as input for the RHS source term
    dqdp = np.array([ddx_fwd(q[0], dp, periodic = True, order = 1), #dur/dp
                     ddx_fwd(q[1], dp, periodic = True, order = 1), #drho/dp
                     ddx_fwd(q[2], dp, periodic = True, order = 1), #dPr/dp
                     ddx_fwd(q[3], dp, periodic = True, order = 1)]) #dup/dp
    H = H_forward_phi(dqdp, q, r)
    G = G_vec(q,r)
    q_pred = q + dr.value*(H + G)
    q_pred = boundary_conditions(q = q_pred)
    return q_pred 
def MacCormack_2D_corr(q_pred, q, r, dp, dr, theta = np.pi / 2):
    """This function computes the correcter step of the MacCormack scheme. The predictor step is equivalent to
    the Euler upwind scheme, where q_pred is returned. 
    ------
    parameters: 
        q_pred = predicted state vector
        r = radius 
        dp = phi step index
        dr = radial step index 
    """
    dqdp_pred = np.array([ddx_bwd(q_pred[0], dp, periodic = True, order = 1), #dur/dp
                     ddx_bwd(q_pred[1], dp, periodic = True, order = 1), #drho/dp
                     ddx_bwd(q_pred[2], dp, periodic = True, order = 1), #dPr/dp
                     ddx_bwd(q_pred[3], dp, periodic = True, order = 1)]) #dup/dp
    #getting predicted values of H, G
    H_pred = H_forward_phi(dqdp_pred, q_pred, r + dr)
    G_pred = G_vec(q_pred, r + dr)
    
    #corrector step
    q_final = 0.5 * (q_pred + q + dr.value * (H_pred + G_pred)) # or taking the average, 0.5 * (U_pred + U + dr.value * (G_pred + H_pred))
    q_final = boundary_conditions(q = q_pred)
    return q_final 
def Pizzo_MacCormack_2D(q, dp, dr, r, theta = np.pi / 2):
    """This function combines the predictor and corrector step of the MacCormack scheme,
    to be called as only one function when modeling
    ------
    parameters:
        q = solution state vector 
        r = radius
        dp = phi step index
        dr = radial step index 
        theta = longitudinal slice, equatorial slice = np.pi/2
    """
    #calling predicator step
    q_pred = Pizzo_2D_Euler_Upwind(q = q, r = r, dp = dp, dr = dr)
    #corrector step 
    q_final = MacCormack_2D_corr(q_pred = q_pred, q = q, r = r, dp = dp, dr = dr, theta = theta)
    q_final = boundary_conditions(q = q_final)
    return q_final
#%% Modified MacCormack
def Euler_Upwind_2D(Cx, q, r, dp, dr):
    """This function computes the upwind or propogated radial step of any primitive variable in the state vector q
    ------
    parameters:
        q = state vector
        r = radius
        dp = phi step index
        dr = radial step index
        Cx = artifical viscosity coefficient
    """
    ur, rho, Pr, up = q
    #computing derivative w.r.t. phi as input for the RHS source term
    dqdp = np.array([ddx_fwd(q[0], dp, periodic = True, order = 1), #dur/dp
                     ddx_fwd(q[1], dp, periodic = True, order = 1), #drho/dp
                     ddx_fwd(q[2], dp, periodic = True, order = 1), #dPr/dp
                     ddx_fwd(q[3], dp, periodic = True, order = 1)]) #dup/dp
    #computing 2nd derivative
    #dq2dp2 = np.array([ddx_central(q[0], dp, periodic = True, order = 2), #d2ur/dp2
                       #ddx_central(q[1], dp, periodic = True, order = 2), #d2rho/dp2
                       #ddx_central(q[2], dp, periodic = True, order = 2), #d2Pr/dp2
                       #ddx_central(q[3], dp, periodic = True, order = 2)]) #d2up/dp2
    #print(dq2dp2)
    #diff = np.array([Cx*dq2dp2[0], Cx*dq2dp2[1], Cx*dq2dp2[2], Cx*dq2dp2[3] ])
   
    d2qdp2 = np.array([diffusive(Cx, q[0], dr.value, dp, 129, 400), 
                       diffusive(Cx, q[1], dr.value, dp, 129, 400), 
                       diffusive(Cx, q[2], dr.value, dp, 129, 400), 
                       diffusive(Cx, q[3], dr.value, dp, 129, 400)])
    H = H_forward_phi(dqdp, q, r)
    G = G_vec(q,r)
    q_pred = q + dr.value*(H + G + d2qdp2)
    q_pred = boundary_conditions(q = q_pred)
    return q_pred 

def MacCormack_2D_corr_v(Cx, q_pred, q, r, dp, dr, theta = np.pi / 2):
    """This function computes the correcter step of the MacCormack scheme. The predictor step is equivalent to
    the Euler upwind scheme, where q_pred is returned. 
    ------
    parameters: 
        q_pred = predicted state vector
        r = radius 
        dp = phi step index
        dr = radial step index 
        Cx = artificial viscosity coefficient
    """
    #computed predicted convective term 
    dqdp_pred = np.array([ddx_bwd(q_pred[0], dp, periodic = True, order = 1), #dur/dp
                          ddx_bwd(q_pred[1], dp, periodic = True, order = 1), #drho/dp
                          ddx_bwd(q_pred[2], dp, periodic = True, order = 1), #dPr/dp
                          ddx_bwd(q_pred[3], dp, periodic = True, order = 1)]) #dup/dp
    #computing predicted diffusive term 
    #dq2dp2_pred = np.array([ddx_central(q_pred[0], dp, periodic = True, order = 2), #d2ur/dp2
                            #ddx_central(q_pred[1], dp, periodic = True, order = 2), #d2rho/dp2
                            #ddx_central(q_pred[2], dp, periodic = True, order = 2), #d2Pr/dp2
                            #ddx_central(q_pred[3], dp, periodic = True, order = 2)])
    
    #diff = np.array([Cx*dq2dp2_pred[0], Cx*dq2dp2_pred[1], Cx*dq2dp2_pred[2], Cx*dq2dp2_pred[3]])
    
    d2qdp2_pred = np.array([diffusive(Cx, q_pred[0], dr.value, dp, 129, 400), 
                            diffusive(Cx, q_pred[1], dr.value, dp, 129, 400), 
                            diffusive(Cx, q_pred[2], dr.value, dp, 129, 400), 
                            diffusive(Cx, q_pred[3], dr.value, dp, 129, 400)])
    
    #getting predicted values of H, G
    H_pred = H_forward_phi(dqdp_pred, q_pred, r + dr) #returns an array of H*dqdp
    G_pred = G_vec(q_pred, r + dr)
    
    #corrector step
    q_final = 0.5 * (q_pred + q + dr.value * (H_pred + G_pred + d2qdp2_pred)) 
    q_final = boundary_conditions(q = q_pred)
    return q_final 

def MacCormack_2D(q, dp, dr, r, Cx, theta = np.pi / 2):
    """This function combines the predictor and corrector step of the MacCormack scheme,
    to be called as only one function when simulating
    ------
    parameters:
        q = solution state vector 
        r = radius
        dp = phi step index
        dr = radial step index 
        Cx = artificial viscosity coefficient 
        theta = longitudinal slice, equatorial slice = np.pi/2
    """
    #calling predicator step
    q_pred = Euler_Upwind_2D(Cx = Cx, q = q, r = r, dp = dp, dr = dr)
    #corrector step 
    q_final = MacCormack_2D_corr_v(Cx = Cx, q_pred = q_pred, q = q, r = r, dp = dp, dr = dr,theta = theta) 
    q_sol = boundary_conditions(q = q_final)
    return q_sol