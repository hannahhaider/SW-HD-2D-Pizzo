#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 13:34:51 2022

@author: hannahhaider

This is the main script for modeling and simulating the 2D Pizzo governing PDE
Adapted from github.com/opaliss/3D_HD_SW
"""
#importing packages and functions
from functionsfortools import get_mas_path
from psipy.model import MASOutput
from pizzo2dfunctiontools import Pizzo_2D_Euler_Upwind, Pizzo_MacCormack_2D, MacCormack_2D, Euler_Upwind_2D
import numpy as np
from astropy.constants import m_p
import matplotlib.pyplot as plt
import matplotlib
import astropy.units as u 
#import matplotlib.animation as animation
import imageio
import os 

# first, we need to load the data and save our data as a model
# load data using get_mas_path
cr = "2210" #defined as a string for url purposes, getting carrington rotation 2210
mas_path = get_mas_path(cr = cr) #sets cr2210 as the MAS helio data 
model = MASOutput(mas_path)
print(model.variables) #prints ['bp', 'br', 'bt', 'p', 'rho', 't', 'vp', 'vr', 'vt']

# MHD mesh coordinates Heliographic (rotating) coordinate system 

# phi - (0, 2pi)
phi =model["vr"].phi_coords
# delta phi
dp = phi[1]-phi[0]

# latitude, theta
theta = model["vr"].theta_coords

# we want to access the radii from Ro = 30 solar radii to 1 AU
# 1 solar radii = 695,700 km
r = (model["vr"].r_coords * u.solRad).to(u.km) #to is a function in astropy units
new_r = np.linspace(r[10], r[-1], int(400)) #creating 400 equally spaced radial steps
# delta r 
dr = new_r[1] - new_r[0]

#last phi index is 6.23 < 2pi, append 2pi to phi
phi = np.append(phi, 2*np.pi)
vp = model["vp"].data * (u.km / u.s)
vp = np.append(vp, [vp[0, :, :]], axis = 0)

vr = model["vr"].data * (u.km / u.s)
vr = np.append(vr, [vr[0, :, :]], axis = 0)

rho = np.array(model["rho"].data) * m_p  # multiply by kg
rho = (rho * (1 / u.cm ** 3)).to(u.kg / u.km ** 3).value  # convert to mks (km)
rho = np.append(rho, [rho[0, :, :]], axis=0)

Pr = np.array(model["p"].data)
Pr = ((Pr * (u.dyne / u.cm ** 2)).to(u.kg / (u.s ** 2 * u.km)))  # convert to mks (km)
Pr = np.append(Pr, [Pr[0, :, :]], axis=0)

# convert phi to degrees
phi_deg = 180 / np.pi * phi
# convert theta to degrees 
theta_deg = 180 / np.pi * theta
#%% Plotting Upwind Euler
U_SOL = np.zeros((4, len(phi), len(new_r))) #preallocating data structure
k1 = vr[:, 55, 0, 0]*(1 + 0.25*(1-np.exp(-30/50))) #for all phi, at equatorial slice, and initial Ro
k2 = rho[:, 55, 0, 0]
k3 = Pr[:, 55, 0, 0]
k4 = vp[:, 55, 0, 0]*(1 + 0.25*(1-np.exp(-30/50)))
U_SOL[:, :, 0] = np.array([k1,k2,k3,k4])  #initial condition (initial radial step)

#now creating main loop to propogate forward/upwind using euler
filenames = [] # initializing filename data structure 
for i in range(len(new_r) - 1):
    U_SOL[:,:, i +1] = Pizzo_2D_Euler_Upwind(q = U_SOL[:,:,i], 
                                             r = new_r[i],
                                             dp = dp, 
                                             dr = dr) 
    #solving for U_i^n+1 by evaluating Euler with incremented r values, and state vector = U_SOL
    if i %25 == 0: #modulo operator, if the division between the incremented r and 25 is equal to 0, then
        #print(i) #prints radius in increments of 25, from 0 to 400
        #print((new_r[i]).to(u.AU)) #prints radius in AU
        #begin plotting
        fig, ax = plt.subplots(nrows = 4, sharex = True, figsize = (5,10))
        #plot radial velocity vs phi
        #pos = ax[0].pcolormesh([phi_deg, theta_deg], U_SOL[0,:,i + 1], 
                               #shading='gouraud', 
                               #cmap="viridis") #plot solution vector (flow parameter[0] = vr) against phi in degrees
        #cbar = fig.colorbar(pos, ax = ax[0])
        #cbar.ax[0].set_ylabel(r'$\frac{km}{s}$')
        pos = ax[0].plot(phi_deg, U_SOL[0,:,i+1])
        ax[0].set_ylabel(r'$\frac{km}{s}$')
        ax[0].set_title(r"$v_{r}$") #labeling radial velocity 
        #plot number density vs phi, convert to cm^-3 by dividing by mass_proton and converting from km^-3 to cm^-3
        pos = ax[1].plot(phi_deg,
                        ((U_SOL[1, :, i + 1] / m_p.value) * (1 / u.km ** 3)).to(1 / u.cm ** 3))
        ax[1].set_ylabel(r'$\frac{1}{cm^3}$')
        ax[1].set_title(r"$n_{p}$")
        #plot pressure vs. phi, convert to dyne/cm^2 from kg/(kms^2) 
        pos = ax[2].plot(phi_deg,
                        (U_SOL[2, :, i + 1] * (u.kg / (u.s ** 2 * u.km))).to(u.dyne / (u.cm ** 2)))
        ax[2].set_ylabel(r'$\frac{dyne}{cm^2}$')
        ax[2].set_title(r"$P$") 
        #plot longitudinal velocity v_phi vs. phi
        pos = ax[3].plot(phi_deg, U_SOL[3, :, i + 1])
        ax[3].set_ylabel(r'$\frac{km}{s}$')
        ax[3].set_title(r"$v_{\phi}$")
        ax[3].set_xticks([0, 90, 180, 270, 360])
        ax[3].set_xlabel("Carrington Longitude (Deg.)")
        fig.suptitle("[Euler] r = " + str(round(new_r[i + 1].to(u.AU).value, 3))) #plot for each r value in AU units
        plt.tight_layout()
        
        filename = f'{i}.png' # creating filename for each r value
        filenames.append(filename) # appending filename to the list of filenames 
        fig.savefig(filename)
        plt.close()

# build gif
with imageio.get_writer('Euler.gif', mode = 'I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
for filename in set(filenames):
    os.remove(filename)  
"""Animation of the results   
states = ["vr", "rho", "Pr", "vp"]
    for state in range(4):
    
        fig1 = plt.figure()
        axs = plt.axes()
        myAnimation, = axs.plot([],[], ':ob', linewidth = 2)
        plt.grid()
        plt.xlabel("Carrington Longitude (deg.)", fontsize = 16)
        plt.ylabel(states[state], fontsize = 16)

        def animate(i):
            if state == 0 or state == 3: #if state is vr or vp
                plt.plot(phi_deg, U_SOL[state,:,i])
                myAnimation.set_data(phi_deg, U_SOL[state,:,i])
            elif state == 1: #rho
                plt.plot(phi_deg, U_SOL[state,:,i]/ m_p.value * (1 / u.km ** 3).to(1 / u.cm ** 3))
                myAnimation.set_data(phi_deg, U_SOL[state,:,i]/ m_p.value * (1 / u.km ** 3).to(1 / u.cm ** 3))
            elif state == 2: #Pr
                plt.plot(phi_deg, U_SOL[state,:,i]* (u.kg / (u.s ** 2 * u.km)).to(u.dyne / (u.cm ** 2)))
                myAnimation.set_data(phi_deg, U_SOL[state,:,i]* (u.kg / (u.s ** 2 * u.km)).to(u.dyne / (u.cm ** 2)))
            return myAnimation,

        anim = animation.FuncAnimation(fig,animate,frames=range(1,len(r)),blit=True)"""
        
#%% Adopting MacCormack technique 
U_SOL_mac = np.zeros((4, len(phi), len(new_r))) #preallocating data structure
k1 = vr[:, 55, 0, 0]*(1 + 0.25*(1-np.exp(-30/50))) #for all phi, at equatorial slice, and initial Ro
k2 = rho[:, 55, 0, 0]
k3 = Pr[:, 55, 0, 0]
k4 = vp[:, 55, 0, 0]*(1 + 0.25*(1-np.exp(-30/50)))
U_SOL_mac[:, :, 0] = np.array([k1,k2,k3,k4])  #initial condition (initial radial step)

filenames = [] # preallocating data structure 
for i in range(len(new_r)-1):
    U_SOL_mac[:,:, i + 1] = Pizzo_MacCormack_2D(q = U_SOL_mac[:,:,i], 
                                                dp = dp, 
                                                dr = dr, 
                                                r = new_r[i])
    if i %25 == 0: #modulo operator, if the division between the incremented r and 25 is equal to 0, then
       # print(i) #prints radius in increments of 25, from 0 to 400
       # print((new_r[i]).to(u.AU)) #prints radius in AU
        #begin plotting
        fig2, ax = plt.subplots(nrows = 4, sharex = True, figsize = (5,10))
        #plot radial velocity vs phi
        pos = ax[0].plot(phi_deg, U_SOL_mac[0,:,i + 1]) #plot solution vector (flow parameter[0] = vr) against phi in degrees
        ax[0].set_ylabel(r'$\frac{km}{s}$')
        ax[0].set_title(r"$v_{r}$") #labeling radial velocity 
        #plot number density vs phi, convert to cm^-3 by dividing by mass_proton and converting from km^-3 to cm^-3
        pos = ax[1].plot(phi_deg,
                        ((U_SOL_mac[1, :, i + 1] / m_p.value) * (1 / u.km ** 3)).to(1 / u.cm ** 3))
        ax[1].set_ylabel(r'$\frac{1}{cm^3}$')
        ax[1].set_title(r"$n_{p}$")
        #plot pressure vs. phi, convert to dyne/cm^2 from kg/(kms^2) 
        pos = ax[2].plot(phi_deg,
                        (U_SOL_mac[2, :, i + 1] * (u.kg / (u.s ** 2 * u.km))).to(u.dyne / (u.cm ** 2)))
        ax[2].set_ylabel(r'$\frac{dyne}{cm^2}$')
        ax[2].set_title(r"$P$") 
        #plot longitudinal velocity v_phi vs. phi
        pos = ax[3].plot(phi_deg, 
                         U_SOL_mac[3, :, i + 1])
        ax[3].set_ylabel(r'$\frac{km}{s}$')
        ax[3].set_title(r"$v_{\phi}$")
        ax[3].set_xticks([0, 90, 180, 270, 360])
        ax[3].set_xlabel(" Carrington Longitude (Deg.)")
        fig2.suptitle("[MacCormack] r = " + str(round(new_r[i + 1].to(u.AU).value, 3))) #plot for each r value in AU units
        plt.tight_layout()
        
        filename = f'Mac{i}.png' # creating filename for each r value
        filenames.append(filename) # appending filename to the list of filenames 
        fig2.savefig(filename)
       # fig.close()
# build gif
with imageio.get_writer('MacCormack.gif', mode = 'I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
for filename in set(filenames):
    os.remove(filename)  
#%% Adopting Euler + MacCormack Technique with an added artifical viscosity 
U_SOL_mod = np.zeros((4, len(phi), len(new_r))) #preallocating data structure
k1 = vr[:, 55, 0, 0]*(1 + 0.25*(1-np.exp(-30/50))) #for all phi, at equatorial slice, and initial Ro
k2 = rho[:, 55, 0, 0]
k3 = Pr[:, 55, 0, 0]
k4 = vp[:, 55, 0, 0]*(1 + 0.25*(1-np.exp(-30/50)))
U_SOL_mod[:, :, 0] = np.array([k1,k2,k3,k4])  #initial condition (initial radial step)

filenames = [] # preallocating data structure 
for i in range(len(new_r)-1):
    U_SOL_mod[:,:, i + 1] = MacCormack_2D(q = U_SOL_mod[:,:,i], 
                                                dp = dp, 
                                                dr = dr, 
                                                r = new_r[i], 
                                                Cx = 0.0000000001) # works for 1e-9, 10 and smaller
    # run using 1e-7
    
    if i %25 == 0: #modulo operator, if the division between the incremented r and 25 is equal to 0, then
        #print(i) #prints radius in increments of 25, from 0 to 400
        #print((new_r[i]).to(u.AU)) #prints radius in AU
        #begin plotting
        fig3, ax = plt.subplots(nrows = 4, sharex = True, figsize = (5,10))
        #plot radial velocity vs phi
        pos = ax[0].plot(phi_deg, U_SOL_mod[0,:,i + 1]) #plot solution vector (flow parameter[0] = vr) against phi in degrees
        ax[0].set_ylabel(r'$\frac{km}{s}$')
        ax[0].set_title(r"$v_{r}$") #labeling radial velocity 
        #plot number density vs phi, convert to cm^-3 by dividing by mass_proton and converting from km^-3 to cm^-3
        pos = ax[1].plot(phi_deg,
                        ((U_SOL_mod[1, :, i + 1] / m_p.value) * (1 / u.km ** 3)).to(1 / u.cm ** 3))
        ax[1].set_ylabel(r'$\frac{1}{cm^3}$')
        ax[1].set_title(r"$n_{p}$")
        #plot pressure vs. phi, convert to dyne/cm^2 from kg/(kms^2) 
        pos = ax[2].plot(phi_deg,
                        (U_SOL_mod[2, :, i + 1] * (u.kg / (u.s ** 2 * u.km))).to(u.dyne / (u.cm ** 2)))
        ax[2].set_ylabel(r'$\frac{dyne}{cm^2}$')
        ax[2].set_title(r"$P$") 
        #plot longitudinal velocity v_phi vs. phi
        pos = ax[3].plot(phi_deg, 
                         U_SOL_mod[3, :, i + 1])
        ax[3].set_ylabel(r'$\frac{km}{s}$')
        ax[3].set_title(r"$v_{\phi}$")
        ax[3].set_xticks([0, 90, 180, 270, 360])
        ax[3].set_xlabel(" Carrington Longitude (Deg.)")
        fig3.suptitle("[Modified MacCormack] r = " + str(round(new_r[i + 1].to(u.AU).value, 3))) #plot for each r value in AU units
        plt.tight_layout()
        filename = f'ModMac{i}.png' # creating filename for each r value
        filenames.append(filename) # appending filename to the list of filenames 
        fig3.savefig(filename)
       # fig.close()
# build gif
with imageio.get_writer('Modified_MacCormack.gif', mode = 'I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
for filename in set(filenames):
    os.remove(filename)  