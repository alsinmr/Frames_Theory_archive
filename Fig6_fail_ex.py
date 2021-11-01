#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2021 Albert Smith-Penzel

This file is part of Frames Theory Archive (FTA).

FTA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FTA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FTA.  If not, see <https://www.gnu.org/licenses/>.


Questions, contact me at:
albert.smith-penzel@medizin.uni-leipzig.de

Created on Thu Sep 24 14:52:56 2020

@author: albertsmith
"""

import numpy as np
from numpy.random import binomial
import matplotlib.pyplot as plt
import matplotlib
import pyDIFRATE as DR
ef=DR.frames
vft=ef.vft


#%% List of examples
"""
1) Correlation due to poorly defined frame
2) With and without timescale separation
3) Uncorrelated and correlated
4) Correlation due to geometry
""" 

#%% Some useful functions

"""
calc_ct calculates the directly calculated correlation function and the product
of correlation functions
"""
def calc_ct(v,vXZ,vF,vFXZ=None):
    Ct=ef.Ct_D2inf(v,cmpt='00',mode='ct')
    
    Ct0pf,Af=ef.Ct_D2inf(v,vXZ,nuZ_F=vF,nuXZ_F=vFXZ,cmpt='0p')
    Ctf=Ct0pf[2].real
    
    Ctp0F=ef.Ct_D2inf(v,vXZ,nuZ_f=vF,nuXZ_f=vFXZ,cmpt='p0',mode='ct')
    CtF=np.sum(Af*Ctp0F.T,1).T.real/Af[2].real
    
    Ctp=Ctf*CtF
    return Ct,Ctp

#xl is sort of like a formatting string, inserting prefixes and units
xl=DR.tools.nice_str('{:q1}')
xl.unit='s'

#Set the default font size in matplotlib
matplotlib.rcParams['font.size']=8

"""
plot_ct plots the directly calculated correlation function and the product of
two correlation functions separated by a frame into a given axis
"""
def plot_ct(ax,v,vXZ,vF,vFXZ=None,dt=.005):
    t=np.arange(v.shape[1])*dt
    Ct,Ctp=calc_ct(v,vXZ,vF,vFXZ)
    ax.cla()
    ax.semilogx(t[1:],Ct[1:],color='red')
    ax.semilogx(t[1:],Ctp[1:],color='black',linestyle=':')
    ax.set_ylim([0,1])
    ax.set_xticks(np.logspace(-2,2,5))
    ax.set_xlim([t[1],250])
    s=ax.get_subplotspec()
    if s.is_first_col():
        ax.set_ylabel(r'$C(t)$')
    else:
        ax.set_yticklabels([])
    if s.is_last_row():
        ax.set_xlabel(r'$t$/ns')
        ax.set_xticklabels([xl.format(t*1e-9) if np.mod(k,2)==0 else '' for k,t in enumerate(ax.get_xticks())])
#        ax.set_xticklabels([xl.format(t*1e-9) for k,t in enumerate(ax.get_xticks())])
    else:
        ax.set_xticklabels([])
    if s.is_first_col() and s.is_first_row():
        ax.legend(['direct','product'],loc='upper right')
    

"""
sym_hop generates a trajectory with hopping between nsites (default 3), where
all sites are populated equally
"""
def sym_hop(tau,theta=np.pi/6,n=1e5,dt=.005,nsites=3):
    n=int(n)
    p=dt/tau
    state=np.mod((binomial(1,p,n)*(1-2*binomial(1,.5,n))).cumsum(),3).astype(int)
    v0=np.array([[np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)] \
         for phi in np.linspace(0,2*np.pi,nsites,endpoint=False)])
    v=v0.T[:,state]
    vXZ=v0.T[:,state]
    return v,vXZ

"""
two_site_hop generates a trajectory with hops  between two sites
"""
def two_site_hop(tau,theta=np.pi/4,phi=0,n=1e5,dt=.005):
    n=int(n)
    p=dt/tau
    state=np.mod(binomial(1,p,n).cumsum(),2).astype(int)
    v0=np.array([[0,0,1],[np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]])
    v=v0.T[:,state]
    return v,np.repeat(np.atleast_2d([np.cos(phi),np.sin(phi),0]).T,n,axis=1)


"""
corr_met_2site generates two trajectories resulting from both a three-site hopping
and a two-site hopping, where the two-site hop only occurs for a particular
orientaion of the three site hoping (correlated motion)
"""
def corr_met_2site(tau_m,tau_h,theta_m=np.pi/6,theta_h=np.pi/4,n=1e5,dt=.005):
    n=int(n)
    p=dt/tau_m
    state=np.mod((binomial(1,p,n)*(1-2*binomial(1,.5,n))).cumsum(),3).astype(int)
    v0=np.array([[np.sin(theta_m)*np.cos(phi),np.sin(theta_m)*np.sin(phi),np.cos(theta_m)] \
         for phi in np.linspace(0,2*np.pi,3,endpoint=False)])
    vm=v0.T[:,state]
    vXZm=v0.T[:,state]
    
    p=3*dt/tau_h
    hop=binomial(1,p,n)
    hop[state!=0]=0
    state=np.mod(hop.cumsum(),2).astype(int)
    phi=0
    v0=np.array([[0,0,1],[np.sin(theta_h)*np.cos(phi),np.sin(theta_h)*np.sin(phi),np.cos(theta_h)]])
    vh=v0.T[:,state]

    return vm,vXZm,vh,np.repeat(np.atleast_2d([np.cos(phi),np.sin(phi),0]).T,n,axis=1)

"""
Takes two trajectories resulting from individual motions and generates the motion
resulting from both motions acting together on an NMR interaction
"""
def v_overall(v1,vXZ1,v2,v2XZ=None):
    sc2=vft.getFrame(v2,v2XZ)
    return vft.R(v1,*sc2),vft.R(vXZ1,*sc2)



#%% Set up plots
fig=plt.figure('Correlation Functions')
fig.clear()
ax=[fig.add_subplot(2,4,k+1) for k in [0,4,1,5,2,6,3,7]]
fig.set_size_inches([180/25.4,100/25.4])



#%% Some default values
tau_m=.05
tau_h=5
dt=.005
n=4e5
    
"Example 1: Badly defined frame (frame uncorrelated with motion)"
vm,vmXZ=sym_hop(tau=tau_m,n=n)
vh,vhXZ=two_site_hop(tau=tau_h,n=n)

v,vXZ=v_overall(vm,vmXZ,vh,vhXZ)
plot_ct(ax[0],v,vXZ,vh,vhXZ)

vh,vhXZ=two_site_hop(tau=tau_h,n=n)  #Here we generate an uncorrelated motion
plot_ct(ax[1],v,vXZ,vh,vhXZ)

"Example 2: With and without timescale separation"
vm,vmXZ=sym_hop(tau=tau_m,n=n)
vh,vhXZ=two_site_hop(tau=tau_h,n=n)

v,vXZ=v_overall(vm,vmXZ,vh,vhXZ)
plot_ct(ax[2],v,vXZ,vh,vhXZ)

vm,vmXZ=sym_hop(tau=5,n=n)
vh,vhXZ=two_site_hop(tau=.05,n=n)

v,vXZ=v_overall(vm,vmXZ,vh,vhXZ)
plot_ct(ax[3],v,vXZ,vh,vhXZ)

"Example 3: Uncorrelated vs. correlated"
vm,vmXZ=sym_hop(tau=tau_m,n=n)
vh,vhXZ=two_site_hop(tau=tau_h,n=n)

v,vXZ=v_overall(vm,vmXZ,vh,vhXZ)
plot_ct(ax[4],v,vXZ,vh,vhXZ)

vm,vmXZ,vh,vhXZ=corr_met_2site(tau_m=tau_m,tau_h=tau_h,n=n)

v,vXZ=v_overall(vm,vmXZ,vh,vhXZ)
plot_ct(ax[5],v,vXZ,vh,vhXZ)

"Example 4: Correlation due to geometry"
v1,v1XZ=two_site_hop(tau=tau_m,n=n)
v2,v2XZ=two_site_hop(tau=tau_h,n=n)
v,vXZ=v_overall(v1,v1XZ,v2,v2XZ)
plot_ct(ax[6],v,vXZ,v2,v2XZ)

v1,v1XZ=two_site_hop(tau=tau_m,phi=np.pi/2,n=n)
v2,v2XZ=two_site_hop(tau=tau_h,n=n)
v,vXZ=v_overall(v1,v1XZ,v2,v2XZ)
plot_ct(ax[7],v,vXZ,v2,v2XZ)

fig.tight_layout()

plt.show()