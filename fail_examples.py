#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
5) No correlation with adiabaticity
""" 

#%% Some useful functions

def calc_ct(v,vXZ,vF,vFXZ=None):
    Ct=ef.Ct_D2inf(v,cmpt='00',mode='ct')
    
    Ct0pf,Af=ef.Ct_D2inf(v,vXZ,nuZ_F=vF,nuXZ_F=vFXZ,cmpt='0p')
    Ctf=Ct0pf[2].real
    
    Ctp0F=ef.Ct_D2inf(v,vXZ,nuZ_f=vF,nuXZ_f=vFXZ,cmpt='p0',mode='ct')
    CtF=np.sum(Af*Ctp0F.T,1).T.real/Af[2].real
    
    Ctp=Ctf*CtF
    return Ct,Ctp

xl=DR.tools.nice_str('{:q1}')
xl.unit='s'
matplotlib.rcParams['font.size']=8

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
    


def sym_hop(tau,theta=np.pi/6,n=1e5,dt=.005,nsites=3):
    n=int(n)
    p=dt/tau
    state=np.mod((binomial(1,p,n)*(1-2*binomial(1,.5,n))).cumsum(),3).astype(int)
    v0=np.array([[np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)] \
         for phi in np.linspace(0,2*np.pi,nsites,endpoint=False)])
    v=v0.T[:,state]
    vXZ=v0.T[:,state]
    return v,vXZ

def three_site_asym(tau,theta=np.pi/6,n=1e5,dt=.005):
    n=int(n)
    p=dt/tau
    state=np.mod((binomial(1,p,n)*(1-2*binomial(1,.5,n))).cumsum(),3).astype(int)
    v0=np.array([[np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)] \
         for phi in [-np.pi/2,0,np.pi/2]])
    v=v0.T[:,state]
    vXZ=v0.T[:,state]
    return v,vXZ

def two_site_hop(tau,theta=np.pi/4,phi=0,n=1e5,dt=.005):
    n=int(n)
    p=dt/tau
    state=np.mod(binomial(1,p,n).cumsum(),2).astype(int)
    v0=np.array([[0,0,1],[np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)]])
    v=v0.T[:,state]
    return v,np.repeat(np.atleast_2d([np.cos(phi),np.sin(phi),0]).T,n,axis=1)

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

def restrict_diff(step=1*np.pi/180,theta=np.pi/4,n=1e5,dt=.005,phi=0):
    n=int(n)
    rb=(1-2*binomial(1,.5,n)).cumsum()*step+theta/2
    i=np.logical_or(rb<0,rb>theta)
    counter=0
    while np.any(i) and counter<n:
        counter+=1
        i0=np.argwhere(i)[0,0]
        if rb[i0]<0:
            rb[i0:]*=-1
        else:
            rb[i0:]*=-1
            rb[i0:]+=2*theta
        i=np.logical_or(rb<0,rb>theta)
    return np.array([np.sin(rb)*np.cos(phi),np.sin(rb)*np.sin(phi),np.cos(rb)]),\
        np.repeat(np.atleast_2d([np.cos(phi),np.sin(phi),0]).T,n,axis=1)

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

"Example 5: No correlation due to adiabaticity"
#v1,v1XZ=two_site_hop(tau=tau_m)
#v2,v2XZ=restrict_diff(theta=np.pi/2)
#v,vXZ=v_overall(v1,v1XZ,v2,v2XZ)
#plot_ct(ax[8],v,vXZ,v2,v2XZ)
#
#v1,v1XZ=two_site_hop(tau=tau_m,theta=np.pi/4,phi=np.pi/2)
#v2,v2XZ=restrict_diff(phi=np.pi/2,theta=np.pi/2)
#v,vXZ=v_overall(v1,v1XZ,v2,v2XZ)
#plot_ct(ax[9],v,vXZ,v2,v2XZ)

fig.tight_layout()