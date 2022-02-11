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
    return Ct.real,Ctp

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
full_corr generates a trajectory based on a pair of two-site hops, where the
hops are either independent (set corr=False) or where the hops always occur
simultaneously. theta determines the size of the hop. mode can be set to
parallel (para), inverse parallel (ipara) for which the two hops cancel, or
perpendicular (perp) for which the hops occur around perpendicular axes
"""
def full_corr(tau,theta=np.pi/4,mode='para',corr=True,n=1e5,dt=.005):
    v01Z=np.array([[0,0,1],[np.sin(theta),0,np.cos(theta)]])
    v01XZ=np.array([[1,0,0],[np.cos(theta),0,-np.sin(theta)]])
    if mode=='para':
        v02Z,v02XZ=v01Z,v01XZ
    if mode=='ipara':
        v02Z,v02XZ=v01Z[::-1],v01XZ[::-1]
    if mode=='perp':
        v02Z=np.array([[1,0,0],[np.cos(theta),np.sin(theta),0]])
        v02XZ=np.array([[0,0,1],[0,0,1]])
    
    n=int(n)
    p=dt/tau
    state=np.mod(binomial(1,p,n).cumsum(),2).astype(int)
    state1=state if corr else np.mod(binomial(1,p,n).cumsum(),2).astype(int)
        
    return v01Z[state].T,v01XZ[state].T,v02Z[state1].T,v02XZ[state1].T

"""
chi12_corr generates a trajectory based on a pair of two-site hops, where the
two-site hops are for a tetrahedral geometry, such as one might expect for
a saturated carbon chain. Four relative directions of the two hops are available,
achieved by setting switch from 0-3. The two hops can be set to be independent
(corr=False) or always occur simultaneously (corr=True)
"""
def chi12_corr(tau,mode='counter',switch=0,corr=True,n=1e5,dt=.005):
    ta=np.arccos(-1/3) #Tetrahedral angle
    v02Z=np.array([[np.sin(ta),0,-np.cos(ta)],
                    [np.sin(ta)*np.cos(2/3*np.pi),np.sin(ta)*np.sin(2/3*np.pi),-np.cos(ta)]])
    v02XZ=np.array([[0,0,-1],[0,0,-1]])



    v01XZ=np.array([[0,0,-1],[0,0,-1]])
    
    if switch==0:
        v01Z=np.array([[np.sin(ta)*np.cos(1/3*np.pi),np.sin(ta)*np.sin(1/3*np.pi),-np.cos(ta)],
                    [np.sin(ta)*np.cos(1/3*np.pi),-np.sin(ta)*np.sin(1/3*np.pi),-np.cos(ta)]])
    elif switch==1:
        v01Z=np.array([[np.sin(ta)*np.cos(1/3*np.pi),-np.sin(ta)*np.sin(1/3*np.pi),-np.cos(ta)],
                    [np.sin(ta)*np.cos(1/3*np.pi),np.sin(ta)*np.sin(1/3*np.pi),-np.cos(ta)]])
    elif switch==2:
        v01Z=np.array([[np.sin(ta)*np.cos(1/3*np.pi),np.sin(ta)*np.sin(1/3*np.pi),-np.cos(ta)],
                    [-np.sin(ta),0,-np.cos(ta)]])
    elif switch==3:
        v01Z=np.array([[-np.sin(ta),0,-np.cos(ta)],
                        [np.sin(ta)*np.cos(1/3*np.pi),np.sin(ta)*np.sin(1/3*np.pi),-np.cos(ta)]])

    
    n=int(n)
    p=dt/tau
    state=np.mod(binomial(1,p,n).cumsum(),2).astype(int)
    state1=state if corr else np.mod(binomial(1,p,n).cumsum(),2).astype(int)

    return v01Z[state].T,v01XZ[state].T,v02Z[state1].T,v02XZ[state1].T


"""
Takes two trajectories resulting from individual motions and generates the motion
resulting from both motions acting together on an NMR interaction
"""
def v_overall(v1,vXZ1,v2,v2XZ=None):
    sc2=vft.getFrame(v2,v2XZ)
    return vft.R(v1,*sc2),vft.R(vXZ1,*sc2)



#%% Set up plots
fig=plt.figure('Full Correlation')
fig.clear()
theta=np.pi/4
ax=[fig.add_subplot(2,3,k+1) for k in [0,3,1,4,2,5]]
fig.set_size_inches([180/25.4,80/25.4])


#%% Sweep through three cases of correlated motion (parallel, inverse parallel, perpendicular)
theta=np.pi/4

"Example 6: Simultaneous motions (parallel)"
v1,v1XZ,v2,v2XZ=full_corr(tau=.5,theta=theta,mode='para',corr=False)
v,vXZ=v_overall(v1,v1XZ,v2,v2XZ)
plot_ct(ax[0],v,vXZ,v2,v2XZ)

v1,v1XZ,v2,v2XZ=full_corr(tau=.5,theta=theta,mode='para',corr=True)
v,vXZ=v_overall(v1,v1XZ,v2,v2XZ)
plot_ct(ax[1],v,vXZ,v2,v2XZ)

"Example 7: Simultaneous motions (inverse parallel)"
v1,v1XZ,v2,v2XZ=full_corr(tau=.5,theta=theta,mode='ipara',corr=False)
v,vXZ=v_overall(v1,v1XZ,v2,v2XZ)
plot_ct(ax[2],v,vXZ,v2,v2XZ)

v1,v1XZ,v2,v2XZ=full_corr(tau=.5,theta=theta,mode='ipara',corr=True)
v,vXZ=v_overall(v1,v1XZ,v2,v2XZ)
plot_ct(ax[3],v,vXZ,v2,v2XZ)

"Example 7: Simultaneous motions (perpendicular)"
v1,v1XZ,v2,v2XZ=full_corr(tau=.5,theta=theta,mode='perp',corr=False)
v,vXZ=v_overall(v1,v1XZ,v2,v2XZ)
plot_ct(ax[4],v,vXZ,v2,v2XZ)

v1,v1XZ,v2,v2XZ=full_corr(tau=.5,theta=theta,mode='perp',corr=True) 
v,vXZ=v_overall(v1,v1XZ,v2,v2XZ)
plot_ct(ax[5],v,vXZ,v2,v2XZ)

fig.tight_layout()
for a in ax:a.set_xlim([.001,250])

#%% Sweep through four cases of tetrahedral hops
fig=plt.figure('Tetrahedral')
fig.clear()
theta=np.pi/4
ax=[fig.add_subplot(2,4,k+1) for k in [0,4,1,5,2,6,3,7]]
fig.set_size_inches([180/25.4,80/25.4])

for a,sw,corr in zip(ax,[0,0,1,1,2,2,3,3],[False,True,False,True,False,True,False,True]):
    v1,v1XZ,v2,v2XZ=chi12_corr(tau=.5,switch=sw,corr=corr)
    v,vXZ=v_overall(v1,v1XZ,v2,v2XZ)
    plot_ct(a,v,vXZ,v2,v2XZ)


fig.tight_layout()
for a in ax:a.set_xlim([.001,250])

plt.show()