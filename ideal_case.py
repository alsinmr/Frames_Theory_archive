#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 14:52:56 2020

@author: albertsmith
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDIFRATE import frames as ef  #Here the package for separating motion by frames
vft=ef.vft                          #Set of tools for rotations, angles, etc.


#%% Simulation Parameters
n=int(1e6)          #Number of time points in simulated trajectory
dt=5e-3             #Time step (ns)
t=np.arange(n)*dt   #Time axis (10 us)
t[0]=1e-3           #We plot on a log scale. We replace 0 with 1 ps to keep it on the scale (labeling later indicates this)

tau_f=0             #Fast motion (immediate decay)
tau_m=0.1           #Methyl rotation (hop time- not decay time constant of the correlation function)
tau_h=1             #Two-site hop time
dr=0.5              #Step size (in degrees) for rotation of the molecule in solution (determines correlation time)

theta_f=15          #Opening angle of fast motion (wobbling in a cone)
theta_m=30          #Symmetry axis for "methyl hop"
theta_h=45          #Angle of two-site hop



#%% Transistions
state_m=np.mod((np.random.binomial(1,dt/tau_m,n)*(1-2*np.random.binomial(1,0.5,n))).cumsum(),3).astype(int)
state_h=np.mod(np.random.binomial(1,dt/tau_h,n).cumsum(),2).astype(int)


br=np.random.rand(n)
beta_f0=np.arccos(1-br+br*np.cos(theta_f*np.pi/180))
gamma_f=np.random.rand(n)*2*np.pi
beta_f=beta_f0*(0.5+1.5*np.abs(gamma_f-np.pi)/np.pi)

gamma_r=np.random.rand(n)*2*np.pi



#%% Build up the trajectory
vf=np.array([np.sin(beta_f)*np.cos(gamma_f),np.sin(beta_f)*np.sin(gamma_f),np.cos(beta_f)])


vm0=[np.sin(theta_m*np.pi/180),0,np.cos(theta_m*np.pi/180)]
vm1=[np.sin(theta_m*np.pi/180)*np.cos(2*np.pi/3),np.sin(theta_m*np.pi/180)*np.sin(2*np.pi/3),np.cos(theta_m*np.pi/180)]
vm2=[np.sin(theta_m*np.pi/180)*np.cos(-2*np.pi/3),np.sin(theta_m*np.pi/180)*np.sin(-2*np.pi/3),np.cos(theta_m*np.pi/180)]

vm=np.array([vm0,vm1,vm2]).T[:,state_m]
vmXZ=np.array([vm1,vm2,vm0]).T[:,state_m]

vh0=[0,0,1]
vh1=[np.sin(theta_h*np.pi/180),0,np.cos(theta_h*np.pi/180)]
vh=np.array([vh0,vh1]).T[:,state_h]

"Here the rotation in solution....slow, don't re-run!"
vr=[np.array([0,0,1])]
for k in range(n-1):
    sc=vft.getFrame(vr[-1])
#    sc[0],sc[1]=sc[4],-sc[5]
    vr0=[np.cos(gamma_r[k])*np.sin(dr*np.pi/180),np.sin(gamma_r[k])*np.sin(dr*np.pi/180),np.cos(dr*np.pi/180)]
    vr.append(vft.R(vr0,*sc))
vr=np.array(vr).T

"Here the full set of motions"
scf=vft.getFrame(vf)
scm=vft.getFrame(vm)
sch=vft.getFrame(vh)
scr=vft.getFrame(vr)

v=vft.R(vft.R(vft.R(vf,*scm),*sch),*scr)
vXZ=vft.R(vft.R(vmXZ,*sch),*scr)


#%% Get each vector in the lab frame
"Frame in lab frame"
vf_LF=vft.R(vft.R(vft.R(vf,*scm),*sch),*scr)
vm_LF=vft.R(vft.R(vm,*sch),*scr)
vh_LF=vft.R(vh,*scr)
vr_LF=vr



#%% Try to reconstruct correlation function assuming timescale separation

"Total correlation function"
Ct=ef.Ct_D2inf(v,cmpt='00',mode='ct')

"Construction correlation function from product of individual motions"

"""
Inner motion (librations).
Af is the residual tensor of fast motion, and Ct0pf
contains the correlation functions describing the evolution from t=0 to the
equilibrium tensor. Af is used in the subsequent step for calculating the 
correlation function for methyl rotation motion

Ctp0f contains components of the correlation functions
"""
Ct0pf,Af=ef.Ct_D2inf(v,vXZ,nuZ_F=vm_LF,cmpt='0p')
Ctp0f=ef.Ct_D2inf(v,vXZ,nuZ_F=vm_LF,cmpt='p0',mode='ct')
Ct_f=Ct0pf[2].real

Ct0pm,Am=ef.Ct_D2inf(v,vXZ,nuZ_F=vh_LF,cmpt='0p')
Ctp0m=ef.Ct_D2inf(v,vXZ,nuZ_F=vh_LF,nuZ_f=vm_LF,cmpt='p0',mode='ct')
Ct_m=np.sum(Af*Ctp0m.T,1).T.real/Af[2].real

Ct0ph,Ah=ef.Ct_D2inf(v,vXZ,nuZ_F=vr_LF,cmpt='0p')
Ctp0h=ef.Ct_D2inf(v,vXZ,nuZ_F=vr_LF,nuZ_f=vh_LF,cmpt='p0',mode='ct')
Ct_h=np.sum(Am*Ctp0h.T,1).T.real/Am[2].real

Ct0pr,Ar=ef.Ct_D2inf(v,vXZ,cmpt='0p')
Ctp0r=ef.Ct_D2inf(v,vXZ,nuZ_f=vr_LF,cmpt='p0',mode='ct')
Ct_r=np.sum(Ah*Ctp0r.T,1).T.real/Ah[2].real


Ct_p=Ct_f*Ct_m*Ct_h*Ct_r


Ct0p=[Ct0pf,Ct0pm,Ct0ph,Ct0pr]      #Correlation functions for the equilibration of residual tensors (right plots)
Ctp0=[Ctp0f,Ctp0m,Ctp0h,Ctp0r]      #Correlation functions to constructtotal correlation function of given motion
A0p=[np.array([0,0,1,0,0]),Af,Am,Ah,Ar]
Ct_x=[Ct_f,Ct_m,Ct_h,Ct_r]

#%% Plot all correlation functions
"Create axes, store titles, text, legends for later"
fig=plt.figure('Ideal Frame Separation')
fig.clear()
n2=int(n/2)

ax=[fig.add_subplot(5,2,k) for k in [1,2,3,6,5,8,7,10,9]]

titles=['Total Correlation Function',
        'Components',
        'C(t) for f in F',
        'C(t) for tensor PAS in f',
        '','','','','']
legends=[[r'$C(t)$',r'$C_\mathrm{prod}(t)$'],
         [r'$C_\mathrm{f}(t)$',r'$C_\mathrm{hop}(t)$',r'$C_\mathrm{met}(t)$',r'$C_\mathrm{f}(t)$'],
         [r'$C_{00}(t)$',r'Re[$C_{10}(t)$]',r'Im[$C_{10}(t)$]',r'Re[$C_{20}(t)$]',r'Im[$C_{20}(t)$]'],
         [r'$C_{00}(t)$',r'Re[$C_{01}(t)$]',r'Im[$C_{01}(t)$]',r'Re[$C_{02}(t)$]',r'Im[$C_{02}(t)$]'],
         '','','','','']
text=['','',
      'PAS in ME\n(fast wobbling)',
      'PAS in ME',
      'ME in CC\n(rotation)',
      'PAS in CC',
      'CC in MO\n(hopping)',
      'PAS in MO',
      'MO in LF\n(tumbling)']


"Plot total correlation function"
ax[0].semilogx(t[:n2],Ct[:n2],color='red')
ax[0].semilogx(t[:n2],Ct_p[:n2],color='black',linestyle=':')

"Plot components of correlation function"
for ct in Ct_x:ax[1].semilogx(t[:n2],ct[:n2])
    
"Plot components of correlation function for each motion"
for k,(ctx,ct0p,ctp0,a0p) in enumerate(zip(Ct_x,Ct0p,Ctp0,A0p)):
    "Ctp0 and the scaled Ctp0"
#    for m,(a0,ct) in enumerate(zip(a0p[2:],ctp0[2:])):
#        ax[2*k+2].semilogx(t[:n2],(a0/a0p[2].real*ct[:n2]).real,color='grey')
#        if m!=0:
#            ax[2*k+2].semilogx(t[:n2],(a0/a0p[2]*ct[:n2]).imag,color='grey',linestyle=':')
    for m,ct in enumerate(ctp0[2:]):
        ax[2*k+2].semilogx(t[:n2],ct[:n2].real)
        if m!=0:
            ax[2*k+2].semilogx(t[:n2],ct[:n2].imag,linestyle=':')
    ax[2*k+2].semilogx(t[:n2],ctx[:n2],color='black')        
    
    "Ct0p"
    if k<3:
        for m,ct in enumerate(ct0p[2:]):
            ax[2*k+3].semilogx(t[:n2],ct[:n2].real)
            if m!=0:
                ax[2*k+3].semilogx(t[:n2],ct[:n2].imag,linestyle=':')
                    
"Legends, titles, texts, axes"
for a,ttl,l,txt in zip(ax,titles,legends,text):
    a.set_title(ttl)
    
    if len(l):a.legend(l,loc='upper center')
    a.text(50,0.75,txt)
    
    a.set_ylim([-.05,1.05])
    a.set_yticks(np.linspace(0,1,5))
    a.set_ylabel('C(t)')
    a.set_yticklabels([0,None,.5,None,1])

    a.set_xticks(np.logspace(-3,3,7))
    a.set_xlim([1e-3,t[n2]])
    if a.is_last_row():
        a.set_xlabel('t')
        a.set_xticklabels(['0 s','','100 ps','','10 ns','',r'1 $\mu$s'])
    else:
        a.set_xticklabels([])

fig.show()
