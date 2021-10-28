#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 13:42:37 2021

@author: albertsmith
"""

import numpy as np
import pyDIFRATE as DR
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import least_squares

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **font)


"This loads the MD trajectory into pyDIFRATE"
mol=DR.molecule('HETs_ILE254.pdb','HETs_ILE254.xtc')


#%% Define the frames

"We store the frames in a list, and then later load them into the frame analysis"
frames=list()
"""
'Type': Name of the function used for generating the frame 
(see pyDIFRATE/Struct/frames.py, pyDIFRATE/Struct/special_frames.py)
'Nuc': Shortcut for selecting a particular pair of nuclei. 
ivlal selects one group on every Ile, Val, Leu, Ala
'resids': List of residues to include (not required here – only one residue in trajectory)
'segids': List of segments to include (not required here – only one segment in trajectory)
'sigma': Time constant for post-processing (averaging of frame direction with Gaussian)
'n_bonds': Number of bonds away from methyl C–C group to define frame
"""

"Frames without post-process smoothing"
frames.append({'Type':'methylCC','Nuc':'ivlal','sigma':0})
frames.append({'Type':'side_chain_chi','Nuc':'ivlal','n_bonds':1,'sigma':0})
frames.append({'Type':'side_chain_chi','Nuc':'ivlal','n_bonds':2,'sigma':0})

"Frames with post-process smoothing"
frames.append({'Type':'hops_3site','Nuc':'ivlal','sigma':5})
frames.append({'Type':'methylCC','Nuc':'ivlal','sigma':5})
frames.append({'Type':'chi_hop','Nuc':'ivlal','n_bonds':1,'sigma':50})
frames.append({'Type':'side_chain_chi','Nuc':'ivlal','n_bonds':1,'sigma':50})
frames.append({'Type':'chi_hop','Nuc':'ivlal','n_bonds':2,'sigma':50})
frames.append({'Type':'side_chain_chi','Nuc':'ivlal','n_bonds':2,'sigma':50})


#%% Create a figure for plotting
titles=[['Methyl rot.',r'$\chi_2$ rot.',r'$\chi_1$ rot.',r'C$\alpha$-C$\beta$ motion','Total'],
        ['Methyl rot.',r'$\chi_2$ rot.',r'$\chi_1$ rot.',r'C$\alpha$-C$\beta$ motion','Total'],
        ['Methyl lib.','Methyl hop.',r'$\chi_2$ lib.',r'$\chi_2$ hop.',
          r'$\chi_1$ lib.',r'$\chi_1$ hop.',r'C$\alpha$-C$\beta$ motion','Total']]
fig=plt.figure('Methyl Dynamics')
fig.clear()
fig.set_size_inches([10.27,  9.03])
ax=[]
ax.append([fig.add_subplot(5,3,k+1) for k in range(0,15,3)])
ax.append([fig.add_subplot(5,3,k+1) for k in range(1,15,3)])
ax.append([fig.add_subplot(5,6,k+1) for k in [4,5,10,11,16,17,22]])
ax[-1].append(fig.add_subplot(5,3,15))

for a0,t0 in zip(ax,titles):
    for a,t in zip(a0,t0):
        a.set_title(t)
        

#%% Analyze with just one frame
tf=20000
mol.select_atoms(Nuc='ivlal')
fr_obj=DR.frames.FrameObj(mol)
fr_obj.tensor_frame(sel1=1,sel2=2)
for f in frames:fr_obj.new_frame(**f)
fr_obj.load_frames(tf=tf,n=-1)
fr_obj.post_process()

include=np.zeros([3,9],dtype=bool)
include[0][:3]=True    #Only methylCC,side_chain_chi frames without post processing
include[1][[4,6,8]]=True  #Only methylCC,side_chain_chi frames with post processing
include[2][3:]=True #All frames with post processing

t=np.arange(int(tf/2))*.005     #Only plot the first half of the correlation function, where noise is lower

for inc,ax0 in zip(include,ax):
    out=fr_obj.frames2ct(include=inc,mode='full')
    
    for ct,a in zip(out['ct_finF'],ax0):
        a.cla()
        a.plot(t,ct.mean(0)[:int(tf/2)])
        a.set_ylim([0,1.05])
        S2=ct.mean(0)[int(tf/4):int(tf/2)].mean()
        b=np.argwhere(ct.mean(0)-S2<0)[0,0]
#        tc0=np.max([.001,((ct.mean(0)[:b]-S2)/(1-S2)).sum()*.005])
        tc0=t[np.argmin(np.abs((ct.mean(0)[:b]-S2)/(1-S2)-np.exp(-1)))]
        fun=lambda x:(((x[0]+(1-x[0])*np.exp(-t[:b]/x[1]))-ct.mean(0)[:b])**2).sum()
        S2,tc=least_squares(fun,[S2,tc0]).x
        a.plot(t,S2+(1-S2)*np.exp(-t/tc),color='grey',linestyle=':')
        a.set_xlim([0,np.min([10*tc,fr_obj.t[int(tf/2)]])])
    ax0[-1].semilogx(out['t'][:int(tf/2)],out['ct'].mean(0)[:int(tf/2)])
    ax0[-1].semilogx(out['t'][:int(tf/2)],out['ct_prod'].mean(0)[:int(tf/2)])
    ax0[-1].set_ylim([0,.5])
    
fig.set_size_inches([180/25.4,220/25.4])

#"Some plotting of the correlation functions"
#fig=plt.figure('Correlation functions')
#fig.clear()
#n=(fr_obj.Ct['ct_finF'].shape[0]+1)
#ax=[fig.add_subplot(int(n/2),2,k+1) for k in range(n)]
#xl=[2,100,100,500,500]
#xl=[.1,.5,5,100,5,100,500]
##xl=[2]
#t=fr_obj.t[:int(tf/2)]
#for ct,a in zip(out['ct_finF'],ax):
#    a.cla()
#    a.plot(t,ct.mean(0)[:int(tf/2)])
#    a.set_ylim([0,1.05])
#    S2=ct.mean(0)[int(tf/4):int(tf/2)].mean()
##    tc0=t[np.argmin(np.abs((ct.mean(0)[:int(tf/2)]-S2)/(1-S2)-np.exp(-1)))]
#    b=np.argwhere(ct.mean(0)-S2<0)[0,0]
#    tc0=np.max([.001,((ct.mean(0)[:b]-S2)/(1-S2)).sum()*.005])
#    fun=lambda x:(((x[0]+(1-x[0])*np.exp(-t[:b]/x[1]))-ct.mean(0)[:b])**2).sum()
#    S2,tc=least_squares(fun,[S2,tc0]).x
#    a.plot(t,S2+(1-S2)*np.exp(-t/tc),color='grey',linestyle=':')
#    a.set_xlim([0,np.min([10*tc,fr_obj.t[int(tf/2)]])])
#ax[-1].semilogx(out['t'][:int(tf/2)],out['ct'].mean(0)[:int(tf/2)])
#ax[-1].semilogx(out['t'][:int(tf/2)],out['ct_prod'].mean(0)[:int(tf/2)])
##fig.tight_layout()


