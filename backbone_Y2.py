#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 13:42:37 2021

@author: albertsmith
"""

import numpy as np
import os
import sys
sys.path.append('/Users/albertsmith/Documents/GitHub/')
sys.path.append('/Users/albertsmith/Documents/Dynamics/HETs_Methyl_Loquet/HETs_python_scripts')
import pyDIFRATE as DR
import matplotlib.pyplot as plt
import matplotlib
import cv2
from pyDIFRATE.chimera import vis4D


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **font)

#%% Load MD simulation
folder='/Users/albertsmith/Documents/Dynamics/GPCR_MD_data/GPCR_run1/'
dcd=folder+'reduced_1ns.dcd'
psf=folder+'step5_assembly.xplor.psf'

mol=DR.molecule(psf,dcd)
uni=mol.mda_object

bsheets=[np.arange(48,83),np.arange(84,111),np.arange(123,151),np.arange(164,182),
                  np.arange(216,247),np.arange(260,295),np.arange(301,329),np.arange(329,341)]
bsheets=[np.arange(48,83),np.arange(84,101),np.arange(101,111),np.arange(123,151),
             np.arange(164,182),np.arange(216,228),np.arange(228,247),
             np.arange(260,278),np.arange(278,295),np.arange(301,319),
             np.arange(319,329),np.arange(329,341)]

res0=np.unique(uni.atoms.select_atoms('not resname PRO and protein').resids)
bsheets=[bs[np.isin(bs,res0)] for bs in bsheets]
resids=np.concatenate(bsheets)

mol.select_atoms(Nuc='15N',resids=resids)

#%% Define the frames

frames=list()
frames.append({'Type':'peptide_plane','resids':resids,'sigma':2000,'full':True})
frames.append({'Type':'superimpose','filter_str':'name CA','resids':bsheets,'sigma':50,\
               'frame_index':np.concatenate([np.ones(bs.shape[0])*k for k,bs in enumerate(bsheets)])})

frames.append({'Type':'superimpose','resids':np.concatenate(bsheets),'filter_str':'name CA',\
                    'sigma':50,'frame_index':np.zeros(len(mol.sel1))})


#%% Analyze with just one frame
tf=20000

fr_obj=DR.frames.FrameObj(mol)
fr_obj.tensor_frame(sel1=1,sel2=2)
for f in frames:fr_obj.new_frame(**f)
fr_obj.load_frames(n=-1,dt=1000)




fr_obj.remove_post_process()
#fr_obj.post_process()
data=fr_obj.frames2data(mode='full')
r=data[0].detect
n=5
r.r_auto(n)


for d in data:d.detect=r
fit=[d.fit() for d in data]

fig=plt.figure('Compare detectors1')
fig.clear()
ax=fit[0].plot_rho(fig=fig)
for k,a in enumerate(ax):
    a.plot(fit[1].label,fit[1].R[:,k],color='black',linestyle=':')
fig.axes[0].set_xlim([-10,-4])
fig.set_size_inches([ 7.41, 12.16])

ax=plt.figure('Detector plots').add_subplot(111)
ax.cla()

directory='/Users/albertsmith/Documents/Dynamics/MF_MD_theory/Figures/backbone3D/'
scene=os.path.join(directory,'scene_Y2.cxs')
fileout=os.path.join(directory,'fr{0}_rho{1}.png')
nf=len(fit)-2
filenames=list()
chimera_cmds=['window 600 900']
for k in range(nf):
    for m in range(n-1):
        fit[k+2].draw_rho3D(m,scaling=1/fit[k+2].R[:,:-1].max(),scene=scene,\
           chimera_cmds=chimera_cmds,fileout=fileout.format(k,m))
        
for k in range(n-1):
    for m in range(nf):
        filenames.append(fileout.format(m,k))
      
 
#%% Make a few correlation function plots
out=fr_obj.frames2ct(mode='full')
resi=[82,228]
t=out['t']

for r in resi:
    b=np.argwhere(fit[0].label==r).squeeze()
    fig=plt.figure('resi{0}'.format(r))
    fig.clear()
    ax=[fig.add_subplot(2,3,k+1) for k in range(nf+1)]
    ax=[ax[0],ax[3],ax[1],ax[4],ax[2]]
    xlim=[20,20,1000,1000]
    for a,ct,xl in zip(ax,out['ct_finF'][:,b],xlim):
        a.plot(t,ct,color='black')
        a.set_xlim([0,xl])
        a.set_ylim([.8,1])
    ax[-1].semilogx(t[1:],out['ct'][b,1:],color='red')
    ax[-1].semilogx(t[1:],out['ct_prod'][b,1:],color='black',linestyle=':')
    ax[-1].set_xlim([0,1000])
    ax[-1].set_ylim([.65,1])
    fig.set_size_inches([150/25.4,100/25.4])
    fig.tight_layout()
        
    
    
    
    