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

Created on Tue Jan 1 13:42:37 2022

@author: albertsmith
"""

import numpy as np
import pyDIFRATE as DR
import matplotlib.pyplot as plt
import matplotlib

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


"Frames with post-process smoothing"
frames.append({'Type':'hops_3site','Nuc':'ivlal','sigma':5})
frames.append({'Type':'methylCC','Nuc':'ivlal','sigma':5})
frames.append({'Type':'chi_hop','Nuc':'ivlal','n_bonds':1,'sigma':50})
frames.append({'Type':'side_chain_chi','Nuc':'ivlal','n_bonds':1,'sigma':50})
frames.append({'Type':'chi_hop','Nuc':'ivlal','n_bonds':2,'sigma':50})
frames.append({'Type':'side_chain_chi','Nuc':'ivlal','n_bonds':2,'sigma':50})
        
#%% Analyze with just one frame
tf=200000
mol.select_atoms(Nuc='ivlal')   #Select 1 methyl group from all isoleucine, valine, leucine, and alanine residues
"""We could also specify the residue, but the provided trajectory just has one residue
A selection command populates mol.sel1 and mol.sel2 with atom groups, where sel1 and sel2 
then define a list of bonds
"""
fr_obj=DR.frames.FrameObj(mol)  #This creates a frame object based on the above molecule object
fr_obj.tensor_frame(sel1=1,sel2=2) #Here we specify to use the same bonds that were selected above in mol.select_atoms
#1,2 means we use for the first atom selection 1 in mol and for the second atom the second selection in mol

for f in frames:fr_obj.new_frame(**f) #This defines all frames in the frame object 
#(arguments were previously stored in the frames list)
fr_obj.load_frames(tf=tf,n=-1)  #This sweeps through the trajectory and collects the directions of all frames at each time point
fr_obj.post_process()   #This applies post processing to the frames where it is defined (i.e. sigma!=0)

"""
For each calculation, we only include some of the 9 frames that were defined above.

1) 3 rotational frames without post procesing (frames 0-2)
2) 3 rotational frames with averaging (frames 4,6,8)
3) 6 frames (rotational+hopping frames, frames 3-8)
"""

t=np.arange(int(tf/2))*.005     #Only plot the first half of the correlation function, where noise is lower

direct,prod,*frames=fr_obj.frames2data(mode='full')



#%% Sweep over various combinations of motions to determine the relaxation and detector behavior
"""
1) Methyl librations only
2) Methyl librations+methyl hopping
3) Methyl libration+methyl hopping+chi2 libration
4) Methyl librations+methyl hopping+chi2 libration+chi2 hopping
5) Methyl librations+methyl hopping+chi2 libration+chi2 hopping+chi1 libration
6) Methyl librations+methyl hopping+chi2 libration+chi2 hopping+chi1 libration+chi1 hopping
7) All motion
"""
    
index=[np.arange(k) for k in range(1,8)] #Index to determine which frames to include

fit0=list()
  
#We edit the standard deviation that is automatically inserted for correlation functions
#This definition assumes the dominant source of error is undersampling of frequent events
frames[0].R_std[:]=np.sqrt(frames[0].sens.info.loc['t'].to_numpy()/frames[0].sens.info[tf-1]['t'])
frames[0].R_std[:,0]=1e-6
frames[0].sens.info.loc['stdev']=frames[0].R_std[0] 
frames[0].new_detect()
r0=frames[0].detect
r0.r_no_opt(20)     #This sets the detector sensitivities (this setting is a temporary processing)
for i in index:     #Loop over all combinations of frames (listed above in the comments)
    data=frames[0].copy()
    data.R_std[:]=np.sqrt(data.sens.info.loc['t'].to_numpy()/data.sens.info[tf-1]['t'])
    data.R_std[:,0]=1e-6
    data.R=np.prod([frames[i0].R for i0 in i],axis=0) #Take product of all frames in i
    data.detect=r0  #This is the detector object, copied into the data object for processing
    fit0.append(data.fit(bounds=False)) #Process the data and store as a fit
    
#%% Here we sweep over the motions individually (i.e. not a product)    
fit1=list()
for f in frames:
    f.detect=r0  #Copy detector object to individual frames
    f.R_std=frames[0].R_std #Copy standard deviations
    fit1.append(f.fit(bounds=False)) #Fit the results from the individual motion

#%% Calculate relaxation rate constant sensitivities and detector sensitivities
"""Use nmr to calculate sensitivities of NMR and R1 experiments. First line
adds NOE experiments, second line adds R1 experiments
"""
nmr=DR.sens.NMR(Type='NOE',Nuc='13CHD2',v0=[1000,700,400]) 
nmr.new_exp(Type='R1',Nuc='13CHD2',v0=[1000,700,400])

nNMR=nmr.R().shape[0]   #Number of NMR experiments
nD=4 #Number of  detectors to analyze these experiments with

r=DR.sens.detect(nmr)   #Calculate detectors from NMR expeirments
r.r_auto(nD-1,inclS2=True,Normalization='MP') #Detector optimization with S2

targetNMR=np.concatenate([[np.ones(200)],nmr.R()]) #Target for calculating NMR relaxation rates
targetNMR[0,140:]=0     #This zeros out the S2 sensitivity for long correlation times 
targetDET=r.rhoz()      #Target for calculating detector responses
targetDET[0,75:]=0      #This zeros out the rho0 sensitivity for long correlation times


#%% Calculate rate constants/detector responses and plot results
rNMR=fit0[0].detect.copy()          #Detector object for calculating NMR rate constants
rNMR.r_target(targetNMR,n=15)
rDET=fit0[0].detect.copy()          #Detector object for calculating detector responses
#rDET.r_auto(n=6,NegAllow=0)
rDET.r_target(targetDET,n=18)

#%% Set up figures
fig=plt.figure('Experiments and Detectors with Different motions')
fig.clear()
fig.set_size_inches([10.27,  8.03])
ax=[fig.add_subplot(nmr.info.shape[1]+1,3,k+1) for k in range(0,3*(nmr.info.shape[1]+1),3)]
ax1=[fig.add_subplot(r.info.shape[1],3,k+1) for k in range(1,3*r.info.shape[1],3)]
ax2=[fig.add_subplot(r.info.shape[1],3,k+1) for k in range(2,3*r.info.shape[1],3)]



label=['methyl libr.','methyl rot.\n(total)','methyl rot.+\n'+r'$\chi_2$ lib.',
       'methyl rot.+\n'+r'$\chi_2$ rot.',r'methyl+$\chi_2$ rot.+'+'\n'+r'$\chi_1$ rot.',
       'methyl+\n'+r'$\chi_1+\chi_2$ rot.','all motion']
label1=['methyl libr.','methyl hop',r'$\chi_2$ libr.',r'$\chi_2$ hop',r'$\chi_1$ libr.',
       r'$\chi_1$ hop',r'C$\alpha$-C$\beta$ motion']

nmrp=list()
detp=list()
det=list()
for k,(f,f1) in enumerate(zip(fit0,fit1)):
    f.detect=rNMR
    nmrp.append(DR.fitting.opt2dist(f.fit()))

    for m,a in enumerate(ax):a.bar(k,nmrp[-1].R[:,m].mean(0)/rNMR.rhoz()[m].max())
    f.detect=rDET
    detp.append(DR.fitting.opt2dist(f.fit()))

    for m,a in enumerate(ax1):a.bar(k,detp[-1].R[:,m].mean(0))
    
    f1.detect=rDET
    det.append(DR.fitting.opt2dist(f1.fit()))

    for m,a in enumerate(ax2):a.bar(k,det[-1].R[:,m].mean(0))
    


string=DR.tools.nice_str(r'{0} at {1} MHz, $\tau_c\approx${2:q2},$\Delta z$={3:.1f}')
string.unit='s'
pad=3
dz=nmr.z()[1]-nmr.z()[0]
z=(nmr.z()*nmr.R()).sum(1)/nmr.R().sum(1)   #Calculates the center of each NMR sensitivity
Delz=nmr.R().sum(1)*dz/nmr.R().max(axis=1)  #Calculates the width of each NMr sensitivity

"Put the centers and widths into the title"
for info,a,z0,Dz in zip(nmr.info.items(),ax[1:],z,Delz):
    a.set_title(string.format(info[1]['Type'],info[1]['v0'],10**z0,Dz),y=1,pad=pad)
ax[0].set_title(r'$1-S^2,\tau_c<500$ ns',y=1,pad=pad)

string=DR.tools.nice_str(r'$\rho_{0}, \tau_c\approx${1:q2},$\Delta z$={2:.1f}')
string.unit='s'
for k,(a,z0,Dz) in enumerate(zip(ax1,rDET.info.loc['z0'],rDET.info.loc['Del_z'])):
    a.set_title(string.format(k,10**z0,Dz),y=1,pad=pad)
for k,(a,z0,Dz) in enumerate(zip(ax2,rDET.info.loc['z0'],rDET.info.loc['Del_z'])):
    a.set_title(string.format(k,10**z0,Dz),y=1,pad=pad)
    
for a in [*ax,*ax1]:
    if a.is_last_row():
        a.set_xticks(range(7))
        a.set_xticklabels(label,rotation=90)
    else:
        a.set_xticks(range(7))
        a.set_xticklabels([])
        
for a,lbl in zip(ax,[r'$1-S^2$',r'$\Gamma_{HC}$/s',r'$\Gamma_{HC}$/s',r'$\Gamma_{HC}$/s',
                     r'$R_1$/s',r'$R_1$/s',r'$R_1$/s']):a.set_ylabel(lbl)
ax[0].set_ylabel(r'$1-S^2$')
for k,a in enumerate(ax1):a.set_ylabel(r'$\rho_'+'{0}'.format(k)+'^{(\\theta,S)}$')
for k,a in enumerate(ax2):a.set_ylabel(r'$\rho_'+'{0}'.format(k)+'^{(\\theta,S)}$')


for a in ax2:
    if a.is_last_row():
        a.set_xticks(range(7))
        a.set_xticklabels(label1,rotation=90)
    else:
        a.set_xticks(range(7))
        a.set_xticklabels([])

for a in [*ax,*ax1,*ax2]:
    yl=a.get_ylim()
    a.set_ylim([0,yl[1]])

plt.show()