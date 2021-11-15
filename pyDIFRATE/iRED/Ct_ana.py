#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:58:49 2019

@author: albertsmith
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


"""

"""
We do basic analysis of correlation functions, without iRED analysis, to 
compare to iRED results
"""

import numpy as np
import multiprocessing as mp
#import os
#os.chdir('../data')
from pyDIFRATE.data.data_class import data
#os.chdir('../iRED')
from pyDIFRATE.iRED.iRED_ana import get_vec
from pyDIFRATE.iRED.iRED_ana import align_vec

#%% Create a data object from the Correlation function results
def Ct2data(molecule,**kwargs):
    
    
    
    if molecule.sel1in is None:
        in1=np.arange(molecule.sel1.atoms.n_atoms)
    else:
        in1=molecule.sel1in
    if molecule.sel2in is None:
        in2=np.arange(molecule.sel2.atoms.n_atoms)
    else:
        in2=molecule.sel2in
        
    vec=get_vec(molecule.sel1,molecule.sel2,in1=in1,in2=in2,**kwargs)
    

    if 'align' in kwargs and kwargs.get('align').lower()[0]=='y':
        vec=align_vec(vec)
    
    ct=Ct(vec,**kwargs)
    
    S2=S2calc(vec)
    
    
    Ctdata=data(molecule=molecule,Ct=ct,S2=S2)
    

    return Ctdata

#%% Calculate correlation functions
def Ct(vec,**kwargs):    
    if 'dt' in kwargs:
        dt=kwargs.get('dt')
        nt=vec.get('t').size
        t=np.arange(0,dt*nt,dt)
    else:
        t=vec.get('t')      
    
    nb=vec.get('X').shape[0]
 
    "Prepare the data needed for each correlation function"    
    v1=list()
    for k in range(0,nb):
        v1.append(np.array([vec.get('X')[k,:],vec.get('Y')[k,:],vec.get('Z')[k,:]]))
    
    "Run in series or in parallel"
    if 'parallel' in kwargs and kwargs.get('parallel').lower()[0]=='n':
        ct0=list()
        for k in range(0,nb):
            ct0.append(Ct_parfun(v1[k]))         
    else:             
        nc=mp.cpu_count()
        if'n_cores' in kwargs:
            nc=np.min([kwargs.get('n_cores'),nc])
            
        with mp.Pool(processes=nc) as pool:
            ct0=pool.map(Ct_parfun,v1)
    

    ct={'t':t,'Ct':np.array(ct0)}
    
    return ct
        
                
           
def Ct_parfun(v):
    nt=np.shape(v)[1]
    for m in range(0,nt):
        v0=np.repeat(np.transpose([v[:,m]]),nt-m,axis=1)
        if m==0:
            ct=(3*np.sum(v0*v[:,m:],axis=0)**2-1)/2
        else:
            ct[0:-m]+=(3*np.sum(v0*v[:,m:],axis=0)**2-1)/2
            
    ct=ct/np.arange(nt,0,-1)
            
    return ct

def Ct_kj(vec,bond1,bond2,**kwargs):
    #We can change the time axis here (should we move this into iRED_ana.get_vec?)
    if 'dt' in kwargs:
        dt=kwargs.get('dt')
        nt=vec.get('t').size
        t=np.arange(0,dt*nt,dt)
    else:
        t=vec.get('t')    
    t=np.concatenate((-t[-1:0:-1],t))
    
    v1=np.array([vec.get('X')[bond1],vec.get('Y')[bond1],vec.get('Z')[bond1]])
    v2=np.array([vec.get('X')[bond2],vec.get('Y')[bond2],vec.get('Z')[bond2]])
    
    nt=np.shape(v1)[1]
    Ct=np.zeros([2*nt-1])
    for k in range(0,nt):
        v0=np.repeat(np.transpose([v1[:,k]]),nt,axis=1)
        Ct[nt-k-1:2*nt-k-1]+=(3*np.sum(v0*v2,axis=0)**2-1)/2
    
    norm=np.arange(1,nt+1)
    norm=np.concatenate((norm,norm[-2::-1]))
    
    Ct=Ct/norm
    
    ct={'t':t,'Ct':Ct}
    
    return ct

#%% Calculate the order parameter
def S2calc(vec):
    v=[vec.get('X'),vec.get('Y'),vec.get('Z')]
    S2=np.zeros(np.shape(vec.get('X'))[0])
    for k in v:
        for m in v:
            S2+=np.mean(k*m,axis=1)**2
    
    S2=3/2*S2-1/2
    
    return S2        