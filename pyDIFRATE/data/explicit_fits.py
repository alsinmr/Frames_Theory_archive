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


Created on Wed May 15 13:39:40 2019

@author: albertsmith
"""

import pyDIFRATE as DR
import numpy as np
from scipy.signal import savgol_filter

#%% Fit all correlation functions in data object
def fitCtdata(data):
    n=np.shape(data.R)[0]
    ne=np.shape(data.R)[1]
    
    
    dist=np.zeros([np.shape(data.R)[0],np.size(data.sens.tc())])
    tc=data.sens.tc()
    Rz=data.sens._rho(np.arange(0,ne))
    
    for k in range(0,n):
        Af,As,_,index=Ctfit3p((tc,data.R[k,:],Rz))
        dist[k,0]=Af
        dist[k,index]=As
        
    out=DR.data()
    out.R=dist
    out.ired=data.ired
    
    
    
    return out

#%% Minimization function
def Ctfit3p(X):
    tc0=X[0]
    Ct=X[1]
    Rz=X[2]
    
    Af=Ct[0]-Ct[1]
    As=1-Af
    Ct=Ct/As
    
    
    n=int(np.size(Ct)/2)
    Rz=Rz[1:n,:]
    Ct=Ct[1:n]
    
    err=np.sum((Rz-np.repeat(np.transpose([Ct]),np.shape(Rz)[1],axis=1))**2,axis=0)
    
    a=np.argmin(err)
    
    tc=tc0[a]
    
    return Af,As,tc,a

def ired2dist(data):
    
    n=data.R.shape[0]
    n0=n-data.ired.get('n_added_vecs')
    ne=data.ired.get('rank')*2+1
    
    nd=data.R.shape[1]
    dist=np.zeros([n0,nd])
    
    for k in range(0,nd):
        lambda_dist=np.repeat([data.ired.get('lambda')[0:-ne]*data.R[0:-ne,k]],n0,axis=0)
        dist[:,k]=np.sum(lambda_dist*data.ired.get('m')[0:n0,0:-ne]**2,axis=1)
        
    return dist

def smooth(dist0,box_pts):
    
    dist0=np.atleast_2d(dist0)
    box = np.ones(box_pts)/box_pts
    
    dist=np.zeros(np.shape(dist0))
    for k in range(0,np.shape(dist0)[0]):
#        dist[k,:]=savgol_filter(dist0[k,:],11,3)
        dist[k,:]=np.convolve(dist0[k,:],box,mode='same')
        
    return dist