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

Created on Wed Aug 28 10:06:35 2019

@author: albertsmith
"""

import numpy as np
import multiprocessing as mp
#import os
#os.chdir('../data')
from pyDIFRATE.data.data_class import data
#os.chdir('../iRED')
from MDAnalysis.analysis.align import rotation_matrix
from psutil import virtual_memory
from pyDIFRATE.iRED.fast_funs import S2calc,Ct,get_trunc_vec,align_mean
from pyDIFRATE.iRED.fast_index import trunc_t_axis,get_count
from pyDIFRATE.iRED.par_iRED import par_class as ipc
from time import time

#%% Run the full iRED analysis
def iRED_full(mol,rank=2,n=100,nr=10,align_iRED=False,refVecs=None,**kwargs):
    """
    Runs the full iRED analysis for a given selection (or set of vec_special functions)
    Arguments are the rank (0 or 1), the sampling (n,nr), whether to align the 
    vectors (align_iRED='y'/'n', and refVecs, which may be a dict containing 
    a vector, created by DIFRATE, a tuple of strings selecting two sets of atoms
    defining bonds), or simply 'y', which will default to using the N-CA bonds in
    a protein.
    
    ired=iRED_full(mol,rank=2,n=100,nr=10,align_iRED='n',refVecs='n',**kwargs)
    
    """

    if 'nt' in kwargs:
        nt=np.min([mol.mda_object.trajectory.n_frames,kwargs.get('nt')])
    else:
        nt=mol.mda_object.trajectory.n_frames
    index=trunc_t_axis(nt,n,nr)
    vec=get_trunc_vec(mol,index,**kwargs)
    
    if align_iRED:
        if refVecs is not None:
            vec0=refVecs
            if isinstance(vec0,dict):
                pass
            elif len(vec0)==2 and isinstance(vec0[0],str) and isinstance(vec0[1],str):
                mol1=mol.copy()
                mol1.select_atoms(sel1=vec0[0],sel2=vec0[1])
                vec0=get_trunc_vec(mol1,index)
            elif isinstance(vec0,str) and vec0.lower()[0]=='y':
                s1='protein and name CA and around 1.6 N'
                s2='protein and name N and around 1.6 CA'
                mol1=mol.copy()
                mol1.select_atoms(sel1=s1,sel2=s2)
                vec0=get_trunc_vec(mol1,index)
            else:
                print('Warning: refVecs entry not valid, using input vectors as reference (without aligning)')
                vec0=vec
        else:
            vec0=vec
    else:
        vec0=None
        
    ired=vec2iRED(vec,rank,align_iRED,refVecs=vec0,molecule=mol,**kwargs)
        
    return ired

#%% Process with iRED from a vector
def vec2iRED(vec,rank=2,align_iRED=False,align_type='ZDir',refVecs=None,**kwargs):
    """
    Takes a vector object and returns the iRED object (vec contains X,Y,Z,t, and
    usually an index for sparse sampling of the time axis)
    
    If align_iRED is set to True, then by default, vec will be used aligned and
    unaligned for a reference vector. The reference vectors (refVecs) may be
    used to replace the unaligned input vector
    
    iRED=vec2iRED(vec,rank=2,align_iRED=False,**kwargs)
    """

    if refVecs is not None:
        vec0=refVecs
        n_added_vecs=vec0.get('X').shape[1]
    elif align_iRED:
        vec0=vec.copy()
        n_added_vecs=vec0.get('X').shape[1]
    else:
        vec0=None
        n_added_vecs=0
        

    if align_iRED:
        vec=align_mean(vec,rank,align_type)
        aligned=True
    else:
        aligned=False
        
    if vec0 is not None:
        for k in ['X','Y','Z']:
            vec[k]=np.concatenate((vec[k],vec0[k]),axis=1)
        
    M=Mmat(vec,rank)
    Yl=Ylm(vec,rank)
    aqt=Aqt(Yl,M)
    
    "parallel calculation of correlation functions"
    ct=Cqt(aqt)
    ctinf=CtInf(aqt)
    dct=DelCt(ct,ctinf)
    ired={'rank':rank,'M':M['M'],'lambda':M['lambda'],'m':M['m'],'t':ct['t'],\
          'N':ct['N'],'index':ct['index'],'DelCt':dct['DelCt'].T,'CtInf':ctinf,\
          'Aligned':aligned,'n_added_vecs':n_added_vecs}
    
    Ctdata=data(iRED=ired,**kwargs)
#    Ctdata.sens.molecule=molecule
#    Ctdata.detect.molecule=Ctdata.sens.molecule
    
    return Ctdata

#%% Generate a data object with iRED results
def iRED2data(molecule,rank=2,**kwargs):
    """Input a molecule object with selections already made, to get a full iRED 
    analysis, moved into a data object
    """
    
    
    """
    Not sure what happened here. Looks like iRED_full performs all steps of this
    calculation. Should get rid of one name or the other...
    """
    Ctdata=iRED_full(molecule,rank,**kwargs)
#    ired=iRED_full(molecule,**kwargs)
    
#    Ctdata=data(iRED=ired,molecule=molecule,**kwargs)
#    Ctdata.sens.molecule=molecule
##    Ctdata.sens.molecule.set_selection()
#    Ctdata.detect.molecule=Ctdata.sens.molecule
#    
    return Ctdata
#%% Calculate the iRED M matrix
def Mmat(vec,rank=2):
    """Calculates the iRED M-matrix, yielding correlation of vectors at time t=0
    M = Mmat(vec,rank=2)
    M is returned as dictionary object, including the matrix itself, and also
    the 
    """
    
    X=vec['X'].T
    Y=vec['Y'].T
    Z=vec['Z'].T
    
    nb=X.shape[0]
    
    M=np.eye(nb)
    
    for k in range(0,nb-1):
        "These are the x,y,z positions for one bond"
        x0=np.repeat([X[k,:]],nb-k-1,axis=0)
        y0=np.repeat([Y[k,:]],nb-k-1,axis=0)
        z0=np.repeat([Z[k,:]],nb-k-1,axis=0)
        
        "We correlate those positions with all bonds having a larger index (symmetry of matrix allows this)"
        dot=x0*X[k+1:,:]+y0*Y[k+1:,:]+z0*Z[k+1:,:]
        
        if rank==1:
            val=np.mean(dot,axis=1)
        elif rank==2:
            val=np.mean((3*dot**2-1)/2,axis=1)
            
        M[k,k+1:]=val
        M[k+1:,k]=val
        
    Lambda,m=np.linalg.eigh(M)
    return {'M':M,'lambda':Lambda,'m':m,'rank':rank}

def Mlagged(vec,lag,rank=2):
    """Calculates the iRED M-matrix, with a lag time, which is provided by an 
    index or range of indices (corresponding to the separation in time points)
    M = Mlagged(vec,rank=2,lag)
    
    lag=10
    or 
    lag=[10,20]
    
    The first instance calculates M using time points separated by exactly the
    lag index. The second takes all time points separated by the first argument, 
    up to one less the last argument (here, separated by 10 up to 19)
    
    """
    
    X=vec['X'].T
    Y=vec['Y'].T
    Z=vec['Z'].T
    
    index0=vec['index']
    
    if np.size(lag)==1:
        lag=np.atleast_1d(lag)
    elif np.size(lag)==2:
        lag=np.arange(lag[0],lag[1])
    
    "Calculate indices for pairing time points separated within the range given in lag"
    index1=np.zeros(0,dtype=int)
    index2=np.zeros(0,dtype=int)
    for k in lag:
        i=np.isin(index0+k,index0)
        j=np.isin(index0,index0+k)
        index1=np.concatenate((index1,np.where(i)[0]))
        index2=np.concatenate((index2,np.where(j)[0]))
    
    nb=X.shape[0]
    M=np.eye(nb)
    
    for k in range(0,nb):
        "We correlate all times that have a second time within the lag range"
        x0=np.repeat([X[k,index1]],nb,axis=0)
        y0=np.repeat([Y[k,index1]],nb,axis=0)
        z0=np.repeat([Z[k,index1]],nb,axis=0)

        dot=x0*X[:,index2]+y0*Y[:,index2]+z0*Z[:,index2]
        
        if rank==1:
            val=np.mean(dot,axis=1)
        elif rank==2:
            val=np.mean((3*dot**2-1)/2,axis=1)
            
        M[k,:]=val
        
    return M
#%% Estimates cross-correlation of the eigenvectors of the M matrix
def Mrange(vec,rank,i0,i1):
    """Estimates the Mmatrix for frames offset by a minimum distance of i0 and
    a maximum distance of i1-1. All M-matrices are simply added together
    M=Mrange(vec,rank,i0,i1)
    """
    pass

#%% Calculates the spherical tensor components for the individual bonds
def Ylm(vec,rank=2):
    """
    Calculates the values of the rank-2 spherical components of a set of vectors
    Yl=Ylm(vec,rank)
    """
    X=vec.get('X')
    Y=vec.get('Y')
    Z=vec.get('Z')
    
    
    Yl=dict()
    if rank==1:
        c=np.sqrt(3/(2*np.pi))
        Yl['1,0']=c/np.sqrt(2)*Z
        a=(X+Y*1j)
#        b=np.sqrt(X**2+Y**2)
#        Yl['1,+1']=-c/2*b*a  #a was supposed to equal exp(i*phi), but wasn't normalized (should be normalized by b)
#        Yl['1,-1']=c/2*b*a.conjugate()  #Correction below
        Yl['1,+1']=-c/2*a
        Yl['1,-1']=c/2*a.conjugate()
    elif rank==2:
        c=np.sqrt(15/(32*np.pi))
        Yl['2,0']=c*np.sqrt(2/3)*(3*Z**2-1)
        a=(X+Y*1j)
#        b=np.sqrt(X**2+Y**2)
#        b2=b**2
#        b[b==0]=1
#        Yl['2,+1']=2*c*Z*b*a
#        Yl['2,-1']=2*c*Z*b*a.conjugate()
        Yl['2,+1']=2*c*Z*a
        Yl['2,-1']=2*c*Z*a.conjugate()
#        a=np.exp(2*np.log(X+Y*1j))
#        b=b**2
#        Yl['2,+2']=c*b*a
#        Yl['2,-2']=c*b*a.conjugate()
        a2=a**2
#        a2[a!=0]=np.exp(2*np.log(a[a!=0]/b[a!=0]))
        Yl['2,+2']=c*a2
        Yl['2,-2']=c*a2.conjugate()
        
    Yl['t']=vec['t']
    Yl['index']=vec['index']
    return Yl

def Aqt(Yl,M):
    """
    Project the Ylm onto the eigenmodes
    aqt=Aqt(Yl,M)
    """
    aqt=dict()
    for k,y in Yl.items():
        if k!='t' and k!='index':
            aqt[k]=np.dot(M['m'].T,y.T).T
        else:
            aqt[k]=y
        
    return aqt

def Cqt(aqt,**kwargs):
    
    "Get number of cores"
    if 'parallel' in kwargs:
        p=kwargs.get('parallel')
        if isinstance(p,str) and p.lower()[0]=='n':
            nc=1
        elif isinstance(p,int):
            nc=p if p>0 else 1   #Check the # cores is bigger than 0
        else:                   #Default use parallel processing
            nc=mp.cpu_count()   #Use all available cores
    else:
        nc=mp.cpu_count()
        
    ref_num,v0=ipc.store_vecs(aqt,nc)
    try:
        t0=time()
        with mp.Pool(processes=nc) as pool:
            ct=pool.map(ipc.Ct,v0)
#            print('t={0}'.format(time()-t0))
        ct=ipc.returnCt(ref_num,ct)
    except:
        print('Error in calculating correlation functions')
    finally:
        ipc.clear_data(ref_num)
    
    index=aqt['index']
    N=get_count(index)
    dt=np.diff(aqt['t'][0:2])/np.diff(index[0:2])
    t=np.linspace(0,dt.squeeze()*np.max(index),index[-1]+1)
    i=N!=0
    N=N[i]
    t=t[i]
    ct=dict({'Ct':ct,'t':t,'index':index,'N':N})
    
    return ct

def Cij_t(aqt,i,j,**kwargs):
    """
    Calculates the cross correlation between modes in the iRED analysis, indexed
    by i and j 
    (this function should later be improved using parallel processing for multiple
    pairs of modes. Currently supports only one pair)
    c_ij=Cij_t(aqt,i,j,**kwargs)
    """
    
    index=aqt['index']
    n=np.size(index)
    
    
    for p,(name,a) in enumerate(aqt.items()):
        if p==0:
            ct=np.zeros(index[-1]+1)+0j
        if name!='index' and name!='t':
            for k in range(n):
                ct[index[k:]-index[k]]+=np.multiply(a[k:,i],a[k,j].conjugate())
    N0=get_count(index)
    nz=N0!=0
    N=N0[nz]
    dt=np.diff(aqt['t'][0:2])/np.diff(index[0:2])
    t=np.linspace(0,dt.squeeze()*np.max(index),index[-1]+1)
    t=t[nz]
    ct=np.divide(ct[nz].real,N)
    
    ct=dict({'Ct':ct,'t':t,'index':index,'N':N})
    
    return ct
    
#%% Estimate the correlation function at t=infinity
def CtInf(aqt):
    "Get final value of correlation function"
    ctinf=None
    for k in aqt.keys():
        if k!='t' and k!='index':
            a=aqt.get(k).mean(axis=0)
            if np.shape(ctinf)==():
                ctinf=np.real(a*a.conj())
            else:
                ctinf+=np.real(a*a.conj())
            
    return ctinf

#%% Estimate the correlation function at t=infinity
def Cij_Inf(aqt,i,j):
    "Get final value of correlation function"
    ctinf=None
    for k in aqt.keys():
        if k!='t' and k!='index':
            a=aqt.get(k)[:,i].mean()
            b=aqt.get(k)[:,j].mean()
            if np.shape(ctinf)==():
                ctinf=np.real(a*b.conj())
            else:
                ctinf+=np.real(a*b.conj())
            
    return ctinf

#%% Returns normalized correlation function
def DelCt(ct,ctinf):
    "Get a normalized version of the correlation function (starts at 1, decays to 0)"
    t=ct.get('t')
    ct=ct.get('Ct')
    nt=ct.shape[0]
    ctinf=np.repeat([ctinf],nt,axis=0)
    ct0=np.repeat([ct[0,:]],nt,axis=0)
    delCt={'t':t,'DelCt':(ct-ctinf)/(ct0-ctinf)}
    
    return delCt



def iRED2dist(bond,data,nbins=None,all_modes=False,Type='avg'):
    """
    Estimates a distribution of correlation times for a given bond in the iRED 
    analysis. We calculate a correlation time for each mode (we fit detector 
    responses to a single mode). Then, we calculate the amplitude of each mode
    on the selected bond. Finally, we calculate a histogram from the results.
    
    z,A=iRED2dist(bond,fit,nbins=None)
    
    Note, that fit needs to be the detector fit of the iRED modes, not the final
    fit (resulting from fit.iRED2rho())
    """
    
    "Get the best-fit correlation time for each mode"
#    z0,_,_=fit2tc(data.R,data.sens.rhoz(),data.sens.z(),data.R_std)
    if Type[0].lower()=='a':
        z0=avgz(data.R,data.sens.z(),data.sens.rhoz())
    else:
        z0,_,_=fit2tc(data.R,data.sens.rhoz(),data.sens.z())
    
    if bond in data.label:
        i=np.argwhere(bond==data.label).squeeze()
    else:
        i=bond
    
    m0=data.ired['m'].T
    l0=data.ired['lambda']
    
    A0=np.zeros(z0.shape)
    
    for k,(l,m) in enumerate(zip(l0,m0)):
        A0[k]=m[i]**2*l
        
    
    if nbins is None:
        nbins=np.min([data.sens.z().size,z0.size/10])
        
    #Axis for histogram    
    z=np.linspace(data.sens.z()[0],data.sens.z()[-1],nbins)
    
    i=np.digitize(z0,z)-1
    
    if all_modes:
        ne=-A0.size
    else:
        ne=data.ired['rank']*2+1
    
    A=np.zeros(z.shape)
    for k,a in enumerate(A0[:-ne]):
        A[i[k]]+=a
    
    return z,A

def avgz(R,z,rhoz):
    """
    Estimates an "average" z for a set of detector responses, determined simply
    by the weighted average of the z0 for each detector (weighted by the
    detector responses). Note that we use max-normalized detectors for this 
    calculation
    """
    nd,nz=np.shape(rhoz)
    z0=np.sum(np.repeat([z],nd,axis=0)*rhoz,axis=1)/np.sum(rhoz,axis=1)
    nb=R.shape[0]
    norm=np.max(rhoz,axis=1)
    
    R=np.divide(R,np.repeat([norm],nb,axis=0))
    
    z=np.divide(np.multiply(R,np.repeat([z0],nb,axis=0)).sum(axis=1),R.sum(axis=1))
    
    return z
    
def fit2tc(R,rhoz,tc,R_std=None):
    """
    Estimates a single correlation time for a set of detector responses, based 
    on the sensitivities of thoses detectors (in principle, may be applied to
    any sensitivity object, but with better performance for optimized detectors)
    
    tc,A=fit2tc(R,sens)
    
    R may be a 2D matrix, in which case each row is a separate set of detector
    responses (and will be analyzed separately)
    """    
    
    R=np.atleast_2d(R)  #Make sure R is a 2D matrix
    if R_std is None:
        R_std=np.ones(R.shape)
    
    
    nd,nz=rhoz.shape    #Number of detectors, correlation times
    nb=R.shape[0]       #Number of bonds
    
    err=list()      #Storage for error
    A=list()        #Storage for fit amplitudes
    
    
    for X in rhoz.T:
        R0=np.divide(R,R_std)
        rho=np.divide(np.repeat([X],nb,axis=0),R_std)
        A.append(np.divide(np.mean(np.multiply(rho,R0),axis=1),np.mean(rho**2,axis=1)))
        err.append(np.power(R0-rho*np.repeat(np.transpose([A[-1]]),nd,axis=1),2).sum(axis=1))
        
    A0=np.array(A)
    err=np.array(err)
    
    i=err.argmin(axis=0)
    tc=np.array(tc[i])
    
    A=np.zeros(nb)
    Rc=np.zeros(R.shape)
    
    for k in range(nb):
        A[k]=A0[i[k],k]
        Rc[k]=A[k]*rhoz[:,i[k]]
    
    return tc,A,Rc