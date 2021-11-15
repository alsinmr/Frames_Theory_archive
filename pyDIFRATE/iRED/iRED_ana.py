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


Created on Mon May  6 13:47:45 2019

@author: albertsmith
"""

"""
We create a number of functions for performing the iRED analysis. These depend 
on the input of one or two MDanalysis objects, that can be used to specify the
bond vector direction. 
"""

import numpy as np
import multiprocessing as mp
#import os
import MDAnalysis as md
from MDAnalysis.analysis import align
#os.chdir('../data')
from pyDIFRATE.data.data_class import data
#os.chdir('../iRED')


#%% Run the full iRED analysis
def iRED_full(sel1,sel2,rank,**kwargs):
    if 'alignCA' not in kwargs:
        kwargs['alignCA']='n'
        "We don't need this functionality for the iRED analysis, although user can still force it"
    
    vec=get_vec(sel1,sel2,**kwargs)
    
    if 'align' in kwargs and kwargs.get('align').lower()[0]=='y':
        vec0=vec
        vec=align_vec(vec)
        
        
        if not('refvecs' in kwargs and kwargs.get('refVecs').lower()[0]=='y'):
            n_added_vecs=vec0.get('X').shape[0]
            for k in vec.keys():
                if k!='t':
                    vec[k]=np.concatenate((vec.get(k),vec0.get(k)),axis=0)        
            aligned=True
    else:
        aligned=False
        
    if  'refVecs' in kwargs and kwargs.get('refVecs').lower()[0]=='y':
        """If we align the vectors, we need reference vectors as well
        This allows us to properly separate overall motion"
        Otherwise, overall motion is gathered into 1 eigenvector, instead of 2*rank+1 vectors
        """
        sel01=sel1.universe.select_atoms('protein and name CA and bonded name N')
        sel02=sel1.universe.select_atoms('protein and name N and bonded name CA')
        
        if sel01.n_atoms==0:
            sel01=sel1
            sel02=sel2
        
        vec0=get_vec(sel01,sel02,**kwargs)
        n_added_vecs=vec0.get('X').shape[0]
        
        for k in vec.keys():
            if k!='t':
                vec[k]=np.concatenate((vec.get(k),vec0.get(k)),axis=0)
    else:
        n_added_vecs=0
        
    M=Mmat(vec,rank)
    Yl=Ylm(vec,rank)
    aqt=Aqt(Yl,M)

    "Default is to use parallel processing"    
    if 'parallel' in kwargs and kwargs.get('parallel').lower()[0]=='n':
        cqt=Cqt(aqt)
    else:
        cqt=Cqt_par(aqt,**kwargs)
        
    ct=Ct(cqt)
    ctinf=CtInf(aqt)
    dct=DelCt(ct,ctinf)
    
    if 'dt' in kwargs:
        "mdanalysis seems to import the wrong time step in some instances."
        "This can be corrected by providing dt"
        dt=kwargs.get('dt')
        nt=np.size(vec.get('t'))
        t=np.arange(0,dt*nt,dt)
        vec['t']=t
    
    ired={'rank':rank,'M':M.get('M'),'lambda':M.get('lambda'),'m':M.get('m'),\
          't':vec.get('t'),'Ct':ct.get('Ct'),'DelCt':dct.get('DelCt'),'CtInf':ctinf,\
          'Aligned':aligned,'n_added_vecs':n_added_vecs}
    
        
    return ired
#%% Create a data object from the iRED results (also runs the analysis)
def iRED2data(molecule,rank,**kwargs):
    """Input a molecule object with selections already made, to get a full iRED 
    analysis, moved into a data object
    """
    
    if molecule.sel1in is None:
        in1=np.arange(molecule.sel1.n_atoms)
    else:
        in1=molecule.sel1in
    if molecule.sel2in is None:
        in2=np.arange(molecule.sel2.n_atoms)
    else:
        in2=molecule.sel2in
    
    ired=iRED_full(molecule.sel1,molecule.sel2,rank,in1=in1,in2=in2,**kwargs)
    
    
    Ctdata=data(iRED=ired,molecule=molecule)
    Ctdata.sens.molecule=molecule
    Ctdata.sens.molecule.set_selection()
    Ctdata.detect.molecule=Ctdata.sens.molecule
    
    return Ctdata
    
#%% Load in vectors for the iRED analysis    
def get_vec(sel1,sel2,**kwargs):
    "Gets vectors from an MDanalysis selection, returns X,Y,Z in dictionary"
    a=sel1.universe
    b=sel2.universe
    
    if 'in1' in kwargs:
        in1=kwargs.get('in1')
    else:
        "Just changed this line. Could be wrong!!!"
        in1=np.arange(sel1.n_atoms)
        
    if 'in2' in kwargs:
        in2=kwargs.get('in2')
    else:
        "Also changed this line. "
        in2=np.arange(sel2.n_atoms)
    
    
    if a!=b:
        print('sel1 and sel2 must be generated from the same MDAnalysis universe!')
        return

    if sel1.n_atoms!=sel2.n_atoms and np.size(in1)!=np.size(in2):
        print('sel1 and sel2 or indices sel1in and sel2in must have the same number of atoms')
        return


    if 'tstep'in kwargs:
        tstep=kwargs.get('tstep')
        print('Take every {0}th frame'.format(tstep))
    else:
        tstep=1
    

    nt=int((a.trajectory.n_frames-1)/tstep)+1
    na=np.size(in1)
    
    X=np.zeros([na,nt])
    Y=np.zeros([na,nt])
    Z=np.zeros([na,nt])
    
    k=0
    
    
    try:
        for k in range(0,nt):
            a.trajectory[k*tstep]
            pos=sel1.positions[in1]-sel2.positions[in2]
            
            X0=pos[:,0]
            Y0=pos[:,1]
            Z0=pos[:,2]
            
            length=np.sqrt(X0**2+Y0**2+Z0**2)
            
            X[:,k]=np.divide(X0,length)
            Y[:,k]=np.divide(Y0,length)
            Z[:,k]=np.divide(Z0,length)
            if k%int(nt/100)==0 or k+1==nt:
                printProgressBar(k+1, nt, prefix = 'Loading:', suffix = 'Complete', length = 50)
    except:
        ts0=iter(a.trajectory)

        for ts in ts0:
            for _ in range(tstep-1):
                next(ts0,None)
            pos=sel1.positions[in1]-sel2.positions[in2]
            X0=pos[:,0]
            Y0=pos[:,1]
            Z0=pos[:,2]
            
            length=np.sqrt(X0**2+Y0**2+Z0**2)
            
            X[:,k]=np.divide(X0,length)
            Y[:,k]=np.divide(Y0,length)
            Z[:,k]=np.divide(Z0,length)
            
            k=k+1
            if k%int(nt/100)==0 or k+1==nt:
                printProgressBar(k+1, nt, prefix = 'Loading:', suffix = 'Complete', length = 50)   
    dt=a.trajectory.dt*tstep
    t=np.arange(0,nt*dt,dt)
 
    vec={'X':X,'Y':Y,'Z':Z,'t':t}
    
    if not('alignCA' in kwargs and kwargs.get('alignCA').lower()[0]=='n'):
        "Default is to always align the CA"
        vec=alignCA(vec,a,**kwargs)

    
    

    return vec       
    
def alignCA(vec0,uni,tstep=1,**kwargs):
    "reference CA positions"
    
    if 'align_ref' in kwargs:
        uni0=uni.select_atoms(kwargs.get('align_ref'))
    else:
        uni0=uni.select_atoms('name CA')
        
    if uni0.n_atoms==0:
        print('No atoms found for alignment, specify atom for alignment with align_ref')
        return vec0

    ref0=uni0.positions-uni0.atoms.center_of_mass()
    
    SZ=np.shape(vec0.get('X'))
    "Pre-allocate the direction vector"
    vec={'X':np.zeros(SZ),'Y':np.zeros(SZ),'Z':np.zeros(SZ),'t':vec0.get('t')} 
    
    nt=vec0['t'].size
    
    for k in range(0,nt):
        try:
            uni.trajectory[k*tstep]
        except:
            if k!=0:
                for _ in range(0,tstep):
                    uni.next()
        "CA positions"
        pos=uni0.positions-uni0.atoms.center_of_mass()

        "Rotation matrix for this time point"
        R,_=align.rotation_matrix(pos,ref0)
        "Apply rotation to vectors"
        vec['X'][:,k]=vec0['X'][:,k]*R[0,0]+vec0['Y'][:,k]*R[0,1]+vec0['Z'][:,k]*R[0,2]
        vec['Y'][:,k]=vec0['X'][:,k]*R[1,0]+vec0['Y'][:,k]*R[1,1]+vec0['Z'][:,k]*R[1,2]
        vec['Z'][:,k]=vec0['X'][:,k]*R[2,0]+vec0['Y'][:,k]*R[2,1]+vec0['Z'][:,k]*R[2,2]

        if k%int(nt/100)==0 or k+1==nt:
            printProgressBar(k+1, nt, prefix = 'Aligning positions:', suffix = 'Complete', length = 50)
            
    return vec
        

#%% Make all vectors point in the same directon (remove influence of orientation on analysis)    
def align_vec(vec0):
    "Aligns the mean direction of a set of vectors along the z-axis"
    
    nt=vec0.get('X').shape[1]
    
    "Mean direction of the vectors"
    X0=vec0.get('X').mean(axis=1)
    Y0=vec0.get('Y').mean(axis=1)
    Z0=vec0.get('Z').mean(axis=1)
    
    length=np.sqrt(X0**2+Y0**2+Z0**2)
    X0=np.divide(X0,length)
    Y0=np.divide(Y0,length)
    Z0=np.divide(Z0,length)
    
    "Angle away from the z-axis"
    beta=np.arccos(Z0)
    
    "Angle of rotation axis away from y-axis"
    "Rotation axis is at (-Y0,X0): cross product of X0,Y0,Z0 and (0,0,1)"
    theta=np.arctan2(-Y0,X0)
    
    xx=np.cos(-theta)*np.cos(-beta)*np.cos(theta)-np.sin(-theta)*np.sin(theta)
    yx=-np.cos(theta)*np.sin(-theta)-np.cos(-theta)*np.cos(-beta)*np.sin(theta)
    zx=np.cos(-theta)*np.sin(-beta)
    
    X=np.repeat(np.transpose([xx]),nt,axis=1)*vec0.get('X')+\
    np.repeat(np.transpose([yx]),nt,axis=1)*vec0.get('Y')+\
    np.repeat(np.transpose([zx]),nt,axis=1)*vec0.get('Z')
    
    xy=np.cos(-theta)*np.sin(theta)+np.cos(-beta)*np.cos(theta)*np.sin(-theta)
    yy=np.cos(-theta)*np.cos(theta)-np.cos(-beta)*np.sin(-theta)*np.sin(theta)
    zy=np.sin(-theta)*np.sin(-beta)
    
    Y=np.repeat(np.transpose([xy]),nt,axis=1)*vec0.get('X')+\
    np.repeat(np.transpose([yy]),nt,axis=1)*vec0.get('Y')+\
    np.repeat(np.transpose([zy]),nt,axis=1)*vec0.get('Z')

    xz=-np.cos(theta)*np.sin(-beta)
    yz=np.sin(-beta)*np.sin(theta)
    zz=np.cos(-beta)
    
    Z=np.repeat(np.transpose([xz]),nt,axis=1)*vec0.get('X')+\
    np.repeat(np.transpose([yz]),nt,axis=1)*vec0.get('Y')+\
    np.repeat(np.transpose([zz]),nt,axis=1)*vec0.get('Z')

    
#    "Some code here to make a specific pair of vectors anticorrelated"
#    "DELETE ME"
#    print('Making first and second bond anti-correlated')
#    X[139,:]=-X[140,:]
#    Y[139,:]=-Y[140,:]
#    Z[139,:]=Z[140,:]
#    
    vec={'X':X,'Y':Y,'Z':Z,'t':vec0.get('t')}
    
    return vec

def Mmat(vec,rank):
    
    nb=vec.get('X').shape[0]
    
    M=np.eye(nb)
    
    for k in range(0,nb-1):
        x0=np.repeat([vec.get('X')[k,:]],nb-k-1,axis=0)
        y0=np.repeat([vec.get('Y')[k,:]],nb-k-1,axis=0)
        z0=np.repeat([vec.get('Z')[k,:]],nb-k-1,axis=0)
        
        dot=x0*vec.get('X')[k+1:,:]+y0*vec.get('Y')[k+1:,:]+z0*vec.get('Z')[k+1:,:]
        
        if rank==1:
            val=np.mean(dot,axis=1)
        elif rank==2:
            val=np.mean((3*dot**2-1)/2,axis=1)
            
        M[k,k+1:]=val
        M[k+1:,k]=val
        
    a=np.linalg.eigh(M)
    return {'M':M,'lambda':a[0],'m':a[1],'rank':rank}

def Mt(vec,rank,tstep):
    nb=vec.get('X').shape[0]
    
    M=np.eye(nb)
    for k in range(0,nb):
        x0=np.repeat([vec.get('X')[k,tstep:]],nb,axis=0)
        y0=np.repeat([vec.get('Y')[k,tstep:]],nb,axis=0)
        z0=np.repeat([vec.get('Z')[k,tstep:]],nb,axis=0)

        if tstep!=0:
            dot=x0*vec.get('X')[:,0:-tstep]+y0*vec.get('Y')[:,0:-tstep]+z0*vec.get('Z')[:,0:-tstep]
        else:
            dot=x0*vec.get('X')+y0*vec.get('Y')+z0*vec.get('Z')
            
        if rank==1:
            val=np.mean(dot,axis=2)
        elif rank==2:
            val=np.mean((3*dot**2-1)/2,axis=1)
            
        M[k,:]=val
            
    return M
        
def Ylm(vec,rank):
    
    X=vec.get('X')
    Y=vec.get('Y')
    Z=vec.get('Z')
    
    
    Yl=dict()
    if rank==1:
        c=np.sqrt(3/(2*np.pi))
        Yl['1,0']=c/np.sqrt(2)*Z
        a=(X+Y*1j)
        b=np.sqrt(X**2+Y**2)
        Yl['1,+1']=-c/2*b*a
        Yl['1,-1']=c/2*b*a.conjugate()
    elif rank==2:
        c=np.sqrt(15/(32*np.pi))
        Yl['2,0']=c*np.sqrt(2/3)*(3*Z**2-1)
        a=(X+Y*1j)
        b=np.sqrt(X**2+Y**2)
        Yl['2,+1']=2*c*Z*b*a
        Yl['2,-1']=2*c*Z*b*a.conjugate()
        a=np.exp(2*np.log(X+Y*1j))
        b=b**2
        Yl['2,+2']=c*b*a
        Yl['2,-2']=c*b*a.conjugate()
        
    Yl['t']=vec.get('t')
    
    return Yl

def Aqt(Yl,M):
    "Project the Ylm onto the eigenmodes"
    aqt=dict()
    for k in Yl.keys():
        if k!='t':
            aqt[k]=np.dot(M.get('m').T,Yl.get(k))
        
    aqt['t']=Yl.get('t')
    
    return aqt


def Cqt(aqt):
    "Get correlation functions for each spherical component"
    cqt=dict()
    for k in aqt.keys():
        if k!='t':
            "Loop over each component"
            nt=aqt.get(k).shape[1]
            nb=aqt.get(k).shape[0]
            for m in range(0,nt):
                "Correlate the mth time point with all other time points"
                a0=np.repeat(np.conj(np.transpose([aqt.get(k)[:,m]])),nt-m,axis=1)
                if m==0:
                    c0=a0*aqt.get(k)+np.zeros([nb,nt])*1j #Make c0 complex
                else:
                    c0[:,0:-m]+=a0*aqt.get(k)[:,m:]
                    
                if m%int(nt/100)==0 or m+1==nt:
                    printProgressBar(m+1, nt, prefix = 'Ct({}):'.format(k), suffix = 'Complete', length = 50)
            print()
            "Divide to normalize for more time points at beginning than end"
            cqt[k]=c0/np.repeat([np.arange(nt,0,-1)],nb,axis=0)
            

      
    cqt['t']=aqt['t']
    
    return cqt

def Cqt_par(aqt,**kwargs):
    "Performs same operation as Cqt, but using parallel processing"
    X=list()
    
    nc=mp.cpu_count()
    if'n_cores' in kwargs:
        nc=np.min([kwargs.get('n_cores'),nc])
        
        
    for k in range(0,nc):
        X.append((aqt,k,nc))

        
    with mp.Pool(processes=nc) as pool:
        X=pool.map(Cqt_parfun,X)
        
    cqt=dict()

    for k in aqt.keys():
        if k!='t':
            nt=aqt.get(k).shape[1]
            nb=aqt.get(k).shape[0]
            cqt[k]=np.zeros([nb,nt])+0*1j
        
    for cqt0 in X:
        for k in cqt0.keys():
            cqt[k]+=cqt0[k]

    for k in cqt.keys():               
        cqt[k]=cqt[k]/np.repeat([np.arange(nt,0,-1)],nb,axis=0)
                
    cqt['t']=aqt['t']
    
    return cqt

def Cqt_parfun(X):
    "Function to be run by Cqt_par in parallel"
    aqt=X[0]
    index=X[1]
    nc=X[2]
    
    cqt0=dict()
    for k in aqt.keys():
        if k!='t':
            "Loop over each component"
            nt=aqt.get(k).shape[1]
            nb=aqt.get(k).shape[0]
            c0=np.zeros([nb,nt])+0*1j
            for l,m in enumerate(range(index,nt,nc)):
                "Correlate the mth time point with all other time points"
                a0=np.repeat(np.conj(np.transpose([aqt.get(k)[:,m]])),nt-m,axis=1)           
                if m==0:
                    c0=a0*aqt.get(k)+0*1j #Make c0 complex
                else:
                    c0[:,0:-m]+=a0*aqt.get(k)[:,m:]

            cqt0[k]=c0
    
    return cqt0

def Ct(cqt):
    "Sum up all components to get the overall correlation function"
    ct0=None
    for k in cqt.keys():
        if k!='t':
            if np.shape(ct0)==():
                ct0=cqt.get(k)
            else:
                ct0+=cqt.get(k)
            
    ct={'t':cqt.get('t'),'Ct':ct0.real}
    
    return ct

def CtInf(aqt):
    "Get final value of correlation function"
    ctinf=None
    for k in aqt.keys():
        if k!='t':
            a=aqt.get(k).mean(axis=1)
            if np.shape(ctinf)==():
                ctinf=np.real(a*a.conj())
            else:
                ctinf+=np.real(a*a.conj())
            
    return ctinf

def DelCt(ct,ctinf):
    "Get a normalized version of the correlation function (starts at 1, decays to 0)"
    t=ct.get('t')
    ct=ct.get('Ct')
    nt=ct.shape[1]
    ctinf=np.repeat(np.transpose([ctinf]),nt,axis=1)
    ct0=np.repeat(np.transpose([ct[:,0]]),nt,axis=1)
    delCt={'t':t,'DelCt':(ct-ctinf)/(ct0-ctinf)}
    
    return delCt


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()