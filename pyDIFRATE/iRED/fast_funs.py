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


Created on Wed Aug 28 10:24:19 2019

@author: albertsmith
"""
#import os
import numpy as np
import multiprocessing as mp
from pyDIFRATE.iRED.parCt import par_class as pct
from pyDIFRATE.iRED.fast_index import get_count
#os.chdir('../Struct')
import pyDIFRATE.Struct.vf_tools as vft
#os.chdir('../iRED')

#%% Estimate the order parameter
def S2calc(vec):
    """
    Calculates an estimate of the order parameter, according to
    3/2*(<x^2>^2+<y^2>^2+<z^2>^2+2<x*y>^2+2<x*z>^2+2<y*z>^2)-1/2 with averages performed
    over the complete vector
    
    S2=S2calc(vec)
    """
    if 'Y' in vec.keys():
        v=np.array([vec.get('X'),vec.get('Y'),vec.get('Z')])
        SZ=vec['X'].shape[1]

    else:
        v=np.array([vec['Z']['X'],vec['Z']['Y'],vec['Z']['Z']])
        SZ=vec['Z']['X'].shape[1]
    v=v/np.sqrt((v**2).sum(axis=0))
    
    S2=np.zeros(SZ)    
    for k in v:
        for m in v:
            S2+=np.mean(k*m,axis=0)**2
    
    S2=3/2*S2-1/2    
    return S2

#%% Returns the correlation function defined by vec
def Ct(vec,**kwargs):
    """
    Calculates the correlation functions for vectors with unequal spacing in the
    time axis. By default, uses parallel processing (using all available cores)
    Optional arguments are parallel, which determines whether or not to use 
    parallel processing ('y'/'n'), or optionally one may simply set parallel to
    the desired number of cores (parallel=4, for example)
    """    
    
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
           
    
    if 'n_cores' in kwargs:
        nc=np.min([kwargs.get('n_cores'),nc])
        print('Warning: n_cores argument will be removed in a later version. set parallel=n_cores')
        "Optional second argument. Not documented- possibly will be removed"
        
        
    if 'Y' not in vec.keys():
        nc=1
        print('Only series processing if eta is non-zero')
        nb=vec['X']['X'].shape[1]
    else:
        nb=vec['X'].shape[1]

    if nc==1:
        "Might later replace this with the code in place"
        "But, should keep some variant- parallel version isn't necessarily stable"
        v0=list()   #Store date required for each core
        k=0
        if 'Y' in vec.keys():
            v0.append((vec['X'][:,range(k,nb,nc)],vec['Y'][:,range(k,nb,nc)],vec['Z'][:,range(k,nb,nc)],vec['index']))
        else:
            v0.append((vec['X']['X'][:,range(k,nb,nc)],vec['X']['Y'][:,range(k,nb,nc)],vec['X']['Z'][:,range(k,nb,nc)],\
                       vec['Z']['X'][:,range(k,nb,nc)],vec['Z']['Y'][:,range(k,nb,nc)],vec['Z']['Z'][:,range(k,nb,nc)],\
                       vec['eta'],vec['index']))
    if nc==1:   #Series processing
        ct0=list()
        for v in v0:
            ct0.append(Ct_par(v))
    else:
        ref_num,v0=pct.store_vecs(vec,nc)
        print('Success')
        try:
            with mp.Pool(processes=nc) as pool:
#                ct=pool.map(ctpar.Ct,v0)
                ct=pool.map(pct.Ct,v0)
            ct=pct.returnCt(ref_num,ct)
        finally:
            pct.clear_data(ref_num)
            
    "Get the count of number of averages"
    index=vec['index']
    N=get_count(index)
    
    "i finds all times points for which we have some averaging of the correlation function"    
    i=N!=0
    N=N[i]
    

    N0=N
    
    if nc==1:
        ct=np.zeros([np.size(N0),nb])
        for k in range(nc):
            N=np.repeat([N0],np.shape(ct0[k])[1],axis=0).T  #N with same shape as ct
            ct[:,range(k,nb,nc)]=np.divide(ct0[k][i],N) #Normalize correlation function based on how many averages
        
    
    dt=(vec['t'][1]-vec['t'][0])/(index[1]-index[0])
    t=np.linspace(0,dt*np.max(index),index[-1]+1)
    t=t[i]
    
    Ct={'t':t,'Ct':ct.T,'N':N0,'index':index}
    
    return Ct


#%% Parallel function to calculate correlation functions
def Ct_par(v):
    if len(v)==8:
        X_X=v[0]
        Y_X=v[1]
        Z_X=v[2]
        X_Z=v[3]
        Y_Z=v[4]
        Z_Z=v[5]
        eta=v[6]
        index=v[7]
        
        n=np.size(index)
        c=np.zeros([np.max(index)+1,np.shape(X_X)[1]])
        
        for k in range(n):
            Cb2=(np.multiply(X_Z[k:],X_Z[k])+np.multiply(Y_Z[k:],Y_Z[k])+np.multiply(Z_Z[k:],Z_Z[k]))**2
            Ca2Sb2=(np.multiply(X_Z[k:],X_X[k])+np.multiply(Y_Z[k:],Y_X[k])+np.multiply(Z_Z[k:],Z_X[k]))**2
#            c[index[k:]-index[k]]+=Cb2*(3-eta)/2-eta*Ca2Sb2+(eta-1)/2
            c[index[k:]-index[k]]+=(3-eta)/2*Cb2-eta*Ca2Sb2+(eta-1)/2
        return c
    else:
        index=v[3]
        X=v[0]
        Y=v[1]
        Z=v[2]
    
        n=np.size(index)
        c=np.zeros([np.max(index)+1,np.shape(X)[1]])
        
        for k in range(n):
            c[index[k:]-index[k]]+=(3*(np.multiply(X[k:],X[k])+np.multiply(Y[k:],Y[k])\
                 +np.multiply(Z[k:],Z[k]))**2-1)/2
    #        if k%int(n/100)==0 or k+1==n:
    #            printProgressBar(k+1, n, prefix = 'C(t) calc:', suffix = 'Complete', length = 50) 
        return c

#%% Load in the truncated vectors from the trajectory
def get_trunc_vec(molecule,index,**kwargs):
    """
    vec=get_trunc_vec(molecule,index,**kwargs)
    
    Returns time-dependent vectors defined in the molecule object. Usually this
    is vectors defined by atom selections in sel1 and sel2 (and possibly indexed
    by sel1in and sel2in). Alternatively, if function-defined vectors are stored
    in molecule._vf (molecule.vec_fun() returns vectors), then these will be 
    returned instead
    
    One must provide the molecule object, and an index determining which time 
    points to analyze.
    
    Optional arguments are dt, which re-defines the time step
    (vs. the time step returned by MDAnalysis), and align, which can be set to
    'y' and will remove overall motion by aligning all frames to a reference
    set of atoms. Default is CA in proteins. To change default, provide a second
    argument, align_ref, which is an MDAnalysis selection string. This string 
    will select from all atoms in the trajectory, and align them.
    
    
    """
    
    
    
    if molecule._vf is not None and False:  #De-activate this functionality. Replace with frames
        vf=molecule.vec_fun
        special=True
    else:
        sel1=molecule.sel1
        sel2=molecule.sel2
        sel1in=molecule.sel1in
        sel2in=molecule.sel2in
        
        "Indices to allow using the same atom more than once"
        if sel1in is None:
            sel1in=np.arange(sel1.n_atoms)
        if sel2in is None:
            sel2in=np.arange(sel2.n_atoms)
            
        if sel1.universe!=sel2.universe:
            print('sel1 and sel2 must be generated from the same MDAnalysis universe')
            return
            
        if np.size(sel1in)!=np.size(sel2in):
            print('sel1 and sel2 or sel1in and sel2in must have the same number of atoms')
            return
        special=False
    
    nt=np.size(index) #Number of time steps
    if special:
        na=vf().shape[1]
    else:
        na=np.size(sel1in) #Number of vectors
    
    X=np.zeros([nt,na])
    Y=np.zeros([nt,na])
    Z=np.zeros([nt,na])
    t=np.zeros([nt])

    uni=molecule.mda_object
    traj=uni.trajectory
    if 'dt' in kwargs:
        dt=kwargs.get('dt')
    else:
        dt=traj.dt/1e3
#        if traj.units['time']=='ps':    #Convert time units into ns
#            dt=dt/1e3
#        elif traj.units['time']=='ms':
#            dt=dt*1e3
        

    ts=iter(traj)
    for k,t0 in enumerate(index):
        try:
            traj[t0]     #This jumps to time point t in the trajectory
        except:
            "Maybe traj[t] doesn't work, so we skip through the iterable manually"
            if k!=0:    
                for _ in range(index[k]-index[k-1]):
                    next(ts,None)
                    
        if special:
            "Run the function to return vector"
            X0,Y0,Z0=vf()
        else:
            "Else just get difference in atom positions"
            v=sel1[sel1in].positions-sel2[sel2in].positions
            "We correct here for vectors extended across the simulation box"
            box=np.repeat([uni.dimensions[0:3]],v.shape[0],axis=0)
            
            i=v>box/2
            v[i]=v[i]-box[i]
            
            i=v<-box/2
            v[i]=v[i]+box[i]
            
            "Store the results"
            X0=v[:,0]
            Y0=v[:,1]
            Z0=v[:,2]
        
        "Make sure length is one"
        length=np.sqrt(X0**2+Y0**2+Z0**2)
        if np.any(length>2):
#            print(molecule.sel1[molecule.sel1in[length>3]].names)
#            print(molecule.sel2[molecule.sel2in[length>3]].names)
#            print(length[length>3])
            print(k)
        X[k,:]=np.divide(X0,length)
        Y[k,:]=np.divide(Y0,length)
        Z[k,:]=np.divide(Z0,length)
        "Keep track of the time axis"
        t[k]=dt*t0
        if k%np.ceil(nt/100).astype(int)==0 or k+1==nt:
            printProgressBar(k+1, nt, prefix = 'Loading:', suffix = 'Complete', length = 50) 

    vec={'X':X,'Y':Y,'Z':Z,'t':t,'index':index}
    
    "Re-align vectors to some set of reference atoms"
    if 'align' in kwargs and kwargs.get('align').lower()[0]=='y':
        "Default does not align molecule"
        vec=align(vec,uni,**kwargs)
        
#    "Re-align vectors so they all point along z"
#    if 'align_iRED' in kwargs and kwargs.get('align_iRED').lower()[0]=='y':
#        vec=align_mean(vec)
           
    return vec

def align_mean(vec0,rank=2,align_type='ZDir'):
    """
    Aligns the mean direction of a set of vectors along the z-axis. This can be
    useful for iRED analysis, to mitigate the orientational dependence of the 
    iRED analysis procedure.
    
    vec = align_mean(vec0)
    
    Options are introduced for the rotation of third angle:
        Type='ZDir' : Sets gamma = -alpha (only option for rank 1 calc)
        Type='tensor' : Aligns the rank 2 tensor, including the asymmetry
        Type='xy-motion' : Aligns the z-component of the rank-2 tensor, and
        maximizes correlation of the x and y components to the previous bond
    """
    
    
    """At some point, we should consider whether it would make sense to use a 
    tensor alignment instead of a vector alignment.
    """
    vec=vec0.copy()  #Just operate on the copy here, to avoid accidental edits
    
    X,Y,Z=vec['X'],vec['Y'],vec['Z']    #Coordinates
    
#    nt=X.shape[0]


    
    #%% Calculate sines and cosines of beta,gamma rotations
    if rank==1:
        "Mean direction of the vectors"
        X0,Y0,Z0=X.mean(axis=0),Y.mean(axis=0),Z.mean(axis=0)
        
        "Normalize the length"
        length=np.sqrt(X0**2+Y0**2+Z0**2)
        X0,Y0,Z0=np.divide([X0,Y0,Z0],length)

        "beta"
        cB,sB=Z0,np.sqrt(1-Z0**2)
        
        "gamma"
        lXY=np.sqrt(X0**2+Y0**2)
        i=lXY==0
        lXY[i]=1.
        cA,sA=[X0,Y0]/lXY
        cA[i]=1.
        cG,sG=cA,-sA
    elif rank==2:
        "Note, rank 2 also aligns the asymmetry of a motion so is a better alignment"
        cossin=vft.getFrame([X,Y,Z]) #Get euler angles for this vector
        D2c=vft.D2(*cossin)    #Calculate Spherical components
        D20=D2c.mean(axis=1) #Calculate average
        sc=vft.Spher2pars(D20)[2:] #Get euler angles
        cA,sA,cB,sB,cG,sG=vft.pass2act(*sc)
            
    "apply rotations"    
    X,Y=cA*X+sA*Y,-sA*X+cA*Y    #Apply alpha
    X,Z=cB*X-sB*Z,sB*X+cB*Z   #Apply beta
    if align_type.lower()[0]=='z':
        X,Y=cA*X-sA*Y,sA*X+cA*Y #Rotate back by -alpha
    else: #Tensor- default option (undone later if using xy-motion)
        X,Y=cG*X+sG*Y,-sG*X+cG*Y  #Apply gamma
    
    if rank==2:
        "Make sure axis is pointing along +z (rotate 180 around y)"
        i=Z.mean(axis=0)<0
        X[:,i],Z[:,i]=-X[:,i],-Z[:,i]
        
        
    "Try to maximize the correlation by aligning X/Y deviations"
    if align_type.lower()[0]=='x':
        c,s=RMS2Dalign(X,Y)
        X,Y=c*X+s*Y,-s*X+c*Y
    
    """
    Note, I'd eventually like to try aligning the vectors to maximimize correlation
    in all three dimensions, that is, an RMS3Dalign function...maybe wouldn't
    make such a difference since we already align the mean tensor directions along z,
    but worth some consideration
    """
    
#        "Check that deviations from average direction go same way for all bonds "
#        iX=(X[:,0]*X).mean(axis=0)-X[:,0].mean()*X.mean(axis=0)<0
#        iY=(Y[:,0]*Y).mean(axis=0)-Y[:,0].mean()*Y.mean(axis=0)<0
#        
#        X[:,iX],Y[:,iY]=-X[:,iX],-Y[:,iY]
#        
#        iX,iY,iZ=X.mean(axis=0)<0,Y.mean(axis=0)<0,Z.mean(axis=0)<0
#        X[:,iX],Y[:,iY],Z[:,iZ]=-X[:,iX],-Y[:,iY],-Z[:,iZ]
#        "Can we really do this? I think flipping the axes should not influence dynamics"
#        "Check 1- uncommenting below still yields D2 along z"
#        "Check 2- small changes to detector responses observed....less convincing"

#    "Check that rotation is correct (if uncommented, D20[1] and D20[3] should be ~zeros, and all elements ~real"
#    cossin=vft.getFrame([X,Y,Z]) #Get euler angles for this vector
#    D2c=vft.D2(*cossin)    #Calculate Spherical components
#    D20=D2c.mean(axis=1) #Calculate average
#    print(D20)
    
    "return results"
    vec['X'],vec['Y'],vec['Z']=X,Y,Z
    return vec
#    nt=X.shape[0]   
#    X0,Y0,Z0=X.mean(axis=0),Y.mean(axis=0),Z.mean(axis=0)    #Coordinates
#    length=np.sqrt(X0**2+Y0**2+Z0**2)
#    X0,Y0,Z0=np.divide([X0,Y0,Z0],length)
#    
#    "Angle away from the z-axis"
#    beta=np.arccos(Z0)
#    
#    "Angle of rotation axis away from y-axis"
#    "Rotation axis is at (-Y0,X0): cross product of X0,Y0,Z0 and (0,0,1)"
#    theta=np.arctan2(-Y0,X0)
#    
#    
#    xx=np.cos(-theta)*np.cos(-beta)*np.cos(theta)-np.sin(-theta)*np.sin(theta)
#    yx=-np.cos(theta)*np.sin(-theta)-np.cos(-theta)*np.cos(-beta)*np.sin(theta)
#    zx=np.cos(-theta)*np.sin(-beta)
#    
#    X=np.repeat([xx],nt,axis=0)*vec0.get('X')+\
#    np.repeat([yx],nt,axis=0)*vec0.get('Y')+\
#    np.repeat([zx],nt,axis=0)*vec0.get('Z')
#    
#    xy=np.cos(-theta)*np.sin(theta)+np.cos(-beta)*np.cos(theta)*np.sin(-theta)
#    yy=np.cos(-theta)*np.cos(theta)-np.cos(-beta)*np.sin(-theta)*np.sin(theta)
#    zy=np.sin(-theta)*np.sin(-beta)
#    
#    Y=np.repeat([xy],nt,axis=0)*vec0.get('X')+\
#    np.repeat([yy],nt,axis=0)*vec0.get('Y')+\
#    np.repeat([zy],nt,axis=0)*vec0.get('Z')
#
#    xz=-np.cos(theta)*np.sin(-beta)
#    yz=np.sin(-beta)*np.sin(theta)
#    zz=np.cos(-beta)
#    
#    Z=np.repeat([xz],nt,axis=0)*vec0.get('X')+\
#    np.repeat([yz],nt,axis=0)*vec0.get('Y')+\
#    np.repeat([zz],nt,axis=0)*vec0.get('Z')
#
#    vec={'X':X,'Y':Y,'Z':Z,'t':vec0['t'],'index':vec0['index']}
#    
#    return vec
    
def RMS2Dalign(X,Y,return_angles=False):
    """
    Returns the optimal 2D rotation to bring a vector of X and Y coordinates onto
    a reference set of coordinates (the X and Y may be a 2D matrix, where each
    column is a new bond, for example). Returns, by default c and s, the cosine
    and sine for the optimal rotation matrix (set return_angles=True to get the
    angle directly)
    """    

    "Consider eliminating for-loop with direct SVD calc"
    "see: https://lucidar.me/en/mathematics/singular-value-decomposition-of-a-2x2-matrix/"
    
    c=list()
    s=list()
    xr,yr=X[:,0],Y[:,0]
    for x,y in zip(X.T,Y.T):
        H=np.array([[(x*xr).sum(),(x*yr).sum()],[(y*xr).sum(),(y*yr).sum()]])
        U,S,Vt=np.linalg.svd(H)
        Ut,V=U.T,Vt.T
        d=np.linalg.det(np.dot(V,Ut))
        R=np.dot(V,np.dot([[1,0],[0,d]],Ut))
        c.append(R[0,0])
        s.append(R[0,1])
        xr,yr=c[-1]*x+s[-1],-s[-1]*x+c[-1]*y
#        print(R)
    c=np.array(c)
    s=np.array(s)
    
    if return_angles:
        return np.arctan2(s,c)
    else:
        return c,s


#%% Removes 
def align(vec0,uni,**kwargs):
    """
    Removes overall rotation from a trajectory, by aligning to a set of reference
    atoms. Default is protein backbone CA. If no CA found, try C11 for lipids 
    (possibly this isn't standard- shouldn't create problems for the time being).
    Next try all carbons, and finally all atoms)
    """
#    if 'align_ref' in kwargs:
#        uni0=uni.select_atoms(kwargs.get('align_ref'))
#    else:
#        uni0=uni.select_atoms('name CA')    #Standard alignment for proteins
#        if uni0.n_atoms==0:
#            uni0=uni.select_atoms('name C11')   #Not sure about this. Alignment for lipids?
#        if uni0.n_atoms==0:
#            uni0=uni.select_atoms('type C') #Try for all carbons
#        if uni0.n_atoms==0:
#            uni0=uni.atoms #Take all atoms
#            
#    if uni0.n_segments>1:
#        "DIfferent segments may be split up after unwrapping. We'll take the segment with the most atoms"
#        count=list()
#        for s in uni0.segments:
#            count.append(s.atoms.n_atoms)
#        uni0=uni0.segments[np.argmax(count)].atoms
#    
#    "Unwrap the segment before this calculation"
##    make_whole(uni0)
#    
#    ref0=uni0.positions-uni0.atoms.center_of_mass()
#    
#    SZ=np.shape(vec0.get('X'))
#    index=vec0['index']
#    "Pre-allocate the direction vector"
#    vec={'X':np.zeros(SZ),'Y':np.zeros(SZ),'Z':np.zeros(SZ),'t':vec0.get('t'),'index':index} 
#
#    nt=vec0['t'].size
#
#    
#    traj=uni.trajectory
#    ts=iter(traj)
#    for k,t0 in enumerate(index):
#        try:
#            traj[t0]     #This jumps to time point t in the trajectory
#        except:
#            "Maybe traj[t] doesn't work, so we skip through the iterable manually"
#            if k!=0:    
#                for _ in range(index[k]-index[k-1]):
#                    next(ts,None) 
#        "Ref positions, first unwrapping the reference segment"
##        make_whole(uni0)
#        pos=uni0.positions-uni0.atoms.center_of_mass()
#        
#        "Rotation matrix for this time point"
#        R,_=rotation_matrix(pos,ref0)
#        "Apply the rotation matrix to the input vector"
#        vec['X'][k,:]=vec0['X'][k,:]*R[0,0]+vec0['Y'][k,:]*R[0,1]+vec0['Z'][k,:]*R[0,2]
#        vec['Y'][k,:]=vec0['X'][k,:]*R[1,0]+vec0['Y'][k,:]*R[1,1]+vec0['Z'][k,:]*R[1,2]
#        vec['Z'][k,:]=vec0['X'][k,:]*R[2,0]+vec0['Y'][k,:]*R[2,1]+vec0['Z'][k,:]*R[2,2]
#        
##        vec['X'][k,:]=vec0['X'][k,:]*R[0,0]+vec0['Y'][k,:]*R[1,0]+vec0['Z'][k,:]*R[2,0]
##        vec['Y'][k,:]=vec0['X'][k,:]*R[0,1]+vec0['Y'][k,:]*R[1,1]+vec0['Z'][k,:]*R[2,1]
##        vec['Z'][k,:]=vec0['X'][k,:]*R[0,2]+vec0['Y'][k,:]*R[1,2]+vec0['Z'][k,:]*R[2,2]
#        "Print out progress"
#        if k%int(np.size(index)/100)==0 or k+1==nt:
#            printProgressBar(k+1, np.size(index), prefix = 'Aligning:', suffix = 'Complete', length = 50) 
#        
#    return vec
    print('Warning: the align function has been removed- please pre-align the trajectory')
    print('Use molecule.align(sel) prior to processing')
    return vec0



#%% Progress bar for loading/aligning
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