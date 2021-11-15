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

Created on Wed May  8 12:26:05 2019

@author: albertsmith
"""

import numpy as np
#import data.data_class as dc
from scipy.optimize import lsq_linear as lsq
from scipy.stats import norm
#import os
import multiprocessing as mp
#os.chdir('../r_class')
from pyDIFRATE.r_class.detectors import detect as dt
#os.chdir('../data')

def fit_data(data,detect=None,bounds=True,ErrorAna=None,save_input=True,parallel=True,subS2=False,**kwargs):
    """
    Subsequent fitting is currently failing (I think), because we are later trying to 
    fit the detectors that result from the R2 exchange correction. Should have an 
    automatic mechanism to discard these in later fits.
    """
    if detect is None:
        if data.detect is None:
            print('A detect object must be provided in the input or as part of the data object')
            return
        else:
            detect=data.detect
    
    if detect.r(bond=0) is None:
        print('First optimize a set of detectors for analysis')
        return
    
    nb=data.R.shape[0]  #number of data points to fit (n_bonds)
    
    "Output object"
#    out=dc.data()
    out=data.__class__()
    "The new sensitivities of the output data are the detectors used"
    out.sens=detect.copy()
    out.sens._disable()    #Clear the input sensitivities (restricts ability to further edit sens)
    
    "Delete the estimation of R2 due to exchange if included in the data here"
    if hasattr(data.sens,'detect_par') and data.sens.detect_par['R2_ex_corr']:
        R=data.R.copy()[:,:-1]
        R_std=data.R_std.copy()[:,:-1]
        data=data.copy('shallow')    #We don't want to edit the original data object by deleting some of the R data
        "The shallow copy alone would still edit the original R data"
        "Replacing the matrices, however, should leave the orignal matrices untouched"
        data.R=R
        data.R_std=R_std
        
    
    nd=detect.r(bond=0).shape[1]    #Number of detectors
    out.R=np.zeros([nb,nd])
    
    "Set some defaults for error analysis"
    if ErrorAna is not None:
        ea=ErrorAna
        if ea.lower()[0:2]=='mc':
            if len(ea)>2:
                nmc=int(ea[2:])
            else:
                nmc=100
        else:
            nmc=0
    else:
        nmc=0
            
    if 'Conf' in kwargs:
        conf=kwargs.get('Conf')
    else:
        conf=0.68
    out.conf=conf
    
    inclS2=detect.detect_par['inclS2']
    if data.S2 is not None and subS2 and not(inclS2):
        print('Subtracting S2')
        subS2=True
    else:
        subS2=False
    

#    "Set up parallel processing"
#    if 'parallel' in kwargs:
#        if kwargs.get('parallel')[0].lower()=='y':
#            para=True
#        else:
#            para=False
#    elif not(bounds):
#        para=False
#    else:
#        if nmc==0:
#            para=True
#        else:
#            para=True
#    
    if not(bounds):
        Y=list()
        for k in range(nb):
            r,R,_,_=fit_prep(k,data,detect,subS2)
                        
            nstd=norm.ppf(1/2+conf/2)
            std=np.sqrt(np.sum(np.linalg.pinv(r)**2,axis=1))
            u=nstd*std
            l=nstd*std
            rho=np.dot(np.linalg.pinv(r),R)
            Y.append((rho,std,u,l))
        
    elif not(parallel):
        "Series processing (only on specific user request)"
        Y=list()
        for k in range(nb):
            r,R,UB,LB=fit_prep(k,data,detect,subS2)
            X=(r,R,LB,UB,conf,nmc)
            Y.append(para_fit(X))
    else:
        "Here, we buildup up X with all the information required for each fit"
        "required: normalized data, normalized r, upper and lower bounds"
        X0=list()
        for k in range(0,nb):
            r,R,UB,LB=fit_prep(k,data,detect,subS2)
            X0.append((r,R,LB,UB,conf,nmc))
        
        "Parallel processing (default)"
        nc=mp.cpu_count()
        if 'n_cores' in kwargs:
            nc=np.min([kwargs.get('n_cores'),nc])
            
        with mp.Pool(processes=nc) as pool:
            Y=pool.map(para_fit,X0)



    Rc=np.zeros(data.R.shape)
    S2c=np.zeros(data.R.shape[0])

    nd=detect.r(bond=0).shape[1]
    out.R=np.zeros([nb,nd])
    out.R_std=np.zeros([nb,nd])
    out.R_l=np.zeros([nb,nd])
    out.R_u=np.zeros([nb,nd])        
    for k in range(0,nb):
        out.R[k,:]=Y[k][0]
        out.R_std[k,:]=Y[k][1]
        out.R_l[k,:]=Y[k][2]
        out.R_u[k,:]=Y[k][3]
#        if detect.detect_par['inclS2'] and data.S2 is not None:
        if inclS2:
            R0in=np.concatenate((detect.R0in(k),[0]))
            Rc0=np.dot(detect.r(bond=k),out.R[k,:])+R0in
            Rc[k,:]=Rc0[:-1]
            S2c[k]=Rc0[-1]
        else:
            Rc[k,:]=np.dot(detect.r(bond=k),out.R[k,:])+detect.R0in(k)

    if save_input:
        out.Rc=Rc
        if inclS2:
            out.S2c=1-S2c
        
    out.sens.info.loc['stdev']=np.median(out.R_std,axis=0)
        
    if save_input:
        out.Rin=data.R
        out.Rin_std=data.R_std
        if inclS2:
            out.S2in=data.S2
            out.S2in_std=data.S2_std
            
    
        
    out.detect=dt(detect)
    
    out.ired=data.ired
    out.label=data.label
    

    out.chi2=np.sum((data.R-Rc)**2/(data.R_std**2),axis=1)
    
    return out

def fit_prep(k,data,detect,subS2):
    """
    Function that prepares data for fitting (builds the R matrix), re-normalizes
    the detector matrix, r, establishes bounds
    """
    rhoz=detect.rhoz(bond=k)
    UB=rhoz.max(axis=1)
    LB=rhoz.min(axis=1)
    r=detect.r(bond=k)
    
    if data.S2 is not None and detect.detect_par['inclS2']:
        R0=np.concatenate((data.R[k,:]-detect.R0in(k),[1-data.S2[k]]))
        Rstd=np.concatenate((data.R_std[k,:],[data.S2_std[k]]))
        R=R0/Rstd
    elif data.S2 is not None and subS2:
        Rstd=data.R_std[k,:]
        R=(data.R[k,:]-data.S2[k]-detect.R0in(k))/Rstd
    else:
        Rstd=data.R_std[k,:]
        R=(data.R[k,:]-detect.R0in(k))/Rstd
        
    r=r/np.repeat(np.transpose([Rstd]),r.shape[1],axis=1)
    
    return r,R,UB,LB
    


def para_fit(X):
    """Function to calculate results in parallel
    Input is the r matrix, after normalization by the standard deviations, R/R_std,
    such that the data is normalized to a standard deviation of 1, upper and
    lower bounds, the desired confidence interval (.95, for example), and finally
    the number of Monte-Carlo repetitions to perform (if set to 0, performs
    linear propagation-of-erro) 
    """
    Y=lsq(X[0],X[1],bounds=(X[2],X[3]))
    rho=Y['x']
    Rc=Y['fun']+X[1]
    
    if X[5]==0:
        std=np.sqrt(np.sum(np.linalg.pinv(X[0])**2,axis=1))
        nstd=norm.ppf(1/2+X[4]/2)
        u=nstd*std
        l=nstd*std
        
    else:
        Y1=list()
        nmc=max([X[5],np.ceil(2/X[4])])
        for k in range(0,X[5]):
            Y0=lsq(X[0],Rc+np.random.normal(size=X[1].shape))
            Y1.append(Y0['x'])
        std=np.std(Y1,axis=0)
        Y1sort=np.sort(Y1,axis=0)
        in_l=np.round(nmc*(1/2-X[4]/2))
        in_u=np.round(nmc*(1/2+X[4]/2))
        l=rho-Y1sort[int(in_l)]
        u=Y1sort[int(in_u)]-rho
       
    return rho,std,l,u

#%% Function to force a data object to be fully consistent with a positive dynamics distribution
def opt2dist(data,sens=None,parallel=True,return_dist=False,in_place=False,detect=None,**kwargs):
    """
    Takes a distribution and sensitivity object (usually contained in the data
    object, but can be provided separately), and for each bond/residue, optimizes
    a distribution that approximately yields the set of detectors, while requiring
    that the distribution itself only contains positive values and has an integral
    of 1 (or 1-S2, if S2 is stored in data). Note that the distribution itself 
    is not a good reporter on dynamics; it is neither regularized or a stable
    description of dynamics. However, its use makes the detector responses more
    physically consistent
    
    If the original detector object is provided, the original data fit will be recalculated
    
    opt_data=opt2dist(data,sens=None,para=True,return_dist=False,in_place=False,detect=None)
    
    returns 0, 1, or 2 values, depending on the setting of return_dist and in_place
    
    """
    
    nb=data.R.shape[0]
    
    if data.S2 is None:
        S2=np.zeros(nb)
    else:
        S2=data.S2
        
    if sens is None:
        sens=data.sens

    "data required for optimization"
    X=[(R,R_std,sens._rho(bond=k),S2r) for k,(R,R_std,S2r) in enumerate(zip(data.R,data.R_std,S2))]        
    
    if parallel:
        nc=mp.cpu_count()
        if 'n_cores' in kwargs:
            nc=np.min([kwargs.get('n_cores'),nc])
            
        with mp.Pool(processes=nc) as pool:
            Y=pool.map(dist_opt,X)
    else:
        Y=[dist_opt(X0) for X0 in X]
    
    out=data if in_place else data.copy() #We'll edit out, which might be the same object as data
    
    dist=list()
    for k,y in enumerate(Y):
        out.R[k]=y[0]
        dist.append(y[1])

    "If these are detector responses, we'll recalculate the data fit if detector object provided"  
    if detect is not None:
        Rc=list()
        if detect.detect_par['inclS2']:
            for k in range(out.R.shape[0]):
                R0in=np.concatenate((detect.R0in(k),[0]))
                Rc0=np.dot(detect.r(bond=k),out.R[k,:])+R0in
                Rc.append(Rc0[:-1])
        else:
            for k in range(out.R.shape[0]):
                Rc.append(np.dot(detect.r(bond=k),out.R[k,:])+detect.R0in(k))
        out.Rc=np.array(Rc)
    
    
    "Output"
    if in_place and return_dist:
        return dist
    elif in_place:
        return
    elif return_dist:
        return (out,dist)
    else:
        return out
    
def dist_opt(X):
    """
    Optimizes a distribution that yields detector responses, R, where the 
    distribution is required to be positive, and have an integral of 1-S2
    
    Ropt,dist=dist_opt((R,R_std,rhoz,S2,dz))
    
    Note- intput is via tuple
    """
    
    R,R_std,rhoz,S2=X
    total=np.atleast_1d(1-S2)
    """Later, we may need to play with the weighting here- at the moment, we
    fit to having a sum of 1, but in fact it is not forced....it should be
    """
    
    ntc=rhoz.shape[1]
    rhoz=np.concatenate((rhoz/np.repeat(np.atleast_2d(R_std).T,ntc,axis=1),
        np.atleast_2d(np.ones(ntc))),axis=0)
    Rin=np.concatenate((R/R_std,total))
    
    dist=0
    while np.abs(np.sum(dist)-total)>1e-3:  #This is a check to see that the sum condition has been satisfied
        dist=lsq(rhoz,Rin,bounds=(0,1))['x']
        Rin[-1]=Rin[-1]*10
        rhoz[-1]=rhoz[-1]*10
    Ropt=np.dot(rhoz[:-1],dist)*R_std
    
    return Ropt,dist

#%% Function to fit a set of detector responses to a single correlation time
def fit2tc(data,df=2,sens=None,z=None,Abounds=False):
    """
    Takes a data object, and corresponding sensitivity (optional if included in
    data), and fits each set of detector responses to a single correlation time
    (mono-exponential fit). Returns the log-correlation time for each data entry,
    corresponding amplitudes, error, and back-calculated values.
    
    Note that the sensitivity may be a sensitivity object, or a numpy array, but
    in the latter case, the log-correlation time axis, z, must also be included
    
    One may change the fitting function:
        df=1:   exp(-t/tc)
        df=2:   A*exp(-t/tc)
        df=3:   A*exp(-t/tc)+C
    
    Note- in the case of df=3, C is *not* calculated. Instead, any detector that
    reaches its max (test:>.95*max(rhoz)) at the last correlation time is omitted 
    from the fit. Its predicted value is still included in Rc
    
    Setting Abounds to True will force A to fall within the range of 0 and 1
        
    z,A,err,Rc=fit2tc(data,df=2,sens=None,z=None,Abounds=False)
    """
    
    if sens is None:
        sens=data.sens
        
    z0=sens.z() if z is None else z
        
    err=list()
    z=list()
    A=list()
    rho_c=list()
    for k,(rho,rho_std) in enumerate(zip(data.R,data.R_std)):
        if hasattr(sens,'rhoz'):
            rhoz=sens.rhoz(k)
        else:
            rhoz=sens

        
        if df==3:
            y=(rho-rhoz[:,-1])/rho_std
            x=((rhoz.T-rhoz[:,-1])/rho_std).T
            x=x[:,:-1]
        else:
            y=rho/rho_std
            x=(rhoz.T/rho_std).T
#        if df==3:
#            j=rhoz[:,-1]/rhoz.max(axis=1)<.95
#            jj=np.logical_not(j)
#            y=y[j]
#            x=x[j]


        
        if df!=1:
            beta=(((1/(x**2).sum(axis=0))*x).T*y).sum(axis=1)
        else:
            beta=1

        err0=(((beta*x).T-y)**2).sum(axis=1)
        if Abounds:
            err0[beta>1]=1e10

        i=np.argmin(err0)
        err.append(err0[i])
        z.append(z0[i])
        if df==2:
            rho_c.append(rhoz[:,i]*beta[i])
            A.append(beta[i])
        elif df==3:
            rho_c.append((rhoz[:,i]-rhoz[:,-1])*beta[i]+rhoz[:,-1])
            
            A.append(beta[i])
        else:
            rho_c.append(rhoz[:,i])
            A.append(1)
    
    return np.array(z),np.array(A),np.array(err),np.array(rho_c)
        
    
