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


Created on Thu Apr  4 14:51:02 2019

@author: albertsmith
"""

"""
Here we store all models for motion. We keep the basic input in the mdl_sens
class very general, so you can add whatever parameters desired here (pass with 
**kwargs)
To add a new model, just make a new definition. Should return an array of 
correlation times and amplitudes. If model is anisotropic, there should be an 
array of arrays, that is, each bond should have an array. The inner array should
have the same size as the array of correlation times. Note that **kwargs must
be in the arguments, even if it isn't used. We always pass the structure and 
direction to the model function, regardless if it's required, so these can be 
collected with **kwargs. Note, all models should return a third parameter that
states whether the model is bond-specific or not. Value should be a string, 'yes'
or 'no'

Note that we furthermore can have access to the structure, imported via 
the mdanalysis module 
"""

import numpy as np
from numpy import inf

#def ModelBondSpfc(Model):
#    "Add models with bond-specific dynamics to this list"
#    mdls=np.array(['AnisoDif']) 
#    return np.any(Model==mdls)
    
    
def ModelSel(Model,direct='dXY',struct=None,**kwargs):
    """
    General function to select the correct model
    """
    
    if Model=='Combined':
        tMdl,AMdl,BndSpfc=Combined(tMdl1=kwargs.get('tMdl1'),AMdl1=kwargs.get('AMdl1'),\
                           tMdl2=kwargs.get('tMdl2'),AMdl2=kwargs.get('AMdl2'))
        BndSpfc='no'
    else:
        try:
            if Model in globals():
                fun=globals()[Model]
            else:
                print('Model "{0}" was not recognized'.format(Model))
                return
#            if 'struct' in fun.__code__.co_varnames[range(fun.__code__.co_argcount)]:
#                print('Bond Specific')
#                if struct.vXY.size==0:
#                    print('Before defining an model with anisotropic motion, import a structure and select the desired bonds')
#                    return
            tMdl,AMdl,BndSpfc=fun(struct=struct,direct=direct,**kwargs)
            
        except:
            print('Model "{0}" failed. Check parameters'.format(Model))
            return
    
#    if Model=='IsoDif':
#        tMdl,AMdl,BndSpfc=IsoDif(**kwargs)
##        BndSpfc='no'
#        """We must always say if the model is bond specific, so we know if an
#        array of models is being returned"""
#    elif Model=='AnisoDif':
#        tMdl,AMdl,BndSpfc=AnisoDif(struct,direct,**kwargs)
##        BndSpfc='yes'
#    elif Model=='Combined':
#        tMdl,AMdl,BndSpfc=Combined(tMdl1=kwargs.get('tMdl1'),AMdl1=kwargs.get('AMdl1'),\
#                           tMdl2=kwargs.get('tMdl2'),AMdl2=kwargs.get('AMdl2'))
#        BndSpfc='no'

    "Make sure we return np arrays with a dimension"
    tMdl=np.atleast_1d(tMdl)
    AMdl=np.atleast_1d(AMdl)
#    if not isinstance(tMdl,np.ndarray):
#        tMdl=np.array(tMdl)
#    if tMdl.shape==():
#        tMdl=np.array([tMdl])
#    if not isinstance(AMdl,np.ndarray):
#        AMdl=np.array(AMdl)
#    if AMdl.shape==():
#        AMdl=np.array([AMdl])
        
    tMdl[tMdl==inf]=1000
    return tMdl,AMdl,BndSpfc
#%% Simple isotropic diffusion
"Isotropic tumbling in solution"
def IsoDif(**kwargs):

    if 'tM' in kwargs:        
        tMdl=kwargs.get('tM')
    elif 'tm' in kwargs:
        tMdl=kwargs.get('tm')
    elif 'tr' in kwargs:
        tMdl=kwargs.get('tr')
    elif 'tR' in kwargs:
        tMdl=kwargs.get('tR')
        
    AMdl=1
    BndSpfc='no'
    return tMdl,AMdl,BndSpfc

#%% Simple fast motion
"Fast motion (too fast to be detected by relaxation, or occuring within 1st pt of trajectory"
def FastMotion(S2=None,**kwargs):
    tMdl=1e-14    #Arbitrarily short correlation time
    if 'AMdl' in kwargs:
        AMdl=kwargs.get('AMdl')
    elif 'A' in kwargs:
        AMdl=kwargs.get('A')
    elif S2 is None:
        print('You must provide S2 to define the FastMotion model')
        return
    else:
        AMdl=1-S2
        
    if np.size(S2)!=1:
        struct=kwargs.get('struct')
        if struct.sel1in is not None:
            nb=np.size(struct.sel1in)
        elif struct.sel1 is not None:
            nb=struct.sel1.n_atoms
        else:
            nb=None
        if nb is not None and np.size(S2)!=nb:
            print('The size of S2 must be 1 or equal the number of bonds being analyzed')
            return
        else:
            BndSpfc='yes'
        AMdl=np.atleast_2d(AMdl).T
    else:
        BndSpfc='no'
    
    return tMdl,AMdl,BndSpfc
#%% Anisotropic diffusion
def AnisoDif(struct,direct='vXY',**kwargs):
   
    """First we get the diffusion tensor, and also Diso and D2. This can be 
    input either as the principle values, Dxx, Dyy, and Dzz, or as the trace of
    the tensor (the isotropic value, tM), plus optionally the anisotropy, xi, 
    and the asymmetry, eta
    """
    if 'Dxx' in kwargs and 'Dyy' in kwargs and 'Dzz' in kwargs:
        Dzz=kwargs.get('Dzz')
        Dxx=kwargs.get('Dxx')
        Dyy=kwargs.get('Dyy')
        Diso=1/3*(Dxx+Dyy+Dzz)
        Dsq=(Dxx*Dyy+Dyy*Dzz+Dzz*Dxx)/3;
    else:
        if 'tM' in kwargs:        
            tM=kwargs.get('tM')
        elif 'tm' in kwargs:
            tM=kwargs.get('tm')
        elif 'tr' in kwargs:
            tM=kwargs.get('tr')
        elif 'tR' in kwargs:
            tM=kwargs.get('tR')
            
        if 'xi' in kwargs:
            xi=kwargs.get('xi')
        else:
            xi=1
        if 'eta' in kwargs:
            eta=kwargs.get('eta')
        else:
            eta=0
            
        Diso=1/(6*tM);
        Dzz=3*Diso*xi/(2+xi);
        Dxx=(3*Diso-(2/3*eta*(xi-1)/xi+1)*Dzz)/2;
        Dyy=2/3*eta*Dzz*(xi-1)/xi+Dxx;
        Dsq=(Dxx*Dyy+Dyy*Dzz+Dzz*Dxx)/3;
        
    "We the relaxation rates"    
    D1=4*Dxx+Dyy+Dzz;
    D2=Dxx+4*Dyy+Dzz;
    D3=Dxx+Dyy+4*Dzz;
    D4=6*Diso+6*np.sqrt(Diso**2-Dsq);
    D5=6*Diso-6*np.sqrt(Diso**2-Dsq);
    


    dx=(Dxx-Diso)/np.sqrt(Diso**2-Dsq);
    dy=(Dyy-Diso)/np.sqrt(Diso**2-Dsq);
    dz=(Dzz-Diso)/np.sqrt(Diso**2-Dsq);
    
    
    "We rotate the vectors in structure"
    if 'euler' in kwargs and direct=='vXY':
        vec=RotVec(kwargs.get('euler'),struct.vXY)
    elif 'euler' in kwargs:
#        vec=RotVec(kwargs.get('euler'),struct.vCSA) 
        "Use the ABOVE LINE! We need to add support for calculating the CSA direction first...."
        vec=RotVec(kwargs.get('euler'),struct.vXY)
    else:
        print('Euler angles are required')
        return
        
    
    n=vec.shape[0]
    tM=np.zeros([5])
    A=np.zeros([n,5])
    
    for k in range(0,n):
        m=vec[k,:]
        res1=(1/4)*(3*(m[0]**4+m[1]**4+m[2]**4)-1)
        res2=(1/12)*(dx*(3*m[0]**4+6*m[1]**2*m[2]**2-1)\
        +dy*(3*m[1]**4+6*m[2]**2*m[0]**2-1)\
        +dz*(3*m[2]**4+6*m[0]**2*m[1]**2-1))
        
        A[k,0]=3*(m[1]**2)*(m[2]**2);
        A[k,1]=3*(m[0]**2)*(m[2]**2); 
        A[k,2]=3*(m[0]**2)*(m[1]**2); 
        A[k,3]=res1-res2;
        A[k,4]=res1+res2;
        
    tM[0]=1/D1
    tM[1]=1/D2
    tM[2]=1/D3
    tM[3]=1/D4
    tM[4]=1/D5
    
    BndSpfc='yes'
    
    return tM,A,BndSpfc

#%% Combine two models
def Combined(tMdl1,AMdl1,tMdl2,AMdl2):
    if np.ndim(tMdl1)==np.ndim(AMdl1) and np.ndim(tMdl2)==np.ndim(AMdl2):
        BndSpfc='no'
    else:
        BndSpfc='yes'
    
    nt1=tMdl1.size
    nt2=tMdl2.size
    if np.size(tMdl1)==0:
        tMdl=tMdl2
        AMdl=AMdl2
    elif np.size(tMdl2)==0:
        tMdl=tMdl1
        AMdl=AMdl1
    else:
        tMdl=np.zeros((nt1+1)*(nt2+1)-1)
    
        tMdl[0:nt1]=tMdl1
        tMdl[nt1:nt1+nt2]=tMdl2
        
        
        for k in range(0,nt1):
            for m in range(0,nt2):
                tMdl[nt1+nt2+m+k*nt2]=tMdl1[k]*tMdl2[m]/(tMdl1[k]+tMdl2[m])
                
        AMdl1=np.swapaxes(AMdl1,0,-1)
        AMdl2=np.swapaxes(AMdl2,0,-1)

        
        if AMdl1.shape[1:]!=AMdl2.shape[1:]:
            if AMdl1.ndim>AMdl2.ndim:
                for k in range(1,AMdl1.ndim):
                    AMdl2=np.repeat(np.array([AMdl2.T]),AMdl1.shape[k],axis=k)
            else:
                for k in range(1,AMdl2.ndim):
                    AMdl1=np.repeat(np.array([AMdl1.T]),AMdl2.shape[k],axis=k) 
        
        S21=1-np.sum(AMdl1,axis=0)  
        S22=1-np.sum(AMdl2,axis=0)          



        AMdl=np.zeros(np.concatenate(([(nt1+1)*(nt2+1)-1],AMdl2.shape[1:])).astype(int))
        AMdl[0:nt1]=np.multiply(np.repeat([S22],AMdl1.shape[0],axis=0),AMdl1)
        AMdl[nt1:nt1+nt2]=np.multiply(np.repeat([S21],AMdl2.shape[0],axis=0),AMdl2)
        
        for k in range(0,nt1):
            for m in range(0,nt2):
                AMdl[nt1+nt2+m+k*nt2]=np.multiply(AMdl1[k],AMdl2[m])
        
        AMdl=np.swapaxes(AMdl,0,-1)

    return tMdl,AMdl,BndSpfc
    
def RotVec(euler,vec):
    def Rz(theta):
        return np.array([[np.cos(theta),np.sin(theta),0],[-np.sin(theta),np.cos(theta),0],[0,0,1]])
    def Ry(theta):
        return np.array([[np.cos(theta),0,-np.sin(theta)],[0,1,0],[np.sin(theta),0,np.cos(theta)]])
    
    
    
    return Rz(euler[2]).dot(Ry(euler[1]).dot(Rz(euler[0]).dot(vec.T))).T 
    
    