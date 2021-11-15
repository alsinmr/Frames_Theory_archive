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

Created on Thu Aug 29 12:36:42 2019

@author: albertsmith
"""
import numpy as np
import traceback
from time import time
from pyDIFRATE.iRED.fast_index import get_count

#%% Parallel class for fast parallel calculation
class par_class():
    """
    We create this class to perform parallel calculation of correlation functions.
    The variables required for calculation are stored as *class* variables, which
    are dictionaries. Because they are class variables, these are available to
    any instance of par_class (that is, if multiple par_class *objects* are 
    created by different processes, they will all have access to these dicts. To
    make problems unlikely, we assign the dictionary keys using a random number.
    That number is passed out to the parent process, and then used to find
    the correct data later. In principle, although different processes are 
    looking into the same dictionaries, they should avoid using/editing the same
    data)
    """
    aqt=dict()
    
    keys=dict()
    nb=dict()
    nk=dict()
    index=dict()
        
    @classmethod    
    def Ct(cls,v):
        i,ref_num=v
        index=cls.index[ref_num]
        aqt=cls.aqt[i]
        n=np.size(index)
        
        ct=dict()
        
        nb=cls.nb[ref_num]
        
        for p,a in enumerate(aqt.values()):
            if p==0:
                ct=np.zeros([index[-1]+1,a.shape[1]])+0j
            for k in range(n):
                ct[index[k:]-index[k]]+=np.multiply(a[k:],a[k].conjugate())
        return ct.real
#    
    @classmethod
    def store_vecs(cls,aqt,nc):
        """Responsible for sorting out the vectors for each process.
        Uses class variables, which are effectively global, but indexes them randomly
        so that we shouldn't end up accessing the same variables in multiple processes
        """
        
        nk=nc   #Maybe we should change this to reduce memory usage. Right now just nc steps
        
        """nc is the number of cores to be used, and nk the number of chunks to
        do the calculation in. Currently equal.
        """
        
        ref_num=np.random.randint(0,1e9) 
        
        cls.keys[ref_num]=ref_num+np.arange(nk) #Keys where the data is stored
        if '1,0' in aqt:
            cls.nb[ref_num]=aqt['1,0'].shape[1]
        elif '2,0' in aqt:
            cls.nb[ref_num]=aqt['2,0'].shape[1]
        cls.nk[ref_num]=nk  #Number of chunks
        cls.index[ref_num]=aqt['index'] #Index of frames taken
        
        nb=cls.nb[ref_num]
        
        for k,i in enumerate(cls.keys[ref_num]):     #Separate and store parts of the vector
            cls.aqt[i]=dict()
            for m,a in aqt.items():
                if m!='t' and m!='index':
                    cls.aqt[i][m]=a[:,range(k,nb,nk)]
            
        v0=list()    
        for i in cls.keys[ref_num]:
            v0.append((i,ref_num))
        
        return ref_num,v0
    
    @classmethod
    def returnCt(cls,ref_num,ct):
        "Still needs updated"
        nk=cls.nk[ref_num]
        index=cls.index[ref_num]
        N0=get_count(index)
        nz=N0!=0
        N0=N0[nz]
        nb=cls.nb[ref_num]
        
        ct0=np.zeros([np.size(N0),nb])
        for k,c in enumerate(ct):
            N=np.repeat([N0],np.shape(c)[1],axis=0).T
            ct0[:,range(k,nb,nk)]=np.divide(c[nz],N)
    
        return ct0
    
    @classmethod
    def clear_data(cls,ref_num):
        locs=['aqt','ct']
        if ref_num in cls.keys:
            for ref0 in cls.keys[ref_num]:
                if ref0 in cls.aqt:
                    del cls.aqt[ref0]
        else:
            print('Data already deleted')
            
        locs=['keys','nb','nk','index']
        for loc in locs:
            if ref_num in getattr(cls,loc):
                del getattr(cls,loc)[ref_num]

    @classmethod
    def _clear_all(cls):
        while len(cls.aqt.keys())!=0:
            try:
                k=list(cls.aqt.keys())
                cls.clear_data(k[0])
            except:
                pass
                