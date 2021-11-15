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
    X=dict()
    Y=dict()
    Z=dict()
    
    ct=dict()
    
    keys=dict()
    nb=dict()
    nk=dict()
    index=dict()
    
    def __init__(self,vec,nc):
        "Random number for storage of data of this process"
        self.ref_num=np.random.randint(0,1e9)
        self.store_vecs(vec,nc,self.ref_num)
    
    def __enter__(self):
        "Required for use in with statement"
        return self  

    def __exit__(self, exc_type, exc_value, tb):
        self.clear_data(self.ref_num)   #Clears data created by this instance (with self.ref_num)
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
     
    def v0(self):
        "Creates a list of tuples for parallel processing"
        v0=list()
        for k in self.keys[self.ref_num]:
            v0.append((k,self.ref_num))
        return v0
    
    @classmethod    
    def Ct(cls,v):
        i,ref_num=v
        index=cls.index[ref_num]
        X=cls.X[i]
        Y=cls.Y[i]
        Z=cls.Z[i]
        n=np.size(index)
        
        "Delete data out of the dictionary after stored here"
        cls.clearXYZ(i)
        
        ct=np.zeros([index[-1]+1,np.shape(X)[1]])
        
        for k in range(n):
            ct[index[k:]-index[k]]+=(3*(np.multiply(X[k:],X[k])+np.multiply(Y[k:],Y[k])\
                 +np.multiply(Z[k:],Z[k]))**2-1)/2
        
        "Store results of correlation function calculation"
#        cls.storeCt(i,ct)
        cls.ct[i]=ct
        return ct
#    
    @classmethod
    def store_vecs(cls,vec,nc,ref_num):
        """Responsible for sorting out the vectors for each process.
        Uses class variables, which are effectively global, but indexes them randomly
        so that we shouldn't end up accessing the same variables in multiple processes
        """
        
        nk=nc   #Maybe we should change this to reduce memory usage. Right now just nc steps
        
        """nc is the number of cores to be used, and nk the number of chunks to
        do the calculation in. Currently equal.
        """
        
        cls.keys[ref_num]=ref_num+np.arange(nk) #Keys where the data is stored
        cls.nb[ref_num]=vec['X'].shape[1]   #Number of correlation functions (n bonds)
        cls.nk[ref_num]=nk  #Number of chunks
        cls.index[ref_num]=vec['index'] #Index of frames taken
        nb=cls.nb[ref_num]  
        for k,i in enumerate(cls.keys[ref_num]):     #Separate and store parts of the vector
            cls.X[i]=vec['X'][:,range(k,nb,nk)]
            cls.Y[i]=vec['Y'][:,range(k,nb,nk)]
            cls.Z[i]=vec['Z'][:,range(k,nb,nk)]
            
        return ref_num
    
    @classmethod
    def clearXYZ(cls,i):
        "Responsible for deleting vectors for a given job"
        del cls.X[i]
        del cls.Y[i]
        del cls.Z[i]
        
#    @classmethod
#    def storeCt(cls,ref0,ct):
#        cls.ct[ref0]=ct
         
    def returnCt(self,ct):
        ref_num=self.ref_num
        nk=self.nk[ref_num]
        keys=range(nk)
        index=self.index[ref_num]
        N0=get_count(index)
        nz=N0!=0
        N0=N0[nz]
        nb=self.nb[ref_num]
        
        ct0=np.zeros([np.size(N0),nb])
        for k,c in enumerate(ct):
            N=np.repeat([N0],np.shape(c)[1],axis=0).T
            ct0[:,range(k,nb,nk)]=np.divide(c[nz],N)
    
        return ct0
    
    @classmethod
    def clear_data(cls,ref_num):
        locs=['X','Y','Z','ct']
        for ref0 in cls.keys[ref_num]:
            for loc in locs:
                if ref0 in getattr(cls,loc):
                    del getattr(cls,loc)[ref0]
        
        locs=['keys','nb','nk','index']
        for loc in locs:
            if ref_num in getattr(cls,loc):
                del getattr(cls,loc)[ref_num]


#%% Determine how many frame pairs are averaged into each time point
def get_count(index):
    """
    Returns the number of averages for each time point in the sparsely sampled 
    correlation function
    """
    N=np.zeros(index[-1]+1)
    n=np.size(index)
   
    for k in range(n):
        N[index[k:]-index[k]]+=1
        
    return N