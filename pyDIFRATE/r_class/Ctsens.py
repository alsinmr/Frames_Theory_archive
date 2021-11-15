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


Created on Wed Apr 10 16:49:23 2019

@author: albertsmith
"""

import numpy as np
import pandas as pd
#import DIFRATE_funs as dff
#import matplotlib.pyplot as plt
import pyDIFRATE.r_class.mdl_sens as mdl
import os
#os.chdir('../iRED')
from pyDIFRATE.iRED.fast_index import trunc_t_axis,get_count
#os.chdir('../r_class')

class Ct(mdl.model):
    def __init__(self,tc=None,z=None,t=None,**kwargs):
        
        """Probably a better way to do this, but I need to identify which
        child of mdl_sens is which later. Using isinstance requires me to 
        import the children into mdl_sens, but also import mdl_sens into its
        children. This seems to create some strange dependence so that I can't
        actually load any of the classes any more"""
        
        self._class='Ct'
        self._origin='Ct'
        
        """The detectors class may have bond-specific sensitivities in _rho. We
        need to know if this is the case for the mdl_sens class to work 
        properly
        """
        self._BondSpfc='no'
        
        """Get user defined tc if provided. Options are to provide the tc 
        vector directly, to provide the log directly (define z instead of tc), 
        or to specify it as a start, end, and number of steps, which are then 
        log-spaced (3 entries for tc or z)
        """
    
        if tc is None:
            if z is not None:
                if np.size(z)==3:
                    self.__tc=np.logspace(z[0],z[1],z[2])
                else:
                    self.__tc=np.power(10,z)
                "Allow users to input z instead of tc"
            else:
                self.__tc=np.logspace(-14,-3,200)
        elif np.size(tc)==3:
            self.__tc=np.logspace(np.log10(tc[0]),np.log10(tc[1]),tc[2])
        else:
            self.__tc=np.array(tc)
        """We don't allow editing of the tc vector; you must initialize a new 
        instance of rates if you want to change it"""
        
        
        """If you want to edit the code to include new experiments, and these 
        require new variables, they MUST be added to one of these lists
        """
        
        "We need to initialize self.info"
        self.info=None  
    
        a=dict()
        if t is not None:
            if np.size(t)==3:
                self.__t=np.arange(t[0],t[1],t[2])    
            elif np.size(t)==2:
                self.__t=np.arange(0,t[0],t[1])
            else:
                self.__t=t
        elif 'sparse' in kwargs:
            "Include nt, n, nr and dt in dict object"
            sparse=kwargs.get('sparse')
            if 'dt' not in sparse or 'nt' not in sparse:
                print('dt and nt are required arguments for generating a sparse sensitivity object')
                return
            index=trunc_t_axis(**sparse)
            
            "Get the count of number of averages"            
            N=get_count(index)
                
            t=sparse.get('dt')*np.arange(index[-1]+1)
            i=N!=0
            N=N[i]
            self.__t=t[i]
            
            if 'stdev' not in kwargs:
                stdev=1/np.sqrt(N)
                stdev[0]=1e-6
                kwargs.update({'stdev':stdev})
        else:
            self.__t=np.arange(0,500.001,.005)
            
        a.update({'t' : self.__t})
        
        nt=self.__t.size
        
        if 'stdev' in kwargs:
            stdev=kwargs.get('stdev')
            if np.size(stdev)==1:
                vec=1/np.sqrt(np.arange(nt,0,-1))
                vec=vec/vec[0]
                stdev=vec*stdev
                stdev[0]=1e-6
            elif np.size(stdev)!=np.size(self.__t):
                vec=1/np.sqrt(np.arange(nt,0,-1))
                stdev=vec/vec[-1]
                stdev[0]=1e-6
        else:
            vec=1/np.sqrt(np.arange(nt,0,-1))
            stdev=vec/vec[-1]
            stdev[0]=1e-6
        
        a.update({'stdev' : stdev})
        
        if 'median_val' in kwargs:
            median_val=kwargs.get('median_val')
            if np.size(median_val)==1:
                median_val=np.ones(nt)*median_val
        else:
            median_val=np.ones(nt)
        a.update({'median_val' : median_val})
        
        self.info=pd.DataFrame.from_dict(a).T  
        
        
        if 'S2' in kwargs:
#            self.__R=np.exp(-1e-9*np.dot(np.atleast_2d(self.__t).T,1/np.atleast_2d(self.__tc)))\
#                -np.repeat([np.exp(-1e-9*self.__t[-1]/self.__tc)],self.__t.shape[0],axis=0)
            "Note the new formula for sensitivity after S2 subtraction. Based on Poisson distribution"
            T=self.__t[-1]*1e-9 #Length of the trajectory
            Lambda=1./(2.*self.__tc) #Constant for Poisson distribution
            self.__R=np.exp(-1e-9*np.dot(np.atleast_2d(self.__t).T,1/np.atleast_2d(self.__tc)))\
                -np.repeat([1./(T*Lambda)*(1-np.exp(-T*Lambda))],self.__t.shape[0],axis=0)
        else:
            self.__R=np.exp(-1e-9*np.dot(np.atleast_2d(self.__t).T,1/np.atleast_2d(self.__tc)))
#            self.__R=np.exp(-1e-9*np.dot(np.transpose([self.__t]),np.divide(1,[self.__tc])))
        "Names of the experimental variables that are available"
        self.__exper=['t','stdev']
        "Names of the spin system variables that are available"
        self.__spinsys=[]
       
        super().__init__()
        
    def Ct(self,exp_num=None,**kwargs):
        
        if exp_num is None:
            exp_num=self.info.columns.values
        
        "Make sure we're working with numpy array for exp_num"
        if not isinstance(exp_num,np.ndarray):
            exp_num=np.array(exp_num)
        if exp_num.shape==():
            exp_num=np.array([exp_num])
        "Make sure we're working with numpy array"
        if not isinstance(exp_num,np.ndarray):
            exp_num=np.array(exp_num)
        if exp_num.shape==():
            exp_num=np.array([exp_num])
            
        return self.__R[exp_num,:]

    def t(self):
        return self.__t
    
    def tc(self):
        return self.__tc.copy()
    
    def z(self):
        return np.log10(self.__tc)
    
    def retExper(self):
        return self.__exper
    
    def retSpinSys(self):
        return self.__spinsys
 
    #%% Hidden output of rates (semi-hidden, can be found if the user knows about it ;-) )
    def _rho(self,exp_num=None,bond=None):
        """The different children of mdl_sens will have different names for 
        their sensitivities. For example, this class returns R, which are the 
        rate constant sensitivities, but the correlation function class returns
        Ct, and the detector class returns rho. Then, we have a function, 
        __rho(self), that exists and functions the same way in all classes
        """
        
        if exp_num is None:
            exp_num=self.info.columns.values
        
        R=self.Ct(exp_num)
        
        
        return R
    
    def _rhoCSA(self,exp_num=None,bond=None):
        
        if exp_num is None:
            exp_num=self.info.columns.values
            
        if bond==-1 & self.molecule.vXY.shape[0]>0:
            nb=self.molecule.vXY.shape[0]
            R=np.zeros([nb,np.size(exp_num),np.size(self.__tc)])
        else:
            R=np.zeros([np.size(exp_num),np.size(self.__tc)])    
            
            
        return R
    
    def Cteff(self,exp_num=None,mdl_num=0,bond=None):
        R,R0=self._rho_eff(exp_num,mdl_num,bond)
        
        return R,R0