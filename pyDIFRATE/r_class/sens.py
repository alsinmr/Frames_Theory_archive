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


Created on Tue Apr  2 21:41:57 2019

@author: albertsmith
"""

import numpy as np
import pandas as pd
import pyDIFRATE.r_class.DIFRATE_funs as dff
import matplotlib.pyplot as plt
import pyDIFRATE.r_class.mdl_sens as mdl
#import sens
from pyDIFRATE.tools.DRtools import dipole_coupling
#os.chdir('../plotting')
from pyDIFRATE.plots.plotting_funs import plot_rhoz
#os.chdir('../r_class')

class rates(mdl.model):  
    def __init__(self,tc=None,z=None,**kwargs):
        
        """Probably a better way to do this, but I need to identify which
        child of mdl_sens is which later. Using isinstance requires me to 
        import the children into mdl_sens, but also import mdl_sens into its
        children. This seems to create some strange dependence so that I can't
        actually load any of the classes any more"""
        
        self._class='rates'
        self._origin='rates'
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
        
        "Names of the experimental variables that are available"
        self.__exper=['Type','v0','v1','vr','offset','stdev']
        "Names of the spin system variables that are available"
        self.__spinsys=['Nuc','Nuc1','dXY','CSA','CSoff','QC','eta','theta']

        "Initialize storage for rate constant calculation"
        self.__R=list()
        self.__RCSA=list()
    
        "We need to initialize self.info"
        self.info=None  
        

        "Initialize some storage for rate constant calculation"
#        self.__R=np.zeros([0,np.size(self.__tc)])

#        self.__info=pd.DataFrame(index=self.__exper+self.__spinsys)
        
        super().__init__()
        "Here we feed in all the information on the experiments"
        self.new_exp(**kwargs)
    
    def new_exp(self,info=None,**kwargs):
        """Adds new experiments to a sensitivity object. Options are to input 
        info as a pandas array, with the appropriate values (usually from another)
        sensitivity object), or list the variables directly. 
        
        Experiment: Type, v0, v1, vr, offset, stdev. These may be lists of values,
        in which case multiple experiments will be created.
        
        Spin system: Nuc, Nuc1, dXY, CSA, QC, eta, theta. These must be the same
        for all simultaneously entered experiments. Nuc1 and dXY may have multiple
        values if the Nuc is coupled to multiple other nuclei.
        
        """
        

        
        if info is None:      
            "Count how many experiments are given"
            ne=0
            for k in self.__exper:
                if k in kwargs:
                    ne=np.max([ne,np.size(kwargs.get(k))])
            
            "Add a None type element in self.__R for each new experiment"
            for k in range(0,ne):
                self.__R.append(None)   
                self.__RCSA.append(None)
            
            "Move all input variables to __sys and __exp"
            "Defaults that don't depend on the observed nucleus can be set here"
            self.__exp=dict()
            for k in self.__exper:
                if k in kwargs:
                    self.__exp.update({k : kwargs.get(k)})
                else:
                    self.__exp.update({k : None})
            
            self.__sys=dict()
            for k in self.__spinsys:
                if k in kwargs:
                    self.__sys.update({k : kwargs.get(k)})
                elif k=='Nuc':
                    self.__sys.update({k : '15N'})
                else:
                    self.__sys.update({k : None})
    
            
            self.__cleanup(ne)
            self.__set_defaults(ne)
            
            "Create the new pandas array"
            
            info=pd.concat([pd.DataFrame.from_dict(self.__exp),pd.DataFrame.from_dict(self.__sys)],axis=1).T
        else:  
            ne=info.shape[1]
            for k in range(0,ne):
                self.__R.append(None)   
                self.__RCSA.append(None)       
#        try: #I don't like using try....but not sure what to do here
#            info=pd.concat([info0,info.T],axis=1,ignore_index=True)
#        except:
#            info=info.T
        
        if not isinstance(self.info,pd.DataFrame):
            self.info=info
        else:
            self.info=pd.concat([self.info,info],axis=1,ignore_index=True)
            
        self.del_mdl_calcs()
        
#%% Make sure inputs all are the correct type (numpy arrays)         
    "Function to make sure all inputs are arrays, and have the correct sizes"    
    def __cleanup(self,ne):
        "Check that all experimental variables can be treated as arrays"
        
        for k in self.__exper:
            a=np.atleast_1d(self.__exp.get(k))
            rep=np.ceil(ne/np.size(a))
            a=np.repeat(a,rep)
            a=a[0:ne]
#            a=self.__exp.get(k)
#            if not isinstance(a,(list,np.ndarray,pd.DataFrame)):
#                a=[a]*ne
#            elif np.size(a)!=ne:
#                "We tile the output if the number of experiments doesn't match up"
#                a=a*int(np.ceil(ne/np.size(a)))
#                a=a[0:ne]
#            else:
#            if not isinstance(a,np.ndarray):
#                a=np.array(a)
            self.__exp.update({k:a})
                
        for k in self.__spinsys:
            a=np.atleast_1d(self.__sys.get(k))

            if (k=='dXY' or k=='Nuc1' or k=='CSoff') and np.size(a)>1:
                a=[a]*ne
            else:
                a=[a[0]]*ne

#            a=self.__sys.get(k)
#            
#            if not isinstance(a,(list,np.ndarray)):
#                a=[a]*ne
#            elif k=='dXY' or k=='Nuc1':
#                b=np.array([None]*ne)
#                for m in range(0,ne):
#                    b[m]=a
#                a=b
#            if not isinstance(a,np.ndarray):
#                a=np.array(a)
#            
#            if a.dtype.str[0:2]=='<U':
#                a=a.astype('<U6')
                
            self.__sys.update({k:a})
            
#%% Setting defaults             
    "Function for setting defaults"
    def __set_defaults(self,ne):
        Nuc=self.__sys.get('Nuc')
        for k in range(0,ne):
            if Nuc[k].upper()=='15N' or Nuc[k].upper()=='N15' or Nuc[k].upper()=='N':
                self.__N15_def(k)
            elif Nuc[k].upper()=='13CA' or Nuc[k].upper()=='CA':
                self.__CA_def(k)
            elif Nuc[k].upper()=='13CO' or Nuc[k].upper()=='CO':
                self.__CO_def(k)
            elif Nuc[k].upper()=='CD2H' or Nuc[k].upper()=='13CD2H' or Nuc[k].upper()=='CHD2' or Nuc[k].upper()=='13CHD2':
                self.__CD2H_def(k)
            elif Nuc[k].upper()=='2H' or Nuc[k].upper()=='D':
                self.__2H_def(k)
            elif Nuc[k].upper()=='H217O':
                self.__H217O_def(k)
        
            
            
            for m in self.__spinsys:
                a=self.__sys.get(m)[k]
                if a is None and m!='Nuc1':
                    a=0
                    self.__sys.get(m)[k]=a
                    
            for m in self.__exper:
                a=self.__exp.get(m)[k]
                if np.size(a)==1 and a is None and m!='stdev':
                    a=0
                    self.__exp.get(m)[k]=a
                
    "Function called to set N15 defaults"
    def __N15_def(self,k):
        self.__sys.get('Nuc')[k]='15N'
        if np.size(self.__sys.get('Nuc1')[k])==1 and self.__sys.get('Nuc1')[k] is None:
            self.__sys.get('Nuc1')[k]='1H'
        if (np.size(self.__sys.get('dXY')[k])==1 and np.size(self.__sys.get('dXY')[k])==1) and any([self.__sys.get('dXY')[k] is None]):
            self.__sys.get('dXY')[k]=dipole_coupling(.102,'15N',self.__sys.get('Nuc1')[k])
        if self.__sys.get('CSA')[k] is None:
            self.__sys.get('CSA')[k]=113
        if self.__sys.get('theta')[k] is None:
            self.__sys.get('theta')[k]=23
        
                
    "Function called to set 13CA defaults"
    def __CA_def(self,k):
        self.__sys.get('Nuc')[k]='13C'
        if self.__sys.get('Nuc1')[k] is None:
            self.__sys.get('Nuc1')[k]='1H'
        if self.__sys.get('dXY')[k] is None:
            self.__sys.get('dXY')[k]=dipole_coupling(.1105,'13C',self.__sys['Nuc1'][k])
        if self.__sys.get('CSA')[k] is None:
            self.__sys.get('CSA')[k]=20
        if self.__sys.get('theta')[k] is None:
            self.__sys.get('theta')[k]=0
            
    "Function called to set 13CO defaults"
    def __CO_def(self,k):
        self.__sys.get('Nuc')[k]='13C'

        if self.__sys.get('CSA')[k] is None:
            self.__sys.get('CSA')[k]=155
        if self.__sys.get('theta')[k] is None:
            self.__sys.get('theta')[k]=0
        self.__sys.get('Nuc1')[k]=None
        self.__sys.get('dXY')[k]=0.0   
         
    "Function called to set Methyl CD2H defaults"
    def __CD2H_def(self,k):
        if self.__exp.get('Type')[k]=='NOE':
            self.__sys['Nuc'][k]='13C'
            if self.__sys['Nuc1'][k] is None:
                self.__sys['Nuc1'][k]='1H'
            if self.__sys['dXY'][k] is None:
                self.__sys['dXY'][k]=dipole_coupling(.1115,'13C',self.__sys['Nuc1'][k])
            if self.__sys['CSA'][k] is None:
                self.__sys['CSA'][k]=50 #16.6667*3, see below why we multiply by 3
        else:
            self.__sys.get('Nuc')[k]='13C'
            if self.__sys.get('Nuc1')[k] is None:
                self.__sys.get('Nuc1')[k]=['1H','2H','2H']
            if self.__sys.get('dXY')[k] is None:
                self.__sys.get('dXY')[k]=[dipole_coupling(.1115,'1H','13C'),
                          dipole_coupling(.1110,'2H','13C'),
                          dipole_coupling(.1110,'2H','13C')]
            if self.__sys.get('CSA')[k] is None:
                self.__sys.get('CSA')[k]=50 #16.6667*3
                """
                We've multiplied the desired CSA by 3, because we assume this is methyl rotation
                Then: methyl rotation will induce little relaxation via the CSA,
                but at lower frequencies, where both CSA and the methyl group
                reorientation incude relaxation, we will then have the correct
                relative sizes of the residual tensors
                """
            if self.__sys.get('theta')[k] is None:
                self.__sys.get('theta')[k]=0
            
    "Function called to set 2H defaults (Quadrupole only)"     
    def __2H_def(self,k):
        self.__sys.get('Nuc')[k]='2H'
        if self.__sys.get('dXY')[k] is None:
            self.__sys.get('dXY')[k]=0.0
        if self.__sys.get('CSA')[k] is None:
            self.__sys.get('CSA')[k]=0
        if self.__sys.get('theta')[k] is None:
            self.__sys.get('theta')[k]=0
        if self.__sys.get('QC')[k] is None:
#            self.__sys.get('QC')[k]=170e3
            self.__sys.get('QC')[k]=60104
            
        "Function called to set O17 in water defaults (Quadrupole only)"     
    def __H217O_def(self,k):
        self.__sys.get('Nuc')[k]='17O'
        if self.__sys.get('dXY')[k] is None:
            self.__sys.get('dXY')[k]=0.0
        if self.__sys.get('CSA')[k] is None:
            self.__sys.get('CSA')[k]=0
        if self.__sys.get('theta')[k] is None:
            self.__sys.get('theta')[k]=0
        if self.__sys.get('QC')[k] is None:
            self.__sys.get('QC')[k]=8.2e6

#%% Delete experiment
    def del_exp(self,exp_num):
        
        if np.size(exp_num)>1:
            exp_num=np.array(np.atleast_1d(exp_num))
            exp_num[::-1].sort()    #Crazy, but this sorts exp_num in descending order
            "delete largest index first, because otherwise the indices will be wrong for later deletions"
            for m in exp_num:
                self.del_exp(m)
        else:
            if np.ndim(exp_num)>0:
                exp_num=np.array(exp_num[0])
            self.info=self.info.drop(exp_num,axis=1)
            del self.__R[exp_num]
            del self.__RCSA[exp_num]
                
            self.info.set_axis(np.arange(np.size(self.info.axes[1])),axis=1,inplace=True)
            self._clear_stored(exp_num)
            
#%% Adjust a parameter
    "We can adjust all parameters of a given type, or just one with the experiment index"
    def set_par(self,type,value,exp_num=None):
        if exp_num is None:
            if hasattr(value,'__len__') and len(value)!=1:
                for k in range(self.info.shape[1]):
                    self.info.at[type,k]=value
            else:
                self.info.at[type,:]=value
            self.__R[:]=[None]*len(self.__R)
        else:
            self.info.at[type,exp_num]=value
            self.__R[exp_num]=None
            self.__RCSA[exp_num]=None
         
        self._clear_stored()
        
        self._reset_exp(exp_num)
        
#%% Correlation time axes   
    "Return correlation times or log of correlation times"        
    def tc(self):
        return self.__tc.copy()
    
    def z(self):
        return np.log10(self.__tc)
    
#%% Rate constant calculations
    "Calculate rate constants for given experiment"
    def _rho(self,exp_num=None,bond=None):
        
        if exp_num is None:
            exp_num=self.info.axes[1]
        
        "Make sure we're working with numpy array"
        exp_num=np.atleast_1d(exp_num)
            
        ntc=self.__tc.size
        ne=exp_num.size
        R=np.zeros([ne,ntc])
        for k in range(0,ne):
            "Look to see if we've already calculated this sensitivity, return it if so"
            if self.__R[exp_num[k]] is not None:
                R[k,:]=self.__R[exp_num[k]]
            else:
                "Otherwise, calculate the new sensitivity, and store it"
                R[k,:]=dff.rate(self.__tc,self.info.loc[:,exp_num[k]])
                self.__R[exp_num[k]]=R[k,:]
#                self.__R=np.vstack([self.__R,R[k,:]])
#                self.__info=pd.concat([self.__info,self.info.loc[:,exp_num[k]]],axis=1,ignore_index=True)
              
        
        return R.copy()
    
    def _reset_exp(self,exp_num=None):
        """
        Deletes sensitivity data for a given experiment, or for all experiments.
        Should be run in case a parameter for an experiment is updated.
        """
        if exp_num is None:
            for m in self.__R:
                m=None
        else:
            for k,m in enumerate(self.__R):
                if m is not  None:
                    self.__R[k]=np.delete(m,exp_num,axis=1)
        
    def Reff(self,exp_num=None,mdl_num=0,bond=None,**kwargs):
        R,_=self._rho_eff(exp_num,mdl_num,bond,**kwargs)
        return R

    def R0(self,exp_num=None,mdl_num=None,bond=None,**kwargs):
        _,R0=self._rho_eff(exp_num,mdl_num,bond,**kwargs)
        return R0
    
    def _rhoCSA(self,exp_num,bond=None):
        """Calculates relaxation due to CSA only. We need this function to 
        allow application of anisotropic models, which then have different 
        influence depending on the direction of the interaction tensor. CSA
        points in a different direction (slighlty) than the dipole coupling
        """
        "Make sure we're working with numpy array"
        exp_num=np.atleast_1d(exp_num)            
            

        ntc=self.__tc.size
        ne=exp_num.size
        R=np.zeros([ne,ntc])
        for k in range(0,ne):
            "Get the information for this experiment"
            exper=self.info.loc[:,exp_num[k]].copy()
            "Turn off other interactions"
            exper.at['Nuc1']=None
            exper.at['dXY']=0
            exper.at['QC']=0
            
            "Look to see if we've already calculated this sensitivity, return it if so"
#            count=0
#            test=False
#            n=self.__R.shape[0]
#            while count<n and not test:
#                if self.__info.iloc[:,count].eq(exper).all():
#                    test=True
#                else:
#                    count=count+1
                    
            if self.__RCSA[exp_num[k]] is not None:
                R[k,:]=self.__RCSA[exp_num[k]]
            else:
                "Otherwise, calculate the new sensitivity, and store it"
                R[k,:]=dff.rate(self.__tc,exper)
                self.__RCSA[exp_num[k]]=R[k,:]
#                self.__R=np.vstack([self.__R,R[k,:]])
#                self.__info=pd.concat([self.__info,exper],axis=1,ignore_index=True)
                
            if bond==-1 & self.molecule.vXY.shape[0]>0:
                nb=self.molecule.vXY.shape[0]
                R=np.repeat([R],nb,axis=0)
        return R.copy()
    
    
##%% Plot the rate constant sensitivites
#    def plot_R(self,exp_num=None,ax=None,**kwargs):
#        
#        if exp_num is None:
#            exp_num=self.info.columns.values
#            
#        a=self.R(exp_num).T
#        if 'norm' in kwargs and kwargs.get('norm')[0].lower()=='y':
#            norm=np.max(a,axis=0)
#            a=a/np.tile(norm,[np.size(self.tc()),1])      
#        
#        if ax is None:
#            fig=plt.figure()
#            ax=fig.add_subplot(111)
#            hdl=ax.plot(self.z(),a)
##            ax=hdl[0].axes
#        else:
#            hdl=ax.plot(self.z(),a)
#        
#        self._set_plot_attr(hdl,**kwargs)
#        
#            
#        ax.set_xlabel(r'$\log_{10}(\tau$ / s)')
#        if 'norm' in kwargs and kwargs.get('norm')[0].lower()=='y':
#            ax.set_ylabel(r'$R$ (normalized)')
#        else:
#            ax.set_ylabel(r'$R$ / s$^{-1}$')
#        ax.set_xlim(self.z()[[0,-1]])
#        ax.set_title('Rate Constant Sensitivity (no model)')
#        
#        return hdl
        
    def plot_R(self,exp_num=None,norm=False,ax=None,**kwargs):
        """
        Plots the sensitivites of the experiments. Default plots all experiments 
        without normalization. Set norm=True to normalize all experiments to 1. 
        Specify exp_num to only plot selected experiments. Set ax to specify the
        axis on which to plot
        
        plot_R(exp_num=None,norm=False,ax=None,**kwargs)
        """
        hdl=plot_rhoz(self,index=exp_num,norm=norm,ax=ax,**kwargs)
        ax=hdl[0].axes
        ax.set_ylabel(r'$R / s^{-1}$')
        ax.set_title('Experimental Sensitivities')
        return hdl
        

#%% Return the names of the experiment and sys variables
    def retSpinSys(self):
        return self.__spinsys
    def retExper(self):
        return self.__exper
        
#%% Hidden output of rates (semi-hidden, can be found if the user knows about it ;-) )
    def R(self,exp_num=None):
        """The different children of mdl_sens will have different names for 
        their sensitivities. For example, this class returns R, which are the 
        rate constant sensitivities, but the correlation function class returns
        Ct, and the detector class returns rho. Then, we have a function, 
        _rho(self), that exists and functions the same way in all children
        """
        return self._rho(exp_num)
    
    def Reff(self,exp_num=None,mdl_num=0,bond=None):
        R,R0=self._rho_eff(exp_num,mdl_num,bond)
        
        return R,R0