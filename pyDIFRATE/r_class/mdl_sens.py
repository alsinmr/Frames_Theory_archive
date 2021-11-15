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



Created on Wed Apr  3 22:07:08 2019

@author: albertsmith
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import copy
import pyDIFRATE.r_class.DynamicModels as dm
#import os
#os.chdir('../Struct')
from pyDIFRATE.Struct.structure import molecule
#os.chdir('../plotting')
from pyDIFRATE.plots.plotting_funs import plot_rhoz
#os.chdir('../r_class')
#import detectors
from scipy.interpolate import interp1d as interp


class model(object):
    def __init__(self):
        
#        if self._class=='Ct':
#            self.__Reff=np.zeros([0,np.size(self.t()),np.size(self.tc())])
#            self.__R0=np.zeros([0,np.size(self.t())])
#        else:
#            self.__Reff=np.zeros([0,np.size(self.tc())])
#            self.__R0=np.zeros([0,1])
    
#        self.__mdlinfo=pd.DataFrame(index=self.retExper()+self.retSpinSys())
#        self.__tMdl=list()
#        self.__AMdl=list()
        
        self.__Reff=list()
        self.__R0=list()
        self.__ReffCSA=list()
        self.__R0CSA=list()
        
        self.MdlPar=list()
        self.tMdl=list()
        self.AMdl=list()
        self.molecule=molecule()
        
            
    def new_mdl(self,tMdl=None,AMdl=None,Model=None,**kwargs):
        
        if tMdl is not None and AMdl is not None:
            tMdl=np.atleast_1d(tMdl)
            AMdl=np.atleast_1d(AMdl)
            
            if AMdl.ndim==3:
                MdlPar=dict(Model='Direct',BondSpfc='yes')
            elif AMdl.ndim==1:
                MdlPar=dict(Model='Direct',BondSpfc='no')
            else:
                print('AMdl must be a single value, a 1D array, or a 3D array')
                return
            
            self.MdlPar.append(MdlPar)
            self.tMdl.append(tMdl)
            self.AMdl.append(AMdl)
        elif Model=='Combined' and 'mdl_nums' in kwargs:
            
            mdl_nums=kwargs.get('mdl_nums')
            if not isinstance(mdl_nums,np.ndarray):
                mdl_nums=np.array(mdl_nums)
            if mdl_nums.shape==():
                mdl_nums=np.array([mdl_nums])
            
            BndSpfc='no'
            Models=list()
            "Maybe we can determine bond specificity inside of the Combined function? (below)"
            for k in mdl_nums:
                Models.append(self.MdlPar[k])
                if self.MdlPar[k]['BondSpfc']=='yes':
                    BndSpfc='yes'
            
            MdlPar=dict(Model='Combined',BondSpfc=BndSpfc,SubModels=Models)
            
            
            tMdl=np.array([]);     #Empty model
            AMdl=np.array([]);   #
            for k in mdl_nums:
                tMdl,AMdl,_=dm.ModelSel('Combined',tMdl1=tMdl,AMdl1=AMdl,tMdl2=self.tMdl[k],AMdl2=self.AMdl[k])
            
            self.MdlPar.append(MdlPar)
            self.tMdl.append(tMdl)
            self.AMdl.append(AMdl)
            
        else:
#            if dm.ModelBondSpfc(Model) and self.molecule.vXY.size==0:
#                print('Before defining an model with anisotropic motion, import a structure and select the desired bonds')
#            else:
            tMdl,AMdl,BndSp=dm.ModelSel(Model,'dXY',self.molecule,**kwargs)
#                if BndSp=='yes' and self._class!='Ct':
            if BndSp=='yes':
                _,A,_=dm.ModelSel(Model,'dCSA',self.molecule,**kwargs)
                AMdl=[AMdl,A]
                AMdl=np.swapaxes(AMdl,0,1)
                
            MdlPar=dict(Model=Model,BondSpfc=BndSp,**kwargs)
            
            self.MdlPar.append(MdlPar)
            self.tMdl.append(tMdl)
            self.AMdl.append(AMdl)
    
        self.__Reff.append(None)
        self.__R0.append(None)
        self.__ReffCSA.append(None)
        self.__R0CSA.append(None)
        
            
    def del_mdl(self,mdl_num):
        del self.AMdl[mdl_num]
        del self.tMdl[mdl_num]
        del self.MdlPar[mdl_num]
        del self.__Reff[mdl_num]
        del self.__R0[mdl_num]
        del self.__ReffCSA[mdl_num]
        del self.__R0CSA[mdl_num]
        
    def del_mdl_calcs(self):
        self.__Reff=list(np.repeat(None,np.size(self.MdlPar)))
        self.__R0=list(np.repeat(None,np.size(self.MdlPar)))
        self.__ReffCSA=list(np.repeat(None,np.size(self.MdlPar)))
        self.__R0CSA=list(np.repeat(None,np.size(self.MdlPar)))
    
    def _rho_eff(self,exp_num=None,mdl_num=0,bond=None,**kwargs):
        """This function is mostly responsible for searching for a pre-existing
        calculation of the model and experiment
        """
        
#        if bond==-1:
#            bond=None
           
        if len(self.MdlPar)==0:  #If no models present, set mdl_num to None
            mdl_num=None
        
        if exp_num is not None:
            exp_num=np.atleast_1d(exp_num)
            
        if (mdl_num is None) or (mdl_num==-1):
            R=self._rho(exp_num,bond)
            R0=np.zeros(R.shape[0:-1])
            return R,R0
        
        if self.__Reff[mdl_num] is None:
            Reff,R0,ReffCSA,R0CSA=self.__apply_mdl(self.tMdl[mdl_num],self.AMdl[mdl_num])
            self.__Reff[mdl_num]=Reff
            self.__R0[mdl_num]=R0
            self.__ReffCSA[mdl_num]=ReffCSA
            self.__R0CSA[mdl_num]=R0CSA
            
        if np.shape(self.__Reff[mdl_num])[0]==1:
            bond=None
        
        if exp_num is None and (bond is None or bond==-1):
            R=self.__Reff[mdl_num]
            R0=self.__R0[mdl_num]
        elif exp_num is None:
            R=self.__Reff[mdl_num][bond,:,:]
            R0=self.__R0[mdl_num][bond,:]
        elif bond is None or bond==-1:
            R=self.__Reff[mdl_num][:,exp_num,:]
            R0=self.__R0[mdl_num][:,exp_num]
        else:
            R=self.__Reff[mdl_num][bond,exp_num,:]
            R0=self.__R0[mdl_num][bond,exp_num]
            
        if R.shape[0]==1:
            R=R[0]
            R0=R0[0]
        
            
        return R.copy(),R0.copy()
        
    
    def _rho_effCSA(self,exp_num=None,mdl_num=0,bond=None):
        """Same as above, but only for the CSA interaction
        """
        
#        if bond==-1:
#            bond=None
            
        if exp_num is not None:
            exp_num=np.atleast_1d(exp_num)
        
        if (mdl_num is None) or (mdl_num==-1):
            R=self._rhoCSA(exp_num,bond)
            R0=np.zeros(R.shape[0:-1])
            return R,R0
        
        if self.__ReffCSA[mdl_num] is None:
            Reff,R0,ReffCSA,R0CSA=self.__apply_mdl(self.tMdl[mdl_num],self.AMdl[mdl_num])
            self.__Reff[mdl_num]=Reff
            self.__R0[mdl_num]=R0
            self.__ReffCSA[mdl_num]=ReffCSA
            self.__R0CSA[mdl_num]=R0CSA
            
        if np.shape(self.__ReffCSA[mdl_num])[0]==1:
            bond=None
        
        if exp_num is None and (bond is None or bond==-1):
            R=self.__ReffCSA[mdl_num]
            R0=self.__R0CSA[mdl_num]
        elif exp_num is None:
            R=self.__ReffCSA[mdl_num][bond,:,:]
            R0=self.__R0CSA[mdl_num][bond,:]
        elif bond is None or bond==-1:
            R=self.__ReffCSA[mdl_num][:,exp_num,:]
            R0=self.__R0CSA[mdl_num][:,exp_num]
        else:
            R=self.__ReffCSA[mdl_num][bond,exp_num,:]
            R0=self.__R0CSA[mdl_num][bond,exp_num]
            
        if R.shape[0]==1:
            R=R[0]
            R0=R0[0]
        
            
        return R.copy(),R0.copy()
    
    def __apply_mdl(self,tMdl,A):
        "tMdl is a list of correlation times in the model, and A the amplitudes"
        "Note that if A does not add to 1, we assume that S2 is non-zero (S2=1-sum(A))"
        
        
        "Get the experimental sensitivities"
        R=self._rho(self.info.columns,bond=-1)
        RCSA=self._rhoCSA(self.info.columns,bond=-1)
        
        R+=-RCSA #We operate on relaxation from dipole and CSA separately
        
        "Shapes of matrices, preallocation"
        SZA=np.shape(A)
        if np.size(SZA)>1:
            SZA=SZA[0]
            iso=False
        else:
            iso=True
            SZA=1
        
        "We repeat R and RCSA for every bond in A if R and RCSA are not already bond specific"
        SZR=R.shape

        if np.size(SZR)==3:
            if iso:
                A=np.repeat([np.repeat([A],2,axis=0)],SZR[0],axis=0)
                SZA=SZR[0]
                iso=False
            SZR=SZR[1:]
        else:
            R=np.repeat([R],SZA,axis=0)
            RCSA=np.repeat([RCSA],SZA,axis=0)
        
        SZeff=np.concatenate([np.atleast_1d(SZA),np.atleast_1d(SZR)])
        SZ0=np.concatenate([np.atleast_1d(SZA),[SZR[0]]])
        
        "Contributions to relaxation coming from model with non-zero S2"
        if np.ndim(A)>1:
            S2=1-np.sum(A[:,0,:],axis=1)
            S2CSA=1-np.sum(A[:,1,:],axis=1)
        else:
            S2=[1-np.sum(A)]
            S2CSA=S2
            
        SZ1=[SZeff[0],np.prod(SZeff[1:])]

        """
        The order parameter of the input model yields the fraction of the model correlation that does not
        change the internal effective correlation time. 
        """
        Reff=np.multiply(np.repeat(np.transpose([S2]),SZ1[1],axis=1),np.reshape(R,SZ1))
        ReffCSA=np.multiply(np.repeat(np.transpose([S2CSA]),SZ1[1],axis=1),np.reshape(RCSA,SZ1))
        Reff=np.reshape(Reff,SZeff)
        ReffCSA=np.reshape(ReffCSA,SZeff)
        
        R0=np.zeros(SZ0)
        R0CSA=np.zeros(SZ0)
        
        "Loop over all correlation times in model"
        for k,tc in enumerate(tMdl):
            "Matrix to transform from z to zeff (or simply to evaluate at z=log10(tc) with M0)"
            M,M0=self.z2zeff(tc)
            
            
            SZ1=[np.prod(SZeff[0:2]),SZeff[2]]
            R00=np.dot(M0,np.reshape(R,SZ1).T)
            R0CSA0=np.dot(M0,np.reshape(RCSA,SZ1).T)
                
            Reff0=np.reshape(np.dot(M,np.reshape(R,SZ1).T).T-np.transpose([R00]),[1,np.prod(SZeff)])
            ReffCSA0=np.reshape(np.dot(M,np.reshape(RCSA,SZ1).T).T-np.transpose([R0CSA0]),[1,np.prod(SZeff)])
            if iso:
                Reff+=A[k]*np.reshape(Reff0,SZeff)
                R0+=A[k]*np.reshape(R00,SZ0)
                ReffCSA+=A[k]*np.reshape(ReffCSA0,SZeff)
                R0CSA+=A[k]*np.reshape(R0CSA0,SZ0)
            else:
                A0=A[:,0,k]
                Reff+=np.reshape(np.multiply(np.repeat(np.transpose([A0]),np.prod(SZR)),Reff0),SZeff)
                R0+=np.reshape(np.multiply(np.repeat(np.transpose([A0]),SZR[0]),R00),SZ0)
                
                A0=A[:,1,k]
                ReffCSA+=np.reshape(np.multiply(np.repeat(np.transpose([A0]),np.prod(SZR)),ReffCSA0),SZeff)
                R0CSA+=np.reshape(np.multiply(np.repeat(np.transpose([A0]),SZR[0]),R0CSA0),SZ0)
                
        
        Reff+=ReffCSA
        R0+=R0CSA
        return Reff,R0,ReffCSA,R0CSA
        
    def z2zeff(self,tc):
        
        z=self.z()
        zeff=z+np.log10(tc)-np.log10(tc+10**z)  #Calculate the effective log-correlation time
        zeff[zeff<=z[0]]=z[0]+1e-12                    #Cleanup: no z shorter than z[0]
        zeff[zeff>=z[-1]]=z[-1]-1e-12           #Cleanup: no z longer than z[-1]
        i=np.digitize(zeff,z,right=False)-1     #Index to find longest z such that z<zeff
        sz=np.size(z)
        M=np.zeros([sz,sz])                     #Pre-allocate Matrix for rho->rho_eff transform
        
        dz=z[1:]-z[0:-1]
        wt=(z[i+1]-zeff)/dz[i]
        M[np.arange(0,sz),i]=wt
        M[np.arange(0,sz),i+1]=1-wt
        
        zi=np.log10(tc)                        #Calculate the log of input tc
        if zi<=z[0]:
            zi=z[0]+1e-12                     #Cleanup: no z shorter than z[0]
        if zi>=z[-1]:
            zi=z[-1]-1e-12
        i=np.digitize(zi,z,right=False)-1     #Index to find longest z such that z<zeff
        
        M0=np.zeros([sz])                     #Pre-allocate Matrix for rho->rho_eff transform
        
        wt=(z[i+1]-zi)/dz[i]
        M0[i]=wt
        M0[i+1]=1-wt
        
        return M,M0
    
#    def detect(self,exp_num=None,mdl_num=None):
#        """
#        r=self.detect(exp_num=None,mdl_num=None)
#        Creates a detector object from the current sensitivity object. can
#        specifiy particular experiments and models to use (default is all
#        experiments) and no model
#        """
#        
#        r=detectors.detect(self,exp_num,mdl_num)
#        
#        return r
        
    def __temp_exper(self,exp_num,inter):
        """When we calculate dipole/CSA relaxation under a bond-specific model 
        (ex. Anisotropic diffusion), we actually need to apply a different 
        model to the motion of the CSA and dipole. To do this, we create a new 
        experiment without the dipole, or without the CSA, calculate its 
        sensitivity, and then delete the experiment from the users's scope 
        after we're done with it. This gets passed back to _rho_eff, where the 
        new model is applied to experiments with CSA and dipole separately"""
        
        exper=self.info.loc[:,exp_num].copy()
        if inter=='dXY':
            exper.at['CSA']=0
        else:
            exper.at['dXY']=0
            exper.at['QC']=0    
            """We should make sure the quadrupole doesn't count twice. I guess
            this shouldn't matter, because we usually neglect dipole and CSA
            relaxation when a quadrupole is present, but if the user puts them
            in for some reason, it would result in a double-counting of the
            quadrupole relaxation"""
        self.new_exp(info=exper)  #Add the new experiment
        n=self.info.columns.values[-1] #Index of the new experiment
        R=self._rho(n)
        
        self.del_exp(n) #Delete the experiment to hide this operation from the user
        
        return R 

    def _clear_stored(self,exp_num=None):
        "Unfortunately, we only have methods to apply models to all experiments at once"
        "This means a change in the experiment list requires recalculation of all models for all experiments"
        "This function deletes all model calculations"
        

        if exp_num is None:
            for m in self.__Reff:
                m=None
            for m in self.__ReffCSA:
                m=None
            for m in self.__R0:
                m=None
            for m in self.__R0CSA:
                m=None
        else:    
            for k,m in enumerate(self.__Reff):
                if m is not None:
                    self.__Reff[k]=np.delete(m,exp_num,axis=1)
            for k,m in enumerate(self.__ReffCSA):
                if m is not None:
                    self.__ReffCSA[k]=np.delete(m,exp_num,axis=1)
            for k,m in enumerate(self.__R0):
                if m is not None:
                    self.__R0[k]=np.delete(m,exp_num,axis=1)
            for k,m in enumerate(self.__R0CSA):
                if m is not None:
                    self.__R0CSA[k]=np.delete(m,exp_num,axis=1)

                
    def zeff(self,t,tau=None):
        if tau==None:
            return self.z()+np.log10(t)-np.log10(10**self.z()+t)
        else:
            return np.log10(t)+np.log10(tau)-np.log10(t+tau)
        
#    def plot_eff(self,exp_num=None,mdl_num=0,bond=None,ax=None,**kwargs):
#        
#        if bond==-1:
#            bond=None
#        
#        a,b=self._rho_eff(exp_num,mdl_num,bond)
#        
#        if bond is None and np.size(a.shape)==3:
#            maxi=np.max(a,axis=0)
#            mini=np.min(a,axis=0)
#            a=np.mean(a,axis=0)
#            pltrange=True
#            maxi=maxi.T
#            mini=mini.T
#        else:
#            pltrange=False
#        
#        a=a.T
#
#        if 'norm' in kwargs and kwargs.get('norm')[0].lower()=='y':
#            norm=np.max(np.abs(a),axis=0)
#            a=a/np.tile(norm,[np.size(self.tc()),1])      
#            
#            if pltrange:
#                maxi=maxi/np.tile(norm,[np.size(self.tc()),1])
#                mini=mini/np.tile(norm,[np.size(self.tc()),1])
#        
#        if ax==None:
#            fig=plt.figure()
#            ax=fig.add_subplot(111)
#            hdl=ax.plot(self.z(),a)
##            hdl=plt.plot(self.z(),a)
##            ax=hdl[0].axes
#        else:
#            hdl=ax.plot(self.z(),a)
#            
#        if pltrange:
#            x=np.concatenate([self.z(),self.z()[-1::-1]],axis=0)
#            for k in range(0,a.shape[1]):
#                y=np.concatenate([mini[:,k],maxi[-1::-1,k]],axis=0)
#                xy=np.concatenate(([x],[y]),axis=0).T
#                patch=Polygon(xy,facecolor=hdl[k].get_color(),edgecolor=None,alpha=0.5)
#                ax.add_patch(patch)
#            
#            
#        ax.set_xlabel(r'$\log_{10}(\tau$ / s)')
#        if 'norm' in kwargs and kwargs.get('norm')[0].lower()=='y':
#            ax.set_ylabel(r'$R$ (normalized)')
#        else:
#            ax.set_ylabel(r'$R$ / s$^{-1}$')
#        ax.set_xlim(self.z()[[0,-1]])
#        ax.set_title('Sensitivity for Model #{0}'.format(mdl_num))
#        
#        return hdl
            
    def plot_eff(self,exp_num=None,mdl_num=0,bond=None,ax=None,norm=False,**kwargs):
        if bond==-1:
            bond=None
            
        hdl=plot_rhoz(self,index=exp_num,mdl_num=mdl_num,norm=norm,ax=ax,bond=bond,**kwargs)
        ax=hdl[0].axes
        ax.set_title('Sensitivity for Model #{0}'.format(mdl_num))
           
        return hdl
    
    def _set_plot_attr(self,hdl,**kwargs):
        props=hdl[0].properties().keys()
        for k in kwargs:
            if k in props:
                for m in hdl:
                    getattr(m,'set_{}'.format(k))(kwargs.get(k))
                    
    def copy(self,type='deep'):
        """
        |
        |Returns a copy of the object. Default is deep copy (all objects except the molecule object)
        | obj = obj0.copy(type='deep')
        |To also create a copy of the molecule object, set type='ddeep'
        |To do a shallow copy, set type='shallow'
        """
        if type=='ddeep':
            out=copy.deepcopy(self)
        elif type!='deep':
            out=copy.copy(self)
        else:
            mol=self.molecule
            self.molecule=None
            out=copy.deepcopy(self)
            self.molecule=mol
            out.molecule=mol
            
        return out
        
        