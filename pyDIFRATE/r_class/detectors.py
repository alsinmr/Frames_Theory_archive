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


Created on Thu Apr 11 20:32:25 2019

@author: albertsmith
"""
#import os
#cwd=os.getcwd()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib.patches import Polygon
import pyDIFRATE.r_class.mdl_sens as mdl
from numpy.linalg import svd
#from scipy.sparse.linalg import svds
from scipy.sparse.linalg import eigs
from scipy.optimize import linprog
from scipy.optimize import lsq_linear as lsqlin
from pyDIFRATE.tools.DRtools import linear_ex
import multiprocessing as mp
import warnings
#os.chdir('../plotting')
import pyDIFRATE.plots.plotting_funs as pf
#os.chdir(cwd)

warnings.filterwarnings("ignore",r"Ill-conditioned matrix*")
warnings.filterwarnings("ignore",r"Solving system with option*")

class detect(mdl.model):
    def __init__(self,sens,exp_num=None,mdl_num=None):
        """ We initiate the detectors class by giving it a sens/Ctsens class, from
        which it extracts the specified experiments and models for each 
        experiment.
        
        I've replaced the normalization here with 1/max of the sensitivity, and
        then using a relative standard deviation, as opposed to the absolute 
        standard deviation. The question is- do we want to minimize the 
        """

        self.n=None;
        self._class='detector'
        self._origin=sens._origin
        
        self.__tc=sens.tc()
        ntc=np.size(self.__tc)
        """This is the maximum number of singular values to return if not
        specified. Probably its higher than necessary.
        """
        self.__maxN=20;
        
        
        if np.size(exp_num)==1 and exp_num is None:
            exp_num=sens.info.columns.values
                    
        "Make sure we're working with numpy array for exp_num"
        exp_num=np.atleast_1d(exp_num)
        
        ne=np.size(exp_num)
        
        if mdl_num is None:
            mdl_num=-1

        "Make sure we're working with numpy array for mdl_num"
        mdl_num=np.atleast_1d(mdl_num)

#        "If all mdl_num are the same, replace with a single entry"
#        if np.size(mdl_num)>1 and np.all(mdl_num[0]==mdl_num):
#            mdl_num=mdl_num[0]
#        
        "Delete detector used for R2 exchange correction"
        if hasattr(sens,'detect_par') and sens.detect_par['R2_ex_corr']:
            sens=sens.copy()    #We don't want to edit the original sensitivy object
            ne=ne-1
            exp_num=exp_num[exp_num!=sens.info.axes[1][-1]]
            sens._remove_R2_ex()

        
        "Store all the experiment and model information"
        self.info_in=sens.info.loc[:,exp_num].copy()
        self.MdlPar_in=sens.MdlPar.copy()
        self.mdl_num=mdl_num.copy()
        
        
        k=0
        nM=np.size(self.MdlPar_in)
        while k<nM:
            if not(np.any(self.mdl_num==k)):
                del self.MdlPar_in[k]
                in1=np.where(np.array(self.mdl_num)!=None)[0]
                in2=np.where(self.mdl_num[in1]>k)[0]
                self.mdl_num[in1[in2]]+=-1
                nM=nM-1
            else:
                k=k+1  
        
        if np.all(self.mdl_num==-1):
            self.mdl_num=[]
        
        "Determine if any models are bond specific"
        self.BondSpfc='no'
        if sens._rho(bond=-1).ndim==3:
            self.BondSpfc='yes'     #If the previously applied models are bond-specific, we need to maintain bond specificity
        else:
            for k in self.MdlPar_in:
                if k.get('BondSpfc')=='yes':
                    self.BondSpfc='yes'

        "Pass the molecule object"
        """Note that this is not a copy, but rather a pointer to the same object.
        If you edit the object, it will be changed in the original sens object
        Best to set the selections first with the sens object, and leave alone
        here"""
        self.molecule=sens.molecule                       

        
        "How many bonds are there?"
        nb=self._nb()
        
        "Storage for the input rate constants"
        self.__R=list()         #Store experimental sensitivities
        self.__R0=list()
        self.__RCSA=list()      #Store experimental sensitivities for CSA only
        self.__R0CSA=list()

        "Load in the sensitivity of the selected experiments"
        if np.size(mdl_num)==1:
            if self.BondSpfc=='yes':
#                for k in range(0,nb):
#                    a,b=sens._rho_eff(exp_num=exp_num,mdl_num=mdl_num[0],bond=k)
#                    self.__R.append(a)
#                    self.__R0.append(b)
#                    a,b=sens._rho_effCSA(exp_num=exp_num,mdl_num=mdl_num[0],bond=k)
#                    self.__RCSA.append(a)
#                    self.__R0CSA.append(b)
                a,b=sens._rho_eff(exp_num=exp_num,mdl_num=mdl_num[0],bond=-1)
                c,d=sens._rho_effCSA(exp_num=exp_num,mdl_num=mdl_num[0],bond=-1)
                for k in range(0,nb):
                    self.__R.append(a[k])
                    self.__R0.append(b[k])
                    self.__RCSA.append(c[k])
                    self.__R0CSA.append(d[k])
            elif mdl_num!=-1:
                a,b=sens._rho_eff(exp_num=exp_num,mdl_num=mdl_num[0])
                self.__R.append(a)
                self.__R0.append(b)
                a,b=sens._rho_effCSA(exp_num=exp_num,mdl_num=mdl_num[0])
                self.__RCSA.append(a)
                self.__R0CSA.append(b)
            else:
                self.__R.append(sens._rho(exp_num))
                self.__R0.append(np.zeros(self.__R[0].shape[0]))
                self.__RCSA.append(sens._rhoCSA(exp_num))
                self.__R0CSA.append(np.zeros(self.__RCSA[0].shape[0]))
        else:
            "In this case, we have to get the experiments one at a time"
            if self.BondSpfc=='yes':
                for k in range(0,nb):
                    self.__R.append(np.zeros([ne,ntc]))
                    self.__R0.append(np.zeros(ne))
                    self.__RCSA.append(np.zeros([ne,ntc]))
                    self.__R0CSA.append(np.zeros(ne))
                    for m in range(0,ne):
                        a,b=sens._rho_eff(exp_num=exp_num[m],mdl_num=mdl_num[m],bond=k)
                        self.__R[k][m,:]=a
                        self.__R0[k][m]=b
                        a,b=sens._rho_effCSA(exp_num=exp_num[m],mdl_num=mdl_num[m],bond=k)
                        self.__RCSA[k][m,:]=a
                        self.__R0CSA[k][m]=b
            else:
                self.__R.append(np.zeros([ne,ntc]))
                self.__R0.append(np.zeros(ne))
                self.__RCSA.append(np.zeros([ne,ntc]))
                self.__R0CSA.append(np.zeros(ne))
                for m in range(0,ne):
                    a,b=sens._rho_eff(exp_num=exp_num[m],mdl_num=mdl_num[m])
                    self.__R[0][m,:]=a
                    self.__R0[0][m]=b
                    a,b=sens._rho_effCSA(exp_num=exp_num[m],mdl_num=mdl_num[m])
                    self.__RCSA[0][m,:]=a
                    self.__R0CSA[0][m]=b
       
        "Names of the experimental variables that are available"
        self.__exper=['rho','z0','z0_std','Del_z','Del_z_std','stdev']
        "Names of the spin system variables that are available"
        self.__spinsys=[]
        "Initialize self.info"
        self.info=None
        
        
        "Some global defaults"
        self.detect_par={'Normalization':'M',   #Normalization of detectors
                         'inclS2':False,
                         'NegAllow':0.5,
                         'R2_ex_corr':False} 
        


        ####################Critical edits################        
        "Pass the normalization"
        a=self.info_in.loc['stdev'].to_numpy()   
        b=np.max(np.abs(np.mean(self.__R,axis=0)),axis=-1)  #Mean over bonds, absolute value, max of abs val.
        b[b==0]=1 #Doesn't really help- zeros in sensitivity doesn't work in SVD
        """
        IMPORTANT EDITS HERE. Make Sure to document
        """
        "Replace None with 1"
        index=a==None
        a[index]=1
        self.norm=np.divide(1,a*b).astype('float64')
#        self.norm=np.divide(1,a).astype('float64')
        ##################################################
        "Storage for the detection vectors"
        self.__r=[None]*nb   #Store the detection vectors
        self.__rho=[None]*nb   #Store the detector sensitivities
        self.__rhoAvg=None
        self.__rAvg=None
        self.__Rc=[None]*nb        #Store the back-calculated sensitivities
        self.__RcAvg=None
        self.__rhoCSA=[None]*nb #CSA only sensitivities
        
        "Store SVD matrix for parallel function"
#        self.__Vt=None

        self.z0=[None]*nb
        self.Del_z=[None]*nb
        self.stdev=[None]*nb
        
        self.SVD=list(np.zeros(nb))
        self.SVDavg=dict()
        
        "Store error for r_auto routine"
        self.__r_auto=dict()
        
        super().__init__()

        "The previous line clears self.molecule, so we have to load it again :-/"        
        self.molecule=sens.molecule        
        

#%% Performs and stores results of singular value decomposition of experimental sensitivities    
    def getSVD(self,bond=None,n=None):
        "Function to perform (and store) all singular value decomposition calculations"
        ne=np.shape(self.__R)[1]
        if n is None:
            n=np.min([np.shape(self.__R)[1],self.__maxN])
        
        if bond is None:
            if 'S' in self.SVDavg.keys() and self.SVDavg['S'].size>=n:
                U=self.SVDavg['U'][:,0:n]
                S=self.SVDavg['S'][0:n]
                Vt=self.SVDavg['Vt'][0:n,:]
                VtCSA=0
            else:
                norm=np.repeat(np.transpose([self.norm]),np.size(self.__tc),axis=1)
                U,S,Vt=svd0(np.multiply(np.mean(self.__R,axis=0),norm),n)
                
                self.SVDavg['U']=U
                self.SVDavg['Vt']=Vt
                self.SVDavg['S']=S
                VtCSA=0
                
                U=U[:,0:n]
                S=S[0:n]
                Vt=Vt[0:n,:]
        else:
            if self.SVD[bond]!=0 and self.SVD[bond]['S'].size>=n:
                U=self.SVD[bond]['U'][:,0:n]
                S=self.SVD[bond]['S'][0:n]
                Vt=self.SVD[bond]['Vt'][0:n,:]
                VtCSA=self.SVD[bond]['VtCSA'][0:n,:]
                
            else:                    
                norm=np.repeat(np.transpose([self.norm]),np.size(self.__tc),axis=1)

                U,S,Vt=svd0(np.multiply(self.Rin(bond),norm),n)
                U=U[:,0:np.size(S)]
                    
                
                VtCSA=np.dot(np.diag(1/S),np.dot(U.T,np.multiply(self._RCSAin(bond),norm)))
                
                if self.SVD[bond]==0:
                    self.SVD[bond]=dict()
                
                self.SVD[bond]['U']=U
                self.SVD[bond]['S']=S
                self.SVD[bond]['Vt']=Vt
                self.SVD[bond]['VtCSA']=VtCSA
                
                U=U[:,0:n]
                S=S[0:n]
                Vt=Vt[0:n,:]
                VtCSA=VtCSA[0:n,:]
    
        return U,S,Vt,VtCSA
#%% Generate r matrix for fitting tests (detector sensitivies are not optimized- and not well-separated)
    def r_no_opt(self,n,bond=None,R2_ex_corr=False,**kwargs):
        
        self.detect_par['inclS2']=False
        self.detect_par['R2_ex_corr']=False
        
        self.n=n
        nb=self._nb()

        if nb==1:
            bond=0
            
        if bond is not None and np.size(bond)==1 and np.atleast_1d(bond)[0]==-1:
            bond=np.arange(0,nb)
            
        if bond is None:
            U,S,Vt,VCSA=self.getSVD(None,n)
            self.__rAvg=np.multiply(np.repeat(np.transpose([1/self.norm]),n,axis=1),np.dot(U,np.diag(S)))
            self.__rhoAvg=Vt
            norm=np.repeat(np.transpose([self.norm]),np.size(self.__tc),axis=1)
            self.__RcAvg=np.divide(np.dot(U,np.dot(np.diag(S),Vt)),norm)
            self.SVDavg['T']=np.eye(n)
            self.SVDavg['stdev']=1/S
            if 'sort_rho' not in kwargs:
                kwargs['sort_rho']='n'

            self.__r_info(None,**kwargs)
        else:
            
            bond=np.atleast_1d(bond)

            for k in bond:
                U,S,Vt,VCSA=self.getSVD(k,n)
                 #Here, we try to control the sign returned for Vt 
                 #(it would be nice if repeated runs of r_no_opt returned the same results)
                sgn=np.sign(Vt.sum(axis=1))
                sgn[sgn==0]=Vt[sgn==0,:].max(axis=1)
                Vt=(sgn*Vt.T).T
                VCSA=(sgn*VCSA.T).T
                U=sgn*U
                self.__r[k]=np.multiply(np.repeat(np.transpose([1/self.norm]),n,axis=1),np.dot(U,np.diag(S)))
                self.__rho[k]=Vt
                self.__rhoCSA[k]=VCSA
                norm=np.repeat(np.transpose([self.norm]),np.size(self.__tc),axis=1)
                self.__Rc[k]=np.divide(np.dot(U,np.dot(np.diag(S),Vt)),norm)
                self.SVD[k]['T']=np.eye(n)
                
                if R2_ex_corr:
                    self.R2_ex_corr(bond=k,**kwargs)
                
                self.__r_info(k,**kwargs)
                
                
            if 'sort_rho' not in kwargs:
                kwargs['sort_rho']='n'
            self.__r_info(None,**kwargs)

#%% Automatic generation of detectors from a set of sensitivities
    def r_auto(self,n,Normalization='Max',inclS2=False,NegAllow=0.5,R2_ex_corr=False,bond=None,parallel=True,z0=None,**kwargs):

        assert n<=self.Rin().shape[0],'Number of detectors cannot be larger than the number of experiments'
        
        self.n=n
        
        self.detect_par['inclS2']=inclS2

        "A little bit silly that the variable names changed...fix later"
        Neg=NegAllow  
        R2ex=R2_ex_corr
        
        "Store some of the inputs"
        self.detect_par.update({'Normalization':Normalization,'inclS2':inclS2,'R2_ex_corr':R2_ex_corr,'NegAllow':NegAllow})

            
        nb=self._nb()
        "If bond set to -1, run through all orientations."
        if bond is None:
            bonds=np.zeros(0)
        elif np.size(bond)==1 and np.atleast_1d(bond)[0]==-1:
            bond=None
            bonds=np.arange(0,nb)
        else:
            bond=np.atleast_1d(bond)
            bonds=bond[1:]
            bond=bond[0]
            
            
        if nb==1:
            "If we only have one set of sensitivities (that is, no orientation dependence, then don't use averages"
            bond=0
            
        if bond is None:
            "Here we operate on the average sensitivities"
            U,S,Vt,VCSA=self.getSVD(None,n)
            norm=np.repeat(np.transpose([self.norm]),np.size(self.__tc),axis=1)
            self.__RcAvg=np.divide(np.dot(U,np.dot(np.diag(S),Vt)),norm)
        else:                
            "We work on the first bond given, and use r_target for the remaining bonds"
            U,S,Vt,VCSA=self.getSVD(bond,n)
            norm=np.repeat(np.transpose([self.norm]),np.size(self.__tc),axis=1)
            self.__Rc[bond]=np.divide(np.dot(U,np.dot(np.diag(S),Vt)),norm)
     
        ntc=np.size(self.__tc) #Number of correlation times
        

        """
        In the follow lines (loop over ntc, or z0), we optimize detectors at 
        either every possible correlation time, or correlation times specified
        by z0. 
        """

        def true_range(k,untried):
            """Finds the range around k in untried where all values are True
            """
            i=np.nonzero(np.logical_not(untried[k:]))[0]
            right=(k+i[0]) if len(i)!=0 else len(untried)
            i=np.nonzero(np.logical_not(untried[:k]))[0]
            left=(i[-1]+1) if len(i)!=0 else 0
            
            return left,right

        def find_nearest(Vt,k,untried,error=None,endpoints=False):
            """Finds the location of the best detector near index k. Note that the
            vector untried indicates where detectors may still exist. k must fall 
            inside a range of True elements in untried, and we will only search within
            that range. Note that by default, finding the best detector at the
            end of that range will be disallowed, since the range is usually bound
            by detectors that have already been identified. Exceptions are the first
            and last positions. untried will be modified in-place
            """
            
            left,right=true_range(k,untried)
            
            maxi=100000
            test=k
            while k!=maxi:
                if not(np.any(untried[left:right])):return     #Give up if the whole range of untried around k is False
                k=test
                rhoz0,x,maxi=det_opt(Vt,k)
                error[k]=np.abs(k-maxi)
                if k<=maxi:untried[k:maxi+1]=False  #Update the untried index
                else:untried[maxi:k+1]=False
                test=maxi
            
            if (k<=left or k>=right-1) and not(endpoints):
                return None #Don't return ends of the range unless 0 or ntc
            else:
                return rhoz0,x,k
         
        def biggest_gap(untried):
            """Finds the longest range of True values in the untried index
            """
            k=np.nonzero(untried)[0][0]
            gap=0
            biggest=0
            while True:
                left,right=true_range(k,untried)
                if right-left>gap:
                    gap=right-left
                    biggest=np.mean([left,right],dtype=int)
                i0=np.nonzero(untried[right:])[0]
                if len(i0)>0:
                    k=right+np.nonzero(untried[right:])[0][0]
                else:
                    break
            return biggest

                
        
        def det_opt(Vt,k,target=None):
            """Performs the optimization of a detectors having a value of 1 at the kth
            correlation time, and minimized elsewhere. Target is the minimum allowed
            value for the detector as a function of correlation time. Default is zeros
            everywhere.
            
            Returns the optimized detector and the location of the maximum of that 
            detector
            """
            ntc=Vt.shape[1]
            target=target if target else np.zeros(ntc)
            x=linprog(Vt.sum(1),-Vt.T,-target,[Vt[:,k]],1,bounds=(-500,500),\
                      method='interior-point',options={'disp':False})
            rhoz=(Vt.T@x['x']).T
            maxi=np.argmax(np.abs(rhoz))
            return rhoz,x['x'],maxi
    
        #Locate where the Vt are sufficiently large to have maxima
        i0=np.nonzero(np.any(np.abs(Vt.T)>(np.abs(Vt).max(1)*.75),1))[0]
        
        untried=np.ones(ntc,dtype=bool)
        untried[:i0[0]]=False
        untried[i0[-1]+1:]=False
        count=0     #How many detectors have we found?
        index=list()    #List of indices where detectors are found
        rhoz=list()     #Optimized sensitivity
        X=list()        #Columns of the T-matrix
        err=np.ones(ntc,dtype=int)*ntc #Keep track of error at all time points tried

        "Locate the left-most detector"
        if untried[0]:
            rhoz0,x,k=find_nearest(Vt,0,untried,error=err,endpoints=True)
            rhoz.append(rhoz0)
            X.append(x)
            index.append(k)
            count+=1
        "Locate the right-most detector"
        if untried[-1] and n>1:
            rhoz0,x,k=find_nearest(Vt,ntc-1,untried,error=err,endpoints=True)
            rhoz.append(rhoz0)
            X.append(x)
            index.append(k)
            count+=1
        "Locate remaining detectors"
        while count<n:  
            "Look in the middle of the first untried range"
            k=biggest_gap(untried)
            out=find_nearest(Vt,k,untried,error=err)  #Try to find detectors
            if out: #Store if succesful
                rhoz.append(out[0])
                X.append(out[1])
                index.append(out[2])
#                untried[out[2]-1:out[2]+2]=False #No neighboring detectors
                count+=1
        
        
        i=np.argsort(index).astype(int)
        pks=np.array(index)[i]
        rhoz=np.array(rhoz)[i]
        T=np.array(X)[i]
        
        
        """Detectors that are not approaching zero at the end of the range of
        correlation times tend to oscillate where they do approach zero. We want
        to push that oscillation slightly below zero
        """
        
        for k in [0,n-1]:
            try:
                if (rhoz[k,0]>0.95*np.max(rhoz[k,:]) or rhoz[k,-1]>0.95*np.max(rhoz[k,:])) and Neg!=0:
    
                    reopt=True #Option to cancel the re-optimization in special cases
                    
                    if rhoz[k,0]>0.95*np.max(rhoz[k,:]):
                        pm=1;
                    else:
                        pm=-1;                        
                    
                    temp=rhoz[k,:]
                    "Locate maxima and minima in the detector"
                    mini=np.where((temp[2:]-temp[1:-1]>=0) & (temp[1:-1]-temp[0:-2]<=0))[0]+1
                    maxi=np.where((temp[2:]-temp[1:-1]<=0) & (temp[1:-1]-temp[0:-2]>=0))[0]+1
                    
                    """Filter out minima that occur at more than 90% of the sensitivity max,
                    since these are probably just glitches in the optimization.
                    """
                    if np.size(mini)>=2 and np.size(maxi)>=2:
                        mini=mini[(temp[mini]<.9) & (temp[mini]<.05*np.max(-pm*np.diff(temp[maxi])))]
                    elif np.size(mini)>=2:
                        mini=mini[temp[mini]<.9]
                        
                    if np.size(maxi)>=2:
                        maxi=maxi[(temp[maxi]<.9) & (temp[maxi]>0.0)]
    #                    maxi=maxi[(temp[maxi]<.9) & (temp[maxi]>0.0*np.max(-pm*np.diff(temp[maxi])))]
                    
                    
                    if rhoz[k,0]>0.95*np.max(rhoz[k,:]):
                    
                        "Calculation for the first detection vector"
    
                        if np.size(maxi)>=2 & np.size(mini)>=2:
                            step=int(np.round(np.diff(mini[0:2])/2))
                            slope2=-(temp[maxi[-1]]-temp[maxi[0]])*Neg/(maxi[-1]-maxi[0])
                        elif np.size(maxi)==1 and np.size(mini)>=1:
                            step=maxi[0]-mini[0]
                            slope2=temp[maxi[0]]*Neg/step
                        else:
                            reopt=False
                            
                        if reopt:
                            a=np.max([1,mini[0]-step])
                            slope1=-temp[maxi[0]]/step*Neg
                            line1=np.arange(0,-temp[maxi[0]]*Neg-1e-12,slope1)
                            line2=np.arange(-temp[maxi[0]]*Neg,1e-12,slope2)
                            try:
                                target=np.concatenate((np.zeros(a),line1,line2,np.zeros(ntc-a-np.size(line1)-np.size(line2))))
                            except:
                                reopt=False
                                    
                    else:
                        "Calculation otherwise (last detection vector)"
                        if np.size(maxi)>=2 & np.size(mini)>=2:
                            step=int(np.round(np.diff(mini[-2:])/2))
                            slope2=-(temp[maxi[0]]-temp[maxi[-1]])*Neg/(maxi[0]-maxi[-1])
                        elif np.size(maxi)==1 and np.size(mini)>=1:
                            step=mini[-1]-maxi[0]
                            slope2=-temp[maxi[0]]*Neg/step
                        else:
                            reopt=False
                            
                        if reopt:
                            a=np.min([ntc,mini[-1]+step])
                            slope1=temp[maxi[-1]]/step*Neg
        
                            line1=np.arange(-temp[maxi[-1]]*Neg,1e-12,slope1)
                            line2=np.arange(0,-temp[maxi[-1]]*Neg-1e-12,slope2)                    
                            target=np.concatenate((np.zeros(a-np.size(line1)-np.size(line2)),line2,line1,np.zeros(ntc-a)))
                        
    
                    if reopt:
                        Y=(Vt,pks[k],target)
                        
                        X=linprog_par(Y)
                        T[k,:]=X
                        rhoz[k,:]=np.dot(T[k,:],Vt)
            except:
                pass

        "Save the results into the detect object"
#        self.r0=self.__r
        if bond is None:
            self.__rAvg=np.multiply(np.repeat(np.transpose([1/self.norm]),n,axis=1),\
                        np.dot(U,np.linalg.solve(T.T,np.diag(S)).T))
            self.__rhoAvg=rhoz
            self.SVDavg['T']=T
            self.__r_auto={'Error':err,'Peaks':pks,'rho_z':self.__rhoAvg.copy()}
            if R2ex:
                self.R2_ex_corr(bond,**kwargs)
            self.__r_norm(bond,**kwargs)
            if inclS2:
                self.inclS2(bond=None,**kwargs)
            self.__r_info(bond,**kwargs)
            if np.size(bonds)>0:
                if 'NT' in kwargs: #We don't re-normalize the results of detectors obtained with r_target
                    kwargs.pop('NT')
                if 'Normalization' in kwargs:
                    kwargs.pop('Normalization')
                self.r_target(n,bond=bonds,Normalization=None,**kwargs)
        else:

            """This isn't correct yet- if more than one bond, we want to 
            use the result for the average calculation as a target for the 
            individual bonds, not loop over all bonds with the result here
            """
            self.__r[bond]=np.multiply(np.repeat(np.transpose([1/self.norm]),n,axis=1),\
                        np.dot(U,np.linalg.solve(T.T,np.diag(S)).T))
            self.__rho[bond]=rhoz
            self.__rhoCSA[bond]=np.dot(T,VCSA)
            self.SVD[bond]['T']=T
            self.__r_auto={'Error':err,'Peaks':pks,'rho_z':self.__rho[bond].copy()}
            if R2ex:                
                self.R2_ex_corr(bond,**kwargs)
                        
            self.__r_norm(bond,**kwargs)
            if inclS2:
                self.inclS2(bond=k,**kwargs)
            self.__r_info(bond,**kwargs)
            if np.size(bonds)>0:
                if 'NT' in kwargs: #We don't re-normalize the results of detectors obtained with r_target
                    kwargs.pop('NT')
                if 'Normalization' in kwargs:
                    kwargs.pop('Normalization')
                self.r_target(n,self.__rho[bond],bonds,Normalization=None,**kwargs)
                 
    def r_auto2(self,n,Normalization='Max',inclS2=False,NegAllow=0.5,R2_ex_corr=False,bond=None,parallel=True,z0=None,**kwargs):

        assert n<=self.Rin().shape[0],'Number of detectors cannot be larger than the number of experiments'
        
        self.n=n
        
        self.detect_par['inclS2']=inclS2

        "A little bit silly that the variable names changed...fix later"
        Neg=NegAllow  
        R2ex=R2_ex_corr
        
        "Store some of the inputs"
        self.detect_par.update({'Normalization':Normalization,'inclS2':inclS2,'R2_ex_corr':R2_ex_corr,'NegAllow':NegAllow})

            
        nb=self._nb()
        "If bond set to -1, run through all orientations."
        if bond is None:
            bonds=np.zeros(0)
        elif np.size(bond)==1 and np.atleast_1d(bond)[0]==-1:
            bond=None
            bonds=np.arange(0,nb)
        else:
            bond=np.atleast_1d(bond)
            bonds=bond[1:]
            bond=bond[0]
            
            
        if nb==1:
            "If we only have one set of sensitivities (that is, no orientation dependence, then don't use averages"
            bond=0
            
        if bond is None:
            "Here we operate on the average sensitivities"
            U,S,Vt,VCSA=self.getSVD(None,n)
            norm=np.repeat(np.transpose([self.norm]),np.size(self.__tc),axis=1)
            self.__RcAvg=np.divide(np.dot(U,np.dot(np.diag(S),Vt)),norm)
        else:                
            "We work on the first bond given, and use r_target for the remaining bonds"
            U,S,Vt,VCSA=self.getSVD(bond,n)
            norm=np.repeat(np.transpose([self.norm]),np.size(self.__tc),axis=1)
            self.__Rc[bond]=np.divide(np.dot(U,np.dot(np.diag(S),Vt)),norm)
     
        ntc=np.size(self.__tc) #Number of correlation times
        err=np.zeros(ntc)       #Error of fit
        

        """
        In the follow lines (loop over ntc, or z0), we optimize detectors at 
        either every possible correlation time, or correlation times specified
        by z0. 
        """

        "Prepare data for parallel processing"
        Y=list()
        if z0 is None:
            z0index=range(ntc)
        else:
            if n>len(z0):
                print('z0 must have at least length n')
                return
            else:
                z0index=list()
                for z1 in z0:
                    z0index.append(np.argmin(np.abs(self.z()-z1)))
        
        for k in z0index:
            Y.append((Vt,k))            

        "Default is parallel processing"
        if not(parallel):
            X=list()
            for Y0 in Y:
                X.append(linprog_par(Y0))
        else:
            with mp.Pool() as pool:
                X=pool.map(linprog_par,Y)
            
        """We optimized detectors at every correlation time (see __linprog_par),
        which have values at 1 for the given correlation time. We want to keep 
        those detectors where the maximum is closest to the correlation time set
        to 1. We search for those here:
        """
        if z0 is None:
            for k in range(0,ntc):
                err[k]=np.abs(np.argmax(np.dot(Vt.T,X[k]))-k)
        else:
            err=np.ones(ntc)*ntc
            for m,k in enumerate(z0index):
                err[k]=np.abs(np.argmax(np.dot(Vt.T,X[m]))-k)
            x0=X.__iter__()
            X=[x0.__next__() if k in z0index else None for k in range(ntc)]
                
        
        if 'Type' in self.info_in.index and 'S2' in self.info_in.loc['Type'].to_numpy() and z0 is None:
            err[0]=0    #Forces a detector that is non-zero at the shortest correlation time if S2 included
            "Possibly need to delete above two lines...not fully tested"
        
        """Ideally, the number of detectors equals the number of minima in err,
        however, due to calculation error, this may not always be the case. We
        start searching for values where err=0. If we don't have enough, we
        raise this value in steps (looking for err=1, err=2), until we have
        enough. If, in one step, we go from too few to too many, we eliminate, 
        one at a time, the peak that is closest to another peak.
        """
        test=True
        thresh=0
        while test:
            pks=np.where(err<=thresh)[0]
            if pks.size==n:
                test=False
            elif pks.size<n:
                thresh=thresh+1
            elif pks.size>n:

                while pks.size>n:
                    a=np.argsort(np.diff(pks))
#                    pks=np.concatenate([pks[a[np.size(pks)-n:]],[pks[-1]]])
                    pks=np.concatenate([pks[a[1:]],[pks[-1]]])
                    pks.sort()
                test=False
        
        "Save the linear combinations for the best detectors"
        T=np.zeros([n,n])
        for k in range(0,n):
            T[k,:]=X[pks[k]]
        
        rhoz=np.dot(T,Vt)
        
        """Detectors that are not approaching zero at the end of the range of
        correlation times tend to oscillate where they do approach zero. We want
        to push that oscillation slightly below zero
        """
        
        for k in range(0,n):
            try:
                if (rhoz[k,0]>0.95*np.max(rhoz[k,:]) or rhoz[k,-1]>0.95*np.max(rhoz[k,:])) and Neg!=0:
    
                    reopt=True #Option to cancel the re-optimization in special cases
                    
                    if rhoz[k,0]>0.95*np.max(rhoz[k,:]):
                        pm=1;
                    else:
                        pm=-1;                        
                    
                    temp=rhoz[k,:]
                    "Locate maxima and minima in the detector"
                    mini=np.where((temp[2:]-temp[1:-1]>=0) & (temp[1:-1]-temp[0:-2]<=0))[0]+1
                    maxi=np.where((temp[2:]-temp[1:-1]<=0) & (temp[1:-1]-temp[0:-2]>=0))[0]+1
                    
                    """Filter out minima that occur at more than 90% of the sensitivity max,
                    since these are probably just glitches in the optimization.
                    """
                    if np.size(mini)>=2 and np.size(maxi)>=2:
                        mini=mini[(temp[mini]<.9) & (temp[mini]<.05*np.max(-pm*np.diff(temp[maxi])))]
                    elif np.size(mini)>=2:
                        mini=mini[temp[mini]<.9]
                        
                    if np.size(maxi)>=2:
                        maxi=maxi[(temp[maxi]<.9) & (temp[maxi]>0.0)]
    #                    maxi=maxi[(temp[maxi]<.9) & (temp[maxi]>0.0*np.max(-pm*np.diff(temp[maxi])))]
                    
                    
                    if rhoz[k,0]>0.95*np.max(rhoz[k,:]):
                        "Calculation for the first detection vector"
    
                        if np.size(maxi)>=2 & np.size(mini)>=2:
                            step=int(np.round(np.diff(mini[0:2])/2))
                            slope2=-(temp[maxi[-1]]-temp[maxi[0]])*Neg/(maxi[-1]-maxi[0])
                        elif np.size(maxi)==1 and np.size(mini)>=1:
                            step=maxi[0]-mini[0]
                            slope2=temp[maxi[0]]*Neg/step
                        else:
                            reopt=False
                            
                        if reopt:
                            a=np.max([1,mini[0]-step])
                            slope1=-temp[maxi[0]]/step*Neg
                            line1=np.arange(0,-temp[maxi[0]]*Neg-1e-12,slope1)
                            line2=np.arange(-temp[maxi[0]]*Neg,1e-12,slope2)
                            try:
                                target=np.concatenate((np.zeros(a),line1,line2,np.zeros(ntc-a-np.size(line1)-np.size(line2))))
                            except:
                                reopt=False
                                    
                    else:
                        "Calculation otherwise (last detection vector)"
                        if np.size(maxi)>=2 & np.size(mini)>=2:
                            step=int(np.round(np.diff(mini[-2:])/2))
                            slope2=-(temp[maxi[0]]-temp[maxi[-1]])*Neg/(maxi[0]-maxi[-1])
                        elif np.size(maxi)==1 and np.size(mini)>=1:
                            step=mini[-1]-maxi[0]
                            slope2=-temp[maxi[0]]*Neg/step
                        else:
                            reopt=False
                            
                        if reopt:
                            a=np.min([ntc,mini[-1]+step])
                            slope1=temp[maxi[-1]]/step*Neg
        
                            line1=np.arange(-temp[maxi[-1]]*Neg,1e-12,slope1)
                            line2=np.arange(0,-temp[maxi[-1]]*Neg-1e-12,slope2)                    
                            target=np.concatenate((np.zeros(a-np.size(line1)-np.size(line2)),line2,line1,np.zeros(ntc-a)))
                        
    
                    if reopt:
                        Y=(Vt,pks[k],target)
                        
                        X=linprog_par(Y)
                        T[k,:]=X
                        rhoz[k,:]=np.dot(T[k,:],Vt)
            except:
                pass

   
        "Save the results into the detect object"
#        self.r0=self.__r
        if bond is None:
            self.__rAvg=np.multiply(np.repeat(np.transpose([1/self.norm]),n,axis=1),\
                        np.dot(U,np.linalg.solve(T.T,np.diag(S)).T))
            self.__rhoAvg=rhoz
            self.SVDavg['T']=T
            self.__r_auto={'Error':err,'Peaks':pks,'rho_z':self.__rhoAvg.copy()}
            if R2ex:
                self.R2_ex_corr(bond,**kwargs)
            self.__r_norm(bond,**kwargs)
            if inclS2:
                self.inclS2(bond=None,**kwargs)
            self.__r_info(bond,**kwargs)
            if np.size(bonds)>0:
                if 'NT' in kwargs: #We don't re-normalize the results of detectors obtained with r_target
                    kwargs.pop('NT')
                if 'Normalization' in kwargs:
                    kwargs.pop('Normalization')
                self.r_target(n,bond=bonds,Normalization=None,**kwargs)
        else:

            """This isn't correct yet- if more than one bond, we want to 
            use the result for the average calculation as a target for the 
            individual bonds, not loop over all bonds with the result here
            """
            self.__r[bond]=np.multiply(np.repeat(np.transpose([1/self.norm]),n,axis=1),\
                        np.dot(U,np.linalg.solve(T.T,np.diag(S)).T))
            self.__rho[bond]=rhoz
            self.__rhoCSA[bond]=np.dot(T,VCSA)
            self.SVD[bond]['T']=T
            self.__r_auto={'Error':err,'Peaks':pks,'rho_z':self.__rho[bond].copy()}
            if R2ex:                
                self.R2_ex_corr(bond,**kwargs)
                        
            self.__r_norm(bond,**kwargs)
            if inclS2:
                self.inclS2(bond=k,**kwargs)
            self.__r_info(bond,**kwargs)
            if np.size(bonds)>0:
                if 'NT' in kwargs: #We don't re-normalize the results of detectors obtained with r_target
                    kwargs.pop('NT')
                if 'Normalization' in kwargs:
                    kwargs.pop('Normalization')
                self.r_target(n,self.__rho[bond],bonds,Normalization=None,**kwargs)

    def r_target(self,target=None,n=None,bond=None,Normalization=None,inclS2=None,R2_ex_corr=None,parallel=True,**kwargs):
        """Set sensitivities as close to some target function as possible
        
        Note, if no target given, this function updates bonds to match the r_auto
        sensitivity results. Then, the settings R2_ex_corr, and inclS2 are taken
        from the previous settings. Otherwise, these are set to False.
        """
        

        
        if target is None:
            try:
#                target=self.__r_auto.get('rho_z')
                target=self.rhoz()
            except:
                print('No target provided, and no sensitivity from r_auto available')
                return

            R2ex=self.detect_par['R2_ex_corr']
            inS2=self.detect_par['inclS2']
            target=self.rhoz(bond=None)
            if R2ex:
                target=target[:-1]
            if inS2:
                target=target[1:]
            if R2_ex_corr is None:
                R2_ex_corr=R2ex
            if inclS2 is None:
                inclS2=inS2
    
        
        target=np.atleast_2d(target)        

        "Store some of the inputs"
        self.detect_par.update({'Normalization':Normalization,'inclS2':inclS2,'R2_ex_corr':R2_ex_corr})


        
        
        if n is None:
            n=target.shape[0]
        
        self.n=n
        nb=self._nb()
        
        "If bond set to -1, run through all orientations."
        if bond is not None and np.size(bond)==1 and np.atleast_1d(bond)[0]==-1:
            bond=np.arange(0,nb)
                
        if nb==1:
            bond=0
            

        if bond is None:
            "Here we operate on the average sensitivities"
            U,S,Vt,VCSA=self.getSVD(None,n)
            norm=np.repeat(np.transpose([self.norm]),np.size(self.__tc),axis=1)
            self.__RcAvg=np.divide(np.dot(U,np.dot(np.diag(S),Vt)),norm)
            
            T=lsqlin_par((Vt,target))
            
            rhoz=np.dot(T,Vt)
            self.__rAvg=np.multiply(np.repeat(np.transpose([1/self.norm]),n,axis=1),\
                        np.dot(U,np.linalg.solve(T.T,np.diag(S)).T))
            self.__rhoAvg=rhoz
            self.SVDavg['T']=T
            if Normalization is not None:
                self.__r_norm(None,**kwargs)  
            if ('inclS2' in kwargs and kwargs['inclS2']) or\
                self.detect_par['inclS2']:
                self.inclS2(bond=None,**kwargs)
            self.__r_info(bond,**kwargs)
        else:
            Y=list()
            bond=np.atleast_1d(bond)
            
            for k in bond:
                U,S,Vt,VCSA=self.getSVD(k,n)
                norm=np.repeat(np.transpose([self.norm]),np.size(self.__tc),axis=1)
                self.__Rc[k]=np.divide(np.dot(U,np.dot(np.diag(S),Vt)),norm)
                Y.append((Vt,target))
                
            "Default is parallel processing"
            if not(parallel) or len(Y)==1:
                T=[lsqlin_par(k) for k in Y]
            else:
                with mp.Pool() as pool:
                    T=pool.map(lsqlin_par,Y)
                    
            for index,k in enumerate(bond):
                U,S,Vt,VCSA=self.getSVD(k,n)
                self.SVD[k]['T']=T[index]
                self.__r[k]=np.multiply(np.repeat(np.transpose([1/self.norm]),n,axis=1),\
                    np.dot(U,np.linalg.solve(T[index].T,np.diag(S)).T))
                self.__rho[k]=np.dot(T[index],Vt)
                self.__rhoCSA[k]=np.dot(T[index],VCSA)
                self.SVD[k]['T']=T[index]
                if Normalization is not None:
                    self.__r_norm(k,**kwargs)
                if self.detect_par['R2_ex_corr']:
                    self.R2_ex_corr(bond=k,**kwargs)
                if self.detect_par['inclS2']:
                    self.inclS2(bond=k,**kwargs)
                    
                
        if 'sort_rho' not in kwargs:
            kwargs['sort_rho']='n'
        self.__r_info(bond,**kwargs)
        
    
#    def r_IMPACT(self,n=None,tc=None,z=None,IMPACTbnds=True,inclS2=False,Normalization='MP',unidist_range=[-11,-8],**kwargs):
#        """
#        Optimizes a set of detectors that behave as an IMPACT fit. That is, 
#        we set up an array of correlation times to fit data to. 
#        
#        Options are to request a specific number of correlation times, in which
#        case we will optimize the fit of a uniform distribution with n 
#        correlation times, where we vary the width and center of the array of
#        correlation times. Otherwise, one may input the array of correlation 
#        times directly (set either n or tc as an array).
#        
#        Furthermore, by default we calculate the sensitivities for a single
#        correlation time using bounds on each correlation time (min=0,max=1),
#        and with the sum of all amplitudes adding to 1 (IMPACTbnds=True). This 
#        is the default IMPACT behavior, although it is not the recommended 
#        detectors methodology (detectors optimized to use the IMPACT methodology
#        will not behave like true detectors!). Alternatively, we may set 
#        IMPACTbnds False for standard detectors behavior.
#        
#        Note that IMPACT required all amplitudes to sum to 1. We do not require
#        this, although setting inclS2 to True, and Normalization to MP will add
#        a detector that will cause the total amplitude to be 1. This is equivalent
#        to adding a very short (~10 ps) correlation time to the correlation 
#        time array.
#        
#        IMPACT Reference:
#        Khan, S. N., C. Charlier, R. Augustyniak, N. Salvi, V. Dejean, 
#        G. Bodenhausen, O. Lequin, P. Pelupessy, and F. Ferrage. 
#        “Distribution of Pico- and Nanosecond Motions in Disordered Proteins 
#        from Nuclear Spin Relaxation.” Biophys J, 2015. 
#        https://doi.org/10.1016/j.bpj.2015.06.069.
#        """
#    
#    
#        assert not(n is None and tc is None and z is None),"Set n, tc, or z"
#        
#        self.detect_par['inclS2']=inclS2
#        if Normalization is not None:self.detect_par['Normalization']=Normalization
#        if 'R2_ex_corr' in kwargs:self.detect_par['R2_ex_corr']=kwargs['R2_ex_corr']
#        
#        if tc is None and z is None:
#            dz=np.diff(self.z()[:2])[0]
#            i1=np.argmin(np.abs(self.z()-unidist_range[0]))
#            i2=np.argmin(np.abs(self.z()-unidist_range[1]))
#            R=self.__R[0][:,i1:i2].sum(1)/(i2-i1)
##            assert n<=R.size,"n must be less than or equal to the number of experiments"
#            
#            err=list()
#            w=list()
#            c=list()
#            for width in np.linspace(2,8,30):
#                zswp=self.z()
#                zswp=np.linspace(self.z()[0]+width/2,self.z()[-1]-width/2,100)
#                for center in zswp:
#                    z=np.linspace(-width/2,width/2,n)+center
#                    r=linear_ex(self.z(),self.__R[0],z)
#                    err.append(lsqlin(r,R,(0,1) if IMPACTbnds else (-np.inf,np.inf))['cost'])
#                    w.append(width)
#                    c.append(center)
#            i=np.argmin(np.array(err))
#            z=np.linspace(-w[i]/2,w[i]/2,n)+c[i]
#                
#
#        z=np.array(z) if tc is None else np.log10(tc)
#        print(z)
#        assert len(z)<=self.__R[0].shape[0],\
#            "Number of correlation times must be less than or equal to the number of experiments"
#        r=linear_ex(self.z(),self.__R[0],z)
#        
#        ntc=self.tc().size
#        rhoz=np.zeros([len(z),ntc])
#        rhozCSA=np.zeros([len(z),ntc])
#        for k in range(ntc):
#            rhoz[:,k]=lsqlin(r,self.__R[0][:,k],(0,1) if IMPACTbnds else (-np.inf,np.inf))['x']
#            rhozCSA[:,k]=lsqlin(r,self.__RCSA[0][:,k],(0,1) if IMPACTbnds else (-np.inf,np.inf))['x']
#        self.__rho[0]=rhoz
#        self.__rhoCSA[0]=rhozCSA
#        self.__r[0]=r
#        self.n=self.__r[0].shape[1]
#        self.SVD[0]={'U':None,'S':None,'Vt':None,'VtCSA':None,'T':np.eye(self.n)}
#        if Normalization is not None:
#            self.__r_norm(0,Normalization=Normalization)
#        if self.detect_par['R2_ex_corr']:
#            self.R2_ex_corr(bond=0,**kwargs)
#        if self.detect_par['inclS2']:
#            self.inclS2(bond=0,Normalization=Normalization)     
#        self.__r_info()
            
    
    def __addS2(self,bond=None,**kwargs):
        if 'NT' in kwargs:
            NT=kwargs.get('NT')
        elif 'Normalization' in kwargs:
            NT=kwargs.get('Normalization')
        else:
            NT=self.detect_par.get('Normalization')
            
    
    def R2_ex_corr(self,bond=None,v_ref=None,**kwargs):
        """
        detect.R2_ex_corr(bond=None,v_ref=None,**kwargs)
        Attempts to fit exchange contributions to R2 relaxation. Requires R2 
        measured at at least two fields. By default, adds a detection vector which
        corrects for exchange, and returns the estimated exchange contribution at 
        the lowest field at which R2 was measured.
        """
        self.detect_par.update({'R2_ex_corr':True})
        
        index=self.info_in.loc['Type']=='R2'
        if np.where(index)[0].size<2:
            print('Warning: At least 2 R2 experiments are required to perform the exchange correction')
            return
        
        nb=self._nb()
        if nb==1:
            "If bond is not specified, and we don't have bond specificity, operate on bond 0"
            bond=0
            

        
        
        r_ex_vec=np.zeros(self.info_in.shape[1])
        v0=np.atleast_1d(self.info_in.loc['v0'][index])
        if v_ref is None:
            v_ref=np.min(v0)
    
        r_ex_vec[index]=np.divide(v0**2,v_ref**2)
        
        rhoz=np.zeros(self.tc().size)
        rhoz[-1]=1e6
        if bond is None:
            self.__rAvg=np.concatenate((self.__rAvg,np.transpose([r_ex_vec])),axis=1)
            self.__rhoAvg=np.concatenate((self.__rhoAvg,[rhoz]),axis=0)
        else:
            bond=np.atleast_1d(bond) #Work with np arrays
            for k in bond:
                self.__r[k]=np.concatenate((self.__r[k],np.transpose([r_ex_vec])),axis=1)
                self.__rho[k]=np.concatenate((self.__rho[k],[rhoz]),axis=0)
                self.__rhoCSA[k]=np.concatenate((self.__rhoCSA[k],np.zeros([1,self.tc().size])))
    
    def inclS2(self,bond=0,**kwargs):
        """
        Adds an additional detector calculated from a measured order parameter.
        If using, one must include the last column of data.R as the order parameter
        measurement (input as 1-S2)
        """
        self.detect_par['inclS2']=True
        
        nb=self._nb()
        if nb==1:
            bond=0  #No bond specificity, operate on bond 0

        "Put together pretty quickly- should review and verify CSA behavior is correct"
        "Note that we don't really expect users to use CSA w/ S2...could be though"
        
        bond=np.atleast_1d(bond)
        for k in bond:
            if self.detect_par['Normalization'][:2].lower()=='mp':
                wt=linprog(-(self.__rho[k].sum(axis=1)).T,self.__rho[k].T,np.ones(self.__rho[k].shape[1]),\
                           bounds=(-500,500),method='interior-point',options={'disp' :False,})['x']
                rhoz0=[1-np.dot(self.__rho[k].T,wt).T]
                rhoz0CSA=[1-np.dot(self.__rhoCSA[k].T,wt).T]
                sc=np.atleast_1d(rhoz0[0].max())
                self.__rho[k]=np.concatenate((rhoz0/sc,self.__rho[k]))
                self.__rhoCSA[k]=np.concatenate((rhoz0CSA/sc,self.__rhoCSA[k]))
                mat1=np.concatenate((np.zeros([self.__r[k].shape[0],1]),self.__r[k]),axis=1)
                mat2=np.atleast_2d(np.concatenate((sc,wt.T),axis=0))
                self.__r[k]=np.concatenate((mat1,mat2),axis=0)
            elif self.detect_par['Normalization'][0].lower()=='m':
                self.__r[k]=np.concatenate((\
                        np.concatenate((np.zeros([self.__r[k].shape[0],1]),self.__r[k]),axis=1),\
                        np.ones([1,self.__r[k].shape[1]+1])),axis=0)
                self.__rho[k]=np.concatenate(([1-self.__rho[k].sum(axis=0)],\
                          self.__rho[k]),axis=0)
                self.__rhoCSA[k]=np.concatenate(([1-self.__rhoCSA[k].sum(axis=0)],\
                          self.__rhoCSA[k]),axis=0)
            elif self.detect_par['Normalization'][0].lower()=='i':
                wt=linprog(-(self.__rho[k].sum(axis=1)).T,self.__rho[k].T,np.ones(self.__rho[k].shape[1]),\
                           bounds=(-500,500),method='interior-point',options={'disp' :False,})['x']
                rhoz0=[1-np.dot(self.__rho[k].T,wt).T]
                rhoz0CSA=[1-np.dot(self.__rhoCSA[k].T,wt).T]
                sc=rhoz0[0].sum()*np.diff(self.z()[:2])
                self.__rho[k]=np.concatenate((rhoz0/sc,self.__rho[k]))
                self.__rhoCSA[k]=np.concatenate((rhoz0CSA/sc,self.__rhoCSA[k]))
                mat1=np.concatenate((np.zeros([self.__r[k].shape[0],1]),self.__r[k]),axis=1)
                mat2=np.atleast_2d(np.concatenate((sc,wt.T),axis=0))
                self.__r[k]=np.concatenate((mat1,mat2),axis=0)
            
    def _remove_R2_ex(self):
        """
        Deletes the R2 exchange correction from all bonds and from the average
        sensitivity calculation. If the user has manually set detect_par['R2_ex_corr']
        to 'no', this function will do nothing (so don't edit this parameter
        manually!)
        detect._remove_R2_ex()
        """
        
        if not(self.detect_par['R2_ex_corr']):
            return
        else:
            self.detect_par['R2_ex_corr']=False
            if self.info is not None:
                self.info=self.info.drop(self.info.axes[1][-1],axis=1)
            if self.__rAvg is not None:
                self.__rAvg=self.__rAvg[:,:-1]
                self.__rhoAvg=self.__rhoAvg[:-1]
#            nb=np.shape(self.__r)[0]
            nb=self._nb()
            
            for k in range(nb):
                if self.__r is not None and self.__r[k] is not None:
                    self.__r[k]=self.__r[k][:,:-1]
                if self.__rho[k] is not None:
                    self.__rho[k]=self.__rho[k][:-1]
                    self.__rhoCSA[k]=self.__rhoCSA[k][:-1]
                    
    def del_exp(self,exp_num):
        
        if self.info is not None and self.info_in is not None:
            print('Deleting experiments from the detector object requires disabling the detectors')
            self._disable()
            print('Detectors now disabled')

        if np.size(exp_num)>1:  #Multiple experiments: Just run this function for each experiment
            exp_num=np.atleast_1d(exp_num)
            exp_num[::-1].sort()    #Sorts exp_num in descending order
            for m in exp_num:
                self.del_exp(m)
        else:
            if exp_num==self.n and self.detect_par['R2_ex_corr']:
                self._remove_R2_ex()    #In case we try to delete the last experient, which is R2 exchange, we remove this way
            else:
                if np.ndim(exp_num)>0:
                    exp_num=exp_num[0]
                self.info=self.info.drop(exp_num,axis=1)
                self.info.columns=range(len(self.info.columns))
                if self.__rhoAvg is not None:
                    self.__rhoAvg=np.delete(self.__rhoAvg,exp_num,axis=0)
                nb=self._nb()
                for k in range(nb):
                    self.__rho[k]=np.delete(self.__rho[k],exp_num,axis=0)
                    self.__rhoCSA[k]=np.delete(self.__rhoCSA[k],exp_num,axis=0)
                    
                self.n+=-1
            
                    
    def _disable(self):
        """
        Clears many of the variables that allow a detectors object to be used
        for fitting and for further detector optimization. This is useful when
        passing the detector object as a sensitivity object resulting from a fit.
        The reasoning is that the sensitivity stored in a fit should not be changed
        for any reason, since the fit has already been performed and therefore
        the detector sensitivities should not be changed (hidden, only intended
        for internal use)
        
        Note, there is an added benefit that detectors generated for direct 
        application to MD-derived correlation functions may be rather large, and
        so we save considerable memory here as well (esp. when saving)
        """
        nb=self._nb()
        
        self.__R=None         #Stores experimental sensitivities
        self.__R0=list()
        self.__RCSA=list()      #Stores experimental sensitivities for CSA only
        self.__R0CSA=list()

        self.__r=None

        self.__Rc=None        #Stores the back-calculated sensitivities
        self.__RcAvg=None
        
        self.SVD=None #Stores SVD results
        self.SVDavg=None
            
#        self.MdlPar_in=None
        if self.info_in.shape[1]>100000: #We cleared because the info_in is huge for MD data
            self.info_in=None #Commenting this. Why did we clear these? 
            #Still, the question remains, why would we ever want to keep info_in?

        self.norm=None
        
    def __r_norm(self,bond=None,**kwargs):
        "Applies equal-max or equal-integral normalization"
        if 'NT' in kwargs:
            NT=kwargs.get('NT')
            self.detect_par['Normalization']=NT
        elif 'Normalization' in kwargs:
            NT=kwargs.get('Normalization')
            self.detect_par['Normalization']=NT
        else:
            NT=self.detect_par.get('Normalization')
            
        nb=self._nb()
        if nb==1:
            bond=0
        
        rhoz=self.rhoz(bond)
        
        NT=self.detect_par.get('Normalization')
        nd=self.n
        
        dz=np.diff(self.z()[0:2])
        
        for k in range(0,nd):
            if NT.upper()[0]=='I':
                sc=np.sum(rhoz[k,:])*dz
            elif NT.upper()[0]=='M':
                sc=np.max(rhoz[k,:])
            else:
                print('Normalization type not recognized (use "N" or "I"). Defaulting to equal-max')
                sc=np.max(rhoz[k,:])                

            if bond is None:
                self.__rhoAvg[k,:]=rhoz[k,:]/sc
                self.SVDavg['T'][k,:]=self.SVDavg['T'][k,:]/sc
                self.__rAvg[:,k]=self.__rAvg[:,k]*sc
            else:
                self.__rho[bond][k,:]=rhoz[k,:]/sc
                self.__rhoCSA[bond][k,:]=self.__rhoCSA[bond][k,:]/sc
                self.SVD[bond]['T'][k,:]=self.SVD[bond]['T'][k,:]/sc
                self.__r[bond][:,k]=self.__r[bond][:,k]*sc
                
                
    def __r_info(self,bond=None,**kwargs):
        """ Calculates paramaters describing the detector sensitivities (z0, Del_z,
        and standard deviation of the detectors). Also resorts the detectors by
        z0, unless 'sort_rho' is set to 'no'. Does not return anything, but edits 
        internal values, z0, Del_z, stdev
        """
        
#        nb=np.shape(self.__R)[0]
        nb=self._nb()
        """Trying to determine how many detectors to characterize (possible that 
        not all bonds have same number of detectors. Note- this situation is not
        allowed for data processing.
        """
        
        if np.ndim(bond)>0:
            bond=bond[0]
            
        if bond is None:
            if self.__rAvg is not None:
                nd0=self.__rAvg.shape[1]
            else:
                cont=True
                k=0
                while cont:
                    if self.__r[k] is not None:
                        cont=False
                        nd0=self.__r[k].shape[1]
                    else:
                        k=k+1
                        if k==nb:
                            print('Warning: no detectors are defined. detect.info cannot be calculated')
                            return
        elif self.__r[bond] is not None:
            nd0=self.__r[bond].shape[1]
        elif self.__rAvg is not None:
            nd0=self.__rAvg.shape[1]
        else:
            cont=True
            k=0
            while cont:
                if self.__r[k] is not None:
                    cont=False
                    nd0=self.__r[k].shape[1]
                else:
                    k=k+1
                    if k==nb:
                        print('Warning: no detectors are defined. detect.info cannot be calculated')
                        return
        
        index=[False]*nb
        for k in range(nb):
            if self.__r[k] is not None:
                z0,_,_=self.r_info(k)
                nd=z0.shape[0]
                if nd==nd0:
                    index[k]=True
                    
        a=dict()
        flds=['z0','Del_z','stdev']
        
        if np.any(index):        
            for f in flds:
                x=list()
                x0=getattr(self,f)
                for k in np.where(index)[0]:
                    x.append(x0[k])
                a.update({f : np.mean(x,axis=0)})
                if f!='stdev':
                    a.update({f+'_std':np.std(x,axis=0)})
                
        else:
            "Re-do calculation for average detectors"
            z0,Del_z,stdev=self.r_info(bond=None)
            a.update({'z0':z0,'Del_z':Del_z,'stdev':stdev})
            
        self.info=pd.DataFrame.from_dict(a)
        self.info=self.info.transpose()

    def r_info(self,bond=None,**kwargs):
        """
        |Returns z0, Del_z, and the standard deviation of a detector
        |
        |z0,Del_z,stdev = detect.r_info(bond)
        |
        |If requested, these will be sorted with z0 ascending (set sort_rho='y' 
        |as argument). Note this will also sort the internal detector!
        """
        r=self.r(bond)
        rhoz=self.rhoz(bond)
        if r is not None:
            nd=self.r(bond).shape[1]
            z0=np.divide(np.sum(np.multiply(rhoz,\
                np.repeat([self.z()],nd,axis=0)),axis=1),\
                np.sum(self.rhoz(bond),axis=1))
                        
            iS2=self.detect_par['inclS2']
            R2ex=self.detect_par['R2_ex_corr']
                    
            if iS2 and R2ex:
                i0=np.argsort(z0[1:-1])
                i=np.concatenate(([0],i0+1,[nd-1]))
            elif iS2:
                i0=np.argsort(z0[1:])
                i=np.concatenate(([0],i0+1))
            elif R2ex:
                i0=np.argsort(z0[0:-1])
                i=np.concatenate((i0,[nd-1]))
            else:
                i0=np.argsort(z0)
                i=i0
            if 'sort_rho' not in kwargs or kwargs.get('sort_rho')[0].lower()=='n':
                i0=np.arange(np.size(i0))
                i=np.arange(np.size(i))
                
            z0=z0[i]
            rhoz=rhoz[i,:]
            r=r[:,i]
            Del_z=np.diff(self.z()[0:2])*np.divide(np.sum(rhoz,axis=1),
                          np.max(rhoz,axis=1))
            
            if R2ex:
                #Dummy value for z0 of R2ex
                z0[-1]=0
                Del_z[-1]=0

            

            
            if self.detect_par['inclS2']:
                st0=np.concatenate(([.1],self.info_in.loc['stdev']))
                stdev=np.power(np.dot(np.linalg.pinv(r)**2,st0**2),0.5)
            else:
                stdev=np.power(np.dot(np.linalg.pinv(r)**2,self.info_in.loc['stdev']**2),0.5)
            
            if bond is not None:
                self.z0[bond]=z0
                self.SVD[bond]['T']=self.SVD[bond]['T'][i0,:]
                self.__r[bond]=r
                self.__rho[bond]=rhoz
                self.Del_z[bond]=Del_z
                self.stdev[bond]=stdev
            else:
                self.SVDavg['T']=self.SVDavg['T'][i0,:]
                self.__rAvg=r
                self.__rhoAvg=rhoz
                
            return z0,Del_z,stdev
        else:
            return
    
    def ___r_info(self,bond=None,**kwargs):
        """Calculates some parameters related to the detectors generates, z0,
        Del_z, and standard deviation of resulting detectors. Also resorts the
        detectors according to z0
        """
        nb=self._nb()
        
        match=True
        if self.__r[0].ndim==2:
            nd0=np.shape(self.__r[0])[1]
        else:
            match=False
            nd0=0
            
        stdev=np.zeros(nd0)
        
        if bond is None:
#            a=np.arange(0,nb)
            a=np.arange(0,0)
            match=False
        else:
            a=np.atleast_1d(bond)
            
        for k in a:
            if self.__r[0].ndim==2:
                nd=np.shape(self.__r[k])[1]
            else:
                nd=0
                            
            
            if nd0!=nd:
                match=False
            
            if nd>0:
                z0=np.divide(np.sum(np.multiply(self.__rho[k],\
                        np.repeat([self.z()],nd,axis=0)),axis=1),\
                        np.sum(self.__rho[k],axis=1))
    
                if 'sort_rho' in kwargs and kwargs.get('sort_rho').lower()[0]=='n':
                    i=np.arange(0,np.size(z0))
                    i0=i
                    if self.detect_par['inclS2']:
                        i0=i0[1:]
                    if self.detect_par['R2_ex_corr']:
                        i0=i0[0:-1]
                else:
                    if self.detect_par['inclS2'] and self.detect_par['R2_ex_corr']:
                        i0=np.argsort(z0[1:-1])
                        i=np.concatenate(([0],i0,[np.size(z0)]))
                    elif self.detect_par['inclS2']:
                        i0=np.argsort(z0[1:])
                        i=np.concatenate(([0],i0))
                    elif self.detect_par['R2_ex_corr']:
                        i0=np.argsort(z0[0:-1])
                        i=np.concatenate((i0,[np.size(z0)]))
                    else:
                        i0=np.argsort(z0)
                        i=i0
                
                self.z0[k]=z0[i]
                self.SVD[k]['T']=self.SVD[k]['T'][i0,:]
                
                self.__r[k]=self.__r[k][:,i]
                self.__rho[k]=self.__rho[k][i,:]
                self.Del_z[k]=np.diff(self.z()[0:2])*np.divide(np.sum(self.__rho[k],axis=1),
                          np.max(self.__rho[k],axis=1))
                stdev=np.sqrt(np.dot(self.SVD[k]['T']**2,1/self.SVD[k]['S'][0:np.size(i0)]**2))
                if self.detect_par['inclS2']:
                    "THIS IS WRONG. ADD STANDARD DEVIATION LATER!!!"
                    stdev=np.concatenate(([0],stdev))
                if self.detect_par['R2_ex_corr']:
                    stdev=np.concatenate((stdev,[0]))
                self.SVD[k]['stdev']=stdev
                if match:
                    stdev+=self.SVD[k]['stdev']
                
        if match:
            a=dict()
            a.update({'z0' : np.mean(self.z0,axis=0)})
            a.update({'Del_z' : np.mean(self.Del_z,axis=0)})
            a.update({'stdev' : stdev/nb})
            if nb>1:
                a.update({'z0_std' : np.std(self.z0,axis=0)})
                a.update({'Del_z_std': np.std(self.Del_z,axis=0)})
            else:
                a.update({'z0_std' : np.zeros(nd)})
                a.update({'Del_z_std' : np.zeros(nd)})
                
            self.info=pd.DataFrame.from_dict(a)
            self.info=self.info.transpose()
            
            
            
    
    def r(self,bond=None):
        nb=self._nb()
        if nb==1:
            bond=0
            
        if bond is None:
            if self.__rAvg is None:
                print('First generate the detectors for the average sensitivities')
                return
            else:
                return self.__rAvg
        else:
            if np.size(self.__r[bond])==1:
                print('First generate the detectors for the selected bond')
                return
            else:
                return self.__r[bond]
    
    def rhoz(self,bond=None):
        nb=self._nb()
        if nb==1:
            bond=0
            
        if bond is None:
            if self.__rAvg is None:
                print('First generate the detectors for the average sensitivities')
            else:
                return self.__rhoAvg.copy()
        else:
            if np.size(self.__rho[bond])==1:
                print('First generate the detectors for the selected bond')
                return
            else:
                if bond==-1:
                    return np.array(self.__rho).copy()
                else:
                    return self.__rho[bond].copy()
            
    def Rc(self,bond=None):
        nb=self._nb()
        if nb==1:
            bond=0
            
        if bond is None:
            if self.__RcAvg is None:
                print('First generate the detectors to back-calculate rate constant sensitivities')
            else:
                return self.__RcAvg
        else:
            if np.size(self.__Rc[bond])==1:
                print('First generate the detectors for the selected bond')
                return
            else:
                return self.__Rc[bond]
            
    
    
    def Rin(self,bond=0):
        nb=self._nb()
        if nb==1:
            bond=0
        return self.__R[bond]
    
    def R0in(self,bond=0):
        nb=self._nb()
        if nb==1:
            bond=0
        return self.__R0[bond]
    
    def rho_eff(self,exp_num=None,mdl_num=0,bond=None,**kwargs):
        rho_eff,_=self._rho_eff(exp_num,mdl_num,bond,**kwargs)
        return rho_eff
    
    def rho0(self,exp_num=None,mdl_num=0,bond=None,**kwargs):
        _,rho0=self._rho_eff(exp_num,mdl_num,bond,**kwargs)
        return rho0
    
    def _RCSAin(self,bond=0):
        nb=self._nb()
        if nb==1:
            bond=0
        return self.__RCSA[bond]
    
    def _R0CSAin(self,bond=0):
        nb=self._nb()
        if nb==1:
            bond=0
        return self.__R0CSA[bond]
    
    def retExper(self):
        return self.__exper
    
    def retSpinSys(self):
        return self.__spinsys
        
    def tc(self):
        return self.__tc.copy()
    
    def z(self):
        return np.log10(self.__tc)
    
    def _rho(self,exp_num=None,bond=None):
        """The different children of mdl_sens will have different names for 
        their sensitivities. For example, this class returns rho_z, which are the 
        rate constant sensitivities, but the correlation function class returns
        Ct, and the detector class returns rho. Then, we have a function, 
        _rho(self), that exists and functions the same way in all children
        """
        

             
        if bond is None or self._nb()==1:
            bond=0
        
        if np.size(self.__rho[bond])==1:
            print('First generate the detectors for the selected bond')
            return
                
        if exp_num is None:
            exp_num=self.info.columns
        
        exp_num=np.atleast_1d(exp_num)
        
        
        if bond==-1:
            rhoz=self.rhoz(bond)
            if rhoz.ndim==3:
                rhoz=rhoz[:,exp_num,:]
            elif rhoz.ndim==2:
                rhoz=rhoz[exp_num,:]
        else:
            rhoz=self.rhoz(bond)[exp_num,:]
                
        return rhoz
    
    def _rhoCSA(self,exp_num=None,bond=None):
        """The different children of mdl_sens will have different names for 
        their sensitivities. For example, this class returns R, which are the 
        rate constant sensitivities, but the correlation function class returns
        Ct, and the detector class returns rho. Then, we have a function, 
        _rho(self), that exists and functions the same way in all children
        """
        
        if bond is None:
            bond=0
        
        if np.size(self.__rhoCSA[bond])==1:
            print('First generate the detectors for the selected bond')
            return
        
        
        if exp_num is None:
            exp_num=self.info.columns
        
        exp_num=np.atleast_1d(exp_num)
        
        
        if bond==-1:
            rhoz=np.array(self.__rhoCSA)
            if rhoz.ndim==3:
                rhoz=rhoz[:,exp_num,:]
            elif rhoz.ndim==2:
                rhoz=rhoz[exp_num,:]
            
            if rhoz.shape[0]==1:
                rhoz=rhoz[0]
        else:
            rhoz=self.__rhoCSA[bond]
            rhoz=rhoz[exp_num,:]
            
        
        return rhoz
    
    def plot_rhoz(self,bond=None,rho_index=None,ax=None,norm=False,**kwargs):
        """
        Plots the sensitivities. Options are to specify the bond, the rho_index,
        the 
        """
        hdl=pf.plot_rhoz(self,bond=bond,index=rho_index,ax=ax,norm=norm,**kwargs)
        ax=hdl[0].axes
        ax.set_ylabel(r'$\rho_n(z)$')
        ax.set_title('Detector Sensitivities')
        return hdl
        
    
    def plot_r_opt(self,fig=None):
        if fig is None:
            fig=plt.figure()
        
        ax1=fig.add_subplot(211)
        ax1.plot(self.z(),self.__r_auto.get('Error'))
        max_err=self.__r_auto.get('Error').max()
        
        ax2=fig.add_subplot(212)
        hdls=ax2.plot(self.z(),self.__r_auto.get('rho_z').T)
        
        for index, k in enumerate(np.sort(self.__r_auto.get('Peaks'))):
            ax1.plot(np.repeat(self.z()[k],2),[-max_err/20,0],linewidth=.5,color=[0,0,0])
            ax1.text(self.z()[k],-max_err*1/10,r'$\rho_{'+str(index+1)+'}$',horizontalalignment='center',\
            verticalalignment='center',color=hdls[index].get_color())
            ax2.text(self.z()[k],self.__r_auto.get('rho_z')[index,:].max()+.05,\
            r'$\rho_{'+str(index+1)+'}$',horizontalalignment='center',\
            verticalalignment='center',color=hdls[index].get_color())
        
        ax1.set_xlabel(r'$\log_{10}(\tau$ / s)')
        ax1.set_ylabel(r'Opt. Error, $\Delta$(max)')
        ax1.set_xlim(self.z()[[0,-1]])
        ax1.set_ylim([-max_err*3/20,max_err*21/20])
        
        ax2.set_xlabel(r'$\log_{10}(\tau$ / s)')
        ax2.set_ylabel(r'$\rho_n(z)$')
        min_rho=self.__r_auto.get('rho_z').min()
        max_rho=self.__r_auto.get('rho_z').max()
        
        ax2.set_xlim(self.z()[[0,-1]])
        ax2.set_ylim([min_rho-.05,max_rho+.1])
        
        return hdls 
        
        
        
    def plot_Rc(self,exp_num=None,norm=True,bond=None,ax=None):
        """
        Plots the input sensitivities compared to their reproduction by fitting to
        detectors. Options are to specifiy experiments (exp_num), to normalize (norm),
        to specify a specific bond (bond), and a specific axis to plot onto (ax). 
        
        plot_Rc(exp_num=None,norm=True,bond=None,ax=None)
        """
        
        hdl=pf.plot_Rc(sens=self,exp_num=exp_num,norm=norm,bond=bond,ax=ax)
        
        return hdl

    def _nb(self):
    #    nb=np.shape(self.__R)[0]
        if self.BondSpfc=='yes':
            nb=self.molecule.vXY.shape[0]
        else:
            nb=1
        return nb

def svd0(X,n):
    if np.shape(X)[0]>np.shape(X)[1]:
#        U,S,Vt=svds(X,k=n,tol=0,which='LM')    #Large data sets use sparse svd to avoid memory overload
#        U=U[:,-1::-1]      #svds puts out eigenvalues in opposite order of svd
#        S=S[-1::-1]
#        Vt=Vt[-1::-1,:]
        S2,V=eigs(np.dot(np.transpose(X),X),k=n)
        S=np.sqrt(S2.real)
        U=np.dot(np.dot(X,V.real),np.diag(1/S))
        Vt=V.real.T
    else:
        U,S,Vt=svd(X)       #But, typically better results from full calculation
        U=U[:,0:np.size(S)] #Drop all the empty vectors
        Vt=Vt[0:np.size(S),:]
   
    return U,S,Vt



    
def linprog_par(Y):
    """This function optimizes a detector sensitivity that has a value of 1
    at correlation time k, and cannot go below some specific value at all
    other correlation times (usually 0). While satisfying these 
    requirements, the sensitivity is minimized.
    """
    Vt=Y[0]
    k=Y[1]
    ntc=np.shape(Vt)[1]
    
    if np.size(Y)==3:
        target=Y[2]
    else:
        target=np.zeros(ntc)
        
    try:
#        if k<Vt.shape[1]/2:
#            equals=Vt[:,[k,-1]].T
#            v=[1,0]
#        else:
#            equals=Vt[:,[0,k]].T
#            v=[0,1]
#        if k==Vt.shape[1]-1:
#            equals=[Vt[:,k].T]
#            v=1
#        else:
#            equals=Vt[:,[k,-1]].T
#            v=[1,0]

#        x=linprog(np.sum(Vt,axis=1),-Vt.T,-target,equals,v,bounds=(-500,500),method='interior-point',options={'disp' :False,})
        x=linprog(np.sum(Vt,axis=1),-Vt.T,-target,[Vt[:,k]],1,bounds=(-500,500),method='interior-point',options={'disp' :False,})
        x=x['x']
        if np.any(np.dot(Vt.T,x)<(np.min(target)-.001)):
            x=np.ones(Vt.shape[0])
    except:
        x=np.ones(Vt.shape[0])
        
    return x

def lsqlin_par(Y):
    Vt=Y[0]
    target=Y[1]
    
    nSVD=np.shape(Vt)[0]
    n=np.shape(target)[0]
    
    T=np.zeros([nSVD,nSVD])
    

    for k in range(0,n):        
        x0=lsqlin(np.transpose(Vt),target[k,:],lsq_solver='exact')
        T[k,:]=x0['x']
        
    for k in range(n,nSVD):
#        a=np.argmin(np.sum(T**2,axis=0))
#        T[k,a]=1
        T[k,k]=1
    
    return T
