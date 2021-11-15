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

Created on Tue May  7 16:51:28 2019

@author: albertsmith
"""

import numpy as np
import pandas as pd
from pyDIFRATE.r_class.Ctsens import Ct
from pyDIFRATE.r_class.detectors import detect
from pyDIFRATE.chimera.chimeraX_funs import run_chimeraX,get_default_colors
import pyDIFRATE.plots.plotting_funs as pf
from pyDIFRATE.data.fitting import fit_data
from pyDIFRATE.data.bin_in_out import save_DIFRATE
import copy

class data(object):
#%% Basic container for experimental and MD data
    def __init__(self,**kwargs):
        
        self.vars=dict()    #Location for extra variables by name
        
        self.label=None
        self.chi2=None
        
        self.R=None
        self.R_std=None
        self.R_u=None
        self.R_l=None
        self.conf=0.68
        
        self.S2=None
        self._S2=None   #Hidden location for S2 calc in case we don't include it
        self.S2_std=None
        self.tot_cc=None
        self.tot_cc_norm=None
        
        self.Rcc=list()
        self.Rcc_norm=list()
        
        self.Rin=None
        self.Rin_std=None
        self.Rc=None
        
        self.S2in=None
        self.S2in_std=None
        self.S2c=None
        
        self.ired=None
        self.type=None
        
        self.sens=None
        self.detect=None
        
        
        
        self.load(**kwargs)
#%% Some options for loading in data        
    def load(self,subS2=False,**kwargs):
        EstErr=False #Default don't estimate error. This is overridden for 'Ct'
        "Load in correlation functions from an iRED calculation"
        if 'iRED' in kwargs:
            ired=kwargs['iRED']
            self.ired=ired
            self.R=ired['DelCt']
            del ired['DelCt']
            nt=ired['t'].size
            
            if subS2:
                self.sens=Ct(t=ired['t'],S2=None,**kwargs)
            else:
                self.sens=Ct(t=ired['t'],**kwargs)
            
            if 'N' in self.ired:
                stdev=1/np.sqrt(self.ired['N'])
                stdev[0]=1e-6
                self.sens.info.loc['stdev']=stdev
                self.R_std=np.repeat([stdev],self.R.shape[0],axis=0)
            else:
                norm=1/(self.ired.get('Ct')[:,0]-self.ired.get('CtInf'))
                norm=np.transpose([norm/norm[0:-(self.ired.get('rank')*2+1)].mean()])

                self.R_std=np.dot(norm,[self.sens.info.loc['stdev']])
            
        elif 'Ct' in kwargs:
            EstErr=False #Default estimate error for correlation functions
            ct=kwargs.get('Ct')
            self.R=ct.get('Ct')

            
            if not(subS2) and 'S2' in kwargs:
                "Removes the S2 correction"
                self._S2=kwargs.pop('S2')
                                
            self.sens=Ct(t=ct.get('t'),**kwargs)
            nt=ct['t'].size
            if 'R_std' in ct:
                self.R_std=ct['R_std']
                EstErr=False
            elif 'N' in ct:
                stdev=1/np.sqrt(ct['N'])
                stdev[0]=1e-6
                self.sens.info.loc['stdev']=stdev
                self.new_detect()
                self.R_std=np.repeat([stdev],self.R.shape[0],axis=0)
            else:
                self.R_std=np.repeat([self.sens.info.loc['stdev']],self.R.shape[0],axis=0)
            if 'S2' in kwargs:
                self.S2=kwargs.get('S2')
            
            
        if 'EstErr' in kwargs:
            if kwargs.get('EstErr')[0].lower()=='n':
                EstErr=False
            elif kwargs.get('EstErr')[0].lower()=='y':
                EstErr=True

        if self.sens is not None:
            self.detect=detect(self.sens)
            
        if EstErr:
            try:
                self.detect.r_no_opt(np.min([15,nt]))
                fit0=self.fit()
#                plt.plot(self.sens.t(),self.R[0,:])
#                plt.plot(self.sens.t(),fit0.Rc[0,:])
                self.R_std=np.sqrt((1/nt)*(np.atleast_2d(fit0.chi)*self.R_std.T**2)).T
    #            self.R_std[:,0]=self.R_std.min()/1e3
                self.sens.info.loc['stdev']=np.median(self.R_std,axis=0)
                self.detect=detect(self.sens)
            except:
                print('Warning: Error estimation failed')

            
        if 'molecule' in kwargs:
            mol=kwargs.get('molecule')
            if self is not None:
                self.sens.molecule=mol
                self.detect.molecule=mol
                if self.sens.molecule.sel1 is not None and self.sens.molecule.sel2 is not None:
                    self.sens.molecule.set_selection()
            self.label=mol.label
    
    def new_detect(self,mdl_num=None,sens=None,exp_num=None):
        """
        Creates a new detector object. Usually for updating the detectors after
        a new model has been created (Typically, we create the data object
        with the correct sensitivity object already in place, such that it 
        doesn't make sens to update the detectors unless some model of motion
        is changed. However, if this is not the case, then the detector object
        may also need to be updated)
        
        data.detect(mdl_num=None,sens=None,exp_num=None)
        
        mdl_num and exp_num should either be the same length or mdl_num should
        just have one element
        """
        
        if sens is not None:
            self.detect=detect(sens,exp_num=exp_num,mdl_num=mdl_num)
        else:
            self.detect=detect(self.sens,exp_num=exp_num,mdl_num=mdl_num)
    
#%% Option for deleting experiments
    def del_exp(self,exp_num):
        """
        |Deletes an experiment or experiments
        |obj.del_exp(exp_num)
        |
        |Note that this method will automatically delete the detector object,
        |since it is no longer valid after deletion of an experiment. Add it back
        |with obj.new_detect()
        """
        
        if hasattr(exp_num,'__len__'):
            exp_num=np.atleast_1d(exp_num)
            exp_num[::-1].sort()
            for m in exp_num:
                self.del_exp(m)
        else:
            if self.R is not None and self.R.shape[1]>exp_num:
                self.R=np.delete(self.R,exp_num,axis=1)
                if self.R_std is not None:
                    self.R_std=np.delete(self.R_std,exp_num,axis=1)
                if self.R_u is not None:
                    self.R_u=np.delete(self.R_u,exp_num,axis=1)
                if self.R_l is not None:
                    self.R_l=np.delete(self.R_l,exp_num,axis=1)

                if self.sens is not None:
                    self.sens.del_exp(exp_num)
                self.new_detect() #Detectors are no longer valid, and so are reset here
            else:
                print('Warning: exp_num {0} was not found'.format(exp_num))
                
        
#%% Option for deleting a data point or points
    def del_data_pt(self,index):
        """
        Deletes a particular residue number (or list of number), given their 
        index (or indices). Deletes values out of R, R_std, R_l, R_u, Rc, Rin, 
        and Rin_std.
        
        obj.del_data_pt(index)
        
        Warning: This will not edit the selections in the sensitivity's molecule
        object, or delete bond-specific sensitivities
        """
        
        if np.size(index)>1:
            index=np.atleast_1d(index)
            index[::-1].sort()
            for m in index:
                self.del_data_pt(m)
        else:
            if index>=self.R.shape[0]:
                print('Warning: index of {0} is greater than or equal to the number of data points ({1})'.format(index,self.R.shape[0]))
                return
            attr=['R','R_l','R_u','R_std','Rc','Rin','Rin_std','label',\
                  'S2','S2_std','S2c','S2in','S2in_std']
            for at in attr:
                if hasattr(self,at):
                    x=getattr(self,at)
                    if x is not None:
                        setattr(self,at,np.delete(x,index,axis=0))
                
#%% Run fit_data from the object     
    def fit(self,detect=None,**kwargs):
        if detect is None:
            detect=self.detect
            
        return fit_data(self,detect,**kwargs)
      
#%% Convert iRED data types into normal detector responses and cross-correlation matrices            
    def iRED2rho(self,mode_index=None):
        """
        Convert the fitting of iRED data to the auto-correlated detectors and
        cross correlation matrices. By default, omits the last 2*rank+1 modes 
        (3 or 5 modes). Alternatively, the user may provide a boolean array with
        size equal to the number of modes, or a list of the modes to be used
        
        fit=fit0.iRED2rho()
        
        or
        
        fit=fit0.iRED2rho(mode_index)
        """
        if self.ired is None or not isinstance(self.sens,detect):
            print('Function only applicable to iRED-derived detector responses')
            return
        
        out=data()

        nd=self.sens.rhoz(bond=0).shape[0]
        nb=self.R.shape[0]
                
        rank=self.ired.get('rank')
        ne=2*rank+1
        
        if mode_index is None:
            mode_index=np.ones(nb,dtype=bool)
            mode_index[-ne:]=False
        else:
            if len(mode_index)==self.ired['M'].shape[0]:
                mode_index=np.array(mode_index,dtype=bool)
            else:
                mo=mode_index.copy()
                mode_index=np.zeros(nb,dtype=bool)
                mode_index[mo]=True
        
        
#        if self.sens.molecule.sel1in is not None:
#            nb0=np.size(self.sens.molecule.sel1in)
#        elif self.sens.molecule.sel2in is not None:
#            nb0=np.size(self.sens.molecule.sel2in)
#        else:
#            nb0=self.sens.molecule.sel1.n_atoms
        nb0=self.R.shape[0]-self.ired['n_added_vecs']
        
        out.R=np.zeros([nb0,nd])
        out.R_std=np.zeros([nb0,nd])
        out.R_l=np.zeros([nb0,nd])
        out.R_u=np.zeros([nb0,nd])
        
        for k in range(0,nd):
            lambda_rho=np.repeat([self.ired.get('lambda')[mode_index]*self.R[mode_index,k]],nb0,axis=0)
            out.R[:,k]=np.sum(lambda_rho*self.ired.get('m')[0:nb0,mode_index]**2,axis=1)
             
            lambda_rho=np.repeat([self.ired.get('lambda')[mode_index]*self.R_std[mode_index,k]],nb0,axis=0)
            out.R_std[:,k]=np.sqrt(np.sum(lambda_rho*self.ired.get('m')[0:nb0,mode_index]**2,axis=1))
            
            lambda_rho=np.repeat([self.ired.get('lambda')[mode_index]*self.R_l[mode_index,k]],nb0,axis=0)
            out.R_l[:,k]=np.sqrt(np.sum(lambda_rho*self.ired.get('m')[0:nb0,mode_index]**2,axis=1))
            
            lambda_rho=np.repeat([self.ired.get('lambda')[mode_index]*self.R_u[mode_index,k]],nb0,axis=0)
            out.R_u[:,k]=np.sqrt(np.sum(lambda_rho*self.ired.get('m')[0:nb0,mode_index]**2,axis=1))
        
        
        out.sens=self.sens
        
        "Pre-allocate nd matrices for the cross-correlation calculations"
        out.tot_cc=np.zeros([nb0,nb0])
        for k in range(0,nd):
            out.Rcc.append(np.zeros([nb0,nb0]))
        "Loop over all eigenmodes"
        for k in np.argwhere(mode_index).squeeze(): #We only use the user-specified modes (or by default, leave at last 2*rank+1 modes)
            m=self.ired.get('m')[0:nb0,k]
            mat=self.ired.get('lambda')[k]*np.dot(np.transpose([m]),[m])
            "Calculate total correlation"
            out.tot_cc+=mat
            "Loop over all detectors"
            for m in range(0,nd):
                out.Rcc[m]+=mat*self.R[k,m]
                
        "Calculate the normalized correlation"
        dg=np.sqrt([np.diag(out.tot_cc)])
        out.tot_cc_norm=out.tot_cc/np.dot(np.transpose(dg),dg)
        for k in range(0,nd):
            dg=np.sqrt([np.diag(out.Rcc[k])])
            out.Rcc_norm.append(out.Rcc[k]/np.dot(np.transpose(dg),dg))
        
        if self.label is not None:
            out.label=self.label    
        elif self.sens is not None and np.size(self.sens.molecule.label)!=0:
            out.label=self.sens.molecule.label
            
        "Calculate the order parameters"

        lda=np.repeat([self.ired.get('lambda')[0:-ne]],nb0,axis=0)
        out.S2=1-np.sum(lda*self.ired.get('m')[0:nb0,0:-ne]**2,axis=1)

        return out
    
    def plot_rho(self,fig=None,plot_sens=True,index=None,rho_index=None,errorbars=False,style='plot',**kwargs):
        """
        Plots the full series of detector responses
        Arguments:
            fig: Specify which figure to plot into
            plot_sens (True/False): Plot the sensitivity at the top of the figure
            index: Specify which residues to plot
            rho_index: Specify which detectors to plot
            errobars (True/False): Display error bars
            style ('p'/'s'/'b'): Plot style (line plot, scatter plot, bar plot)
            **kwargs: Plotting arguments (passed to plotting functions)
        """
        return pf.plot_rho_series(self,fig,plot_sens,index,rho_index,errorbars,style,**kwargs)
            
    def plot_cc(self,det_num,cutoff=None,ax=None,norm=True,index=None,**kwargs):
        if np.size(self.Rcc)==0:
            print('Data object does not contain cross-correlation data')
            print('First, create a data object from iRED analysis (data=iRED2data(...))')
            print('Then, analyze with detectors, data.r_auto(...);fit0=data.fit(...)')
            print('Finally, convert fit into normal detector responses, fit=fit0.iRED2rho()')
            return
        
        if det_num is None:
            x=self.tot_cc
        else:
            x=self.Rcc[det_num]
        if index is None:
            index=np.arange(x.shape[0])
         
        ax=pf.plot_cc(x,self.label,ax,norm,index,**kwargs) 
        if det_num is None:
            ax.set_title('Total cross correlation')
        else:
            ax.set_title(r'Cross correlation for $\rho_{' + '{}'.format(det_num) + '}$')   
        
        return ax
      
    def plot_fit(self,errorbars=True,index=None,exp_index=None,fig=None,style='log',**kwargs):
        """
        Plots the fit quality of the input data. This produces bar plots with 
        errorbars for the input data and scatter points for the fit, in the case
        of experimental data. If correlation functions are being fit, then
        line plots (without errorbars) are used (specify style as log or linear,
        log is default)
        
        One may specify the residue index and also the index of experiments to
        be plotted
        
        plot_fit(errorbars=True,index=None,exp_index=None,fig=None,ax=None)
        """
        
        "Return if data missing"
        if self.Rc is None:
            print('data object is not a fit or calculated values are not stored')
            return
        
        info=self.sens.info_in
        
        if info is None or 't' in info.index.values:
            if info is None:
                t=np.arange(1,self.Rin.shape[1]+1)
            else:
                t=info.loc['t'].to_numpy()
            ax=pf.plot_all_Ct(t,Ct=self.Rin,Ct_fit=self.Rc,lbl=self.label,index=index,fig=fig,style=style,**kwargs)
        else:
            if self.S2c is not None:
                Rin=np.concatenate((self.Rin,np.atleast_2d(self.S2in).T),1)
                Rin_std=np.concatenate((self.Rin_std,np.atleast_2d(self.S2in_std).T),1)
                Rc=np.concatenate((self.Rc,np.atleast_2d(self.S2c).T),1)
                info0=info[0].copy()
                for a,b in info0.items():
                    info0[a]='' if isinstance(b,str) else 0
                info0['Type']='S2'
                info=pd.concat((info,info0),1,ignore_index=True)
                    
            else:
                Rin,Rin_std,Rc=self.Rin,self.Rin_std,self.Rc
            if not(errorbars):Rin_std=None
            
            ax=pf.plot_fit(self.label,Rin,Rc,Rin_std,info,index,exp_index,fig)
        
        return ax
    
    def draw_cc3D(self,bond,det_num=None,index=None,scaling=1,norm=True,absval=True,\
                   disp_mode=None,chimera_cmds=None,fileout=None,save_opts=None,\
                   scene=None,x0=None,colors=None):
        
        if self.label is None:
            print('User has not defined any bond labels, bond will now refer to the absolute index')
            assert bond<self.R.shape[0],'Invalid bond selection (0<=bond<{0})'.format(self.R.shape[0])
            i=bond
        else:
            assert np.any(bond==np.array(self.label)),'bond not found in self.label'
            i=np.argwhere(bond==np.array(self.label))[0,0]
        

        x=(self.Rcc_norm[det_num][i] if det_num else self.tot_cc_norm[i]) if norm\
            else (self.Rcc[det_num][i] if det_num else self.tot_cc[i])
        
        if absval:
            x=np.abs(x)
        else:
            x0=np.array([-np.max(np.abs(x)),np.max(np.abs(x))] if colors else [-np.max(np.abs(x)),0,np.max(np.abs(x))])
        
        if colors is None:
#            colors=[[210,180,140,255],[255,100,0,255]] if absval else [[0,100,255,255],[210,180,140,255],[255,100,0,255]]
            colors=[[210,180,140,255],get_default_colors(det_num) if det_num is not None else [255,100,0,255]]\
                if absval else [[0,0,200,255],[210,180,140,255],[200,0,0,255]]
                 
        mol=self.sens.molecule
        if index is not None:
            index=np.unique(np.concatenate((index,[i]))) #Make sure bond is within index
            if np.max(index)==1 and len(index)>2:
                index=np.array(index,dtype=bool)
            else:
                index=np.array(index,dtype=int)
            s1,s2=mol.sel1.copy(),mol.sel2.copy()
            mol.sel1,mol.sel2=mol.sel1[index],mol.sel2[index]
            x=x[index]
            i=np.argwhere((self.label[index]==bond) if self.label is not None else (i==index))[0,0]

        x*=scaling
        
        run_chimeraX(mol=mol,disp_mode=disp_mode,x=x,chimera_cmds=chimera_cmds,\
                     fileout=fileout,save_opts=save_opts,scene=scene,x0=x0,
                     colors=colors,marker=i)
        if index is not None:mol.sel1,mol.sel2=s1,s2
            
#    def draw_cc3D(self,bond,det_num=None,chain=None,fileout=None,scaling=None,norm='y',**kwargs):
#        "bond is the user-defined label! Not the absolute index..."
#
#        if self.label is None:
#            print('User has not defined any bond labels, bond will now refer to the absolute index')
#            index=bond
#        elif any(np.atleast_1d(self.label)==bond):
#            index=np.where(np.array(self.label)==bond)[0][0]
#        else:
#            print('Invalid bond selection')
#            return
#            
#        if norm.lower()[0]=='y':
#            if det_num is None:
#                values=self.tot_cc_norm[index,:]
#            else:
#                values=self.Rcc_norm[det_num][index,:]
#        else:
#            if det_num is None:
#                values=self.tot_cc[index,:]
#            else:
#                values=self.Rcc[det_num][index,:]
#        
#        "Take absolute value- I'm not convinced about this yet..."
#        values=np.abs(values)
#
#        if scaling is None:
##            "Default is to scale to the maximum of all correlations"
##            scale0=0
##            for k in range(0,np.shape(self.Rcc_norm)[0]):
##                a=self.Rcc_norm[k]-np.eye(np.shape(self.Rcc_norm)[1])
##                scale0=np.max([scale0,np.max(np.abs(a))])
#            if norm.lower()[0]=='y':
##                if det_num is None:
##                    scale0=np.max(np.abs(self.tot_cc_norm)-np.eye(np.shape(self.tot_cc_norm)[0]))
##                else:
##                    scale0=np.max(np.abs(self.Rcc_norm[det_num]-np.eye(np.shape(self.Rcc_norm)[1])))
#                scale0=1
#            else:
#                scale0=np.max(np.abs(values))
#            scaling=1/scale0
#
#        res1=self.sens.molecule.sel1.resids
#        chain1=self.sens.molecule.sel1.segids
#        res2=self.sens.molecule.sel2.resids
#        chain2=self.sens.molecule.sel2.segids
#
#        color_scheme=kwargs.pop('color_scheme') if 'color_scheme' in kwargs else 'blue'
#            
#        if np.all(self.sens.molecule.sel1.names=='N') or np.all(self.sens.molecule.sel2.names=='N') and\
#            np.all(res1==res2) and np.all(chain1==chain2):
#            style='pp'
#        else:
#            style='bond'
#
#        if style=='pp':
#            "Color the whole peptide plane one color"
#            resi=res1
#            chain=chain1
#            plt_cc3D(self.sens.molecule,resi,values,resi0=bond,chain=chain,chain0=chain[index],\
#                     fileout=fileout,scaling=scaling,color_scheme=color_scheme,style=style,**kwargs)
#        else:
#            "Color the individual bonds specified in the molecule selections"
#            "I'm not sure the indexing of resi0 is correct here!!!"
#            plt_cc3D(self.sens.molecule,None,values,resi0=index,scaling=scaling,color_scheme=color_scheme,style=style,**kwargs)
#            """I'm going in circles here for some reason. Just switched resi0=res[index]
#            to resi0=index. plot_cc in 'bond' mode expects the index found in molecule.sel1
#            and molecule.sel2. So this seems like it should be correct...but let's see
#            if it glitches again for lipids"""
##            print('Selections over multiple residues/chains- not currently implemented')
#        
        
        
#    def draw_rho3D(self,det_num=None,resi=None,fileout=None,scaling=None,**kwargs):
#        
#        if det_num is None:
#            values=1-self.S2
#        else:
#            values=self.R[:,det_num]
#        
#      
#        res1=self.sens.molecule.sel1.resids
#        chain1=self.sens.molecule.sel1.segids
#        res2=self.sens.molecule.sel2.resids
#        chain2=self.sens.molecule.sel2.segids
#        
#
#        if np.size(res1)==np.size(res2) and (np.all(res1==res2) and np.all(chain1==chain2)):
#            resi0=resi
#            resi=res1
#            chain=chain1
##            chain[chain=='PROA']='p'
#            
#            
#            
#            if resi0 is not None:
#                index=np.in1d(resi,resi0)
#                resi=resi[index]
#                chain=chain[index]
#                values=values[index]
#              
#            if scaling is None:
#                scale0=np.max(values)
#                scaling=1/scale0
#                
#            plot_rho(self.sens.molecule,resi,values,chain=chain,\
#                     fileout=fileout,scaling=scaling,**kwargs)
#                
#        else:
#            if scaling is None:
#                scale0=np.max(values)
#                scaling=1/scale0
#            
#            plot_rho(self.sens.molecule,None,values,scaling=scaling,**kwargs)
##            print('Selections over multiple residues/chains- not currently implemented')
            
    def draw_rho3D(self,det_num=None,index=None,scaling=1,disp_mode=None,\
                   chimera_cmds=None,fileout=None,save_opts=None,\
                   scene=None,x0=None,colors=None):
        
        if colors is None:
            if det_num is None:
                colors=[[255,255,0,255],[255,0,0,255]]
            else:
                clr=get_default_colors(det_num)
                colors=[np.array([210,180,140,255]),clr]

        
        if det_num is None:
            x=1-self.S2.copy()
        else:
            x=self.R[:,det_num].copy()
         
        mol=self.sens.molecule
        if index is not None:
            if np.max(index)==1 and len(index)>2:
                index=np.array(index,dtype=bool)
            else:
                index=np.array(index,dtype=int)
            s1,s2=mol.sel1,mol.sel2
            mol.sel1=mol.sel1[index]
            mol.sel2=mol.sel2[index]
            x=x[index]

        x*=scaling
        x[x<0]=0
        
        run_chimeraX(mol=mol,disp_mode=disp_mode,x=x,chimera_cmds=chimera_cmds,\
                     fileout=fileout,save_opts=save_opts,scene=scene,x0=x0,
                     colors=colors)
        if index is not None:mol.sel1,mol.sel2=s1,s2
            
    def draw_mode(self,mode_num=None,resi=None,fileout=None,scaling=None,**kwargs):
        

        values=self.ired['m'][mode_num]
#        values=values/np.abs(values).max()
        if self.ired['n_added_vecs']!=0:
            values=values[0:-self.ired['n_added_vecs']]
      
        res1=self.sens.molecule.sel1.resids
        chain1=self.sens.molecule.sel1.segids
        res2=self.sens.molecule.sel2.resids
        chain2=self.sens.molecule.sel2.segids
        

        if np.size(res1)==np.size(res2) and (np.all(res1==res2) and np.all(chain1==chain2)):
            resi0=resi
            resi=res1
            chain=chain1
#            chain[chain=='PROA']='p'
            
            
            
            if resi0 is not None:
                index=np.in1d(resi,resi0)
                resi=resi[index]
                chain=chain[index]
                values=values[index]
              
            if scaling is None:
                scale0=np.max(values)
                scaling=1/scale0
                
            plot_rho(self.sens.molecule,resi,values,chain=chain,\
                     fileout=fileout,scaling=scaling,color_scheme='rb',**kwargs)
                
        else:
            if scaling is None:
                scale0=np.max(values)
                scaling=1/scale0
            
            plot_rho(self.sens.molecule,None,values,scaling=scaling,color_scheme='rb',**kwargs)
#            print('Selections over multiple residues/chains- not currently implemented')        
        
    def save(self,filename):
        """
        |Save data to filename
        |self.save(filename)
        """
        save_DIFRATE(filename,self)
        
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
            if self.sens is not None and self.detect is not None:
                mol=self.sens.molecule
                self.sens.molecule=None
                self.detect.molecule=None
                out=copy.deepcopy(self)
                self.sens.molecule=mol
                self.detect.molecule=mol
                out.sens.molecule=mol
                out.detect.molecule=mol
            elif self.sens is not None:
                mol=self.sens.molecule
                self.sens.molecule=None
                out=copy.deepcopy(self)
                self.sens.molecule=mol
                out.sens.molecule=mol
            else:
                out=copy.deepcopy(self)
            
        return out
        
    def print2text(self,filename,variables=['label','R','R_std'],precision=4):
        """
        Prints data to a text file, specified by filename. The user may specify
        which variables to print (default: label, R, and R_std)
        """
        form='{0:.{1}f}'
        
        with open(filename,'w') as f:
            f.write('data')
            for v in variables:
                f.write('\n'+v)
                X=np.array(getattr(self,v))
                if X.ndim==1:
                    sz0=np.size(X)
                    if isinstance(X[0],str):
                        for k in range(sz0):
                            f.write('\n'+X[k])
                    else:
                        for k in range(sz0):
                            f.write('\n'+form.format(X[k],precision))
                elif X.ndim==2:
                    sz0,sz1=np.shape(X)
                    for k in range(sz0):
                        for m in range(sz1):
                            if m==0:
                                f.write('\n')
                            else:
                                f.write('\t')
                            f.write(form.format(X[k,m],precision))
                f.write('\n')
                
            
            
#            f.write('\nlabel')
#            sz0=np.size(self.label)
#            for k in range(sz0):
#                f.write('\n{0}'.format(self.label[k]))
#            f.write('\nR')
#            sz0,sz1=self.R.shape
#            for k in range(sz0):
#                for m in range(sz1):
#                    if m==0:
#                        f.write('\n')
#                    else:
#                        f.write('\t')   
#                    f.write(form.format(self.R[k,m],precision))
#            f.write('\nRstd')
#            for k in range(sz0):
#                for m in range(sz1):
#                    if m==0:
#                        f.write('\n')
#                    else:
#                        f.write('\t')   
#                    f.write(form.format(self.R_std[k,m],precision))
#            if conf[0].lower()=='y' and self.R_l is not None:
#                f.write('\nR_l, conf={0}'.format(self.conf))
#                for k in range(sz0):
#                    for m in range(sz1):
#                        if m==0:
#                            f.write('\n')
#                        else:
#                            f.write('\t')   
#                        f.write(form.format(self.R_l[k,m],precision))
#                f.write('\nR_u, conf={0}'.format(self.conf))
#                for k in range(sz0):
#                    for m in range(sz1):
#                        if m==0:
#                            f.write('\n')
#                        else:
#                            f.write('\t')   
#                        f.write(form.format(self.R[k,m],precision))
#            