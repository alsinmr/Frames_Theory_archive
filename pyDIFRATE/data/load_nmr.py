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

Functions for loading data from files

Created on Mon Jul 29 14:14:04 2019

@author: albertsmith
"""

import numpy as np
from pyDIFRATE.r_class.sens import rates
from pyDIFRATE.r_class.detectors import detect
from pyDIFRATE.data import data_class as dc
#import pyDIFRATE.data.data_class as dc

def load_NMR(filename):
    """
    |load_NMR loads NMR data and experimental information from a text file
    |data = load_NMR(filename)
    |
    |The file can contain experimental data, information about those experiments,
    |and information about models (currently only isotropic models)
    |Each part of the file is initiated by keywords 'data', 'info', or 'model'
    |A section terminates at the occurence of another one of the keywords or at
    |the end of the file. 
    |
    |Within the data section, 'R', 'Rstd', and 'label' initiate sections 
    |containing data (contains all data, regardless of whether that data is 
    |usually referred to as a rate, ex. order parameters), standard deviation of
    |the data, and labels. 'R' and 'Rstd' can be input as a matrix, where each
    |column is a different experiment. One may also have multiple 'R' and 'Rstd'
    |sections, where each new section will be appended as a new set of experiments
    |The 'label' section is a single column with numbers or strings, naming the
    |different residues or bonds. This is only used for plot labeling.
    |
    |Within the info section, one inputs parameters describing a given experiment
    |for a given system. 'Type','v0','v1','vr','offset','stdev','Nuc','Nuc1','dXY',
    |'CSA','QC','eta','theta' all are possible parameters. These should appear on
    |one line with the following line containing the value for the parameter. 
    |Multiple experiments may be entered by including multiple parameters for the
    |experimental parameters (although only one spin system may be entered this way)
    |However, if a parameter is repeated in the 'info' section, this will initiate
    |entry of a new set of experiments.
    |
    |Isotropic models may be entered in the model section. However, a molecule needs
    |to be loaded before entry of anisotropic models, so that it is not currently
    |possible to enter anisotropic model parameters from a file. Within the model
    |section, tthe first line names the model, and subsequent lines come in pairs,
    |where we name the parameter and give its value in the second line
    |
    |An example file would look like
    |
    |data
    |R
    |1.6337    1.6337    3.4796    2.1221
    |2.2000    2.0245    9.3194    3.6051
    |...
    |0.2500    0.2900    4.0425    2.0151
    |
    |Rstd
    |0.0327    0.0327    0.0696    0.0424
    |0.0440    0.0405    0.1864    0.0721
    |...
    |0.0050    0.0058    0.0808    0.0403
    |
    |label
    |gamma
    |beta
    |...
    |C16
    |
    |info
    |Type
    |R1
    |v0
    |500 800
    |Type
    |R1p
    |v0
    |600
    |vr
    |10
    |v1
    |15,35
    |
    |model
    |IsoDif
    |tM
    |4.84e-9
    """
    
    keys0=np.array(['info','data','model'])
    
    data=dc.data()
    data.sens=rates()
    
    rate_args=list()
    mdl_args=list()
    
    with open(filename,'r') as f:
        while not eof(f):
            a=f.readline()
            if a.strip().lower()=='info':
                rate_args=read_info(f,keys0)
                for k in rate_args:
                    data.sens.new_exp(**k)
            elif a.strip().lower()=='data':
                R,Rstd,label,S2,S2_std=read_data(f,keys0)
                data.R=R
                data.R_std=Rstd
                data.label=label
                if S2 is not None:
                    data.S2=S2
                    data.S2_std=S2_std
            elif a.strip().lower()=='model':
                mdl_args.append(read_model(f,keys0))
    
    mdl=False
    for mdls in mdl_args:
        data.sens.new_mdl(**mdls)
        mdl=True
    
    if mdl:
        data.detect=detect(data.sens,mdl_num=0)
    else:
        data.detect=detect(data.sens)
        
    if data.sens.info.shape[1]!=0:
        if data.sens.info.shape[1]!=data.R.shape[1]:
            print('Warning: number of data sets does not match number of experiments in info')
        else:
            for k in range(data.sens.info.shape[1]):
                if data.sens.info.loc['stdev'][k] is None:
                    data.sens.info.loc['stdev'][k]=np.median(data.R_std[:,k]/data.R[:,k])
                           
    return data

def load_NMR_info(filename):
    """
    |load_NMR_info loads a description of NMR experiments from a file. Formatting
    |is as for load_NMR (which loads a full data set and experimental info).
    |
    |rates = load_NMR_info(filename)
    |
    |Note that this simply calls load_NMR, and extracts the 'sens' object from
    |data
    """
    data=load_NMR(filename)
    rates=data.sens
    
    return rates

def read_data(f,keys0):    
    """
    Reads data out of a file (called by load_NMR)
    """
    cont=True
    R=list()
    Rstd=list()
    S2=list()
    S2_std=list()
    label=None
    ne=0
    
    keys1=['R','Rstd','label','R_std','S2','S2_std']
    
    while not(eof(f)) and cont:
        pos=f.tell()
        a=f.readline()
        
        if np.isin(a.strip(),keys1):
            if a.strip()=='R':
                R.append(read_lines(f,np.concatenate((keys0,keys1))))
            elif a.strip()=='Rstd' or a.strip()=='R_std':
                Rstd.append(read_lines(f,np.concatenate((keys0,keys1))))
            elif a.strip()=='label':
                label=read_label(f,np.concatenate((keys0,keys1)))
            elif a.strip()=='S2':
                S2.append(read_lines(f,np.concatenate((keys0,keys1))))
            elif a.strip()=='S2_std':
                S2_std.append(read_lines(f,np.concatenate((keys0,keys1))))
        elif np.isin(a.strip(),keys0):
            cont=False
            f.seek(pos)
    
    if np.size(R)!=0:
        R=np.concatenate(R,axis=1)
    else:
        R=None
        print('Warning: no data found in data entry')
        return None,None

    if np.size(Rstd)!=0:
        Rstd=np.concatenate(Rstd,axis=1)
    else:
        Rstd=None
    if len(S2)!=0:
        S2=np.atleast_1d(np.concatenate(S2,axis=0).squeeze())
    else:
        S2=None
        
    if len(S2_std)!=0:
        S2_std=np.atleast_1d(np.concatenate(S2_std,axis=0).squeeze())
    else:
        S2_std=None
        

    if Rstd is None:
        print('Warning: Standard deviations are not provided')
        print('Standard deviations set equal to 1/10 of the median of the rate constants')
        ne=R.shape[0]
        Rstd=np.repeat([np.median(R,axis=0)],ne,axis=0)
    elif np.any(R.shape!=Rstd.shape):
        print('Warning: Shape of standard deviation does not match shape of rate constants')
        print('Standard deviations set equal to 1/10 of the median of the rate constants')
        ne=R.shape[0]        
        Rstd=np.repeat([np.median(R,axis=0)]/10,ne,axis=0)
        
    if (S2 is not None and S2_std is None) or (S2 is not None and S2.size!=S2_std.size):
        print('Warning: Shape of S2 does not match the shape of S2_std')
        print('Standard deviations set to 0.01')
        S2_std=np.ones(S2.shape)*0.01
    
    return R,Rstd,label,S2,S2_std

def read_lines(f,keys0):
    """
    Reads individual lines of data from a file
    """
    
    R=list()
    ne=0
    cont=True
    
    while not(eof(f)) and cont:
        pos=f.tell()
        a=f.readline()
        if np.isin(a.strip(),keys0):
            cont=False
            f.seek(pos)
        else:
            try:
                R0=np.atleast_1d(a.strip().split()).astype('float')
            except:
                print('Warning: Could not convert data in file into float')
                return None
            if R0.size>0:
                if ne==0:
                    ne=R0.size
                elif ne!=R0.size:
                    print('Inconsistent row lengths, data input aborted')
                    return None
                R.append(R0)
                
    return np.atleast_2d(R)

def read_label(f,keys0):
    """
    Reads out labels from a file. Tries to convert to label to float, returns strings if fails
    """
    label=list()
    cont=True
    while not(eof(f)) and cont:
        pos=f.tell()
        a=f.readline()
        if np.isin(a.strip(),keys0):
            cont=False
            f.seek(pos)
        else:
            label.append(a.strip())
    
    label=np.atleast_1d(label)
    
    try:
        label=label.astype('float')
    except:
        pass
    
    return label
        
def read_model(f,keys0):
    """
    Reads out description of a model
    """
    mdl_pars=dict()
    cont=True
    
    a=f.readline()
    mdl_pars.update({'Model':a.strip()})
    
    while not(eof(f)) and cont:
        pos=f.tell()
        a=f.readline()
        if np.isin(a.strip(),keys0):
            cont=False
            f.seek(pos)
        elif a.strip():
            name=a.strip()
            a=f.readline()
            val=np.atleast_1d(a.strip().split())
            try:
                val=val.astype('float')
            except:
                pass
            if val.size==1:
                val=val[0]
            mdl_pars.update({name:val})
    
    return mdl_pars

def read_info(f,keys0):
    """
    Reads out information on an experiment from file (called by load_NMR)
    """
    temp=rates()
    keywords=np.concatenate((temp.retExper(),temp.retSpinSys())) #These are the possible variables to load

    rate_args=list()
    args=dict()
    used=list()
    cont=True
    while not eof(f) and cont:
        pos=f.tell()
        a=f.readline()

        if np.isin(a.strip(),keywords):
            name=a.strip()
            if name in used:    #We reset to a new set of experiments if a parameter is repeated (usually 'Type')
                rate_args.append(args)
                used=list()
                used.append(name)
#                print(args)
                args=dict()
            else:
                used.append(name)
                
            val=f.readline().strip().split()
            try:
                val=np.array(val).astype('float')
            except:
                pass
            
            args.update({name:val})
        
        elif np.isin(a.strip().lower(),keys0):
            cont=False
            f.seek(pos)
        
    if args:
        rate_args.append(args)
        
    return rate_args

def eof(f):
    "Determines if we are at the end of the file"
    pos=f.tell()    #Current position in the file
    f.readline()    #Read out a line
    if pos==f.tell(): #If position unchanged, we're at end of file
        return True
    else:       #Otherwise, reset pointer, return False
        f.seek(pos)
        return False
        