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


Created on Fri Mar 22 09:32:49 2019

@author: albertsmith

Collection of useful functions for DIFRATE
"""

import numpy as np
#import os 
#os.chdir('../tools')
from pyDIFRATE.tools.DRtools import NucInfo
#os.chdir('../r_class')

    
    
def J(tc,v):
    "Returns the spectral density"
    return 2/5*tc/(1+(2*np.pi*v*tc)**2)

def rate(tc,exper):
    """Returns the sensitivity of an experiment, specified by exper (one 
    column of a pandas array), for a given set of correlation times, tc.
    """
    try:
        if exper.loc['Type'] in globals():
            fun=globals()[exper.loc['Type']]
            R=fun(tc,exper)
            return R
        else:
            print('Experiment type {0} was not recognized'.format(exper.loc['Type']))
            return
    except:
        print('Calculation of experiment {0} failed. Check parameters.'.format(exper.loc['Type']))
        return
    
def S2(tc,exper):
    """
    Order parameter (note- one must provide 1-S2 into the data.R matrix!)
    
    Returns a uniform sensitivity, independent of correlation time
    """
    return np.ones(np.shape(tc))    
    
def R1(tc,exper):
    "Returns longitudinal relaxation rate constant"
    "Dipolar relaxation updated to include MAS spinning, relevant for homonuclear couplings"
    v0=exper['v0']*1e6
    vr=exper['vr']*1e3
    dXY=exper['dXY']
    Nuc=exper['Nuc']
    Nuc1=exper['Nuc1']
    QC=exper['QC']
    eta=exper['eta']
    vX=NucInfo(Nuc)/NucInfo('1H')*v0
    CSA=exper['CSA']*vX/1e6
    R=np.zeros(tc.shape)

    if Nuc1 is not None and dXY is not None:
        "Dipolar relaxation"
        if np.size(dXY)==1:
            vY=NucInfo(Nuc1)/NucInfo('1H')*v0
            S=NucInfo(Nuc1,'spin')
            sc=S*(S+1)*4/3 # Scaling factor depending on the spin, =1 for spin 1/2
            
            if vX==vY:
                Delv=exper['CSoff']*vX/1e6
                R+=sc*(np.pi*dXY/2)**2*(1/6*J(tc,Delv+2*vr)+1/6*J(tc,Delv-2*vr)\
                   +1/3*J(tc,Delv+vr)+1/3*J(tc,Delv-vr)+3*J(tc,vX)+6*J(tc,2*vX))
            else:
               R+=sc*(np.pi*dXY/2)**2*(J(tc,vX-vY)+3*J(tc,vX)+6*J(tc,vY+vX))

        else:
            for k in range(0,np.size(dXY)):
                S=NucInfo(Nuc1[k],'spin')
                sc=S*(S+1)*4/3 # Scaling factor depending on the spin, =1 for spin 1/2
                vY=NucInfo(Nuc1[k])/NucInfo('1H')*v0
                if vX==vY:
                    Delv=exper['CSoff'][k]*vX/1e6
                    R+=sc*(np.pi*dXY[k]/2)**2*(1/6*J(tc,Delv+2*vr)+1/6*J(tc,Delv-2*vr)\
                       +1/3*J(tc,Delv+vr)+1/3*J(tc,Delv-vr)+3*J(tc,vX)+6*J(tc,2*vX))
                else:
                    R+=sc*(np.pi*dXY[k]/2)**2*(J(tc,vX-vY)+3*J(tc,vX)+6*J(tc,vY+vX))
                
    "CSA relaxation"
    R+=3/4*(2*np.pi*CSA)**2*J(tc,vX)

    if QC!=0:
        "Quadrupolar relaxation"
        """Note that these formulas give the initial rate of relaxation, that 
        is, the average rate of relaxation for all orientations, and furthermore
        does not include deviations due to multi-exponential relaxation
        """
        S=NucInfo(Nuc,'spin')
        deltaQ=1/(2*S*(2*S-1))*QC*2*np.pi
        C=(deltaQ/2)**2*(1+eta**2/3) #Constant that scales the relaxation
        if S==0.5:
            print('No quadrupole coupling for S=1/2')
        elif S==1:
            R+=C*(3*J(tc,vX)+12*J(tc,2*vX))
        elif S==1.5:
            R+=C*(36/5*J(tc,vX)+144/5*J(tc,2*vX))
        elif S==2.5:
            R+=C*(96/5*J(tc,vX)+384/5*J(tc,2*vX))
        else:
            print('Spin={0} not implemented for quadrupolar relaxation'.format(S))
            
    return R

def R1Q(tc,exper):
    """This function calculates the relaxation rate constant for relaxation of
    quadrupolar order
    """
    v0=exper['v0']*1e6
    Nuc=exper['Nuc']
    QC=exper['QC']
    eta=exper['eta']
    vX=NucInfo(Nuc)/NucInfo('1H')*v0
    
    S=NucInfo(Nuc,'spin')
    deltaQ=1/(2*S*(2*S-1))*QC*2*np.pi
    C=(deltaQ/2)**2*(1+eta**2/3)    #Constant scaling the relaxation
    if S==0.5:
        print('No quadruple coupling for spin=1/2')
    elif S==1:
        R=C*9*J(tc,vX)
    elif S==1.5:
        R=C*(36*J(tc,vX)+36*J(tc,2*vX))
    elif S==2.5:
        R=C*(792/7*J(tc,vX)+972/7*J(tc,2*vX))
    else:
        print('Spin not implemented')
        
    return R

def R1p(tc,exper):
    v0=exper['v0']*1e6
    dXY=exper['dXY']
    Nuc=exper['Nuc']
    Nuc1=exper['Nuc1']
    QC=exper['QC']
    eta=exper['eta']
    vr=exper['vr']*1e3
    v1=exper['v1']*1e3
    off=exper['offset']*1e3
    vX=NucInfo(Nuc)/NucInfo('1H')*v0
    CSA=exper['CSA']*vX/1e6
    R=np.zeros(tc.shape)
    
    "Treat off-resonance spin-lock"
    ve=np.sqrt(v1**2+off**2)
    if ve==0:
        theta=np.pi/2
    else:
        theta=np.arccos(off/ve)
    
    R10=R1(tc,exper)    #We do this first, because it includes all R1 contributions
    "Start here with the dipole contributions"
    if Nuc1 is not None:
        if np.size(dXY)==1:
            vY=NucInfo(Nuc1)/NucInfo('1H')*v0
            S=NucInfo(Nuc1,'spin')
            sc=S*(S+1)*4/3 #Scaling depending on spin of second nucleus
            R1del=sc*(np.pi*dXY/2)**2*(3*J(tc,vY)+
                      1/3*J(tc,2*vr-ve)+2/3*J(tc,vr-ve)+2/3*J(tc,vr+ve)+1/3*J(tc,2*vr+ve))
        else:            
            R1del=np.zeros(tc.shape)
            for k in range(0,np.size(dXY)):
                vY=NucInfo(Nuc1[k])/NucInfo('1H')*v0
                S=NucInfo(Nuc1[k],'spin')
                sc=S*(S+1)*4/3 #Scaling depending on spin of second nucleus
                R1del+=sc*(np.pi*dXY[k]/2)**2*(3*J(tc,vY)+
                          1/3*J(tc,2*vr-ve)+2/3*J(tc,vr-ve)+2/3*J(tc,vr+ve)+1/3*J(tc,2*vr+ve))
    else:
        R1del=np.zeros(tc.shape)
    "CSA contributions"
    R1del+=1/6*(2*np.pi*CSA)**2*(1/2*J(tc,2*vr-ve)+J(tc,vr-ve)+J(tc,vr+ve)+1/2*J(tc,2*vr+ve))
    "Here should follow the quadrupole treatment!!!"    
    
    "Add together R1 and R1p contributions, depending on the offset"
    R+=R10+np.sin(theta)**2*(R1del-R10/2) #Add together the transverse and longitudinal contributions   
    return R

def R2(tc,exper):
    exper['off']=0
    exper['v1']=0
    
    return R1p(tc,exper)   

def NOE(tc,exper):
    v0=exper['v0']*1e6
    dXY=exper['dXY']
    Nuc=exper['Nuc']
    Nuc1=exper['Nuc1']
    vX=NucInfo(Nuc)/NucInfo('1H')*v0
    R=np.zeros(tc.shape)
    
    if Nuc1!=None:
        vY=NucInfo(Nuc1)/NucInfo('1H')*v0
        S=NucInfo(Nuc1,'spin')
        sc=S*(S+1)*4/3 # Scaling factor depending on the spin, =1 for spin 1/2
        R+=sc*(np.pi*dXY/2)**2*(-J(tc,vX-vY)+6*J(tc,vY+vX))
        
    return R

def ccXY(tc,exper):
    """
    CSA-dipole cross-correlated transverse relaxation
    """
    v0,dXY,Nuc,Nuc1,theta=exper['v0']*1e6,exper['dXY'],exper['Nuc'],exper['Nuc1'],exper['theta']*np.pi/180
    vX=NucInfo(Nuc)/NucInfo('1H')*v0
    CSA=exper['CSA']*vX/1e6

    if Nuc1 is not None:
        S=NucInfo(Nuc1,'spin')
        if S!=0.5:
            print('Warning: Formulas for cross-correlated relaxation have only been checked for S=1/2')
        sc=S*(S+1)*4/3  
        R=np.sqrt(sc)*1/8*(2*np.pi*dXY)*(2*np.pi*CSA)*(3*np.cos(theta)**2-1)/2.*(4*J(tc,0)+3*J(tc,vX))
    else:
        R=np.zeros(tc.shape)
    return R

def ccZ(tc,exper):
    """
    CSA-dipole cross-correlated longitudinal relaxation
    """
    v0,dXY,Nuc,Nuc1,theta=exper['v0']*1e6,exper['dXY'],exper['Nuc'],exper['Nuc1'],exper['theta']*np.pi/180
    vX=NucInfo(Nuc)/NucInfo('1H')*v0
    CSA=exper['CSA']*vX/1e6
    
    if Nuc1 is not None:
        S=NucInfo(Nuc1,'spin')
        if S!=0.5:
            print('Warning: Formulas for cross-correlated relaxation have only been checked for S=1/2')
        sc=S*(S+1)*4/3  
        R=np.sqrt(sc)*1/8*(2*np.pi*dXY)*(2*np.pi*CSA)*(3*np.cos(theta)**2-1)/2.*6*J(tc,vX)
    else:
        R=np.zeros(tc.shape)
    return R