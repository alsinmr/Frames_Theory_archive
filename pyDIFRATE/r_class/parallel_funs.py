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


Created on Tue Apr 16 13:53:01 2019

@author: albertsmith
"""

from scipy.optimize import linprog
import numpy as np

def linprog_par(Y):
    Vt=Y[0]
    k=Y[1]
    ntc=np.shape(Vt)[1]
    try:
        x=linprog(np.sum(Vt,axis=1),-Vt.T,np.zeros(ntc),[Vt[:,k]],1,bounds=(-500,500),method='interior-point',options={'disp' :False})
        x=x['x']
    except:
        x=np.ones(Vt.shape[0])
#    x=linprog(np.sum(Vt,axis=1),-Vt.T,np.zeros(ntc),[Vt[:,k]],1,bounds=(None,None))
#    X[k]=x['x']
    return x