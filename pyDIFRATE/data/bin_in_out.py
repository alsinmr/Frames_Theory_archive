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

Created on Wed Jul 31 16:37:08 2019

@author: albertsmith
"""

import pickle
#from data import data_class
#from data.data_class import data

def save_bin(filename,obj):
    """
    |save_bin saves a python object. 
    |
    |save_bin(filename,obj)
    |
    |Fails if that object contains an MDanalysis object
    """
    
    with open(filename,'wb') as f:
        pickle.dump(obj,f)
        
def load_bin(filename):
    """
    |Loads a python object
    |
    |obj = load_bin(filename)
    |
    |If object saved with save_DIFRATE, reload with load_DIFRATE
    """
    with open(filename,'rb') as f:
        obj=pickle.load(f)
        
    return obj


def save_DIFRATE(filename,obj):
    """
    |save_DIFRATE saves a DIFRATE object. 
    |
    |save_DIFRATE(filename,obj)
    |
    |Deletes the MDanalysis object before saving- this object otherwise creates
    |a pickling error 
    """
    
    """Note- I don't understand why this function is necessary. The MDAnalysis
    universe exists in the atom selections, and can be recovered from these. 
    Nonetheless, pickling fails if we don't first remove the saved universe.
    """
    
    if hasattr(obj,'copy'):
        obj=obj.copy()
    
    if hasattr(obj,'sens') and hasattr(obj,'detect'):
        if obj.sens is not None and obj.sens.molecule is not None:
            obj.sens.molecule.del_MDA_object()
        if obj.detect is not None and obj.detect.molecule is not None:
            obj.detect.molecule.del_MDA_object()
    elif hasattr(obj,'molecule'):
        obj.molecule.del_MDA_object()
    elif hasattr(obj,'mda_object'):
        obj.del_MDA_object()        
    
    save_bin(filename,obj)
    
    if hasattr(obj,'sens') and hasattr(obj,'detect'):
        if obj.sens is not None and obj.sens.molecule is not None:
            obj.sens.molecule.reload_MDA()
        if obj.detect is not None and obj.detect.molecule is not None:
            obj.detect.molecule.reload_MDA()
    elif hasattr(obj,'molecule'):
        obj.molecule.reload_MDA()
    elif hasattr(obj,'mda_object'):
        obj.reload_MDA()
    
    
def load_DIFRATE(filename):
    """
    |load_DIFRATE loads a DIFRATE object from a file
    |
    |obj = load_DIFRATE(filename)
    |
    |Replaces the mda_object in the various DIFRATE objects
    """
    
    obj=load_bin(filename)
    
    
    if hasattr(obj,'sens') and hasattr(obj,'detect'):
        if obj.sens is not None and obj.sens.molecule is not None and obj.sens.molecule.sel1 is not None:
            obj.sens.molecule.mda_object=obj.sens.molecule.sel1.universe
        if obj.detect is not None and obj.detect.molecule is not None and obj.detect.molecule.sel1 is not None:
            obj.detect.molecule.mda_object=obj.detect.molecule.sel1.universe
    elif hasattr(obj,'molecule') and obj.molecule.sel1 is not None:
        obj.molecule.mda_object=obj.molecule.sel1.universe
    elif hasattr(obj,'mda_object') and obj.sel1 is not None:
        obj.mda_object=obj.sel1.universe
        
    return obj