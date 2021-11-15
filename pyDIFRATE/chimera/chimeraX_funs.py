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

Created on Tue Jul 13 13:42:37 2021

@author: albertsmith

Created on Wed Nov 11 15:13:30 2020

@author: albertsmith
"""

import os
import numpy as np
import MDAnalysis as md
import matplotlib.pyplot as plt
from shutil import copyfile
#os.chdir('../Struct')
import pyDIFRATE.Struct.select_tools as selt
#os.chdir('../Struct')
from pyDIFRATE.Struct.vf_tools import Spher2pars,norm,getFrame,Rspher,pbc_corr,pars2Spher,sc2angles,pass2act
#os.chdir('../chimera')

def chimera_path(**kwargs):
    "Returns the location of the ChimeraX program"
    
    assert is_chimera_setup(),\
        "ChimeraX path does not exist. Run chimeraX.set_chimera_path(path) first, with "+\
        "path set to the ChimeraX executable file location."
    
    with open(os.path.join(get_path(),'ChimeraX_program_path.txt'),'r') as f:
        path=f.readline()
    
    return path

def is_chimera_setup():
    "Determines whether chimeraX executable path has been provided"
    return os.path.exists(os.path.join(get_path(),'ChimeraX_program_path.txt'))

def clean_up():
    """Deletes chimera scripts and tensor files that may have been created but 
    not deleted
    
    (Under ideal circumstances, this shouldn't happen, but may occur due to errors)
    """
    
    names=[fn for fn in os.listdir(get_path()) \
           if fn.startswith('chimera_script') and fn.endswith('.py') and len(fn)==23]
    
    tensors=[fn for fn in os.listdir(get_path()) \
           if fn.startswith('tensors') and fn.endswith('.txt') and len(fn)==18]
    
    for n in names:
        os.remove(os.path.join(get_path(),n))
    for t in tensors:
        os.remove(os.path.join(get_path(),t))
    
    print('{0} files removed'.format(len(names)+len(tensors)))

def set_chimera_path(path):
    """
    Stores the location of ChimeraX in a file, entitled ChimeraX_program_path.txt
    
    This function needs to be run before execution of Chimera functions (only
    once)
    """
    assert os.path.exists(path),"No file found at '{0}'".format(path)
    
    with open(os.path.join(get_path(),'ChimeraX_program_path.txt'),'w') as f:
        f.write(path)
        

def run_command(**kwargs):
    "Code to import runCommand from chimeraX"
    return 'from chimerax.core.commands import run as rc\n'

def get_path(filename=None):
    """
    Determines the location of THIS script, and returns a path to the 
    chimera_script given by filename.
    
    full_path=get_path(filename)
    """
    dir_path=os.path.dirname(os.path.realpath(__file__))
    return dir_path if filename is None else os.path.join(dir_path,filename)

def WrCC(f,command,nt=0):
    """Function to print chimera commands correctly, using the runCommand function
    within ChimeraX. nt specifies the number of tabs in python to use.
    
    f:          File handle
    command:    command to print
    nt:         number of tabs
    
    WrCC(f,command,nt)
    """
    for _ in range(nt):
        f.write('\t')
    f.write('rc(session,"{0}")\n'.format(command))
    

def py_line(f,text,nt=0):
    """
    Prints a line to a file for reading as python code. Inserts the newline and
    also leading tabs (if nt specified)
    
    python_line(f,text,nt=0)
    """
    
    for _ in range(nt):
        f.write('\t')
    f.write(text)
    f.write('\n')

def guess_disp_mode(mol):
    """
    Attempts to guess how to plot a molecule, based on the contents of the pdb
    and if specified, the selections in mol.sel1 and mol.sel2.
    
    Current options are: 
        'bond'  :   Plots the full molecule (or molecules), and displays dynamics
                    on the given bond
        'backbone': Plots a protein backbone, with dynamics displayed on the
                    H,N,CA,C, and O atoms
        'equiv':    If selections (mol.sel1,mol.sel2) include a heteroatom in 
                    sel1(or sel2) and a proton in sel2(or sel1), dynamics will
                    be plotted on those atoms and any other equivalent atoms. The
                    full molecule will be displayed
        'methyl':   If atoms in mol.sel1 and mol.sel2 are members of methyl groups,
                    and the full molecule is determined to be a protein, then 
                    dynamics will be displayed on the fully methyl group. 
                    Furthermore, the protein backbone will be displayed along
                    with sidechains containing the relevant sidechains.
                
    disp_mode=guess_disp_mode(mol)
    """
    
    sel0=mol.mda_object.atoms[0:0] #An empty atom group
    if mol.sel1 is not None:
        sel0=sel0+mol.sel1
    if mol.sel2 is not None:
        sel0=sel0+mol.sel1
    if len(sel0)==0:
        sel0=mol.mda_object.atoms #No selections: sel0 is the full universe
    
    segids=np.unique(sel0.segids)
    resids=np.unique(sel0.resids)
    if len(segids)!=0:
        i=np.isin(mol.mda_object.atoms.segids,segids)
        sel1=mol.mda_object.atoms[i]
    elif len(resids)!=0:
        i=np.isin(mol.mda_object.atoms.resids,resids)
        sel1=mol.mda_object.atoms[i]
    else:
        sel1=mol.mda_object.atoms
    
    
    if 'N' in sel1.names and 'CA' in sel1.names and 'C' in sel1.names and 'O' in sel1.names:
        "This is probably a protein"
        if mol.sel1 is None or mol.sel2 is None:
            return 'backbone' #Without further info, we assume this is a backbone plote
        
        is_met=True     #Switch to false if we find a selection not consistent with methyl
        is_bb=True      #Switch to false if we find a selection not consistent with backbone
        for s1,s2 in zip(mol.sel1,mol.sel2):
            if not(s1.name in ['N','CA','C','O'] or s2.name in ['N','CA','C','O']):
                is_bb=False
            types1=[s.types[0] for s in selt.find_bonded(s1,sel0=sel1,sort='massi',n=3)]
            types2=[s.types[0] for s in selt.find_bonded(s2,sel0=sel1,sort='massi',n=3)]
            if not(np.all([t=='H' for t in types1]) or np.all([t=='H' for t in types2])):
                is_met=False
            
            if not(is_met or is_bb):
                return 'bond' #Neither methyl or backbone, so return 'bond' mode
        
        if is_met:
            return 'methyl'
        return 'backbone'
    else:
        "Probably not a protein"
        if len(np.unique(mol.sel1))==len(mol.sel1) and len(np.unique(mol.sel2))==len(mol.sel2):
            "Then, no repeated selections, so probably ok to highlight other bonded 1H"
            return 'equiv'
        else:
            return 'bond'

def sel_indices(mol,disp_mode,mode='all'):
    """
    Generates list of indices plotting dynamics or for showing the correct 
    selection. Set mode to all to select all atoms to be displayed and to 
    'value' to get an index for each selection to be plotted.
    
    str=sel_str(mol,disp_mode='protein',mode='all')
    
    str_list1,str_list2=sel_str(mol,disp_mode='bond',mode='value')
        In bond mode, there may be overlaps in selections, so we return two lists,
        otherwise:
    str_list=sel_str(mol,disp_mode='methyl,mode='value')
    """
    uni=mol.mda_object
    if mode.lower()=='all':
        "First get all atoms in mol.sel1 and mol.sel2, or just everything in the universe if no selections"
        if mol.sel1 is not None and mol.sel2 is not None:
            sel0=mol.sel1+mol.sel2
        elif mol.sel1 is not None:
            sel0=mol.sel1
        elif mol.sel2 is not None:
            sel0=mol.sel2
        else:
            sel0=uni.atoms
        
        "Resids and Segids in the selection"
        resind=np.unique(sel0.resindices)
        segind=np.unique(sel0.segindices)
        sel0=uni.atoms
        
        "If all selections in one segment or one residue, only display that segment/residue"
        if len(segind)==1:
            sel0=sel0.segments[np.isin(uni.segments.segindices,segind)]
        if len(resind)==1:
            sel0=sel0.residues[np.isin(sel0.residues.resindices,resind)]
        sel0=sel0.atoms
        
        "Get selections according to mode"
        if disp_mode.lower()=='backbone':
            sel0=sel0.select_atoms('name N C CA')
            for s1,s2 in zip(mol.sel1,mol.sel2):
                sel0=sel0+uni.residues[s1.resindex].atoms.select_atoms('name H HN')+\
                uni.residues[s2.resindex].atoms.select_atoms('name H HN')
                sel0=sel0+uni.residues[s1.resindex-1].atoms.select_atoms('name O')+\
                uni.residues[s2.resindex-1].atoms.select_atoms('name O')
        elif disp_mode.lower()=='methyl':
            "In methyl mode, get the backbone and residues that were in the mol.sel1/mol.sel2"
            sel0=sel0.select_atoms('name N C CA')+uni.residues[resind].atoms.select_atoms('type C N O S')
            "And now, add on the protons attached to the bonded carbon"
            for s1,s2 in zip(mol.sel1,mol.sel2):
                if s1.type=='C':
                    bonded=selt.find_bonded(s1,uni.residues[s1.resindex].atoms,sort='massi',n=3)
                else:
                    bonded=selt.find_bonded(s1,uni.residues[s2.resindex].atoms,sort='massi',n=3)
                
                for b in bonded:sel0=sel0+b[0]
        out=uni2pdb_index(np.unique(sel0.ids),mol.pdb_id)
        
        return out[out!=-1]
        
    else:
        if disp_mode.lower()=='backbone':
            ids=list()
            sel0=uni.atoms
            sel=mol.sel2 if mol.sel1 is None else mol.sel1
            for s in sel:
                resindex=s.resindex
                sel1=sel0.residues[resindex].atoms.select_atoms('name N H HN')
                sel2=sel0.residues[resindex-1].atoms.select_atoms('name C O')
                if len(sel2)==2 and np.sqrt(((sel1.select_atoms('name N').positions-\
                                             sel2.select_atoms('name C').positions)**2).sum())<1.6:
                    sel1=sel1+sel2
                ids.append(uni2pdb_index(sel1.ids,mol.pdb_id))
        elif disp_mode.lower()=='bond':
            ids=list()
            for s1,s2 in zip(mol.sel1,mol.sel2):
                ids.append(uni2pdb_index((s1+s2).ids,mol.pdb_id))
        elif disp_mode.lower()=='methyl':
            ids=list()
            sel0=uni.atoms
            for s1,s2 in zip(mol.sel1,mol.sel2):
                s=s1 if s1.type=='C' else s2
                sel0=uni.residues[s.resindex].atoms
                sel=selt.find_bonded(s,sel0=sel0,sort='massi',n=3)
                ids.append([uni2pdb_index(s.id,mol.pdb_id)[0]])
                ids[-1].extend(uni2pdb_index([s1[0].id for s1 in sel],mol.pdb_id))
        elif disp_mode.lower()=='equiv':
            ids=list()
            sel0=uni.residues[np.unique((mol.sel1+mol.sel2).resindices)].atoms
            for s1,s2 in zip(mol.sel1,mol.sel2):
                s1,s2=(s1,s2) if s1.mass>s2.mass else (s2,s1)
                bond=selt.find_bonded(s1,sel0,sort='massi',n=3)
                id0=[s1.id]
                for b in bond:
                    if b[0].type==s2.type:id0.append(b[0].id)
                ids.append(uni2pdb(id0,mol.pdb_id))
        else:
            print('Unrecognized display mode ({0}) in sel_indices'.format(disp_mode))
            print('Use backbone,bond,methyl, or equiv')
            return
        
#        ids=[i*(i!=-1) for i in ids]
        
        return ids

def py_print_npa(f,name,x,format_str='.6f',dtype=None,nt=0):
    """
    Hard-codes an array into a python script for running within ChimeraX. Provide
    the file handle, the name for the variable within ChimeraX, the values to 
    be stored in the array, and the number of tabs in for the variable  (nt)
    
    A format string may be used to determine the precision written to file
    (example .6f, .3e, etc.)
    
    The data type stored in ChimeraX may also be controlled, by specifying the
    data type
    
    Only for single elements, 1D, and 2D arrays (will create a numpy array in 
    ChimeraX)
    
    py_print_npa(f,name,x,format_str='.6f',dtype=None,nt=0)
    """
    
    x=np.array(x)
    
    f.write('\n')
    for _ in range(nt):f.write('\t')
    
    if x.size==0:
        print('Warning: writing an empty matrix to ChimeraX')
        f.write(name+'=np.zeros({0})'.format(x.shape))
    elif x.ndim==0:
        f.write((name+'=np.array({0:'+format_str+'})').format(x))
    elif x.ndim==1:
        f.write(name+'=np.array([')
        for k,x0 in enumerate(x):
            f.write(('{0:'+format_str+'},').format(x0))
            if np.mod(k,10)==9 and k!=len(x)-1:
                f.write('\\\n') #Only 10 numbers per line
                for _ in range(nt+1):f.write('\t')
        f.seek(f.tell()-1)  #Delete the last comma
        f.write('])')
    elif x.ndim==2:
        f.write(name+'=np.array([')
        for m,x0 in enumerate(x):
            if m!=0:  #Tab in lines following the first line
                for _ in range(nt+1):f.write('\t')
            f.write('[')    #Start the current row
            for k,x00 in enumerate(x0):
                f.write(('{0:'+format_str+'},').format(x00))
                if np.mod(k,10)==9 and k!=len(x0)-1:
                    f.write('\\\n') #Only 10 numbers per line
                    for _ in range(nt+1):f.write('\t')
            f.seek(f.tell()-1) #Delete the last comma
            f.write('],\n') #Close the current row
            
        f.seek(f.tell()-2) #Delete the new line and comma
        f.write('])')   #Close the matrix
    else:
        print('Too many dimensions for py_print_npa')
        return
        
    if dtype is not None:
        f.write('.astype("{0}")'.format(dtype))
    f.write('\n')

def py_print_lists(f,name,x,format_str='.6f',nt=0):
    """
    Hard-codes a list or list of lists into a python script for running within
    ChimeraX. Provide the file handle, the name for the variable within ChimeraX
    the values to be stored in the lists, and the number of tabs in for the 
    variable (nt)
    
    A format string may be used to determine the precision written to file
    (example .6f, .3e, etc.)
    
    Only for single elements, 1D, and 2D lists
    
    py_print_lists(f,name,x,nt=0)
    """
    
    f.write('\n')
    for _ in range(nt):f.write('\t')
    
    if not(hasattr(x,'__len__')):
        f.write((name+'{0:'+format_str+'}').format(x))
    elif not(hasattr(x[0],'__len__')):
        f.write(name+'=[')
        for k,x0 in enumerate(x):
            f.write(('{0:'+format_str+'},').format(x0))
            if np.mod(k,10)==9 and k!=len(x)-1:
                f.write('\\\n') #10 numbers per line
                for _ in range(nt+1):f.write('\t')
        f.seek(f.tell()-1) #Delete the last comma
        f.write(']')
    elif not(hasattr(x[0][0],'__len__')):
        f.write(name+'=[')
        for m,x0 in enumerate(x):
            if m!=0:
                for _ in range(nt+1):f.write('\t')
            f.write('[')
            for k,x00 in enumerate(x0):
                f.write(('{0:'+format_str+'},').format(x00))
                if np.mod(k,10)==9 and k!=len(x0)-1:
                    f.write('\\\n') #Only 10 numbers per line
                    for _ in range(nt+1):f.write('\t')
            f.seek(f.tell()-1) #Delete last comma
            f.write('],\n') #Close the current row
        f.seek(f.tell()-2) #Delete the new line and comma
        f.write(']') #Close the matrix
    else:
        print('Too many dimensions for py_print_lists')
        return
     
    f.write('\n')           


   

def color_calc(x,x0=None,colors=[[0,0,255,255],[210,180,140,255],[255,0,0,255]]):
    """
    Calculates color values for a list of values in x (x ranges from 0 to 1).
    
    These values are linear combinations of reference values provided in colors.
    We provide a list of N colors, and a list of N x0 values (if x0 is not provided,
    it is set to x0=np.linspace(0,1,N). If x is between the 0th and 1st values
    of x0, then the color is somewhere in between the first and second color 
    provided. Default colors are blue at x=0, tan at x=0.5, and red at x=1.
    
    color_calc(x,x0=None,colors=[[0,0,255,255],[210,180,140,255],[255,0,0,255]])
    """
    
    colors=np.array(colors,dtype='uint8')
    N=len(colors)
    if x0 is None:x0=np.linspace(0,1,N)
    x=np.array(x)
    if x.min()<x0.min():
        print('Warning: x values less than min(x0) are set to min(x0)')
        x[x<x0.min()]=x0.min()
    if x.max()>x0.max():
        print('Warning: x values greater than max(x0) are set to max(x0)')
        x[x>x0.max()]=x0.max()
    
    i=np.digitize(x,x0)
    i[i==len(x0)]=len(x0)-1
    
    clr=(((x-x0[i-1])*colors[i].T+(x0[i]-x)*colors[i-1].T)/(x0[i]-x0[i-1])).T
    
    return clr.astype('uint8')

def get_default_colors(det_num):
    """
    Returns in RGBA the default color for a given detector number. 
    """
    
    clr0=plt.rcParams['axes.prop_cycle'].by_key()['color']
    clr=clr0[np.mod(det_num,len(clr0))]
    
    return np.concatenate((hex_to_rgb(clr),[255]))
    

def hex_to_rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip('#')
    lv = len(value)
    return [int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]  

def run_chimeraX(mol,disp_mode=None,x=None,chimera_cmds=None,fileout=None,save_opts=None,\
                scene=None,x0=None,marker=None,absval=True,colors=[[255,255,0,255],[255,0,0,255]]):
    """
    Opens an instance of chimera, displaying the current pdb (if no pdb exists
    in mol, it will also create the pdb). Atoms will be displayed in accordance
    with the display mode:
        'bond'  :   Shows all atoms in the same segment/residues if mol.sel1
                    and mol.sel2 contain only one segment/residue. Parameter
                    encoding only shown on particular bond
        'equiv' :   Same display as bond, but parameter encoding will be shown 
                    on all equivalent bonds
        'backbone': Shows only protein backbone, along with parameter encoding
                    on all atoms in the peptide plane
        'methyl':   Shows protein backbone, plus side chains of residues found
                    in mol.sel1,mol.sel2. Parameter encoding is shown all the
                    methyl protons, carbon, and the next bonded carbon.
    If the display mode is not specified, it will be guessed based on the atoms
    in mol.sel1, mol.sel2
    
    One may also specify x, a vector with values between 0 and 1, which is 
    encoded into the radius and color of atoms. The displayed radius is 
    calculcated from x as 4*x+0.9 Angstroms.
    
    chimera_cmds:   A list of commands to execute in chimera after setup
    scene       :   A saved chimera session to open and execute run_chimera in.
                    Note: the pdb open in scene must have the same indices as the
                    pdb in mol.pdb
    fileout     :   File to save chimera image to.
    save_opts   :   Options for file saving
    colors      :   List of colors used for encoding
    x0          :   Values of x corresponding to each color
    marker      :   Index of selection to color black (for example, for cc plots)
    
    run_chimera(mol,disp_mode=None,x=None,chimera_cmds=None,fileout=None,save_opts=None,\
                scene=None,x0=None,colors=[[0,0,255,255],[210,180,140,255],[255,0,0,255]])
    """
    
    if mol.pdb is None:
        mol.MDA2pdb()
    pdb=mol.pdb #Get pdb name

    rand_index=np.random.randint(1e6)   #We'll tag a random number onto the filename
                                        #This lets us run multiple instances without interference
    full_path=get_path('chimera_script{0:06d}.py'.format(rand_index))     #Location to write out chimera script
    
    
    "Here we try to guess the display mode if not given"
    if disp_mode is None:
        disp_mode=guess_disp_mode(mol)
    
    di=sel_indices(mol,disp_mode,mode='all')
    
    
    with open(full_path,'w') as f:

        py_line(f,'try:')
        py_line(f,run_command(),1)
        py_line(f,'import os',1)
        py_line(f,'import numpy as np',1)
        py_print_npa(f,'di',di,format_str='d',dtype='uint32',nt=1)
        
        if x is not None:
            "Print out matrices required for parameter encoding"
            if len(mol.sel1)!=len(x) or len(mol.sel2)!=len(x):
                print('The lengths of mol.sel1, mol.sel2, and x (the parameter to be encoded)')
                print('must all match')
                return

            id0=sel_indices(mol,disp_mode,mode='value')
            x=np.concatenate([x0*np.ones(len(i)) for i,x0 in zip(id0,x)])
            id0=np.concatenate([i for i in id0]).astype(int)
            ids,b=np.unique(id0,return_index=True)
            x=np.array([x[id0[b0]==id0].mean() for b0 in b])
            clrs=color_calc(x=x,x0=x0,colors=colors)
            
            i=ids!=-1
            ids,x,clrs=ids[i],x[i],clrs[i]
            
            if marker:
                id_mark=sel_indices(mol,disp_mode,mode='value')[marker]
                py_print_npa(f,'id_mark',id_mark,format_str='d',dtype='uint32',nt=1)
            py_print_npa(f,'ids',ids,format_str='d',dtype='uint32',nt=1)
            py_print_npa(f,'r',4*np.abs(x)+0.9,format_str='.6f',dtype='float',nt=1) #Scale up radius ??   
            py_print_npa(f,'clr',clrs,format_str='d',dtype='uint8',nt=1)
            
        if scene:
#            WrCC(f,'open '+scene,1)
            py_line(f,'mdl=session.open_command.open_data("{0}")[0]'.format(scene),1)
            py_line(f,'session.models.add(mdl)',1)
        else:
#            WrCC(f,"open '"+pdb+"'",1)
            py_line(f,'mdl=session.open_command.open_data("{0}")[0]'.format(pdb),1)
            py_line(f,'session.models.add(mdl)',1)
        WrCC(f,'~display',1)
        WrCC(f,'~ribbon',1)
        
        
        #Get the atoms to be displayed
        py_line(f,'if len(session.models)>1:',1)
        py_line(f,'atoms=session.models[1].atoms',2)
        WrCC(f,'display #1.1',2)
        py_line(f,'else:',1)
        py_line(f,'atoms=session.models[0].atoms',2)
        WrCC(f,'display #1',2)
             
        #Set to ball and stick
         
        
        #Display the correct selection

#        py_line(f,'display=getattr(atoms.atoms,"visibles")',1)
#        py_line(f,'display[:]=0',1)
#        py_line(f,'display[di]=1',1)
#        py_line(f,'setattr(atoms.atoms,"visibles",display)',1)
        py_line(f,'hide=getattr(atoms,"hides")',1)
        py_line(f,'hide[:]=1',1)
        py_line(f,'hide[di]=0',1)
        py_line(f,'setattr(atoms,"hides",hide)',1)
        
        #Parameter encoding
        if x is not None:
            WrCC(f,'style ball',1)
            WrCC(f,'size stickRadius 0.2',1)
            WrCC(f,'color all tan',1)
            py_line(f,'r0=getattr(atoms,"radii").copy()',1)
            py_line(f,'clr0=getattr(atoms,"colors").copy()',1)
            py_line(f,'r0[:]=.8',1)
            py_line(f,'r0[ids]=r',1)
            py_line(f,'clr0[ids]=clr',1)
            if marker:
                py_line(f,'clr0[id_mark]=[70,70,70,255]',1)
            py_line(f,'setattr(atoms,"radii",r0)',1)
            py_line(f,'setattr(atoms,"colors",clr0)',1)

        if chimera_cmds is not None:
            if isinstance(chimera_cmds,str):chimera_cmds=[chimera_cmds]
            for cc in chimera_cmds:
                WrCC(f,cc,1)
        
        if fileout is not None:
            if len(fileout)>=4 and fileout[-4]!='.':fileout=fileout+'.png'
            if save_opts is None:save_opts=''
            WrCC(f,"save " +fileout+' '+save_opts,1)
        
        py_line(f,'except:')
        py_line(f,'pass',1)
        py_line(f,'finally:')
        py_line(f,'os.remove("{0}")'.format(full_path),1)
        if fileout is not None: #Exit if a file is saved
            WrCC(f,'exit',1)
    copyfile(full_path,full_path[:-9]+'.py')

    os.spawnl(os.P_NOWAIT,chimera_path(),chimera_path(),full_path)
#    import subprocess
#    subprocess.Popen([chimera_path(),'--start shell',full_path])

def molecule_only(mol,disp_mode=None):
    """
    Displays the molecule in ChimeraX
    """
    
    if mol.pdb is None:
        mol.MDA2pdb()
    pdb=mol.pdb #Get pdb name

    rand_index=np.random.randint(1e6)   #We'll tag a random number onto the filename
    full_path=get_path('chimera_script{0:06d}.py'.format(rand_index))     #Location to write out chimera script
    
    "Here we try to guess the display mode if not given"
    if disp_mode is None and (mol.sel1 is not None or mol.sel2 is not None):
        disp_mode=guess_disp_mode(mol)
        
        di=sel_indices(mol,disp_mode,mode='all')
    else:
        di=None
    
    
    with open(full_path,'w') as f:

        py_line(f,'try:')
        py_line(f,run_command(),1)
        py_line(f,'import os',1)
        py_line(f,'import numpy as np',1)
        if di is not None:
            py_print_npa(f,'di',di,format_str='d',dtype='uint32',nt=1)
        
            

        WrCC(f,'open '+pdb,1)
        WrCC(f,'~display',1)
        WrCC(f,'~ribbon',1)
        
        #Get the atoms to be displayed
        py_line(f,'if len(session.models)>1:',1)
        py_line(f,'atoms=session.models[1].atoms',2)
        WrCC(f,'display #1.1',2)
        py_line(f,'else:',1)
        py_line(f,'atoms=session.models[0].atoms',2)
        WrCC(f,'display #1',2)
             
        py_line(f,'hide=getattr(atoms,"hides")',1)
        py_line(f,'hide[:]=1',1)
        py_line(f,'hide[di]=0',1)
        py_line(f,'setattr(atoms,"hides",hide)',1)
        
        py_line(f,'except:')
        py_line(f,'pass',1)
        py_line(f,'finally:')
        py_line(f,'os.remove("{0}")'.format(full_path),1)

    copyfile(full_path,full_path[:-9]+'.py')

    os.spawnl(os.P_NOWAIT,chimera_path(),chimera_path(),full_path)

def draw_tensors(A,mol=None,sc=2.09,tstep=0,disp_mode=None,index=None,scene=None,\
                 fileout=None,save_opts=None,chimera_cmds=None,\
                 colors=[[255,100,100,255],[100,100,255,255]],marker=None,\
                 marker_color=[[100,255,100,255],[255,255,100,255]],deabg=False,\
                 pos=None,frame='inter',vft=None):
    """
    Plots tensors onto bonds of a molecule. One must provide the tensors, A, which
    are plotted onto a molecule (if molecule object provided), the molecule
    object, where mol.sel1 and mol.sel2 define the bonds corresponding to the 
    elements in A.
    
        A:      Tensors, where input should be 5xN. Tensors should be provided
                in the frame of the bond. By default, these are the complex
                components of the tensor itself. Alternatively, provide delta,
                eta, alpha, beta, and gamma (radians) in a 5xN matrix. Set
                deabg to True.
        mol:    molecule object, with N atoms selected in mol.sel1 and mol.sel2,
                and where mol._vft is defined (that is, mol.tensor_frame was
                run, and mol.clear_frames has not been executed. mol._vft() should
                also return N vectors. We need both these pieces of information
                to determine both the orientation of the tensors and their 
                positions.
                If mol not provided, tensors will be simply plotted in space,
                instead of on a pdb
            
        tstep:  Time step to use for generating pdb, orienting tensors
        index:  Only show some tensors. Index can select which, out of A and
                mol.sel1, mol.sel2, etc. to use
        sel
        
        frame:  Provides the frame that tensors are defined in. By default, this
                frame is the frame of the interaction (so, A=[0,0,1,0,0] would
                lie along the bond (set frame='inter'). However, one may also 
                choose the lab frame (set frame ='LF'), for which A=[0,0,1,0,0]
                would lie along z.
    """
    
    "Filenames"
    rand_index=np.random.randint(1e6)   #We'll tag a random number onto the filenames
                                    #This lets us run multiple instances without interference
    full_path=get_path('chimera_script{0:06d}.py'.format(rand_index))     #Location to write out chimera script 
    tensor_file=get_path('tensors_{0:06d}.txt'.format(rand_index))
    
    
    A=np.atleast_2d(A)
    if A.shape[0]!=5 and A.shape[1]==5:
        A=A.T
    elif A.shape[0]!=5:
        print('A must be 5xN, where N is number of tensors')


    if mol is not None and (len(mol.sel1)!=len(A[0]) or len(mol.sel2)!=len(A[0])):
        print('The lengths of mol.sel1, mol.sel2, and A (the tensor)')
        print('must all match')
        print('Length of: [mol.sel1, mol.sel1, A]:[{0},{1},{2}]'.format(len(mol.sel1),len(mol.sel2),len(A[0])))
        return

    if index is None:
        index=np.ones(len(A[0]),dtype=bool)
    else:
        index=np.array(index)
        if index.max()>1 or (len(index)<3 and len(mol.sel1)>3): #Then probably not a logical index
            in0=index
            index=np.zeros(len(mol.sel1),dtype=bool)
            index[in0]=True

    if marker is None:
        marker=np.zeros(len(A[0]),dtype=bool)
    else:
        marker=np.array(marker)
        if marker.max()>1 or (len(marker)<3 and len(marker.sel1)>3): #Then probably not a logical index
            in0=marker
            marker=np.zeros(len(mol.sel1),dtype=bool)
            marker[in0]=True
    marker=marker[index]       

    "Here we try to guess the display mode if not given"
    if mol is not None and disp_mode is None:
        disp_mode=guess_disp_mode(mol)                                       
    
    
    if scene is None:
        if mol is not None:
            if mol.pdb is None:
                if disp_mode.lower()=='methyl' or disp_mode.lower()=='backbone':
                    mol.MDA2pdb(tstep=tstep,select='protein')
                else:
                    resids=np.unique((mol.sel1+mol.sel2).resids)
                    select='resid'
                    for r in resids:select+=' {0}'.format(r)
                    mol.MDA2pdb(tstep=tstep,select=select)
            pdb=mol.pdb
        else:
            pdb=None
    
    if mol is not None:
        di=sel_indices(mol,disp_mode,mode='all') #INdex for which atom to display
        
#    id0=np.concatenate([i for i in id0]).astype(int)
#    ids,b=np.unique(id0,return_index=True)
    
    "Calculate parameters required for tensor file"
    if mol is not None:
        mol.mda_object.trajectory[tstep] #Go to the correct time step
    
    if deabg:A=pars2Spher(*A)   #Move to tensor components

    A=[a[index] for a in A]     #Index the tensors    
    if mol is None:
        if pos is None:
            pos=np.zeros([3,len(A[0])])
            pos[0]=np.arange(len(A[0]))*3
    else:
        if frame[0].lower()!='l':
            vZ,vXZ=mol._vft() if vft is None else vft()   #Get bond directions
            scF=getFrame(vZ[:,index],vXZ[:,index]) #Get frame of bond, apply index
            A=Rspher(A,*scF)     #Apply frame to tensors
            
        pos=(mol.sel1.positions[index]+mol.sel2.positions[index]).T/2 #Get the positions, along with index
    delta,eta,*euler=Spher2pars(A,return_angles=True)
    
    write_tensor(tensor_file,delta=delta*sc,eta=eta,euler=euler,pos=pos,marker=marker)      #Write out the tensor file
    
    with open(full_path,'w') as f:
        py_line(f,'import os')
        py_line(f,'import numpy as np')
        py_line(f,run_command())
        
        copy_funs(f)    #Copy required functions into chimeraX script
                
        py_line(f,'\n')
        py_line(f,'try:')
        
        if mol is not None:
            py_print_npa(f,'di',di,format_str='d',dtype='uint32',nt=1)  #Print out ids for visualizing
        
        if scene is not None:
            WrCC(f,'open '+scene,1)
        elif pdb is not None:
            WrCC(f,'open '+pdb,1)
        WrCC(f,'~display',1)
        WrCC(f,'~ribbon',1)
        
        #Get the atoms to be displayed
        if mol is not None:
            py_line(f,'if len(session.models)>1:',1)
            py_line(f,'atoms=session.models[1].atoms',2)
            WrCC(f,'display #1.1',2)
            py_line(f,'else:',1)
            py_line(f,'atoms=session.models[0].atoms',2)
            WrCC(f,'display #1',2)
                 
             #Display the correct selection
            py_line(f,'hide=getattr(atoms,"hides")',1)
            py_line(f,'hide[:]=1',1)
            py_line(f,'hide[di]=0',1)
            py_line(f,'setattr(atoms,"hides",hide)',1)
        
        WrCC(f,'style stick',1)
        

        
        negative_color=[int(c) for c in colors[1]]
        positive_color=[int(c) for c in colors[0]]
        nc=[int(c) for c in marker_color[1]]
        pc=[int(c) for c in marker_color[0]]
        
        py_line(f,('load_surface(session,"{0}",sc={1},theta_steps={2},phi_steps={3},positive_color={4},negative_color={5},'\
                +'marker_pos_color={6},marker_neg_color={7})')\
            .format(tensor_file,sc,50,25,positive_color,negative_color,pc,nc),1)
    
        WrCC(f,'display',1)
        
        if chimera_cmds is not None:
            if isinstance(chimera_cmds,str):chimera_cmds=[chimera_cmds]
            for cmd in chimera_cmds:
                WrCC(f,cmd,1)
    
        
        
        if fileout is not None:
            if len(fileout)<=4 or fileout[-4]!='.':fileout=fileout+'.png'
            if save_opts is None:save_opts=''
            WrCC(f,"save " +fileout+' '+save_opts,1)
        py_line(f,'except:')
        py_line(f,'pass',1)
        py_line(f,'finally:')
        py_line(f,'os.remove("{0}")'.format(full_path),1)
        py_line(f,'os.remove("{0}")'.format(tensor_file),1)
#        py_line(f,'pass',1)
        if fileout is not None:
            WrCC(f,'exit',1)
    copyfile(full_path,full_path[:-9]+'.py')
    copyfile(tensor_file,tensor_file[:-11]+'.txt')
    
    os.spawnl(os.P_NOWAIT,chimera_path(),chimera_path(),full_path)

            
    
def uni2pdb_index(index,pdb_index,report_err=False):
    "Converts the universe index to the index for a stored pdb"
    "The stored pdb is in molecule.pdb, and the index is in molecule.pdb_in"
    
    index=np.atleast_1d(index)
    
    i=-np.ones(np.size(index),dtype=int)
    for k,ind in enumerate(index):
        if np.any(ind==pdb_index):
            i[k]=np.argwhere(ind==pdb_index)[0,0]
        elif report_err:
            print('Index: {0} not found in pdb_index'.format(ind))
    return i.astype(int)


def write_tensor(filename,delta,eta=None,euler=None,pos=None,marker=None):
    """
    Writes out a tab-separated file with delta, eta, alpha, beta, gamma, and
    x,y,z for tensors. For reading within ChimeraX
    
    write_tensor(filename,delta,eta=None,euler=None,pos=None,marker=None)
    """
    
    delta=np.array(delta)
    n=delta.size
    
    #Defaults, make sure all numpy arrays
    eta=np.zeros(n) if eta is None else np.array(eta)
    euler=np.zeros([3,n]) if euler is None else np.array(euler)
    pos=np.zeros([3,n]) if pos is None else np.array(pos)
    if marker is None:
        marker=np.zeros(n)
    else:
        if not(hasattr(marker,'__len__')):marker=[marker]
        if len(marker)<len(eta) or np.max(marker)>1:
            m1=marker
            marker=np.zeros(n)
            marker[np.array(m1,dtype=int)]=1
    
    if len(euler)==3:
        alpha,beta,gamma=euler
    else:
        alpha,beta,gamma=sc2angles(*euler)
    X,Y,Z=pos
    

    with open(filename,'w') as f:
        for vals in zip(delta,eta,alpha,beta,gamma,X,Y,Z,marker):
            for v in vals[:-1]:f.write('{0:16.8}\t'.format(v))
            f.write('{0:d}\t'.format(int(vals[-1])))
            f.write('\n')

def copy_funs(f):
    """
    Copys all functions in THIS file below the comment "Files used inside ChimeraX"
    
    Input is the file handle, f, to which the pythons functions should be copied
    
    copy_funs(f)
    """
    
    with open(get_path('chimeraX_funs.py'),'r') as funs:
        start_copy=False
        for line in funs:
            if start_copy:
                f.write(line)
            else:
                if len(line)>=30 and line[:30]=="#%% Files used inside ChimeraX":
                    start_copy=True
        f.write('\n')

#%% Files used inside ChimeraX (don't edit this comment!!..it will break the code)
"""
Everything after these lines is printed into the chimeraX script, so don't add
anything below that you don't need in chimeraX
"""
def sphere_triangles(theta_steps=100,phi_steps=50):
    """
    Creates arrays of theta and phi angles for plotting spherical tensors in ChimeraX.
    Also returns the corresponding triangles for creating the surfaces
    """
    
    theta=np.linspace(0,2*np.pi,theta_steps,endpoint=False).repeat(phi_steps)
    phi=np.repeat([np.linspace(0,np.pi,phi_steps,endpoint=True)],theta_steps,axis=0).reshape(theta_steps*phi_steps)
    
    triangles = []
    for t in range(theta_steps):
        for p in range(phi_steps-1):
            i = t*phi_steps + p
            t1 = (t+1)%theta_steps
            i1 = t1*phi_steps + p
            triangles.append((i,i+1,i1+1))
            triangles.append((i,i1+1,i1))
    
    return theta,phi,triangles
    
def spherical_surface(delta,eta=None,euler=None,pos=None,sc=2.09,
                      theta_steps = 100,
                      phi_steps = 50,
                      positive_color = (255,100,100,255), # red, green, blue, alpha, 0-255 
                      negative_color = (100,100,255,255)):
    """
    Function for generating a surface in ChimeraX. delta, eta, and euler angles
    should be provided, as well positions for each tensor (length of all arrays
    should be the same, that is (N,), (N,), (3,N), (3,N) respectively.
    
    Returns arrays with the vertices positions (Nx3), the triangles definitions
    (list of index triples, Nx3), and a list of colors (Nx4)
    
    xyz,tri,colors=spherical_surface(delta,eta=None,euler=None,pos=None,
                                     theta_steps=100,phi_steps=50,
                                     positive_color=(255,100,100,255),
                                     negative_color=(100,100,255,255))
    """
    # Compute vertices and vertex colors
    a,b,triangles=sphere_triangles(theta_steps,phi_steps)
    
    if euler is None:euler=[0,0,0]
    if pos is None:pos=[0,0,0]
    if eta is None:eta=0
    
    # Compute r for each set of angles
    sc=np.sqrt(2/3)*sc
    
    A=[-1/2*delta*eta,0,np.sqrt(3/2)*delta,0,-1/2*delta*eta]   #Components in PAS
    
    #0 component after rotation by a and b
    A0=np.array([A[mp+2]*d2(b,m=0,mp=mp)*np.exp(1j*mp*a) for mp in range(-2,3)]).sum(axis=0).real
    
    #Coordinates before rotation by alpha, beta, gamma
    x0=np.cos(a)*np.sin(b)*np.abs(A0)*sc/2
    y0=np.sin(a)*np.sin(b)*np.abs(A0)*sc/2
    z0=np.cos(b)*np.abs(A0)*sc/2

    alpha,beta,gamma=euler
    alpha,beta,gamma=-alpha,-beta,-gamma    #Added 30.09.21 along with edits to vf_tools>R2euler
    #Rotate by alpha
    x1,y1,z1=x0*np.cos(alpha)+y0*np.sin(alpha),-x0*np.sin(alpha)+y0*np.cos(alpha),z0
    #Rotate by beta
    x2,y2,z2=x1*np.cos(beta)-z1*np.sin(beta),y1,np.sin(beta)*x1+np.cos(beta)*z1
    #Rotate by gamma
    x,y,z=x2*np.cos(gamma)+y2*np.sin(gamma),-x2*np.sin(gamma)+y2*np.cos(gamma),z2

    x=x+pos[0]
    y=y+pos[1]
    z=z+pos[2]
    
#    xyz=[[x0,y0,z0] for x0,y0,z0 in zip(x,y,z)]
    #Determine colors
    colors=np.zeros([A0.size,4],np.uint8)
    colors[A0>=0]=positive_color
    colors[A0<0]=negative_color
    

    # Create numpy arrays
#    xyz = np.array(xyz, np.float32)
    xyz=np.ascontiguousarray(np.array([x,y,z]).T,np.float32)       #ascontiguousarray forces a transpose in memory- not just editing the stride
    colors = np.array(colors, np.uint8)
    tri = np.array(triangles, np.int32)

    return xyz,tri,colors
 

def load_tensor(filename):
    """
    Reads in a tab-separated file with delta, eta, alpha,beta, gamma, and x,y,z
    for a set of tensors. 
    
    delta,eta,euler,pos=load_tensor(filename)
    """
    delta=list()
    eta=list()
    alpha=list()
    beta=list()
    gamma=list()
    x=list()
    y=list()
    z=list()
    marker=list()
    with open(filename,'r') as f:
        for line in f:
            out=line.strip().split('\t')
            out=[np.array(o,float) for o in out]
            delta.append(out[0])
            eta.append(out[1])
            alpha.append(out[2])
            beta.append(out[3])
            gamma.append(out[4])
            x.append(out[5])
            y.append(out[6])
            z.append(out[7])
            marker.append(out[8])

    delta=np.array(delta)
    eta=np.array(eta)
    euler=np.array([alpha,beta,gamma]).T
    pos=np.array([x,y,z]).T
    marker=np.array(marker)

    return delta,eta,euler,pos,marker            
        
    

def load_surface(session,tensor_file,sc=2.09,theta_steps=100,phi_steps=50,
                 positive_color=(255,100,100,255),negative_color=(100,100,255,255),
                 marker_pos_color=(100,255,100,255),marker_neg_color=(255,255,100,255)):
    
    Delta,Eta,Euler,Pos,Marker=load_tensor(tensor_file)
    
    from chimerax.core.models import Surface
    from chimerax.surface import calculate_vertex_normals,combine_geometry_vntc
    
    geom=list()
    
    for k,(delta,eta,euler,pos,marker) in enumerate(zip(Delta,Eta,Euler,Pos,Marker)):
        if marker==1:
            pc=marker_pos_color
            nc=marker_neg_color
        else:
            pc=positive_color
            nc=negative_color
        xyz,tri,colors=spherical_surface(\
                                         delta=delta,eta=eta,euler=euler,pos=pos,\
                                         sc=sc,theta_steps=theta_steps,\
                                         phi_steps=phi_steps,\
                                         positive_color=pc,\
                                         negative_color=nc)

        norm_vecs=calculate_vertex_normals(xyz,tri)
        
        geom.append((xyz,norm_vecs,tri,colors))    
        
    xyz,norm_vecs,tri,colors=combine_geometry_vntc(geom)    
    s = Surface('surface',session)
    s.set_geometry(xyz,norm_vecs,tri)
    s.vertex_colors = colors
    session.models.add([s])

    return s


def d2(c=0,s=None,m=None,mp=0):
    """
    Calculates components of the d2 matrix. By default only calculates the components
    starting at m=0 and returns five components, from -2,-1,0,1,2. One may also
    edit the starting component and select a specific final component 
    (mp=None returns all components, whereas mp may be specified between -2 and 2)
    
    d2_m_mp=d2(m,mp,c,s)  #c and s are the cosine and sine of the desired beta angle
    
        or
        
    d2_m_mp=d2(m,mp,beta) #Give the angle directly
    
    Setting mp to None will return all values for mp in a 2D array
    
    (Note that m is the final index)
    """
    
    if s is None:
        c,s=np.cos(c),np.sin(c)
    
    """
    Here we define each of the components as functions. We'll collect these into
    an array, and then call them out with the m and mp indices
    """
    "First, for m=-2"
    
    if m is None or mp is None:
        if m is None and mp is None:
            print('m or mp must be specified')
            return
        elif m is None:
            if mp==-2:
                index=range(0,5)
            elif mp==-1:
                index=range(5,10)
            elif mp==0:
                index=range(10,15)
            elif mp==1:
                index=range(15,20)
            elif mp==2:
                index=range(20,25)
        elif mp is None:
            if m==-2:
                index=range(0,25,5)
            elif m==-1:
                index=range(1,25,5)
            elif m==0:
                index=range(2,25,5)
            elif m==1:
                index=range(3,25,5)
            elif m==2:
                index=range(4,25,5)
    else:
        index=[(mp+2)*5+(m+2)]
    
    out=list()    
    for i in index:
        #mp=-2
        if i==0:x=0.25*(1+c)**2
        if i==1:x=0.5*(1+c)*s
        if i==2:x=np.sqrt(3/8)*s**2
        if i==3:x=0.5*(1-c)*s
        if i==4:x=0.25*(1-c)**2
        #mp=-1
        if i==5:x=-0.5*(1+c)*s
        if i==6:x=c**2-0.5*(1-c)
        if i==7:x=np.sqrt(3/8)*2*c*s
        if i==8:x=0.5*(1+c)-c**2
        if i==9:x=0.5*(1-c)*s
        #mp=0
        if i==10:x=np.sqrt(3/8)*s**2
        if i==11:x=-np.sqrt(3/8)*2*s*c
        if i==12:x=0.5*(3*c**2-1)
        if i==13:x=np.sqrt(3/8)*2*s*c
        if i==14:x=np.sqrt(3/8)*s**2
        #mp=1
        if i==15:x=-0.5*(1-c)*s
        if i==16:x=0.5*(1+c)-c**2
        if i==17:x=-np.sqrt(3/8)*2*s*c
        if i==18:x=c**2-0.5*(1-c)
        if i==19:x=0.5*(1+c)*s
        #mp=2
        if i==20:x=0.25*(1-c)**2
        if i==21:x=-0.5*(1-c)*s
        if i==22:x=np.sqrt(3/8)*s**2
        if i==23:x=-0.5*(1+c)*s
        if i==24:x=0.25*(1+c)**2
        out.append(x)
        
    if m is None or mp is None:
        return np.array(out)
    else:
        return out[0]
