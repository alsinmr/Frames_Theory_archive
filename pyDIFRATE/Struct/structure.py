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


Created on Thu Apr  4 15:05:19 2019

@author: albertsmith
"""

import MDAnalysis as mda
from MDAnalysis.lib.mdamath import make_whole
import MDAnalysis.analysis.align
import numpy as np
import os
from pyDIFRATE.Struct.vec_funs import new_fun,print_frame_info
import copy
#os.chdir('../chimera')
#from chimera.chimera_funs import open_chimera
from pyDIFRATE.chimera.chimeraX_funs import molecule_only
#os.chdir('../Struct')
import pyDIFRATE.Struct.select_tools as selt

class molecule(object):
    def __init__(self,*args):
        self.mda_object=None
        self.sel1=None
        self.sel2=None
        self.sel1in=None
        self.sel2in=None
        self.label_in=list()
        self.label=list()
        self.vXY=np.array([])
        self.vCSA=np.array([])
        self.Ralign=list()
        self._vf=None
        self._vft=None
        self._frame_info={'frame_index':list(),'label':None}
        
        self.pdb=None #Container for a pdb extracted from the mda_object
        self.pdb_id=None
        "We might want to delete this pdb upon object deletion"
        
        self.__MDA_info=None
        
        if np.size(args)>0:
            self.load_struct(*args)

    def load_struct(self,*args,**kwargs):   
        self.mda_object=mda.Universe(*args,**kwargs)
        
#    def vec_special(self,Type,**kwargs):
#        """
#        Allows user defined vectors to be created, from a function defined in
#        vec_vuns.py (a function handle should be returned, where that function
#        returns a value dependent on the current position in the md analysis
#        trajectory. The function should return x,y,z components at the given time)
#        """
##        if self._vf is None:
##            self._vf=list()
##            
##        self._vf.append(new_fun(Type,self,**kwargs))
#        
#        """I'm joining the vec_special and frames functionality.
#        vec_special as its own attribute will eventually be removed
#        """
#        self.new_frame(Type,**kwargs)   
        
    def clear_vec_special(self):
        self._vf=None
        
    def vec_fun(self):
        """
        Evaluates all vectors generated with vec_special at the current time point
        of the MD trajectory. Returns a 3xN vector, where N is the number of 
        vectors (ex. moment_of_inertia or rot_axis vectors)
        """
        vec=list()
        if self._vf is not None:
            for f in self._vf:
                vec.append(f()) #Run all of the functions in self._vf
            return np.concatenate(vec,axis=1)
        else:
            print('No vector functions defined, run vec_special first')
        
#    def select_atoms(self,sel1=None,sel2=None,sel1in=None,sel2in=None,index1=None,index2=None,Nuc=None,resi=None,select=None,**kwargs):
    
    def new_frame(self,Type=None,frame_index=None,**kwargs):
        """
        Create a new frame, where possible frame types are found in vec_funs.
        Note that if the frame function produces a different number of reference
        frames than there are bonds (that is, vectors produced by the tensor 
        frame), then a frame_index is required, to map the frame to the appropriate
        bond. The length of the frame_index should be equal to the number of 
        vectors produced by the tensor frame, and those elements should have 
        values ranging from 0 to one minus the number of frames defined by this
        frame. 
        
        To get a list of all implemented frames and their arguments, call this
        function without any arguments. To get arguments for a particular frame,
        call this function with only Type defined.
        """
        if Type is None:
            print_frame_info()
        elif len(kwargs)==0:
            print_frame_info(Type)
        else:
            assert self._vft is not None,'Define the tensor frame first (run mol.tensor_frame)'
            vft=self._vft()
            nb=vft[0].shape[1] if len(vft)==2 else vft.shape[1] #Number of bonds in the tensor frame
            if self._vf is None: self._vf=list()
            fun,fi,*_=new_fun(Type,self,**kwargs)
            if frame_index is None:frame_index=fi #Assign fi to frame_index if frame_index not provided
            f=fun()    #Output of the vector function (test its behavior)
            nf=f[0].shape[1] if len(f)==2 else f.shape[1]
            if fun is not None:
                "Run some checks on the validity of the frame before storing it"
                if frame_index is not None:
                    assert frame_index.size==nb,'frame_index size does not match the size of the tensor_fun output'
                    assert frame_index[np.logical_not(np.isnan(frame_index))].max()<nf,'frame_index contains values that exceed the number of frames'
                    self._frame_info['frame_index'].append(frame_index)
                else:
                    assert nf==nb,'No frame_index was provided, but the size of the tensor_fun and the frame_fun do not match'
                    self._frame_info['frame_index'].append(np.arange(nb))
                self._vf.append(fun)    #Append the new function
                    
    
    def tensor_frame(self,Type='bond',label=None,**kwargs):
        """
        Creates a frame that defines the NMR tensor orientation. Usually, this
        is the 'bond' frame (default Type). However, other frames may be used
        in case a dipole coupling is not the relevant interaction. The chosen
        frame should return vectors defining both a z-axis and the xz-plane. A
        warning will be returned if this is not the case.
        """
        if Type is None:
            print_frame_info()
        elif len(kwargs)==0:
            print_frame_info(Type)
        else:
            if Type=='bond' and 'sel3' not in kwargs:
                kwargs['sel3']='auto'     #Define sel3 for the bond frame (define vXZ)
            
            self._vft,*_=new_fun(Type,self,**kwargs) #New tensor function
            if len(self._vft())!=2:
                print('Warning: This frame only defines vZ, and not vXZ;')
                print('In this case, correlation functions may not be properly defined')
            if label is not None:
                self._frame_info['label']=label
                  
    
    def clear_frames(self):
        "Clears out all informatin about frames"
        self._vf=None
        self._vft=None
        self._frame_info={'frame_index':list(),'label':None}
        
    def select_atoms(self,sel1=None,sel2=None,sel1in=None,sel2in=None,Nuc=None,resids=None,segids=None,filter_str=None):
        """
        Selects the atoms to be used for bond definitions. 
        sel1/sel2 : A string or an atom group (MDanalysis) defining the 
                    first/second atom in the bond
        sel1in/sel2in : Index to re-assign sel1/sel2 possibly to multiple bonds
                        (example: if using string assignment, maybe to calculate
                        for multiple H's bonded to the same C)
        Nuc : Keyword argument for selecting a particular type of bond 
              (for example, N, C, CA would selection NH, C=O bonds, or CA-HA 
              bonds, respectively)
        resids : Filter the selection defined by the above arguments for only
                 certain residues
        segids : Filter the selection defined by the above arguments for only 
                 certain segments
        filter_str : Filter the selection by a string (MDAnalysis string selection)
        """
        
        if Nuc is None:
            "Apply sel1 and sel2 selections directly"
            if sel1 is not None:
                self.sel1=selt.sel_simple(self,sel1,resids,segids,filter_str)
                if sel1in is not None:
                    self.sel1=self.sel1[sel1in]
            if sel2 is not None:
                self.sel2=selt.sel_simple(self,sel2,resids,segids,filter_str)
                if sel2in is not None:
                    self.sel2=self.sel2[sel2in]
        else:
            self.sel1,self.sel2=selt.protein_defaults(Nuc,self,resids,segids,filter_str)
                
        if self.sel1 is not None and self.sel2 is not None and self.sel1.n_atoms==self.sel2.n_atoms:
            self.set_selection()
            
            "An attempt to generate a unique label under various conditions"
            count,cont=0,True
            while cont:
                if count==0: #One bond per residue- just take the residue number
                    label=self.sel1.resids  
                elif count==1: #Multiple segments with same residue numbers
                    label=np.array(['{0}_{1}'.format(s.segid,s.resid) for s in self.sel1])
                elif count==2: #Same segment, but multiple bonds on the same residue (include names)
                    label=np.array(['{0}_{1}_{2}'.format(s1.resid,s1.name,s2.name) for s1,s2 in zip(self.sel1,self.sel2)])
                elif count==3: #Multiple bonds per residue, and multiple segments
                    label=np.array(['{0}_{1}_{2}_{3}'.format(s1.segid,s1.resid,s1.name,s2.name) \
                                    for s1,s2 in zip(self.sel1,self.sel2)])
                "We give up after this"
                count=count+1
                if np.unique(label).size==label.size or count==4:
                    cont=False
                    
            self.label=label
            
#    def select_atoms(self,sel1=None,sel2=None,sel1in=None,sel2in=None,Nuc=None,resi=None,select=None,**kwargs):
#        
#        
#        
#        if select is not None:
#            sel=self.mda_object.select_atoms(select)
#        else:
#            sel=self.mda_object.atoms
#            
#        if resi is not None:
#            string=''
#            for res in resi:
#                string=string+'resid {0:.0f} or '.format(res)
#            string=string[0:-4]
#            sel=sel.select_atoms(string)
#        
#        if Nuc is not None:
#            if Nuc.lower()=='15n' or Nuc.lower()=='n':                    
#                self.sel1=sel.select_atoms('(name H or name HN) and around 1.1 name N')
#                self.sel2=sel.select_atoms('name N and around 1.1 (name H or name HN)')
#            elif Nuc.lower()=='co':
#                self.sel1=sel.select_atoms('name C and around 1.4 name O')
#                self.sel2=sel.select_atoms('name O and around 1.4 name C')
#            elif Nuc.lower()=='ca':
#                self.sel1=sel.select_atoms('name CA and around 1.4 (name HA or name HA2)')
#                self.sel2=sel.select_atoms('(name HA or name HA2) and around 1.4 name CA')
#                print('Warning: selecting HA2 for glycines. Use manual selection to get HA1 or both bonds')
##            self.label_in=self.sel1.resids           
#            self.label_in=self.sel1.resnums                     
#        else:
#            if sel1!=None:
##                if index1!=None:
##                    self.sel1=sel.select_atoms(sel1)[index1]
##                else:
##                    self.sel1=sel.select_atoms(sel1)
#                self.sel1=sel.select_atoms(sel1)
#            if sel2!=None:
##                if index2!=None:
##                    self.sel2=sel.select_atoms(sel2)[index2]
##                else:
##                    self.sel2=sel.select_atoms(sel2)
#                self.sel2=sel.select_atoms(sel2)
#              
#            "I'm gradually eliminating any use of sel1in and sel2in, and replacing using this approach"
#            "This also makes index1 and index2 redundant"
#            if sel1in is not None:
##                self.sel1in=sel1in  
#                self.sel1=self.sel1[sel1in]
#            if sel2in is not None:
##                self.sel2in=sel2in
#                self.sel2=self.sel2[sel2in]
#                
#        try:
#            self.set_selection()    #Is this a good idea? Sets the selection, even if the user isn't done
#        except:
#            pass
    
        
    def clear_selection(self):
        self.sel1=None
        self.sel2=None
        self.sel1in=None
        self.sel2in=None
    
    #Seems pretty unnecessary....remove if nothing breaks
#    def add_label(self,label=None):
#        self.label_in=label
        
    def set_selection(self,**kwargs):
        
        if self.sel1in is None and self.sel2in is None:
            nr=np.size(self.sel1.resids)
        elif self.sel1in is None:
            nr=np.size(self.sel2in)
        else:
            nr=np.size(self.sel1in)

        
        vec=np.zeros([nr,3])
        
    #    for k in range(0, nt-2):
    
        if 'tstep' in kwargs:
            tstep=kwargs.get('tstep')
        else:
            tstep=int(self.mda_object.trajectory.n_frames/100)
            if tstep==0:
                tstep=1
    
        nt=self.mda_object.trajectory.n_frames
    
        for k in range(0,nt,tstep):
            """
            I think that this averaging over the trajectory should probably include
            a re-alignment of each molecule with itself at each time point. Consider
            the problems otherwise- for a lipid rotating in a membrane, we'd end
            up with all average bonds pointing approximately along the lipid normal.
            """
            try:
                self.mda_object.trajectory[k]
            except:
                if k!=0:
                    for _ in range(0,tstep):
                        self.mda_object.trajectory.next()
    
            if self.sel1in is None and self.sel2in is None:
                vec+=self.sel1.positions-self.sel2.positions
            elif self.sel1in is None:
                for m,q in enumerate(self.sel2in):
                    vec[m,:]+=self.sel1.positions[m]-self.sel2.positions[q]
            elif self.sel2in is None:
                for m,q in enumerate(self.sel1in):
                    vec[m,:]+=self.sel1.positions[q]-self.sel2.positions[m]
            else:
                for m,q in enumerate(self.sel1in):
                    vec[m,:]+=self.sel1.positions[q]-self.sel2.positions[self.sel2in[m]]
                    
        len=np.sqrt(np.sum(np.power(vec,2),axis=1))
        vec=np.divide(vec,np.reshape(np.repeat(len,3),vec.shape)) #Normalize the vector
        
        self.vXY=vec
        if np.shape(self.label_in)[0]==nr:
            self.label=self.label_in
        
    def MDA2pdb(self,tstep=None,select='protein',make_whole=False,**kwargs):
        "Provide a molecule, print a certain frame to pdb for later use in chimera"
        

        uni=self.mda_object

        
        if tstep is None:
            tstep=int(uni.trajectory.n_frames/2)
            
        dir_path = os.path.dirname(os.path.realpath(__file__))
    
    
        if self.pdb is not None and os.path.exists(self.pdb):
            os.remove(self.pdb)
            
        full_path=os.path.join(dir_path,os.path.basename(self.mda_object.trajectory.filename)+'_{0}'.format(tstep)+'.pdb')
        
        try:
            uni.trajectory[tstep]
        except:
            uni.trajectory.rewind()
            for k in range(0,tstep):
                uni.trajectory.next()
        
        if select is not None:
            a=uni.select_atoms(select)
        else:
            a=uni.atoms

        
        if make_whole:
            self.mk_whole(sel=a)
        
        a.write(full_path)
        
        self.pdb=full_path
        self.pdb_id=np.array(a.ids)
        
        return full_path
    
    def chimera(self,disp_mode=None):
        """
        Starts a chimera session with the pdb stored in molecule.pdb
        """
#        open_chimera(self,**kwargs)
        molecule_only(self,disp_mode=None)
    def mk_whole(self,sel=None):
        """
        Unwraps all segments in a selection of the MD analysis universe 
        (by default, the selection is the union self.sel1 snd self.sel2)
        
        self.unwrap(sel=None)
        
        Note that the unwrapping only remains valid while the trajectory remains
        on the current frame (re-run for every frame)
        
        This program needs to be run with MDA2pdb in most cases (automatic)
        If obtaining vectors, we need to run this if the 'align' option is being
        used. It is not necessary if we only look at individual bonds without
        aligning, since we can correct box crossings simply by searching for bonds
        that are too long.
        """
        
        "Default selection (segments in self.sel1 and self.sel2)"
        if sel is None:
            if self.sel1 is not None:
                if self.sel2 is not None:
                    sel=self.sel1.union(self.sel2)
                else:
                    sel=self.sel1
            elif self.sel2 is not None:
                sel=self.sel2
            else:
                sel=self.mda_object.atoms
        elif isinstance(sel,str):
            sel=self.mda_object.select_atoms(sel=sel)
            
        for seg in sel.segments:
            try:
                make_whole(seg.atoms)
            except:
                print('Failed to make segment {0} whole'.format(seg.segid))
            
    def align(self,select,tstep=0,overwrite=False):
        """
        Alignment of the trajectory to a reference selection (usually the CA of
        the protein. Important if the molecule rotates during the MD simulation.
        Should be performed before any analysis is performed (this will write
        a new trajectory in the same location as the original trajectory)
        
        align(select,tstep=0,overwrite=False)
        
        select should be an MDAnalysis selection string
        
        tstep determines the reference time step (index)
        
        overwrite allows one to re-align the trajectory (otherwise, an existing
        alignment may be loaded, possibly with a different reference selection)
        """
        
        
        "Get the mda universe object, and set trajectory to start"
        uni=self.mda_object
        uni.trajectory.rewind()
        "Where will the new trajectory be written?"
        directory=os.path.dirname(uni.trajectory.filename)
        base=os.path.basename(uni.trajectory.filename)

        if len(base)>=7 and base[0:7]=='rmsfit_' and overwrite:
            print('Warning: re-aligned a trajectory that has already been aligned before')
        
        filename='rmsfit_'+base
        newfile=os.path.join(directory,filename)
        

        "Load the file if it already exists"
        if not(overwrite) and os.path.exists(newfile):
            print('Aligned file:\n {0}\n already exists. Loading existing file'.format(newfile))
            print('To re-calculate aligned trajectory, set overwrite=True')
            self.load_struct(uni.filename,newfile)
        else:
            "Create a reference pdb (the first frame of the trajectory)"
            if self.pdb is not None:
                "We won't delete an existing pdb"
                pdb=self.pdb
                pdb_id=self.pdb_id
                self.pdb=None
            else:
                pdb=None
            
            "Get the reference pdb from the first trajectory"
            self.MDA2pdb(tstep=0,select=None)
            
            ref=mda.Universe(uni.filename,self.pdb)
            
            alignment=mda.analysis.align.AlignTraj(uni,ref,select=select,verbose=True,pbc=True)
            alignment.run()
            
            if newfile!=alignment.filename:
                print('Warning: Unexpected filename used by MDanalysis')
            
            self.load_struct(uni.filename,alignment.filename)
            
            "Reload existing pdb if given"
            if pdb is not None:
                os.remove(self.pdb)
                self.pdb=pdb
                self.pdb_id=pdb_id
        
        
        "Reset the selections"
        if self.sel1 is not None:
            self.sel1=self.mda_object.atoms[self.sel1.indices]
        if self.sel2 is not None:
            self.sel2=self.mda_object.atoms[self.sel2.indices]
        try:
            self.set_selection()
        except:
            pass
            

        return
    
    def del_MDA_object(self):
        """
        In some cases, it is necessary to delete the MD analysis objects 
        (for example, when saving, we can't pickle the MD object). This function
        deletes the object after first saving information required to reload
        it and the atom selections
        """
        if self.mda_object is None:
            "Do nothing if no universe is stored"
            return
        else:
            uni=self.mda_object
            info=dict()
            self.__MDA_info=info
        "Save the filenames used for the universe"
        info.update({'filename':uni.filename})
        if hasattr(uni.trajectory,'filenames'):
            info.update({'filenames':uni.trajectory.filenames})
        elif hasattr(uni.trajectory,'filename'):
            info.update({'filenames':np.atleast_1d(uni.trajectory.filename)})
            
        "Save the id numbers of the selections"
        if self.sel1 is not None:
            info.update({'sel1':self.sel1.ids})
        if self.sel2 is not None:
            info.update({'sel2':self.sel2.ids})
        
        "Set the MD analysis objects to None"
        self.mda_object=None
        self.sel1=None
        self.sel2=None
        
    def reload_MDA(self):
        if self.__MDA_info is None:
            "Do nothing if MD analysis object hasn't been deleted"
            return
        info=self.__MDA_info
        if 'filenames' in info:
            uni=mda.Universe(info['filename'],info['filenames'].tolist())
        else:
            uni=mda.Universe(info['filename'])
        self.mda_object=uni
        
        sel0=uni.atoms
        if 'sel1' in info:
            self.sel1=sel0[info['sel1']-1]
        if 'sel2' in info:
            self.sel2=sel0[info['sel2']-1]
        
        self.__MDA_info=None
        
    def copy(self,type='deep'):
        """
        |
        |Returns a copy of the object. Default is deep copy (all objects except the 
        |MDanalysis object, mda_object)
        | obj = obj0.copy(type='deep')
        |To also create a copy of the molecule object, set type='ddeep'
        |To do a shallow copy, set type='shallow'
        """
        if type=='ddeep':
            out=copy.deepcopy(self)
        elif type!='deep':
            out=copy.copy(self)
        else:
            uni=self.mda_object
            self.mda_object=None
            out=copy.deepcopy(self)
            self.mda_object=uni
            out.mda_object=uni
        return out
        
    def __del__(self):
        if self.pdb is not None and os.path.exists(self.pdb):
            os.remove(self.pdb)
    