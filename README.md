# Frames_Theory_archive
This archive provides the python scripts required to generate Figs. 5-7
shown in the paper "Interpreting NMR dynamic parameters via the separation of reorientation motion in MD simulation", 

by A.A. Smith

albert.smith-penzel@medizin.uni-leipzig.de

Scripts that can be run at the command line:
Fig5_ideal_case.py
Fig6_fail_ex.py
Fig7_methyl_frames.py

There is NO INSTALLATION required for the code. Just place everything in a folder, navigate there, and run. However, python3 and the following modules must be installed from other sources (these are the tested versions, although other versions may work).

Python v. 3.7.3
numpy v. 1.17.2,
scipy v. 1.3.0,
pandas v. 0.25.1,
MDAnalysis v. 0.19.2,
matplotlib v. 3.0.3,

We recommend installing Anaconda: https://docs.continuum.io/anaconda/install/
The Anaconda installation includes numpy, scipy, pandas, and matplotlib. 

MDAnalysis is installed by running:
conda config --add channels conda-forge
conda install mdanalysis
(https://www.mdanalysis.org/pages/installation_quick_start/)

All files are copyrighted under the GNU General Public License. A copy of the license has been provided in the file LICENSE

Copyright 2021 Albert Smith-Penzel