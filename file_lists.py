##file_lists.py

##a record of files to include in behavioral analysis.

import os
from sys import platform as _platform

if _platform == 'win32':
	root = "J:"
elif _platform == 'darwin':
	root = "/Volumes/Untitled"

##save location
save_loc = os.path.join(root,"Ryan/DS_animals/results")

behavior_files = [
##S1 files
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R1.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R2.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R3.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R4.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R5.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R6.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R7.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R8.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R9.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R10.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R11.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R12.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R13.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R14.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R15.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R16.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R17.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/behavior/S1_R18.hdf5"),
##S2 files
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R1.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R2.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R3.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R4.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R5.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R6.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R7.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R8.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R9.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R10.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R11.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R12.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R13.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R14.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R15.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R16.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R17.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/behavior/S2_R18.hdf5"),
##S3 files
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R1.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R2.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R3.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R4.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R5.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R6.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R7.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R8.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R9.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R10.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R11.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R12.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R14.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R15.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R16.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/behavior/S3_R17.hdf5")
]

ephys_files = [
##S1 files
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R1r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R2r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R3.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R4r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R5r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R6r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R7r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R8r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R9r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R10r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R11r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R12r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R13r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R14r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R15r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R16r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R17r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S1/neural_data/S1_R18r.hdf5"),
##S2 files
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R1r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R2r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R3r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R4r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R5r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R6r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R7r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R8r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R9r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R10r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R11r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R12r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R13r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R14r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R15r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R16r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R17r.hdf5"),
os.path.join(root,"Ryan/DS_animals/S2/neural_data/S2_R18r.hdf5"),
##S3 files
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R1.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R2.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R3.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R4.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R5.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R6.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R7.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R8.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R9.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R10.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R11.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R12.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R14.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R15.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R16.hdf5"),
os.path.join(root,"Ryan/DS_animals/S3/neural_data/S3_R17.hdf5"),
]