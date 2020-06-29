import cPickle as pickle
import numpy as np
import os, glob

"""
Store symmetry operations in .pickle format, with keys denoting the point group operation.
For each point group, the symmetry operations contain both the rotation and translational
operations; the final column corresponds to the translational element, and the zeroeth key
corresponds to the identity operation.

Also store restricted phase information in .pickle format, with keys denoting the point
group operation and values as np.array([caxis_value, cslice_value]) in degrees. Central
axis and central slice reflections are expected to be integer multiples of these values.

Usage: python sym_ops.py
Output is saved to reference/sym_ops.pickle and reference/restricted_phases.pickle.
"""

################################## USEFUL FUNCTIONS ##################################

def str_to_matrix(remarks):
    """
    Convert a dictionary whose values are symmetry operations in string
    format (as listed in REMARK 290 section of PDB header) to a dict of
    matrices corresponding to the translational and rotational elements.

    Inputs:
    -------
    remarks: dictionary of symmetry operations in string format
    
    Outputs:
    --------
    sym_ops: dictionary of symmetry operations in matrix format
    """
    
    sym_ops = dict()
    
    for key in remarks: 
        m = np.zeros((3,4))
        elems = remarks[key].split(",")
        
        for row in range(3):
            r_t = elems[row].split("+")
            
            # handle rotational portion
            if r_t[0][-1] == "X": m[row][0] = 1.0
            if r_t[0][-1] == "Y": m[row][1] = 1.0
            if r_t[0][-1] == "Z": m[row][2] = 1.0
            if r_t[0][0] == "-": m[row] *= -1.0
                
            # handle translational portion, if present
            if len(r_t) > 1:
                num, den = r_t[1].split("/")
                m[row][-1] = float(num)/float(den)
                
        sym_ops[key] = m

    return sym_ops


def extract_remarks290(filename):
    """
    Extract the symmetry operations listed in the REMARKS 290 section of 
    PDB header and store the string representations in output dictionary.
    
    Inputs:
    -------
    filename: path to PDB file
    
    Outputs:
    --------
    remarks: dictionary of symmetry operations in string format
    """
    
    counter = 0
    remarks = dict()
    
    with open(filename, "r") as f:
        
        # only extract lines corresonding to symmetry operations
        extract = False
        for line in f:
            
            if "REMARK 290     NNNMMM   OPERATOR" in line:
                extract = True
                continue
                
            elif line.split()[-1] == "290":
                extract = False
                continue
                
            elif extract:
                remarks[counter] = line.split()[-1]
                counter += 1
                
    return remarks


def extract_space_group(filename):
    """
    Extract the space group symbol (Hermann-Mauguin notation) listed in the
    CRYST1 record of the PDB header.

    Inputs:
    -------
    filename: path to PDB file

    Outputs:
    --------
    sg_symbol: space group symbol, string
    """
    with open(filename, "r") as f:
        for line in f:
            if "CRYST1" in line:
                sg_symbol = line[55:65].rstrip()

    return sg_symbol


################################# SYMMETRY OPERATIONS #################################

sym_ops = dict()
    
# symmetry relationships from PDB files
filenames = dict()
filenames = glob.glob("reference/pdb_files/*.pdb")

for fname in filenames:
    key = extract_space_group(fname)
    as_str = extract_remarks290(fname)
    sym_ops[key] = str_to_matrix(as_str)

# generate reference directory if it doesn't already exist  
if not os.path.isdir("reference"):
    os.mkdir("reference")

# save dictionary as reference/sym_ops.pickle
with open("reference/sym_ops.pickle", "wb") as handle:
    pickle.dump(sym_ops, handle)

################################# RESTRICTED PHASES ###################################

res_phases = dict()
res_phases['P 21 21 21'] = np.array([180.0, 90.0])
res_phases['P 43 21 2'] = np.array([180.0, 45.0])
res_phases['F 4 3 2'] = np.array([180.0, 180.0])

# save dictionary as reference/restricted_phases.pickle
with open("reference/phase_restrictions.pickle", "wb") as handle:
    pickle.dump(res_phases, handle)


############################## REINDEXING OPERATIONS #################################

reidx_ops = dict()
reidx_ops['P 43 21 2'] = dict()
reidx_ops['P 43 21 2'][0] = np.array([[1,0,0],[0,1,0],[0,0,1]]) # (h,k,l)
reidx_ops['P 43 21 2'][1] = np.array([[-1,0,0],[0,-1,0],[0,0,1]]) # (-h,-k,l)
reidx_ops['P 43 21 2'][2] = np.array([[0,1,0],[1,0,0],[0,0,-1]]) # (k,h,-l)
reidx_ops['P 43 21 2'][3] = np.array([[0,-1,0],[-1,0,0],[0,0,-1]]) # (-k,-h,-l)

# save dictionary as reference/reindexing_ops.pickle
with open("reference/reindexing_ops.pickle", "wb") as handle:
    pickle.dump(reidx_ops, handle)
