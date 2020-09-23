from collections import OrderedDict
import time, argparse, os
import itertools, glob, random, mrcfile
import cPickle as pickle
import numpy as np
import pathos.pools as pp
from natsort import natsorted

import ProcessCrystals as proc
import cctbx_utils, mock_data
import utils, random_eulers
import iotbx.pdb
from cctbx import miller, maptbx
from cctbx.array_family import flex


def parse_commandline():
    """
    Parse command line input and set default values for inputs not given.
    """
    parser = argparse.ArgumentParser(description='Find origin for 2d simulated data.')
    parser.add_argument('-p','--pdb_path', help='Path to coordinates file', required=True)
    parser.add_argument('-a','--angle', help='Tilt angle', required=True, type=float)
    parser.add_argument('-np','--n_processes', help='Number of CPU processors', required=True, type=int)
    parser.add_argument('-g','--grid_spacing', help='Grid spacing in Angstroms', required=True, type=float)
    parser.add_argument('-r','--resolution', help='High-resolution limit', required=True, type=float)
    parser.add_argument('-w','--ang_width', help='Angular width', required=False, type=float, default=1.0)
    parser.add_argument('-m','--A', help='Crystallographic orientation matrix, .npy file', required=False, type=str)
    parser.add_argument('-o','--output', help='Path for saving shifts and metrics -- prefix only', required=True)

    return vars(parser.parse_args())


def predict_angles(pdb_path, resolution, A):
    """
    Predict the tilt angle at which each reflection will be observed. Here a positive
    tilt angle corresponds to images with a +y coordinate. Reflections that lie in the
    missing wedge are excluded.
    
    Inputs:
    -------
    pdb_path: path to reference PDB file
    resolution: high-resolution limit of structure factors
    A: crystal setting matrix

    Outputs:
    --------
    hkl_t: dict with keys as Millers and values as tilt angles
    """
    # predict coordinates of all reflections in reciprocal pixels
    sg_symbol, sg_no, cell, cs = cctbx_utils.unit_cell_info(pdb_path)
    hkl = np.array(miller.build_set(crystal_symmetry=cs,
                                    anomalous_flag=True,
                                    d_min=resolution).expand_to_p1().indices())
    qvecs = np.inner(A, np.squeeze(hkl)).T

    # predict tilt angle from associated coordinates
    t = np.rad2deg(np.arctan2(qvecs[:,1], qvecs[:,0]))
    t[(t>90) & (t<=180)] = utils.wraptopi(t[(t>90) & (t<180)] + 180.0) # shift q2 to q4
    t[(t>=-180) & (t<=-90)] += 180.0 # shift q3 to q1
    
    # generate a dict with keys as Millers and values as tilt angles
    hkl_t = OrderedDict((tuple(key),val) for key,val in zip(hkl,t))    
    return hkl_t


def simulate_one_tilt(pdb_path, resolution, angle, A=np.eye(3), ang_width=1.0):
    """
    Generate reflection data within +/- ang_width of input angle, and subject to a 
    random, global phase shift. 
    
    Inputs:
    -------
    pdb_path: path to coordinates file
    resolution: maximum resolution to which to compute reflection data
    angle: tilt angle reflection data are centered on
    A: orientation matrix by which to rotate crystal, optional
    ang_width: half of increment within angle for which reflections will be kept
    
    Outputs:
    --------
    hklI_sel: OrderedDict of retained Miller indices and corresponding intensities
    hklp_sel: OrderedDict of retained Miller indices and corresponding phases
    shifts: global phase shift reflection data were subjected to
    """
    # compute reference information
    sg_symbol, sg_no, cell, cs = cctbx_utils.unit_cell_info(pdb_path)
    refI, refp = cctbx_utils.reference_sf(pdb_path, resolution, expand_to_p1=True)    
        
    # rotate reflection data and compute tilt angles
    hklt = predict_angles(pdb_path, resolution, A)
    hkl, tilts = np.array(hklt.keys()), np.array(hklt.values())

    # retain reflection data in observed angular range; add random phase shift
    hkl_sel = hkl[np.where((tilts > angle - ang_width) & (tilts < angle + ang_width))[0]]
    hkl_sel = [tuple(h) for h in hkl_sel]
    hkl_sel = utils.remove_Friedels(hkl_sel)
    
    hklI_sel = OrderedDict((key,val) for key,val in refI.items() if key in hkl_sel)
    hklp_sel = OrderedDict((key,val) for key,val in refp.items() if key in hkl_sel)
    hklp_sel, shifts = mock_data.add_random_phase_shift(hklp_sel)
    
    return hklI_sel, hklp_sel, shifts


def find_origin(pdb_path, hklI, hklp, grid_spacing, n_processes):
    """
    Determine shifts required to locate crystallographic phase origin.
    
    Inputs:
    -------
    pdb_path: path to coordinates file
    hklI: OrderedDict of Miller indices and corresponding intensities
    hklp: OrderedDict of Miller indices and corresponding intensities
    grid_spacing: search interval for finding origin in Angstrom
    n_processes: number of processes over which to parallelize computation
    
    Outputs:
    --------
    dmetrics: dictionary of evaluating phase origin metrics
    shifts: tuple of predicted fractional origin shifts
    """
    
    sg_symbol, sg_no, cell, cs = cctbx_utils.unit_cell_info(pdb_path)
    fo = proc.FindOrigin(sg_symbol, cell, cs, hklp, hklI)
    dmetrics, shifts = fo.scan_candidates(grid_spacing, n_processes)

    print "No. centric reflections: %i" %(len(fo.cslice_idx) + len(fo.caxis_idx))
    print "No. symm-equivalent hkl: %i" %(len(np.nonzero(fo.base_idx - fo.sym_idx)[0]))
    
    return dmetrics, shifts


if __name__ == '__main__':
    
    start_time = time.time()

    # extract command line arguments
    args = parse_commandline()
    
    # load or generate a crystallographic orientation matrix
    if args['A'] is None:
        A = mock_data.generate_random_A(args['pdb_path'])
    else:
        A = np.load(args['A'])

    # simulate and find origin for given tilt angle   
    hklI, hklp, r_shifts = simulate_one_tilt(args['pdb_path'], args['resolution'], 
                                             args['angle'], A=A, ang_width=args['ang_width'])
    dmetrics, c_shifts = find_origin(args['pdb_path'], hklI, hklp, args['grid_spacing'], args['n_processes'])
    shifts = np.vstack((r_shifts, c_shifts))
    
    # save shifts and metrics
    np.save(args['output'] + ".npy", shifts)
    with open(args['output'] + ".pickle", "wb") as handle:
        pickle.dump(dmetrics, handle)

    # save selection of phases and intensities recorded for this tilt image
    for d,suffix in zip([hklI, hklp],['_I','_p']):
        with open(args['output'] + suffix + ".pickle", "wb") as handle:
            pickle.dump(d, handle)

    print "elapsed time is %.2f" %((time.time() - start_time)/60.0)
