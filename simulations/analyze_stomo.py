from collections import OrderedDict
import time, argparse, os
import itertools, glob, random, mrcfile
import cPickle as pickle
import numpy as np
import pathos.pools as pp
from natsort import natsorted
import ProcessCrystals as proc

import utils, cctbx_utils, iotbx.pdb
from cctbx import miller
from cctbx.array_family import flex

def parse_commandline():
    """
    Parse command line input and set default values for inputs not given.
    """
    parser = argparse.ArgumentParser(description='Merge, find origin, and reduce simulated tomograms.')
    parser.add_argument('-p','--pdb_path', help='Path to reference structure file in PDB format', required=True)
    parser.add_argument('-b','--base_path', help='Path to base directory containing tomo directories', required=True)
    parser.add_argument('-np','--n_processes', help='Number of CPU processors', required=True, type=int)
    parser.add_argument('-g','--grid_spacing', help='Grid spacing in Angstroms for phase origin search', required=True, type=float)
    parser.add_argument('-r','--resolution', help='High-resolution limit', required=True, type=float)
    parser.add_argument('-o','--output', help='Path at which to save merged information', required=True)
    parser.add_argument('-d','--dsel', help='Indices of datasets to retain, formatted eg 2,1,0', required=False)

    return vars(parser.parse_args())


def data_paths(args):
    """
    Collect ordered paths to indexed.json, estimated phase, and estimated intensity files.
    
    Inputs:
    -------
    args: dict specifying base_path to results from indexing and peak fitting
    
    Outputs:
    --------
    data_paths: dict containing paths to json and fitted peak files
    """
    if args['dsel'] is not None:
        dsel = [int(x) for x in args['dsel'].split(",")]

    data_paths = dict()   
    for ftype in ['json', 'phases', 'intensities']:
        data_paths[ftype] = list()

    for ds in dsel:
        data_paths['json'].append(os.path.join(args['base_path'], "tomo_%i/dials_tomo%i/indexed.json" %(ds,ds)))
        data_paths['phases'].append(os.path.join(args['base_path'], "tomo_%i/peaks_tomo%i/estp.pickle" %(ds,ds)))
        data_paths['intensities'].append(os.path.join(args['base_path'], "tomo_%i/peaks_tomo%i/estI.pickle" %(ds,ds)))

    n_files = np.array([len(data_paths[key]) for key in data_paths.keys()])
    assert all([i==n_files[0] for i in n_files])
    
    return data_paths


def order_datasets(args):
    """
    Reorder the datasets based on maximizing the number of shared reflections
    being merged at each iteration.
    
    Inputs:
    -------
    args: dict that includes a 'data_paths' key to intensity file names
    
    Outputs:
    --------
    args: updated dict, with 'data_paths' key reordered
    """
    order = np.zeros(len(args['data_paths']['intensities'])).astype(int)

    # retrieve lists of reflections available in each dataset
    hkl = dict()
    for i,fname in enumerate(args['data_paths']['intensities']):
        hkl[i] = pickle.load(open(fname)).keys()

    # determine which pair of tomgorams will be added first
    combinations = list(itertools.combinations(hkl.keys(), 2))
    n_shared = np.zeros(len(combinations))

    hkl_list = list()
    for i,c in enumerate(combinations):
        n_shared[i] = len(set(hkl[c[0]]).intersection(set(hkl[c[1]])))

    c_interest = combinations[np.where(n_shared == n_shared.max())[0][0]]
    for i,c in enumerate(c_interest):
        hkl_list.extend(hkl[c])
        order[i] = c
        hkl.pop(c)
        
    # determine order of the remaining tomograms
    num = 2
    while num < len(order):
        n_shared = dict()
        for i in hkl.keys():
            n_shared[i] = len(set(hkl[i]).intersection(set(hkl_list)))
        c = max(n_shared, key=n_shared.get)
        hkl_list.extend(hkl[c])
        order[num] = c
        hkl.pop(c)
        num += 1

    # reorder json, intensities, and phase filenames in data_paths
    for ftype in args['data_paths'].keys():
        args['data_paths'][ftype] = [args['data_paths'][ftype][num] for num in order]
        
    return args


def assess_tr_data(args):
    """
    Evaluate a series of mock crystals from randomly oriented tomograms.
    
    Inputs:
    -------
    args: dict of pdb_path, n_processes, grid_spacing, resolution, n_cryst, data_paths
    
    Outputs:
    --------
    results: dict of r_ref, r_sym, completeness, cc_map
    shifts: dict of shifts for merging, finding origin, comparing to reference
    """
    # compute reference information
    pdb_input = iotbx.pdb.input(file_name=args['pdb_path'])
    ref_sg_symbol, ref_sg_no, ref_cell, ref_cs = cctbx_utils.unit_cell_info(args['pdb_path'])
    refI, refp = cctbx_utils.reference_sf(args['pdb_path'], 
                                          args['resolution'],
                                          expand_to_p1=True, 
                                          table='electron')
    refp_mod = OrderedDict((key,np.array([val])) for key,val in refp.items())
    
    # set up dictionary for storing results
    results, shifts = dict(), dict()
    for key in ['r_ref', 'r_sym', 'completeness_p1', 'completeness_sg', 'cc_map', 'cc_I']:
        results[key] = np.zeros(args['n_cryst'])
    for key in ['reference', 'origin', 'merge']:
        shifts[key] = np.zeros((args['n_cryst'], 3))
    if ref_sg_no == 1: shifts.pop('origin')
        
    # set up MergeCrystals class for scan over crystals
    mc = proc.MergeCrystals(space_group=ref_sg_no, grid_spacing=args['grid_spacing'])
    
    for num in range(args['n_cryst']):
        print "Intensites from: %s" %(args['data_paths']['intensities'][num])
        print "Phases from: %s" %(args['data_paths']['phases'][num])
        print "Indexing matrix from: %s" %(args['data_paths']['json'][num])

        # load simulated data
        hklI = pickle.load(open(args['data_paths']['intensities'][num]))
        hklp = pickle.load(open(args['data_paths']['phases'][num]))
        sg_symbol, sg_no, cell, cs = cctbx_utils.unit_cell_info(args['data_paths']['json'][num])
        assert sg_symbol == ref_sg_symbol
        
        # filter Millers below resolution cutoff and with negative intensities
        res = utils.compute_resolution(sg_no, cell, np.array(hklp.keys()))
        indices = np.where((res<args['resolution']) | (np.array(hklI.values())<0))[0]
        hkl_to_remove =  list()
        for idx in indices: hkl_to_remove.append(hklp.keys()[idx])
        for htr in hkl_to_remove:
            hklp.pop(htr)
            hklI.pop(htr)
        
        # merge crystal
        mc.add_crystal(hklI, hklp, np.array(cell), n_processes=args['n_processes'], weighted=True)
        hklI_avg, hklp_avg = mc.merge_values()  
        avg_cell = tuple(np.average(mc.cell_constants.values(), axis=0))
        
        # locate crystallographic origin if not triclinic
        if sg_no != 1:
            fo = proc.FindOrigin(sg_symbol, avg_cell, cs, hklp_avg, hklI_avg)
            dmetrics, shifts['origin'][num] = fo.scan_candidates(args['grid_spacing'], 
                                                                 args['n_processes'])
            
        # shift phases to match reference for easier comparison 
        comp = proc.CompareCrystals(avg_cell, hklp_avg, hklI_avg)
        m_grid, hklp_shifted, shifts['reference'][num] = comp.grid_shift(refp, 
                                                                         args['grid_spacing'], 
                                                                         args['n_processes'],
                                                                         refI)
        hklI_merge = OrderedDict((key,val) for key,val in mc.hklI.items())
        hklp_merge = OrderedDict((key,val) for key,val in mc.hklp.items())
    
        # reduce phases to asymmetric unit
        rc = proc.ReduceCrystals(hklI_merge, hklp_merge, avg_cell, sg_symbol)
        rc.shift_phases(shifts['reference'][num])
        p_asu = rc.reduce_phases(weighted=True)
        rc.reduce_intensities()
        mtz_object = rc.generate_mtz()
        
        # examine cross-correlation with reference map
        savename = os.path.join(args['output'], "reduced%i.ccp4" %num)
        ma = cctbx_utils.mtz_to_miller_array(mtz_object)
        map_sim = cctbx_utils.compute_map(ma, save_name = savename)
        results['cc_map'][num] = cctbx_utils.compare_maps(map_sim, args['pdb_path'])

        # compute completeness -- both (non-anomalous) P1 and space group
        for cs_obj,tag in zip([cs, cs.cell_equivalent_p1()],['completeness_sg', 'completeness_p1']):
            ma_calc = miller.array(miller_set=miller.set(cs_obj,
                                                         flex.miller_index(rc.hklI.keys()),
                                                         anomalous_flag=False),
                                   data=flex.double(np.ones(len(rc.hklI.keys()))))
            results[tag][num] = ma_calc.merge_equivalents().array().completeness()

        # assess quality of phases
        phib = OrderedDict((key, np.array([val])) for key,val in rc.data['PHIB'].items())
        results['r_ref'][num] = np.mean(utils.residual_phases(phib, refp_mod))
        results['r_sym'][num] = np.mean(utils.residual_to_avgphase(p_asu))

        # assess accuracy of intensities: log-log cross-correlation
        ivals = np.array(utils.shared_dict(refI, rc.data['IMEAN']).values())
        results['cc_I'][num] = np.corrcoef(np.log10(ivals[:,0]), np.log10(ivals[:,1]))[0,1]
        
        # examine errors in predicted fractional shifts
        if num != 0:
            shifts['merge'][num] = mc.fshifts[num]
    
        print "\n"

    return results, shifts


if __name__ == '__main__':
    
    start_time = time.time()

    # retrieve parameters from command line input
    args = parse_commandline()
    args['data_paths'] = data_paths(args)
    args['n_cryst'] = len(args['data_paths']['json'])
    if not os.path.isdir(args['output']):
        os.mkdir(args['output'])

    # reduce, find origin, merge
    results, shifts = assess_tr_data(args)

    # save statistics to pickle files
    for d,fname in zip([results,shifts],['stats','shifts']):
        with open(os.path.join(args['output'],"%s.pickle" %fname), "wb") as handle:
            pickle.dump(d,handle)

    print "elapsed time is %.2f" %((time.time() - start_time)/60.0)
