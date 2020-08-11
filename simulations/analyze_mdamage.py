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
    parser = argparse.ArgumentParser(description='Generate and analyze mock data for a range of phase errors and completeness.')
    parser.add_argument('-p','--pdb_path', help='Path to reference structure file in PDB format', required=True)
    parser.add_argument('-sp','--specs_path', help='Path to dict specifying data collection details', required=True)
    parser.add_argument('-n','--n_cryst', help='Number of crystals to merge per run', required=True, type=int)
    parser.add_argument('-nr','--n_rep', help='Number of runs per error/completeness combination', required=True, type=int)
    parser.add_argument('-np','--n_processes', help='Number of CPU processors', required=True, type=int)
    parser.add_argument('-g','--grid_spacing', help='Grid spacing in Angstroms for phase origin search', required=True, type=float)
    parser.add_argument('-r','--resolution', help='High-resolution limit', required=True, type=float)
    parser.add_argument('-c','--completeness', help='Fractional completeness in P1', required=True, type=float) 
    parser.add_argument('-s','--sigma', help='Sigma determining phase error distribution', required=True, type=float)
    parser.add_argument('-b','--bfactor', help='Initial B-factor in Angstrom squared', required=True, type=float)
    parser.add_argument('-o','--output', help='Path and prefix for saving merged information', required=True)

    return vars(parser.parse_args())


def initial_errors(hklp, refp, shifts):
    """
    Compute the starting phase errors to reference; output metrics are sigma
    of the normal distribution and mean of phase errors to reference.
    
    Inputs:
    -------
    hklp: dict with Millers as keys and values as phases
    refp: dict with Millers as keys and values as reference phases
    shifts: fractional shifts that reposition hklp on reference phase origin
    
    Outputs:
    --------
    sigma: variance of normal distribution fit to phase residuals to reference
    m_error: mean of phase residuals to reference
    """
    # shift phases back to reference origin
    hkl, p = np.array(hklp.keys()), np.array(hklp.values())
    p_unshift = utils.wraptopi(p + 360 * np.dot(hkl, shifts).ravel())
    
    # remove Friedel mates from the shifted phases and compute error metrics
    hkl_sel = utils.remove_Friedels(hklp.keys())    
    hklp_unshift = OrderedDict((key,val) for key,val in zip(hklp.keys(),p_unshift) if key in hkl_sel)
    sigma, m_error = utils.residual_phase_distribution(refp, hklp_unshift)
    
    return sigma, m_error


def err_origin_shifts(eq_origins, dshifts, cell):
    """
    Compute mean error in shifts predicted to locate crystal on a valid origin.
    Calculation uses the shifts used to generate the mock data (expected key).
    
    Inputs:
    -------
    eq_origins: array of equivalent origins in fractional cell shifts
    dshifts: dict of fractional shifts, including 'origin' and 'reference' keys
    cell: input crystal's cell information, (a,b,c,alpha,beta,gamma)

    Outputs:
    --------
    error: array of magnitude of origin shifts errors in Ang, shape n_cryst
    """
    p_origins = dshifts['origin'] + dshifts['expected'][0]
    p_origins[p_origins>1.0] = p_origins[p_origins>1.0] - 1.0
    p_origins[p_origins<0.0] = p_origins[p_origins<0.0] + 1.0
    p_origins = p_origins[:,:,np.newaxis]

    e_origins = np.min(np.abs(eq_origins - p_origins), axis=2)
    e_origins_A = e_origins * np.array(cell)[:3]

    return np.linalg.norm(e_origins_A, axis=1)
    
    
def err_merge_shifts(dshifts, cell):
    """
    Compute mean error in shifts predicted to merge crystals. The expected origin is 
    the origin of the first crystal added during merging, dshifts['expected'][0], and 
    this should match the vector sum of the shifts identified during merging and the
    expected shifts for each subsequently added crystal.
    
    Inputs:
    -------
    dshifts: dict of fractional shifts, including 'expected' and 'merge' keys
    cell: input crystal's cell information, (a,b,c,alpha,beta,gamma)

    Outputs:
    --------
    error: array of magnitude of origin shifts errors in Ang, shape n_cryst
    """
    eq_positions = np.array([0.0,1.0])
    p_merge = np.abs(dshifts['expected'][0] - dshifts['merge'] - dshifts['expected'])
    p_merge = p_merge[:,:,np.newaxis]

    e_merge = np.min(np.abs(eq_positions - p_merge), axis=2)
    e_merge_A = e_merge * np.array(cell)[:3]
    
    return np.linalg.norm(e_merge_A, axis=1)


def select_hkl(cs, resolution, completeness, hklId):
    """
    Retain a list of (non-unique) reflections from hklId based on their intensities
    to achieve a list with desired P1 (but not anomalous) completeness, so Friedels
    are assumed to be equivalent, but crystallographic symmetry is not considered.
    
    Inputs:
    -------
    cs: CCTBX crystal symmetry object
    resolution: high-resolution limit
    completeness: desired (unique) completeness
    hklId: dict with keys as Millers and values as intensities
    
    Outputs:
    --------
    hkl_sel: list of retained reflections
    """
    # extract list of all P1 reflections and compute number for desired completeness
    hkl_p1 = list(cs.build_miller_set(anomalous_flag=False, 
                                       d_min=resolution).expand_to_p1().indices())
    n_est = int(np.around(completeness * len(hkl_p1)))
    
    print "Number of unique, nonanomalous P1 reflections: %i" %len(hkl_p1)
    print "Number of desired reflections: %i" %(n_est)
    
    # sort available reflections in order of decreasing intensity
    s_idx = np.argsort(np.array(hklId.values()))[::-1]

    # extract list of reflections to retain for desired completeness
    if n_est > len(hklId.keys()):
        print "Warning: more reflections required than available in tilt range"
        print "All available reflections will be retained"
        hkl_sel = hklId.keys()
    else:
        hkl_sel = list()
        i, counter = 0, 0
        while (counter < n_est) and (i < len(s_idx)):
            key = hklId.keys()[s_idx[i]]
            hkl_sel.append(key)
            if key in hkl_p1: counter += 1
            i += 1
            
    print "Number of total retained reflections is: %i" % len(hkl_sel)
    hkl_unique_ret = [key for key in hkl_sel if key in hkl_p1]
    print "Number of retained unique reflections is: %i" %(len(hkl_unique_ret))
            
    # print information about ASU / P1 completeness
    for cs_obj,tag in zip([cs, cs.cell_equivalent_p1()], ['SG', 'P1']):
        ma_calc = miller.array(miller_set=miller.set(cs_obj,
                                                     flex.miller_index(hkl_sel),
                                                     anomalous_flag=False), 
                               data=flex.double(np.ones_like(hkl_sel)))
        print "%s completeness: %.2f" %(tag, ma_calc.merge_equivalents().array().completeness())
            
    return hkl_sel


def assess_mock_data(args, rep):
    """
    Evaluate a series of mock crystals with specified completeness and phase errors.
    
    Inputs:
    -------
    args: dict specifying n_processes, grid_spacing, pdb_path, resolution, n_cryst,
          completeness, sigma, specs_path, tilt_list, bfactor
    rep: repetition number currently being computed

    Outputs:
    --------
    results: dict of r_ref, r_sym, completeness, cc_map
    shifts: dict of expected, merging, and origin shifts
    hkl_tilts_Id: dict of retained hkl: np.array([tilt angle, intensity])
    """
    # compute reference information
    sg_symbol, sg_no, cell, cs = cctbx_utils.unit_cell_info(args['pdb_path'])
    refI, refp = cctbx_utils.reference_sf(args['pdb_path'], 
                                          args['resolution'],
                                          expand_to_p1=True, 
                                          table='electron')
    refp_mod = OrderedDict((key,np.array([val])) for key,val in refp.items())
    
    # set up dictionary for storing results
    results, shifts, hkl_tilts_Id, A_matrices = dict(), dict(), dict(), dict()
    for key in ['sigma_i', 'r_ref_i', 'sigma', 'r_ref', 'r_sym', 'completeness_p1', 'completeness_sg', 'cc_map', 'cc_I']:
        results[key] = np.zeros(args['n_cryst'])
    for key in ['reference', 'expected', 'origin', 'merge']:
        shifts[key] = np.zeros((args['n_cryst'], 3))
    if sg_no == 1: 
        shifts.pop('origin')
        
    # set up MergeCrystals class for scan over crystals
    mc = proc.MergeCrystals(space_group=sg_no, grid_spacing=args['grid_spacing'])
    
    for num in range(args['n_cryst']):
        # apply analytic model of damage to a randomly-oriented crystal
        hklId, hklpd, hkltd, A_matrices[num] = mock_data.simulate_damage(args['specs_path'],
                                                                         args['pdb_path'],
                                                                         args['resolution'],
                                                                         args['tilt_list'],
                                                                         args['bfactor'])

        # retain highest intensity reflections for specified completeness
        hkl_sel = select_hkl(cs, args['resolution'], args['completeness'], hklId) 
    
        # generate mock data based on above hkl list
        hklI, hklp, shifts['expected'][num] = mock_data.generate_mock_data(args['pdb_path'],
                                                                           args['resolution'],
                                                                           completeness=1.0,
                                                                           hkl_sel=hkl_sel,
                                                                           sigma=args['sigma'])
        hklI_d = OrderedDict((key,hklId[key]) for key in hklI.keys())
        hkl_tilts_Id[num] = OrderedDict((key,np.array([hkltd[key],hklI_d[key]])) for key in hklI.keys())
                
        # compute starting phase errors and store
        sigma_i, m_error = initial_errors(hklp, refp, shifts['expected'][num])
        results['sigma_i'][num], results['r_ref_i'][num] = sigma_i, m_error

        # add to MergeCrystals object
        mc.add_crystal(hklI_d, hklp, np.array(cell), n_processes=args['n_processes'], weighted=True)
        hklI_avg, hklp_avg = mc.merge_values()  
        
        # locate crystallographic origin if not triclinic
        ##if sg_no != 1:
        ##    fo = proc.FindOrigin(sg_symbol, cell, cs, hklp_avg, hklI_avg)
        ##    dmetrics, shifts['origin'][num] = fo.scan_candidates(args['grid_spacing'], 
        ##                                                         args['n_processes'])

        # shift phases to match reference for easier comparison 
        comp = proc.CompareCrystals(cell, hklp_avg, hklI_avg)
        m_grid, hklp_shifted, shifts['reference'][num] = comp.grid_shift(refp, 
                                                                         args['grid_spacing'],
                                                                         args['n_processes'],
                                                                         refI)
        hklI_merge = OrderedDict((key,val) for key,val in mc.hklI.items())
        hklp_merge = OrderedDict((key,val) for key,val in mc.hklp.items())

        # reduce phases to asymmetric unit
        rc = proc.ReduceCrystals(hklI_merge, hklp_merge, cell, sg_symbol)
        rc.shift_phases(shifts['reference'][num])
        p_asu = rc.reduce_phases(weighted=True)
        rc.reduce_intensities()
        mtz_object = rc.generate_mtz()
            
        # examine completeness and cross-correlation with reference map
        #savename = "%s_r%in%i.ccp4" %(args['output'],rep,num)
        ma = cctbx_utils.mtz_to_miller_array(mtz_object)
        map_sim = cctbx_utils.compute_map(ma)
        results['cc_map'][num] = cctbx_utils.compare_maps(map_sim, args['pdb_path'])

        # compute completeness -- both (non-anomalous) P1 and space group
        for cs_obj,tag in zip([cs, cs.cell_equivalent_p1()],['completeness_sg', 'completeness_p1']):
            ma_calc = miller.array(miller_set=miller.set(cs_obj,
                                                         flex.miller_index(rc.hklI.keys()),
                                                         anomalous_flag=False),
                                   data=flex.double(np.ones(len(rc.hklI.keys()))))
            results[tag][num] = ma_calc.merge_equivalents().array().completeness()        
                        
        # assess quality of phases
        results['sigma'][num], results['r_ref'][num] = utils.residual_phase_distribution(refp, rc.data['PHIB'])
        results['r_sym'][num] = np.mean(utils.residual_to_avgphase(p_asu))
        
        # assess accuracy of intensities: log-log cross-correlation
        ivals = np.array(utils.shared_dict(refI, rc.data['IMEAN']).values())
        results['cc_I'][num] = np.corrcoef(np.log10(ivals[:,0]), np.log10(ivals[:,1]))[0,1]

        # examine errors in predicted fractional shifts
        if num != 0:
            shifts['merge'][num] = mc.fshifts[num]

        print "\n"

    return results, shifts, hkl_tilts_Id, A_matrices


if __name__ == '__main__':
    
    start_time = time.time()

    # extract command line arguments
    args = parse_commandline()

    # dose-symmetric tilt-scheme
    specs = pickle.load(open(args['specs_path']))
    tilts = np.zeros(41)
    n_tilts = np.arange(-1*specs['increment'], -1*(specs['angle']+specs['increment']), -1*specs['increment'])
    p_tilts = np.arange(0, specs['angle']+specs['increment'], specs['increment'])
    tilts[1::2], tilts[::2] = n_tilts, p_tilts
    args['tilt_list'] = tilts

    # crystal information and equivalent fractional origins for P 21 21 21
    sg_symbol, sg_no, cell, cs = cctbx_utils.unit_cell_info(args['pdb_path'])
    eq_origins = np.array([0.0,0.5,1.0])

    # set up dictionaries and format parameters key
    print "Completeness: %.2f, Sigma: %.2f" %(args['completeness'], args['sigma'])
    dstore, dshifts, tstore, Astore = dict(), dict(), dict(), dict()
    
    # generate and analyze n_rep runs for the initial (completeness,sigma) combination
    for r in range(args['n_rep']):
        print "Rep. %i" %(r)
        dstore[r], dshifts[r], tstore[r], Astore[r] = assess_mock_data(args, r)
        ##if 'origin' in dshifts[r].keys():
        ##    dstore[r]['rs_origin'] = err_origin_shifts(eq_origins, dshifts[r], cell)
        dstore[r]['rs_merge'] = err_merge_shifts(dshifts[r], cell)

    # reformat output for easier analysis, values have shape (n_rep,n_cryst)
    d_series = OrderedDict()
    for label in dstore[0].keys():
        d_series[label] = np.array([dstore[key][label] for key in dstore.keys()])

    # save series analysis, fractional shifts information, and tilt angle distributions
    with open("%s.pickle" %(args['output']), "wb") as handle:
        pickle.dump(d_series, handle)
    with open("%s_tilts_Id.pickle" %(args['output']), "wb") as handle:
        pickle.dump(tstore, handle)
    with open("%s_fshifts.pickle" %(args['output']), "wb") as handle:
        pickle.dump(dshifts, handle)
    with open("%s_A.pickle" %(args['output']), "wb") as handle:
        pickle.dump(Astore, handle)

    print "elapsed time is %.2f" %((time.time() - start_time)/60.0)


