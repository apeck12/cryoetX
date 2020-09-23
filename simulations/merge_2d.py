from collections import OrderedDict
import time, argparse, os
import itertools, glob, random, mrcfile, operator
import cPickle as pickle
import numpy as np
from natsort import natsorted
import ProcessCrystals as proc
import utils, cctbx_utils

from cctbx import miller, maptbx
import iotbx.pdb
from iotbx import ccp4_map, file_reader, mtz
from dxtbx.model.experiment_list import ExperimentListFactory
from cctbx.array_family import flex


def parse_commandline():
    """
    Parse command line input and return as dictionary.
    """
    parser = argparse.ArgumentParser(description='Merge reflection data from tilt images.')
    parser.add_argument('-d','--dpath', help='Input directory', required=True)
    parser.add_argument('-p','--pdb_path', help='path to coordinates files', required=True)
    parser.add_argument('-t','--tilt_angles', help='path to tilt angles file', required=True)
    parser.add_argument('-a','--start_angle', help='Start angle', required=True, type=float)
    parser.add_argument('-n','--n_processes', help='Number of CPU processors', required=True, type=int)
    parser.add_argument('-o','--output', help='Output directory', required=True)

    return vars(parser.parse_args())


class MergeImages:
    
    """
    Class for merging reflections from tilt images.
    """
    
    def __init__(self, grid_spacing, ref_path):
        """
        Set up instance of class.
        """
        self.grid_spacing = grid_spacing # grid spaicng used during crystallographic origin search
        self.sg_symbol, self.sg_no, self.cell, self.cs = cctbx_utils.unit_cell_info(ref_path)
        self.merge_p, self.merge_I = OrderedDict(), OrderedDict()
        
        
    def candidate_origins(self, metrics, n_candidates):
        """
        Determine top candidate origins, and return their corresponding fractional shifts.

        Inputs:
        -------
        metrics: 3d np.array of origin scores
        n_candidates: number of top-ranking origins to return

        Outputs:
        --------
        cand_shifts: fractional shifts for highest-ranked candidate crystallographic origins
        """
        # determine the fractional shift corresponding to each grid node
        xshifts, yshifts, zshifts = [np.arange(0, self.cell[i], 
                                               self.grid_spacing) for i in range(3)]
        fshifts_list = list(itertools.product(xshifts/self.cell[0], 
                                              yshifts/self.cell[1], 
                                              zshifts/self.cell[2]))
        # sort the metrics
        sorted_idx = np.argsort(metrics.flatten())[:n_candidates]
        cand_shifts = np.array(fshifts_list)[sorted_idx]

        return cand_shifts
    
    
    def reduce_tilt(self, hklp, hklI, shifts):
        """
        Shift phases to input origin and reduce.
        
        Inputs:
        -------
        hklp: OrderedDict of hkl: phases
        hklI: OrderedDict of hkl: intensities
        shifts: fractional shifts for changing phase origin
        
        Outputs:
        --------
        hklp_r: OrderedDict of reduced hkl: phases
        hklp_I: OrderedDict of reduced hkl: intensities
        """
        # retrieve reference structure factors
        hklI_mod = OrderedDict((key,np.array([val])) for key,val in hklI.items())
        hklp_mod = OrderedDict((key,np.array([val])) for key,val in hklp.items())

        # reduce data
        rc = proc.ReduceCrystals(hklI_mod, hklp_mod, self.cell, self.sg_symbol)
        rc.shift_phases(shifts)
        p_asu = rc.reduce_phases(weighted=True)
        rc.reduce_intensities()

        return rc.data['PHIB'], rc.data['IMEAN']

                
    def add_first_image(self, hklp, hklI, metrics):
        """
        Add data from first image, reducing based on the top crystallographic origin.
        
        Inputs:
        -------
        hklp: OrderedDict of hkl: phases
        hklI: OrderedDict of hkl: intensities
        metrics: 3d array of crystallographic origin scores for each fractional cell shift
        """
        origin = self.candidate_origins(metrics, 1)
        hklp_r, hklI_r = self.reduce_tilt(hklp, hklI, origin[0]) 
        self.merge_p = OrderedDict((key,np.array([val])) for key,val in hklp_r.items())
        self.merge_I = OrderedDict((key,np.array([val])) for key,val in hklI_r.items())
        
        return origin
    
    
    def score_image(self, hklp, hklI, metrics, n_candidates):
        """
        Score image by computing lowest phase residual of the top n_candidates
        performing crystallographic phase origins.
        
        Inputs:
        -------
        hklp: OrderedDict of hkl: phases
        hklI: OrderedDict of hkl: intensities
        metrics: 3d array of crystallographic origin scores for each fractional cell shift
        n_candidates: number of candidate phase origins to consider
        
        Outputs:
        --------
        score: score of top-performing origin
        origin: fractional shifts of top-performing origin
        """
        # create modified version of merge_p for comparison to reduced
        merge_p = OrderedDict((key,np.mean(np.array(val))) for key,val in self.merge_p.items())
        
        # rank n_candidate origins based on phase residual
        cand_or = self.candidate_origins(metrics, n_candidates)
        scored = np.zeros(n_candidates)
        for j,origin in enumerate(cand_or):
            hklp_r, hklI_r = self.reduce_tilt(hklp, hklI, origin)
            scored[j] = np.mean(utils.residual_phases(merge_p, hklp_r))
        
        # determine top score and associated fractional shift
        score = np.min(scored)
        origin = cand_or[np.argmin(scored)]
        
        return score, origin
    

    def score_image_merge(self, hklp, hklI, n_processes=4):
        """
        Score image based on merging data in P1 to already recorded reflections.
        
        Inputs:
        -------
        hklp: OrderedDict of hkl: phases
        hklI: OrderedDict of hkl: intensities
        
        Outputs:
        --------
        score: score of top-performing origin
        origin: fractional shifts of top-performing origin
        """
        # create modified version of merge_p and merge_I for addition to MergeCrystals class
        merge_p = OrderedDict((key,np.mean(np.array(val))) for key,val in self.merge_p.items())
        merge_I = OrderedDict((key,np.mean(np.array(val))) for key,val in self.merge_I.items())
        
        # find origin using MergeCrystals class
        mc = proc.MergeCrystals(space_group=self.sg_no, grid_spacing=self.grid_spacing)
        mc.add_crystal(merge_I, merge_p, np.array(self.cell), n_processes=n_processes, weighted=True)
        mc.add_crystal(hklI, hklp, np.array(self.cell), n_processes=n_processes, weighted=True)
        origin = mc.fshifts[1]
        
        # determine phase residual score associated with that origin
        hklp_r, hklI_r = self.reduce_tilt(hklp, hklI, origin)
        score = np.mean(utils.residual_phases(merge_p, hklp_r))
        
        return score, origin
    
    
    def add_next_image(self, hklp, hklI, origin):
        """
        Add data from any image after the first, reducing based on the best-
        performing origin of the top n_candidate fractional shifts.
        
        Inputs:
        -------
        hklp: OrderedDict of hkl: phases
        hklI: OrderedDict of hkl: intensities
        metrics: 3d array of crystallographic origin scores for each fractional cell shift
        n_candidates: number of candidate phase origins to consider
        """
        
        # reduce data using top-scoring origin
        hklp_r, hklI_r = self.reduce_tilt(hklp, hklI, origin)

        # add reduced data
        for m in hklp_r.keys():
            if m in self.merge_p.keys():
                self.merge_p[m] = np.append(self.merge_p[m], hklp_r[m])
                self.merge_I[m] = np.append(self.merge_I[m], hklI_r[m])
            else:
                self.merge_p[m] = np.array([hklp_r[m]])
                self.merge_I[m] = np.array([hklI_r[m]])
                
        return 


def load_tilt_data(dpath, angle, metrics=True):
    """
    Load unshifted phases, intensities, and origin shift for a given tilt angle.
    
    Inputs:
    -------
    dpath: data path containing pimage_* files
    angle: tilt angle of data to load
    
    Outputs:
    --------
    data: dict of hklI, hklp, shifts (input and best), metrics
    """
    # map tilt angles to number in data collection scheme
    tilt_angles_d = OrderedDict((key,val) for key,val in zip(tilt_angles, range(41)))
    i = tilt_angles_d[angle]
    
    # load data and store in dictionary
    data = dict()
    data['I'] = pickle.load(open(os.path.join(dpath, "pimage_%i_I.pickle" %i)))
    data['p'] = pickle.load(open(os.path.join(dpath, "pimage_%i_p.pickle" %i)))
    data['shifts'] = np.load(os.path.join(dpath, "pimage_%i.npy" %i))
    if metrics is True:
        data['metrics'] = pickle.load(open(os.path.join(dpath, "pimage_%i.pickle" %i)))['combined']
    
    return data


def retrieve_hkl(fnames, cs, tilt_angles):
    """
    Retrieve the list of Miller indices at each tilt angle, both in P1 and reduced.
    
    Inputs:
    -------
    fnames: glob-interpretable list of filenames, ordered by tilt angle order
    cs: CCTBX crystal symmetry object
    tilt_angles: list of tilt angles
    
    Outputs:
    --------
    hkl_p1: OrderedDict of tilt_angle: list of all Millers
    hkl_asu: OrderedDict of tilt_angle: list of reduced Millers
    """
    # generate list of reflections in ASU for each tilt image
    hkl_asu, hkl_p1 = OrderedDict(), OrderedDict()
    fnames_ord = natsorted(glob.glob(fnames))
    
    for i,angle in enumerate(tilt_angles):
        hkl_p1[angle] = pickle.load(open(fnames_ord[i])).keys()
        ma = miller.array(miller_set = miller.set(cs,
                                                  flex.miller_index(hkl_p1[angle]),
                                                  anomalous_flag=False),
                              data = flex.double(np.ones(len(hkl_p1[angle]))))
        merged = ma.merge_equivalents().array()
        hkl_asu[angle] = list(merged.indices())
        
    return hkl_p1, hkl_asu


def compute_completeness(cs, hkl_list):
    """
    Compute space group completeness for a given hkl set; note that resolution
    range is likely dictated by the resolution range of hkl_list.
    
    Inputs:
    -------
    cs: CCTBX crystal symmetry object
    hkl_list: list of Miller indices in tuple format
    
    Output:
    -------
    c: fractional space group completeness
    """
    ma_calc = miller.array(miller_set=miller.set(cs,
                                                 flex.miller_index(hkl_list),
                                                 anomalous_flag=False),
                           data=flex.double(np.ones(len(hkl_list))))
    return ma_calc.merge_equivalents().array().completeness()


def merged_statistics(args, refp, merge_p, merge_I, eq_origins, savename=None):
    """
    Compute phase residual to reference and completeness of merged data.
    
    Inputs:
    -------
    args: dict with cell, cs, sg_symbol, and pdb_path keys
    refp: dict of reference hkl: phase values
    merge_p: dict of merged hkl: phase values
    merge_I: dict of merged hkl: intensity values
    eq_origins: np.array of equivalent origins
    savename: if not None, save real space map
    
    Outputs:
    --------
    p_res: mean phase residual to reference
    completeness: fractional space group completeness
    """
    # compute space group completeness
    completeness = compute_completeness(args['cs'], merge_p.keys())
    
    # compute mean phase residual to reference, checking equivalent origins
    p_residuals = np.zeros(eq_origins.shape[0])
    for i,origin in enumerate(eq_origins):
        rc = proc.ReduceCrystals(merge_I.copy(), merge_p.copy(), args['cell'], args['sg_symbol'])
        rc.shift_phases(origin)
        p_asu = rc.reduce_phases(weighted=True)
        p_residuals[i] = np.mean(utils.residual_phases(refp, rc.data['PHIB']))
    p_res = np.min(p_residuals)

    # use best equivalent origin to reduce to generate map on same origin as reference
    rc1 = proc.ReduceCrystals(merge_I.copy(), merge_p.copy(), args['cell'], args['sg_symbol'])
    rc1.shift_phases(eq_origins[np.argmin(p_residuals)])
    p_asu = rc1.reduce_phases(weighted=True)
    rc1.reduce_intensities()

    # compute correlation to map and optionally generate map
    mtz_object = rc1.generate_mtz()
    ma = cctbx_utils.mtz_to_miller_array(mtz_object)
    map_sim = cctbx_utils.compute_map(ma, save_name = savename)
    cc_map = cctbx_utils.compare_maps(map_sim, args['pdb_path'])

    return p_res, completeness, cc_map


def next_angles(hkl_angle, hkl_list, select_angles=None):
    """
    Determine next tilt angle that shares maximum number of reflections with 
    current set, given by hkl_list. 
    
    Inputs:
    -------
    hkl_asu: OrderedDict of angle: list of reflections 
    hkl_list: list of already added reflections
    select_angles: list of keys in hkl_asu to consider, optional
    
    Outputs:
    --------
    next_angles: list of (tilt_angle, n_shared) in decreasing order of n_shared
    """
    if select_angles is None:
        select_angles = np.array(hkl_angle.keys())
    
    n_shared = dict()
    for angle in select_angles:
        n_shared[angle] = len(set(hkl_angle[angle]).intersection(set(hkl_list)))

    sorted_ns = sorted(n_shared.items(), key=operator.itemgetter(1))[::-1]    
    return sorted_ns


def add_tilt_image(args, mi, remaining_angles):
    """
    Add next tilt image to MergeImages object. Remaining images are sorted based on 
    the number of shared reflections, and the first image that has a phase residual
    score below threshold is added.
    
    Inputs:
    -------
    args: dict containing hkl_asu, dpath, threshold, and n_candidates keys
    mi: MergeImage object
    remaining_angles: list of remaining tilt angles that haven't been added
    
    Outputs:
    --------
    mi: updated MergeImage object
    remaining_angles: udpated list of remaining tilt angles
    origin: origin shift used
    t_angle: tilt angle of image added
    """

    ranked_angles = next_angles(args['hkl_asu'], 
                                mi.merge_p.keys(), 
                                select_angles=remaining_angles)
    
    score = args['threshold'] + 5.0
    tested_angles = list()

    while score > args['threshold']:
        for t_angle, n_refl in ranked_angles:

            tested_angles.append(t_angle)
            print "testing angle %i, with %i shared reflections" %(t_angle, n_refl)

            data = load_tilt_data(args['dpath'], t_angle)
            score, origin = mi.score_image(data['p'], data['I'], data['metrics'], args['n_candidates'])

            if score < args['threshold']:
                print "adding data from angle: %i, with score: %.2f" %(t_angle, score)
                mi.add_next_image(data['p'], data['I'], origin)
                remaining_angles.remove(t_angle)
                return mi, remaining_angles, origin, t_angle
            
            if len(tested_angles) == len(remaining_angles):
                return None

            
def add_tilt_image_merge(args, mi, remaining_angles, min_shared=10):
    """
    Add next tilt image to MergeImages object by finding merging origin shift
    rather than using one of the ranked crystallographic phase origins.
    
    Inputs:
    -------
    args: dict containing hkl_p1, dpath, threshold, and n_processes keys
    mi: MergeImage object
    remaining_angles: list of remaining tilt angles that haven't been added
    min_shared: minimum number of shared reflections to attempt merging
    
    Outputs:
    --------
    mi: updated MergeImage object
    remaining_angles: udpated list of remaining tilt angles
    origin: origin shift used
    t_angle: tilt angle of image added
    """

    ranked_angles = next_angles(args['hkl_p1'], 
                                mi.merge_p.keys(), 
                                select_angles=remaining_angles)
    
    score = args['threshold'] + 5.0
    tested_angles = list()

    while score > args['threshold']:
        for t_angle, n_refl in ranked_angles:

            if n_refl < min_shared:
                print "insufficient reflections to attempt merging"
                return None

            tested_angles.append(t_angle)
            print "testing angle %i by merging, with %i shared reflections" %(t_angle, n_refl)

            data = load_tilt_data(args['dpath'], t_angle)
            score, origin = mi.score_image_merge(data['p'], data['I'], n_processes=args['n_processes'])

            if score < args['threshold']:
                print "adding data from angle: %i, with score: %.2f" %(t_angle, score)
                mi.add_next_image(data['p'], data['I'], origin)
                remaining_angles.remove(t_angle)
                return mi, remaining_angles, origin, t_angle
            
            if len(tested_angles) == len(remaining_angles):
                return None


if __name__ == '__main__':
    
    start_time = time.time()

    # extract command line arguments
    args = parse_commandline()
    args['resolution'] = 3.3
    args['grid_spacing'] = 0.5 # grid spacing used for crystallographic origin search
    args['n_candidates'] = 150 # number of origin candidates to consider per tilt image
    args['threshold'] = 4.0 # phase residual threshold for adding a tilt image

    # reference information
    args['sg_symbol'], args['sg_no'], args['cell'], args['cs'] = cctbx_utils.unit_cell_info(args['pdb_path'])
    refI, refp = cctbx_utils.reference_sf(args['pdb_path'], args['resolution'], expand_to_p1=True)
    refp_mod = OrderedDict((key,np.array([val])) for key,val in refp.items())
    eq_origins = np.array([[0,0,0],[0,0,0.5],[0.5,0.5,0],[0.5,0.5,0.5]])

    # additional set-up information
    pnames = os.path.join(args['dpath'], "pimage_*_p.pickle")
    tilt_angles = np.loadtxt(args['tilt_angles'])
    args['hkl_p1'], args['hkl_asu'] = retrieve_hkl(pnames, args['cs'], tilt_angles)
    remaining_angles = list(tilt_angles)

    # information to track during merging
    comp_sg, p_residuals = np.zeros(len(tilt_angles)), np.zeros(len(tilt_angles))
    added_tilts, cc_map = np.zeros(len(tilt_angles)), np.zeros(len(tilt_angles))
    f_shifts = np.zeros((len(tilt_angles),3))

    # set up output folder
    if not os.path.isdir(args['output']):
        os.mkdir(args['output'])

    # set up class and add first image
    mi = MergeImages(grid_spacing=args['grid_spacing'], ref_path=args['pdb_path'])
    data = load_tilt_data(args['dpath'], args['start_angle'])
    f_shifts[0] = mi.add_first_image(data['p'], data['I'], data['metrics'])
    merge_I = OrderedDict((key,val) for key,val in mi.merge_I.items())
    merge_p = OrderedDict((key,val) for key,val in mi.merge_p.items())
    p_residuals[0], comp_sg[0], cc_map[0] = merged_statistics(args, 
                                                              refp, 
                                                              merge_p, 
                                                              merge_I,
                                                              eq_origins,
                                                              savename=os.path.join(args['output'], "map0.ccp4"))
    remaining_angles.remove(args['start_angle'])
    added_tilts[0] = args['start_angle']

    # add remaining images
    first_pass = True
    for counter in range(1,len(tilt_angles)):
        print "on projection image %i" %counter
        vals = add_tilt_image(args, mi, remaining_angles)
        savename = os.path.join(args['output'], "map%i.ccp4" %counter)
        
        # first attempt by using top-ranked crystallographic phase origins
        if vals is not None:
            mi, remaining_angles, f_shifts[counter], added_tilts[counter] = vals
            merge_I = OrderedDict((key,val) for key,val in mi.merge_I.items())
            merge_p = OrderedDict((key,val) for key,val in mi.merge_p.items())
            p_residuals[counter], comp_sg[counter], cc_map[counter] = merged_statistics(args,
                                                                                        refp, 
                                                                                        merge_p, 
                                                                                        merge_I,
                                                                                        eq_origins,
                                                                                        savename=savename)
        # otherwise, try by merging
        elif first_pass is True:
            vals = add_tilt_image_merge(args, mi, remaining_angles)
            if vals is not None:
                mi, remaining_angles, f_shifts[counter], added_tilts[counter] = vals
                merge_I = OrderedDict((key,val) for key,val in mi.merge_I.items())
                merge_p = OrderedDict((key,val) for key,val in mi.merge_p.items())
                p_residuals[counter], comp_sg[counter], cc_map[counter] = merged_statistics(args,
                                                                                            refp, 
                                                                                            merge_p, 
                                                                                            merge_I,
                                                                                            eq_origins,
                                                                                            savename=savename)
            else:
                first_pass = False
                break

    # save to output
    stats = dict()
    stats['p_residuals'], stats['comp_sg'], stats['cc_map'] = p_residuals, comp_sg, cc_map
    stats['f_shifts'], stats['tilt_angles'] = f_shifts, added_tilts
    with open(os.path.join(args['output'], "stats.pickle"), "wb") as handle:
        pickle.dump(stats, handle)
    with open(os.path.join(args['output'], "merge_hklp.pickle"), "wb") as handle:
        pickle.dump(mi.merge_p, handle)

    print "elapsed time is %.2f" %((time.time() - start_time)/60.0)
