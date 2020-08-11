from dxtbx.model.experiment_list import ExperimentListFactory
from cctbx import miller
import cPickle as pickle
import numpy as np
import sys, time, argparse, os

"""
Extract indexing matrices and reflection information from DIALS indexing output so that 
contents are accessible without libtbx.python; also, predict all observed positions to 
the specified resolution. Contents are saved to matrix_info.pickle, indexed_info.pickle
and predicted_info.pickle in given path.
"""

def parse_commandline():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser(description='Convert DIALS indexing output to non-libtbx accessible format.')
    parser.add_argument('-p','--path', help='Directory with output of DIALS indexing', required=True)
    parser.add_argument('-i','--info', help='Dictionary that specifies mag, res, pixel size, and shape', required=True)
    return vars(parser.parse_args())


def extract_matrices(crystal, savepath):
    """
    Extract orientation matrices and save to pickle file.
    """
    matrices = dict()
    matrices['A'] = np.array(crystal.get_A()).reshape(3,3)
    matrices['U'] = np.array(crystal.get_U()).reshape(3,3)
    matrices['B'] = np.array(crystal.get_B()).reshape(3,3)

    with open(os.path.join(savepath, "matrix_info.pickle"), "wb") as handle:
        pickle.dump(matrices, handle)

    return matrices


def find_duplicates(millers, qvecs, A_matrix):
    """
    Find duplicate hkl entries and record the entry (noted by value of 1) whose associated 
    qvector is more distant from the calculated qvector.
    """
    from collections import Counter

    # convert from np.array format to list of tuples
    millers_as_tuple = list()
    for hkl in millers:
        millers_as_tuple.append(tuple(hkl))

    # identify duplicate hkl
    counts = Counter(millers_as_tuple)
    suspect_idx = np.where(np.array(counts.values())!=1)[0]
    
    # track down indices of duplicate hkl that don't match calculated qvector
    discard_idx = list()
    for idx in suspect_idx:
        h,k,l = counts.keys()[idx]
        print "Duplicate found for Miller (%i,%i,%i)" %(h,k,l)
        mult_idx = np.where((millers[:,0]==h) & (millers[:,1]==k) & (millers[:,2]==l))[0]
        q_calc = np.inner(A_matrix, np.array([h,k,l])).T
        q_recorded = qvecs[mult_idx]
        delta = np.sum(np.abs(q_calc - q_recorded), axis=1)
        discard_idx.append(mult_idx[np.where(delta!=delta.min())[0]][0])

    # convert to np.array of 1s (valid) and 0 (for discard)
    discard_idx = np.array(discard_idx)
    discard = np.zeros(millers.shape[0])
    if len(discard_idx) > 0:
        discard[discard_idx]=1

    return discard


def extract_rlp_info(indexed, crystal, A_matrix, savepath):
    """
    Extract Miller indices, resolution, and positional information of indexed reflections.
    """
    idx_info = dict()
    hkl = indexed.select(indexed['miller_index']!=(0,0,0))['miller_index']
    idx_info['res'] = np.array(crystal.get_unit_cell().d(hkl))
    idx_info['hkl'] = np.array(hkl)

    # extract info from indexed object and remove unindexed Miller indices
    for key,tag in zip(['I', 'sigI', 'xyz', 'qvec'], ['intensity.sum.value', 'intensity.sum.variance', 'xyzobs.px.value', 'rlp']):
        idx_info[key] = np.array(indexed.select(indexed['miller_index']!=(0,0,0))[tag])

    # remove duplicate entries from each data type if any exist
    discard = find_duplicates(idx_info['hkl'], idx_info['qvec'], A_matrix)
    if np.sum(discard) > 0:
        for key in idx_info.keys():
            idx_info[key] = idx_info[key][discard==0]

    # dump to pickle file
    with open(os.path.join(savepath, "indexed_info.pickle"), "wb") as handle:
        pickle.dump(idx_info, handle)

    return idx_info


def missing_wedge_mask(angle, shape):
    """
    Generate a volume of the predicted missing wedge region based on the tomogram's
    shape and maximum tilt angle. A value of zero corresponds to pixels that belong
    to the missing wedge. 
    """
    # determine the slope of missing wedge plane
    rise, run = shape[2]/2 * np.tan(np.deg2rad(angle)), shape[2]/2    
    if rise > shape[2]/2:
        segment = np.tan(np.deg2rad(90 - angle)) * (rise - shape[2]/2)
        run = shape[2]/2 - segment
        rise = shape[2]/2
    m = float(rise) / float(run)
    
    # generate octant mask -- 1/8 of total volume for efficiency
    xc, yc, zc = int(shape[0]/2), int(shape[1]/2), int(shape[2]/2)
    c = np.vstack((np.meshgrid(range(xc), range(yc), range(zc)))).reshape(3,-1).T
    idx1 = np.where((c[:,0]>=0) & (c[:,2]>=0) & (c[:,2] > m * c[:,0]))[0]
    octant = np.ones((xc,yc,zc)).flatten()
    octant[idx1] = 0
    octant = octant.reshape((xc,yc,zc))
    
    # generate full volume from octant
    m_wedge = np.ones(shape)
    m_wedge[xc:,yc:,zc:] = octant
    m_wedge[0:xc,yc:,zc:] = octant
    m_wedge[xc:,0:yc,zc:] = np.fliplr(octant)
    m_wedge[0:xc,0:yc:,zc:] = np.fliplr(octant)

    m_wedge[xc:,yc:,0:zc] = np.flip(m_wedge[xc:,yc:,zc:], 2)
    m_wedge[xc:,0:yc,0:zc] = np.flip(m_wedge[xc:,0:yc,zc:], 2)
    m_wedge[0:xc:,yc:,0:zc] = np.flip(m_wedge[0:xc:,yc:,zc:], 2)
    m_wedge[0:xc:,0:yc:,0:zc] = np.flip(m_wedge[0:xc:,0:yc:,zc:], 2)

    m_wedge = np.transpose(m_wedge, (1,0,2)).T # for experimental tomograms
    #m_wedge = np.transpose(m_wedge, (1,2,0)).T # for simulated tomograms
    
    return m_wedge


def predict_positions(A, crystal, specs, savepath):
    """
    Predict the locations of Bragg peaks out to the specified resolution, and remove
    any predicted to fall inside the missing wedge region. Output positions and their
    associated Miller indices as separate keys in a dictionary.
    """
    # generate complete list of Miller indices to given resolution
    ms = miller.build_set(crystal_symmetry = crystal.get_crystal_symmetry(),
                          anomalous_flag=True,
                          d_min = specs['res']).expand_to_p1()
    hkl = np.array(ms.indices())

    # predict the xyz positions of each peak in the FFT of the tomogram
    qvecs = np.inner(A, np.squeeze(hkl)).T
    px_coords = qvecs * 1.0 / specs['mag'] * specs['px_size'] + np.array(specs['shape']) / 2.0
    print "Predicted %i reflections to %.1f resolution" %(len(hkl), specs['res'])
    
    # remove any Millers located inside the missing wedge
    mwedge = missing_wedge_mask(specs['angle'], specs['shape'])
    sel = np.fliplr(px_coords.copy()) 
    sel = np.around(sel).astype(int) 
    valid_idx = mwedge.flatten()[np.ravel_multi_index(sel.T, specs['shape'])] 
    hkl_valid, px_valid = hkl[valid_idx==1], px_coords[valid_idx==1]
    print "Retained %i reflection outside missing wedge" %(len(hkl_valid))

    # store in dictionary, save as pickle, and return
    predicted = dict()
    predicted['hkl'], predicted['xyz'] = hkl_valid, px_valid
    
    with open(os.path.join(savepath, "predicted_info.pickle"), "wb") as handle:
        pickle.dump(predicted, handle)

    return predicted
    

if __name__ == '__main__':

    start_time = time.time()
    args = parse_commandline()

    # loading command line input
    indexed = pickle.load(open(os.path.join(args['path'], "indexed.pickle")))
    exp = ExperimentListFactory.from_json_file(os.path.join(args['path'], "indexed.json"), check_format=False)
    crystal = exp[0].crystal

    # extracting and saving information
    matrices = extract_matrices(crystal, args['path'])
    rlp_info = extract_rlp_info(indexed, crystal, matrices['A'], args['path'])
    specs = pickle.load(open(args['info']))
    predicted = predict_positions(matrices['A'], crystal, specs, args['path'])

    print "elapsed time is %.2f minutes" %((time.time() - start_time)/60.0)
