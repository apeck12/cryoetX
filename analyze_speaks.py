from collections import OrderedDict
import time, argparse, os, mrcfile
import cPickle as pickle
import numpy as np
import ProcessCrystals as proc

"""
Fit Bragg peaks, estimating both intensity and phase values, from simulated tomograms.
"""

def parse_commandline():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser(description='Fit Bragg peak profiles, estimating intensity and phase values.')
    parser.add_argument('-t','--tomogram', help='Tomogram / real space volume in MRC format', required=True)
    parser.add_argument('-i','--hkl_xyz', help='Dict containing hkl and xyz information, e.g. indexed.pickle', required=True)
    parser.add_argument('-s','--savepath', help='Path at which to save estimated intensities and phases', required=True)
    parser.add_argument('-p','--params', help='Dict containing parameters for profile fitting', required = False)

    return vars(parser.parse_args())


def setup_params():
    """
    Set up parameters for profile fitting: thresholds for retaining pixels, etc.
    """
    params = dict()
    params['length'] = 15 # length of subvolumes to extract
    params['i_sigma'] = 4 # sigma threshold for retaining pixel 
    params['b_sigma'] = 2 # background sigma threshold
    params['p_std'] = 15 # std dev threshold for peak phases
    params['p_mode'] = 'weighted_mean' # method for estimating phases
    params['n_pixels'] = 3 # min number of pixels for peak to be retained
    params['c_threshold'] = 2.0 # max dist in pixels between obs and pred peak center

    return params


if __name__ == '__main__':
    
    start_time = time.time()
    args = parse_commandline()

    # load input files and set up dictionary of profile-fitting parameters
    tomogram = mrcfile.open(args['tomogram']).data
    hkl_info = pickle.load(open(args['hkl_xyz']))
    hkl_xyz = OrderedDict((tuple(key),val) for key,val in zip(hkl_info['hkl'], hkl_info['xyz']))
    if args['params'] is None:
        params = setup_params()
    else:
        params = pickle.load(open(args['params']))

    # set up instance of ProfileFitting class
    pf = proc.PeakFitting(tomogram, hkl_info, params['length'])
    imasks = pf.compute_imasks(params['i_sigma'])
    bmasks = pf.compute_imasks(params['b_sigma'], coincident_masks=imasks)
    bgd_vols = pf.estimate_background(bmasks)
    pmasks = pf.compute_pmasks(imasks, params['p_std'])
    pf.npixels_threshold(pmasks, params['n_pixels'])

    i_est, i_std = pf.estimate_intensities(pmasks, bgd_ests=bgd_vols)
    p_est, p_std = pf.estimate_phases(pmasks, mode=params['p_mode'])

    # compute residual between predicted and observed peak center
    r_coord = OrderedDict()
    for key in pf.xyz_obs.keys():
        r_coord[key] = np.sqrt(np.sum(np.square(pf.xyz_obs[key] - hkl_xyz[key])))
    n_reflections = len(p_est.keys())

    # remove any peaks whose observed center is too distant from predicted center
    p_est = OrderedDict((key,val) for key,val in p_est.items() if r_coord[key]<params['c_threshold'])
    p_std = OrderedDict((key,val) for key,val in p_std.items() if r_coord[key]<params['c_threshold'])
    i_std = OrderedDict((key,val) for key,val in i_std.items() if r_coord[key]<params['c_threshold'])
    i_est = OrderedDict((key,val) for key,val in i_est.items() if r_coord[key]<params['c_threshold'])
    assert i_est.viewkeys() == p_est.viewkeys()
    print "%i reflections removed due to peak coordinates residual" %(n_reflections - len(p_est.keys()))

    # eliminate any Millers associated with negative intensities
    p_est = OrderedDict((key,val) for key,val in p_est.items() if i_est[key]>0)
    p_std = OrderedDict((key,val) for key,val in p_std.items() if i_est[key]>0)
    i_std = OrderedDict((key,val) for key,val in i_std.items() if i_est[key]>0)
    i_est = OrderedDict((key,val) for key,val in i_est.items() if i_est[key]>0)
    assert i_est.viewkeys() == p_est.viewkeys()

    # save information to savepath
    if not os.path.isdir(args['savepath']):
        os.mkdir(args['savepath'])

    for data,fname in zip([i_est, i_std, p_est, p_std], ['estI', 'stdI', 'estp', 'stdp']):
        with open(os.path.join(args['savepath'], "%s.pickle" %fname), "wb") as handle:
            pickle.dump(data, handle)

    with open(os.path.join(args['savepath'], "peak_centers.pickle"), "wb") as handle:
        pickle.dump(pf.xyz_obs, handle)

    print "elapsed time is %.2f" % ((time.time() - start_time) / 60.0)
