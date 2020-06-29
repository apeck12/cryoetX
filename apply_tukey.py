import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse, time
import numpy as np
import scipy.signal
import mrcfile, utils

"""
Multiply input volume by a Tukey window function. This is a necessary pre-processing step before 
centering the density and shifting the center to the origin in order to eliminate phase splitting.
"""

def parse_commandline():
    """
    Parse command line input and return as dictionary.
    """
    parser = argparse.ArgumentParser(description='Multiply input volume by a Tukey-shaped filter.')
    parser.add_argument('-i','--input', help='Input volume in .MRC format', required=True)
    parser.add_argument('-o','--output', help='Output volume in .MRC format', required=True)

    return vars(parser.parse_args())


def tukey_kernel(shape):
    """
    Generate a 3d tukey kernel; multiplication by this kernel will cause edge values to 
    be dampened (and at extreme edges, to have a value of 0).
    """
    import scipy.signal
    
    kx = scipy.signal.tukey(shape[0])
    ky = scipy.signal.tukey(shape[1])
    kz = scipy.signal.tukey(shape[2])

    k3d = kx[:,np.newaxis,np.newaxis] * kx[np.newaxis,:,np.newaxis] * kx[np.newaxis,np.newaxis,:]
    return k3d


def plot_vols(pre_vol, post_vol):
    """
    Plot central slices from the volumes before and after applying the Tukey filter.
    """
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 8))

    ax1.imshow(pre_vol[pre_vol.shape[0]/2,:,:])
    ax2.imshow(pre_vol[:,pre_vol.shape[1]/2,:])
    ax3.imshow(pre_vol[:,:,pre_vol.shape[2]/2])

    ax4.imshow(post_vol[post_vol.shape[0]/2,:,:])
    ax5.imshow(post_vol[:,post_vol.shape[1]/2,:])
    ax6.imshow(post_vol[:,:,post_vol.shape[2]/2])

    f.savefig("applied_tukey.png", dpi=300, bbox_inches='tight')

    return

if __name__ == '__main__':
    
    start_time = time.time()
    args = parse_commandline()

    vol = mrcfile.open(args['input']).data
    k3 = tukey_kernel(vol.shape)
    t_vol = vol * k3
    utils.save_mrc(t_vol, args['output'])
    #plot_vols(vol, t_vol)

    print "elapsed time is %.2f" %((time.time() - start_time) / 60.0) 

