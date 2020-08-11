from scipy.ndimage.filters import gaussian_filter
import argparse, time, utils, os, mrcfile
import numpy as np

"""
Simulate the effects of radiation damage in real space. Each damage event is modeled by replacing
a randomly-chosen subregion in the real-space volume by its Gaussian-blurred copy. After applying
a user-specified number of this, the damaged volume is saved in MRC format.
"""


def parse_commandline():
    """
    Parse command line input and return as dictionary.
    """
    parser = argparse.ArgumentParser(description='Simulate effects of radiation damage to input volume.')
    parser.add_argument('-i','--input', help='Input MRC file', required=True)
    parser.add_argument('-o','--output', help='Output MRC file', required=True)
    parser.add_argument('-n','--n_hits', help='Number of hits or damage events to apply', required=True, type=int)
    parser.add_argument('-s','--sigma', help='Sigma for Gaussian blur used to simulate a hit', required=True, type=float)
    parser.add_argument('-l','--length', help='Length of cubic box of hit region', required=True, type=int)
    parser.add_argument('-c','--centers',help='Record centers of hits in given file',required=False)

    return vars(parser.parse_args())


def apply_hit(volume, sigma, length, record=None):
    """
    Apply a 'hit' to the input volume by Gaussian-blurring a random cubic region.

    Inputs:
    -------
    volume: 3d array to be radiation-damaged
    sigma: parameter describing decay rate of Gaussian blur
    length: dimensions of cubic subregion to blur.

    Output:
    -------
    volume: input volume after simulating exposure to radiation
    """
    shape = volume.shape

    # retrieve a random, valid location in input volume
    xi,yi,zi = np.random.randint(shape[0]), np.random.randint(shape[1]), np.random.randint(shape[2])
    while (xi<int(length/2)) or (xi>=shape[0]-(int(length/2))) or (yi<int(length/2)) \
            or (yi>=shape[1]-(int(length/2))) or (zi<int(length/2)) or (zi>=shape[2]-(int(length/2))):
        xi,yi,zi = np.random.randint(shape[0]), np.random.randint(shape[1]), np.random.randint(shape[2])

    # optionally record the location of damage
    if record is not None:
        with open(record, 'a') as hits_file:
            hits_file.write('%i %i %i\n' %(xi,yi,zi))

    # replace the surrounding area with a Gaussian-filtered version
    region = volume[xi-int(length/2):xi+int(length/2)+1, 
                    yi-int(length/2):yi+int(length/2)+1, 
                    zi-int(length/2):zi+int(length/2)+1]
    region_g = gaussian_filter(region, sigma=sigma, order=0, mode='reflect')
    volume[xi-int(length/2):xi+int(length/2)+1,
           yi-int(length/2):yi+int(length/2)+1,
           zi-int(length/2):zi+int(length/2)+1] = region_g

    return volume


if __name__ == '__main__':
    
    start_time = time.time()
    args = parse_commandline()

    vol = mrcfile.open(args['input']).data.copy()
    for i in range(args['n_hits']):
        vol = apply_hit(vol, args['sigma'], args['length'], record=args['centers'])

    if args['input'] == args['output']:
        print "Overwriting %s" %(args['input'])
        os.remove(args['input'])
    utils.save_mrc(vol, args['output'])

    print "elapsed time is %.2f" %((time.time() - start_time) / 60.0) 
