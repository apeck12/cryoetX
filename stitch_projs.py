from natsort import natsorted
import argparse, time, glob
import utils, mrcfile, os
import numpy as np


def parse_commandline():
    """
    Parse command line input and return as dictionary.
    """
    parser = argparse.ArgumentParser(description='Stitch projections as separate MRC files into a tilt-stack.')
    parser.add_argument('-i','--input', help='Input projection names using glob syntax', required=True)
    parser.add_argument('-o','--output', help='Output tilt-stack in MRC format', required=True)

    return vars(parser.parse_args())


if __name__ == '__main__':
    
    start_time = time.time()
    args = parse_commandline()

    fnames = natsorted(glob.glob(args['input']))
    shape = mrcfile.open(fnames[0]).data.shape

    # add each projection into tilt-stack array
    stack = np.zeros((len(fnames), shape[-1], shape[-2]))
    for i in range(len(fnames)):
        if len(shape) == 3:
            stack[i] = mrcfile.open(fnames[i]).data[0]
        else:
            stack[i] = mrcfile.open(fnames[i]).data

    # save tilt-stack and remove individuals projections
    utils.save_mrc(stack, args['output'])
    map(os.remove, glob.glob(args['input']))

    print "elapsed time is %.2f" %((time.time() - start_time) / 60.0) 
