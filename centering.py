import argparse, time
from EMAN2 import *
import numpy as np

"""
Center the global phase origin of input volume using the EMAN2 wrapper for the SPARX 
autoconvolution function, and then shift the volume's center to the corner (as done
by fftshift). Requires an EMAN2 python distribution. Applying centering plus an origin 
shift eliminates the phenomenon of phase splitting at Bragg peak positions by removing
the circular discontinuity at the origin of the FFT calculation.
"""

def parse_commandline():
    """
    Parse command line input and return as dictionary.
    """
    parser = argparse.ArgumentParser(description='Use EMAN2 functions to center global origin of input volume.')
    parser.add_argument('-i','--input_volume', help='Input volume in .MRC format', required=True)
    parser.add_argument('-o','--output_volume', help='Output volume in .MRC format', required=True)

    return vars(parser.parse_args())

if __name__ == "__main__":
    
    start_time = time.time()
    args = parse_commandline()

    # center data using EMAN2 centering functions
    e = EMData()
    e.read_image(args['input_volume'])
    e.process_inplace("xform.centeracf")
    e.process_inplace("xform.phaseorigin.tocorner")
    e.write_image(args['output_volume'])

    print "elapsed time is %.4f" %((time.time() - start_time) / 60.0)
