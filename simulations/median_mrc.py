import argparse, mrcfile
import numpy as np

"""
Compute the median value of an input MRC file and save to a text file.
"""

def parse_commandline():
    """
    Parse command line input and return as dictionary.
    """
    parser = argparse.ArgumentParser(description='Compute median value of input MRC file.')
    parser.add_argument('-i','--input', help='Input MRC file', required=True)
    parser.add_argument('-o','--output', help='Output file containing median value', required=True)

    return vars(parser.parse_args())


if __name__ == '__main__':
    
    args = parse_commandline()
    
    volume = mrcfile.open(args['input']).data
    median = np.median(volume)
    
    with open(args['output'], "w") as f:
        f.write("%.5f \n" %median)
    f.close()
