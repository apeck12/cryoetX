import argparse, time, utils
import cPickle as pickle
import numpy as np
from cctbx import miller, maptbx
import iotbx.pdb

"""
Simulate density map from a coordinates file using CCTBX and electron scattering factors.
"""

def parse_commandline():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Simulate electron density from PDB.")
    parser.add_argument('-i', '--input', help='input coordinates file in PDB format')
    parser.add_argument('-o', '--output', help='output density in MRC format')
    parser.add_argument('-gs', '--grid_step', help='grid step for computing map', type=float)
    parser.add_argument('-r', '--resolution', help='minimum resolution of map', type=float)
    parser.add_argument('-s', '--shape', help='length of cubic volume to output', type=int)
    return vars(parser.parse_args())


def add_buffer(shape, fft_map):
    """
    Embed/center the computed map in a larger cubic volume of length shape.
    """
    # compute how much buffer region to add around fft_map to achieve desired shape
    hbuffer = shape - np.array(fft_map.shape)
    hbuffer = hbuffer / 2
    hbuffer = hbuffer.astype(int)
    
    # set the buffer value to the map's median, approximately the solvent value
    volume = np.ones((shape,shape,shape)) * np.median(fft_map)
    volume[hbuffer[0]: hbuffer[0]+fft_map.shape[0],
           hbuffer[1]: hbuffer[1]+fft_map.shape[1],
           hbuffer[2]: hbuffer[2]+fft_map.shape[2]] = fft_map

    return volume


def remove_blanks(fft_map):
    """
    Remove any blank slices along the borders of the input volume.
    """
    xdel, ydel, zdel = list(), list(), list()
    for i in range(fft_map.shape[0]):
        if np.sum(np.abs(fft_map[i,:,:]))==0: 
            xdel.append(i)
    for i in range(fft_map.shape[1]):
        if np.sum(np.abs(fft_map[:,i,:]))==0: 
            ydel.append(i)
    for i in range(fft_map.shape[2]):
        if np.sum(np.abs(fft_map[:,:,i]))==0: 
            zdel.append(i)

    if len(xdel)!=0: fft_map = np.delete(fft_map, np.array(xdel), axis=0)
    if len(ydel)!=0: fft_map = np.delete(fft_map, np.array(ydel), axis=1)
    if len(zdel)!=0: fft_map = np.delete(fft_map, np.array(zdel), axis=2)

    return fft_map


def compute_map(args):
    """
    Compute a CCP4-style map using CCTBX to the specified resolution and grid spacing.
    """
    # compute structure factors to desired resolution
    pdb_cryst = iotbx.pdb.input(file_name = args['input'])
    xrs = pdb_cryst.xray_structure_simple(crystal_symmetry = pdb_cryst.crystal_symmetry_from_cryst1())
    xrs.scattering_type_registry(table='electron')
    f_calc = xrs.structure_factors(d_min = args['resolution']).f_calc()

    # compute FFT of structure factors for real space map
    fft_map = f_calc.fft_map(grid_step = args['grid_step'],
                             symmetry_flags = maptbx.use_space_group_symmetry)
    fft_map = fft_map.apply_sigma_scaling()
    fft_map = fft_map.real_map().as_numpy_array()

    fft_map = remove_blanks(fft_map)
    return fft_map


if __name__ == "__main__":
    
    start_time = time.time()
    args = parse_commandline()

    fft_map = compute_map(args)
    fft_map = add_buffer(args['shape'], fft_map)
    utils.save_mrc(fft_map, args['output'])

    print "elapsed time is %.2f minutes" %((time.time() - start_time)/60.0)
