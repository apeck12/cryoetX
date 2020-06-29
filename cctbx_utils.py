# --------------------------------------------- # 
#                    Imports                    #
# --------------------------------------------- #


from collections import OrderedDict
from cctbx.array_family import flex
from cctbx import miller, maptbx
from iotbx import ccp4_map, mtz
import iotbx.pdb
import numpy as np
import cPickle as pickle


# --------------------------------------------- #
#                  Functions                    #
# --------------------------------------------- #


def unit_cell_info(filename):
    """
    Extract unit cell information from an indexed.json file output by DIALS
    or a reference PDB file.
   
    Input:
    ------
    filename: path to indexed.json or reference pdb
   
    Outputs:
    --------
    sg_symbol: space group symbol in Hermann-Mauguin notation
    sg_no: space group number, integer
    cell: tuple of unit cell (a,b,c,alpha,beta,gamma)
    cs: CCTBX crystal symmetry object
    """
    from dxtbx.model.experiment_list import ExperimentListFactory
   
    ext = filename.split(".")[-1]
    if ext == 'json':
        exp = ExperimentListFactory.from_json_file(filename, check_format=False)
        crystal = exp[0].crystal
        cs = crystal.get_crystal_symmetry()
       
    elif ext == "pdb":
        pdb_input = iotbx.pdb.input(file_name=filename)
        cs = pdb_input.crystal_symmetry()

    cell = cs.unit_cell().parameters()
    sg_info = cs.space_group().info().symbol_and_number()
    sg_symbol, sg_no = sg_info.split("(")[0][:-1], int(sg_info.split(".")[1][1:-1])

    return sg_symbol, sg_no, cell, cs


def reference_sf(ref_path, resolution, expand_to_p1=False, table='electron'):
    """
    Compute structure factors from a coordinates file to the specified resolution.

    Inputs:
    -------
    ref_path: path to coordinates file in PDB format
    resolution: maximum resolution, float
    expand_to_p1: whether to expand structure factors to P1, boolean
    table: scattering factor type, optional. use 'n_gaussian' for X-ray.

    Output:
    -------
    refI: dictionary whose keys are Millers and values are intensities
    refp: dictionary whose keys are Millers and values are phases in degrees
    """
    pdb_input = iotbx.pdb.input(file_name=ref_path)
    xrs = pdb_input.xray_structure_simple(crystal_symmetry=pdb_input.crystal_symmetry_from_cryst1())
    xrs.scattering_type_registry(table=table)
    sf = xrs.structure_factors(d_min=resolution, anomalous_flag=True).f_calc()
    if expand_to_p1 is True:
        sf = sf.expand_to_p1()

    refp = OrderedDict((key,np.rad2deg(val)) for key,val in sf.phases())
    refI = OrderedDict((key,val) for key,val in sf.intensities())

    return refI, refp


def mtz_to_miller_array(mtz_object):
    """
    Recombine intensities and phases (under 'IMEAN' and 'PHIB' labels, respectively)
    from input MTZ file into a Miller array object. Probably there's a CCTBX utility
    to accomplish this, but I haven't been able to locate it.
    
    Inputs:
    -------
    mtz_object: a CCTBX MTZ object (from mtz.object(file_name=filename.mtz))
    
    Outputs:
    --------
    ma: a CCTBX Miller array object of complex structure factors
    """
    
    # extract Miller arrays and crystal symmetry
    I_ma = mtz_object.as_miller_arrays_dict()[('crystal', 'dataset', 'IMEAN')]
    p_ma = mtz_object.as_miller_arrays_dict()[('crystal', 'dataset', 'PHIB')]
    assert list(I_ma.indices()) == list(p_ma.indices())
    cs = mtz_object.crystals()[1].crystal_symmetry()
    
    # compute complex structure factors
    I, p = np.array(I_ma.data()), np.deg2rad(np.array(p_ma.data()))
    A, B = np.sqrt(I) * np.cos(p), np.sqrt(I) * np.sin(p)
    indices = I_ma.indices()
    sf_data = flex.complex_double(flex.double(A), flex.double(B))

    # convert complex structure factors to CCTBX-style Miller array
    ma = miller.array(miller_set = miller.set(cs, indices, anomalous_flag=False), data = sf_data)
    
    return ma


def ccp4_to_npy(map_name, save_name = None):
    """
    Convert a ccp4 map to numpy format and optionally save to given path.
    
    Inputs:
    -------
    map_name: path to .CCP4 map file
    save_name: path to output numpy file, optional

    Output:
    -------
    f_as_npy: converted CCP4 map volume as numpy array
    """
    f = ccp4_map.map_reader(file_name = map_name)
    shape = f.data.focus()
    f_as_npy = np.array(f.data).reshape(shape)

    if save_name is not None:
        np.save(save_name, f_as_npy)

    return f_as_npy


def convert_to_sf(hkl, intensities, phases, cs):
    """
    Reformat intensities and phases into a CCTBX-style Miller array, with space group
    symmetry enforced. 

    Inputs:
    -------
    hkl: list of Miller indices, formatted as tuples
    intensities: array of intensities, with ordering matching hkl
    phases: array of phases with ordering matching hkl
    cs: CCTBX crystal.symmetry object

    Outputs:
    --------
    ma: CCTBX-style Miller array
    """
    # compute structure factors in complex number format
    if (phases.min() < -1*np.pi) or (phases.max() > np.pi):
        print "Error: Invalid phase values; may be in degrees rather than radians."
    A, B = np.sqrt(intensities)*np.cos(phases), np.sqrt(intensities)*np.sin(phases)
    B[np.abs(B)<1e-12] = 0

    # reformat miller indices and structure factor information into CCTBX format
    indices = flex.miller_index(hkl)
    sf_data = flex.complex_double(flex.double(A),flex.double(B))
    ma = miller.array(miller_set=miller.set(cs, indices, anomalous_flag=False), data=sf_data)

    return ma


def compute_map(ma, save_name = None, grid_step = 0.3):
    """
    Compute CCP4 map from a CCTBX-style Miller array and save if an output path
    is given. Default grid step is 0.5 Angstroms.
    
    Inputs:
    -------
    ma: CCTBX Miller array
    save_name: filepath at which to save CCP4 map, optional
    grid_step: grid step for computing real space map in Angstrom, default 0.3
    
    Output:
    -------
    fft_map: maptbx real space map
    """
    fft_map = ma.fft_map(grid_step = grid_step,
                         symmetry_flags = maptbx.use_space_group_symmetry)
    fft_map = fft_map.apply_volume_scaling()

    if save_name is not None:
        fft_map.as_ccp4_map(save_name)

    return fft_map


def compare_maps(map_sim, pdb_path, table='electron'):
    """
    Compute correlation coefficient between a simulated map and the reference map 
    generated from the input PDB file. Map resolution / gridding is determined by 
    the grid spacing of the simulated map.
   
    Inputs:
    -------
    map_sim: simulated map generated by maptbx
    pdb_path: path to reference PDB file
    table: scattering factor type, optional; use 'n_gaussian' for X-ray.
   
    Outputs:
    --------
    map_cc: correlation coefficient between simulated and reference maps
    """
    # compute structure factors for the reference file
    pdb_input = iotbx.pdb.input(file_name=pdb_path)
    xrs = pdb_input.xray_structure_simple(crystal_symmetry=
                                          pdb_input.crystal_symmetry_from_cryst1())
    xrs.scattering_type_registry(table=table)
    f_calc = xrs.structure_factors(d_min=2.0/map_sim.d_min()).f_calc()

    # compute reference real space map from structure factors
    crystal_gridding = maptbx.crystal_gridding(unit_cell = xrs.unit_cell(),
                                               space_group_info = xrs.space_group_info(),
                                               pre_determined_n_real = map_sim.n_real(),
                                               symmetry_flags = maptbx.use_space_group_symmetry)
    map_ref = f_calc.fft_map(crystal_gridding = crystal_gridding,
                             symmetry_flags = maptbx.use_space_group_symmetry)
   
    # extract density values and ensure that map means are zero
    m1 = map_sim.real_map_unpadded().as_numpy_array().flatten()
    m2 = map_ref.real_map_unpadded().as_numpy_array().flatten()
    assert np.allclose(m1.mean(), 0)
    assert np.allclose(m2.mean(), 0)    

    return np.corrcoef(m1,m2)[0,1]
