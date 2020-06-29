# --------------------------------------------- # 
#                    Imports                    #
# --------------------------------------------- #

from collections import OrderedDict
import cPickle as pickle
import numpy as np
import os, glob, utils
import scipy.stats

import cctbx_utils, mock_data
import iotbx.pdb

# --------------------------------------------- # 
#               Tests of utils.py               #
# --------------------------------------------- #

def test_wraptopi():
    """
    Confirming that utils.wraptopi yields same results as Matlab's wrapTo180 function,
    except that former wraps to domain [-180,180) rather than [-180,180] as in Matlab.
    See: https://www.mathworks.com/help/map/ref/wrapto180.html for more details.
    """
    x = np.array([-400, -190, -180, -175, 175, 180, 190, 380]).astype(float)
    xw = np.array([-40, 170, -180, -175, 175, 180, -170, 20]).astype(float)
    xw[xw==180] = -180
    yw = utils.wraptopi(x)

    np.testing.assert_array_almost_equal(yw, xw)
    return


def test_average_phases():
    """
    Confirming that mean of circular data is correctly computed by utils.average_phases.
    Comparison is to scipy.stats.circmean function, but this doesn't have the option of 
    computing the weighted mean.
    """
    p_vals = (np.random.rand(10) - 0.5) * 360.0
    p_ref = scipy.stats.circmean(np.deg2rad(p_vals), low=-1*np.pi, high=np.pi)
    p_ref = utils.wraptopi(np.array(np.rad2deg(p_ref)))
    p_est = utils.average_phases(p_vals)

    np.testing.assert_allclose(p_ref, p_est)
    return


def test_std_phases():
    """
    Compare utils.std_phases to the output of scipy.stats.circmean. Note that the results
    diverge when the phases are randomly distributed over the unit circle, but match well
    when the range is not too large. Here it's checked that for phases randomly drawn from
    a domain of 45 degrees, the two functions match to within 1%.
    """
    p_domain = 45.0
    p_vals = (np.random.rand(10) - 0.5) * p_domain + np.random.randint(-180,180)
    p_vals = utils.wraptopi(p_vals)
    p_ref = np.rad2deg(scipy.stats.circstd(np.deg2rad(p_vals)))
    p_est = utils.std_phases(p_vals)

    assert np.abs(p_est - p_ref) / p_ref * 100.0 < 1.0
    return


def test_residual_phases():
    """
    Validating residual_phases function with a different (slightly slower) implementation.
    """
    pdb_path, res = "./reference/pdb_files/4bfh.pdb", 3.0
    refI, refp = cctbx_utils.reference_sf(pdb_path, res, expand_to_p1=True, table='electron')
    refp_err = mock_data.add_phase_errors(refp, 10.0)

    p_shared = utils.shared_dict(refp, refp_err)
    p_vals = np.deg2rad(np.array(p_shared.values()))
    p1, p2 = p_vals[:,0], p_vals[:,1]
    pr_ref = np.rad2deg(np.arccos(np.cos(p1 - p2)))
    pr_est = utils.residual_phases(refp, refp_err)

    np.testing.assert_allclose(pr_ref, pr_est)
    return 


# --------------------------------------------- # 
#              Tests of mock_data.py            #
# --------------------------------------------- #


def test_remove_millers():
    """
    Confirming that remove_millers function yields the desired P1 completeness.
    """
    base_path = "./reference/pdb_files/"
    pdb_ids = ['6d6g', '4bfh', '2id8']
    c_vals = np.random.rand(3)
   
    for pdb,c in zip(pdb_ids, c_vals):
        # generate reference structure factors
        pdb_path = os.path.join(base_path, "%s.pdb" %pdb)
        refI, refp = cctbx_utils.reference_sf(pdb_path, 4.0, expand_to_p1=True, table='electron')
       
        # extract crystal symmetry object
        pdb_input = iotbx.pdb.input(file_name=pdb_path)
        cs = pdb_input.crystal_symmetry()
       
        # test that remove millers yields correct completeness
        refI_sel, refp_sel = mock_data.remove_millers(refI, refp, c)
        ma = cctbx_utils.convert_to_sf(refI_sel.keys(),
                                       np.array(refI_sel.values()),
                                       np.deg2rad(np.array(refp_sel.values())),
                                       cs.cell_equivalent_p1())

        # ascertain within 2 percent of expected completeness
        np.testing.assert_allclose(ma.completeness(), c, atol=0.02)
    
    return


# --------------------------------------------- # 
#           Tests of ProcessCrystals.py         #
# --------------------------------------------- #


def test_find_origin():
   
    import ProcessCrystals as proc

    # set up paths, default values, reference structure factors
    res, n_processes, grid_spacing = 3.0, 8, 1.0
    pdb_path = "./reference/pdb_files/4bfh.pdb"
    sg_symbol, sg_no, cell, cs = cctbx_utils.unit_cell_info(pdb_path)
    refI, refp = cctbx_utils.reference_sf(pdb_path, res, expand_to_p1=True, table='electron')
   
    # confirm phase origin at (0,0,0) for reference structure
    # equivalent origins should be ranked lower due to interpolation errors
    fo = proc.FindOrigin(sg_symbol, cell, cs, refp, refI)
    dmetrics, shifts = fo.scan_candidates(grid_spacing, n_processes)
    np.testing.assert_array_equal(shifts, np.zeros(3))
   
    # confirm that correct origin is identified if phases are shifted
    t_shifts = np.random.random(3)
    p_shifted = fo.shift_phases(t_shifts)
    refp_s = OrderedDict((key,val) for key,val in zip(fo.hkl, p_shifted))
   
    fo_s = proc.FindOrigin(sg_symbol, cell, cs, refp_s, refI)
    data, shifts = fo_s.scan_candidates(grid_spacing, n_processes)
   
    # compare identified shifts to permitted fractional shifts (pf_origins) for P 21 21 21
    pf_origins = np.array([0.0, 0.5, 1.0])
    tol = grid_spacing / np.array(cell)[:3]
    residual = np.abs(1.0 - np.array(shifts) - t_shifts)
    assert all(np.min(np.abs(pf_origins - residual.reshape(3,1)), axis=1) < tol)
   
    return


def test_merge_crystals():
   
    import ProcessCrystals as proc
   
    res, grid_spacing, n_processes = 3.3, 0.5, 8
   
    for pdb_id in ['4bfh', '6d6g']:
        # generate reference information and structure factors
        pdb_path = "./reference/pdb_files/%s.pdb" %pdb_id
        sg_symbol, sg_no, cell, cs = cctbx_utils.unit_cell_info(pdb_path)
        refI, refp = cctbx_utils.reference_sf(pdb_path, res, expand_to_p1=True, table='electron')
       
        # generate randomly-shifted, half-complete (in P1) mock data
        hklI1, hklp1, eshifts1 = mock_data.generate_mock_data(pdb_path, res, completeness=0.5, sigma=0.0)
        hklI2, hklp2, eshifts2 = mock_data.generate_mock_data(pdb_path, res, completeness=0.5, sigma=0.0)

        # merge crystals using MergeCrystals class
        mc = proc.MergeCrystals(space_group=sg_no, grid_spacing=grid_spacing)
        mc.add_crystal(hklI1, hklp1, np.array(cell), n_processes=n_processes, weighted=True)
        mc.add_crystal(hklI2, hklp2, np.array(cell), n_processes=n_processes, weighted=True)
       
        # check that fractional shift for merge matches expected value
        tol = np.max(grid_spacing/np.array(cell)[:3])
        p_origins = np.array([0.0,1.0])
        c_origin = eshifts2 - eshifts1 + mc.fshifts[1]
        assert all(np.min(np.abs(p_origins - c_origin.reshape(3,1)), axis=1) < tol)
       
        # check that intensities of merged data match reference
        Imerge, pmerge = mc.merge_values()
        assert all(np.array([(Imerge[hkl] - refI[hkl])/Imerge[hkl] for hkl in Imerge.keys()])<1e-6)
           
    return


def test_compare_crystals():

    import ProcessCrystals as proc

    # set up paths and default values
    res, n_processes, grid_spacing = 3.0, 8, 1.0
   
    # make sure correct shifts are found for two different PDB files
    for pdb_id in ['4bfh', '6d6g']:
        pdb_path = "./reference/pdb_files/%s.pdb" %pdb_id
        sg_symbol, sg_no, cell, cs = cctbx_utils.unit_cell_info(pdb_path)
   
        # compute reference and randomly-shifted data
        tarI, tarp, eshifts = mock_data.generate_mock_data(pdb_path,
                                                           res,
                                                           completeness=1.0,
                                                           hkl_sel=None,
                                                           sigma=0.0)
        refI, refp = cctbx_utils.reference_sf(pdb_path, res, expand_to_p1=True, table='electron')
        
        # use CompareCrystals class to calculate shifts that relate data
        comp = proc.CompareCrystals(cell, tarp, hklI=tarI)                      
        mgrid, hklp_shifted, fshifts = comp.grid_shift(refp, grid_spacing, n_processes, hklI_ref=refI)

        # ensure computed shifts are below tolerance
        tol = grid_spacing / np.array(cell)[:3]
        assert all(np.abs(1.0 - fshifts - eshifts) < tol)
       
    return


def test_reduce_crystals():

    import ProcessCrystals as proc

    # set up paths and default values
    pdb_path, res = "./reference/pdb_files/4bfh.pdb", 3.0
    sg_symbol, sg_no, cell, cs = cctbx_utils.unit_cell_info(pdb_path)
    refI, refp = cctbx_utils.reference_sf(pdb_path, res, expand_to_p1=True, table='electron')
    refI_mod = OrderedDict((key,np.array([val])) for key,val in refI.items())
    refp_mod = OrderedDict((key,np.array([val])) for key,val in refp.items())

    # check that reduced phases are internally consistent
    rc = proc.ReduceCrystals(refI_mod, refp_mod, cell, sg_symbol)
    p_asu = rc.reduce_phases(weighted=True)
    rc.reduce_intensities()
    eq = np.array([np.allclose(v,v[0]) for v in p_asu.values()])
    assert all([np.allclose(np.around(p_asu.values()[i]) % 180, 0) for i in np.where(eq==False)[0]])
   
    # check that reduced phases and intensities match reference
    hkl_asu = list(cs.build_miller_set(anomalous_flag=False, d_min=res).indices())
    assert np.allclose(utils.wraptopi(np.array([rc.data['PHIB'][hkl] - refp[hkl] for hkl in hkl_asu])), 0)
    assert np.allclose(np.array([rc.data['IMEAN'][hkl] - refI[hkl] for hkl in hkl_asu]), 0)
   
    # check that shifting phases from origin leads to loss of symmetry-expected relationships
    rc.shift_phases(np.random.random(3))
    p_asu = rc.reduce_phases(weighted=True)
    eq = np.array([np.allclose(v,v[0]) for v in p_asu.values()])
    assert not all([np.allclose(np.around(p_asu.values()[i]) % 180, 0) for i in np.where(eq==False)[0]])
   
    return
