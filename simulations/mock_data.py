# --------------------------------------------- # 
#                    Imports                    #
# --------------------------------------------- #

from collections import OrderedDict
import random_eulers, utils
import numpy as np
import cPickle as pickle
import random
import cctbx_utils, iotbx.pdb
from cctbx import miller
from cctbx.array_family import flex

# --------------------------------------------- #
#           From 3d structure factors           #
# --------------------------------------------- #

def remove_millers(hklI, hklp, fraction):
    """
    Remove Millers randomly from input data dictionaries until the specified 
    fraction of original Millers remains. The parameter fraction corresponds
    to the P1 completeness if inputs data is P1-complete.
    
    Inputs:
    -------
    hklI: dict whose keys are Millers and values are intensities
    hklp: dict whose keys are Millers and values are phases, ordered as hklI
    fraction: fraction of Millers to retain, float
    
    Outputs:
    --------
    hklI_sel: subset of input hklI
    hklp_sel: subset of input hklp, ordered as output hklI_sel
    """
    assert hklI.viewkeys() == hklp.viewkeys()
    
    hkl = list(hklI.keys())
    n_hkl = int(np.around(0.5 * fraction * len(hkl))) # factor of 2 accounts for P1 degeneracy
    hkl_sel = random.sample(hkl, n_hkl)
    
    hklI_sel = OrderedDict((key,val) for key,val in hklI.items() if key in hkl_sel)
    hklp_sel = OrderedDict((key,val) for key,val in hklp.items() if key in hkl_sel)
    
    return hklI_sel, hklp_sel


def random_hkl_selection(cs, resolution, completeness):
    """
    Retrieve a list of random Miller indices to achieve the desired completeness
    in P1, assuming no anomalous signal.
    
    Inputs:
    -------
    cs: CCTBX crystal symmetry object
    resolution: high-resolution limit
    completeness: desired P1 completeness
    
    Outputs:
    --------
    hkl_sel: list of Miller indices to retain   
    """
    # get list of P1 reflections, assuming no anomalous signal
    hkl_p1 = list(cs.build_miller_set(anomalous_flag=False,
                                      d_min=resolution).expand_to_p1().indices())

    # retrieve random list of reflections and their friedel mates
    n_est = int(np.around(completeness * len(hkl_p1)))
    hkl_sel = random.sample(hkl_p1, n_est)
    fhkl_sel = [(-1*h[0],-1*h[1],-1*h[2]) for h in hkl_sel]
    hkl_sel += fhkl_sel

    # print information about P1 and SG completeness
    for cs_obj,tag in zip([cs, cs.cell_equivalent_p1()], ['SG', 'P1']):
        ma_calc = miller.array(miller_set=miller.set(cs_obj,
                                                 flex.miller_index(hkl_sel),
                                                 anomalous_flag=False), 
                               data=flex.double(np.ones_like(hkl_sel)))
        print "%s completeness: %.2f" %(tag, ma_calc.merge_equivalents().array().completeness())
        
    return hkl_sel


def retain_millers(hklI, hklp, mlist):
    """
    Retain Millers in mlist from input data.
    
    Inputs:
    -------
    hklI: dict whose keys are Millers and values are intensities
    hklp: dict whose keys are Miller and values are phases, ordered as hklI
    mlist: list of Miller indices (as tuples) to be retained
    
    Outputs:
    --------
    hklI_sel: subset of input hklI
    hklp_sel: subset of input hklp, ordered as output hklI_sel
    """
    assert hklI.viewkeys() == hklp.viewkeys()
    
    hklI_sel = OrderedDict((key,val) for key,val in hklI.items() if key in mlist)
    hklp_sel = OrderedDict((key,val) for key,val in hklp.items() if key in mlist)
    
    return hklI_sel, hklp_sel


def add_random_phase_shift(hklp):
    """
    Introduce a random phase shift, at most one unit cell length along each axis.
    
    Inputs:
    -------
    hklp: dict whose keys are Millers and values are phases in degrees
    
    Outputs:
    --------
    hklp_shifted: dict whose keys are Millers and values are shifted phases
    fshifts: fractional shifts by which origin was translated
    """   
    fshifts = np.array([random.random() for i in range(3)])    
    hkl, p = np.array(hklp.keys()), np.array(hklp.values())
    p_shifted = utils.wraptopi(p - 360 * np.dot(hkl, fshifts).ravel())
    hklp_shifted = OrderedDict((tuple(key),val) for key,val in zip(hkl, p_shifted))
    
    return hklp_shifted, fshifts


def add_phase_errors(hklp, sigma, friedels_same=True):
    """
    Add phase errors drawn from a normal distribution, N(mu, sigma), to the data
    in hklp. Parameters and data should be in degrees. Note that mu is set to 0,
    as this corresponds to a global phase shift.
    
    Inputs:
    -------
    hklp: dict whose keys are Millers and values are phases in degrees
    sigma: standard deviation of error normal distribution
    friedels_same: force phase relationship between Friedel mates, boolean
    
    Outputs:
    --------
    hklp_error: dict whose keys are Millers and values are phases with errors
    """
    # draw errors from a normal distribution and add to phases
    errors = sigma * np.random.randn(len(hklp.keys())) 
    p_errors = utils.wraptopi(np.array(hklp.values()) + errors)
    hklp_error = OrderedDict((key,val) for key,val in zip(hklp.keys(), p_errors))

    # force phase relationship between Friedel mates
    if friedels_same is True:
        for key in hklp_error:
            fkey = (-1*key[0], -1*key[1], -1*key[2])
            if fkey in hklp_error.keys():
                hklp_error[fkey] = -1*hklp_error[key]

    return hklp_error


def generate_mock_data(ref_path, resolution, completeness=1.0, hkl_sel=None, sigma=0.0):
    """
    Generate a modified hklp dictionary in which phases have been shifted from
    the reference origin and phase errors have been introduced. Remove Millers 
    either not listed in hkl_sel or to achieve specified completeness.
    
    Inputs:
    -------
    ref_path: path to reference PDB file
    resolution: high-resolution limit of structure factors
    completeness: fraction of Millers to retain, default is 1.0
    hkl_sel: list of Millers to retain; if None (default), retain all
    sigma: standard deviation for error distribution, default is 0.0
    
    Outputs:
    --------
    hklI: dict whose keys are Millers and values are intensities
    hklp: dict whose keys are Millers and values are phases, ordered as hklI
    shifts: fractional shifts by which phase origin has been translated
    """
    hklI, hklp = cctbx_utils.reference_sf(ref_path, resolution, expand_to_p1=True, table='electron')
    
    if completeness != 1.0:
        hklI, hklp = remove_millers(hklI, hklp, completeness)
    
    if hkl_sel is not None:
        hklI, hklp = retain_millers(hklI, hklp, hkl_sel)
    
    hklp, shifts = add_random_phase_shift(hklp)
    hklp = add_phase_errors(hklp, sigma, friedels_same=True)
    
    return hklI, hklp, shifts


def missing_wedge_mask(angle, shape):
    """
    Generate a volume of the predicted missing wedge region, with a value of zero 
    indicating pixels that belong to the missing wedge. This is oriented to match
    simulated tomographic data.
    
    Inputs:
    -------
    angle: maximum tilt angle
    shape: cubic length of tomogram
    
    Outputs:
    --------
    m_wedge: mask corresponding to missing wedge volume
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
    idx1 = np.where((c[:,0]>=0) & (c[:,2]>=0) & (c[:,2] >= m * c[:,0]))[0]
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

    m_wedge = np.transpose(m_wedge, (1,2,0)).T
    
    return m_wedge


def generate_random_A(pdb_path):
    """
    Generate a crystal setting (A) matrix that is randomly oriented; notation for
    matrices follows DIALS' convention.
    
    Inputs:
    -------
    pdb_path: path to PDB file with SCALES recoreds
    
    Outputs:
    --------
    A: randomly oriented crystal setting matrix
    """
    
    pdb = iotbx.pdb.input(file_name = pdb_path)
    B = np.array(pdb.scale_matrix()[0]).reshape(3,3)
    U = random_eulers.random_rmatrix()
    
    return np.dot(U,B)


def predict_angles(specs_path, pdb_path, resolution, A):
    """
    Predict the tilt angle at which each reflection will be observed. Here a positive
    tilt angle corresponds to images with a +y coordinate. Reflections that lie in the
    missing wedge are excluded.
    
    Inputs:
    -------
    specs_path: dict specifying details of data collection strategy
    pdb_path: path to reference PDB file
    resolution: high-resolution limit of structure factors
    A: crystal setting matrix

    Outputs:
    --------
    hkl_t: dict with keys as Millers and values as tilt angles
    """
    # load information about collection strategy 
    specs = pickle.load(open(specs_path))

    # predict coordinates of all reflections in reciprocal pixels
    sg_symbol, sg_no, cell, cs = cctbx_utils.unit_cell_info(pdb_path)
    hkl = np.array(miller.build_set(crystal_symmetry=cs,
                                    anomalous_flag=True,
                                    d_min=resolution).expand_to_p1().indices())
    qvecs = np.inner(A, np.squeeze(hkl)).T
    xyz = qvecs * 1.0 / specs['mag'] * specs['px_size']

    # predict tilt angle from associated coordinates
    t = np.rad2deg(np.arctan2(xyz[:,1], xyz[:,0]))
    t[(t>90) & (t<=180)] = utils.wraptopi(t[(t>90) & (t<180)] + 180.0) # shift q2 to q4
    t[(t>=-180) & (t<=-90)] += 180.0 # shift q3 to q1
    
    # retain reflections not in missing wedge region
    max_angle = specs['angle'] + 0.5*specs['increment']
    valid_idx = np.where((t > -1*max_angle) & (t < max_angle))[0]
    t, hkl = t[valid_idx], hkl[valid_idx]
    print "Retained %i reflection outside missing wedge" %(len(hkl))

    # generate a dict with keys as Millers and values as tilt angles
    hkl_t = OrderedDict((tuple(key),val) for key,val in zip(hkl,t))    
    return hkl_t


def simulate_damage(specs_path, pdb_path, resolution, tilt_order, bfactor):
    """
    Simulate damage according to the model: I(q,n) = I(q,0)*exp(-n*B*q^2/(16*pi^2)), where 
    n is proportional to the amount of dose received, B is the initial B factor, and I(q,0) 
    is the initial intensity of the reflection under consideration. The B-factor increases
    linearly with dose, which itself increases linearly with the number of tilts collected.
    The simulated structure factors are treated as though they belong to a crystal randomly
    oriented relative to the missing wedge. 
    
    Inputs:
    -------
    pdb_path: path to reference PDB file
    specs_path: path to file indicating details of data collection strategy
    resolution: high-resolution limit of structure factors
    tilt_order: tilt angles in order of image collected, first to last
    bfactor: initial B factor in Angstrom squared at image 0
    
    Outputs:
    --------
    hklI: dict whose keys are Millers and values are 'damaged' intensities
    hklp: dict whose keys are Millers in hklI and values are reference phases
    A: crystal setting matrix, randomly oriented
    """
    # compute reference (undamaged) structure factors
    refI, refp = cctbx_utils.reference_sf(pdb_path, resolution, expand_to_p1=True)
    
    # compute tilt angles for structure factors in a random orientation
    A = generate_random_A(pdb_path)
    hklt = predict_angles(specs_path, pdb_path, resolution, A)
    tilts = np.array(hklt.values())
    
    # convert tilts to n, proportional to dose, by interpolating to nearest tilt
    tilt_order = OrderedDict((v,c) for c,v in enumerate(tilt_order))

    n = np.zeros_like(tilts)
    for i in range(len(tilts)):
        nearest_tilt = min(tilt_order, key=lambda x:abs(x-tilts[i]))
        n[i] = tilt_order[nearest_tilt]
    n += 1 # this way, first image effectively receives some dose
    
    # compute qmags and initial intensity
    qvecs = np.inner(A, np.squeeze(np.array(hklt.keys()))).T
    qmags = np.linalg.norm(qvecs, axis=1)
    I_0 = np.array([refI[key] for key in hklt.keys()])
    
    # compute damaged intensities
    I_d = I_0 * np.exp(-(bfactor/(16*np.square(np.pi))) * n * np.square(qmags))
    hklI = OrderedDict((key,val) for key,val in zip(hklt.keys(), I_d))
    hklp = OrderedDict((key,refp[key]) for key in hklI.keys())
    
    return hklI, hklp, hklt, A


# --------------------------------------------- #
#    Metrics comparing angular distributions    #
# --------------------------------------------- #

def kl_divergence(p, q=np.arange(-60,61,1)):
    """
    Compute KL divergence, assuming that the expected distribution is 
    uniform across sampling frequencies given by q.
    
    Inputs:
    -------
    p: distribution of tilt angles
    q: sampling bins in angular space
    """   
    counts, bins = np.histogram(p, bins=q, normed=True)
    q = 1.0 / len(counts) * np.ones(len(counts))
    kl = np.sum(counts[counts!=0] * np.log(counts[counts!=0] / q[counts!=0]))
    return kl


def kl_divergence_alt(p, q=np.arange(-60,61,1)):
    """
    Compute alternate KL divergence, assuming that the expected distribution 
    is uniform across sampling frequencies given by q. In this case, the abs
    value of the log is taken.
    
    Inputs:
    -------
    p: distribution of tilt angles
    q: sampling bins in angular space
    """   
    counts, bins = np.histogram(p, bins=q, normed=True)
    q = 1.0 / len(counts) * np.ones(len(counts))
    kl = np.sum(counts[counts!=0] * np.abs(np.log(counts[counts!=0] / q[counts!=0])))
    return kl


def chi_squared(p, q=np.arange(-60,61,1)):
    """
    Compute chi-squared value, assuming the expected distribution is 
    uniform across sampling frequencies given by q.
    
    Inputs:
    -------
    p: distribution of tilt angles
    q: sampling bins in angular space
    """   
    obs, bins = np.histogram(p, q, normed=True)
    exp = 1.0 / len(obs) * np.ones(len(obs))
    chi_sq = np.sum(np.square(obs-exp) / np.sqrt(exp))
    return chi_sq
