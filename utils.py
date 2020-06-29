# --------------------------------------------- # 
#                    Imports                    #
# --------------------------------------------- #


from collections import OrderedDict
import pyfftw, mrcfile, itertools
import numpy as np


# --------------------------------------------- #
#           Fourier transform-related           #
# --------------------------------------------- #


def compute_ft(image):
    """ 
    Compute complex, centered DFT of an n-d numpy array using pyFFTW.

    Inputs:
    -------
    image: array of any dimensions

    Outputs:
    --------
    ft_image: array of (complex) structure factors
    """
    ft_image = pyfftw.empty_aligned(image.shape)
    ft_image[:] = image
    f = pyfftw.interfaces.scipy_fftpack.fftn(ft_image)
    return np.fft.fftshift(f)


def ft_to_I_phase(ft):
    """
    Convert structure factors to separate numpy arrays of intensity and phase.

    Inputs:
    -------
    ft: array of complex structure factors

    Outputs:
    --------
    I: array of intensities
    phase: array of phases in radians
    """
    I = np.square(np.abs(ft))
    phase = np.arctan2(ft.imag, ft.real)
    return I, phase


# --------------------------------------------- #
#             Visualization helpers             #
# --------------------------------------------- #


def followup_brightness_scale(data, brightness=0.5, hsize=100):
    """
    Numpy version of Brewster/Zwart code to rescale intensities for visualization.
    """
    qave = np.mean(data)
    histogram = np.zeros(hsize)

    for i in range(data.size):
        temp = int((hsize/2)*data.flatten()[i]/qave)
        if temp < 0: histogram[0]+=1
        elif temp >= hsize: histogram[hsize-1]+=1
        else: histogram[temp]+=1

    percentile, accum = 0, 0
    for i in range(hsize):
        accum+=histogram[i]
        if (accum > 0.9*data.size):
            percentile=i*qave/(hsize/2)
            break

    adjlevel = 0.4 
    if percentile > 0.:
        correction = brightness * adjlevel/percentile
    else:
        correction = brightness / 5.0

    outscale = 256
    corrected = data*correction
    outvalue = outscale * (1.0 - corrected)
    outvalue[outvalue < 0] = 0
    outvalue[outvalue >= outscale] = outscale - 1
    return outvalue


def hsv_phase_plot(I_sel, p_sel):
    """
    Convert intensities and phases to an array that can be plotted by matplotlib's 
    imshow, in which hue / saturation / color is used to visualize intensities and
    phases simultaneously. Inputs are intensities (I_sel) and phases (p_sel) arrays.
    """
    from matplotlib import colors

    I_sel = followup_brightness_scale(I_sel)
    p_sel = p_sel.astype(float)
    
    i_plot = (256.0 - I_sel) / 256.0
    p_plot = (p_sel + 180.0) / 360.0
    ones = np.ones(p_plot.shape)
    c = colors.hsv_to_rgb(np.array(zip(p_plot.flatten(), 
                                       i_plot.flatten(), 
                                       ones.flatten())))

    return c.reshape(p_plot.shape[0],p_plot.shape[1],3)


# --------------------------------------------- # 
#       Handling data in reciprocal space       #
# --------------------------------------------- #


def wraptopi(p_vals):
    """
    Wrap array of input phases (in degrees) to lie in domain [-180, 180).

    Inputs:
    -------
    p_vals: array of phase values in degrees, dtype float

    Outputs:
    --------
    p_vals: updated array of wrapped phase values in domain [-180, 180)
    """
    n1 = (-180.0 - p_vals) / 360.0
    n2 = (180.0 - p_vals) / -360.0
    
    dn1 = (n1 + 1).astype(int) * 360.0
    dn2 = (n2 + 1).astype(int) * 360.0
    
    p_vals[p_vals < -180.0] += dn1[p_vals < -180.0]
    p_vals[p_vals > 180.0] -= dn2[p_vals > 180.0]
    p_vals[p_vals == 180.0] = -180.0

    return p_vals


def average_phases(p_vals, weights=None):
    """
    Average phases using a method that works for circular data, with wrapping
    from -180 to 180. Modified code courtesy:
    https://stackoverflow.com/questions/491738/
    how-do-you-calculate-the-average-of-a-set-of-circular-data
    
    Inputs:
    -------
    p_vals: array of phase values in degrees
    weights: optional, weights for each phase in p_vals
    
    Outputs:
    --------
    p_avg: phase average in degrees
    """
    
    if weights is None:
        weights = np.ones(p_vals.shape)
    
    x,y = 0,0
    for (angle, weight) in zip(p_vals, weights):
        x += np.cos(np.deg2rad(angle)) * weight / np.sum(weights)
        y += np.sin(np.deg2rad(angle)) * weight / np.sum(weights)
    
    return np.rad2deg(np.arctan2(y,x))


def std_phases(p_vals, weights=None):
    """
    Compute the standard deviation of input phases, yielding a value that
    roughly matches the output of scipy.stats.circstd (provided the range
    of phase values is not too large). Unlike scipy.stats.circstd, enable
    value-weighting of the phases.

    Inputs:
    -------
    p_vals: array of phase values in degrees
    weights: optional, weights for each phase in p_vals
    
    Outputs:
    --------
    p_std: phase standard deviation in degrees
    """
    p_avg = average_phases(p_vals, weights=weights)
    p_var = np.average(wraptopi(p_vals - p_avg)**2, weights=weights)

    return np.sqrt(p_var)


def stderr_phases(p_vals, weights = None):
    """
    Estimate the standard error of input phases as the square root of their 
    circular variance, which yields a value in the range [0,1], divided by 
    the number of samples.
    
    Inputs:
    -------
    p_vals: array of phase values in degrees
    weights: optional, weighs of phases in p_vals
    
    Outputs:
    --------
    p_stderr: circular variance, between 0 and 1
    """
    if weights is None:
        weights = np.ones(p_vals.shape)
        
    x,y = 0,0
    for (angle, weight) in zip(p_vals, weights):
        x += np.cos(np.deg2rad(angle)) * weight / np.sum(weights)
        y += np.sin(np.deg2rad(angle)) * weight / np.sum(weights)
        
    r2 = np.square(x) + np.square(y)
    p_var = 1 - np.sqrt(r2)
    p_var = np.clip(p_var, 0.0, 1.0) # avoid floating point errors  
    
    return np.sqrt(p_var / float(len(p_vals))) 


def residual_phases(hklp1, hklp2):
    """
    Compute the difference between the phases of reflections shared between hklp1 
    and hklp2 input dictionaries. Input phases and output residual are in degrees.
    
    Inputs:
    -------
    hklp1: OrderedDict with keys as Millers and phases as np.array([phase]) 
    hklp2: OrderedDict with keys as Millers and phases as np.array([phase]) 

    Output:
    -------
    residuals: array of difference between shared phases in hklp1 and hklp2
    """
    p_shared = shared_dict(hklp1, hklp2)
    p_vals = np.array(p_shared.values())
    
    p1, p2 = p_vals[:,0], p_vals[:,1]
    diff = p1 - p2
    diff[diff>180] -= 360
    diff[diff<-180] += 360
    
    return np.abs(diff)


def residual_to_avgphase(hklp):
    """
    For each reflection in hklp with more than one recorded phase value, compute
    the mean residual of the values to the mean phase -- a measure of precision.
    
    Inputs:
    -------
    hklp: OrderedDict with keys as Millers and phases as np.array([phase(s)]) 

    Output:
    -------
    residuals: array of mean delta between mean phase and consituent phases
    """
    residuals = list()
    
    for key in hklp.keys():
        if len(hklp[key]) > 1:
            diff = hklp[key] - average_phases(hklp[key])
            diff[diff>180] -= 360
            diff[diff<-180] += 360
            residuals.append(np.mean(np.abs(diff)))
    
    return np.array(residuals)


def residual_phase_distribution(hklp1, hklp2):
    """
    Compute the difference between the phases of reflections shared between hklp1 
    and hklp2 input dictionaries. Input phases and output residual are in degrees.
    
    Inputs:
    -------
    hklp1: OrderedDict with keys as Millers and phases as np.array([phase]) 
    hklp2: OrderedDict with keys as Millers and phases as np.array([phase]) 

    Output:
    -------
    sigma: sigma from fitting N(mu, sigma) to phase residuals 
    m_error: mean error of phase residuals
    """
    import scipy.stats

    p_shared = shared_dict(hklp1, hklp2)
    p_vals = np.array(p_shared.values())
    
    p1, p2 = p_vals[:,0], p_vals[:,1]
    diff = p1 - p2
    diff[diff>180] -= 360
    diff[diff<-180] += 360
    
    mu, std = scipy.stats.norm.fit(diff)    
    return std, np.mean(np.abs(diff))


def remove_Friedels(miller_list):
    """
    Remove Friedel pairs from input list of Miller indices (list of tuples format).
    
    Inputs:
    -------
    miller_list: list of Miller indices, each given as a tuple

    Outputs:
    --------
    miller_list: updated list with Friedel mates removed
    """
    for miller in miller_list:
        friedel = (-1*miller[0], -1*miller[1], -1*miller[2])
        if friedel in miller_list:
            miller_list.remove(friedel) 

    return miller_list


def sym_ops_friedels(sym_ops):
    """
    Expand the input dictionary of symmetry operations with those for each of
    the Friedel mates. For each symmetry operation, the rotational element is
    multiplied by negative 1 while the translational component is the same as
    the starting symmetry relationship.
    
    Inputs:
    -------
    sym_ops: dictionary of symmetry operations
    
    Outputs:
    -------
    sym_ops: expanded dictionary of symmetry operations
    """
    for key in sym_ops.keys():
        sym_ops[max(sym_ops.keys())+1] = np.vstack((-1*sym_ops[key][:,:-1].T, 
                                                     sym_ops[key][:,-1])).T
    return sym_ops


def compute_resolution(space_group, cell_constants, hkl):
    """
    Compute d-spacing / resolution for a set of scattering vectors, where q = 2*pi*s.
    If present, set (0,0,0) to arbitrarily high (but not infinite) resolution.
    
    Inputs:
    -------
    space_group: number corresponding to space group
    cell_constants: array of cell constants, (a, b, c, alpha, beta, gamma)
    hkl: array of n Miller indices, n x 3

    Output:
    -------
    resolution: resolution for hkl list (empty if space group is unsupported)
    """

    a, b, c, alpha, beta, gamma = cell_constants
    h, k, l = hkl[:,0], hkl[:,1], hkl[:,2]

    # valid for orthorhombic, cubic, and tetragonal
    if ((space_group >= 16) and (space_group <=142)) or ((space_group >= 195) and (space_group <=230)):
        inv_d = np.sqrt(np.square(h/a) + np.square(k/b) + np.square(l/c))

    # valid for hexagonal (and possibly trigonal? if so change lower bound to 143)
    elif (space_group >= 168) and (space_group <=194):
        inv_d = np.sqrt(4.0*(np.square(h) + h*k + np.square(k))/(3*np.square(a)) + np.square(l/c))

    # valid for monoclinic
    elif (space_group >= 3) and (space_group <=15):
        beta = np.deg2rad(beta)
        inv_d = np.sqrt(np.square(h/(a*np.sin(beta))) + np.square(k/b) + np.square(l/(c*np.sin(beta))) \
                             + 2*h*l*np.cos(beta) / (a*c*np.square(np.sin(beta))))

    # valid for triclinic
    elif (space_group <= 2):
        alpha, beta, gamma = np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)
        s11 = np.square(b*c*np.sin(alpha))
        s22 = np.square(a*c*np.sin(beta))
        s33 = np.square(a*b*np.sin(gamma))
        s12 = a*b*np.square(c)*(np.cos(alpha)*np.cos(beta) - np.cos(gamma))
        s23 = np.square(a)*b*c*(np.cos(beta)*np.cos(gamma) - np.cos(alpha))
        s13 = a*np.square(b)*c*(np.cos(gamma)*np.cos(alpha) - np.cos(beta))
        V = a*b*c*np.sqrt(1 - np.square(np.cos(alpha)) - np.square(np.cos(beta)) - np.square(np.cos(gamma)) \
                          + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))
        inv_d = np.sqrt(1.0/np.square(V)*(s11*np.square(h) + s22*np.square(k) + s33*np.square(l) \
                                          + 2*s12*h*k + 2*s23*k*l + 2*s13*h*l))

    else:
        print "This space group is currently unsupported."
        return np.empty(0)

    inv_d[inv_d==0] = 1e-5
    res = 1.0 / inv_d

    return res


# --------------------------------------------- # 
#           Data wrangling in real space        #
# --------------------------------------------- #


def save_mrc(volume, savename):
    """
    Save 3d numpy array, volume, to path savename in mrc format.
    """
    mrc = mrcfile.new(savename, overwrite=True)
    mrc.header.map = mrcfile.constants.MAP_ID
    mrc.set_data(volume.astype(np.float32))
    mrc.close()
    return


def rescale_vol(volume):
    """
    Rescale the voxel values of a volume to lie between 0 and 1.
    """
    return (volume - volume.min()) / (volume.max() - volume.min())


def reformat_mdtraj_pdb(fname, outname):
    """
    Reformat the input pdb file (fname) in MDTraj format to match PDB conventions.
    Standard PDB convention has one more column in front of the atom label than
    what's written out by MDTraj.
    """

    file1 = open(fname, "r")
    file2 = open(outname, "w")

    for fline in file1.readlines():
        if "ATOM" in fline:
            file2.write(fline[:62] + "5" + fline[63:76] + " " + fline[76:])
        else:
            file2.write(fline)

    file1.close()
    file2.close()

    return

# --------------------------------------------- #
#                  Miscellany                   #
# --------------------------------------------- #


def shared_dict(d1, d2):
    """
    Return a dictionary of common reflections from the input dictionaries.

    Inputs:
    -------
    d1: dict whose keys are Millers, values are unrestricted
    d2: dict whose keys are Millers, values are unrestricted

    Outputs:
    --------
    shared_d: dict of common Millers as keys, np.array([d1[miller], d1[miller]]) as values
    """
    shared_hkl = set(d1.keys()).intersection(d2.keys())
    shared_d = OrderedDict()

    for hkl in shared_hkl:
        shared_d[hkl] = np.array([d1[hkl], d2[hkl]])

    return shared_d
