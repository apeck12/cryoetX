# --------------------------------------------- # 
#                    Imports                    #
# --------------------------------------------- #


from collections import OrderedDict
import cPickle as pickle
import numpy as np
import itertools, warnings, os
import pathos.pools as pp
import scipy.optimize
import cctbx_utils, utils

from cctbx.array_family import flex
from cctbx import miller, crystal
from iotbx import mtz


# --------------------------------------------- # 
#                    Classes                    #
# --------------------------------------------- #


class PeakFitting:

    """
    Class for fitting Bragg peak profiles, with different options to do so based on local
    intensity and phase values. 
    """

    def __init__(self, tomogram, hkl_info, length):
        """
        Initialize class by extracting intensity and phase subvolumes.
        
        Inputs:
        -------
        tomogram: real space volume
        hkl_info: dict containing Miller and peak center information
        length: cubic length of subvolume to consider around each potential Bragg peak
        """
        self.ivols, self.pvols = self.extract_subvolumes(tomogram, hkl_info, length)
        self.length = length

        
    def extract_subvolumes(self, tomogram, hkl_info, length):
        """
        Extract cubic subvolumes around each predicted peak center given by xyz coordinates
        in hkl_info. Return separate dictionaries containing subvolumes for intensities and
        phases (in degrees) as values and corresponding Miller indices as keys. 
        
        Inputs:
        -------
        tomogram: real space volume
        hkl_info: dict containing Miller and peak center information
        length: cubic length of subvolume to consider around each potential Bragg peak

        Outputs:
        --------
        ivols: OrderedDict, keys as Millers and values as subvolumes of the FT intensities
        pvols: OrderedDict, keys as Millers and values as subvolumes of the FT phases
        """
        # FFT volume to compute intensities and phases 
        fftI, fftp = utils.ft_to_I_phase(utils.compute_ft(tomogram))
        ivols, pvols = OrderedDict(), OrderedDict()
        
        # if hkl_info is a regular rather than DIALS-style pickle file
        if type(hkl_info) == dict:
            self.xyz_calc = hkl_info['xyz']
            hkl = list(map(tuple, hkl_info['hkl']))
        
        # otherwise, use DIALS functions to retreive hkl and xyz information
        else:
            hkl = indexed.select(indexed['miller_index']!=(0,0,0))['miller_index']
            self.xyz_calc = np.array(indexed.select(indexed['miller_index']!=(0,0,0))['xyzobs.px.value'])
            
        xyz = np.around(self.xyz_calc).astype(int)
        x0, x1, y0, y1, z0, z1 = xyz[:,0]-int(length/2), xyz[:,0]+int(length/2), xyz[:,1]-int(length/2), \
            xyz[:,1]+int(length/2), xyz[:,2]-int(length/2), xyz[:,2]+int(length/2)
        
        # extract cubic subvolumes for phases and intensities
        for i in range(len(hkl)):
            ivols[hkl[i]] = fftI[z0[i]:z1[i]+1, y0[i]:y1[i]+1, x0[i]:x1[i]+1]
            pvols[hkl[i]] = fftp[z0[i]:z1[i]+1, y0[i]:y1[i]+1, x0[i]:x1[i]+1] * 180/np.pi

        self.retained_hkl = ivols.keys()
        return ivols, pvols
            
        
    def _find_contiguous(self, mapping, start, contig_list):
        """
        Identify pixels contiguous with pixel mapped to start index. This function works
        recursively; on the first iteration, contig_list should be an empty numpy array.
        """
        if len(contig_list)==0:
            contig_list = np.array([start])
        
        to_check = mapping[start]
        for idx in to_check:
            if idx not in contig_list:
                contig_list = np.concatenate((np.array([idx]), contig_list))
                contig_list = self._find_contiguous(mapping, idx, contig_list)

        return contig_list


    def _adjacency_filter(self, mask, coincident_mask = None):
        """
        Remove marked (value of 1) pixels that don't face-border at least one other unmasked 
        pixel. If there are multiple isolated sets of contiguous pixels, retain the set that 
        is closest to the center of the subvolume or the coincident mask, if given.
        """
        from scipy.spatial.distance import squareform, pdist
        import scipy.ndimage

        # compute pairwise distances between all identified pixels
        mask_idx = np.array(np.where(mask!=0), dtype='float').T
        euc_dist = squareform(pdist(mask_idx, 'euclidean'))

        # map each marked pixels to its adjacent marked pixels
        mapping = dict()
        adj_pix = [np.where(euc_dist[i]==1)[0] for i in range(euc_dist.shape[0])]
        for i in range(len(adj_pix)):
            mapping[i] = adj_pix[i]

        # find set of contiguous pixels that is nearest to center of subvolume or coincident mask, if given
        if coincident_mask is None:
            cen_array = np.expand_dims(np.array(mask.shape)/2, 0) 
        else:
            cen_array = np.expand_dims(np.array(scipy.ndimage.measurements.center_of_mass(coincident_mask)), 0)
        d_cen = scipy.spatial.distance.cdist(mask_idx, cen_array, 'euclidean')
        cen = np.where(d_cen==d_cen.min())[0][0]    
        contig_list = np.empty(0)
        contiguous_set = self._find_contiguous(mapping, cen, contig_list)

        # remove any marked pixels that do not belong to the identified contiguous set
        for i in range(mask_idx.shape[0]):
            if i not in contiguous_set:
                xm,ym,zm = mask_idx[i].astype(int)
                mask[xm,ym,zm] = 0

        return mask


    def compute_imasks(self, sigma, k_size=None, coincident_masks=None):
        """
        Generate peak masks, where 1 corresponds to pixels predicted to be spanned by a 
        reflection. Pixels are predicted to belong to a peak if their intensity exceeds: 
           I(px) > median_filter(ivol)_px + mean(ivol) + sigma * std_dev(ivol)
        if a kernel size (k_size) for a median filter is given, or, if k_size is None:
           I(px) > mean(ivol) + sigma * std_dev(ivol)
        and they are adjacent to minimally one other pixel spanned by the peak.
        
        Inputs:
        -------
        sigma: factor determining threshold above which pixels are included
        k_size: kernel size if using a median filter, optional
        coincident_masks: dict with keys as Millers, values as subvolume masks
        
        Outputs:
        --------
        imasks: dict with keys as Millers, values as subvolume masks (0=reject, 1=retain)
        """        
        import scipy.signal
        
        # generate a spherical mask to exclude pixels outside of radius 1 less than half cubic length
        radius = int(self.length / 2)
        r_threshold = radius - 1
        
        mg = np.mgrid[-radius:radius:(2*radius+1)*1j,
                      -radius:radius:(2*radius+1)*1j,
                      -radius:radius:(2*radius+1)*1j]
        r_dist = np.square(mg[0].flatten()) + np.square(mg[1].flatten()) + np.square(mg[2].flatten())
        r_dist = np.sqrt(r_dist).reshape(radius*2+1,radius*2+1,radius*2+1)
        r_mask = np.zeros_like(r_dist)
        r_mask[r_dist < r_threshold] = 1

        # generate a mask for each Miller index in self.ivols
        imasks = OrderedDict()
        counter = 0

        if coincident_masks is None:
            self.xyz_obs = OrderedDict()

        for key in self.ivols.keys():
            ivol = self.ivols[key].copy()
            mask = np.zeros(ivol.shape)
            
            # pixels that exceed I_mean + sigma * I_std are included in peak
            if k_size is None:
                i_threshold = np.mean(ivol) + sigma*np.std(ivol)
                mask[ivol>i_threshold] = 1
            
            # if k_size is given, base threshold on some intensity over median-filtered volume    
            else:
                medfilt = scipy.signal.medfilt(ivol, kernel_size=k_size)
                mask[np.where(ivol - medfilt > np.mean(ivol) + sigma*np.std(ivol))] = 1
                
            # eliminate pixels outside spherical mask and check for adjacency 
            mask[r_mask==0] = 0
            if np.sum(mask) > 0:
                if coincident_masks is None:
                    mask = self._adjacency_filter(mask, coincident_mask = None)
                    
                    # compute peak centroid: intensity-weighted peak position if centroid not already defined
                    xc = np.sum(ivol[np.where(mask==1)] * np.where(mask==1)[0]) / np.sum(ivol[np.where(mask==1)])
                    yc = np.sum(ivol[np.where(mask==1)] * np.where(mask==1)[1]) / np.sum(ivol[np.where(mask==1)])
                    zc = np.sum(ivol[np.where(mask==1)] * np.where(mask==1)[2]) / np.sum(ivol[np.where(mask==1)])
                    self.xyz_obs[key] = self.xyz_calc[counter] - radius + np.array([zc, yc, xc])

                else:
                    if np.sum(coincident_masks[key]) > 0:
                        mask = self._adjacency_filter(mask, coincident_mask = coincident_masks[key])
                    
            imasks[key] = mask
            counter += 1
        
        return imasks
    
    
    def compute_pmasks(self, imasks, p_threshold):
        """
        Generate a submask for each hkl, where 1 corresponds to the set of pixels whose 
        retention yields an intensity-weighted standard deviation less than p_threshold.
        
        Inputs:
        -------
        masks: dict with keys as Millers and values as masks indicating valid pixels
        p_threshold: threshold for standard deviation of phases belonging to peak
        
        Outputs:
        --------
        masks: dict with keys as Millers and values as masks indicating valid pixels
            based on p_threshold criterion
        """       
        pmasks = OrderedDict()
        
        # loop over all Miller indices 
        for key in imasks.keys():
            mask = np.zeros((self.length, self.length, self.length))
            if np.sum(imasks[key]!=0):
                p_vals = self.pvols[key][np.where(imasks[key]!=0)]
                i_vals = self.ivols[key][np.where(imasks[key]!=0)]
            
                # iteratively identify pixels to remove until p_threshold is reached
                valid_idx = np.ones_like(p_vals).astype(int)
                p_temp, i_temp = p_vals.copy(), i_vals.copy()
                for i in range(len(p_vals)):
                    while utils.std_phases(p_temp, i_temp) > p_threshold:
                        delta = np.abs(utils.wraptopi(p_temp - utils.average_phases(p_temp,weights=i_temp)))
                        max_idx = np.where(delta==delta.max())[0][0]
                        valid_idx[p_vals==p_temp[max_idx]] = 0
                        p_temp = np.delete(p_temp, max_idx)
                        i_temp = np.delete(i_temp, max_idx)
        
                # compute mask in shape of subvolume, with pixels to retain marked by a value of 1
                peak_idx = np.where(imasks[key]!=0)
                retained_idx = np.array(peak_idx).T[valid_idx==1]    
                for i in range(retained_idx.shape[0]):
                    xi,yi,zi = retained_idx[i]
                    mask[xi,yi,zi] = 1
                
            pmasks[key] = mask
            
        return pmasks
    
    
    def estimate_phases(self, masks, mode='weighted_mean'):
        """
        Estimate the phase of each Bragg peak as 1) the weighted mean of peak pixels 
        (mode="weighted_mean"), 2) the mean of peak pixels (mode="mean"), or 3) the 
        value of the pixel with the maximum intensity (mode="maxI"). Peak pixels are 
        indicated by a value of 1 in masks.
        
        Inputs:
        -------
        masks: dict with keys as Millers and values as masks indicating valid pixels
        
        Outputs:
        --------
        p_est: dict with keys as Millers and values as estimated phase
        p_std: dict with keys as Millers and values as phase standard deviation
        """
        p_est, p_std = OrderedDict(), OrderedDict()
        
        # loop over all Miller indices
        for key in self.retained_hkl:
            # phase cannot be determined if no valid peak pixels
            if np.sum(masks[key]) == 0:
                print "Phase could not be estimated for Miller %s" %(key,)
            
            else:
                p_vals = self.pvols[key][np.where(masks[key]!=0)]
                i_vals = self.ivols[key][np.where(masks[key]!=0)]
                
                if mode == "weighted_mean":
                    p_est[key] = utils.average_phases(p_vals, weights=i_vals)
                    p_std[key] = utils.std_phases(p_vals, weights=i_vals)
                    
                elif mode == "mean":
                    p_est[key] = utils.average_phases(p_vals)
                    p_std[key] = utils.std_phases(p_vals)
                    
                elif mode == "maxI":
                    p_est[key] = p_vals[i_vals==i_vals.max()]
                    p_std[key] = utils.std_phases(p_vals, weights=i_vals)
                    
                else:
                    print "Error: mode must be weighted_mean, mean or maxI"
                    
        return p_est, p_std
                    
                    
    def estimate_background(self, masks):
        """
        Estimate the background intensity by performing linear interpolation to predict the 
        intensity values of pixels marked by 1 in input masks. The purpose of estimating the 
        underlying background intensity at each pixel rather than simply summing nearby pixels
        is due to the effect of the missing wedge, which yields nonuniform background patterns.
        
        Inputs:
        -------
        masks: dict with keys as Millers and values as mask subvolumes
        
        Outputs:
        --------
        bgd_ests: dict with keys as Millers and values as background subvolumes, optional
        """
        import scipy.interpolate
    
        bgd_ests = OrderedDict()
        for key in masks.keys():
            bgd = self.ivols[key].copy().flatten()
            
            if np.sum(masks[key]) != 0:
                # using np.ravel methods to locate masked (Bragg) pixels
                mg_1d = np.arange(masks[key].shape[0])
                ravel_shape = (len(mg_1d), len(mg_1d), len(mg_1d))
                xyz = np.array(list(itertools.product(mg_1d, mg_1d, mg_1d)), dtype='int')
                xyz_rav = np.ravel_multi_index(xyz.T, ravel_shape)

                mask_idx = np.ravel_multi_index(np.where(masks[key]==1), ravel_shape)
                for idx in mask_idx:
                    xyz_rav[idx] = -1

                xi = np.array(np.where(masks[key]==1)).T
                points = np.array(np.unravel_index(xyz_rav[xyz_rav!=-1], ravel_shape)).T
                values = self.ivols[key][masks[key]==0].copy()

                # estimate masked (Bragg) pixels by interpolation 
                est_xi = scipy.interpolate.griddata(points, values, xi, method='linear')
                est = masks[key].flatten()
                bgd[est==1] = est_xi # fill gaps in original volume with estimated values
            
            bgd_ests[key] = bgd.reshape(masks[key].shape)

        return bgd_ests
    
    
    def estimate_intensities(self, masks, bgd_ests = None):
        """
        Estimate the intensity for each Bragg peak as the sum of retained pixels, indicated by a 
        value of 1 in masks. If bgd_ests is given, subtract the estimated background before 
        computing the sum.
        
        Inputs:
        -------
        masks: dict with keys as Millers and values as mask subvolumes
        bgd_ests: dict with keys as Millers and values as background subvolumes, optional
        
        Outputs:
        --------
        i_est: dict with keys as Millers and values as integrated peak intensity
        i_std: dict with keys as Millers and values as standard deviation of peak intensity
        """
        i_est, i_std = OrderedDict(), OrderedDict()
        
        for key in self.retained_hkl:  
            if np.sum(masks[key])==0:
                print "Intensity could not be estimated for Miller %s" %(key,)
                
            else:
                if bgd_ests is None:
                    i_vals = self.ivols[key][np.where(masks[key]!=0)]                
                else:
                    ivol_mbgd = self.ivols[key].copy() - bgd_ests[key]
                    i_vals = ivol_mbgd[np.where(masks[key]!=0)]
                
                i_est[key], i_std[key] = np.sum(i_vals), np.std(i_vals)
                
        return i_est, i_std


    def npixels_threshold(self, masks, n_threshold):
        """
        Compute the number of pixels belonging to each reflection, and update the list of retained 
        Millers (self.retained_hkl) and peak centers (self.xyz_obs) to only keep reflections with 
        at least n_threshold pixels.
        
        Inputs:
        -------
        masks: dict with keys as Millers and values as mask subvolumes
        n_threshold: threshold number of pixels for retaining peak
        
        Outputs:
        --------
        n_pixels: array of number of retained pixels per peak
        """
        n_pixels = np.array([np.sum(masks[miller]) for miller in masks.keys()])
        counter = 0

        for i,miller in enumerate(masks):
            if (n_pixels[i] < n_threshold) and (miller in self.retained_hkl):
                self.retained_hkl.remove(miller)
                counter += 1

                if miller in self.xyz_obs.keys():
                    self.xyz_obs.pop(miller)

        print "%i Millers removed based on npixel threshold" %counter
        return n_pixels


    def stdratio_threshold(self, masks, sr_threshold):
        """
        Retain any hkl associated with an std_ratio above input sr_threshold, where std_ratio is 
        the ratio of the standard deviations of peak to non-peak pixel intensities for the slices 
        along the x direction that contain at least 1 marked pixel. Miller list self.retained_hkl 
        and self.xyz_obs are updated accordingly, and an np.array of std_ratio values is returned.
        
        Inputs:
        -------
        masks: dict with keys as Millers and values as mask subvolumes
        
        Outputs:
        --------
        std_ratio: array of std_ratio value per peak, as explained above
        """
        std_ratios = np.zeros(len(masks))    
        counter = 0

        for i,miller in enumerate(masks):
            if np.sum(masks[miller]) > 0:
                xslices = np.unique(np.where(masks[miller]!=0)[0])
                num = np.concatenate([self.ivols[miller][xs][masks[miller][xs]==1] for xs in xslices]).ravel()
                den = np.concatenate([self.ivols[miller][xs][masks[miller][xs]==0] for xs in xslices]).ravel()
                std_ratios[i] = np.std(num) / np.std(den)

                if (std_ratios[i] < sr_threshold) and (miller in self.retained_hkl):
                    self.retained_hkl.remove(miller)
                    counter += 1

                    if miller in self.xyz_obs.keys():
                        self.xyz_obs.pop(miller)

        print "%i Millers removed based on std dev ratio threshold" %counter
        return std_ratios


class FindOrigin:

    """
    Class to assist with locating the crystallographic phase origin.
    """
    
    def __init__(self, sg_symbol, cell, cs, hklp, hklI):
        """
        Initialize class, minimally with a dictionary of phases and unit cell dimensions.
        
        Inputs:
        -------
        sg_symbol: space group symbol in Hermann-Mauguin notation
        cell: tuple of unit cell constants, (a,b,c)
        cs: CCTBX symmetry object
        hklp: dictionary whose keys are Millers and values are phases in degrees
        hklI: dictionary whose keys are Millers and values are intensities
        """
        ref_path = os.path.join(os.path.abspath("."), "reference")
        sym_ops = pickle.load(open(os.path.join(ref_path,"sym_ops.pickle")))[sg_symbol]
        self.sym_ops = utils.sym_ops_friedels(sym_ops)
        self.caxis_val, self.cslice_val = pickle.load(open(os.path.join(ref_path,
                                                                        "phase_restrictions.pickle")))[sg_symbol]
        self.cell = cell
        self.cs = cs
        self._set_up(hklp, hklI)
        

    def _centralhkl_indices(self):
        """
        Locate indices of central axis and central slice reflections.
        
        Outputs:
        --------
        caxis_idx: np.array of indices of cenral axis reflections
        cslice_idx: np.array of indices of central slice reflections
        """
        caxis_idx, cslice_idx = list(), list()
        
        for i,miller in enumerate(self.hkl):
            if miller.count(0)==2:
                caxis_idx.append(i) # central axis
            if miller.count(0)==1:
                cslice_idx.append(i) # central slice
        
        return np.array(caxis_idx), np.array(cslice_idx)

    
    def _symhkl_indices(self):
        """
        Return indices of each primary reflection (base_idx); any of its symmetry-equivalents, 
        including identity (sym_idx); and the key of the translational operation that relates
        them., with the ordering according to sym_ops dictionary. Friedel mates are treated as
        symmetry-equivalents.

        Outputs:
        --------
        base_idx: np.array of indices of "parent" reflections
        sym_idx: np.array of indices of "children" reflections (symmetry-related to parent)
        op_idx: np.array of the sym_ops operation that relates base_idx and sym_idx
        """    
        self.R_ops = np.array([self.sym_ops[i][:,:-1] for i in range(len(self.sym_ops.keys()))])
        self.T_ops = np.array([self.sym_ops[i][:,-1] for i in range(len(self.sym_ops.keys()))])

        miller_seen = list()
        base_idx, sym_idx, op_idx = list(), list(), list()

        # loop over all reflections and check for symmetry-related reflections
        for i,miller in enumerate(self.hkl):
            miller_sym = np.inner(np.array(miller), self.R_ops).astype(int)

            for op,ms in enumerate(miller_sym):
                if (tuple(ms) in self.hkl) and (tuple(ms) not in miller_seen):
                    op_idx.append(op)
                    base_idx.append(i)
                    sym_idx.append(self.hkl.index(tuple(ms)))
                    miller_seen.append(tuple(ms))
                    
        return np.array(base_idx), np.array(sym_idx), np.array(op_idx) 
    
       
    def _set_up(self, hklp, hklI):
        """
        Set up symmetry information and indexing information useful for pulling out relevant
        reflections. If intensities are supplied, use as weights.
        
        Inputs:
        -------
        hklp: dictionary whose keys are Millers and values are phases in degrees
        hklI: dictionary whose keys are Millers and values are intensities, or None
        """     
        # extract reflection information, store as Millers, phases, intensities
        assert hklI.keys() == hklp.keys()
        self.hkl = hklp.keys()
        self.phases = np.squeeze(np.array(hklp.values()))
        self.I = np.squeeze(np.array(hklI.values()))
        
        # indices for restricted phase reflections (central axis/slice)
        self.caxis_idx, self.cslice_idx = self._centralhkl_indices()
        
        # indices for symmetry-related reflections        
        self.base_idx, self.sym_idx, self.op_idx = self._symhkl_indices()    
        
        return

    
    def sym_residual(self, phases, weighted=True):
        """
        Compute the average residual of symmetry-equivalent reflections, assuming that given 
        phases are in degrees and their ordering matches that of self.hkl. Expected phase is:
        
        p_sym = p_base + 360 * dot(hkl_sym, T_sym)
        
        where p_base corresponds to the phase of the principal reflection, and hkl_sym and T_sym
        respectively are the symmetry-related reflection and its translational operator.
        
        Inputs:
        -------
        phases: array of phases, with ordering matching self.hkl, in degrees
        weighted: boolean, if True weight phase residual by corresponding intensity
    
        Outputs:
        --------
        r_avg: mean residual of symmetry-equivalent phases to their expected values
        """       
        # if no symmetry related reflections are present, return 0
        if len(self.base_idx) == 0 and len(self.sym_idx) == 0:
            return 0.0

        p_base, p_obs = phases[self.base_idx], phases[self.sym_idx]
        hkl_sym = np.array(self.hkl)[self.sym_idx]
        T = self.T_ops[self.op_idx]

        # map all reflections to ASU, including -1 adjustment for Friedels
        shifts = np.sum(hkl_sym[:,:,np.newaxis] * T[:,:,np.newaxis], axis=1).flatten()
        p_calc = utils.wraptopi(p_obs - 360 * shifts) 
        p_calc[np.where(np.array(self.op_idx) > max(self.sym_ops.keys())/2)] *= -1 

        # compute expected mean phase value for the parent reflection from all observations
        unq_idx = np.unique(np.array(self.base_idx))
        p_avg = np.zeros_like(p_base)
        for unq in unq_idx:
            p_avg[np.array(self.base_idx)==unq] = utils.average_phases(p_calc[np.array(self.base_idx)==unq])
        
        r_vals = np.abs(utils.wraptopi(p_calc - p_avg))
        if weighted is True:
            return np.average(r_vals, weights=self.I[self.sym_idx])
        else:
            return np.average(r_vals)
    
        
    def central_residual(self, phases, weighted=True):
        """
        Compute the average residual of central slice / axis reflections to the nearest integer
        multiple of their expected values.
        
        Inputs:
        -------
        phases: array of phases, with ordering matching self.hkl, in degrees
        weighted: boolean, if True weight phase residual by corresponding intensity
    
        Outputs:
        --------
        r_avg: mean residual of central axis / slice phases to their expected values
               or 0, if no centric reflections are in dataset
        """
        # if not centric reflections are present, return 0
        if len(self.caxis_idx) == 0 and len(self.cslice_idx) == 0:
            return 0.0

        # otherwise, compute mean residual of centric reflections
        e_vals = [self.caxis_val, self.cslice_val]
        idx_lists = [self.caxis_idx, self.cslice_idx]
        
        r_vals, w_vals = list(), list()
        for e_val, idx_list in zip(e_vals, idx_lists):
            if len(idx_list) > 0:
                p_sel = phases[idx_list]
                p_residuals = np.abs(p_sel % float(e_val))
                n_residuals = np.abs(p_sel % float(-1 * e_val))
                stacked = np.vstack((p_residuals, n_residuals)).T
                r_vals.extend(list(np.around(np.abs(np.amin(stacked, axis=1)), 2)))
        r_vals = np.array(r_vals)
        
        if weighted is True:
            return np.average(r_vals, weights=self.I[np.concatenate(idx_lists).astype(int)])
        else:
            return np.average(r_vals)


    def map_skewness(self, phases):
        """
        Compute the skewness of the real space map computed from self.hkl and input phases.
        
        Inputs:
        ------
        phases: array of phases, with order matching self.hkl, in degrees

        Outputs:
        --------
        s_factor: skewness of electron density map as a float
        """
        import scipy.stats

        # compute Miller array, convert to ccp4_map, convert to numpy array
        ma = cctbx_utils.convert_to_sf(self.hkl, self.I, np.deg2rad(phases), self.cs)
        fmap = cctbx_utils.compute_map(ma, save_name = None, grid_step = 0.5)
        fmap_as_npy = fmap.real_map_unpadded().as_numpy_array()

        return scipy.stats.skew(fmap_as_npy.flatten())
    
            
    def shift_phases(self, fshifts):
        """
        Compute new phases after shifting the unit cell origin by fractional shifts.
    
        Inputs:
        -------
        fshifts: fractional shifts along a,b,c by which to translate unit cell origin
    
        Outputs:
        --------
        p_shifted: dictionary with keys as Millers, and values as origin-centered phases
        """
        
        return utils.wraptopi(self.phases - 360.0 * np.dot(np.array(self.hkl), fshifts))    
    
    
    def _eval_shifts(self, fshifts):
        """
        Compute residuals and map skewness for phases translated by input shifts.
        
        Inputs:
        -------
        fshifts: fractional shifts along the crystallographic a,b,c axes

        Outputs:
        --------
        eval: tuple of (central hkl residual, sym-equiv residual, map skewness)
        """
        p_shifted = self.shift_phases(fshifts)
        r_cen = self.central_residual(p_shifted)
        r_sym = self.sym_residual(p_shifted)
        skew = self.map_skewness(p_shifted)
        
        return (r_cen, r_sym, skew)

    
    def scan_candidates(self, grid_spacing, n_processes, w_rsym=1.0, w_rcen=1.0, w_skew=1.0):
        """
        Perform a grid scan to assess each fractional position as a candidate phase origin.
        
        Inputs:
        -------
        grid_spacing: grid spacing in Angstrom for phase origin search
        n_processes: number of processors for multiprocessing
        w_rsym: relative weight for symmetry residual, default is equal weights
        w_rcen: relative weight for central reflection residual, default is equal weights
        w_skew: relative weight for map skewness metric, default is equal weights
        
        Outputs:
        --------
        metrics: dictionary of metrics for sym and central residuals and map skewness
        fshifts: fractional shifts by which to translate unit cell origin
        """
        
        # set up grid of fractional shifts to scan as candidate origins
        xshifts, yshifts, zshifts = [np.arange(0, self.cell[i], grid_spacing) for i in range(3)]
        fshifts_list = list(itertools.product(xshifts/self.cell[0], 
                                              yshifts/self.cell[1], 
                                              zshifts/self.cell[2]))    
        num = len(fshifts_list)
        print "Finding origin: %i grid point to evaluate" %num

        # evaluate shifts using multiprocessing
        pool = pp.ProcessPool(n_processes)
        metrics = pool.map(self._eval_shifts, fshifts_list)
        metrics = np.array(metrics)

        # rearrange results into separate dictionary keys
        dmetrics = dict()
        for i,m in zip(range(3), ['rcen', 'rsym', 'skew']):
            dmetrics[m] = metrics[:,i].reshape(len(xshifts), len(yshifts), len(zshifts))

        # computing combined metric, taking inverse of skew factor, weighting and normalizing
        for key in dmetrics.keys():
            if key == 'skew':
                dmetrics[key] = -1.0 * dmetrics[key]
            if np.all(dmetrics[key]==0):
                print "Could not evaluate %s due to lack of relevant reflections" %key
            else:
                dmetrics[key] = (dmetrics[key]-dmetrics[key].min())/(dmetrics[key].max()-dmetrics[key].min())
        dmetrics['combined'] = w_rsym*dmetrics['rsym'] + w_rcen*dmetrics['rcen'] + w_skew*dmetrics['skew']
        
        # identify top candidate phase origin
        sorted_origins = np.array(np.unravel_index(np.argsort(dmetrics['combined'].flatten()),
                                                   dmetrics['combined'].shape)).T
        xs, ys, zs = sorted_origins[0]
        fshifts = (xshifts[xs]/self.cell[0], yshifts[ys]/self.cell[1], zshifts[zs]/self.cell[2])

        return dmetrics, fshifts


class MergeCrystals:
    
    """
    Class for merging data from different crystals by shifting the nth added crystal to 
    the same phase origin (not necessarily coincident with a crystallographic origin) as 
    the first dataset added and scaling intensities.
    """
    
    def __init__(self, space_group, grid_spacing):
        """
        Initialize class with empty OrderedDicts to store intensities and phases, and
        a dictionary to store unit cell constants. Input grid_spacing is in Angstroms 
        and dictates the sampling rate for the phase shift search.
        """
        self.hklI, self.hklp = OrderedDict(), OrderedDict()
        self.cell_constants = dict()
        self.space_group = space_group
        self.grid_spacing = grid_spacing
        self.fshifts = dict()

        
    def merge_values(self, weighted=True):
        """
        Compute centroid intensity and phase values from self.hklI and self.hklp. For 
        intensities, this is the average; for phases, the intensity-weighted average.
        
        Inputs:
        -------
        weighted: whether to intensity-weight phases, boolean
        
        Outputs:
        --------
        hklI_merge: dict whose keys are Millers and values are averaged intensities
        hklp_merge: dict whose keys are Millers and values are I-weighted avg phases
        """
        assert self.hklI.keys() == self.hklp.keys()
        
        # average intensities, maintaining original Miller ordering
        hklI_merge = OrderedDict((key,np.average(val)) for key,val in self.hklI.items())
                
        # compute (optionally intensity-weighted) phase averages 
        hklp_merge = OrderedDict()
        for miller in self.hklp.keys():
            x,y = 0,0
            for (angle,weight) in zip(self.hklp[miller], self.hklI[miller]):
                if weighted is True:
                    x += np.cos(np.deg2rad(angle)) * weight / np.sum(self.hklI[miller])
                    y += np.sin(np.deg2rad(angle)) * weight / np.sum(self.hklI[miller])
                else:
                    x += np.cos(np.deg2rad(angle)) 
                    y += np.sin(np.deg2rad(angle))
            hklp_merge[miller] = np.rad2deg(np.arctan2(y,x))
            
        return hklI_merge, hklp_merge


    def _compute_presiduals(self, hkl, p_ref, p_tar, weights, fshifts):
        """
        Compute the average residual between target phases (p_tar) shifted by fshifts and 
        reference phases (p_ref).
        
        Inputs:
        -------
        hkl: array of Miller indices
        p_ref: array of reference phases in degrees
        p_tar: array of target phases in degrees
        weights: array of weights, either uniform or mean intensity for that Miller
        fshifts: fractional shifts along (a,b,c) by which to translate phases

        Outputs:
        --------
        p_residual: weighted average residual between shifted and reference phases
        """
        p_shifted = utils.wraptopi(p_tar - 360 * np.dot(hkl, fshifts).ravel())
        diff = p_shifted - p_ref
        diff[diff>180] -= 360
        diff[diff<-180] += 360

        return np.average(np.abs(diff), weights=weights)

    
    def _wrap_presidual(self, args):
        return self._compute_presiduals(*args)


    def _shift_origin(self, hklp_ref, hklp, hklI_ref, hklI, n_processes, weighted):
        """
        Shift phases of added crystal to an origin consistent with the reference phases.
        The best origin is considered the one that minimizes the mean phase residual 
        between the added/target and reference phases, and is identified based on a grid
        search of frequency grid_spacing along the a,b,c crystallographic axes.
        
        Inputs:
        -------
        hklp_ref: dict whose keys are Millers and values are reference phases
        hklp: dict whose keys are Millers and values are phases of crystal to add
        hklI_ref: dict whose keys are Millers and values are reference intensities
        hklI: dict whose keys are Millers and values are intensities of crystal to add
        n_processes: number of processors over which to parallelize task
        weighted: whether to intensity-weight the phase residuals, boolean
        
        Outputs:
        --------
        hklp_shifted: dict whose keys are Millers and values are shifted phases
        """
        # find shared Millers and extract shared information
        hklp_shared = utils.shared_dict(hklp_ref, hklp)
        hkl = np.array(hklp_shared.keys())
        p_ref, p_tar = np.array(hklp_shared.values())[:,0], np.array(hklp_shared.values())[:,1]
        n_crystal = max(self.cell_constants.keys())
        cell = self.cell_constants[n_crystal]
        
        # process intensities for weights, or supply uniform weights if weighted is False
        if weighted is True:
            hklI_shared = utils.shared_dict(hklI_ref, hklI)
            weights = np.mean(np.array(hklI_shared.values()), axis=1)
        else:
            weights = np.ones_like(p_ref)
        
        # locate best origin based on a grid search
        xshifts, yshifts, zshifts = [np.arange(0, cell[i], self.grid_spacing) for i in range(3)]
        fshifts_list = list(itertools.product(xshifts/cell[0], 
                                              yshifts/cell[1], 
                                              zshifts/cell[2]))
        num = len(fshifts_list)
        print "Merging crystals: %i shared reflections, %i grid points" %(len(hkl),num)

        # use multiprocessing to compute residuals for each shift in fshifts_list
        pool = pp.ProcessPool(n_processes)
        args_eval = zip([hkl]*num, [p_ref]*num, [p_tar]*num, [weights]*num, fshifts_list)
        m_grid = pool.map(self._wrap_presidual, args_eval)
        m_grid = np.array(m_grid).reshape((len(xshifts), len(yshifts), len(zshifts)))

        # shift all phases in added crystal to best origin
        xs,ys,zs = np.where(m_grid==m_grid.min())
        if len(xs)>1:
            print "Warning: multiple origins (n=%i) scored equally well; one will be chosen at random" %(len(xs))
            rand_idx = np.random.randint(0, high=len(xs))
            xs, ys, zs = np.array([xs[rand_idx]]), np.array([ys[rand_idx]]), np.array([zs[rand_idx]])

        fshifts = np.array([xshifts[xs]/cell[0],
                            yshifts[ys]/cell[1],
                            zshifts[zs]/cell[2]])
        self.fshifts[n_crystal] = np.squeeze(fshifts)
        hkl, p_vals = np.array(hklp.keys()), np.squeeze(np.array(hklp.values()))
        p_shifted = utils.wraptopi(p_vals - 360 * np.dot(hkl, fshifts).ravel())
        hklp_shifted = OrderedDict((key,val) for key,val in zip(hklp.keys(), p_shifted))    

        print "Minimum mean phase resiudal is %.2f" %(m_grid.min())
        print "Best fractional shifts are: (%.2f, %.2f, %.2f)" %(fshifts[0], fshifts[1], fshifts[2])        
        return hklp_shifted
    

    def _sigma_scale(self, I, qmags, m, b, sigma):
        """
        Perform Wilson factor scaling of input intensities with additional base offset
        and multiplicative factor.
        
        Inputs:
        -------
        I: np.array of intensities to scale
        qmags: np.array of |q_vectors| (inverse resolution), ordered like I
        m: multiplicative factor, float
        b: additive factor, float
        sigma: Wilson scaling parameter, related to B factor
        
        Outputs:
        --------
        I_scaled: scaled intensities 
        """
        return m * np.exp(-1*np.square(qmags * sigma)) * I + b


    def _optimize_scalefactors(self, I_ref, I_tar, qmags):
        """
        Least-squares optimization of parameters to scale I_tar to I_ref using the model:
        I_ref = m * exp(-(q*sigma)^2) * I_tar + b
                
        Inputs:
        -------
        I_ref: np.array of reference intensities
        I_tar: np.array of intensties to scale to reference
        qmags: np.array of |q_vectors| (inverse resolution), ordered like I_ref and I_tar
        
        Outputs:
        --------
        m_f, b_f, sigma_f: best-fit scaling parameters       
        """
        import scipy.optimize
    
        def error(args):
            m,b,sigma = args
            I_scaled = self._sigma_scale(I_tar, qmags, m, b, sigma)
            residuals = np.log10(I_scaled[I_scaled>0]) - np.log10(I_ref[I_scaled>0])
            return residuals
    
        x0 = (np.mean(I_tar)/np.mean(I_ref), 0.3, 1.0)
        res = scipy.optimize.leastsq(error, x0)
        m_f, b_f, sigma_f = res[0]
        I_scaled = self._sigma_scale(I_tar, qmags, m_f, b_f, sigma_f)
    
        residual_s = np.sum(np.abs(np.log10(I_ref[I_scaled>0]) - np.log10(I_tar[I_scaled>0])))
        residual_e = np.sum(np.abs(np.log10(I_ref[I_scaled>0]) - np.log10(I_scaled[I_scaled>0])))
        print "Fitted parameters m, b, sigma are: (%.2f,%.2f,%.2f)" %(m_f, b_f, sigma_f)
        print "Scaling reduced sum of log residuals by %.2f" %(residual_s - residual_e)
        
        return m_f, b_f, sigma_f
    
    
    def _scale_intensities(self, hklI_ref, hklI):
        """
        Scale intensities of added crystal to be consistent with reference intensities.
        
        Inputs:
        -------
        hklI_ref: dict whose keys are Millers and values are reference intensities
        hklI: dict whose keys are Millers and values of are intensities of added crystal
        
        Outputs:
        --------
        hklI_scaled: dict whose keys are Millers and values are scaled phases
        """
        # find shared Millers and extract shared information
        hklI_shared = utils.shared_dict(hklI_ref, hklI)
        hkl = np.array(hklI_shared.keys())
        I_ref, I_tar = np.array(hklI_shared.values())[:,0], np.array(hklI_shared.values())[:,1]
        
        # least-squares optimization to scale I_tar to I_ref
        n_crystal = max(self.cell_constants.keys())
        qmags_sel = 1.0 / utils.compute_resolution(self.space_group,
                                                   self.cell_constants[n_crystal],
                                                   hkl)
        m_f, b_f, sigma_f = self._optimize_scalefactors(I_ref.copy(), I_tar.copy(), qmags_sel)
        
        # scale all intensities and reconstruct into a dictionary
        qmags = 1.0 / utils.compute_resolution(self.space_group,
                                               self.cell_constants[n_crystal],
                                               np.array(hklI.keys()))
        I_scaled = self._sigma_scale(np.array(hklI.values()), qmags, m_f, b_f, sigma_f)
        hklI_scaled = OrderedDict((key,val) for key,val in zip(hklI.keys(), I_scaled))
        
        return hklI_scaled

    
    def add_crystal(self, hklI, hklp, cell_constants, n_processes=1, weighted=True):
        """
        Add another crystal's dataset to exisiting class. Phases of the added crystal are 
        shifted to an origin consistent with the phases already present, while the set of
        added intensities are uniformly scaled to match those already present.
        
        Inputs:
        -------
        hklI: dict whose keys are Miller indices and values are intensities
        hklp: dict whose keys are Miller indices (matching hklI) and values are phases
        cell_constants: tuple of unit cell constants, (a,b,c)   
        n_processes: number of processors over which to parallelize task
        weighted: whether to intensity-weight phase residuals, boolean
        """
        assert hklI.keys() == hklp.keys()
        self.cell_constants[len(self.cell_constants)] = cell_constants
        
        # handle case where this if the first crystal added
        if len(self.hklI) == 0:
            self.hklI = OrderedDict((key,np.array([val])) for key,val in hklI.items() if val>0)
            self.hklp = OrderedDict((key,np.array([val])) for key,val in hklp.items() if key in self.hklI.keys())
            assert self.hklI.viewkeys() == self.hklp.viewkeys()
            
        else:
            # remove any Millers associated with negative intensites from input hklI and hklp
            hklp = OrderedDict((key,val) for key,val in hklp.items() if hklI[key]>0)
            hklI = OrderedDict((key,val) for key,val in hklI.items() if hklI[key]>0)

            # check that there are overlapping Miller indices
            n_overlap = len(set(self.hklp.keys()).intersection(hklp.keys()))
            if n_overlap == 0:
                print "Dataset could not be added: no shared reflections"
                return          

            # shift phases to consistent origin and scale intensities
            hklI_ref, hklp_ref = self.merge_values()
            hklp_shifted = self._shift_origin(hklp_ref, hklp, hklI_ref, hklI, n_processes, weighted)
            hklI_scaled = self._scale_intensities(hklI_ref, hklI)
            
            # add adjusted data to class variables
            assert hklp_shifted.viewkeys() == hklI_scaled.viewkeys()           
            for miller in hklp_shifted.keys():
                if hklI_scaled[miller] > 0:
                    if miller in self.hklp.keys():
                        self.hklp[miller] = np.append(self.hklp[miller], hklp_shifted[miller])
                        self.hklI[miller] = np.append(self.hklI[miller], hklI_scaled[miller])
                    else:
                        self.hklp[miller] = np.array([hklp_shifted[miller]])
                        self.hklI[miller] = np.array([hklI_scaled[miller]])
            
        return


class ReduceCrystals:
    
    """
    Class for reducing data from multiple crystals to the asymmetric unit and easy
    conversion to mtz format. Requires CCTBX.
    """
        
    def __init__(self, hklI, hklp, cell_constants, sg_symbol):
        """
        Initialize class with reflection data and crystal information.

        Inputs:
        -------
        hklI: dict whose keys are Millers and values np.array([intensities])
        hklp: dict whose keys are Millers and values np.array([phases])
        cell_constants: (a,b,c,alpha,beta,gamma)
        sg_symbol: space group in Hermann-Mauguin notation, e.g. "P 21 21 21"
        """
        assert hklI.keys() == hklp.keys()
        self.hklI = hklI
        self.hklp = hklp
        self.cell_constants = cell_constants
        self.sg_symbol = sg_symbol
        self.data = dict() # for storing reduced output
        self.cs = crystal.symmetry(unit_cell = cell_constants, 
                                   space_group_symbol = sg_symbol)
        
    
    def shift_phases(self, fshifts):
        """
        Shift phases in self.hklp along the crystallographic a,b,c axes by the input 
        tuple shifts to relocate the phase origin.
        
        Inputs:
        -------
        shifts: tuple of fractional shifts along the (a,b,c) axes
        """
        for hkl,p_unshifted in self.hklp.items():
            self.hklp[hkl] = utils.wraptopi(p_unshifted - 360 * np.dot(np.array(hkl), fshifts).ravel())

        return
    

    def _indices_asu_hkl(self):
        """
        Map reflections in self.hklp to the asymmetric unit, generating a list of
        reflections that belong to the asymmetric unit and either are observed in
        self.hklp or have at least one symmetry-related reflection in self.hklp.
        
        Outputs:
        --------
        asu_hkl_list: list of Miller indices in asymmetric unit
        """
        # use CCTBX to obtain a list of asymmetric unit reflections
        mock_p = [item[0] for item in self.hklp.values()]
        ma = miller.array(miller_set = miller.set(self.cs,
                                                  flex.miller_index(self.hklp.keys()),
                                                  anomalous_flag=False),
                          data = flex.double(mock_p))
        merged = ma.merge_equivalents().array()        
        asu_hkl_list = list(merged.indices())
        
        return asu_hkl_list
                
    
    def reduce_phases(self, weighted=True):
        """
        Reduce phase information to the asymmetric unit by averaging over data both 
        from symmetry-equivalent reflections and different observations of the same 
        reflection. Optionally weight measurements by their corresponding intensity.
        Estimated phases are stored under "PHIB" in self.data, while their figure of 
        merit, "FOM", is estimated to be 1-std_error(phases).
        
        Inputs:
        -------
        weighted: whether to intensity-weight phases, optional, default is True
        
        Outputs:
        --------
        p_reduced: OrderedDict of reduced phases, with all phases listed
        """
        # identify Millers in asu and expand sym_ops to encompass Friedels
        asu_hkl_list = self._indices_asu_hkl()
        sym_ops_path = os.path.join(os.path.abspath("."), "reference/sym_ops.pickle")
        sym_ops = pickle.load(open(sym_ops_path))[self.sg_symbol]
        sym_ops = utils.sym_ops_friedels(sym_ops)

        # loop over all Millers in the asymmetric unit
        p_reduced, i_reduced, self.sym_hkl_dict = OrderedDict(), OrderedDict(), OrderedDict()
        for asu_hkl in asu_hkl_list:
            related_hkls, p_vals, i_vals = list(), list(), list()
            
            # loop over all symmetry operations for each Miller
            for op in sym_ops.keys():
                R,T = sym_ops[op][:,:-1], sym_ops[op][:,-1]
                sym_hkl = tuple(np.inner(np.array(asu_hkl),R).astype(int))
                
                # exclude reflection if it or its Friedel mate has already been added
                if (sym_hkl not in related_hkls) and (sym_hkl in self.hklp.keys()):
                    
                    # map phase to its value in the asymmetric unit
                    related_hkls.append(sym_hkl)
                    p_sym, i_sym = self.hklp[sym_hkl], self.hklI[sym_hkl]
                    p_base = utils.wraptopi(p_sym - 360 * np.dot(np.array(sym_hkl),T))
                        
                    # multiply phase value by -1 if a Friedel pair
                    if op > max(sym_ops.keys()) / 2:
                        p_base *= -1

                    p_vals.append(p_base)
                    i_vals.append(i_sym)
                        
            p_reduced[asu_hkl] = np.concatenate(p_vals, axis=0)
            i_reduced[asu_hkl] = np.concatenate(i_vals, axis=0)
            self.sym_hkl_dict[asu_hkl] = related_hkls
            
        # add phases and cosine of variance to self.data, optionally intensity-weighted
        if weighted is False:
            self.data['PHIB'] = OrderedDict((key,utils.average_phases(val)) for key,
                                            val in p_reduced.items())
            self.data['FOM'] = OrderedDict((key,1.0 - utils.stderr_phases(val)) for key,
                                           val in p_reduced.items())
        else:
            zip_data = zip(p_reduced.keys(),p_reduced.values(),i_reduced.values())
            self.data['PHIB'] = OrderedDict((k,utils.average_phases(v,weights=w)) for k,v,w in zip_data)
            self.data['FOM'] = OrderedDict((k,utils.stderr_phases(v,weights=w)) for k,v,w in zip_data)

        return p_reduced
        
    
    def reduce_intensities(self):
        """
        Reduce intensity information to the asymmetric unit using CCTBX functions
        to simplify the calculation of sigmas.
        """
        # separate observations of same reflections and convert data to lists
        hkl_list, I_list = list(), list()
        for key, val in self.hklI.items():
            hkl_list.append([key] * len(val))
            I_list.append(val)
        hkl_list = [item for sublist in hkl_list for item in sublist]
        I_list = np.concatenate(I_list, axis=0)
        
        # merge equivalents using CCTBX
        flex_hkl, flex_I = flex.miller_index(hkl_list), flex.double(I_list)
        ma = miller.array(miller_set=miller.set(self.cs,
                                                flex_hkl,
                                                anomalous_flag=False),
                          data=flex_I, sigmas=flex.sqrt(flex_I))
        ma_merged = ma.merge_equivalents().array()
        print "Completeness is %.3f" %(ma_merged.completeness())
        print "Resolution range:", ma_merged.resolution_range()
        
        # add I and sigI to self.data
        indices = list(ma_merged.indices())
        I, sigI = list(ma_merged.data()), list(ma_merged.sigmas())
        
        self.data['IMEAN'] = OrderedDict((key,val) for key,val in zip(indices, I))
        self.data['SIGIMEAN'] = OrderedDict((key,val) for key,val in zip(indices, sigI))
        
        return
    
    
    def generate_mtz(self, save_name = None):
        """
        Convert reduced data stored in self.data to MTZ format and save if a
        path is supplied.
        
        Inputs:
        -------
        save_name: file name for output .MTZ file if saving, optional
        
        Outputs:
        --------
        mtz_object: self.data reformatted as a CCTBX MTZ object 
        """
        # ensure that hkl keys are the same and convert to flex array
        assert self.data['IMEAN'].keys() == self.data['PHIB'].keys()
        indices = flex.miller_index(self.data['IMEAN'].keys())

        # set up column root labels and column types, as specified in
        # http://www.ccp4.ac.uk/html/mtzformat.html
        col_rootlabels = ["IMEAN", "SIGIMEAN", "PHIB", "FOM"]
        col_types = ["J", "Q", "P", "W"]

        # add stored information in self.data to an MTZ object
        mtz_dataset = None
        for crl, ct in zip(col_rootlabels, col_types):
            ma = miller.array(miller_set=miller.set(self.cs,
                                                    indices,
                                                    anomalous_flag=False),
                              data=flex.double(self.data[crl].values()))

            if mtz_dataset is None:
                mtz_dataset = ma.as_mtz_dataset(column_root_label=crl,
                                                column_types=ct)
    
            else:                
                mtz_dataset.add_miller_array(miller_array=ma,
                                             column_root_label=crl, 
                                             column_types=ct)
        
        if save_name is not None:
            mtz_dataset.mtz_object().write(save_name)
        
        return mtz_dataset.mtz_object()


class CompareCrystals:
    
    """
    Class for finding the translations that shift one crystal to the phase origin of another.
    The metric used is the mean phase residual between crystals, evaluated for every set of
    fractional shifts in the unit cell determined by the parameter grid_spacing.
    """

    def __init__(self, cell, hklp, hklI=None):
        """
        Initialize class with reflection data and crystal information.

        Inputs:
        -------
        cell: unit cell constants, (a,b,c,alpha,beta,gamma)
        hklp: dict whose keys are Millers and values np.array([phases])
        hklI: dict whose keys are Millers and values np.array([intensities]), optional
        """
        self.hklp = hklp
        self.cell = np.array(cell)
        self.hklI = hklI    
    
    def _compute_presiduals(self, hkl, p_ref, p_tar, weights, fshifts):
        """
        Compute the average residual between target phases (p_tar) shifted by fshifts and 
        reference phases (p_ref).
        
        Inputs:
        -------
        hkl: array of Miller indices
        p_ref: array of reference phases in degrees
        p_tar: array of target phases in degrees
        weights: array of weights, either uniform or mean intensity for that Miller
        fshifts: fractional shifts along (a,b,c) by which to translate phases

        Outputs:
        --------
        p_residual: (weighted) average residual between shifted and reference phases
        """
        p_shifted = utils.wraptopi(p_tar - 360 * np.dot(hkl, fshifts).ravel())
        diff = p_shifted - p_ref
        diff[diff>180] -= 360
        diff[diff<-180] += 360

        return np.average(np.abs(diff), weights=weights)
    
    
    def _wrap_presidual(self, args):
        """
        Wrapper for _compute_presiduals class to enable passing objects to pool.
        """
        return self._compute_presiduals(*args)
    
    
    def grid_shift(self, hklp_ref, grid_spacing, n_processes, hklI_ref=None):
        """
        Shift phases of self.hkp at intervals of grid_spacing and assess the correlation
        to phases of hklp_ref at every grid position. If hklI_ref is given, then use the
        intensity-weighted phases for the calculaton.
        
        Inputs:
        -------
        hklp_ref: dict with keys are Millers and values are phases in degrees
        grid_spacing: search interval in Angstroms
        n_processes: number of CPU processors at program's disposal
        hklI_ref: dict with keys as Millers and values as intensities, optional
        
        Outputs:
        --------
        m_grid: grid of phase residuals between reference and target phases
        hklp_shifted: dictionary whose keys are Millers and values are shifted phases
        fshifts: tuple corresponding to the best fractional shift
        """
        
        # find shared Millers and extract common phase information
        hklp_shared = utils.shared_dict(hklp_ref, self.hklp)
        hkl = np.array(hklp_shared.keys())
        p_ref = np.squeeze(np.array(hklp_shared.values())[:,0]) 
        p_tar = np.squeeze(np.array(hklp_shared.values())[:,1])
        
        # process intensities for weights, or supply uniform weights if weighted is False
        if (hklI_ref is not None) and (self.hklI is not None):
            hklI_shared = utils.shared_dict(hklI_ref, self.hklI)
            assert hklI_shared.keys() == hklp_shared.keys()
            weights = np.squeeze(np.mean(np.array(hklI_shared.values()), axis=1))
        elif (hklI_ref is not None) and (self.hklI is None):
            print "Warning: no weights will be used, since none were input in setting up class"
            weights = np.ones_like(p_ref)
        else:
            weights = np.ones_like(p_ref)
            
        # perform grid shifts and assess CC
        xshifts, yshifts, zshifts = [np.arange(0, self.cell[i], grid_spacing) for i in range(3)]
        fshifts_list = list(itertools.product(xshifts/self.cell[0], 
                                              yshifts/self.cell[1], 
                                              zshifts/self.cell[2]))
        num = len(fshifts_list)
        print "Comparing crystals: %i shared reflections, %i grid points" %(len(hkl),num)
        
        # evaluate CC at each shift
        pool = pp.ProcessPool(n_processes)
        args_eval = zip([hkl]*num, [p_ref]*num, [p_tar]*num, [weights]*num, fshifts_list)
        m_grid = pool.map(self._wrap_presidual, args_eval)
        m_grid = np.array(m_grid).reshape((len(xshifts), len(yshifts), len(zshifts)))
        
        # shift self.hklp phases based on best position
        xs,ys,zs = np.where(m_grid==m_grid.min())
        fshifts = np.squeeze(np.array([xshifts[xs]/self.cell[0], 
                                       yshifts[ys]/self.cell[1], 
                                       zshifts[zs]/self.cell[2]]))
        hkl_all, p_all = np.array(self.hklp.keys()), np.squeeze(np.array(self.hklp.values()))
        p_shifted = utils.wraptopi(p_all - 360 * np.dot(hkl_all, fshifts).ravel())
        hklp_shifted = OrderedDict((key,np.array([val])) for key,val in zip(self.hklp.keys(), p_shifted))
        
        print "Best fractional shifts are: (%.2f, %.2f, %.2f)" %(fshifts[0], fshifts[1], fshifts[2])
        print "Minimum mean phase residual is %.2f" %(m_grid.min())
        return m_grid, hklp_shifted, fshifts
