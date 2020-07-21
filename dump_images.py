from __future__ import division
import mrcfile, time
import numpy as np
import sys, math, os
from dxtbx.format.FormatCBFMini import FormatCBFMini
from dxtbx.model.detector import DetectorFactory
from dxtbx.model.beam import BeamFactory
from scitbx.array_family import flex
from scitbx import fftpack
from numpy import fft
from scipy.ndimage.filters import gaussian_filter
from dxtbx.model import ScanFactory

"""
Modification of AB's original script dump_images.py. Command line input determines whether
1. Tukey filtering is applied to mrc file and/or 2. Gaussian filtering is applied to amps
and phases, with tunable value of standard deviation

Example usage, running from directory where cbf files will be written:
libtbx.python dump_images.py data.mrc none 2
The above applies only a Gaussian filter with a standard deviation of 2.
"""

def apply_tukey_3d(data_volume):
  """ Generate a 3d-symmetric Tukey kernel. NB: Overall sum of intensities
  is not preserved as currently implemented. """
  print "Applying Tukey filter"
  import scipy.signal
  shape = data_volume.shape
  k = scipy.signal.tukey(shape[0]) # assuming a cubic volume
  k3d = k[:,np.newaxis,np.newaxis] * k[np.newaxis,:,np.newaxis] * k[np.newaxis,np.newaxis,:]
  return k3d * data_volume

def load_fft(filename, Tukey):
  print "Opening file"
  with mrcfile.open(filename) as mrc:
    print "Converting to flex"
    # coerce to double
    if Tukey is True:
      mod_vol = apply_tukey_3d(mrc.data.astype(np.float64))
      realpart = flex.double(mod_vol)
    else:
      realpart = flex.double(mrc.data.astype(np.float64))
    complpart = flex.double(flex.grid(realpart.focus()))
    C3D = flex.complex_double(realpart,complpart)

    print "C3Dfocus",C3D.focus()
    print C3D
    FFT = fftpack.complex_to_complex_3d((C3D.focus()[0],C3D.focus()[1],C3D.focus()[2]))
    complex_flex = FFT.forward(C3D)
    print complex_flex.focus()
    print complex_flex

  return complex_flex

def get_intensity_and_phase(complex_flex):
  real, imag = complex_flex.parts()
  intensity = real**2 + imag**2
  intensity /= 1e10
  print "Shifting intensity..."
  intensity = flex.double(fft.fftshift(intensity.as_numpy_array()))
  print intensity.focus()

  phase = flex.atan2(imag, real)*180/math.pi
  print "Shifting phases..."
  phase = flex.double(fft.fftshift(phase.as_numpy_array()))
  print phase.focus()
  return intensity, phase

def make_images(data, tag):
  pixel_size = 0.1 # mm/pixel
  detector = DetectorFactory.simple(
      'PAD', 100, (pixel_size*data.focus()[1]/2,pixel_size*data.focus()[2]/2), '+x', '-y',(pixel_size, pixel_size),
      (data.focus()[2], data.focus()[1]), (-1, 1e6-1), [],
      None)
  beam = BeamFactory.simple(1.0)
  sf = ScanFactory()
  scan = sf.make_scan(image_range = (1,180),
                      exposure_times = 0.1,
                      oscillation = (0, 1.0),
                      epochs = range(180),
                      deg = True)

  # write images in each of three directions
  for slice_id in [0,1,2]:
    for idx in xrange(data.focus()[slice_id]):
      if slice_id == 0: # slow
        data_slice = data[idx:idx+1,:,:]
        data_slice.reshape(flex.grid(data.focus()[1], data.focus()[2]))
        filename = "fft_frame_%s_mf_%04d.cbf"%(tag, idx)
      elif slice_id == 1: # med
        data_slice = data[:,idx:idx+1,:]
        data_slice.reshape(flex.grid(data.focus()[0], data.focus()[2]))
        filename = "fft_frame_%s_sf_%04d.cbf"%(tag, idx)
      elif slice_id == 2: # fast
        data_slice = data[:,:,idx:idx+1]
        data_slice.reshape(flex.grid(data.focus()[0], data.focus()[1]))
        filename = "fft_frame_%s_sm_%04d.cbf"%(tag, idx)
      print ['slow', 'med', 'fast'][slice_id], idx
      FormatCBFMini.as_file(detector,beam,None,scan,data_slice, filename)


if __name__ == '__main__':

  start_time = time.time()

  # load input arguments
  mrcfilename = sys.argv[1]
  apply_Tukey = False
  if sys.argv[2] == "tukey":
    apply_Tukey = True
  sigma = float(sys.argv[3])
  if sigma == 0: sigma = None
  
  fft_flex = load_fft(mrcfilename, Tukey = apply_Tukey)
  gaussian_filter_sigma = sigma # Change to None to remove filter; usually use value of 2

  for tag, dataset in zip(['I', 'phase'], get_intensity_and_phase(fft_flex)):
    from scipy.ndimage.filters import gaussian_filter
    print 'before apply sig', gaussian_filter_sigma, flex.mean(dataset)
    dataset = flex.double(gaussian_filter(dataset.as_numpy_array(), gaussian_filter_sigma))
    print 'after', flex.mean(dataset)
    make_images(dataset.iround(), tag)

  print "elapsed time is %.4f" %((time.time() - start_time)/60.0)
