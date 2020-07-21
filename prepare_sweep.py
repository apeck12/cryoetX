from dxtbx.datablock import DataBlockFactory, DataBlockDumper
from dxtbx.imageset import ImageSetFactory, ImageSweep, ImageSetData
from dxtbx.model.goniometer import GoniometerFactory
from dxtbx.model.scan import ScanFactory
import glob, os, sys

"""
Modification of AB's prepare_sweep.py.
Usage: libtbx.python prepare_sweep.py img_dir start_img, end_img savefile
"""

root = sys.argv[1]
start, end = int(sys.argv[2]), int(sys.argv[3])

g = GoniometerFactory.single_axis()
s = ScanFactory.make_scan((start,end), 0, (0,1), [0]*(end - start + 1))
sw = ImageSetFactory.from_template(template = os.path.join(root, "fft_frame_I_mf_####.cbf"), 
                                   scan = s, goniometer = g, image_range = (start,end))
dump = DataBlockDumper(DataBlockFactory.from_imageset(sw))
dump.as_file(sys.argv[4])
