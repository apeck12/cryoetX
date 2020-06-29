import cPickle as pickle
import numpy as np
import sys

"""
Helper script to construct a dictionary containing useful information about
values used during fft1d.py. Useful for downstream operations.
Usage: python make_specs.py savename.pickle
"""

specs = dict()
specs['mag'] = float(raw_input('Magnification : '))
specs['px_size'] = float(raw_input('Pixel size : '))
specs['res'] = float(raw_input('Resolution : '))
length = int(raw_input('Tomogram dimension : '))
specs['shape'] = (length, length, length)
specs['angle'] = float(raw_input('Max. tilt angle : '))
specs['increment'] = float(raw_input('Tilt increment : '))

print [(key,val) for key,val in specs.items()]

with open(sys.argv[1], "wb") as handle:
    pickle.dump(specs, handle)
