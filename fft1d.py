from __future__ import division
from libtbx import easy_pickle
from dxtbx.datablock import DataBlockFactory
from dials.array_family import flex
from dials.algorithms.indexing.fft1d import candidate_basis_vectors_fft1d
from rstbx.phil.phil_preferences import indexing_api_defs
from dials.algorithms.indexing.indexer import indexer_base, master_params
from cctbx.sgtbx import space_group_info
import iotbx.phil
import os, sys
from dxtbx.model import Crystal

"""
Usage: libtbx.python fft1d.py strong.pickle sweep.json savepath
"""

hardcoded_phil = iotbx.phil.parse(
  input_string=indexing_api_defs).extract()

#pixel size is 109 nm = 1090 angstroms
reciprocal_pixel_size = 1.0/1090
spacegroup = raw_input('Space group : ')
magnification = float(raw_input('Magnification : '))
length = float(raw_input('Tomogram dimension : '))/2.0
threshold = float(raw_input('IsigI threshold for retaining peak: '))

# Load the data and map from pixel reciprocal space coordinates to actual reciprocal space coordinates
strong = easy_pickle.load(sys.argv[1])
datablock = DataBlockFactory.from_json_file(sys.argv[2])
isigi = strong['intensity.sum.value']/flex.sqrt(strong['intensity.sum.variance'])
strong = strong.select(isigi >= threshold)
x, y, z = (strong['xyzobs.px.value']).parts()
x = (x-length) * reciprocal_pixel_size * magnification
y = (y-length) * reciprocal_pixel_size * magnification
z = (z-length) * reciprocal_pixel_size * magnification

strong['rlp'] = flex.vec3_double(x, y, z)

# get candidates
solutions = candidate_basis_vectors_fft1d(strong['rlp'], hardcoded_phil)
#print solutions

sgi = space_group_info(spacegroup)
master_params.indexing.known_symmetry.space_group = sgi
master_params.indexing.basis_vector_combinations.max_combinations = 5
master_params.refinement.reflections.outlier.algorithm = None
indexer = indexer_base(strong, datablock[0].extract_imagesets(), master_params)
crystals = indexer.find_candidate_orientation_matrices(solutions[0])
crystal = crystals[0] # Warning, we pick here the first one, but it is possible the others should be examined
print "Before applying symmetry"
print crystal
crystal, cob = indexer.apply_symmetry(crystal, sgi.group())
print "After applying symmetry"
print crystal

# Index the reflections
from dxtbx.model.experiment_list import ExperimentListFactory, ExperimentListDumper
from dials.algorithms.indexing import index_reflections
experiments = ExperimentListFactory.from_datablock_and_crystal(datablock, crystal)
strong['xyzobs.mm.value'] = strong['xyzobs.px.value']
strong['imageset_id'] = flex.int(len(strong), 0)
strong['id'] = flex.int(len(strong), -1)
index_reflections(strong, experiments)
print "Indexed", (strong['id'] == 0).count(True), "out of", len(strong)
dump = ExperimentListDumper(experiments)
dump.as_file(sys.argv[3] + "indexed.json")
strong.as_pickle(sys.argv[3] + "indexed.pickle")




