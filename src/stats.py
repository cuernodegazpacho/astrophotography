import glob, os

import rawpy
import numpy as np
from numpy import unravel_index


# path = '../../astrophotography_data/exposure_tests/tests_1/I6400_10.ARW'
# path = '../../astrophotography_data/exposure_tests/tests_1/I1600_10.ARW'
# path = '../../astrophotography_data/MilkyWayPrettyBoy/12800/light/DSC03770.ARW'
# path = '../../astrophotography_data/dynamic_range/25s_ISO1600_electronic/DSC03434.ARW'
# path = '../../astrophotography_data/MilkyWayPrettyBoy/800/light/DSC03532.ARW'

# path = '../../astrophotography_data/MilkyWayPrettyBoy/6400/bias/DSC03892.ARW'
path = '../../astrophotography_data/MilkyWayPrettyBoy/6400/dark/DSC03862.ARW'

raw = rawpy.imread(path)
print(raw)
array = raw.raw_image

print(array)

raw_min = np.min(array)
raw_max = np.max(array)

pos_min = unravel_index(array.argmin(), array.shape)
pos_max = unravel_index(array.argmax(), array.shape)

print(raw_min, raw_max, pos_min, pos_max)



