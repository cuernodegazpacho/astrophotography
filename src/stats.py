import glob

import rawpy
import numpy as np
from numpy import unravel_index


path = '../../astrophotography_data/MilkyWayPrettyBoy/12800/dark/'

list_p = glob.glob(path + '/*.ARW')

for fname in list_p:

    raw = rawpy.imread(fname)
    array = raw.raw_image
    raw_min = np.min(array)
    raw_max = np.max(array)
    raw_mean = np.mean(array)
    raw_std = np.std(array)

    pos_min = unravel_index(array.argmin(), array.shape)
    pos_max = unravel_index(array.argmax(), array.shape)

    print('{:s}  {:s} {:s}  {:5d} {:5d}  {:7.2f} {:6.2f}'.format(str(raw.black_level_per_channel),
        str(pos_min), str(pos_max),
        raw_min, raw_max, raw_mean, raw_std))

# raw.black_level_per_channel is the bias level. In
# short exposures dark is not statistically significant,
# as expected. Lets look for systematic patterns.

summed_array = None

for fname in list_p:

    raw1 = rawpy.imread(fname)
    array = raw.raw_image

    if summed_array is None:
        summed_array = np.zeros(shape=array.shape, dtype=float)

    summed_array += array

summed_array /= len(list_p)

print('SUM:')
raw_min = np.min(summed_array)
raw_max = np.max(summed_array)
raw_mean = np.mean(summed_array)
raw_std = np.std(summed_array)

pos_min = unravel_index(summed_array.argmin(), summed_array.shape)
pos_max = unravel_index(summed_array.argmax(), summed_array.shape)

print('{:f} {:f}  {:7.2f} {:6.2f}'.format(raw_min, raw_max, raw_mean, raw_std))



