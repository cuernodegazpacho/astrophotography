import os

from astropy.io import fits

path = '../../astrophotography_data/MilkyWayPrettyBoy/wcs/'

wcs_file = os.path.join(path, 'wcs.fits')

f = fits.open(wcs_file)

print(f)