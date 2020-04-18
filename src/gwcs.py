import os

from astropy.io import fits
from astropy import wcs


path = '../../astrophotography_data/MilkyWayPrettyBoy/wcs/'


wcs_file = os.path.join(path, 'wcs.fits')
f = fits.open(wcs_file)
print(f)

axy_file = os.path.join(path, 'axy.fits')
f = fits.open(axy_file)
print(f)

corr_file = os.path.join(path, 'corr.fits')
f = fits.open(corr_file)
print(f)

rdls_file = os.path.join(path, 'rdls.fits')
f = fits.open(rdls_file)
print(f)
