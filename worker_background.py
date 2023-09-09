import datetime

import numpy as np

from astropy.io import fits
from astropy.convolution import interpolate_replace_nans
from photutils.background import Background2D

import rawpy


class Worker:
    '''
    A class with callable instances that execute the background calculation
    algorithm over images in a list.
    
    Input is a list of .ARW files, output is a list of FITS files.

    It provides the callable for the `Pool.apply_async` function, and also
    holds all parameters necessary to perform the calculation.
    '''
    def __init__(self, image_list, input_suffix, output_suffix, 
                 red_mask, green_mask, blue_mask,
                 red_norm, blue_norm, bkg_cell_footprint, bkg_filter, 
                 bkg_sigma_clip, bkg_kernel, bkg_estimator):

        self.image_list = image_list
        
        self.input_suffix = input_suffix
        self. output_suffix = output_suffix

        self.red_norm = red_norm
        self.blue_norm = blue_norm
        
        self.red_mask = red_mask
        self.green_mask = green_mask
        self.blue_mask = blue_mask
        
        self.bkg_cell_footprint = bkg_cell_footprint
        self.bkg_filter = bkg_filter
        self.bkg_sigma_clip = bkg_sigma_clip
        self.bkg_kernel = bkg_kernel
        self.bkg_estimator = bkg_estimator

    def __call__(self):

        for image_name in self.image_list:
            raw = rawpy.imread(image_name)
            imarray = raw.raw_image_visible.astype(float)

            subtracted = self.subtract_background(imarray,red_norm=self.red_norm, blue_norm=self.blue_norm)

            self.write_bkg_subtracted(image_name, subtracted)


    def subtract_background(self, imarray, red_norm=1.0, blue_norm=1.0):
        # red_norm and blue_norm are normalization parameters applied to the R and B bands (assume
        # G=1) in order to make the star images as well-behaved as possible, in terms of being
        # well represented, on average, by the daofind Gaussian. Ideally a different normalization
        # should be applied to each star, depending on its color index, but this will be left as
        # a possible (but not very likely) future improvement. For now, we assume that an average,
        # frame-wide single normalization should suffice (statistically).

        # separate color bands
        red_array = imarray * self.red_mask
        green_array = imarray * self.green_mask
        blue_array = imarray * self.blue_mask

        # interpolate over the masked pixels in each band, so the background estimator
        # is presented with a smooth array entirely filled with valid data
        red_array[red_array == 0.0] = np.nan
        green_array[green_array == 0.0] = np.nan
        blue_array[blue_array == 0.0] = np.nan

        red_array = interpolate_replace_nans(red_array, self.bkg_kernel)
        green_array = interpolate_replace_nans(green_array, self.bkg_kernel)
        blue_array = interpolate_replace_nans(blue_array, self.bkg_kernel)

        red_array[np.isnan(red_array)] = 0.
        green_array[np.isnan(green_array)] = 0.
        blue_array[np.isnan(blue_array)] = 0.

        # fit background model to each smoothed-out color band
        red_bkg = Background2D(red_array, self.bkg_cell_footprint, 
                               filter_size=self.bkg_filter, 
                               sigma_clip=self.bkg_sigma_clip,
                               bkg_estimator=self.bkg_estimator)
        green_bkg = Background2D(green_array, self.bkg_cell_footprint, 
                               filter_size=self.bkg_filter, 
                               sigma_clip=self.bkg_sigma_clip,
                               bkg_estimator=self.bkg_estimator)
        blue_bkg = Background2D(blue_array, self.bkg_cell_footprint, 
                               filter_size=self.bkg_filter, 
                               sigma_clip=self.bkg_sigma_clip,
                               bkg_estimator=self.bkg_estimator)

        # subtract background from each masked color array
        subtracted = imarray - red_bkg.background * self.red_mask - \
                     green_bkg.background * self.green_mask - \
                     blue_bkg.background * self.blue_mask

        # after background subtraction, apply color band normalization. This has to be done separately
        # from step above for the background on each band to remain zero on average.
        subtracted = (subtracted * self.red_mask * red_norm) + \
                     (subtracted * self.green_mask) + \
                     (subtracted * self.blue_mask * blue_norm)

        return subtracted


    def write_bkg_subtracted(self, image_name, subtracted):
        today = datetime.datetime.now().ctime()

        fits_name = image_name.replace(self.input_suffix, self.output_suffix)

        print("Writing bkg-subtracted image: ", fits_name, today, flush=True)

        # Create FITS 32-bit floating point file with data array
        hdr = fits.Header()
        hdr['DATE'] = today
        hdr['PATH'] = fits_name
        primary_hdu = fits.PrimaryHDU(header=hdr)

        hdu = fits.ImageHDU(subtracted.astype('float32'))

        hdul = fits.HDUList([primary_hdu, hdu])
        hdul.writeto(fits_name, overwrite=True)

        return fits_name