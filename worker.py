import numpy as np

# Comparison functions used to figure out which stars are closest to a given pixel
gt_zero = lambda x: x > 0.0
lt_zero = lambda x: x < 0.0


# Gets the index of the closest star in table.
# The differences are in the sense pixel - star centroid.
# The comparison functions define from which quadrant the star is drawn from.
def closest(diff_x, diff_y, compare_x, compare_y):
    # Compute mask that masks out everything that is outside
    # the quadrant defined by the comparison functions
    mask_x = np.where(compare_x(diff_x), 1, 0)
    mask_y = np.where(compare_y(diff_y), 1, 0)
    mask = mask_x * mask_y

    # Get index of star at minimum distance
    distance = np.sqrt((diff_x * diff_x + diff_y * diff_y)) * mask
    if np.nonzero(distance)[0].size > 0:
        mindist = np.min(distance[np.nonzero(distance)])
        index = np.where(distance == mindist)[0][0]
        return index, mindist
    else:
        return -1, 0.0


class Worker:
    '''
    A class with callable instances that execute the offset calculation
    algorithm over a section of the input image.

    It provides the callable for the `Pool.apply_async` function, and also
    holds all parameters necessary to perform the calculation.

    The 'step' parameters help save time by allowing the algorithm to work
    only on pixels separated by 'step' (in both X and Y). The remaining pixels
    are filled elsewhere by interpolation with a 9x9 Gaussian kernel.
    '''
    def __init__(self, x0, y0, size_x, size_y, step_x, step_y, centroid_x, centroid_y,
                offset_x, offset_y):
        '''
        Parameters:

        x0, y0 - top left pixel of the image section designated for this instance
        size_x, size_y - size of the image section
        step_x - step used in the x direction when looping over pixels
        step_y - step used in the y direction when looping over pixels
        centroid_x - 1-D data from the `xcentroid` column in the offsets table
        centroid_y - 1-D data from the `ycentroid` column in the offsets table
        offset_x - 1-D data from the `xoffset` column in the offsets table
        offset_y - 1-D data from the `yoffset` column in the offsets table

        Returns:

        dict with output arrays. To be collected by a callback function.
        '''
        self.x0 = x0
        self.y0 = y0
        self.size_x = size_x
        self.size_y = size_y
        self.step_x = step_x
        self.step_y = step_y

        self.centroid_x = centroid_x
        self.centroid_y = centroid_y
        self.offset_x = offset_x
        self.offset_y = offset_y

        # create local output arrays. These have the shape of one single
        # section of the entire image. Once filled up, they are returned
        # to a callback function that takes care of storing them into
        # the appropriate section of the result arrays.
        self.offset_array_x = np.zeros(shape=(self.size_y, self.size_x))
        self.offset_array_y = np.zeros(shape=(self.size_y, self.size_x))

    def __call__(self):

        range_i = list(range(0, self.size_x, self.step_x))
        range_j = list(range(0, self.size_y, self.step_y))

        for i in range_i:
            for j in range_j:

                pixel_x = int(i + self.x0)
                pixel_y = int(j + self.y0)

                diff_x = pixel_x - self.centroid_x
                diff_y = pixel_y - self.centroid_y

                index = np.array(range(4), dtype=int)
                dist  = np.array(range(4), dtype=float)

                # get index and distance of the closest star, one per quadrant
                index[0], dist[0] = closest(diff_x, diff_y, gt_zero, gt_zero)
                index[1], dist[1] = closest(diff_x, diff_y, lt_zero, gt_zero)
                index[2], dist[2] = closest(diff_x, diff_y, gt_zero, lt_zero)
                index[3], dist[3] = closest(diff_x, diff_y, lt_zero, lt_zero)

                # weighted average of the offset values. The weight is the inverse
                # distance pixel-star. Beware of zeroed or non-existent distances.
                sumweights = 0.0
                for k in range(len(dist)):
                    if dist[k] > 0.:
                        sumweights += 1./dist[k]

                weighted_offset_x = 0.0
                weighted_offset_y = 0.0

                for k in range(len(index)):
                    if index[k] > 0:
                        weighted_offset_x += self.offset_x[index[k]] * (1./dist[k] / sumweights)
                        weighted_offset_y += self.offset_y[index[k]] * (1./dist[k] / sumweights)

                self.offset_array_x[j][i] = weighted_offset_x
                self.offset_array_y[j][i] = weighted_offset_y

        # return the local output arrays with offsets for this section of the image,
        # plus metadata to locate the section on the offsets arrays.

        return {'x0': self.x0,
                'y0': self.y0,
                'size_x': self.size_x,
                'size_y': self.size_y,
                'offset_array_x': self.offset_array_x,
                'offset_array_y': self.offset_array_y
               }
# end of Worker class definition

