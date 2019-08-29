import numpy as np
from scipy.stats import multivariate_normal

from contour_visualization.Distribution import Distribution

import logging

logger = logging.getLogger(__name__)


def get_gaussian_from_list(dist_list):
    if len(dist_list) == 7:
        return Gaussian(x_min=dist_list[0],
                        x_max=dist_list[1],
                        y_min=dist_list[2],
                        y_max=dist_list[3],
                        means=dist_list[4],
                        cov_matrix=dist_list[5],
                        size=dist_list[6])
    else:
        raise ValueError("Expected length of 7 instead got {}".format(len(dist_list)))


class Gaussian(Distribution):
    def __init__(self, means=None, cov_matrix=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if means is None:
            means = [0, 0]
        self.means = means

        if cov_matrix is None:
            cov_matrix = [[1, 0], [0, 1]]
        self.cov_matrix = cov_matrix

        self.gau = multivariate_normal(self.means, self.cov_matrix)
        logger.debug(self)

    def get_density(self, x):
        return self.weight * self.gau.pdf(x)

    def get_density_grid(self, size=None, x_min=None, x_max=None, y_min=None, y_max=None):
        if x_min is None:
            x_min = self.x_min
        if x_max is None:
            x_max = self.x_max
        if y_min is None:
            y_min = self.y_min
        if y_max is None:
            y_max = self.y_max
        if size is None:
            size = self.size
        x_list = np.linspace(x_min, x_max, size)
        y_list = np.linspace(y_min, y_max, size)
        x, y = np.meshgrid(x_list, y_list)
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        return x, y, self.weight * self.gau.pdf(pos)

    def get_attributes(self):
        return [self.x_min, self.x_max, self.y_min, self.y_max, self.means, self.cov_matrix, self.weight, self.size]

    def __str__(self):
        return "[{}, {}, {}, {}, {}, {}, {}, {}]".format(self.x_min, self.x_max, self.y_min, self.y_max, self.means,
                                                         self.cov_matrix, self.weight, self.size)
