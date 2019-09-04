from math import sqrt

import numpy as np
from scipy.stats import multivariate_normal

from contour_visualization.Distribution import Distribution

import logging

logger = logging.getLogger(__name__)


def learn_gaussian_from_model(model):
    if hasattr(model.model, "data"):
        mean = np.mean(
            model.model.data.loc[model.model.data[model.categorical_attribute] == model.used_categorical_name].loc[:,
            model.model.data.columns != 'species'].values, axis=0)
        cov = np.cov(
            model.model.data.loc[model.model.data[model.categorical_attribute] == model.used_categorical_name].loc[:,
            model.model.data.columns != 'species'].values, rowvar=False)
        logger.debug("Mean: {}".format(mean))
        logger.debug("Cov-matrix: {}".format(cov))
        return Gaussian(means=mean,
                        cov_matrix=cov)


def get_gaussian_from_list(dist_list):
    """
    creates a gaussian-distribution by an array of values

    :param dist_list: [x_min, x_max, y_min, y_max, means, cov_matrix, size]
    :return: Gaussian
    """
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
    """
    Generates a multivariate normal distribution of dimension 2.
    """

    def __init__(self, means=None, cov_matrix=None, *args, **kwargs):
        """
        Generates a multivariate normal distribution of dimension 2.
        Rest can be set like in the abstract distribution class

        :param means: means / expectations of the distribution
        :param cov_matrix: covariance matrix of the distribution
        """
        super().__init__(*args, **kwargs)

        if means is None:
            means = [0, 0]
            logger.warn("No means defined using default {}".format(means))
        elif len(means) != 2:
            raise ValueError("len({}) != 2".format(means))
        self.means = means

        if cov_matrix is None:
            cov_matrix = [[1, 0], [0, 1]]
            logger.warn("No covariance-matrix defined. Using default {}".format(cov_matrix))
        elif len(cov_matrix) != 2 and any([len(cov_matrix[i]) != 2 for i in [0, 1]]):
            raise ValueError("{}.shape != (2,2)".format(cov_matrix))
        self.cov_matrix = cov_matrix
        self.gau = multivariate_normal(self.means, self.cov_matrix)
        if "x_min" not in kwargs.keys():
            min_values = -5 * sqrt(cov_matrix[0][0]) + means[0], -5 * sqrt(cov_matrix[1][1]) + means[1]
            max_values = 5 * sqrt(cov_matrix[0][0]) + means[0], 5 * sqrt(cov_matrix[1][1]) + means[1]
        else:
            min_values = kwargs["x_min"], kwargs["y_min"]
            max_values = kwargs["x_max"], kwargs["y_max"]
        self.x_min = min_values[0]
        self.x_max = max_values[0]
        self.y_min = min_values[1]
        self.y_max = max_values[1]

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
