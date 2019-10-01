import random
from collections import namedtuple
from typing import List

import numpy as np
from scipy.stats import multivariate_normal

from contour_visualization import picture_contours, color_schemes
from contour_visualization.Distribution import Distribution

import logging

logger = logging.getLogger(__name__)
c_handler = logging.StreamHandler()
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)


def normalize_array(X, old_min, old_max, new_min, new_max):
    X = np.asarray(X)
    if old_min != old_max:
        return (X - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min
    else:
        logger.warning("min == max | {}=={}".format(old_min, old_max))
        return []


def normalize_2d_array(X, x_min_old, x_max_old, x_min_new, x_max_new, y_min_old=None, y_max_old=None, y_min_new=None,
                       y_max_new=None):
    if y_min_old is None:
        y_min_old = x_min_old
    if y_max_old is None:
        y_max_old = x_max_old
    if y_min_new is None:
        y_min_new = x_min_new
    if y_max_new is None:
        y_max_new = x_max_new
    x_shape = X.shape
    splite = x_shape[0]
    x_flatt = X.flatten("F")
    x_flatt[:splite] = normalize_array(x_flatt[:splite], x_min_old, x_max_old, x_min_new, x_max_new)
    x_flatt[splite:] = normalize_array(x_flatt[splite:], y_min_old, y_max_old, y_min_new, y_max_new)
    return np.reshape(x_flatt, x_shape, order="F")


class Gaussian():
    def __init__(self, mean, cov):
        self.gau = multivariate_normal(mean, cov)

    def get(self, x, y):
        return self.gau.pdf([x, y])


def get_gaussian(x_min, x_max, y_min, y_max, mean=None, cov=None, size=500000):
    """
    returns a gaussian-distribution

    :param cov:
    :param mean:
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :param size:
    :return:
    """
    if mean is None:
        mean = [0, 0]
    if cov is None:
        cov = [[0, 0], [0, 0]]
    x_list = np.linspace(x_min, x_max, size)
    y_list = np.linspace(y_min, y_max, size)
    x, y = np.meshgrid(x_list, y_list)
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    z = multivariate_normal(mean, cov)
    return x, y, z.pdf(pos)


def get_random_gaussian(x_min, x_max, y_min, y_max, variance_min, variance_max, size, scale_factor=1.):
    """
    generates a random gaussian inbetween the given min and max values

    :param x_min: minimal x-value for the x-expectation
    :param x_max: maximal x-value for the x-expectation
    :param y_min: minimal y-value for the y-expectation
    :param y_max: maximal y-value for the y-expectation
    :param variance_min: minimal variance of the gaussian
    :param variance_max: maximal variance of the gaussian
    :param size: shape of the 2D-gaussian (size*size)
    :param scale_factor: scalar for x_min, x_max, y_min, y_max
    :return: 2D-gaussian (size, size)
    """
    mu_x_1 = random.randint(int(x_min * scale_factor), int(x_max * scale_factor))
    mu_y_1 = random.randint(int(y_min * scale_factor), int(y_max * scale_factor))
    cov = np.matrix([[random.randint(variance_min, variance_max), random.randint(variance_min, variance_max)],
                     [random.randint(variance_min, variance_max), random.randint(variance_min, variance_max)]])
    cov = np.dot(cov, cov.transpose())
    return get_gaussian(x_min, x_max, y_min, y_max, [mu_x_1, mu_y_1], cov, size)


Limits = namedtuple("Limits", ["x_min", "x_max", "y_min", "y_max"])


def get_limits(distributions):
    return Limits(*get_x_values(distributions), *get_y_values(distributions))


def generate_distribution_grids(distributions: List[Distribution], x_min=None, x_max=None, y_min=None, y_max=None):
    if x_min is None:
        x_min, x_max = get_x_values(distributions)
    if y_min is None:
        y_min, y_max = get_y_values(distributions)
    return [distribution.get_density_grid(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)[2] for distribution in
            distributions]


def generate_gaussians_old(gaussians):
    return [get_gaussian(*gaussian)[2] for gaussian in gaussians]


def get_x_values(distributions: List[Distribution]):
    return min([dist.x_min for dist in distributions]), max([dist.x_max for dist in distributions])


def get_y_values(distributions: List[Distribution]):
    return min([dist.y_min for dist in distributions]), max([dist.y_max for dist in distributions])


def generate_gaussians_xyz(gaussians):
    x_list, y_list, z_list = [], [], []
    for gaussian in gaussians:
        x, y, z = get_gaussian(*gaussian)
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
    return x_list, y_list, z_list


def generate_random_gaussians(num=2, x_min=-10, x_max=10, y_min=-10, y_max=10, variance_min=2, variance_max=10,
                              size=200,
                              scale_factor=0.6):
    return [get_random_gaussian(x_min, x_max, y_min, y_max, variance_min, variance_max, size, scale_factor)[2] for _ in
            range(num)]


def generate_weights(z_values):
    """
    calculates minimum, maximum and sum of a given list of weights.
    :param z_values: [2D-weights_1, ... ,2D-weights_n]
    :return: minimum value of all weights, max of all weights, 2D-matrix with sum of each pixel of 2D-weights
    """
    z_sum = z_values[0].copy()
    z_min, z_max = np.min(z_values[0]), np.max(z_values[0])
    if len(z_values) > 0:
        for i in z_values[1:]:
            i_min = np.min(i)
            i_max = np.max(i)
            z_min = np.min([z_min, i_min])
            z_max = np.max([z_max, i_max])
            z_sum += i
    return z_min, z_max, z_sum


def find_index(number, levels):
    """
    finds the position of a number in a sorted list. If the number is smaller then the smallest element in the list the
    function returns 0 if it is bigger than the biggest number in the list it returns the length of the list

    :param number: the number which position should be found
    :param levels: the list in which is locked for the position
    :param verbose: output for debugging
    :return: position of the number
    """
    start = 0
    end = len(levels) - 1
    logger.debug("Number: {}".format(number))
    logger.debug("Level: [{}]".format(", ".join([str(i) for i in levels])))
    if number < levels[start]:
        return 0
    if number > levels[end]:
        return end + 1
    pivo = int((end - start) / 2 + start)
    while start != pivo:
        logger.debug("{}:{}:{}".format(start, pivo, end))
        if number < levels[pivo]:
            end = pivo
        else:
            start = pivo
        pivo = int((end - start) / 2 + start)
    return pivo + 1


def generate_monochromatic_plot_from_gaussians(z_list, color_schemes_list):
    """
    generates a color-images from a 2D-gaussian

    :param z_list: [2D-gaussian_1, ... ,2D-gaussian_n]
    :param color_schemes_list: [startcolor_1, startcolor_n]
    :return: [2D-image_1, ... , 2D-image_n]
    """
    z_color_list = []
    for z, startcolor in zip(z_list, color_schemes_list):
        z_color, _ = picture_contours.get_colorgrid(z, color_schemes.create_monochromatic_colorscheme, 10, False,
                                                    startcolor=startcolor)
        z_color_list.append(z_color)
    return z_color_list


def generate_brewer_plot_from_gaussians(z_list, color_schemes_list):
    """

    :param z_list: [2D-gaussian_1, ... ,2D-gaussian_n]
    :param color_schemes_list: [colorname_1, colorname_n]
    :return: [2D-image_1, ... , 2D-image_n]
    """
    z_color_list = []
    for z, colorscheme in zip(z_list, color_schemes_list):
        z_color, _ = picture_contours.get_colorgrid(z, color_schemes.create_color_brewer_colorscheme, 10, False,
                                                    colorscheme=colorscheme)
        z_color_list.append(z_color)
    return z_color_list
