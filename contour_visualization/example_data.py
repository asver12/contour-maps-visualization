import copy
import itertools as it
import math

import numpy as np
from contour_visualization import helper, color_schemes
from contour_visualization.Gaussian import Gaussian

import logging

logger = logging.getLogger(__name__)


def generate_gaussians(gaussians):
    return [helper.get_gaussian(*gaussian)[2] for gaussian in gaussians]


def generate_four_moving_gaussians(x_min=-10, x_max=10, y_min=-10, y_max=10, size=200, weights=None):
    if weights is None:
        weights = [1, 1, 1, 1]
    int_gaussians = [None, None, None, None]

    colorschemes = color_schemes.get_colorbrewer_schemes()
    color_codes = [color_schemes.get_main_color(i)[-1] for i in colorschemes]
    cov_matrix = [[5, 0], [0, 5]]
    z_lists = []
    z_sums = []
    gaussians = []
    for i in range(1, 6):
        int_gaussians[0] = Gaussian(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, size=size, means=[+i, +i],
                                    cov_matrix=cov_matrix,
                                    weight=weights[0])
        int_gaussians[1] = Gaussian(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, size=size, means=[-i, -i],
                                    cov_matrix=cov_matrix,
                                    weight=weights[1])
        int_gaussians[2] = Gaussian(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, size=size, means=[+i, -i],
                                    cov_matrix=cov_matrix,
                                    weight=weights[2])
        int_gaussians[3] = Gaussian(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, size=size, means=[-i, +i],
                                    cov_matrix=cov_matrix,
                                    weight=weights[3])
        logger.info(" \n ".join(str(i) for i in int_gaussians))
        z_list = helper.generate_distribution_grids(int_gaussians)
        z_min, z_max, z_sum = helper.generate_weights(z_list)
        z_lists.append(z_list)
        z_sums.append(z_sum)
        gaussians.append([copy.deepcopy(i) for i in int_gaussians])
    return z_lists, z_sums, gaussians, color_codes


def generate_default_gaussian(means=None, cov_matrix=None, *args, **kwargs):
    if cov_matrix is None:
        cov_matrix = [[5, 0], [0, 5]]
    if means is None:
        means = [0, 0]
    return Gaussian(means=means, cov_matrix=cov_matrix, *args, **kwargs)

def generate_reversed_moving_gaussian(x_min=-10, x_max=10, y_min=-10, y_max=10, size=200, weight=0.5):
    variance_x, variance_y = 5, 5

    var_x, var_y = [2, 5, 10, 15, 20], [15, 20]
    gaussians_2d = []

    for i, mu_y in enumerate([5, 2, 0, -2, -5]):
        for j, mu_x in enumerate([5, 2, 0, -2, -5]):
            if mu_y < 0:
                variance_y = var_y[i - 3]
                variance_x = var_x[j]
            else:
                variance_x = 5
            gaussians_2d.append(
                Gaussian(means=[mu_x, mu_y], cov_matrix=[[variance_x, 2], [2, variance_y]], x_min=x_min, x_max=x_max,
                         y_min=y_min, y_max=y_max,
                         size=size, weight=weight))
    return gaussians_2d

def generate_moving_gaussian(x_min=-10, x_max=10, y_min=-10, y_max=10, size=200, weight=0.5):
    var_x, var_y = [2, 5, 10, 15, 20], [15, 20]
    gaussians_2d = []
    variance_x, variance_y = 5, 5

    for i, mu_y in enumerate([-5, -2, 0, 2, 5]):
        for j, mu_x in enumerate([-5, -2, 0, 2, 5]):
            if mu_y > 0:
                variance_x = var_x[j]
                variance_y = var_y[i - 3]
            else:
                variance_x = 5
            gaussians_2d.append(
                Gaussian(means=[mu_x, mu_y], cov_matrix=[[variance_x, 2], [2, variance_y]], x_min=x_min, x_max=x_max,
                         y_min=y_min, y_max=y_max,
                         size=size, weight=weight))
    return gaussians_2d


def generate_two_gaussian(weights=None, *args, **kwargs):
    if weights is None:
        weights = [0.5, 0.5]
    static_gaussian = generate_default_gaussian(weight=weights[0])
    gaussians_2d = generate_moving_gaussian(weight=weights[1], *args, **kwargs)
    example = []
    for i in range(len(gaussians_2d)):
        example.append([static_gaussian, gaussians_2d[i]])
    return example


def generate_three_gaussians(weights=None, *args, **kwargs):
    if weights is None:
        weights = [0.5, 0.5]
    static_gaussian = generate_default_gaussian(weight=weights[0])
    gaussians_2 = generate_moving_gaussian(weight=weights[1], *args, **kwargs)
    gaussians_3 = []
    for i, j in it.combinations(range(len(gaussians_2)), 2):
        gaussians_3.append([static_gaussian, gaussians_2[i], gaussians_2[j]])
    example_data = []
    for i in np.linspace(0, len(gaussians_3) - 1, dtype=int):
        example_data.append(gaussians_3[i])
    return example_data


def generate_four_random_gaussians(weights=None, *args, **kwargs):
    if weights is None:
        weights = [0.25, 0.25, 0.25, 0.25]
    static_gaussian = generate_default_gaussian(weight=weights[0])
    gaussians_1 = generate_moving_gaussian(weight=weights[1], *args, **kwargs)
    gaussians_2 = generate_moving_gaussian(weight=weights[2], *args, **kwargs)
    gaussians_3 = generate_moving_gaussian(weight=weights[3], *args, **kwargs)
    gaussians_4 = []
    for i, j, k in it.combinations(range(len(gaussians_2)), 3):
        gaussians_4.append([static_gaussian, gaussians_1[i], gaussians_2[j], gaussians_3[k]])
    example_data = []
    for i in np.linspace(0, len(gaussians_4) - 1, dtype=int):
        example_data.append(gaussians_4[i])
    return example_data


def generate_four_gaussians(weights=None, *args, **kwargs):
    if weights is None:
        weights = [0.25, 0.25, 0.25, 0.25]
    static_gaussian = generate_default_gaussian(weight=weights[0])
    gaussians_1 = generate_moving_gaussian(weight=weights[1], *args, **kwargs)
    gaussians_2 = generate_moving_gaussian(weight=weights[2], *args, **kwargs)
    gaussians_2_rev = generate_reversed_moving_gaussian(weights[3], *args, **kwargs)
    gaussians_4 = []
    for i in range(len(gaussians_2)):
        if i < math.ceil(len(gaussians_2) / 2):
            j = len(gaussians_2) - i - 1
        else:
            j = i - math.ceil(len(gaussians_2) / 2)
        gaussians_4.append([static_gaussian, gaussians_1[i], gaussians_2[j], gaussians_2_rev[i]])
    return gaussians_4


def generate_five_gaussians(weights=None, *args, **kwargs):
    if weights is None:
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    static_gaussian = generate_default_gaussian(weight=weights[0])
    gaussians_1 = generate_moving_gaussian(weights[1], *args, **kwargs)
    gaussians_2 = generate_moving_gaussian(weights[2], *args, **kwargs)
    gaussians_3 = generate_moving_gaussian(weights[3], *args, **kwargs)
    gaussians_4 = generate_moving_gaussian(weights[4], *args, **kwargs)
    gaussians_5 = []
    for i, j, k, l in it.combinations(range(len(gaussians_1)), 4):
        gaussians_5.append([static_gaussian, gaussians_1[i], gaussians_2[j], gaussians_3[k], gaussians_4[l]])
    example_data = []
    for i in np.linspace(0, len(gaussians_5)-1, dtype=int, num=25):
        example_data.append(gaussians_5[i])
    return example_data
