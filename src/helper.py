import random
import numpy as np
from scipy.stats import multivariate_normal

from src import picture_worker, color_schemes


def get_gaussian(x_min, x_max, y_min, y_max, mu_x=0.0, variance_x=0.0, mu_y=0.0, variance_y=0.0, size=500000):
    """
    returns a gaussian-distribution

    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :param mu_x:
    :param variance_x:
    :param mu_y:
    :param variance_y:
    :param size:
    :return:
    """
    xlist = np.linspace(x_min, x_max, size)
    ylist = np.linspace(y_min, y_max, size)
    X, Y = np.meshgrid(xlist, ylist)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    Z = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
    return X, Y, Z.pdf(pos)


def get_random_gaussian(x_min, x_max, y_min, y_max, variance_min, variance_max, size, scale_factor=1):
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
    mu_variance_x_1 = random.randint(variance_min, variance_max)
    mu_variance_y_1 = random.randint(variance_min, variance_max)
    return get_gaussian(x_min, x_max, y_min, y_max, *(mu_x_1, mu_variance_x_1, mu_y_1, mu_variance_y_1), size)


def generate_gaussians(gaussians):
    return [get_gaussian(*gaussian)[2] for gaussian in gaussians]


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


def find_index(number, levels, verbose=False):
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
    if number < levels[start]:
        return 0
    if number > levels[end]:
        return end + 1
    pivo = int((end - start) / 2 + start)
    while start != pivo:
        if verbose:
            print("{}:{}:{}".format(start, pivo, end))
        if number < levels[pivo]:
            end = pivo
        if number > levels[pivo]:
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
        z_color, _ = picture_worker.get_colorgrid(z, color_schemes.create_monochromatic_colorscheme, 10, False,
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
        z_color, _ = picture_worker.get_colorgrid(z, color_schemes.create_color_brewer_colorscheme, 10, False,
                                                  colorscheme=colorscheme)
        z_color_list.append(z_color)
    return z_color_list
