import random

from scipy.stats import multivariate_normal
import numpy as np


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
    mu_x_1 = random.randint(int(x_min * scale_factor), int(x_max * scale_factor))
    mu_y_1 = random.randint(int(y_min * scale_factor), int(y_max * scale_factor))
    mu_variance_x_1 = random.randint(variance_min, variance_max)
    mu_variance_y_1 = random.randint(variance_min, variance_max)
    return get_gaussian(x_min, x_max, y_min, y_max, *(mu_x_1, mu_variance_x_1, mu_y_1, mu_variance_y_1), size)


def find_index(number, levels, verbose=False):
    """
    finds the position of a number in a list. If the number is smaller then the smallest element in the list the
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
