import numpy as np


def porter_duff_source_over(color_1, color_2, alpha):
    """
    uses the source-over Porter-Duff operator to calculate a new color C with one alpha for each pixel
    if one of the colors is not np.array it will be transformed

    :param color_1: np.array, list or tuple with alpha at last position
    :param color_2: np.array, list or tuple with alpha at last position
    :param alpha: in [0,1]
    :param verbose: to get additional output
    :return: np.array(color) with same size and color-space as input
    """
    color_1 = np.asarray(color_1, dtype=np.float)
    color_2 = np.asarray(color_2, dtype=np.float)
    return color_1 * alpha + color_2 * (1 - alpha)


def alpha_composing_specific(color_1, alpha_1, color_2, alpha_2, verbose=False):
    """
    uses the source-over Porter-Duff operator to calculate a new color C but with an alpha for each pixel
    if one of the colors is not np.array it will be transformed

    :param color_1: np.array, list or tuple with alpha at last position
    :param color_2: np.array, list or tuple with alpha at last position
    :param verbose: to get additional output
    :return: np.array(color) with same size and color-space as input
    """
    color_1 = np.asarray(color_1, dtype=np.float)
    color_2 = np.asarray(color_2, dtype=np.float)
    if verbose:
        print("c1: {} c2: {} = {}".format(color_1, color_2,
                                          color_1 * color_1[3] + color_2 * color_2[3] * (1 - color_1[3])))
    return color_1 * alpha_1 + color_2 * alpha_2


def blend_one_point_color(color_1, color_2, alpha, position=0):
    color_1 = np.asarray(color_1, dtype=np.float)
    color_2 = np.asarray(color_2, dtype=np.float)
    color_1[position] = color_1[position] * alpha + color_2[position] * (1 - alpha)
    return color_1


def simple_color_mult(color_1, color_2, verbose=False):
    """
    multiplies the colors to achieve a new color C
    if one of the colors is not np.array it will be transformed

    :param color_1: np.array, list or tuple with alpha at last position
    :param color_2: np.array, list or tuple with alpha at last position
    :param verbose: to get additional output
    :return: np.array(color) with same size and color-space as input
    """
    color_1 = np.asarray(color_1, dtype=np.float)
    color_2 = np.asarray(color_2, dtype=np.float)
    if verbose:
        print("c1: {} c2: {} = {}".format(color_1, color_2, color_1 * color_2))
    return color_1 * color_2


def hsv_color_operator(color_1, color_2, alpha=.5, position=0):
    if color_1[0] == 0:
        return color_2
    if color_2[0] == 0:
        return color_1
    color_1[position] = color_1[position] * alpha + color_2[position] * (1 - alpha)
    return color_1
