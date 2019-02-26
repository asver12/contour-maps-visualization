import numpy as np

def porter_duff_source_over(color_1, color_2, alpha):
    """
    uses the source-over Porter-Duff operator to calculate a new color C
    if one of the colors is not np.array it will be transformt
    :param color_1: (r,g,b,a)
    :param color_2: (r,g,b,a)
    :param alpha: in [0,1]
    :return: np.array([r,g,b])
    """
    color_1 = np.asarray(color_1)
    color_2 = np.asarray(color_2)
    return color_1*alpha + color_2*(1-alpha)

def simple_alpha_composing(color_1, color_2, verbose=False):
    """
    uses the source-over Porter-Duff operator to calculate a new color C
    if one of the colors is not np.array it will be transformt
    :param color_1: (r,g,b,a)
    :param color_2: (r,g,b,a)
    :param alpha: in [0,1]
    :return: np.array([r,g,b])
    """
    color_1 = np.asarray(color_1)
    color_2 = np.asarray(color_2)
    if verbose:
        print("c1: {} c2: {} = {}".format(color_1, color_2, color_1*color_1[3]+color_2*color_2[3]*(1-color_1[3])))
    return color_1*color_1[3]+color_2*color_2[3]*(1-color_1[3])

def simple_alpha_mult(color_1, color_2, verbose=False):
    """
    uses the source-over Porter-Duff operator to calculate a new color C
    if one of the colors is not np.array it will be transformt
    :param color_1: (r,g,b,a)
    :param color_2: (r,g,b,a)
    :param alpha: in [0,1]
    :return: np.array([r,g,b])
    """
    color_1 = np.asarray(color_1)
    color_2 = np.asarray(color_2)
    if verbose:
        print("c1: {} c2: {} = {}".format(color_1, color_2, color_1*color_2))
    return color_1*color_2

def survival_of_the_stronges(color_1, color_2, verbose=False):
    """
    uses the source-over Porter-Duff operator to calculate a new color C
    if one of the colors is not np.array it will be transformt
    :param color_1: (r,g,b,a)
    :param color_2: (r,g,b,a)
    :param alpha: in [0,1]
    :return: np.array([r,g,b])
    """
    color_1 = np.asarray(color_1)
    color_2 = np.asarray(color_2)
    if verbose:
        print("c1: {} c2: {} = {}".format(color_1, color_2, color_1*color_1[3]+color_2*color_2[3]*(1-color_1[3])))
    return color_1*color_1[3]+color_2*color_2[3]*(1-color_1[3])