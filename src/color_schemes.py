import numpy as np
from matplotlib import pyplot as plt

from src import color_converter


def create_monochromatic_colorscheme(startcolor, levels, lvl_white=1):
    """
    Generates a monochromatic colorscheme from a startcolor. The values of the startcolor are in rgba
    with r,g,b in [0,1] and a = 1

    :param startcolor: rgba with alpha 255
    :param levels: Array with levels at which the color change
    :param lvl_white: First x colors which are supposed to be white
    :return: colorarray with len(levels) + 1 entrys each color is of the form [r,g,b,a], r,g,b,a in [0,1]
    """
    norm_levels = np.linspace(0, 1, len(levels) + 1)
    color_scheme = [[float(startcolor[0]), float(startcolor[1]), float(startcolor[2]), float(i)] for i in
                    norm_levels]
    for i in range(lvl_white + 1 if lvl_white < len(levels) else len(levels) + 1):
        color_scheme[i] = np.array([1., 1., 1., 1.])
    return color_scheme


def matplotlib_colorschemes(colorscheme, levels, lvl_white=1, verbose=False):
    """
    Generates a colorscheme from matplotlib

    :param colorscheme: Colorscheme as String
    :param levels: Array with levels at which the color change
    :return: colorarray with len(levels) + 1 entrys each color is of the form [r,g,b,a], r,g,b,a in [0,1]
    """
    if verbose:
        print("colorscheme: {} | levels: {} |level white: {}".format(colorscheme, levels, lvl_white))
    color_scheme = [i for i in
                    plt.cm.get_cmap(colorscheme)(np.linspace(0, 1, len(levels) + 1))]
    for i in range(lvl_white + 1):
        color_scheme[i] = np.array([1., 1., 1., 1.])
    return color_scheme
