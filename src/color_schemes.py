import numpy as np
from matplotlib import pyplot as plt

from src import color_converter


def create_monochromatic_colorscheme(startcolor, levels):
    """
    Generates a monochromatic colorscheme from a startcolor. The values of the startcolor are in rgba
    with r,g,b in [0,1] and a = 1

    :param startcolor: rgba with alpha 255
    :param levels: Array with levels at which the color change
    :return: colorarray with len(levels) + 1 entrys each color is of the form [r,g,b,a], r,g,b,a in [0,1]
    """
    norm_levels = np.linspace(0, 1, len(levels) + 1)
    return [[startcolor[0], startcolor[1], startcolor[2], startcolor[3] - i] for i in
            norm_levels]


def matplotlib_colorschemes(colorscheme, levels, lvl_white=1, verbose=False):
    """
    Generates a colorscheme from matplotlib

    :param colorscheme: Colorscheme as String
    :param levels: Array with levels at which the color change
    :return: colorarray with len(levels) + 1 entrys each color is of the form [r,g,b,a], r,g,b,a in [0,1]
    """
    if verbose:
        print("colorscheme: {} | levels: {} |level white: {}".format(colorscheme,levels,lvl_white))
    color_scheme = [i for i in
            plt.cm.get_cmap(colorscheme)(np.linspace(0, 1, len(levels) + 1))]
    for i in range(lvl_white+1):
        color_scheme[i] = np.array([1., 1., 1., 1.])
    return color_scheme
