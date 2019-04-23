import random

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from skimage import color


def create_monochromatic_colorscheme(startcolor, levels, min_value = 0, max_value = 1, lvl_white=1, verbose=False):
    """
    Generates a monochromatic colorscheme from a startcolor. The values of the startcolor are in rgba
    with r,g,b in [0,1] and a = 1

    :param startcolor: rgba with alpha 255
    :param levels: Array with levels at which the color change
    :param lvl_white: First x colors which are supposed to be white
    :return: colorarray with len(levels) + 1 entrys each color is of the form [r,g,b,a], r,g,b,a in [0,1]
    """
    norm_levels = np.linspace(min_value, max_value, len(levels) + 1)
    if verbose:
        print("Min: {} | Max: {}".format(min_value, max_value))
    color_scheme = [[float(startcolor[0]), float(startcolor[1]), float(startcolor[2]), float(i)] for i in
                    norm_levels]
    for i in range(lvl_white + 1 if lvl_white < len(levels) else len(levels) + 1):
        color_scheme[i] = np.array([1., 1., 1., 1.])
    if verbose:
        print(color_scheme)
    return color_scheme

def create_hsl_colorscheme(startcolor, levels, min_value = 0, max_value = 1, lvl_white=1, verbose=False):
    """

    :param startcolor:
    :param levels:
    :param min_value:
    :param max_value:
    :param lvl_white:
    :param verbose:
    :return:
    """
    # 234 65 29
    # 232 57 36
    # 231 53 40
    # 231 50 44
    # 230 48 47
    # 231 44 55
    # 230 44 63
    if len(startcolor) == 4:
        rgb = color.rgba2rgb([[startcolor]])
    elif len(startcolor) == 3:
        rgb = [[startcolor]]
    else:
        raise ValueError("Expected RGB or RGBa value")
    norm_levels = np.linspace(min_value, max_value, len(levels) + 1)
    if verbose:
        print("Min: {} | Max: {}".format(min_value, max_value))
    hsv = color.rgb2hsv(rgb)
    print(hsv)
    color_scheme = [color.hsv2rgb([[[float(hsv[0][0][0]),float(i),float(1-i)]]]) for i in norm_levels]
    print([[float(hsv[0][0][0]),float(i),float(1-i)] for i in norm_levels])
    color_scheme = [np.array([i[0][0][0],i[0][0][1],i[0][0][2],1.]) for i in color_scheme]
    print(color_scheme)
    for i in range(lvl_white + 1 if lvl_white < len(levels) else len(levels) + 1):
        color_scheme[i] = np.array([1., 1., 1., 1.])
    if verbose:
        print(color_scheme)
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


def random_matplotlib_colorschemes():
    return list(cm.cmap_d.keys())[random.randrange(len(cm.cmap_d.keys()))]
