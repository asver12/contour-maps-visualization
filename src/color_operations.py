import numpy as np

from src import helper, color_schemes


def blend_color(blending_operator, color_1, color_2, *args, **kwargs):
    """
    blends rgb color with blendingoperator and returns rgb255 data

    :param blending_operator: blending operator which is used to calculate new color, color_1, color_2, alpha
    :param color_1: first color
    :param color_2: second color
    :param alpha: alpha value
    :return: new color
    """
    return blending_operator(color_1, color_2, *args, **kwargs)


def get_colorcodes(colorschemes):
    """
    Works only for brewer-colorschemes atm
    :param colorschemes: brewer-colorscheme
    :return:
    """
    return [color_schemes.get_main_color(i)[-1] for i in colorschemes]


def map_colors(x, colormap, levels, split=True):
    """
    Maps a color to given levels

    :param x: 2D-Array with values to map
    :param colormap: colors for each level
    :param levels: levels at which colors are seperated
    :param split: defines if alpha stay in the color_map or not
    :return: 2D-Array with rgba color-codes
    """
    # if the areas between the contourlines are filled
    # the first color is white, the last one is the strongest
    if len(colormap) == len(levels) + 1:
        split_number = 3 if split else 4
        x_new = np.zeros([len(x), len(x[0]), split_number])
        alpha_new = np.zeros([len(x), len(x[0]), 1])
        for i in range(len(x)):
            for j in range(len(x[0])):
                index = helper.find_index(x[i][j], levels)
                x_new[i][j] = colormap[index][:split_number]
                alpha_new[i][j] = colormap[index][3]
    else:
        raise Exception("{} + 1 !={}".format(len(levels), len(colormap)))
    return x_new, alpha_new
