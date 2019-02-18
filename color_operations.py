import numpy as np

import color_converter
import helper

def blend_rgb255(blending_operator, color_1, color_2, alpha, **kwargs):
    """
    blends rgb color with blendingoperator and returns rgb255 data
    :param blending_operator: blending operator which is used to calculate new color, color_1, color_2, alpha
    :param color_1: first color
    :param color_2: second color
    :param alpha: alpha value
    :return: new color
    """
    #color_blending_operator.porter_duff_source_over
    def convert_rgb01(color):
        return color_converter.rgb255_to_rgb01(color)
    def convert_rgb255(color):
        return color_converter.rgb01_to_rgb255(color)
    return convert_rgb255(blending_operator(convert_rgb01(color_1), convert_rgb01(color_2), alpha, **kwargs))

def map_colors(x ,colormap, levels):
    """

    :param x: 2D-Array with values to map
    :param colormap: colors for each level
    :param levels: levels at which colors are seperated
    :return: 2D-Array with rgba color-codes
    """
    if len(colormap) != len(levels)+ 1:
        raise Exception("{} !={} + 1".format(len(colormap),len(levels)))
    X_new = np.zeros([len(x), len(x[0]), 4], dtype=np.uint8)
    for i in range(len(x)):
        for j in range(len(x[0])):
            X_new[i][j] = colormap[helper.find_index(x[i][j], levels)]
            X_new[i][j][3] = 255 - X_new[i][j][3]
    return X_new