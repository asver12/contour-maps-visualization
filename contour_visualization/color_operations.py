import numpy as np

from contour_visualization import helper, color_schemes
import logging

logger = logging.getLogger(__name__)


def map_colors(x, colormap, levels, split=True, lower_border=None):
    """
    Maps a color to given levels


    :param x: 2D-Array with values to map
    :param colormap: colors for each level
    :param levels: levels at which colors are seperated
    :param split: defines if alpha stay in the color_map or not
    :param lower_border: border at which to cut, if not given uses all level given
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
                if lower_border:
                    if lower_border < x[i][j]:
                        index = helper.find_index(x[i][j], levels)
                        x_new[i][j] = colormap[index][:split_number]
                        logger.debug(x_new[i][j])
                        alpha_new[i][j] = colormap[index][3]
                    else:
                        index = helper.find_index(x[i][j], levels)
                        x_new[i][j] = color_schemes.get_white()[:-1]
                        alpha_new[i][j] = colormap[index][3]
                else:
                    index = helper.find_index(x[i][j], levels)
                    x_new[i][j] = colormap[index][:split_number]
                    logger.debug(x_new[i][j])
                    alpha_new[i][j] = colormap[index][3]
                logger.debug(x_new[i][j])

    else:
        raise Exception("{} + 1 !={}".format(len(levels), len(colormap)))
    return x_new, alpha_new
