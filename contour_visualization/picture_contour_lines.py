import numpy as np
from skimage import measure

from contour_visualization import helper
from contour_visualization.picture_contours import get_iso_levels, logger, norm_levels


def generate_contour_lines(ax, X, distribution_limits, contour_lines_colorscheme, contour_lines_method="equal_density",
                           contour_lines_weighted=True, num_of_levels=8, borders=None, linewidth=2):
    if borders is None:
        borders = [0.5, 1.]
    levels = get_iso_levels(X, contour_lines_method, num_of_levels + 1)
    logger.debug("Level: {}".format(levels))
    if contour_lines_weighted:
        contour_lines_colors = get_contour_line_colors(contour_lines_colorscheme, levels, borders)
    else:

        contour_lines_colors = np.repeat(
            contour_lines_colorscheme["colorscheme"](contour_lines_colorscheme["colorscheme_name"],
                                                     [1.], lvl_white=0), num_of_levels + 1, axis=0)
    plot_contour_lines(ax, X, distribution_limits, levels, contour_lines_colors, linewidth=linewidth)


def plot_contour_lines(ax, X, limits, levels, colors, linewidth=2, *args, **kwargs):
    contours = find_contour_lines(X, levels)
    for i, color in zip(contours[:len(levels)], colors[:len(levels)]):
        for contour in i:
            contour = helper.normalize_2d_array(contour, 0, X.shape[0], limits.y_min, limits.y_max, 0,
                                                X.shape[1],
                                                limits.x_min, limits.x_max)
            ax.plot(contour[:, 1], contour[:, 0], linewidth=linewidth, color=color, *args, **kwargs)


def find_contour_lines(X, levels):
    """
    returns num_of_levels contourlines for given weights
    :param z_value: 2D-weight
    :param num_of_levels: number of contour-lines to return
    :return: [contour-lines_1, ... , contourlines_n]
    """
    # levels = get_iso_levels(X, method, num_of_levels + 1)
    logger.debug(levels)
    contours = []
    for i in levels:
        contours.append(measure.find_contours(X, i))
    return contours


def get_contour_line_colors(contour_lines_colorscheme, level, borders):
    return contour_lines_colorscheme["colorscheme"](contour_lines_colorscheme["colorscheme_name"],
                                                    norm_levels(level, *borders), lvl_white=0)
