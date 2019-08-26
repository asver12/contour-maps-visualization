import numpy as np
from matplotlib.patches import Wedge

from contour_visualization import helper, color_schemes

import logging

logger = logging.getLogger(__name__)

colorschemes = color_schemes.get_colorbrewer_schemes()
color_codes = [color_schemes.get_main_color(i)[-4] for i in colorschemes]


def sort_ratios(sorting_list, sorted_list):
    """
    sorts both lists with with the keys from the first list
    :param sorting_list: list which is used to sort
    :param sorted_list: list which is sorted by first list
    :return: both list sorted by first list
    """
    return zip(*sorted(zip(sorting_list, sorted_list[:len(sorting_list)])))


def draw_pie(ax, ratios, center, colors=None, radius=0.02, angle=0, ):
    if colors is None:
        colors = color_codes
    ratios, colors = sort_ratios(ratios, colors)
    thetas = []
    sum_ratios = sum(ratios)
    for i in ratios:
        thetas.append((i / sum_ratios) * 360)
    cum_thetas = np.cumsum(thetas)
    cum_thetas = np.insert(cum_thetas, 0, 0)
    cum_thetas = [i + angle for i in cum_thetas]
    for k, theta in enumerate(zip(cum_thetas[:-1], cum_thetas[1:])):
        # only works with axis not plt!!! #
        ax.add_artist(Wedge(center, radius, *theta, fc=colors[k]))


def get_radius(distances, value, z_min, z_max, borders):
    norm_values = helper.normalize_array(value, z_min, z_max, *borders)
    return (distances / 2) * norm_values


def container_size(x_min, x_max, y_min, y_max, num_of_pies_row, num_of_pies_column):
    x = np.linspace(x_min, x_max, num_of_pies_row + 1)
    y = np.linspace(y_min, y_max, num_of_pies_column + 1)
    x = [(i + j) / 2 for i, j in zip(x[:-1], x[1:])]
    y = [(i + j) / 2 for i, j in zip(y[:-1], y[1:])]
    distances = abs(x_max - x_min) / num_of_pies_row, abs(y_max - y_min) / num_of_pies_column
    return np.meshgrid(x, y, sparse=True), distances


def get_distance_ratio(num_of_pies_x, x_values, y_values):
    return int(num_of_pies_x * (abs(y_values[1] - y_values[0]) / abs(x_values[1] - x_values[0])))


def input_image(ax, gaussian, z_min, z_max, num_of_pies_x=10, num_of_pies_y=0, angle=0, set_limit=False,
                colorschemes=color_schemes.get_colorbrewer_schemes(),
                modus="light", borders=None):
    if borders is None:
        if modus == "size":
            borders = [0.1, 0.9]
        else:
            borders = [.2, .9]
    if set_limit:
        ax.set_xlim([helper.get_x_values(gaussian)])
        ax.set_ylim([helper.get_y_values(gaussian)])
    if num_of_pies_y == 0:
        num_of_pies_y = get_distance_ratio(num_of_pies_x, helper.get_x_values(gaussian), helper.get_y_values(gaussian))
    container, distances = container_size(*helper.get_x_values(gaussian),
                                          *helper.get_y_values(gaussian),
                                          num_of_pies_x,
                                          num_of_pies_y)
    for k in container[0][0]:
        for l in container[1]:
            middle_point = k, l[0]
            input_values = []
            for j in range(len(gaussian)):
                input_values.append(gaussian[j].get_density(middle_point))
            new_ratio = helper.normalize_array(input_values, min(input_values), max(input_values), 0, 1)
            if new_ratio is not None:
                new_ratio = np.asarray(input_values) / len(input_values)
            if modus == "size":
                use_colors = [color_schemes.get_main_color(i)[-4] for i in colorschemes]
                draw_pie(ax, ratios=new_ratio, angle=angle, center=middle_point,
                         radius=get_radius(min(distances), sum(input_values), z_min, z_max, borders), colors=use_colors)
            elif modus == "light":
                use_colors = []
                for colorscheme in colorschemes:
                    use_colors.append(get_colors_to_use(colorscheme, sum(input_values), z_min, z_max, borders))
                logger.debug("Using colors: {}".format(use_colors))
                draw_pie(ax, ratios=new_ratio, angle=angle, center=middle_point,
                         radius=(min(distances) / 2) * 0.9, colors=use_colors)
            else:
                use_colors = []
                for colorscheme in colorschemes:
                    use_colors.append(get_colors_to_use(colorscheme, sum(input_values), z_min, z_max, borders))
                draw_pie(ax, ratios=new_ratio, angle=angle, center=middle_point,
                         radius=get_radius(min(distances), sum(input_values), z_min, z_max, [0.7, 0.9]),
                         colors=use_colors)


def get_colors_to_use(colorscheme, sum_input_value, z_min, z_max, borders):
    if sum_input_value > z_max:
        logger.warning("Input [{}] bigger than Max-Value [{}] found ".format(sum_input_value, z_max))
        sum_input_value = z_max
    if sum_input_value < z_min:
        logger.warning("Input [{}] smaller than Min-Value [{}] found ".format(sum_input_value, z_max))
        sum_input_value = z_min
    logger.debug("{}{}{}{} = {}".format(sum_input_value, z_min, z_max, borders,
                                        helper.normalize_array(sum_input_value, z_min, z_max,
                                                               *borders)))
    return list(colorscheme["colorscheme"](colorscheme["colorscheme_name"],
                                           [helper.normalize_array(sum_input_value, z_min,
                                                                   z_max, *borders)],
                                           lvl_white=0)[0])
