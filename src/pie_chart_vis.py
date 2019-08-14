import numpy as np
from matplotlib.patches import Wedge

from src import helper, color_schemes

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


def draw_pie(ax, ratios, center, radius=0.02, angle=0, colors=None):
    if not colors:
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
        ax.add_artist(Wedge(center, radius, *theta, fc=colors[k]))


def get_radius(distances, value, z_min, z_max):
    norm_values = helper.normalize_array(value, z_min, z_max, 0.2, 0.9)
    return (distances[0] / 2) * norm_values


def container_size(x_min, x_max, y_min, y_max, num_of_pies_row, num_of_pies_column):
    x = np.linspace(x_min, x_max, num_of_pies_row + 1)
    y = np.linspace(y_min, y_max, num_of_pies_column + 1)
    x = [(i + j) / 2 for i, j in zip(x[:-1], x[1:])]
    y = [(i + j) / 2 for i, j in zip(y[:-1], y[1:])]
    distances = abs(x_max - x_min) / num_of_pies_row, abs(y_max - y_min) / num_of_pies_column
    return np.meshgrid(x, y, sparse=True), distances


def input_image(ax, gaussian, z_min, z_max, num_of_pies_x=10, num_of_pies_y=0, angle=0, set_limit=False, colors=None):
    if not colors:
        colors = color_codes
    if set_limit:
        ax.set_xlim([gaussian[0][0], gaussian[0][1]])
        ax.set_ylim([gaussian[0][2], gaussian[0][3]])
    if num_of_pies_y == 0:
        num_of_pies_y = num_of_pies_x
    container, distances = container_size(gaussian[0][0], gaussian[0][1], gaussian[0][2], gaussian[0][3], num_of_pies_x,
                                          num_of_pies_y)
    gaus = []
    for i in gaussian:
        gaus.append(helper.Gaussian(i[4], i[5]))
    for k in container[0][0]:
        for l in container[1]:
            middle_point = k, l[0]
            input_values = []
            for j in range(len(gaussian)):
                input_values.append(gaus[j].get(*middle_point))
            new_ratio = helper.normalize_array(input_values, min(input_values), max(input_values), 0, 1)
            if new_ratio is not None:
                new_ratio = np.asarray(input_values) / len(input_values)
            draw_pie(ax, ratios=new_ratio, angle=angle, center=middle_point,
                     radius=get_radius(distances, sum(input_values), z_min, z_max), colors=colors)


def generate_image(ax, gaussian, num_of_pies=10, angle=0, set_limit=False, colors=None):
    if not colors:
        colors = color_codes
    z_list = helper.generate_gaussians(gaussian)
    _, _, z_sum = helper.generate_weights(z_list)
    input_image(ax, gaussian, np.min(z_sum), np.max(z_sum), num_of_pies, angle=angle, set_limit=set_limit,
                colors=colors)
