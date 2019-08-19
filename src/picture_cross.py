import math
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import scipy.ndimage
import itertools
import collections
import pyclipper

from src import helper, picture_worker, color_schemes, hierarchic_blending_operator

import logging

from src.Gaussian import Gaussian

logger = logging.getLogger(__name__)


def plot_images(cross_lines, gaussians, z_sums, colors=None, color_space="lab", contour_lines_method="equal_density",
                contour_lines=True,
                contour_lines_weighted=True, num_of_levels=8,
                title="", with_axis=True, borders=None, linewidth_contour=2, linewidth_cross=0.5, columns=5,
                bottom=0.0,
                left=0., right=2.,
                top=2.):
    logger.debug("{}".format(["mu_x", "variance_x", "mu_y", "variance_y"]))
    if len(gaussians) == 1:
        title_j = ""
        if title == "" and gaussians:
            title_j = '\n'.join("{}".format(gau[4:-1]) for gau in gaussians[0])
        elif len(title) > columns:
            title_j = title[columns]
        color_legend = colors if colors else []
        plot_image(plt, cross_lines[0], gaussians[0], z_sums[0], color_legend, color_space, contour_lines_method,
                   contour_lines_weighted,
                   title_j, with_axis, num_of_levels, borders, linewidth_contour, linewidth_cross)
        plt.subplots_adjust(bottom=bottom, left=left, right=right, top=top)
    else:
        for i in range(math.ceil(len(gaussians) / columns)):
            subplot = cross_lines[i * columns:(i + 1) * columns]
            sub_sums = z_sums[i * columns:(i + 1) * columns]
            fig, axes = plt.subplots(1, len(subplot), sharex='col', sharey='row')
            if len(subplot) == 1:
                title_j = ""
                if title == "" and gaussians:
                    title_j = '\n'.join("{}".format(gau[4:-1]) for gau in gaussians[i * columns])
                elif len(title) > i * columns:
                    title_j = title[i * columns]
                plot_image(axes, subplot[0], gaussians[i * columns], sub_sums[0], colors, color_space,
                           contour_lines_method,
                           contour_lines_weighted,
                           title_j,
                           with_axis,
                           num_of_levels,
                           borders,
                           linewidth_contour)
            else:
                for j in range(len(subplot)):
                    title_j = ""
                    if title == "" and gaussians:
                        title_j = '\n'.join("{}".format(gau[4:-1]) for gau in gaussians[j + i * columns])
                    elif len(title) > j + i * columns:
                        title_j = title[j + i * columns]
                    plot_image(axes[j], subplot[j], gaussians[j + i * columns], sub_sums[j], colors, color_space,
                               contour_lines_method,
                               contour_lines_weighted,
                               title_j,
                               with_axis,
                               num_of_levels,
                               borders,
                               linewidth_contour)
                    logger.debug(gaussians[0][0])
                    axes[j].set_xlim([gaussians[j + i * columns][0][0], gaussians[j + i * columns][0][1]])
                    axes[j].set_ylim([gaussians[j + i * columns][0][2], gaussians[j + i * columns][0][3]])
                    axes[j].set_aspect('equal', 'box')
            fig.subplots_adjust(bottom=bottom, left=left, right=right, top=top)


def plot_image(axis, cross_lines, gaussians, z_sum, colors, color_space="lab",
               contour_lines_method="equal_density",
               contour_lines_weighted=True, title="", with_axis=True,
               num_of_levels=6, borders=None, linewidth=2,
               contour_lines_colorscheme=color_schemes.get_background_colorbrewer_scheme()):
    logger.debug("gaussians: {}".format(gaussians))
    for cross in cross_lines:
        generate_cross(axis, *cross[:4])
    fill_between_lines(axis, cross_lines, color_space=color_space)
    picture_worker.generate_contour_lines(axis, z_sum, gaussians[0], contour_lines_colorscheme,
                                          contour_lines_method,
                                          contour_lines_weighted, num_of_levels, borders, linewidth)


def filter_order_list(list_1, idx):
    return [list_1[i] for i in idx]
    # return list(collections.OrderedDict.fromkeys(list_1))


def filter_order_color(list_1):
    new_list = []
    idx = []
    for i, lst in enumerate(list_1):
        for j in new_list:
            if all(k in j for k in lst):
                break
        else:
            new_list.append(lst)
            idx.append(i)
    return idx, new_list


def convert_to_int(point, scale=10 ** 5):
    return int(point[0] * scale), int(point[1] * scale)


def convert_to_float(point, scale=10 ** 5):
    return point[0] / scale, point[1] / scale


def get_intersection(polys):
    pc = pyclipper.Pyclipper()
    # PT_SUBJECT
    # PT_CLIP
    for poly in polys[:-1]:
        pc.AddPath(poly, pyclipper.PT_SUBJECT, True)
    pc.AddPath(polys[-1], pyclipper.PT_CLIP, True)
    return pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)


def generate_polygons(rectangle_1, colors, z_weights):
    for first_point, second_point, color, z_weight in zip(rectangle_1[:-1], rectangle_1[1:], colors, z_weights):
        poly = (convert_to_int(first_point[0]), convert_to_int(second_point[0]), convert_to_int(second_point[1]),
                convert_to_int(first_point[1]))
        yield poly, color, z_weight


def generate_polys(cross_lines):
    crosses = []
    for rectangle_1, rectangle_2, colors_1, colors_2, z_weights_1, z_weights_2 in cross_lines:
        poly_cross = []
        for i in generate_polygons(rectangle_1, colors_1, z_weights_1):
            poly_cross.append(i)
        for i in generate_polygons(rectangle_2, colors_2, z_weights_2):
            poly_cross.append(i)
        crosses.append(poly_cross)
    return crosses


def get_fill_regions(cross_lines, int_condition=1000):
    # create rectangles
    crosses = generate_polys(cross_lines)

    # find all intersections of 2 polygons
    poly_return = []
    for cross_1, cross_2 in itertools.combinations(crosses, 2):
        for a in cross_1:
            for b in cross_2:
                try:
                    solution = get_intersection([a[0], b[0]])
                    if solution:
                        poly_return.append([solution[0], [a[1], b[1]], [a[2], b[2]]])
                except:
                    pass

    # find all intersections of the intersections from 2 polygons
    new_polys = poly_return
    i = 0
    next_polys = []
    while len(new_polys) > 2 and i < len(crosses) + 1 and len(next_polys) < int_condition:
        logger.debug("Iteration[{}]: {}".format(len(crosses), i))
        logger.debug("Polys: {}".format(len(new_polys)))
        next_polys = []
        com_polys = []
        for a, b in itertools.combinations(new_polys, 2):
            if a[0] != b[0]:
                try:
                    solution = get_intersection([a[0], b[0]])
                    if solution and solution not in com_polys:
                        next_polys.append([solution[0], [*a[1], *b[1]], [*a[2], *b[2]]])
                        com_polys.append(solution)
                        poly_return.append([solution[0], [*a[1], *b[1]], [*a[2], *b[2]]])
                except:
                    pass
        logger.debug("New Polys: {}".format(len(next_polys)))
        logger.debug("Total Polys: {}".format(len(poly_return)))
        new_polys = next_polys
        i += 1

    # wichtig für Masterarbeit als Bild um Aufbau zu erklären
    # fig, axes = plt.subplots(1, 1, sharex='col', sharey='row')
    # for poly in polys:
    #     axes.plot(*poly[0].exterior.xy)
    # logger.debug("Points: {}".format(points))
    return poly_return


def mix_colors(colors, z_weights, color_space="lab"):
    color = picture_worker.convert_color_to_colorspace(colors[0], color_space)
    z_weights = sorted(list(zip(z_weights, colors)), key=lambda x: x[0])
    z_weight = z_weights[0][0]
    for z_wei, col in z_weights:
        col = picture_worker.convert_color_to_colorspace(col, color_space)
        color, z_weight = hierarchic_blending_operator.porter_duff_source_over(color, z_weight, col, z_wei)
    return picture_worker.convert_color_to_rgb(color, color_space)


def fill_between_lines(axis, cross_lines, color_space="lab"):
    filled_regions = get_fill_regions(cross_lines)
    for region in filled_regions:
        idx, col = filter_order_color(region[1])
        z_weig = filter_order_list(region[2], idx)
        if len(region[2]) > 2:
            logger.debug("col[{}]: {}".format(region[1], col))
            logger.debug("z[{}]: {}".format(region[2], z_weig))
        color = mix_colors(col, z_weig, color_space)
        axis.add_patch(Polygon([convert_to_float(point) for point in region[0]], closed=True,
                               fill=True, edgecolor=color, facecolor=color, aa=True, linewidth=0.))


def generate_line(axis, line, color_points=None, borders=None):
    if borders is None:
        borders = [0.5, 1]
    if color_points is None:
        contour_lines_colorscheme = color_schemes.get_colorbrewer_schemes()[0]
        color_points = contour_lines_colorscheme["colorscheme"](
            contour_lines_colorscheme["colorscheme_name"],
            picture_worker.norm_levels(np.linspace(0, 1, len(line) - 1), *borders), lvl_white=0)
    points = np.array(list(zip(line[:-1], line[1:])))
    logger.debug("Points: {}".format(points))
    for j, i in enumerate(points):
        idx = [0, 2, 3, 1]
        axis.add_patch(Polygon(np.array(i).reshape(4, 2)[idx], closed=True,
                               fill=True, edgecolor=color_points[j], facecolor=color_points[j], aa=True, linewidth=0.))


def get_half_lines(middlepoint, direction, length, verb=True):
    startpoint = middlepoint[0] - direction[1] * length, middlepoint[1] - direction[0] * length
    endpoint = middlepoint[0] + direction[1] * length, middlepoint[1] + direction[0] * length

    return (startpoint, middlepoint), (middlepoint, endpoint)


def split_half_line(startpoint, endpoint, iso_level, x_list, y_list, z_list):
    num = 100
    index_start_x = helper.find_index(startpoint[1], x_list[0].flatten())
    index_start_y = helper.find_index(startpoint[0], y_list[:, 0].flatten())
    index_end_x = helper.find_index(endpoint[1], x_list[0].flatten())
    index_end_y = helper.find_index(endpoint[0], y_list[:, 0].flatten())
    x, y = np.linspace(index_start_x, index_end_x, num), np.linspace(index_start_y, index_end_y, num)

    # fehlerhaft???
    zi = scipy.ndimage.map_coordinates(z_list, np.vstack((x, y)))

    x_1, y_1 = np.linspace(startpoint[0], endpoint[0], num), np.linspace(startpoint[1], endpoint[1], num)
    used_points = [(i, j) for i, j in zip(x_1, y_1)]
    logger.debug("------------------------------------------------------------")
    logger.debug("ISO-Line: {}".format(iso_level))
    logger.debug("Points on Line: {}".format(used_points))
    logger.debug("Points on Z: {}".format(zi))
    logger.debug("------------------------------------------------------------")
    split_points = []
    start_point = 0
    for i in iso_level:
        best_match = 1
        point = None
        for j, z in enumerate(zi):
            if abs(i - best_match) > abs(i - z):
                logger.debug("New z found: {} [old: {}]".format(z, best_match))
                best_match = z
                point = used_points[j]
                start_point += j + 1
        split_points.append(point)
    return [startpoint, *split_points, endpoint]


def get_color(iso_level, colorscheme, level_white=0):
    return colorscheme["colorscheme"](
        colorscheme["colorscheme_name"], iso_level, lvl_white=level_white)


def get_line(gaussian, x_list, y_list, z_list, eigenvalue, eigenvector, colorscheme, min_value=0., max_value=1.,
             method="equal_density",
             num_of_levels=5):
    iso_level = picture_worker.get_iso_levels(z_list, method=method, num_of_levels=num_of_levels)
    first_line, second_line = get_half_lines(gaussian.means, eigenvector, eigenvalue)
    logger.debug("------------------------------------------------------------")
    logger.debug("First Part of Line {}".format(first_line))
    logger.debug("Second Part of Line {}".format(second_line))
    logger.debug("------------------------------------------------------------")
    first_line = split_half_line(*first_line, iso_level, x_list, y_list, z_list)
    second_line = split_half_line(*second_line, iso_level[::-1], x_list, y_list, z_list)
    iso_lvl = picture_worker.get_iso_levels(z_list, method=method, num_of_levels=num_of_levels + 2)
    iso_lvl = picture_worker.get_color_middlepoint(iso_lvl, min_value, max_value)
    colors = get_color(iso_lvl, colorscheme)
    return [*first_line, *second_line[1:]], [*colors, *colors[::-1]], [*iso_lvl, *iso_lvl[::-1]]


def generate_rectangle_from_line(line, eigenvector, length):
    new_line = []
    for i in line:
        first_point, second_point = get_half_lines(i, eigenvector, length, verb=False)
        new_line.append((first_point[0], second_point[1]))
    return new_line


def get_cross(gaussian: Gaussian, colorscheme, min_value=0., max_value=1., length=3, *args, **kwargs):
    picture_worker.check_constrains(min_value, max_value)
    x_list, y_list, z_list = gaussian.get_density_grid()
    eigenvalues, eigenvectors = np.linalg.eig(gaussian.cov_matrix)
    logger.debug(gaussian)
    logger.debug(eigenvalues)
    logger.debug(eigenvectors)
    line_1, colors_1, z_lvl_1 = get_line(gaussian, x_list, y_list, z_list, eigenvalues[0], eigenvectors[1], colorscheme,
                                         min_value, max_value,
                                         *args,
                                         **kwargs)
    line_2, colors_2, z_lvl_2 = get_line(gaussian, x_list, y_list, z_list, eigenvalues[1], eigenvectors[0], colorscheme,
                                         min_value, max_value,
                                         *args,
                                         **kwargs)
    rectangle_1 = generate_rectangle_from_line(line_1, eigenvectors[0], length)
    rectangle_2 = generate_rectangle_from_line(line_2, eigenvectors[1], length)
    return rectangle_1, rectangle_2, colors_1, colors_2, z_lvl_1, z_lvl_2


def generate_cross(axis, line_1, line_2, colors_1, colors_2):
    logger.debug("Len Line 1: {}".format(len(line_1)))
    logger.debug("Len Colors 1: {}".format(len(colors_1)))
    generate_line(axis, line_1, colors_1)
    generate_line(axis, line_2, colors_2)


def genenerate_crosses(gaussians, z_list, z_min, z_max, colorschemes, length=3, borders=None, *args, **kwargs):
    if borders is None:
        borders = [0, 1]
    lower_border = borders[0]
    upper_border = borders[1]
    z_weights = []
    for z, colorscheme in zip(z_list, colorschemes):
        z_min_weight = (upper_border - lower_border) * (np.min(z) - z_min) / (z_max - z_min) + lower_border
        z_max_weight = (upper_border - lower_border) * (np.max(z) - z_min) / (z_max - z_min) + lower_border
        z_weights.append([z_min_weight, z_max_weight])
    return [get_cross(i, j, *k, length, *args, **kwargs) for i, j, k in zip(gaussians, colorschemes, z_weights)]


def input_crosses(ax, gaussians, z_list, z_min, z_max, colorschemes, length=3, borders=None, color_space="lab", *args,
                  **kwargs):
    if not isinstance(gaussians[0], Gaussian):
        raise ValueError("Expected Gaussian instead got {}".format(type(gaussians[0])))
    cross_lines = genenerate_crosses(gaussians, z_list, z_min, z_max, colorschemes, length, borders, *args, **kwargs)
    for cross in cross_lines:
        generate_cross(ax, *cross[:4])
    fill_between_lines(ax, cross_lines, color_space=color_space)


def generate_image(gaussians, colorschemes, length=3, borders=None, *args, **kwargs):
    if not isinstance(gaussians[0], Gaussian):
        raise ValueError("Expected Gaussian instead got {}".format(type(gaussians[0])))
    logger.debug(gaussians)
    if borders is None:
        borders = [0, 1]
    lower_border = borders[0]
    upper_border = borders[1]
    z_list = helper.generate_gaussians(gaussians)
    z_min, z_max, z_sum = helper.generate_weights(z_list)
    z_weights = []
    for z, colorscheme in zip(z_list, colorschemes):
        z_min_weight = (upper_border - lower_border) * (np.min(z) - z_min) / (z_max - z_min) + lower_border
        z_max_weight = (upper_border - lower_border) * (np.max(z) - z_min) / (z_max - z_min) + lower_border
        z_weights.append([z_min_weight, z_max_weight])
    return z_list, [
        get_cross(i, j, *k, length, *args, **kwargs) for i, j, k
        in
        zip(gaussians, colorschemes, z_weights)], z_sum
