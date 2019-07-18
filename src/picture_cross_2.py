import math
import numpy as np

from matplotlib import pyplot as plt
import scipy.ndimage
from typing import NamedTuple
from itertools import combinations
from shapely import geometry
from shapely.ops import cascaded_union

from matplotlib.patches import Polygon

from src import helper, picture_worker, color_schemes

import logging

logger = logging.getLogger(__name__)


def plot_images(cross_lines, gaussians, z_sums, colors=None, contour_lines_method="equal_density", contour_lines=True,
                contour_lines_weighted=True, num_of_levels=8,
                title="", with_axis=True, borders=None, linewidth_contour=2, linewidth_cross=2, columns=5,
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
        plot_image(plt, cross_lines[0], gaussians[0], z_sums[0], color_legend, contour_lines_method,
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
                plot_image(axes, subplot[0], gaussians[i * columns], sub_sums[0], colors,
                           contour_lines_method,
                           contour_lines_weighted,
                           title_j,
                           with_axis,
                           num_of_levels,
                           borders,
                           linewidth_contour, linewidth_cross)
            else:
                for j in range(len(subplot)):
                    title_j = ""
                    if title == "" and gaussians:
                        title_j = '\n'.join("{}".format(gau[4:-1]) for gau in gaussians[j + i * columns])
                    elif len(title) > j + i * columns:
                        title_j = title[j + i * columns]
                    plot_image(axes[j], subplot[j], gaussians[j + i * columns], sub_sums[j], colors,
                               contour_lines_method,
                               contour_lines_weighted,
                               title_j,
                               with_axis,
                               num_of_levels,
                               borders,
                               linewidth_contour, linewidth_cross)
                    logger.debug(gaussians[0][0])
                    axes[j].set_xlim([gaussians[j + i * columns][0][0], gaussians[j + i * columns][0][1]])
                    axes[j].set_ylim([gaussians[j + i * columns][0][2], gaussians[j + i * columns][0][3]])
                    axes[j].set_aspect('equal', 'box')
            fig.subplots_adjust(bottom=bottom, left=left, right=right, top=top)


def plot_image(axis, cross_lines, gaussians, z_sum, colors,
               contour_lines_method="equal_density",
               contour_lines_weighted=True, title="", with_axis=True,
               num_of_levels=6, borders=None, linewidth=2, linewidth_cross=2,
               contour_lines_colorscheme=color_schemes.get_background_colorbrewer_scheme()):
    logger.debug("gaussians: {}".format(gaussians))
    #fill_between_lines(axis, cross_lines)
    for cross in cross_lines:
        generate_cross(axis, *cross, linewidth_cross=linewidth_cross)
    picture_worker.generate_contour_lines(axis, z_sum, gaussians[0], contour_lines_colorscheme,
                                          contour_lines_method,
                                          contour_lines_weighted, num_of_levels, borders, linewidth)


class Line(NamedTuple):
    m: float
    b: float


def calc_line(pkt_1, pkt_2):
    logger.debug("Pkt 1: {}".format(pkt_1))
    logger.debug("Pkt 2: {}".format(pkt_2))
    if pkt_1[0] == pkt_2[0]:
        m = 0
        b = 0
    elif pkt_2[1] == pkt_1[1]:
        m = 1
        b = 0
    else:
        m = (pkt_1[1] - pkt_2[1]) / (pkt_1[0] - pkt_2[0])
        b = pkt_1[0] * m + pkt_1[1]
    return Line(m, b)


def calc_t(point_1, point_2, point_3, point_4, x, y):
    numerator = (point_1[x] - point_3[x]) * (point_3[y] - point_4[y]) \
                - (point_1[y] - point_3[y]) * (point_3[x] - point_4[x])
    denominator = (point_1[x] - point_2[x]) * (point_3[y] - point_4[y]) \
                  - (point_1[y] - point_2[y]) * (point_3[x] - point_4[x])
    return numerator / denominator


def calc_u(point_1, point_2, point_3, point_4, x, y):
    numerator = (point_1[x] - point_2[x]) * (point_1[y] - point_3[y]) \
                - (point_1[y] - point_2[y]) * (point_1[x] - point_3[x])
    denominator = (point_1[x] - point_2[x]) * (point_3[y] - point_4[y]) \
                  - (point_1[y] - point_2[y]) * (point_3[x] - point_4[x])
    return numerator / denominator


def line_intersect(point_1, point_2, point_3, point_4, x, y):
    if (point_1[x] - point_2[x]) * (point_3[y] - point_4[y]) \
            - (point_1[y] - point_2[y]) * (point_3[x] - point_4[x]) == 0:
        return False


def on_segment(point_seq, point_2, point_3):
    if max(point_2[0], point_3[0]) >= point_seq[0] >= min(point_2[0], point_3[0]) and max(point_2[1], point_3[1]) >= \
            point_seq[1] >= min(point_2[1], point_3[1]):
        return True
    return False


def calc_intersection(point_1, point_2, point_3, point_4):
    """
    see https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection for implementation details


    :param point_1:
    :param point_2:
    :param point_3:
    :param point_4:
    :return:
    """
    x = 0
    y = 1
    if line_intersect(point_1, point_2, point_3, point_4, x, y):
        return None, None
    else:
        p_new = ((point_1[x] * point_2[y] - point_1[y] * point_2[x]) * (point_3[x] - point_4[x])
                 - (point_3[x] * point_4[y] - point_3[y] * point_4[x]) * (point_1[x] - point_2[x])) \
                / ((point_1[x] - point_2[x]) * (point_3[y] - point_4[y]) - (point_1[y] - point_2[y]) * (
                point_3[x] - point_4[x])), (
                        (point_1[x] * point_2[y] - point_1[y] * point_2[x]) * (point_3[y] - point_4[y])
                        - (point_3[x] * point_4[y] - point_3[y] * point_4[x]) * (point_1[y] - point_2[y])) \
                / ((point_1[x] - point_2[x]) * (point_3[y] - point_4[y]) - (point_1[y] - point_2[y]) * (
                point_3[x] - point_4[x]))
        logger.debug("First Line: {}".format([point_1, point_2]))
        logger.debug("Second Line: {}".format([point_3, point_4]))
        logger.debug("New Points: {}".format(p_new))
        if on_segment(p_new, point_1, point_2) and on_segment(p_new, point_3, point_4):
            return p_new

        # t = calc_t(point_1, point_2, point_3, point_4, x, y)
        # u = calc_u(point_1, point_2, point_3, point_4, x, y)
        # if 0. <= t <= 1.:
        #     return point_1[x] + t * (point_2[x] - point_1[x]), point_1[y] + t * (point_2[y] - point_1[y])
        # if 0. <= u <= 1.:
        #     return point_3[x] + u * (point_4[x] - point_3[x]), point_3[y] + u * (point_4[y] - point_3[y])
    return None, None


def get_fill_regions(cross_lines):
    cross_points = []
    points = []
    polys = []
    for line_1, line_2, colors_1, colors_2 in cross_lines:
        for first_point, second_point, color in zip(line_1[:-1], line_1[1:], colors_1):
            logger.debug(first_point)
            point_1 = geometry.Point(*first_point[0])
            point_2 = geometry.Point(*first_point[1])
            point_3 = geometry.Point(*second_point[0])
            point_4 = geometry.Point(*second_point[1])
            poly = geometry.Polygon([[p.x, p.y] for p in [point_1,point_3,point_4,point_2]])
            polys.append(poly)
            #     [circles[pos] for pos in idx.intersection(circle.bounds) if circles[pos] != circle])
            # intersections.append(circle.intersection(merged_circles))
    #intersection = cascaded_union([a.intersection(b) for a, b in combinations(polys, 2)])
    # cross_points.append((cross[0][0], cross[0][-1]))
    # cross_points.append((cross[1][0], cross[1][-1]))
    # for i, j in combinations(cross_points, 2):
    #     x, y = calc_intersection(*i, *j)
    #     if x is not None:
    #         points.append(([x - 1, x - 1, x + 1, x + 1], [y - 1, y + 1, y + 1, y - 1]))
    logger.debug("Points: {}".format(points))
    return points


def fill_between_lines(axis, cross_lines):
    filled_regions = get_fill_regions(cross_lines)
    for region in filled_regions:
        pass
    axis.plot(*filled_regions.exterior.xy, facecolor='lightsalmon', zorder=10)


def generate_line(axis, line, color_points=None, borders=None, linewidth_cross=2.0):
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
        idx = [0,2,3,1]
        axis.add_patch(Polygon(np.array(i).reshape(4, 2)[idx], closed=True,
                               fill=True, color=color_points[j]))
        # axis.plot(i[:, 0].T, i[:, 1].T, color=color_points[j],
        #           linewidth=linewidth_cross)


def get_iso_level(z_list, method="equal_density", num_of_levels=5):
    return picture_worker.get_iso_levels(z_list, method, num_of_levels)


def get_half_lines(middlepoint, direction, length):
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


def get_color(iso_level, colorscheme, level_white=0, borders=None):
    if borders is None:
        borders = [0.5, 1.]
    return colorscheme["colorscheme"](
        colorscheme["colorscheme_name"],
        picture_worker.norm_levels(iso_level, *borders), lvl_white=level_white)


def get_line(gaussian, x_list, y_list, z_list, eigenvalue, eigenvector, colorscheme, num_of_levels=5, *args, **kwargs):
    iso_level = get_iso_level(z_list, num_of_levels=num_of_levels, *args, **kwargs)
    first_line, second_line = get_half_lines(gaussian[4], eigenvector, eigenvalue)
    logger.debug("------------------------------------------------------------")
    logger.debug("First Part of Line {}".format(first_line))
    logger.debug("Second Part of Line {}".format(second_line))
    logger.debug("------------------------------------------------------------")
    first_line = split_half_line(*first_line, iso_level, x_list, y_list, z_list)
    second_line = split_half_line(*second_line, iso_level[::-1], x_list, y_list, z_list)
    colors = get_color(get_iso_level(z_list, num_of_levels=num_of_levels + 1), colorscheme)
    return [*first_line, *second_line[1:]], [*colors, *colors[::-1]]


def generate_rectangle_from_line(line, eigenvector, length):
    new_line = []
    for i in line:
        first_point, second_point = get_half_lines(i, eigenvector, length)
        new_line.append((first_point[0], second_point[1]))
    return new_line


def get_cross(gaussian, colorscheme, length=3, *args, **kwargs):
    x_list, y_list, z_list = helper.get_gaussian(*gaussian)
    eigenvalues, eigenvectors = np.linalg.eig(gaussian[5])
    line_1, colors_1 = get_line(gaussian, x_list, y_list, z_list, eigenvalues[0], eigenvectors[1], colorscheme, *args,
                                **kwargs)
    line_2, colors_2 = get_line(gaussian, x_list, y_list, z_list, eigenvalues[1], eigenvectors[0], colorscheme, *args,
                                **kwargs)
    rectangle_1 = generate_rectangle_from_line(line_1, eigenvectors[0], length)
    rectangle_2 = generate_rectangle_from_line(line_2, eigenvectors[1], length)
    return rectangle_1, rectangle_2, colors_1, colors_2


def generate_cross(axis, line_1, line_2, colors_1, colors_2, linewidth_cross=2.0):
    logger.debug("Len Line 1: {}".format(len(line_1)))
    logger.debug("Len Colors 1: {}".format(len(colors_1)))
    generate_line(axis, line_1, colors_1, linewidth_cross=linewidth_cross)
    generate_line(axis, line_2, colors_2, linewidth_cross=linewidth_cross)


def generate_image_lines(gaussians, colorschemes, length=3, *args, **kwargs):
    logger.debug(gaussians)
    return [
        get_cross(i, j, length, *args, **kwargs) for i, j
        in
        zip(gaussians, colorschemes)]
