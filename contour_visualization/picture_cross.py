import copy

import numpy as np
from matplotlib.patches import Polygon
from scipy import linalg
import itertools
import pyclipper

from contour_visualization import helper, picture_contours, color_schemes, hierarchic_blending_operator
from contour_visualization.Gaussian import Gaussian

import logging

logger = logging.getLogger(__name__)


def filter_order_list(list_1, idx):
    return [list_1[i] for i in idx]


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
    color = convert_color_to_colorspace(colors[0], color_space)
    z_weights = sorted(list(zip(z_weights, colors)), key=lambda x: x[0])
    z_weight = z_weights[0][0]
    for z_wei, col in z_weights:
        col = convert_color_to_colorspace(col, color_space)
        color, z_weight = hierarchic_blending_operator.porter_duff_source_over(color, z_weight, col, z_wei)
    return convert_color_to_rgb(color, color_space)


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
            picture_contours.norm_levels(np.linspace(0, 1, len(line) - 1), *borders), lvl_white=0)
    points = np.array(list(zip(line[:-1], line[1:])))
    logger.debug("Points: {}".format(points))
    for j, i in enumerate(points):
        idx = [0, 2, 3, 1]
        axis.add_patch(Polygon(np.array(i).reshape(4, 2)[idx], closed=True,
                               fill=True, edgecolor=color_points[j], facecolor=color_points[j], aa=True, linewidth=0.))


def get_half_lines(middlepoint, direction, length):
    """
    returns two lines, (startpoint, middlepoint) and (middlepoint, endpoint).

    :param middlepoint: point where the two lines cross
    :param direction: slope of the  line
    :param length:
    :return:
    """
    if len(middlepoint) != 2:
        raise ValueError("Invalid middlepoint given len({}) != 2".format(middlepoint))
    if len(direction) != 2:
        raise ValueError("Invalid slope given len({}) != 2".format(direction))
    if np.linalg.norm(direction) != 1:
        direction = direction / np.linalg.norm(direction)
    startpoint = middlepoint[0] - direction[0] * length, middlepoint[1] - direction[1] * length
    endpoint = middlepoint[0] + direction[0] * length, middlepoint[1] + direction[1] * length
    return (startpoint, middlepoint), (middlepoint, endpoint)


def split_half_line(gaussian, startpoint, endpoint, iso_level):
    num = 100
    logger.debug("Startpoint: {}".format(startpoint))
    logger.debug("Endpoint: {}".format(endpoint))
    _relative_start_point_x, _relative_end_point_x = get_broader_interval(startpoint[0], endpoint[0])
    _relative_start_point_y, _relative_end_point_y = get_broader_interval(startpoint[1], endpoint[1])
    _relative_start_point = (_relative_start_point_x, _relative_start_point_y)
    _relative_end_point = (_relative_end_point_x, _relative_end_point_y)
    logger.debug("relative startpoint {}".format(_relative_start_point))
    logger.debug("relative endpoint {}".format(_relative_end_point))
    x_1, y_1 = np.linspace(_relative_start_point[0], _relative_end_point[0], num), \
               np.linspace(_relative_start_point[1], _relative_end_point[1], num)
    used_points = [(i, j) for i, j in zip(x_1, y_1)]
    zi = [gaussian.get_density(x) for x in used_points]
    logger.debug("------------------------------------------------------------")
    logger.debug("ISO-Line: {}".format(iso_level))
    logger.debug("Points on Line: {}".format(used_points))
    logger.debug("Points on Z: {}".format(zi))
    logger.debug("------------------------------------------------------------")
    split_points = map_points(iso_level, zi, used_points)
    logger.debug("Line: {}".format([startpoint, *split_points, endpoint]))

    return [startpoint, *split_points, endpoint]


def get_broader_interval(point_1, point_2, percentage=0.1):
    if point_1 > point_2:
        return point_1 + get_relative_length(point_2, point_1) * percentage, \
               point_2 - get_relative_length(point_2, point_1) * percentage
    return point_1 - get_relative_length(point_1, point_2) * percentage, \
           point_2 + get_relative_length(point_1, point_2) * percentage


def get_relative_length(start, end):
    return abs(end - start)


def map_points(points, value_line, point_mapping):
    """
    Finds the closest x and y coordinate by the density for each iso-level. The density is given by a line.

    :param points: points for which the x,y-coordinates are wanted
    :param value_line: line on in which the points are approximated
    :param point_mapping: matching x,y-coordinates for the line
    :return:
    """
    if len(points) == 0:
        raise ValueError(
            "Points[{} - {}] outside of datarange[{} - {}]".format(min(points), max(points), min(value_line),
                                                                   max(value_line)))
    if len(value_line) != len(point_mapping):
        raise ValueError("Values[{}] and mapping[{}] doesnt fit".format(len(value_line), len(point_mapping)))
    split_points = []
    for i in points:
        best_match = value_line[0]
        point = None
        for j, z in enumerate(value_line[1:], 1):
            if abs(i - best_match) > abs(i - z):
                # logger.debug("New z found: {} [old: {}]".format(z, best_match))
                best_match = z
                point = point_mapping[j]
        if point:
            split_points.append(point)
        else:
            logger.warning("points in grid of z-coordinates to similar")
    logger.debug("Remaining points: {}".format(points))
    return split_points


def find_point_indices(point, x_list, y_list):
    if len(point) != 2:
        raise ValueError("Point-length {} != 2 [{}]".format(len(point), point))
    if not x_list[0][0] <= point[0] <= x_list[0][-1]:
        raise ValueError("Point {} not in Intervall {} for x-value".format(point[0], (x_list[0][0], x_list[0][-1])))
    if not y_list[:, 0][0] <= point[1] <= y_list[:, 0][-1]:
        raise ValueError("Point {} not in Intervall {} for y-value".format(point[1],
                                                                           (y_list[:, 0][0], y_list[:, 0][-1])))
    index_x = helper.find_index(point[0], x_list[0].flatten())
    index_y = helper.find_index(point[1], y_list[:, 0].flatten())
    return index_x, index_y


def get_color(iso_level, colorscheme, level_white=0):
    return colorscheme["colorscheme"](
        colorscheme["colorscheme_name"], iso_level, lvl_white=level_white)


def get_line(gaussian, eigenvalue, eigenvector, colorscheme, min_value=0., max_value=1.,
             method="equal_density",
             num_of_levels=5):
    """
    generates a line with matching colors and iso-lines from a given gaussian and a fitting grid of densities

    :param gaussian: Gaussian class with means and cov-matrix
    :param eigenvalue:
    :param eigenvector:
    :param colorscheme:
    :param min_value:
    :param max_value:
    :param method:
    :param num_of_levels:
    :return:
    """
    _, _, z_list = gaussian.get_density_grid()
    iso_level = picture_contours.get_iso_levels(z_list, method=method, num_of_levels=num_of_levels)
    first_line, second_line = get_half_lines(gaussian.means, eigenvector, eigenvalue)
    logger.debug("------------------------------------------------------------")
    logger.debug("First Part of Line {}".format(first_line))
    logger.debug("Second Part of Line {}".format(second_line))
    logger.debug("------------------------------------------------------------")
    first_line = split_half_line(gaussian, *first_line, iso_level)
    second_line = split_half_line(gaussian, *second_line, iso_level[::-1])
    iso_lvl = picture_contours.get_iso_levels(z_list, method=method, num_of_levels=num_of_levels + 2)
    logger.debug("Min/max-value: {}/{}".format(min_value, max_value))
    logger.debug("Iso-Level: {}".format(iso_lvl))
    iso_lvl = picture_contours.get_color_middlepoint(iso_lvl, min_value, max_value)
    colors = get_color(iso_lvl, colorscheme)
    logger.debug("Colors: {}".format(colors))
    return [*first_line, *second_line[1:]], [*colors[-len(first_line) + 1:], *colors[len(second_line[1:])::-1]], [
        *iso_lvl, *iso_lvl[::-1]]
    # return [*first_line, *second_line[1:]], [*colors[:len(first_line)], *colors[len(second_line[1:])::-1]], [
    #     *iso_lvl[:len(first_line)], *iso_lvl[len(second_line[1:])::-1]]


"""
[array([0.95630982, 0.97199609, 1.        , 1.        ]), array([0.92511198, 0.94722686, 0.99428568, 1.        ]), 
array([0.87282719, 0.92262225, 0.96968108, 1.        ]), array([0.79999307, 0.87450518, 0.94509573, 1.        ]), 
array([0.75692278, 0.85013561, 0.91987907, 1.        ]), array([0.70031047, 0.8259999 , 0.8979728 , 1.        ])]

[0.00351246 0.00648906 0.00947214 0.01243449 0.01540133 0.01837032 0.0213396 ]
[0.00351246 0.00648906 0.00947214 0.01243449 0.01540133 0.01837032 0.0213396 ]
"""


def generate_rectangle_from_line(line, eigenvector, broad):
    """
    Splites line into two lines. The angle is given by the eigenvector. The Length defines the distance to the given line.
    For the crosses in particular with the correct eigenvector the new two lines are orthogonal to the given one since
    eigenvectors are orthogonal to each other

    new line :  . . . . . .
                |
                | broad
    old line :  . . . . . .
                |
                | broad
    new line :  . . . . . .

    :param line: line which is split into rectangles. [point_1, ... point_n]
    :param eigenvector: vector which defines the angle in which the lines are split.
    The new two lines are always parallel
    :param broad: Distance between each new line and the given one
    :return: list of tuple each tuple giving two points for the new lines [(line_1_1, line_2_1), ... , (line_1_n, line_2_n)]
    """
    new_line = []
    for i in line:
        first_point, second_point = get_half_lines(i, eigenvector, broad)
        new_line.append((first_point[0], second_point[1]))
    return new_line


def get_cross(gaussian, colorscheme, min_value=0., max_value=1., broad=3, *args, **kwargs):
    """
    Caculates the two rectangles with matching colors and the iso-level for a cross.

    :param gaussian: Gaussdistribution with means and cov-matrix
    :param colorscheme: colorscheme to use
    :param min_value: minimum color-percentage to take
    :param max_value: maximum color-percentage to take
    :param broad: broad of the cross
    :return:
    """
    if not hasattr(gaussian, "cov_matrix"):
        raise AttributeError("[{}] property 'cov_matrix is missing".format(type(gaussian)))
    if not hasattr(gaussian, "means"):
        raise AttributeError("[{}] property 'mean' is missing".format(type(gaussian)))
    picture_contours.check_constrains(min_value, max_value)
    eigenvalues, eigenvectors = linalg.eigh(gaussian.cov_matrix)
    if eigenvalues[0] < eigenvalues[1]:
        eigenvalues[0], eigenvalues[1] = eigenvalues[1], eigenvalues[0]
    logger.debug("Distribution: {}".format(gaussian))
    logger.debug("Eigenvalues: {}".format(eigenvalues))
    logger.debug("Eigenvectors: {}".format(eigenvectors))
    line_1, colors_1, z_lvl_1 = get_line(gaussian, eigenvalues[0],
                                         eigenvectors[1],
                                         colorscheme,
                                         min_value, max_value,
                                         *args,
                                         **kwargs)
    line_2, colors_2, z_lvl_2 = get_line(gaussian, eigenvalues[1],
                                         eigenvectors[0],
                                         colorscheme,
                                         min_value, max_value,
                                         *args,
                                         **kwargs)
    rectangle_1 = generate_rectangle_from_line(line_1, eigenvectors[0], broad)
    rectangle_2 = generate_rectangle_from_line(line_2, eigenvectors[1], broad)
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
    if not hasattr(gaussians[0], "cov_matrix"):
        raise AttributeError("[{}] property 'cov_matrix is missing".format(type(gaussians[0])))
    if not hasattr(gaussians[0], "means"):
        raise AttributeError("[{}] property 'mean' is missing".format(type(gaussians[0])))
    cross_lines = genenerate_crosses(gaussians, z_list, z_min, z_max, colorschemes, length, borders, *args, **kwargs)
    for cross in cross_lines:
        generate_cross(ax, *cross[:4])
    fill_between_lines(ax, cross_lines, color_space=color_space)


def convert_color_to_rgb(color, color_space="lab"):
    return picture_contours.convert_color_space_to_rgb(np.array([[color]]), color_space)[0][0]


def convert_color_to_colorspace(color, color_space="lab"):
    return picture_contours.convert_rgb_image(np.array([[color]]), color_space)[0][0]
