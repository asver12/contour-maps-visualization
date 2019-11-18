import numpy as np
from matplotlib.patches import Polygon
from scipy import linalg
import itertools
import pyclipper

from contour_visualization import helper, picture_contours, color_schemes, hierarchic_blending_operator, iso_lines

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
    r_point = rectangle_1[0]
    c_point = colors[0]
    z_point = z_weights[0]

    for point, color, z_weight in zip(rectangle_1[1:], [*colors[1:], None], [*z_weights[1:], None]):
        if not np.array_equal(c_point, color):
            poly = (convert_to_int(r_point[0]), convert_to_int(point[0]), convert_to_int(point[1]),
                    convert_to_int(r_point[1]))
            yield poly, c_point, z_point
            r_point = point
            c_point = color
            z_point = z_weight


def join_inner_polys(crosses):
    return crosses


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


def poly_union(polys):
    poly_return = []
    for cross_1, cross_2 in itertools.combinations(polys, 2):
        for a in cross_1:
            for b in cross_2:
                try:
                    solution = get_intersection([a[0], b[0]])
                    if solution:
                        poly_return.append([solution[0], [a[1], b[1]], [a[2], b[2]]])
                except:
                    pass
    return poly_return


def multi_union(poly_return, number_gaussians, int_condition=1000):
    new_polys = poly_return
    i = 0
    next_polys = []
    while len(new_polys) > 2 and i < + 1 and len(next_polys) < int_condition:
        logger.debug("Iteration[{}]: {}".format(number_gaussians, i))
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
    return poly_return


def get_fill_regions(cross_lines, *args, **kwargs):
    # create rectangles
    crosses = generate_polys(cross_lines)
    # find all intersections of 2 polygons
    poly_return = poly_union(crosses)

    # find all intersections of the intersections from 2 polygons
    poly_return = multi_union(poly_return, len(crosses), *args, **kwargs)
    return poly_return


def mix_colors(colors, z_weights, mode="hierarchic",
               blending_operator=hierarchic_blending_operator.porter_duff_source_over,
               color_space="lab"):
    colors = convert_color_to_colorspace(colors, color_space)
    if mode == "alpha_sum":
        logger.debug("Mode: alpha_sum")
        sum_z_weights = sum(z_weights)
        # normalize values
        colors = [list(map(lambda x: x * (weight / sum_z_weights), col)) for col, weight in zip(colors, z_weights)]
        # sum each value in list
        colors = [sum(x) for x in zip(*colors)]
    elif mode == "alpha_sum_quad":
        logger.debug("Mode: alpha_sum_quad")
        sum_z_weights = sum(map(lambda x: x ** 2, z_weights))
        colors = [list(map(lambda x: x * (weight ** 2 / sum_z_weights), col)) for col, weight in zip(colors, z_weights)]
        colors = [sum(x) for x in zip(*colors)]
    else:
        logger.debug("Mode: hierarchic")
        z_weights = sorted(list(zip(z_weights, colors)), key=lambda x: x[0])
        z_weight = z_weights[0][0]
        color = z_weights[0][1]
        for z_wei, col in z_weights[1:]:
            color, z_weight = blending_operator(color, z_weight, col, z_wei)
        colors = color
    logger.debug("Mixed Color: {}".format(colors))
    return convert_color_to_rgb(colors, color_space)


def fill_between_lines(axis, cross_lines, blending_operator=hierarchic_blending_operator.porter_duff_source_over,
                       mode="alpha_sum",
                       color_space="lab"):
    filled_regions = get_fill_regions(cross_lines)
    for region in filled_regions:
        idx, col = filter_order_color(region[1])
        z_weig = filter_order_list(region[2], idx)
        if len(region[2]) > 2:
            logger.debug("col[{}]: {}".format(region[1], col))
            logger.debug("z[{}]: {}".format(region[2], z_weig))
        color = mix_colors(col, z_weig, mode=mode, blending_operator=blending_operator, color_space=color_space)
        axis.add_patch(Polygon([convert_to_float(point) for point in region[0]], closed=True,
                               fill=True, edgecolor=None, facecolor=color, aa=True, linewidth=0.))


def generate_line(axis, line, color_points=None, borders=None, fill=True, linewidth=0., use_colors=True):
    if borders is None:
        borders = [0.5, 1]
    if color_points is None:
        contour_lines_colorscheme = color_schemes.get_colorbrewer_schemes()[0]
        color_points = contour_lines_colorscheme["colorscheme"](
            contour_lines_colorscheme["colorscheme_name"],
            helper.norm_levels(np.linspace(0, 1, len(line) - 1), *borders), lvl_white=0)
    points = np.array(list(zip(line[:-1], line[1:])))
    logger.debug("Points: {}".format(points))
    for j, i in enumerate(points):
        idx = [0, 2, 3, 1]
        axis.add_patch(Polygon(np.array(i).reshape(4, 2)[idx], closed=True,
                               fill=fill, edgecolor=color_points[j] if use_colors else "black",
                               facecolor=color_points[j] if use_colors else "black", aa=True,
                               linewidth=linewidth))


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
    _relative_start_point, _relative_end_point = get_broader_interval(startpoint, endpoint, .1)
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


def get_distance(point_1, point_2):
    """
    calc the length for a two points in a n-D-grid

    :param point_1: coordinate (x_1, ..., x_n)
    :param point_2: coordinate(y_1, ... , y_n)
    :return: distance between both coordinates
    """
    if len(point_1) != len(point_2):
        raise ValueError("Length of point 1 not equale length of point 2[{} != {}]".format(len(point_1), len(point_2)))
    return sum((point_2[i] - point_1[i]) ** 2 for i in range(len(point_1))) ** (1 / 2)


def get_direction(point_1, point_2):
    if len(point_1) != len(point_2):
        raise ValueError("Length of point 1 not equale length of point 2[{} != {}]".format(len(point_1), len(point_2)))
    direction = [i - j for i, j in zip(point_2, point_1)]
    return direction / np.linalg.norm(direction)


def get_broader_interval(point_1, point_2, percentage=0.1):
    direction = get_direction(point_2, point_1)
    distance = get_distance(point_2, point_1)
    return point_1 + direction * distance * percentage, point_2 - direction * distance * percentage


def map_points(points, value_line, point_mapping):
    """
    Finds the closest x and y coordinate by the density for each iso-level. The density is given by a line.
    The first and the last point on the value_line are exluded!!!

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
        index = 0
        best_match = value_line[0]
        point = None
        for j, z in enumerate(value_line[1:], 1):
            if abs(i - best_match) > abs(i - z):
                logger.debug("New z found: {} [old: {}]".format(z, best_match))
                best_match = z
                point = point_mapping[j]
                index = j
        if point:
            if index < len(value_line) - 1:
                split_points.append(point)
        else:
            logger.warning("points in grid of z-coordinates too similar")
    logger.debug("Remaining points: {}".format(points))
    return split_points


def find_point_indices(point, x_list, y_list):
    if len(point) != 2:
        raise ValueError("Point-length {} != 2 [{}]".format(len(point), point))
    if not x_list[0][0] <= point[0] <= x_list[0][-1]:
        raise ValueError("Point {} not in Interval {} for x-value".format(point[0], (x_list[0][0], x_list[0][-1])))
    if not y_list[:, 0][0] <= point[1] <= y_list[:, 0][-1]:
        raise ValueError("Point {} not in Interval {} for y-value".format(point[1],
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

    def get_limits(line, index):
        return min(line, key=lambda k: k[index])[index], max(line, key=lambda k: k[index])[index]

    first_line, second_line = get_half_lines(gaussian.means, eigenvector, eigenvalue)
    limits = helper.Limits(*get_limits([*first_line, *second_line], 0), *get_limits([*first_line, *second_line], 1))
    logger.debug("Cross-limits: {}".format(limits))
    _, _, z_list = gaussian.get_density_grid(x_min=limits.x_min, x_max=limits.x_max, y_min=limits.y_min,
                                             y_max=limits.y_max)
    iso_level = iso_lines.get_iso_levels(z_list, method=method, num_of_levels=num_of_levels)
    logger.debug("------------------------------------------------------------")
    logger.debug("First Part of Line {}".format(first_line))
    logger.debug("Second Part of Line {}".format(second_line))
    logger.debug("Iso-Level for splitting {}".format(iso_level))
    logger.debug("------------------------------------------------------------")
    first_line = split_half_line(gaussian, *first_line, iso_level)
    second_line = split_half_line(gaussian, *second_line, iso_level[::-1])
    iso_lvl = iso_lines.get_iso_levels(gaussian.get_density_grid()[2], method=method,
                                       num_of_levels=num_of_levels + 2)
    logger.debug("Min/max-value: {}/{}".format(min_value, max_value))
    logger.debug("Iso-Level for colors: {}".format(iso_lvl))
    iso_lvl = picture_contours.get_color_middlepoint(iso_lvl, min_value, max_value)
    colors = get_color(iso_lvl, colorscheme)
    logger.debug("Colors: {}".format(colors))
    return [*first_line, *second_line[1:]], [*colors[-len(first_line) + 1:], *colors[len(second_line[1:])::-1]], [
        *iso_lvl, *iso_lvl[::-1]]


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
        # split each point and move it orthogonal to the line
        first_point, second_point = get_half_lines(i, eigenvector, broad)
        new_line.append((first_point[0], second_point[1]))
    return new_line


def get_cross(gaussian, colorscheme, min_value=0., max_value=1., broad="5%", same_broad=True,
              length_multiplier=2. * np.sqrt(2.), *args, **kwargs):
    """
    Caculates the two rectangles with matching colors and the iso-level for a cross.

    :param gaussian: Gaussdistribution with means and cov-matrix
    :param colorscheme: colorscheme to use
    :param min_value: minimum color-percentage to take
    :param max_value: maximum color-percentage to take
    :param broad: broad of the cross
    :param same_broad: cross size is caculated with the smaller side of the cross otherwise it is calculated for both
    :param length_multiplier: value to multiply the eigenvalues with. results shorter or longer cross
    :return:
    """
    if not hasattr(gaussian, "cov_matrix"):
        raise AttributeError("[{}] property 'cov_matrix is missing".format(type(gaussian)))
    if not hasattr(gaussian, "means"):
        raise AttributeError("[{}] property 'mean' is missing".format(type(gaussian)))
    picture_contours.check_constrains(min_value, max_value)
    eigenvalues, eigenvectors = linalg.eigh(gaussian.cov_matrix)
    eigenvalues = length_multiplier * np.sqrt(eigenvalues)
    if eigenvalues[0] < eigenvalues[1]:
        eigenvalues[0], eigenvalues[1] = eigenvalues[1], eigenvalues[0]
    logger.debug("Distribution: {}".format(gaussian))
    logger.debug("Eigenvalues: {}".format(eigenvalues))
    logger.debug("Eigenvectors: {}".format(eigenvectors))
    line_long, colors_long, z_lvl_long = get_line(gaussian, eigenvalues[0],
                                                  eigenvectors[1],
                                                  colorscheme,
                                                  min_value, max_value,
                                                  *args,
                                                  **helper.filter_kwargs(get_line, **kwargs))
    line_short, colors_short, z_lvl_short = get_line(gaussian, eigenvalues[1],
                                                     eigenvectors[0],
                                                     colorscheme,
                                                     min_value, max_value,
                                                     *args,
                                                     **helper.filter_kwargs(get_line, **kwargs))
    broad_short, broad_long = get_broad(broad, line_short, line_long, same_broad)
    # since eigenvectors are orthogonal to each other just change them for the direction
    rectangle_1 = generate_rectangle_from_line(line_short, eigenvectors[1], broad_short)
    rectangle_2 = generate_rectangle_from_line(line_long, eigenvectors[0], broad_long)
    return rectangle_1, rectangle_2, colors_short, colors_long, z_lvl_short, z_lvl_long


def get_broad(broad, line_short, line_long, same_length=True):
    """
    input ether a number or a relative number in format 50.0% and returns broad fitting to it


    :param broad: int or str in format 50.0% which is used to get a relative broad
    :param line_short: line in format [(x_1, y_1), ... , x_n, y_n)
    :param line_long: line in format [(x_1, y_1), ... , x_n, y_n)
    :param same_length: if set true calculates only the relative distance for the smaller cross
    :return: broad for line short, broad for line long

    """
    broad_short = broad
    broad_long = broad
    if isinstance(broad, str):
        try:
            broad = float(broad[:-1]) / 100
            if same_length:
                broad_short = broad * get_distance(line_short[0], line_short[-1])
                broad_long = broad_short
            else:
                broad_short = broad * get_distance(line_short[0], line_short[-1])
                broad_long = broad * get_distance(line_long[0], line_long[-1])
        except ValueError:
            raise ValueError("Broad not in format 50.0%[{}]".format(broad))
        except Exception as e:
            raise e
    return broad_short, broad_long


def generate_cross(axis, line_1, line_2, colors_1, colors_2, *args, **kwargs):
    logger.debug("Len Line 1: {}".format(len(line_1)))
    logger.debug("Len Colors 1: {}".format(len(colors_1)))
    generate_line(axis, line_1, colors_1, *args, **helper.filter_kwargs(generate_line, **kwargs))
    generate_line(axis, line_2, colors_2, *args, **helper.filter_kwargs(generate_line, **kwargs))


def generate_crosses(gaussians, z_list, z_min, z_max, colorschemes, broad="50%", same_broad=True,
                     length_mutliplier=2. * np.sqrt(2.), borders=None,
                     *args, **kwargs):
    if borders is None:
        borders = [0, 1]
    lower_border = borders[0]
    upper_border = borders[1]
    z_weights = []
    for z in z_list:
        z_min_weight = (upper_border - lower_border) * (np.min(z) - z_min) / (z_max - z_min) + lower_border
        z_max_weight = (upper_border - lower_border) * (np.max(z) - z_min) / (z_max - z_min) + lower_border
        z_weights.append([z_min_weight, z_max_weight])
    return [get_cross(i, j, *k, broad, same_broad, length_mutliplier, *args, **kwargs) for i, j, k in
            zip(gaussians, colorschemes, z_weights)]


def input_crosses(ax, gaussians, z_list, z_min, z_max, colorschemes, broad=3, same_broad=True,
                  length_multiplier=2. * np.sqrt(2.), borders=None, color_space="lab", fill=True, cross_fill=True,
                  blending_operator=hierarchic_blending_operator.porter_duff_source_over, mode="hierarchic",
                  *args,
                  **kwargs):
    if not hasattr(gaussians[0], "cov_matrix"):
        raise AttributeError("[{}] property 'cov_matrix is missing".format(type(gaussians[0])))
    if not hasattr(gaussians[0], "means"):
        raise AttributeError("[{}] property 'mean' is missing".format(type(gaussians[0])))
    cross_lines = generate_crosses(gaussians, z_list, z_min, z_max, colorschemes, broad, same_broad,
                                   length_multiplier,
                                   borders, *args,
                                   **kwargs)
    for cross in cross_lines:
        generate_cross(ax, *cross[:4], fill=cross_fill, *args, **kwargs)
    if fill:
        fill_between_lines(ax, cross_lines, blending_operator=blending_operator, mode=mode, color_space=color_space)


def convert_color_to_rgb(color, color_space="lab"):
    return picture_contours.convert_color_space_to_rgb(np.array([[color]]), color_space)[0][0]


def convert_color_to_colorspace(color, color_space="lab"):
    return picture_contours.convert_rgb_image(np.array([[color]]), color_space)[0][0]
