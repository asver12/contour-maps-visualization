import numpy as np
from skimage import color

import logging

logger = logging.getLogger(__name__)
c_handler = logging.StreamHandler()
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)

from contour_visualization import color_operations, hierarchic_blending_operator, helper, c_picture_worker, \
    color_schemes, iso_lines

try:
    from density_visualization import iso_levels
except ImportError as e:
    logger.warning(e)


def check_constrains(min_value, max_value):
    """
    checks if a value is not normalized correctly

    :param min_value: 0.
    :param max_value: 1.
    """
    if min_value < 0. or min_value > max_value:
        raise Exception("{} is not accepted as minimum value".format(min_value))
    if max_value > 1.:
        raise Exception("{} is not accepted as maximum value".format(max_value))


def get_color_middlepoint(iso_lines, min_value, max_value, normalize=True):
    """
    calculates the middlepoints all adjacent points in the list.

    :param iso_lines: list of points [x_1, ... , x_n]
    :param normalize: if true normalizes the arrary
    :param min_value: minimum of normalization
    :param max_value: maximum of normalization
    :return: array of middlepoints with size len(iso_lines) - 1
    """
    if normalize:
        norm = helper.norm_levels(iso_lines, min_value, max_value)
    else:
        norm = iso_lines
    return [(i + j) / 2 for i, j in zip(norm[:-1], norm[1:])]


def get_colorgrid(X, colorscheme, method="equal_density", num_of_levels=8, min_value=0., max_value=1., split=True,
                  min_border=None,
                  *args,
                  **kwargs):
    """
    Takes a 2D-Grid and maps it to a color-scheme. Therefor it generates a colormap with the given number of levels


    :param X: 2D-Grid with single values
    :param colorscheme: color-scheme from color_schemes
    :param num_of_levels:
    :param min_value:
    :param max_value:
    :param split:
    :param min_border:
    :return: 2D-Grid with color values from the color-scheme
    """
    check_constrains(min_value, max_value)
    logger.debug("Min: {} | Max: {}".format(min_value, max_value))

    # generate colors to chose from
    norm = iso_lines.get_iso_levels(X, method=method, num_of_levels=num_of_levels + 2)
    norm = get_color_middlepoint(norm, min_value, max_value)
    colormap = colorscheme(levels=norm, *args, **kwargs)

    # replace points in image with matching colors
    levels = iso_lines.get_iso_levels(X, method=method, num_of_levels=num_of_levels)
    return color_operations.map_colors(X, colormap, levels, split, lower_border=min_border)


def convert_rgb_image(img, color_space):
    """
    converts rgb-image into a given color-space. If the rgb-image is in rgba its converted to rgb.

    :param img: 2D-image with rgba or rgb values
    :param color_space: "lab" or "hsv"
    :return: 2D-image in given color-space if no color-space is given in rgb
    """
    if img.shape[-1] == 4:
        img = color.rgba2rgb(img)
    if color_space == "lab":
        img = color.rgb2lab(img)
        logger.debug("Lab-Color is used")
    elif color_space == "hsv":
        img = color.rgb2hsv(img)
        logger.debug("Hsv-Color is used")
    else:
        logger.debug("RGB-Color is used")
    return img


def convert_color_space_to_rgb(img, color_space):
    """
    converts image of given color-space into a rgb-image.

    :param img: 2D-image in given color-space
    :param color_space: "lab" or "hsv"
    :return: 2D-image in rgb if no color-space is given in rgb
    """
    if color_space == "lab":
        img = color.lab2rgb(img)
        logger.debug("Lab-Color is used")
    elif color_space == "hsv":
        img = color.hsv2rgb(img)
        logger.debug("Hsv-Color is used")
    else:
        logger.debug("Nothing was converted")
    return img


def generate_img_list(z_list, z_min, z_max, colorschemes, lower_border, upper_border, method="equal_density",
                      num_of_levels=8, *args, **kwargs):
    """
    generates a grid of hights with the corresponding colors from a density-grid.
    It maps the interval of each density into the total density-interval

    x = [density_1, ... , density_n]

    z_min = min(x)

    z_max = max(x)

    z_min <= min(density_i) <= max(density_i) <= z_max for all i in 1 , .. , n

    :param z_list: list of densities each one with the same shape [density_1, ... , density_n]
    :param z_min: minimal density occurring in the z_list min([density_1, ... , density_n])
    :param z_max: maximal density occurring in the z_list max([density_1, ... , density_n])
    :param colorschemes: colorschemes to use for each density-grid
    :param global_border: minimum factor for colorselection
    :param upper_border: maximum factor for colorselection
    :param method: method with which the distance between contour-lines is chosen
    :param num_of_levels: number of contour-lines to use
    :return: list of grid of colors in same shape as input list
    """
    img_list = []
    for z, colorscheme in zip(z_list, colorschemes):
        z_min_weight = (upper_border - lower_border) * (np.min(z) - z_min) / (z_max - z_min) + lower_border
        z_max_weight = (upper_border - lower_border) * (np.max(z) - z_min) / (z_max - z_min) + lower_border
        img, _ = get_colorgrid(z, *args, **colorscheme, method=method, num_of_levels=num_of_levels,
                               min_value=z_min_weight,
                               max_value=z_max_weight,
                               split=True, **kwargs)
        img_list.append(img)
    return img_list


def calculate_image(z_list, z_min, z_max, z_sum, colorschemes,
                    method="equal_density",
                    num_of_levels=8,
                    color_space="lab",
                    use_c_implementation=False,
                    mode="hierarchic",
                    blending_operator=hierarchic_blending_operator.porter_duff_source_over,
                    borders=None,
                    min_gauss=False,
                    lower_border=None,
                    lower_border_to_cut=0):
    """
    generates a merged image from multiple density-grids with same shape


    :param z_list: list of densities each one with the same shape [density_1, ... , density_n]
    :param z_min: minimal density occurring in the z_list min([density_1, ... , density_n])
    :param z_max: maximal density occurring in the z_list max([density_1, ... , density_n])
    :param colorschemes: colorschemes to use for each density-grid
    :param method: method with which the distance between contour-lines is chosen
    :param num_of_levels: number of contour-lines to use
    :param color_space: colorspace to merge the images in "rgb" or "lab"
    :param use_c_implementation: run the merging process with the c-implementation
    :param mode: sets the mode to use. Defaults is hierarchic and defaults to hierarchic
    :param blending_operator: operator with which the pictures are merged
    :param borders: min and max color from colorspace which is used from 0. to 1.
    :param min_gauss: uses minimal gaussian for the threshold for colors
    :param lower_border: min alpha value which is shown in each vis
    :return: colorgrid with merged image
    """
    if borders is None:
        borders = [0, 1]
    if lower_border:
        if lower_border_to_cut < 0:
            barrier = 0
        else:
            if min_gauss:
                barrier = z_max
                for i in z_list:
                    new_barrier = iso_lines.get_iso_levels(i, method, lower_border)[lower_border_to_cut]
                    if new_barrier < barrier:
                        barrier = new_barrier
            else:
                barrier = iso_lines.get_iso_levels(z_sum, method, lower_border)[lower_border_to_cut]
    else:
        barrier = None
    logger.debug("Lower Barrier is: {}".format(barrier))
    if len(z_list) == 1:
        img, _ = get_colorgrid(z_list[0], **colorschemes[0], method=method, num_of_levels=num_of_levels,
                               min_value=borders[0],
                               max_value=borders[1], split=True, min_border=barrier,
                               lvl_white=0 if not (barrier is None) else 1)
        return img, z_list[0]
    img_list = generate_img_list(z_list, z_min, z_max, colorschemes, *borders, method=method,
                                 num_of_levels=num_of_levels,
                                 min_border=barrier,
                                 lvl_white=0 if not (barrier is None) else 1)
    return combine_multiple_images_hierarchic(img_list, z_list, blending_operator, color_space=color_space,
                                              use_c_implementation=use_c_implementation, mode=mode)


def input_image(ax, distributions, z_list=None, z_min=None, z_max=None, z_sum=None, colorschemes=None,
                method="equal_density",
                num_of_levels=8,
                color_space="lab",
                use_c_implementation=False,
                mode="hierarchic",
                blending_operator=hierarchic_blending_operator.porter_duff_source_over,
                borders=None,
                min_gauss=False,
                lower_border=None,
                lower_border_to_cut=0, xlim=None, ylim=None):
    """
    inputs the contours of distributions into an matplotlib axis object

    :param ax: matplotlib axis
    :param distributions: list of :class:`contour_visualization.Distribution.Distribution`
    :param z_list: list of densities each one with the same shape [density_1, ... , density_n]
    :param z_min: minimal density occurring in the z_list min([density_1, ... , density_n])
    :param z_max: maximal density occurring in the z_list max([density_1, ... , density_n])
    :param colorschemes: colorschemes to use for each density-grid
    :param method: method with which the distance between contour-lines is chosen
    :param num_of_levels: number of contour-lines to use
    :param color_space: colorspace to merge the images in "rgb" or "lab"
    :param use_c_implementation: run the merging process with the c-implementation
    :param mode: sets the mode to use. Defaults is hierarchic and defaults to hierarchic
    :param blending_operator: operator with which the pictures are merged
    :param borders: min and max color from colorspace which is used from 0. to 1.
    """
    limits = helper.get_limits(distributions, xlim, ylim)
    if z_list is None:
        z_list = helper.generate_distribution_grids(distributions, limits=limits)
    if z_min is None:
        z_min, z_max, z_sum = helper.generate_weights(z_list)
    if colorschemes is None:
        colorschemes = color_schemes.get_colorbrewer_schemes()
    img, alpha = calculate_image(z_list, z_min, z_max, z_sum, colorschemes, method, num_of_levels, color_space,
                                 use_c_implementation, mode, blending_operator, borders, min_gauss=min_gauss,
                                 lower_border=lower_border, lower_border_to_cut=lower_border_to_cut)
    extent = [limits.x_min, limits.x_max, limits.y_min, limits.y_max]
    ax.imshow(img, extent=extent, origin='lower')


def _check_if_mixable(color_1, color_2):
    """
    checks if one of two colors is zero and returns 2 if color_1 is zero and 3 if color_2 is zero.
    If both colors are none zero returns 1

    :param color_1: rgb-color
    :param color_2: rgb-color
    :return: Int 1 if both none zero, 2 if first and 3 if second colors is zero
    """
    if all(abs(1. - x) < 1e-14 for x in color_1):
        return 2
    if all(abs(1. - x) < 1e-14 for x in color_2):
        return 3
    return 1


def _hierarchic_blending(args, blending_operator, i, image, image2, img, img2, j, kwargs, reduce, z_1, z_2,
                         z_new):
    logger.debug("{},{} = {}".format(i, j, img[i][j]))
    switch = {
        1: blending_operator(img[i][j], z_1[i][j], img2[i][j], z_2[i][j], *args, **kwargs),
        2: (img2[i][j], z_2[i][j]),
        3: (img[i][j], z_1[i][j]),
    }
    reduce[i][j], z_new[i][j] = switch.get(_check_if_mixable(image[i][j], image2[i][j]))
    logger.debug("{},{}: {} + {} = {} \n  max({}|{}) = {}".format(i, j, img[i][j], img2[i][j], reduce[i][j], z_1[i][j],
                                                                  z_2[i][j], z_new[i][j]))


def combine_multiple_images_hierarchic(images, z_values,
                                       blending_operator=hierarchic_blending_operator.porter_duff_source_over,
                                       color_space="lab",
                                       use_c_implementation=False, mode="hierarchic", *args, **kwargs):
    """
    Merges multiple pictures into one using a given blending-operator, the specific grade of blending is weighted by
    the z_values of each image. The pixel of each image is merged by its weight. From lowest to highest

    :param mode: sets the mode to use. Defaults is hierarchic and defaults to hierarchic
    :param use_c_implementation: only works with rgb and lab at the moment
    :param blending_operator: operator which is used to mix the images point by point
    :param images: [image_1, image_2, ... , image_n]
    :param z_values: [z_values_1, z_values_2, ... , z_values_n]
    :param color_space: None, "lab" or "hsv"
    :param args:
    :param kwargs:
    :return:
    """
    images = [convert_rgb_image(np.asarray(img), None) for img in images]
    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        import time
        start = time.time()
    if use_c_implementation:
        logger.debug("Using C-Implementation")
        if mode.lower() == "alpha_sum":
            logger.debug("Mode: alpha_sum")
            reduce, z_new = c_picture_worker.call_hierarchic_alpha_sum_merge(images, z_values, color_space)
        elif mode.lower() == "hierarchic":
            logger.debug("Mode: hierarchic")
            reduce, z_new = c_picture_worker.call_hierarchic_merge(images, z_values, color_space)
        elif mode.lower() == "alpha_sum_quad":
            logger.debug("Mode: alpha_sum_quad")
            reduce, z_new = c_picture_worker.call_l2_sum_merge(images, z_values, color_space)
        else:
            logger.debug("Mode: hierarchic---default")
            reduce, z_new = c_picture_worker.call_hierarchic_merge(images, z_values, color_space)
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            end = time.time()
            logger.debug("{}s elapsed".format(end - start))
        return reduce, z_new
    images = [convert_rgb_image(np.asarray(img), None) for img in images]
    if any(img.ndim != 3 for img in images):
        raise Exception("Images need a dimension of 3")
    np_images = [convert_rgb_image(img, color_space) for img in images]
    logger.debug(np_images)
    logger.debug(images)
    z_new = np.zeros([len(images[0]), len(images[0][0]), 1])
    reduce = np.zeros([len(images[0]), len(images[0][0]), len(images[0][0][0])])
    for i in range(len(images[0])):
        for j in range(len(images[0][0])):
            # sort z_values for the point and remember image it belongs to
            sorted_values = sorted([(k, x[i][j]) for k, x in enumerate(z_values)], key=lambda x: x[1])
            _hierarchic_blending(args, blending_operator, i, images[sorted_values[0][0]],
                                 images[sorted_values[1][0]],
                                 np_images[sorted_values[0][0]], np_images[sorted_values[1][0]],
                                 j, kwargs, reduce,
                                 z_values[sorted_values[0][0]],
                                 z_values[sorted_values[1][0]], z_new)
            if len(images) > 2:
                for k in range(2, len(sorted_values)):
                    # for readability
                    logger.debug("{},{} = {}".format(i, j, reduce[i][j]))
                    switch = {
                        1: blending_operator(reduce[i][j], z_new[i][j], np_images[sorted_values[k][0]][i][j],
                                             z_values[sorted_values[k][0]][i][j], *args, **kwargs),
                        2: (np_images[sorted_values[k][0]][i][j], z_values[sorted_values[k][0]][i][j]),
                        3: (reduce[i][j], z_new[i][j]),
                    }
                    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                        reduce_befor = reduce[i][j].copy()
                        logger.debug(convert_color_space_to_rgb([[reduce[i][j]]], color_space)[0][0])
                    # Select between three cases:
                    # 1: both pixel are not white, use blending_operator
                    # 2: first pixel is white, use second
                    # 3: second pixel is white, use first
                    reduce[i][j], z_new[i][j] = switch.get(
                        _check_if_mixable(convert_color_space_to_rgb([[reduce[i][j]]], color_space)[0][0],
                                          images[sorted_values[k][0]][i][j]))

                    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                        logger.debug(
                            "{},{}: {} + {} = {} \n  max({}|{}) = {}".format(i, j, reduce_befor,
                                                                             images[sorted_values[k][0]][i][j],
                                                                             reduce[i][j],
                                                                             z_new[i][j],
                                                                             z_values[sorted_values[k][0]][i][j],
                                                                             z_new[i][j]))
    reduce = convert_color_space_to_rgb(reduce, color_space)
    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        end = time.time()
        logger.debug("{}s elapsed".format(end - start))
    return reduce, z_new
