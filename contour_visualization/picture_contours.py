import numpy as np
from skimage import color

import logging
from contour_visualization import color_operations, hierarchic_blending_operator, helper, c_picture_worker
from density_visualization import iso_levels

logger = logging.getLogger(__name__)




def check_constrains(min_value, max_value):
    if min_value < 0. or min_value > max_value:
        raise Exception("{} is not accepted as minimum value".format(min_value))
    if max_value > 1.:
        raise Exception("{} is not accepted as maximum value".format(max_value))


def get_iso_levels(X, method="equal_density", num_of_levels=8):
    """
    generates iso-lines to a given 2D-array
    :param X: picture to generate iso-levels to
    :param method: normal or equal_density available right now
    :param num_of_levels: number of iso-lines to generate
    :return: array([iso_line_1, ... , iso_line_n ])
    """
    if method == "equal_density":
        return iso_levels.equi_prob_per_level(X, k=num_of_levels)
    elif method == "equal_horizontal":
        return iso_levels.equi_horizontal_prob_per_level(X, k=num_of_levels)
    elif method == "equal_value":
        return iso_levels.equi_value(X, k=num_of_levels)
    return np.linspace(np.min(X), np.max(X), num_of_levels + 2)[1:-1]


def norm_levels(levels, new_min_value=0., new_max_value=1., old_min=None, old_max=None):
    levels = np.asarray(levels)
    if levels.size != 0:
        if old_min is None or old_max is None:
            return np.interp(levels, (min(levels), max(levels)), (new_min_value, new_max_value))
        else:
            return np.interp(levels, (old_min, old_max), (new_min_value, new_max_value))
    else:
        return levels


def get_color_middlepoint(iso_lines, min_value, max_value):
    norm = norm_levels(iso_lines, min_value, max_value)
    return [(i + j) / 2 for i, j in zip(norm[:-1], norm[1:])]


def get_colorgrid(X, colorscheme, method="equal_density", num_of_levels=8, min_value=0., max_value=1., split=True,
                  *args,
                  **kwargs):
    """
    Takes a 2D-Grid and maps it to a color-scheme. Therefor it generates a colormap with the given number of levels

    :param X: 2D-Grid with single values
    :param colorscheme: color-scheme from color_schemes
    :param num_of_levels:
    :param args:
    :return: 2D-Grid with color values from the color-scheme
    """
    check_constrains(min_value, max_value)
    logger.debug("Min: {} | Max: {}".format(min_value, max_value))

    # generate colors to chose from
    norm = get_iso_levels(X, method=method, num_of_levels=num_of_levels + 2)
    norm = get_color_middlepoint(norm, min_value, max_value)
    colormap = colorscheme(levels=norm, *args, **kwargs)

    # replace points in image with matching colors
    levels = get_iso_levels(X, method=method, num_of_levels=num_of_levels)
    return color_operations.map_colors(X, colormap, levels, split)


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
                      num_of_levels=8):
    img_list = []
    for z, colorscheme in zip(z_list, colorschemes):
        z_min_weight = (upper_border - lower_border) * (np.min(z) - z_min) / (z_max - z_min) + lower_border
        z_max_weight = (upper_border - lower_border) * (np.max(z) - z_min) / (z_max - z_min) + lower_border
        img, _ = get_colorgrid(z, **colorscheme, method=method, num_of_levels=num_of_levels, min_value=z_min_weight,
                               max_value=z_max_weight,
                               split=True)
        img_list.append(img)
    return img_list


def calculate_image(gaussians, z_list, z_min, z_max, colorschemes,
                    method="equal_density",
                    num_of_levels=8,
                    color_space="lab",
                    use_c_implementation=False,
                    use_alpha_sum=True,
                    blending_operator=hierarchic_blending_operator.porter_duff_source_over,
                    borders=None):
    if borders is None:
        borders = [0, 1]
    if len(gaussians) == 1:
        img, _ = get_colorgrid(z_list[0], **colorschemes[0], num_of_levels=num_of_levels, min_value=borders[0],
                               max_value=borders[1], split=True)
        return img, z_list[0]
    img_list = generate_img_list(z_list, z_min, z_max, colorschemes, *borders, method, num_of_levels)
    return combine_multiple_images_hierarchic(blending_operator, img_list, z_list, color_space=color_space,
                                              use_c_implementation=use_c_implementation,
                                              use_alpha_sum=use_alpha_sum)


def input_image(ax, gaussians, z_list, z_min, z_max, colorschemes,
                method="equal_density",
                num_of_levels=8,
                color_space="lab",
                use_c_implementation=False,
                use_alpha_sum=True,
                blending_operator=hierarchic_blending_operator.porter_duff_source_over,
                borders=None):
    img, alpha = calculate_image(gaussians, z_list, z_min, z_max, colorschemes, method, num_of_levels, color_space,
                                 use_c_implementation, use_alpha_sum, blending_operator, borders)
    extent = [*helper.get_x_values(gaussians), *helper.get_y_values(gaussians)]

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


def combine_multiple_images_hierarchic(blending_operator, images, z_values, color_space="lab",
                                       use_c_implementation=False, use_alpha_sum=True, *args,
                                       **kwargs):
    """
    Merges multiple pictures into one using a given blending-operator, the specific grade of blending is weighted by
    the z_values of each image. The pixel of each image is merged by its weight. From lowest to highest
    :param use_alpha_sum:
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
        if use_alpha_sum:
            reduce, z_new = c_picture_worker.call_hierarchic_alpha_sum_merge(images, z_values, color_space)
        else:
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
