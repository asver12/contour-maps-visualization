import math
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from skimage import color, measure

import logging

logger = logging.getLogger(__name__)

from src import color_operations, hierarchic_blending_operator, helper, c_picture_worker, color_schemes

from density_visualization import iso_levels


def get_picture(x_min, x_max, y_min, y_max, X, Y, Z, levels, *args, **kwargs):
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    fig, ax = plt.subplots()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.patch.set_facecolor("none")
    ax.patch.set_edgecolor("none")
    ax.axis('off')
    plt.contourf(X, Y, Z, levels, *args, **kwargs)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, -1).copy(), w, h
    plt.figure()
    return img


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
        if old_min == None or old_max == None:
            return np.interp(levels, (levels.min(), levels.max()), (new_min_value, new_max_value))
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


def get_colors(colorscheme, levels, *args, **kwargs):
    """
    Creates for a given list and a colorscheme a list of num_of_levels colors

    :param colorscheme: color scheme with a variable levels
    :param levels: list with the position of the returned colors
    :param args: unnamed arguments for color_scheme
    :param kwargs: named arguments for color_scheme
    :return: list of size num_of_levels with colors
    """
    return colorscheme(levels=levels, *args, **kwargs)


def convert_color_to_colorspace(color, color_space="lab"):
    wow = _convert_rgb_image(np.array([[color]]), color_space)[0][0]
    return wow


def convert_color_to_rgb(color, color_space="lab"):
    wow = _convert_color_space_to_rgb(np.array([[color]]), color_space)[0][0]
    return wow


def _convert_rgb_image(img, color_space):
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


def _convert_color_space_to_rgb(img, color_space):
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


def get_image_list(gaussians, colorschemes, borders=None):
    """

    :param gaussians: [gaussian_1, ... , gaussian_n]
    :param colorschemes: [{colorscheme: color_scheme_function_1, colorscheme_name: colorscheme_name_1},
                            ... ]{colorscheme: color_scheme_function_n, colorscheme_name: colorscheme_name_n}]
    :param borders: range in which the pixel of the pictures are normalizes
    :return: Imagelist [2D-image_1, ... ,2D-image_n], Weights [2D-weight_1, ... ,2D-weight_n], Sum of all weights 2D-weight
    """
    if borders is None:
        borders = [0, 1]
    z_list = helper.generate_gaussians(gaussians)
    z_min, z_max, z_sum = helper.generate_weights(z_list)
    img_list = []
    lower_border = borders[0]
    upper_border = borders[1]
    for z, colorscheme in zip(z_list, colorschemes):
        z_min_weight = (upper_border - lower_border) * (np.min(z) - z_min) / (z_max - z_min) + lower_border
        z_max_weight = (upper_border - lower_border) * (np.max(z) - z_min) / (z_max - z_min) + lower_border
        img, _ = get_colorgrid(z, **colorscheme, min_value=z_min_weight, max_value=z_max_weight, split=True)
        img_list.append(img)
    return img_list, z_list, z_sum


def generate_image(gaussians, colorschemes, blending_operator=hierarchic_blending_operator.porter_duff_source_over,
                   method="equal_density",
                   num_of_levels=8,
                   color_space="lab",
                   use_c_implementation=False,
                   use_alpha_sum=True,
                   borders=None):
    """
    Generates an image from a list of gaussians and a colorscheme for each

    :param gaussians: [gaussian_1, ... , gaussian_n]
    :param colorschemes: [{colorscheme: color_scheme_function_1, colorscheme_name: colorscheme_name_1},
                            ... ]{colorscheme: color_scheme_function_n, colorscheme_name: colorscheme_name_n}]
    :param blending_operator: default is hierarchic-porter-duff-source-over
    :param use_c_implementation: if true the c-implementation is used which is approc. 3-4 times faster but only works with hierarchic-porter-duff-source-over rgb and lab
    :param borders: range in which the pixel of the pictures are normalizes
    :return: Weights [2D-weight_1, ... ,2D-weight_n], 2D-image, Sum of all weights 2D-weight
    """
    if borders is None:
        borders = [0, 1]
    if len(gaussians) == 1:
        z_list = helper.generate_gaussians(gaussians)
        img, _ = get_colorgrid(z_list[0], **colorschemes[0], num_of_levels=num_of_levels, split=True)
        return z_list, img, z_list[0]
    z_list = helper.generate_gaussians(gaussians)
    z_min, z_max, z_sum = helper.generate_weights(z_list)
    img_list = []
    lower_border = borders[0]
    upper_border = borders[1]
    for z, colorscheme in zip(z_list, colorschemes):
        z_min_weight = (upper_border - lower_border) * (np.min(z) - z_min) / (z_max - z_min) + lower_border
        z_max_weight = (upper_border - lower_border) * (np.max(z) - z_min) / (z_max - z_min) + lower_border
        img, _ = get_colorgrid(z, **colorscheme, method=method, num_of_levels=num_of_levels, min_value=z_min_weight, max_value=z_max_weight,
                               split=True)
        img_list.append(img)
    image, alpha = combine_multiple_images_hierarchic(blending_operator, img_list, z_list, color_space=color_space,
                                                      use_c_implementation=use_c_implementation,
                                                      use_alpha_sum=use_alpha_sum)
    return z_list, image, z_sum


def plot_images(images, gaussians, z_sums, colors=None, contour_lines_method="equal_density", contour_lines=True,
                contour_lines_weighted=True, num_of_levels=8,
                title="", with_axis=True, borders=None, linewidth=2, columns=5,
                bottom=0.0,
                left=0., right=2.,
                top=2.):
    """
    plots images for given gaussians

    :param contour_lines_weighted:
    :param colors: color of each gaussian. Gonna be plotted as legend
    :param images: [image_1, ... , image_n]
    :param gaussians: [[gaussian_1_1, ... gaussian_1_m], ... , [gaussian_n_1, ... gaussian_n_m]] gaussians from which the image is calculated
    :param z_sums: [z_sum_1, ... z_sum_n]
    :param contour_lines: if true plot while be returned with contour-lines
    :param num_of_levels: number of contour-lines returned
    :param columns: number of pictures next to each other
    :return:
    """

    logger.debug("{}".format(["mu_x", "variance_x", "mu_y", "variance_y"]))
    if len(images) == 1:
        title_j = ""
        if title == "" and gaussians:
            title_j = '\n'.join("{}".format(gau[4:-1]) for gau in gaussians[0])
        elif len(title) > columns:
            title_j = title[columns]
        color_legend = colors if colors else []
        plot_image(plt, images[0], gaussians[0], z_sums[0], color_legend, contour_lines_method, contour_lines,
                   contour_lines_weighted,
                   title_j, with_axis, num_of_levels, borders, linewidth)
        plt.subplots_adjust(bottom=bottom, left=left, right=right, top=top)
    else:
        for i in range(math.ceil(len(images) / columns)):
            subplot = images[i * columns:(i + 1) * columns]
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
                           contour_lines,
                           contour_lines_weighted,
                           title_j,
                           with_axis,
                           num_of_levels,
                           borders,
                           linewidth)
            else:
                for j in range(len(subplot)):
                    title_j = ""
                    if title == "" and gaussians:
                        title_j = '\n'.join("{}".format(gau[4:-1]) for gau in gaussians[j + i * columns])
                    elif len(title) > j + i * columns:
                        title_j = title[j + i * columns]
                    plot_image(axes[j], subplot[j], gaussians[j + i * columns], sub_sums[j], colors,
                               contour_lines_method,
                               contour_lines,
                               contour_lines_weighted,
                               title_j,
                               with_axis,
                               num_of_levels,
                               borders,
                               linewidth)
            fig.subplots_adjust(bottom=bottom, left=left, right=right, top=top)


def plot_image(axis, image, gaussians, z_sum, colors=None, contour_lines_method="equal_density", contour_lines=True,
               contour_lines_weighted=True, title="", with_axis=True,
               num_of_levels=8, borders=None, linewidth=2, lw=4):
    """

    :param axis: matplotlib axis at which the image is to plot on
    :param with_axis: if mathematical axis is shown or not
    :param contour_lines_method: method to plot the contourlines with
    :param linewidth: width of the contourlines
    :param borders: intervall in which the contourlinecolors are generated
    :param contour_lines_weighted: if the contourline colors should be weighted or not
    :param colors: [color_1, ... , color_n] colors of each gaussian
    :param image: 2D-image in rgb to plot
    :param gaussians: [gaussian_1, ... , gaussian_n] gaussians from which the image is calculated
    :param z_sum: 2D-weights of combined gaussians
    :param contour_lines: if true contourlines are plotted
    :param title: title of picture
    :param num_of_levels: number of contourlines
    :return:
    """
    if colors is None:
        colors = []
    if contour_lines:
        contour_lines_colorscheme = color_schemes.get_background_colorbrewer_scheme()
        generate_contour_lines(axis, z_sum, gaussians[0], contour_lines_colorscheme, contour_lines_method,
                               contour_lines_weighted, num_of_levels, borders, linewidth)
    extent = gaussians[0][:4]
    axis.imshow(image, extent=extent, origin='lower')
    if isinstance(axis, type(plt)):
        axis.title(title)
    else:
        axis.set_title(title)
    if colors:
        custom_lines = [Line2D([0], [0], color=colors[i], lw=lw) for i in
                        range(len(gaussians))]
        axis.legend(custom_lines, [i for i in range(len(gaussians))],
                    loc='upper left', frameon=False)

    if not with_axis:
        axis.axis("off")


def generate_contour_lines(ax, X, gaussian, contour_lines_colorscheme, contour_lines_method="equal_density",
                           contour_lines_weighted=True, num_of_levels=8, borders=None, linewidth=2):
    if borders is None:
        borders = [0.5, 1]
    levels = get_iso_levels(X, contour_lines_method, num_of_levels + 1)
    if contour_lines_weighted:
        contour_lines_colors = contour_lines_colorscheme["colorscheme"](
            contour_lines_colorscheme["colorscheme_name"],
            norm_levels(levels, *borders), lvl_white=0)
    else:

        contour_lines_colors = np.repeat(
            contour_lines_colorscheme["colorscheme"](contour_lines_colorscheme["colorscheme_name"],
                                                     [1.], lvl_white=0), num_of_levels + 1, axis=0)
    plot_contour_lines(ax, X, gaussian, levels, contour_lines_colors, linewidth=linewidth)


def plot_contour_lines(ax, X, gaussian, levels, colors, linewidth=2):
    contours = find_contour_lines(X, levels)
    for i, color in zip(contours[:len(levels)], colors[:len(levels)]):
        for contour in i:
            contour = helper.normalize_2d_array(np.asarray(contour), 0, X.shape[0], *gaussian[:2], 0, X.shape[1],
                                                *gaussian[2:4])

            ax.plot(contour[:, 1], contour[:, 0], linewidth=linewidth, color=color)


def get_normalized_contours(X, contours, gaussian):
    for i, k in enumerate(contours):
        for j, contour in enumerate(k):
            contours[i][j] = helper.normalize_2d_array(np.asarray(contour), 0, X.shape[0], *gaussian[:2], 0, X.shape[1],
                                                       *gaussian[2:4])
    return contours


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


def normalize_contour_lines(contour_lines, new_min, new_max, old_min, old_max):
    return [norm_levels(np.asarray(i), new_min, new_max, old_min, old_max) for i in contour_lines]


def find_contour_lines_bruteforce(z_value, img, num_of_levels, epsilon=0.00011):
    """

    :param z_value: 2D-weight
    :param img: 2D-image
    :param num_of_levels: number of contour-lines to return
    :param epsilon: everythink closer to the result of a contour-line is concidert as contour-line
    :return: 2D-image with contour-lines
    """
    x_min, x_max = np.min(z_value), np.max(z_value)
    levels = np.linspace(x_min, x_max, num_of_levels)
    logger.debug(levels)
    for i in range(len(z_value)):
        for j in range(len(z_value[0])):
            for k in levels[1:]:
                if z_value[i][j] > 1e-4 and abs(k - z_value[i][j]) < epsilon:
                    img[i][j] = np.array([0., 0., 0.])
    return img


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


def combine_two_images(blending_operator, image, image2, color_space="lab", *args, **kwargs):
    """
    Combines two images with shape [x,y,3/4]. If the 3 dimension is in shape 4 it is expected to be in rgab and will be
    transformed into srgb with shape 3.

    :param blending_operator: operator which is used to mix the two images point by point
    :param image: image with shape [x,y,3/4]
    :param image2: image with shape [x,y,3/4]
    :param color_space: colorspace to use atm lab and rgb are supported
    :param args: extra arguments for the blending operator
    :param kwargs: extra arguments for the blending operator
    :return: img whit shape [x,y,3] as srgb
    """
    image = _convert_rgb_image(np.asarray(image), None)
    image2 = _convert_rgb_image(np.asarray(image2), None)
    if image.ndim != 3 or image2.ndim != 3:
        raise Exception("Images need a dimension of 3")
    img = _convert_rgb_image(image, color_space)
    img2 = _convert_rgb_image(image2, color_space)
    logger.debug(img)
    logger.debug(image)
    reduce = np.zeros([len(img), len(img[0]), len(img[0][0])])
    for i in range(len(img)):
        for j in range(len(img[0])):
            # for readability
            _normal_blending(args, blending_operator, i, image, image2, img, img2, j, kwargs, reduce)
    reduce = _convert_color_space_to_rgb(reduce, color_space)
    return reduce


def _normal_blending(args, blending_operator, i, image, image2, img, img2, j, kwargs, reduce):
    logger.debug("{},{} = {}".format(i, j, img[i][j]))
    switch = {
        1: color_operations.blend_color(blending_operator, img[i][j], img2[i][j],
                                        *args, **kwargs),
        2: img2[i][j],
        3: img[i][j]
    }
    reduce[i][j] = switch.get(_check_if_mixable(image[i][j], image2[i][j]))
    logger.debug("{},{}: {} + {} = {}".format(i, j, img[i][j], img2[i][j], reduce[i][j]))


def combine_two_images_hierarchic(blending_operator, image, z_1, image2, z_2, color_space="lab", *args,
                                  **kwargs):
    """
    Combines two images with shape [x,y,3/4]. If the 3 dimension is in shape 4 it is expected to be in rgab and will be
    transformed into srgb with shape 3.

    :param color_space: None, "lab" or "hsv"
    :param blending_operator: operator which is used to mix the two images point by point
    :param image: image with shape [x,y,3/4]
    :param z_1: weights for the first image
    :param image2: image with shape [x,y,3/4]
    :param z_2: weights for the second image
    :return: img whit shape [x,y,3] as srgb
    """
    image = _convert_rgb_image(np.asarray(image), None)
    image2 = _convert_rgb_image(np.asarray(image2), None)
    if image.ndim != 3 or image2.ndim != 3:
        raise Exception("Images need a dimension of 3")
    img = np.asarray(image)
    img2 = np.asarray(image2)
    img = _convert_rgb_image(img, color_space)
    img2 = _convert_rgb_image(img2, color_space)
    logger.debug(image)
    logger.debug(img)
    z_new = np.zeros([len(img), len(img[0]), 1])
    reduce = np.zeros([len(img), len(img[0]), len(img[0][0])])
    for i in range(len(img)):
        for j in range(len(img[0])):
            # for readability
            _hierarchic_blending(args, blending_operator, i, image, image2, img, img2, j, kwargs, reduce, z_1,
                                 z_2, z_new)
    reduce = _convert_color_space_to_rgb(reduce, color_space)
    return reduce, z_new


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
    images = [_convert_rgb_image(np.asarray(img), None) for img in images]
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
    images = [_convert_rgb_image(np.asarray(img), None) for img in images]
    if any(img.ndim != 3 for img in images):
        raise Exception("Images need a dimension of 3")
    np_images = [_convert_rgb_image(img, color_space) for img in images]
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
                        logger.debug(_convert_color_space_to_rgb([[reduce[i][j]]], color_space)[0][0])
                    # Select between three cases:
                    # 1: both pixel are not white, use blending_operator
                    # 2: first pixel is white, use second
                    # 3: second pixel is white, use first
                    reduce[i][j], z_new[i][j] = switch.get(
                        _check_if_mixable(_convert_color_space_to_rgb([[reduce[i][j]]], color_space)[0][0],
                                          images[sorted_values[k][0]][i][j]))

                    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                        logger.debug(
                            "{},{}: {} + {} = {} \n  max({}|{}) = {}".format(i, j, reduce_befor,
                                                                             images[sorted_values[k][0]][i][j],
                                                                             reduce[i][j],
                                                                             z_new[i][j],
                                                                             z_values[sorted_values[k][0]][i][j],
                                                                             z_new[i][j]))
    reduce = _convert_color_space_to_rgb(reduce, color_space)
    if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
        end = time.time()
        logger.debug("{}s elapsed".format(end - start))
    return reduce, z_new
