from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from numpy.core._multiarray_umath import ndarray
from skimage import color

from src import color_operations


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


def get_colorgrid(X, color_scheme, num_of_levels, split=True, *args, **kwargs):
    """
    Takes a 2D-Grid and maps it to a color-scheme. Therefor it generates a colormap with the given number of levels

    :param X: 2D-Grid with single values
    :param color_scheme: color-scheme from color_schemes
    :param num_of_levels:
    :param args:
    :return: 2D-Grid with color values from the color-scheme
    """
    z_min, z_max = np.min(X), np.max(X)
    levels = np.linspace(z_min, z_max, num_of_levels)  # [1:6] ?
    colormap = color_scheme(levels=levels, *args, **kwargs)
    return color_operations.map_colors(X, colormap, levels, split)


def _convert_rgb_image(img, color_space, verbose=False):
    if img.shape[-1] == 4:
        img = color.rgba2rgb(img)
    if color_space == "lab":
        img = color.rgb2lab(img)
        if verbose:
            print("Lab-Color is used")
    elif color_space == "hsv":
        img = color.rgb2hsv(img)
        if verbose:
            print("Hsv-Color is used")
    else:
        if verbose:
            print("RGB-Color is used")
    return img


def _convert_color_space_to_rgb(img, color_space, verbose=False):
    if color_space == "lab":
        img = color.lab2rgb(img)
        if verbose:
            print("Lab-Color is used")
    elif color_space == "hsv":
        img = color.hsv2rgb(img)
        if verbose:
            print("Hsv-Color is used")
    else:
        print("Nothing was converted")
    return img


def _check_if_mixable(color_1, color_2):
    if all(abs(1-x) < 1e-14 for x in color_1):
        return 2
    if all(abs(1-x) < 1e-14 for x in color_2):
        return 3
    return 1


def combine_two_images(blending_operator, image, image2, color_space=None, verbose=False, *args, **kwargs):
    """
    Combines two images with shape [x,y,3/4]. If the 3 dimension is in shape 4 it is expected to be in rgab and will be
    transformed into srgb with shape 3.

    :param blending_operator: operator which is used to mix the two images point by point
    :param image: image with shape [x,y,3/4]
    :param image2: image with shape [x,y,3/4]
    :param color_space: colorspace to use atm lab and hsv are supported
    :param verbose: show more debugging informations
    :param args: extra arguments for the blending operator
    :param kwargs: extra arguments for the blending operator
    :return: img whit shape [x,y,3] as srgb
    """
    img = np.asarray(image)
    img2 = np.asarray(image2)
    if img.ndim != 3 or img2.ndim != 3:
        raise Exception("Images need a dimension of 3")
    img = _convert_rgb_image(img, color_space)
    img2 = _convert_rgb_image(img2, color_space)
    if verbose:
        print(img)
        print(image)
    reduce = np.zeros([len(img), len(img[0]), len(img[0][0])])
    for i in range(len(img)):
        for j in range(len(img[0])):
            if verbose:
                print("{},{} = {}".format(i, j, img[i][j]))
            switch = {
                1: color_operations.blend_color(blending_operator, img[i][j], img2[i][j],
                                                *args, **kwargs),
                2: img2[i][j],
                3: img[i][j]
            }
            reduce[i][j] = switch.get(_check_if_mixable(image[i][j], image2[i][j]))
            if verbose:
                print("{},{}: {} + {} = {}".format(i, j, img[i][j], img2[i][j], reduce[i][j]))
    reduce = _convert_color_space_to_rgb(reduce, color_space)
    return reduce


def combine_two_images_hierarchic(blending_operator, image, z_1, image2, z_2, color_space=None, verbose=False, *args,
                                  **kwargs):
    """
    Combines two images with shape [x,y,3/4]. If the 3 dimension is in shape 4 it is expected to be in rgab and will be
    transformed into srgb with shape 3.

    :param verbose:
    :param color_space:
    :param blending_operator: operator which is used to mix the two images point by point
    :param image: image with shape [x,y,3/4]
    :param z_1: weights for the first image
    :param image2: image with shape [x,y,3/4]
    :param z_2: weights for the second image
    :return: img whit shape [x,y,3] as srgb
    """
    img = np.asarray(image)
    img2 = np.asarray(image2)
    if img.ndim != 3 or img2.ndim != 3:
        raise Exception("Images need a dimension of 3")
    img = _convert_rgb_image(img, color_space)
    img2 = _convert_rgb_image(img2, color_space)
    if verbose:
        print(image)
        print(img)
    z_new: ndarray = np.zeros([len(img), len(img[0]), 1])
    reduce = np.zeros([len(img), len(img[0]), 3])
    for i in range(len(img)):
        for j in range(len(img[0])):
            if verbose:
                print("{},{} = {}".format(i, j, img[i][j]))
            switch = {
                1: blending_operator(img[i][j], z_1[i][j], img2[i][j], z_2[i][j], *args, **kwargs),
                2: (img2[i][j], z_2[i][j]),
                3: (img[i][j], z_1[i][j]),
            }
            reduce[i][j], z_new[i][j] = switch.get(_check_if_mixable(image[i][j], image2[i][j]))
            if verbose:
                print("{},{}: {} + {} = {} \n  max({}|{}) = {}".format(i, j, img[i][j], img2[i][j], reduce[i][j], z_1[i][j], z_2[i][j], z_new[i][j]))
    reduce = _convert_color_space_to_rgb(reduce, color_space)
    return reduce, z_new
