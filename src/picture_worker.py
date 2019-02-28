from matplotlib import pyplot as plt
import matplotlib
import numpy as np
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


def get_colorgrid(X, color_scheme, num_of_levels, *args, **kwargs):
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
    colormap_green = color_scheme(levels=levels, *args, **kwargs)
    return color_operations.map_colors(X, colormap_green, levels)

def combine_two_images(blending_operator, img, img2, color_space=None, verbose=False, *args, **kwargs):
    if color_space == "lab":
        img = color.rgb2lab(img)
        img2 = color.rgb2lab(img2)
        print("Lab-Color is used")
    if color_space == "hsv":
        img = color.rgb2hsv(img)
        img2 = color.rgb2hsv(img2)
        print("Hsv-Color is used")
    reduce = np.zeros([len(img), len(img[0]), 3])
    for i in range(len(img)):
        for j in range(len(img[0])):
            if verbose:
                print("{}:{} = {}".format(i, j, img[i][j]))
            reduce[i][j] = color_operations.blend_color(blending_operator, img[i][j], img2[i][j],
                                                        *args, **kwargs)
            if verbose:
                print("{}:{}: {} + {} = {}".format(i, j, img[i][j], img2[i][j], reduce[i][j][0]))
    if color_space == "lab":
        reduce = color.lab2rgb(reduce)
    if color_space == "hsv":
        reduce = color.hsv2rgb(reduce)
    return reduce