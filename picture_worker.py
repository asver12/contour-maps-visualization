from matplotlib import pyplot as plt
import matplotlib
import numpy as np

import color_operations
import color_converter


def get_picture(x_min, x_max, y_min, y_max, X, Y, Z, levels, **args):
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    fig, ax = plt.subplots()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.patch.set_facecolor("none")
    ax.patch.set_edgecolor("none")
    ax.axis('off')
    plt.contourf(X, Y, Z, levels, **args)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, -1).copy(), w, h
    plt.figure()
    return img


def get_colorgrid(X, color_scheme, color, num_of_levels, **args):
    z_min, z_max = np.min(X), np.max(X)
    levels = np.linspace(z_min, z_max, num_of_levels)  # [1:6] ?
    colormap_green = color_scheme(color, levels=levels, **args)
    return color_operations.map_colors(X, colormap_green, levels)


def combine_two_images(blending_operator, img, img2, alpha=0.5, **kwargs):
    reduce = np.zeros([len(img), len(img[0]), 4], dtype=np.uint8)
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j][3] == 0:
                reduce[i][j] = img2[i][j]
            elif img2[i][j][3] == 0:
                reduce[i][j] = img[i][j]
            else:
                reduce[i][j] = np.array(
                    color_operations.blend_rgb255(blending_operator, img[i][j], img2[i][j], alpha, **kwargs), np.uint8)
    return reduce
