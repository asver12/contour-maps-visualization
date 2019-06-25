import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from src import picture_worker, helper, color_schemes
from src.picture_worker import generate_contour_lines


def plot_images(images, gaussians, z_lists, colors=None, contour_lines_method="equal_density", contour_lines=True,
                contour_lines_weighted=True, num_of_levels=6,
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
    num_of_levels = num_of_levels - 1
    print("{}".format(["mu_x", "variance_x", "mu_y", "variance_y"]))
    if len(images) == 1:
        title_j = ""
        if title == "" and gaussians:
            title_j = '\n'.join("{}".format(gau[4:-1]) for gau in gaussians[0])
        elif len(title) > columns:
            title_j = title[columns]
        color_legend = colors if colors else []
        plot_image(plt, images[0], gaussians[0], z_lists[0], color_legend, contour_lines_method, contour_lines,
                   contour_lines_weighted,
                   title_j, with_axis, num_of_levels, borders, linewidth)
        plt.subplots_adjust(bottom=bottom, left=left, right=right, top=top)
    else:
        for i in range(math.ceil(len(images) / columns)):
            subplot = images[i * columns:(i + 1) * columns]
            sub_sums = z_lists[i * columns:(i + 1) * columns]
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


def plot_image(axis, image, gaussians, z_lists, colors=None, contour_lines_method="equal_density", contour_lines=True,
               contour_lines_weighted=True, title="", with_axis=True,
               num_of_levels=6, borders=None, linewidth=2):
    if colors is None:
        colors = color_schemes.get_colorbrewer_schemes()
    if borders is None:
        borders = [0.5, 1]
    extent = gaussians[0][:4]
    axis.imshow(image, extent=extent, origin='lower')
    for z_sum, contour_lines_colorscheme in zip(z_lists, colors):
        generate_contour_lines(axis, z_sum, gaussians[0], contour_lines_colorscheme,
                               contour_lines_method,
                               contour_lines_weighted, num_of_levels, borders, linewidth)
    if isinstance(axis, type(plt)):
        axis.title(title)
    else:
        axis.set_title(title)
    if colors:
        custom_lines = [Line2D([0], [0], color=color_schemes.get_main_color(colors[i])[-3], lw=4) for i in
                        range(len(gaussians))]
        axis.legend(custom_lines, [i for i in range(len(gaussians))],
                    loc='upper left', frameon=False)

    if not with_axis:
        axis.axis("off")
