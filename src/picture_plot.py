import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src import pie_chart_vis, helper, picture_worker, color_schemes, hierarchic_blending_operator, picture_cross

import logging

logger = logging.getLogger(__name__)

visualizations = ["contour_lines", "contours", "pie_charts", "crosses"]


def plot_images(gaussians, titles="", colors="", columns=5,
                bottom=0.0,
                left=0., right=2.,
                top=2.,
                *args, **kwargs
                ):
    """
    plots images for given gaussians.
    Can plot contours, contour-lines, crosses, pie-charts

    :param colors: color of each gaussian. Gonna be plotted as legend if non given, the colors from contour are chosen
    :param gaussians: [[gaussian_1_1, ... gaussian_1_m], ... , [gaussian_n_1, ... gaussian_n_m]] gaussians from which the image is calculated
    :param columns: number of pictures next to each other
    :return:
    """
    logger.debug("{}".format(["mu_x", "variance_x", "mu_y", "variance_y"]))
    color_legend = colors if colors else []
    if len(gaussians) == 1:
        title = generate_title(titles, gaussians[0], 0)
        fig, ax = plt.subplots(1, 1)
        plot_image(ax, gaussians[0], title=title, legend_colors=color_legend, *args, **kwargs)
        fig.subplots_adjust(bottom=bottom, left=left, right=right, top=top)
    else:
        for i in range(math.ceil(len(gaussians) / columns)):
            sub_gaussians = gaussians[i * columns:(i + 1) * columns]
            fig, ax = plt.subplots(1, len(sub_gaussians), sharex='col', sharey='row')
            if len(sub_gaussians) == 1:
                title = generate_title(titles, gaussians[i * columns], i * columns)
                plot_image(ax[i * columns], gaussians[i * columns], title=title, legend_colors=color_legend, *args,
                           **kwargs)
            else:
                for j in range(len(sub_gaussians)):
                    title = generate_title(titles, gaussians[j + i * columns], j + i * columns)
                    plot_image(ax[j], sub_gaussians[j], title=title,
                               legend_colors=color_legend, *args, **kwargs)
            fig.subplots_adjust(bottom=bottom, left=left, right=right, top=top)


def generate_title(titles, gaussian, index):
    if (len(titles) <= index or titles == "") and gaussian:
        return '\n'.join("{}".format(gau[4:-1]) for gau in gaussian)
    return titles[index]


def plot_image(ax, gaussians,
               contour_lines=False, contour_line_colorscheme=color_schemes.get_background_colorbrewer_scheme(),
               contour_lines_method="equal_density", contour_lines_weighted=True, contour_line_level=8,
               contour_line_borders=None,
               linewidth=2,
               contours=False, contour_colorscheme=color_schemes.get_colorbrewer_schemes(),
               contour_method="equal_density", contour_lvl=8, color_space="lab", use_c_implementation=True,
               use_alpha_sum=False, blending_operator=hierarchic_blending_operator.porter_duff_source_over,
               contour_borders=None,
               crosses=False, cross_colorscheme=color_schemes.get_colorbrewer_schemes(), cross_width=3,
               cross_borders=None,
               pie_charts=False, num_of_pies=10, angle=90, pie_chart_colors=None,
               legend_lw=2,
               legend_colors=None,
               title=""
               ):
    """
    Plots an image at a given axe, can plot contour, contour-lines, crosses and pie-charts

    :param ax: axis to plot on
    :param gaussians: list of gaussians to plot [gaussian_1, ... gaussian_n]
    :param contour_lines:
    :param contour_line_colorscheme:
    :param contour_lines_method:
    :param contour_lines_weighted:
    :param contour_line_level:
    :param contour_line_borders:
    :param linewidth:
    :param contours:
    :param contour_colorscheme:
    :param contour_method:
    :param contour_lvl:
    :param color_space:
    :param use_c_implementation:
    :param use_alpha_sum:
    :param blending_operator:
    :param contour_borders:
    :param crosses:
    :param cross_colorscheme:
    :param cross_width:
    :param cross_borders:
    :param pie_charts:
    :param num_of_pies:
    :param angle:
    :param pie_chart_colors:
    :param legend_lw:
    :param legend_colors: plots colors as lines to legend if not chosen defaults to contour-colors
    :param title: title specified if not given non is plotted
    :return:
    """

    z_list = helper.generate_gaussians(gaussians)
    z_min, z_max, z_sum = helper.generate_weights(z_list)

    # # to avoid a stretched y-axis
    ax.set_aspect('equal', adjustable='box')
    #
    if not contours:
        if isinstance(ax, type(plt)):
            ax.xlim(gaussians[0][0], gaussians[0][1])
            ax.ylim(gaussians[0][2], gaussians[0][3])
        else:
            ax.set_xlim(gaussians[0][0], gaussians[0][1])
            ax.set_ylim(gaussians[0][2], gaussians[0][3])
    if title:
        if isinstance(ax, type(plt)):
            ax.title(title)
        else:
            ax.set_title(title)
    if not legend_colors:
        legend_colors = color_schemes.get_representiv_colors(
            evaluate_colors([contour_line_colorscheme, contour_colorscheme, pie_chart_colors], len(gaussians)))
    custom_lines = [Line2D([0], [0], color=legend_colors[i], lw=legend_lw) for i in
                    range(len(gaussians))]
    ax.legend(custom_lines, [i for i in range(len(gaussians))],
              loc='upper left', frameon=False)
    if contours:
        if isinstance(contour_colorscheme, dict):
            picture_worker.input_image(ax, [gaussians[0]], [z_sum], np.min(z_sum), np.max(z_sum), [contour_colorscheme],
                                       contour_method,
                                       contour_lvl, color_space, use_c_implementation, use_alpha_sum,
                                       blending_operator=blending_operator, borders=contour_borders)
        else:
            picture_worker.input_image(ax, gaussians, z_list, z_min, z_max, contour_colorscheme, contour_method,
                                       contour_lvl, color_space, use_c_implementation, use_alpha_sum,
                                       blending_operator=blending_operator, borders=contour_borders)
    if crosses:
        picture_cross.input_crosses(ax, gaussians, z_list, z_min, z_max, cross_colorscheme, cross_width, cross_borders)
    if contour_lines:
        if isinstance(contour_line_colorscheme, dict):
            picture_worker.generate_contour_lines(ax, z_sum, gaussians[0], contour_line_colorscheme,
                                                  contour_lines_method,
                                                  contour_lines_weighted, contour_line_level, contour_line_borders,
                                                  linewidth)
        else:
            for z_values, scheme in zip(z_list, contour_line_colorscheme):
                picture_worker.generate_contour_lines(ax, z_values, gaussians[0], scheme, contour_lines_method,
                                                      contour_lines_weighted, contour_line_level, contour_line_borders,
                                                      linewidth)
    if pie_charts:
        pie_chart_vis.input_image(ax, gaussians, np.min(z_sum), np.max(z_sum), num_of_pies, angle=angle,
                                  colors=pie_chart_colors)


def evaluate_colors(colorschemes, number_of_schemes):
    for i in colorschemes:
        if len(i) >= number_of_schemes:
            return i
