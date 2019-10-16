import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from contour_visualization import pie_chart_vis, helper, picture_contours, color_schemes, hierarchic_blending_operator, \
    picture_cross, \
    picture_contour_lines

import logging

logger = logging.getLogger(__name__)

visualizations = ["contour_lines", "contours", "pie_charts", "crosses"]


def plot_all_methods(distributions, *args, **kwargs):
    plot_images(distributions, contours=True, contour_lines=True, *args, **kwargs)
    plot_images(distributions, pie_charts=True, *args, **kwargs)
    plot_images(distributions, crosses=True, contour_lines=True, *args, **kwargs)


def plot_image_variations(distribution, plot_titles=False, titles="", colors="", xlabels="", ylabels="",
                          *args, **kwargs
                          ):
    """
    plots all three types of plots for one distribution

    :param distribution:
    :param args:
    :param kwargs:
    :return:
    """
    logger.debug("{}".format(["mu_x", "variance_x", "mu_y", "variance_y"]))
    color_legend = colors if colors else []
    title = ""
    xlabel = ""
    ylabel = ""
    if plot_titles:
        title = _generate_title(titles, distribution, 0)
    if xlabels:
        xlabel = xlabels
    if ylabels:
        ylabel = ylabels
    fig, ax = plt.subplots(1, 3, sharex='col', sharey='row')
    plot_image(ax[0], distribution, title=title, legend=True, legend_colors=color_legend, xlabel=xlabel,
               ylabel=ylabel, contours=True, contour_lines=True, *args, **kwargs)
    plot_image(ax[1], distribution, title=title, legend=True, legend_colors=color_legend, xlabel=xlabel,
               ylabel=ylabel, crosses=True, contour_lines=True, *args, **kwargs)
    plot_image(ax[2], distribution, title=title, legend=True, legend_colors=color_legend, xlabel=xlabel,
               ylabel=ylabel, pie_charts=True, *args, **kwargs)


def plot_images(distributions, plot_titles=False, titles="", colors="", columns=5, xlabels="", ylabels="", legend=True,
                bottom=0.0,
                left=0., right=2.,
                top=2.,
                *args, **kwargs
                ):
    """
    plots images for given gaussians.
    Can plot contours, contour-lines, crosses, pie-charts

    :param distributions: [[distribution_1_1, ... distribution_1_m], ... , [distribution_n_1, ... distribution_n_m]]
    distributions from which the image is calculated
    :param plot_titles: if a title is plotted or not
    :param titles: title for each image to plot [title_1, ... title_m]
    :param colors: color of each gaussian. Gonna be plotted as legend if non given, the colors from contour are chosen
    :param columns: number of pictures next to each other
    :return:
    """
    logger.debug("{}".format(["mu_x", "variance_x", "mu_y", "variance_y"]))
    color_legend = colors if colors else []
    title = ""
    xlabel = ""
    ylabel = ""
    if len(distributions) == 1:
        if plot_titles:
            title = _generate_title(titles, distributions[0], 0)
        if xlabels:
            xlabel = xlabels
        if ylabels:
            ylabel = ylabels
        fig, ax = plt.subplots(1, 1)
        plot_image(ax, distributions[0], title=title, legend=legend, legend_colors=color_legend, xlabel=xlabel,
                   ylabel=ylabel, *args, **kwargs)
        fig.subplots_adjust(bottom=bottom, left=left, right=right, top=top)
    else:
        for i in range(math.ceil(len(distributions) / columns)):
            sub_gaussians = distributions[i * columns:(i + 1) * columns]
            fig, ax = plt.subplots(1, len(sub_gaussians), sharex='col', sharey='row')
            if len(sub_gaussians) == 1:
                if plot_titles:
                    title = _generate_title(titles, distributions[i * columns], i * columns)
                if xlabels:
                    xlabel = xlabels[i * columns]
                if ylabels:
                    ylabel = ylabels[i * columns]
                try:
                    plot_image(ax, distributions[i * columns], title=title, legend=legend,
                               legend_colors=color_legend, xlabel=xlabel, ylabel=ylabel, *args,
                               **kwargs)
                except Exception as e:
                    print(e)
            else:
                for j in range(len(sub_gaussians)):
                    if plot_titles:
                        title = _generate_title(titles, distributions[j + i * columns], j + i * columns)
                    if xlabels:
                        xlabel = xlabels[j + i * columns]
                    if ylabels:
                        ylabel = ylabels[j + i * columns]
                    try:
                        plot_image(ax[j], sub_gaussians[j], *args, title=title, legend=legend,
                                   legend_colors=color_legend, xlabel=xlabel, ylabel=ylabel, **kwargs)
                    except Exception as e:
                        print(e)
            fig.subplots_adjust(bottom=bottom, left=left, right=right, top=top)


def _generate_title(titles, gaussian, index):
    if (len(titles) < index or titles == "") and gaussian:
        return '\n'.join("{}".format(gau.get_attributes()[4:-1]) for gau in gaussian)
    return titles[index]


def plot_image(ax, distributions,
               contour_lines=False, contour_line_colorscheme=color_schemes.get_background_colorbrewer_scheme(),
               contour_lines_method="equal_density", contour_lines_weighted=True, contour_line_level=5,
               contour_line_borders=None,
               linewidth=2,
               contours=False, contour_colorscheme=color_schemes.get_colorbrewer_schemes(),
               contour_method="equal_density", contour_lvl=8, color_space="lab", use_c_implementation=True,
               contour_mode="hierarchic", blending_operator=hierarchic_blending_operator.porter_duff_source_over,
               contour_min_gauss=False,
               contour_lower_border_lvl=None,
               contour_lower_border_to_cut=0,
               contour_borders=None,
               crosses=False, cross_colorscheme=color_schemes.get_colorbrewer_schemes(), cross_width="5%",
               cross_same_broad=True,
               cross_length_multiplier=2. * np.sqrt(2.),
               cross_borders=None,
               pie_charts=False, pie_num=25, pie_angle=90, pie_chart_colors=None, pie_chart_modus="light",
               pie_chart_scale=1.,
               pie_chart_borders=None,
               pie_chart_iso_level=40,
               pie_chart_level_to_cut=0,
               pie_chart_contour_method="equal_density",
               legend=False,
               legend_lw=2,
               legend_colors=None,
               legend_names=None,
               title="",
               xlabel="",
               ylabel="",
               *args, **kwargs
               ):
    """
    Plots an image at a given axe, can plot contour, contour-lines, crosses and pie-charts

    :param ax: axis to plot on
    :param distributions: list of distributions to plot [distribution_1, ... distribution_n]
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
    :param contour_mode: sets the mode to use. Defaults is hierarchic and defaults to hierarchic
    :param blending_operator:
    :param contour_min_gauss: if min of min gauss is used when True else from z_sum
    :param contour_lower_border_to_cut: defines the global lower border at which to cut the particular each image
    :param contour_lower_border_lvl: def at which level the iso-border gets cut
    :param contour_borders:
    :param crosses:
    :param cross_colorscheme:
    :param cross_width:
    :param cross_same_broad: if True calculates the broad of the crosses depending by the smaller cross
    :param cross_length_multiplier: is multiplied with the lenght to create bigger or smaller crosses
    :param cross_borders:
    :param cross_fill: if cross is filled with color or not
    :param pie_charts:
    :param pie_num:
    :param pie_angle: where the pie-chart begins 0 is horizontal beginning on the right 90 beginns at the top
    :param pie_chart_colors: Colorscheme to use. Defaults is colorbrewer
    :param pie_chart_modus: "light" or "size" if "size" global density is coded with size elif "light" through the colorscheme
    :param pie_chart_scale: when light selected sets the size of the pies
    :param pie_chart_borders: [0.,1.] range of ether size or color lightness of the pies
    :param pie_chart_iso_level:
    :param pie_chart_level_to_cut:
    :param pie_chart_contour_method:
    :param legend: if a legend should be plotted or not
    :param legend_lw:
    :param legend_colors: plots colors as lines to legend if not chosen defaults to contour-colors
    :param legend_names: if set uses names instead of numbers
    :param title: title specified if not given non is plotted
    :param xlabel:
    :param ylabel:
    :return:
    """

    if contours or contour_lines or pie_charts or crosses:
        limits = helper.get_limits(distributions)
        z_list = helper.generate_distribution_grids(distributions, x_min=limits.x_min, x_max=limits.x_max,
                                                    y_min=limits.y_min, y_max=limits.y_max)
        z_min, z_max, z_sum = helper.generate_weights(z_list)

        # # to avoid a stretched y-axis
        if isinstance(ax, type(plt)):
            pass  # ax.aspect('equal', adjustable='box')
        else:
            ax.set_aspect('equal', adjustable='box')
        logger.debug("Axis-limits: {}".format(limits))
        if xlabel:
            if isinstance(ax, type(plt)):
                ax.xlabel(xlabel)
            else:
                ax.set_xlabel(xlabel)
        if ylabel:
            if isinstance(ax, type(plt)):
                ax.ylabel(ylabel)
            else:
                ax.set_ylabel(ylabel)

        if not contours:
            if isinstance(ax, type(plt)):
                ax.xlim((limits.x_min, limits.x_max))
                ax.ylim((limits.y_min, limits.y_max))
            else:
                ax.set_xlim((limits.x_min, limits.x_max), emit=False)
                ax.set_ylim((limits.y_min, limits.y_max), emit=False)
        if title:
            if isinstance(ax, type(plt)):
                ax.title(title)
            else:
                ax.set_title(title)
        if legend:
            if not legend_colors:
                legend_colors = color_schemes.get_representiv_colors(
                    _evaluate_colors([_evaluate_colors(
                        [[contour_line_colorscheme, ] if isinstance(contour_line_colorscheme,
                                                                    dict) else contour_line_colorscheme,
                         [contour_colorscheme, ] if isinstance(contour_colorscheme, dict) else contour_colorscheme,
                         pie_chart_colors], len(distributions)), pie_chart_colors], len(distributions)))[
                                :len(distributions)]
            _generate_legend(ax, legend_colors, legend_names, legend_lw=legend_lw)
        if contours:
            if isinstance(contour_colorscheme, dict):
                picture_contours.input_image(ax, [distributions[0]], [z_sum], np.min(z_sum), np.max(z_sum), z_sum,
                                             [contour_colorscheme],
                                             contour_method,
                                             contour_lvl, color_space, use_c_implementation, contour_mode,
                                             blending_operator=blending_operator, borders=contour_borders,
                                             min_gauss=contour_min_gauss,
                                             lower_border=contour_lower_border_lvl,
                                             lower_border_to_cut=contour_lower_border_to_cut)
            else:
                picture_contours.input_image(ax, distributions, z_list, z_min, z_max, z_sum, contour_colorscheme,
                                             contour_method,
                                             contour_lvl, color_space, use_c_implementation, contour_mode,
                                             blending_operator=blending_operator, borders=contour_borders,
                                             min_gauss=contour_min_gauss,
                                             lower_border=contour_lower_border_lvl,
                                             lower_border_to_cut=contour_lower_border_to_cut)
        if crosses:
            picture_cross.input_crosses(ax, distributions, z_list, z_min, z_max, cross_colorscheme, cross_width,
                                        cross_same_broad,
                                        cross_length_multiplier,
                                        cross_borders, linewidth=linewidth, *args, **kwargs)
        if contour_lines:
            if isinstance(contour_line_colorscheme, dict):
                picture_contour_lines.generate_contour_lines(ax, z_sum, limits,
                                                             contour_line_colorscheme,
                                                             contour_lines_method,
                                                             contour_lines_weighted, contour_line_level,
                                                             contour_line_borders,
                                                             linewidth)
            else:
                for z_values, scheme in zip(z_list, contour_line_colorscheme):
                    picture_contour_lines.generate_contour_lines(ax, z_values, limits,
                                                                 scheme,
                                                                 contour_lines_method,
                                                                 contour_lines_weighted, contour_line_level,
                                                                 contour_line_borders,
                                                                 linewidth)
        if pie_charts:
            if pie_chart_colors is None:
                pie_chart_colors = contour_colorscheme
            pie_chart_vis.input_image(ax, distributions, z_sum, pie_num, angle=pie_angle,
                                      colorschemes=pie_chart_colors, modus=pie_chart_modus, borders=pie_chart_borders,
                                      iso_level=pie_chart_iso_level, level_to_cut=pie_chart_level_to_cut,
                                      contour_method=pie_chart_contour_method, scale=pie_chart_scale, set_limit=False)


def _generate_legend(axis, colors, names=None, legend_lw=2):
    if names is None:
        names = [chr(i + 65) for i in range(len(colors))]
    custom_lines = [Line2D([0], [0], color=colors[i], lw=legend_lw) for i in
                    range(len(colors))]
    axis.legend(custom_lines, names, frameon=False)


def _evaluate_colors(colorschemes, number_of_schemes):
    for i in colorschemes:
        if len(i) >= number_of_schemes:
            return i
