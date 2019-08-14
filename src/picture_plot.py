import numpy as np
from src import pie_chart_vis, helper, picture_worker, color_schemes, hierarchic_blending_operator, picture_cross


def plot_image(ax, gaussians,
               contour_lines=False, contour_line_colorscheme=color_schemes.get_background_colorbrewer_scheme(),
               contour_lines_method="equal_density", contour_lines_weighted=True, contour_line_level=8, borders=None,
               linewidth=2,
               contours=False, contour_colorscheme=color_schemes.get_colorbrewer_schemes(),
               contour_method="equal_density", contour_lvl=8, color_space="lab", use_c_implementation=True,
               use_alpha_sum=False, blending_operator=hierarchic_blending_operator.porter_duff_source_over,
               contour_borders=None,
               crosses=False, cross_colorscheme=color_schemes.get_colorbrewer_schemes(), cross_width=3,
               cross_borders=None,
               pie_charts=False, num_of_pies=10, angle=90, colors=None):
    z_list = helper.generate_gaussians(gaussians)
    z_min, z_max, z_sum = helper.generate_weights(z_list)
    if not contours:
        ax.set_xlim(gaussians[0][0], gaussians[0][1])
        ax.set_ylim(gaussians[0][2], gaussians[0][3])
    if contours:
        picture_worker.input_image(ax, gaussians, z_list, z_min, z_max, contour_colorscheme, contour_method,
                                   contour_lvl, color_space, use_c_implementation, use_alpha_sum,
                                   blending_operator=blending_operator, borders=contour_borders)
    if crosses:
        picture_cross.input_crosses(ax, gaussians, z_list, z_min, z_max, cross_colorscheme, cross_width, cross_borders)
    if contour_lines:
        picture_worker.generate_contour_lines(ax, z_sum, gaussians[0], contour_line_colorscheme, contour_lines_method,
                                              contour_lines_weighted, contour_line_level, borders, linewidth)
    if pie_charts:
        pie_chart_vis.input_image(ax, gaussians, np.min(z_sum), np.max(z_sum), num_of_pies, angle=angle, colors=colors)
