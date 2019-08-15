from src import color_schemes, picture_plot


def plot_images(gaussians, contour_line_colorscheme=color_schemes.get_colorbrewer_schemes(),
                contour_colorscheme=color_schemes.get_background_colorbrewer_scheme(), *args, **kwargs):
    for vis in picture_plot.visualizations:
        kwargs.pop(vis, None)
    if not "contour_line_borders" in kwargs.keys():
        kwargs["contour_line_borders"] = [0.5, 0.9]
    if not "contour_line_level" in kwargs.keys():
        kwargs["contour_line_level"] = 5
    picture_plot.plot_images(gaussians, contour_lines=True, contour_line_colorscheme=contour_line_colorscheme,
                             contours=True, contour_colorscheme=contour_colorscheme, *args, **kwargs)
