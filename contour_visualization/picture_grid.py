from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from contour_visualization import picture_contours, helper, color_schemes
from contour_visualization.picture_plot import _evaluate_colors


def plot_images(images, gaussians, *args, **kwargs):
    for img, gaussian in zip(images, gaussians):
        plot_image(img, gaussian, *args, **kwargs)


def plot_image(gaussians, title="", with_axis=True, xlim=None, ylim=None,
               colorschemes=color_schemes.get_colorbrewer_schemes(),legend_lw=2, *args, **kwargs):
    """

    :param with_axis: if mathematical axis is shown or not
    :param images: List of List of 2D-images in rgb to plot
    :param gaussians: [gaussian_1, ... , gaussian_n] gaussians from which the image is calculated
    :param title: title of picture
    :return:
    """
    fig, axis = plt.subplots(1, len(gaussians), sharex='col', sharey='row')
    limits = helper.get_limits(gaussians, xlim, ylim)
    extent = [limits.x_min, limits.x_max, limits.y_min, limits.y_max]
    z_list = helper.generate_distribution_grids(gaussians, limits=limits)
    z_min, z_max, z_sum = helper.generate_weights(z_list)

    colors = color_schemes.get_representiv_colors(colorschemes)
    names = [chr(i + 65) for i in range(len(gaussians))]
    custom_lines = [Line2D([0], [0], color=colors[i], lw=legend_lw) for i in
                    range(len(colors))]
    for i, gaussian in enumerate(gaussians):
        img, _ = picture_contours.calculate_image([z_list[i]], z_min, z_max, z_sum, [colorschemes[i]], *args, **kwargs)
        axis[i].imshow(img, extent=extent, origin='lower')
        if title == "" and gaussians:
            title = str(gaussian.get_attributes()[4:-1])
        axis[i].set_title(title)
        axis[i].legend([custom_lines[i]], [names[i]], frameon=False)
    if not with_axis:
        axis.axis("off")
    fig.subplots_adjust(wspace=0.1)