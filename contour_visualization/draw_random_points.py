import numpy as np
import itertools
import random
from contour_visualization import picture_contours, iso_lines, helper, color_schemes, color_operations
from contour_visualization.color_schemes import get_main_color


def get_points(gaussian, z_list, colorscheme, num=100, method="equal_density", num_of_levels=8, min_value=0.,
               max_value=1., split=True,
               min_border=None, *args, **kwargs):
    # generate colors to chose from
    z_list = np.asarray(z_list)
    norm = iso_lines.get_iso_levels(z_list, method=method, num_of_levels=num_of_levels + 2)
    norm = picture_contours.get_color_middlepoint(norm, min_value, max_value)
    colormap = colorscheme["colorscheme"](levels=norm, colorscheme_name=colorscheme["colorscheme_name"], *args,
                                          **kwargs)

    # replace points in image with matching colors
    levels = iso_lines.get_iso_levels(z_list, method=method, num_of_levels=num_of_levels)
    rand_coordinates = gaussian.gau.rvs(num * num, random_state=12345)
    rand_z_values = gaussian.gau.pdf(rand_coordinates).reshape((num, num))
    # z_value, alpha_value = color_operations.map_colors(rand_z_values, colormap, levels, split)
    z_value = get_main_color(colorscheme)[-4]

    return rand_coordinates, np.tile(z_value, (num * num, 1))


def generate_random_points(distributions, colorschemes=None, z_list=None, z_min=None, z_max=None, xlim=None, ylim=None,
                           *args, **kwargs):
    limits = helper.get_limits(distributions, xlim, ylim)
    if z_list is None:
        z_list = helper.generate_distribution_grids(distributions, limits=limits)
    if z_min is None:
        z_min, z_max, z_sum = helper.generate_weights(z_list)
    if colorschemes is None:
        colorschemes = color_schemes.get_colorbrewer_schemes()
    return __generate_random_points(distributions, z_list, z_min, z_max, colorschemes[:len(distributions)], *args,
                                    **kwargs)


def __generate_random_points(gaussians, z_list, z_min, z_max, colorschemes, borders=None,
                             *args, **kwargs):
    if borders is None:
        borders = [0, 1]
    lower_border = borders[0]
    upper_border = borders[1]
    z_weights = []
    for z in z_list:
        z_min_weight = (upper_border - lower_border) * (np.min(z) - z_min) / (z_max - z_min) + lower_border
        z_max_weight = (upper_border - lower_border) * (np.max(z) - z_min) / (z_max - z_min) + lower_border
        z_weights.append([z_min_weight, z_max_weight, z])
    return [get_points(gaussian, z_list=z_list, colorscheme=colorscheme, min_value=z_min_weight, max_value=z_max_weight,
                       min_border=lower_border,
                       *args, **helper.filter_kwargs(get_points, **kwargs)) for
            gaussian, colorscheme, [z_min_weight, z_max_weight, z_list] in zip(gaussians, colorschemes, z_weights)]


def input_points(ax, distributions, *args, **kwargs):
    point_lists = generate_random_points(distributions, *args, **kwargs)
    points = itertools.chain(*[x for x, y in point_lists])
    colors = itertools.chain(*[y for x, y in point_lists])
    point_lists = list(zip(list(points), list(colors)))
    random.shuffle(point_lists)
    for points, colors in point_lists:
        ax.scatter(*points, c=[colors, ], **helper.filter_kwargs(ax.scatter, **kwargs))

        # for i in range(len(points)):
        #     print(colors[i])
        #    ax.scatter([a[i],], [b[i],], c=[colors[i],], **helper.filter_kwargs(ax.scatter, **kwargs))
