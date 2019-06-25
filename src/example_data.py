import copy

from src import helper, color_schemes, picture_worker


def generate_gaussians(gaussians):
    return [helper.get_gaussian(*gaussian)[2] for gaussian in gaussians]


def generate_four_moving_gaussians(x_min=-10, x_max=10, y_min=-10, y_max=10, size=200):
    mu_x, mu_y = 0, 0
    variance_x, variance_y = 5, 5
    gaussian_static = [x_min, x_max, y_min, y_max, [mu_x, mu_y], [[variance_x, 0], [0, variance_y]], size]

    colorschemes = color_schemes.get_colorbrewer_schemes()

    color_codes = [color_schemes.get_main_color(i)[-1] for i in colorschemes]

    static_gaussian_rep = [copy.deepcopy(gaussian_static),
                           copy.deepcopy(gaussian_static),
                           copy.deepcopy(gaussian_static), copy.deepcopy(gaussian_static)]
    z_lists = []
    z_sums = []
    gaussians = []

    for i in range(1, 6):
        static_gaussian_rep[0][4][0] = +(i)
        static_gaussian_rep[0][4][1] = +(i)
        static_gaussian_rep[1][4][0] = -(i)
        static_gaussian_rep[1][4][1] = -(i)
        static_gaussian_rep[2][4][0] = +(i)
        static_gaussian_rep[2][4][1] = -(i)
        static_gaussian_rep[3][4][0] = -(i)
        static_gaussian_rep[3][4][1] = +(i)

        print(static_gaussian_rep)
        z_list = helper.generate_gaussians(static_gaussian_rep)
        z_min, z_max, z_sum = helper.generate_weights(z_list)
        z_lists.append(z_list)
        z_sums.append(z_sum)
        gaussians.append([copy.deepcopy(i) for i in static_gaussian_rep])
    return z_lists, z_sums, gaussians, color_codes


def default_gaussian_2d(x_min=-10, x_max=10, y_min=-10, y_max=10, size=200):
    mu_x, mu_y = 2, 2
    variance_x, variance_y = 5, 5
    gaussian_static = [x_min, x_max, y_min, y_max, [mu_x, mu_y], [[variance_x, 0], [0, variance_y]], size]

    var_x, var_y = [2, 5, 10, 15, 20], [15, 20]
    gaussians_2d = []
    for i, mu_y in enumerate([-5, -2, 0, 2, 5]):
        for j, mu_x in enumerate([-5, -2, 0, 2, 5]):
            if mu_y > 0:
                variance_y = var_y[i - 3]
                variance_x = var_x[j]
            else:
                variance_x = 5
            gaussians_2d.append([x_min, x_max, y_min, y_max, [mu_x, mu_y], [[variance_x, 0], [0, variance_y]], size])
    color_codes = [color_schemes.get_main_color(i)[-1] for i in color_schemes.get_colorbrewer_schemes()[:2]]
    gaussians = []
    z_lists = []
    z_sums = []
    for i in range(len(gaussians_2d)):
        gau = [gaussian_static, gaussians_2d[i]]
        z_list = helper.generate_gaussians(gau)
        z_min, z_max, z_sum = helper.generate_weights(z_list)
        z_lists.append(z_list)
        z_sums.append(z_sum)
        gaussians.append(copy.deepcopy(gau))
    return z_lists, z_sums, gaussians, color_codes