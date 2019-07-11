import copy

import numpy as np
import scipy.ndimage

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from src import helper, picture_worker, hierarchic_blending_operator


def get_image_list(gaussians, colorschemes, borders=None, verbose=False):
    """

    :param gaussians: [gaussian_1, ... , gaussian_n]
    :param colorschemes: [{colorscheme: color_scheme_function_1, colorscheme_name: colorscheme_name_1},
                            ... ]{colorscheme: color_scheme_function_n, colorscheme_name: colorscheme_name_n}]
    :param borders: range in which the pixel of the pictures are normalizes
    :param verbose: outputs additional information
    :return: Imagelist [2D-image_1, ... ,2D-image_n], Weights [2D-weight_1, ... ,2D-weight_n], Sum of all weights 2D-weight
    """
    if borders is None:
        borders = [0, 1]
    z_list = helper.generate_gaussians(gaussians)
    z_min, z_max, z_sum = helper.generate_weights(z_list)
    img_list = []
    lower_border = borders[0]
    upper_border = borders[1]
    for z, colorscheme in zip(z_list, colorschemes):
        z_min_weight = (upper_border - lower_border) * (np.min(z) - z_min) / (z_max - z_min) + lower_border
        z_max_weight = (upper_border - lower_border) * (np.max(z) - z_min) / (z_max - z_min) + lower_border
        img, _ = picture_worker.get_colorgrid(z, **colorscheme, min_value=z_min_weight, max_value=z_max_weight,
                                              split=True,
                                              verbose=verbose)
        img_list.append(img)
    return img_list, z_list, z_sum


def gen_line(eigenvector, eigenvalue, middlepoint, x_list, y_list, z_list, verbose=False):
    length = eigenvalue  # 1/(eigenvalue**(1/2))
    if x_list[0][0] <= middlepoint[1] + eigenvector[0] * length <= x_list[0][-1]:
        index_x = helper.find_index(middlepoint[1] + eigenvector[0] * length, x_list[0])
    else:
        index_x = 100
    if y_list[:, 0][0] <= middlepoint[0] + eigenvector[1] * length <= y_list[:, 0][-1]:
        index_y = helper.find_index(middlepoint[0] + eigenvector[1] * length, y_list[:, 0])
    else:
        index_y = 100
    if verbose:
        print("Middlepoint: {}, Direction: {}, Distance: {}".format(middlepoint, eigenvector, eigenvalue))
        print("x-From : {} to {}".format(middlepoint[1] - eigenvector[0] * length,
                                         middlepoint[1] + eigenvector[0] * length))
        print("y-From : {} to {}".format(middlepoint[0] - eigenvector[1] * length,
                                         middlepoint[0] + eigenvector[1] * length))

    if x_list[0][0] <= middlepoint[1] - eigenvector[0] * length <= x_list[0][-1]:
        index_x_begin = helper.find_index(middlepoint[1] - eigenvector[0] * length, x_list[0])
    else:
        index_x_begin = 100
    if y_list[:, 0][0] <= middlepoint[0] - eigenvector[1] * length <= y_list[:, 0][-1]:
        index_y_begin = helper.find_index(middlepoint[0] - eigenvector[1] * length, y_list[:, 0])
    else:
        index_y_begin = 100
    if verbose:
        print("x_begin: {}, x_end: {}".format(index_x_begin, index_x))
        print("y_begin: {}, y_end: {}".format(index_y_begin, index_y))
    return np.linspace(index_x_begin, index_x, z_list.shape[0] * 10, dtype=int), np.linspace(index_y_begin, index_y,
                                                                                             z_list.shape[1] * 10,
                                                                                             dtype=int)


def get_line(eigenvalue, eigenvector, middlepoint):
    length = eigenvalue
    startpoint = middlepoint[1] - eigenvector[0] * length, middlepoint[1] + eigenvector[0] * length
    endpoint = middlepoint[0] - eigenvector[1] * length, middlepoint[0] + eigenvector[1] * length
    return startpoint, endpoint


def split_line(startpoint, endpoint, size=100, method="equal_density", num_of_levels=5):
    level_x = picture_worker.get_iso_levels(np.linspace(startpoint[0], endpoint[0], size), method, num_of_levels)
    level_y = picture_worker.get_iso_levels(np.linspace(startpoint[0], endpoint[0], size), method, num_of_levels)
    return [startpoint, *list(zip(level_x, level_y)), endpoint]


def filter_crosses(gaussian, size=3, method="equal_density", num_of_levels=5, verbose=False):
    eigenvalues, eigenvectors = np.linalg.eig(gaussian[5])
    startpoint_x, endpoint_x = get_line(eigenvalues[0], eigenvectors[1], gaussian)
    startpoint_y, endpoint_y = get_line(eigenvalues[1], eigenvectors[0], gaussian)
    x_line, x_color = split_line(startpoint_x, endpoint_x, size, method, num_of_levels)
    y_line, y_color = split_line(startpoint_y, endpoint_y, size, method, num_of_levels)
    return x_line, y_line


def generate_image_lines(gaussians, colorschemes, cross_size=3, verbose=False,
                   blending_operator=hierarchic_blending_operator.porter_duff_source_over,
                   *args, **kwargs):
    print(gaussians)
    return [
        filter_crosses(gaussians[i], size=cross_size, verbose=verbose) for i
        in
        gaussians]

def filter_cross(image, x_list, y_list, z_list, gaussian, size=3, verbose=False):
    """
    if mue_1 is smaller
    Eigenvector e_1 corrsponds to the major axis direction
    Egenvalue 1/(sqrt(mue_2) corresponds to the major axis lenght
    Eigenvector e_2 corresponds to the minor axis
    Egenvalue 1/(sqrt(mue_1) corresponds to the minor axis lenght

    :param image:
    :param x_list:
    :param y_list:
    :param z_list:
    :param gaussian:
    :param size:
    :return:
    """
    new_image = np.ones(image.shape)
    eigenvalues, eigenvectors = np.linalg.eig(gaussian[5])

    if verbose:
        print("-----------------------------------------------")
        print("Eigenvector: {} \n Eigenvalues: {}".format(eigenvectors, eigenvalues))

    x, y = gen_line(eigenvectors[0], eigenvalues[1], gaussian[4], x_list, y_list, z_list, verbose)
    for j in range(1, size):
        for i in zip(x, y):
            if i[0] < new_image.shape[0] and i[1] < new_image.shape[1]:
                new_image[i[0]][i[1] - j] = image[i[0]][i[1] - j]
                new_image[i[0] - j][i[1]] = image[i[0] - j][i[1]]
    for j in range(size):
        for i in zip(x, y):
            if i[0] < new_image.shape[0] - j and i[1] < new_image.shape[1] - j:
                new_image[i[0]][i[1] + j] = image[i[0]][i[1] + j]
                new_image[i[0] + j][i[1]] = image[i[0] + j][i[1]]
    if verbose:
        print("-----------------------------------------------")
    x, y = gen_line(eigenvectors[1], eigenvalues[0], gaussian[4], x_list, y_list, z_list, verbose)
    # new_image[x, y] = image[x, y]

    for j in range(1, size):
        for i in zip(x, y):
            if i[0] < new_image.shape[0] and i[1] < new_image.shape[1]:
                new_image[i[0]][i[1] - j] = image[i[0]][i[1] - j]
                new_image[i[0] - j][i[1]] = image[i[0] - j][i[1]]
    for j in range(size):
        for i in zip(x, y):
            if i[0] < new_image.shape[0] - j and i[1] < new_image.shape[1] - j:
                new_image[i[0]][i[1] + j] = image[i[0]][i[1] + j]
                new_image[i[0] + j][i[1]] = image[i[0] + j][i[1]]

    if verbose:
        print("-----------------------------------------------")
    return new_image


def generate_image(gaussians, colorschemes, cross_size=3, verbose=False,
                   blending_operator=hierarchic_blending_operator.porter_duff_source_over,
                   *args, **kwargs):
    print(gaussians)
    x_list, y_list, z_list = helper.generate_gaussians_xyz(gaussians)
    img_list, _, z_sum = get_image_list(gaussians, colorschemes)
    img_list = [
        filter_cross(img_list[i], x_list[i], y_list[i], z_list[i], gaussians[i], size=cross_size, verbose=verbose) for i
        in
        range(0, len(gaussians))]
    image, alpha = picture_worker.combine_multiple_images_hierarchic(images=img_list, z_values=z_list,
                                                                     blending_operator=blending_operator, *args,
                                                                     **kwargs)
    return z_list, image, z_sum
