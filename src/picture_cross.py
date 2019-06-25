import numpy as np

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


def filter_cross(image, z_list, size=3):
    middlepoint = np.unravel_index(z_list.argmax(), z_list.shape)
    new_image = np.ones(image.shape, dtype=float)
    # print(image.shape)
    new_image[:, middlepoint[1] - size:middlepoint[1] + size] = image[:, middlepoint[1] - size:middlepoint[1] + size]
    new_image[middlepoint[0] - size:middlepoint[0] + size, :] = image[middlepoint[0] - size:middlepoint[0] + size, :]
    # print(new_image[middlepoint[0], :])
    return new_image


def generate_image(gaussians, colorschemes, cross_size=3,
                   blending_operator=hierarchic_blending_operator.porter_duff_source_over,
                   *args, **kwargs):
    img_list, z_list, z_sum = get_image_list(gaussians, colorschemes)
    img_list = [filter_cross(i, j, cross_size) for i, j in zip(img_list, z_list)]
    image, alpha = picture_worker.combine_multiple_images_hierarchic(images=img_list, z_values=z_list,
                                                                     blending_operator=blending_operator, *args,
                                                                     **kwargs)
    return z_list, image, z_sum
