from ctypes import cdll, c_int, byref, c_double, POINTER, Structure, c_char
import os
import numpy as np

try:
    local_dir = os.path.dirname(__file__)
    lib = cdll.LoadLibrary(local_dir + "/../libs/libblendingOperators.so")
except OSError as e:
    print("File libsvmBlend.so could not be found under {}".format(
        os.path.dirname(__file__) + "/../libs/libblendingOperators.so"))

dtype = np.float


class returnStruct(Structure):
    _fields_ = [("returnList", POINTER(c_double)),
                ("returnWeight", POINTER(c_double))]


lib.mmMultSimple.argtypes = [c_int, c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
                             POINTER(c_double), POINTER(c_double)]

lib.mmMultSimple.restype = None


def create_cdll_type(array):
    return np.ctypeslib.as_ctypes(array.flatten())


def callSimpleMerge(matrizes, weight, verbose=False):
    """
    combines two matrizes by simple weight given

    :param matrizes: [image_1, image_2]
    :param weight: [weight_1, weight_2]
    :param verbose: outputs additional information
    :return: 2D-image
    """
    m, n = len(matrizes[0]), len(matrizes[0][0])
    matrizes = [np.asarray(i, dtype=dtype) for i in matrizes]
    weight = [np.asarray(i, dtype=dtype) for i in weight]
    if verbose:
        for i in range(len(matrizes[0])):
            print("{}: ".format(i), end="")
            for j in range(len(matrizes[0][0])):
                print("{}: ".format(j), end="")
                for k in range(len(matrizes[0][0][0])):
                    print("{} ".format(matrizes[0][i][j][k]), end="")
            print("")
        print("")

        for i in range(len(weight[0])):
            print("{}: ".format(i), end="")
            for j in range(len(weight[0][0])):
                print("{}: ".format(j), end="")
                print("{} ".format(weight[0][i][j]), end="")
            print("")
        print("")

    new_matrix = np.zeros(matrizes[0].shape, dtype=np.float)
    new_weight = np.zeros(weight[0].shape, dtype=np.float)
    c_new_matrix = create_cdll_type(new_matrix)
    c_new_weight = create_cdll_type(new_weight)
    lib.mmMultSimple(m, n, create_cdll_type(matrizes[0]),
                     create_cdll_type(weight[0]), create_cdll_type(matrizes[1]),
                     create_cdll_type(weight[1]), c_new_matrix, c_new_weight)
    return np.ctypeslib.as_array(c_new_matrix).reshape((m, n, 3)), np.ctypeslib.as_array(c_new_weight).reshape(m, n, 1)


doublepp = np.ctypeslib.ndpointer(dtype=np.uintp)

lib.mmMultHierarchic.argtypes = c_int, c_int, c_int, doublepp, doublepp, POINTER(c_double), POINTER(
    c_double), POINTER(c_char)
lib.mmMultHierarchic.restype = None


def call_hierarchic_merge(matrizes, weights, colorspace="lab"):
    """
    Combines multiple matrizes hierarchic and returns the combined image. Uses hierarchic porter-duff-source-over
    Only works with rgb-Images atm

    :param colorspace: rgb and lab supported so far
    :param matrizes: [image_1, ... , image_n] list of images to use
    :param weights: [weight_1, ... , weight_n] corresponding weights to images
    :return: hierarchically build new 2D-image
    """
    num_of_matrizes, m, n = len(matrizes), len(matrizes[0]), len(matrizes[0][0])

    new_matrix = np.zeros(matrizes[0].shape, dtype=np.float)
    new_weight = np.zeros(weights[0].shape, dtype=np.float)
    input_matrizes = np.array([np.ctypeslib.as_array(np.asarray(i, dtype=np.float).flatten()) for i in matrizes])
    input_weights = np.array([np.ctypeslib.as_array(np.asarray(i, dtype=np.float).flatten()) for i in weights])

    c_new_matrix_1 = create_cdll_type(new_matrix)
    c_new_weight_1 = create_cdll_type(new_weight)
    c_input_matrizes = (input_matrizes.__array_interface__['data'][0]
                        + np.arange(input_matrizes.shape[0]) * input_matrizes.strides[0]).astype(np.uintp)
    c_input_weights = (input_weights.__array_interface__['data'][0]
                       + np.arange(input_weights.shape[0]) * input_weights.strides[0]).astype(np.uintp)

    lib.mmMultHierarchic(m, n, num_of_matrizes, c_input_matrizes, c_input_weights, c_new_matrix_1, c_new_weight_1,
                         colorspace.encode('utf-8'))
    return np.ctypeslib.as_array(c_new_matrix_1).reshape(m, n, 3), np.ctypeslib.as_array(c_new_weight_1).reshape(m, n,
                                                                                                                 1)


lib.mmMultQuadraticHierarchic.argtypes = c_int, c_int, c_int, doublepp, doublepp, POINTER(c_double), POINTER(
    c_double), POINTER(c_char)
lib.mmMultQuadraticHierarchic.restype = None


def call_l2_sum_merge(matrizes, weights, colorspace="lab"):
    num_of_matrizes, m, n = len(matrizes), len(matrizes[0]), len(matrizes[0][0])

    new_matrix = np.zeros(matrizes[0].shape, dtype=np.float)
    new_weight = np.zeros(weights[0].shape, dtype=np.float)
    input_matrizes = np.array([np.ctypeslib.as_array(np.asarray(i, dtype=np.float).flatten()) for i in matrizes])
    input_weights = np.array([np.ctypeslib.as_array(np.asarray(i, dtype=np.float).flatten()) for i in weights])

    c_new_matrix_1 = create_cdll_type(new_matrix)
    c_new_weight_1 = create_cdll_type(new_weight)
    c_input_matrizes = (input_matrizes.__array_interface__['data'][0]
                        + np.arange(input_matrizes.shape[0]) * input_matrizes.strides[0]).astype(np.uintp)
    c_input_weights = (input_weights.__array_interface__['data'][0]
                       + np.arange(input_weights.shape[0]) * input_weights.strides[0]).astype(np.uintp)

    lib.mmMultQuadraticHierarchic(m, n, num_of_matrizes, c_input_matrizes, c_input_weights, c_new_matrix_1,
                                  c_new_weight_1,
                                  colorspace.encode('utf-8'))
    return np.ctypeslib.as_array(c_new_matrix_1).reshape(m, n, 3), np.ctypeslib.as_array(c_new_weight_1).reshape(m, n,
                                                                                                                 1)


lib.mmMultSumHierarchic.argtypes = c_int, c_int, c_int, doublepp, doublepp, POINTER(c_double), POINTER(
    c_double), POINTER(c_char)
lib.mmMultSumHierarchic.restype = None


def call_hierarchic_alpha_sum_merge(matrizes, weights, colorspace="lab"):
    """
    Combines multiple matrizes hierarchic and returns the combined image. Uses hierarchic porter-duff-source-over
    Only works with rgb-Images atm

    :param colorspace: rgb and lab supported so far
    :param matrizes: [image_1, ... , image_n] list of images to use
    :param weights: [weight_1, ... , weight_n] corresponding weights to images
    :return: hierarchically build new 2D-image
    """
    num_of_matrizes, m, n = len(matrizes), len(matrizes[0]), len(matrizes[0][0])

    new_matrix = np.zeros(matrizes[0].shape, dtype=np.float)
    new_weight = np.zeros(weights[0].shape, dtype=np.float)
    input_matrizes = np.array([np.ctypeslib.as_array(np.asarray(i, dtype=np.float).flatten()) for i in matrizes])
    input_weights = np.array([np.ctypeslib.as_array(np.asarray(i, dtype=np.float).flatten()) for i in weights])

    c_new_matrix_1 = create_cdll_type(new_matrix)
    c_new_weight_1 = create_cdll_type(new_weight)
    c_input_matrizes = (input_matrizes.__array_interface__['data'][0]
                        + np.arange(input_matrizes.shape[0]) * input_matrizes.strides[0]).astype(np.uintp)
    c_input_weights = (input_weights.__array_interface__['data'][0]
                       + np.arange(input_weights.shape[0]) * input_weights.strides[0]).astype(np.uintp)

    lib.mmMultSumHierarchic(m, n, num_of_matrizes, c_input_matrizes, c_input_weights, c_new_matrix_1, c_new_weight_1,
                            colorspace.encode('utf-8'))
    return np.ctypeslib.as_array(c_new_matrix_1).reshape(m, n, 3), np.ctypeslib.as_array(c_new_weight_1).reshape(m, n,
                                                                                                                 1)


if __name__ == "__main__":
    from src import color_schemes
    from src import picture_contours
    from src import helper

    from src import hierarchic_blending_operator

    x_min, x_max = -10, 10
    y_min, y_max = -10, 10
    size = 3

    mu_x_1 = 0
    mu_y_1 = 0
    mu_variance_x_1 = 3
    mu_variance_y_1 = 15
    gaussian_1 = ([mu_x_1, mu_y_1], [[mu_variance_x_1, 0.], [0., mu_variance_y_1]])
    mu_x_2 = 3
    mu_y_2 = 3
    mu_variance_x_2 = 4
    mu_variance_y_2 = 4
    gaussian_2 = ([mu_x_2, mu_y_2], [[mu_variance_x_2, 0.], [0., mu_variance_y_2]])
    mu_x_3 = -2
    mu_y_3 = -1
    mu_variance_x_3 = 7
    mu_variance_y_3 = 7

    gaussian_3 = ([mu_x_3, mu_y_3], [[mu_variance_x_3, 0.], [0., mu_variance_y_3]])
    X, Y, Z = helper.get_gaussian(x_min, x_max, y_min, y_max, *gaussian_1, size)
    X_1, Y_1, Z_1 = helper.get_gaussian(x_min, x_max, y_min, y_max, *gaussian_2, size)
    X_2, Y_2, Z_2 = helper.get_gaussian(x_min, x_max, y_min, y_max, *gaussian_3, size)
    Z_color, Z_alpha = picture_contours.get_colorgrid(Z, color_schemes.matplotlib_colorschemes, 10,
                                                      colorscheme_name="PuBu")
    Z_color_1, Z_alpha_1 = picture_contours.get_colorgrid(Z_1, color_schemes.matplotlib_colorschemes, 10,
                                                          colorscheme_name="OrRd")
    Z_color_2, Z_alpha_2 = picture_contours.get_colorgrid(Z_2, color_schemes.matplotlib_colorschemes, 10,
                                                          colorscheme_name="RdPu")
    matrizen = [Z, Z_1]
    colors = [Z_color, Z_color_1]
    result = callSimpleMerge(colors, matrizen)
    print(result[1])
    print(picture_contours.combine_multiple_images_hierarchic(hierarchic_blending_operator.porter_duff_source_over,
                                                              [Z_color, Z_color_1], [Z,
                                                                                     Z_1])[1])
    print(result[0])
    print(picture_contours.combine_multiple_images_hierarchic(hierarchic_blending_operator.porter_duff_source_over,
                                                              [Z_color, Z_color_1], [Z,
                                                                                     Z_1])[0])

    print("---------------------------------------------------------")
    print("-----------------Hierarchic------------------------------")

    pics = [Z_color, Z_color_1, Z_color_2]
    weights = [Z, Z_1, Z_2]
    results = call_hierarchic_merge(pics, weights)
    print(results[0])
    print(picture_contours.combine_multiple_images_hierarchic(hierarchic_blending_operator.porter_duff_source_over,
                                                              pics, weights)[0])
