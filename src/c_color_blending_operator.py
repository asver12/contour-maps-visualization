from ctypes import cdll, c_double, POINTER, Structure
import os
from numpy.ctypeslib import ndpointer
import numpy as np

try:
    local_dir = os.path.dirname(__file__)
    lib = cdll.LoadLibrary(local_dir + "/../libs/libblendingOperators.so")
except OSError as e:
    print("File libsvmBlend.so could not be found under {}".format(
        os.path.dirname(__file__) + "/../libs/libblendingOperators.so"))

array_type = c_double * 3

lib.porterDuffSourceOver.argtypes = POINTER(c_double), POINTER(c_double), c_double
lib.porterDuffSourceOver.restype = ndpointer(dtype=c_double, shape=(3,))


def porter_duff_source_over_old(color_1, color_2, alpha):
    """
    uses the source-over Porter-Duff operator to calculate a new color C with one alpha for each pixel
    if one of the colors is not np.array it will be transformed

    :param color_1: np.array, list or tuple without alpha (size 3)
    :param color_2: np.array, list or tuple without alpha (size 3)
    :param alpha: in [0,1]
    :return: np.array(color) with same size and color-space as input
    """
    color_1 = np.asarray(color_1, dtype=np.float)
    color_2 = np.asarray(color_2, dtype=np.float)
    return lib.porterDuffSourceOver(array_type(*color_1), array_type(*color_2), alpha)


lib.weightedPorterDuffSourceOver.argtypes = POINTER(c_double), c_double, POINTER(c_double), c_double


class returnStruct(Structure):
    _fields_ = [("returnList", POINTER(c_double)),
                ("returnWeight", c_double)]


lib.weightedPorterDuffSourceOver.restype = returnStruct


def porter_duff_source_over(color_1, z_1, color_2, z_2):
    resultStruct = lib.weightedPorterDuffSourceOver(array_type(*color_1), z_1, array_type(*color_2), z_2)
    return np.ctypeslib.as_array(resultStruct.returnList, shape=(3,)), resultStruct.returnWeight


if __name__ == "__main__":
    rgb_front = [0.5, 0.5, 0.5]
    rgb_back = [0.3, 0.3, 0.3]
    alpha = 0.5
    weight_1 = 0.1
    weight_2 = 0.4
    print("Front: {}".format(rgb_front))
    print("Back: {}".format(rgb_back))
    print("Result normal: {}".format(porter_duff_source_over_old(rgb_front, rgb_back, alpha)))
    print("Weights: {} , {}".format(weight_1, weight_2))
    print("Expected: {}".format(
        rgb_front[0] * (weight_1 / (weight_1 + weight_2)) + rgb_back[0] * (1 - weight_1 / (weight_1 + weight_2))))
    wow = porter_duff_source_over(rgb_front, weight_1, rgb_back, weight_2)
    print("Result weighted: {}".format(wow))