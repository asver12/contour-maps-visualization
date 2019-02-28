from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from numpy.core._multiarray_umath import ndarray
from skimage import color

from src import color_operations

def combine_two_images(blending_operator, img, z_1, img2, z_2):
    z_new: ndarray = np.zeros([len(img), len(img[0]), 1])
    reduce = np.zeros([len(img), len(img[0]), 3])
    for i in range(len(img)):
        for j in range(len(img[0])):
            reduce[i][j], z_new[i][j] = blending_operator(img[i][j], z_1[i][j], img2[i][j], z_2[i][j])
    return reduce, z_new
