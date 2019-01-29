import numpy as np

def porter_duff_source_over(color_1, color_2, alpha):
    """
    uses the source-over Porter-Duff operator to calculate a new color C
    if one of the colors is not np.array it will be transformt
    :param color_1: (r,g,b,a)
    :param color_2: (r,g,b,a)
    :param alpha: in [0,1]
    :return: np.array([r,g,b])
    """
    if not isinstance(color_1, np.ndarray):
        color_1 = np.array(color_1)
    if not isinstance(color_2, np.ndarray):
        color_2 = np.array(color_2)
    return color_1*alpha + color_2*(1-alpha)