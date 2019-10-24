import sys

import numpy as np

from contour_visualization.picture_contours import logger

import importlib

iso_mod = importlib.util.find_spec("density_visualization.iso_levels")
iso_lvl_exists = iso_mod is not None
if iso_lvl_exists:
    from density_visualization import iso_levels
else:
    logger.warning("Iso-Level-Lib not found. Using default method")


def get_iso_levels(x, method="equal_density", num_of_levels=8):
    """
    generates iso-lines to a given 2D-array

    :param x: picture to generate iso-levels to
    :param method: normal or equal_density available right now
    :param num_of_levels: number of iso-lines to generate
    :return: array([iso_line_1, ... , iso_line_n ])
    """
    if iso_lvl_exists:
        if "density_visualization.iso_levels" not in sys.modules:
            logger.info("module density_visualization.iso_levels is missing. Normal iso-levels have been used")
            return normal_iso_level(x, num_of_levels)
        if method == "equal_density":
            return iso_levels.equi_prob_per_level(x, k=num_of_levels)
        elif method == "equal_horizontal":
            return iso_levels.equi_horizontal_prob_per_level(x, k=num_of_levels)
        elif method == "equal_value":
            return iso_levels.equi_value(x, k=num_of_levels)
    return normal_iso_level(x, num_of_levels)


def normal_iso_level(x, num_of_levels):
    return np.linspace(np.min(x), np.max(x), num_of_levels + 2)[1:-1]
