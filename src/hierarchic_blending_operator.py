from src import ddhp_color_blending


def porter_duff_source_over_weighted(color_1, z_1, color_2, z_2):
    alpha = z_1 / (z_1 + z_2)
    return color_1 * alpha + color_2 * (1 - alpha), z_1 * alpha + z_2 * (1 - alpha)


def select_max(color_1, z_1, color_2, z_2):
    if z_1 > z_2:
        color_new = color_1
        z_new = z_1
    else:
        color_new = color_2
        z_new = z_2
    return color_new, z_new


def porter_duff_source_over_quad_weighted(color_1, z_1, color_2, z_2):
    alpha = z_1 / (z_1 + z_2) ** 2
    return color_1 * alpha + color_2 * (1 - alpha), z_1 * alpha + z_2 * (1 - alpha)


def ddhp_color_blending_weighted(color_1, z_1, color_2, z_2):
    alpha = z_1 / (z_1 + z_2) ** 2
    return ddhp_color_blending.blend_rgb_colors(color_1, color_2, alpha), z_1 * alpha + z_2 * (1 - alpha)
