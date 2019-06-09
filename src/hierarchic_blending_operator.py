from src import ddhp_color_blending


def porter_duff_source_over(color_1, z_1, color_2, z_2):
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


def lab_color_operator_weighted(color_1, z_1, color_2, z_2):
    new_color = color_1.copy()
    if z_1 < z_2:
        new_color[0] = color_2[0]
    alpha = z_1 / (z_1 + z_2)
    new_color[1] = color_1[1] * alpha + color_2[1] * (1 - alpha)
    new_color[2] = color_1[2] * alpha + color_2[2] * (1 - alpha)
    return new_color, max(z_1, z_2)
