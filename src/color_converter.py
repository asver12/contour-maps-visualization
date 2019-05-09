def rgb_to_hex(rgb):
    """
    converts rgb color to hex color

    :param rgb: [r,g,b] each in [0,1]
    :return: #rrggbb
    """
    return "#" + "".join("x{:02x}".format(int(x * 255))[1:] for x in rgb)


def rgb255a_to_rgb01a(rgba):
    return [item / 255 if i + 1 != len(rgba) else item for i, item in enumerate(rgba)]


def hex_to_rgb(hex_rgb):
    value = hex_rgb.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb255_to_rgb01(rgba):
    return [item / 255 for item in rgba]


def rgba01_to_rgba255(rgba):
    return [int(item * 255) if i + 1 != len(rgba) else item for i, item in enumerate(rgba)]


def rgb01_to_rgb255(rgba):
    return [int(item * 255) for item in rgba]


if __name__ == "__main__":
    color = [1, 0, 0, 1]
    color_255 = rgba01_to_rgba255(color)
    print(color_255)
    color_01 = rgb255a_to_rgb01a(color_255)
    print(color_01)
    print("{} =^= {}".format(color_01, rgb_to_hex(color[:3])))
