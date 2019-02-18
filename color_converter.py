def rgb_to_hex(rgb):
    """
    converts rgb color to hex color
    :param rgb: [r,g,b] each in [0,1]
    :return: "xrrggbb
    """
    return "#" + "".join("x{:02x}".format(int(x * 255))[1:] for x in rgb)


def rgb255_to_rgb01(rgba):
    return [i / 255 for i in rgba]


def rgb01_to_rgb255(rgba):
    return [int(i * 255) for i in rgba]



if __name__ == "__main__":
    color = [1, 0, 0]
    color_255 = rgb01_to_rgb255(color)
    print(color_255)
    color_01 = rgb255_to_rgb01(color_255)
    print(color_01)
    print("{} =^= {}".format(color_01, rgb_to_hex(color)))
