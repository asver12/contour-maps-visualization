def rgb_to_hex(rgb):
    """
    converts rgb color to hex color
    :param rgb: [r,g,b] each in [0,1]
    :return: "xrrggbb
    """
    return "#" + "".join("x{:02x}".format(int(x*255))[1:] for x in rgb)

if __name__ == "__main__":
    color = [1,0,0]
    print("{} =^= {}".format(color,rgb_to_hex(color)))