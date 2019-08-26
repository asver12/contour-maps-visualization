from ctypes import cdll, c_int, c_float
import os
try:
    local_dir = os.path.dirname(__file__)
    lib = cdll.LoadLibrary(local_dir +"/../libs/libsvmBlend.so")
except OSError as e:
    print("File libsvmBlend.so could not be found under {}".format(os.path.dirname(__file__) +"/../libs/libsvmBlend.so"))


def blend_rgb_colors(front_rgb, back_rgb, alpha, verbose = False):
    """
    Expects a color in the format rgb each in [0, 1]

    :param front_rgb: [r,g,b]
    :param back_rgb: [r,g,b]
    :param alpha: float in [0,1]
    :return: [r,g,b] in [0,1]
    """
    front_rgb_int = array_to_int(front_rgb)
    back_rgb_int = array_to_int(back_rgb)
    if verbose:
        print("Front: {} | {}".format(front_rgb,type(front_rgb)))
        print("Back: {} | {}".format(back_rgb, type(back_rgb)))
        print("Result: {}".format(color_converter.rgb01_to_rgb255(int_to_array(lib.blend_rgb_colors(front_rgb_int, back_rgb_int, alpha)))))
    lib.blend_rgb_colors.argtypes = c_int, c_int, c_float
    return int_to_array(lib.blend_rgb_colors(front_rgb_int, back_rgb_int, alpha))


def array_to_int(rgb):
    return int(rgb[0]*255) | (int(rgb[1]*255) << 8) | (int(rgb[2]*255) << 16)


def int_to_array(rgb_packed):
    return [((rgb_packed >> (i * 8)) & 0xFF)/255 for i in range(3)]

if __name__ == "__main__":
    from src import color_converter
    rgb_front = [255, 0, 0]
    rgb_back = [0, 128, 0]
    alpha = 0.5
    print("Front: {}".format(rgb_front))
    print("Back: {}".format(rgb_back))
    print("Result: {}".format(color_converter.rgb01_to_rgb255(blend_rgb_colors(color_converter.rgb255_to_rgb01(rgb_front), color_converter.rgb255_to_rgb01(rgb_back), alpha))))

    rgb_front = [219, 217, 234]
    rgb_back = [200, 28, 18]
    alpha = 0.5
    print("Front: {}".format(rgb_front))
    print("Back: {}".format(rgb_back))
    print("Result: {}".format(color_converter.rgb01_to_rgb255(blend_rgb_colors(color_converter.rgb255_to_rgb01(rgb_front), color_converter.rgb255_to_rgb01(rgb_back), alpha))))
