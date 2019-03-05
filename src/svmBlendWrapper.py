from ctypes import cdll, c_int, c_float

try:
    lib = cdll.LoadLibrary("./libs/libsvmBlend.so")
except OSError as e:
    print(e)


def blend_rgb_colors(front_rgb, back_rgb, alpha):
    lib.blend_rgb_colors.argtypes = c_int, c_int, c_float
    return int_to_array(lib.blend_rgb_colors(array_to_int(front_rgb), array_to_int(back_rgb), alpha))


def array_to_int(rgb):
    return int(rgb[0]*255) | (int(rgb[1]*255) << 8) | (int(rgb[2]*255) << 16)


def int_to_array(rgb_packed):
    return [((rgb_packed >> (i * 8)) & 0xFF)/255 for i in range(3)]


if __name__ == "__main__":
    rgb_front = [255, 0, 0]
    rgb_back = [0, 128, 0]
    alpha = 0.5
    print("Front: {}".format(rgb_front))
    print("Back: {}".format(rgb_back))
    print("Result: {}".format(blend_rgb_colors(rgb_front, rgb_back, alpha)))
