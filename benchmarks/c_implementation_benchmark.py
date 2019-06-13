import timeit


mysetup = """

import src.helper as helper
import src.color_schemes as color_schemes
import src.picture_worker as picture_worker
import src.hierarchic_blending_operator as hierarchic_blending_operator

first_color_scheme = "PuBu"
second_color_scheme = "OrRd"
third_color_scheme = "RdPu"

x_min, x_max = -10, 10
y_min, y_max = -10, 10
size = 100

mu_x_1 = 0
mu_y_1 = 0
mu_variance_x_1 = 3
mu_variance_y_1 = 15
gaussian_1 = (mu_x_1,mu_variance_x_1,mu_y_1,mu_variance_y_1)
mu_x_2 = 3
mu_y_2 = 3
mu_variance_x_2 = 4
mu_variance_y_2 = 4
gaussian_2 = (mu_x_2,mu_variance_x_2,mu_y_2,mu_variance_y_2)
mu_x_3 = -2
mu_y_3 = -1
mu_variance_x_3 = 7
mu_variance_y_3 = 7
gaussian_3 = (mu_x_3,mu_variance_x_3,mu_y_3,mu_variance_y_3)

X, Y, Z = helper.get_gaussian(x_min,x_max,y_min,y_max,*gaussian_1,size)
X_1, Y_1, Z_1 = helper.get_gaussian(x_min,x_max,y_min,y_max,*gaussian_2,size)
X_2, Y_2, Z_2 = helper.get_gaussian(x_min,x_max,y_min,y_max,*gaussian_3,size)

Z_new, Z_alpha = picture_worker.get_colorgrid(Z,color_schemes.matplotlib_colorschemes,10,colorscheme_name=first_color_scheme)
Z_new_1, Z_alpha_1 = picture_worker.get_colorgrid(Z_1,color_schemes.matplotlib_colorschemes,10,colorscheme_name=second_color_scheme)
Z_new_2, Z_alpha_2 = picture_worker.get_colorgrid(Z_2,color_schemes.matplotlib_colorschemes,10,colorscheme_name=third_color_scheme)


"""

py = timeit.timeit("""


mixed, alpha_new = picture_worker.combine_multiple_images_hierarchic(hierarchic_blending_operator.porter_duff_source_over, [Z_new, Z_new_1, Z_new_2], [Z, Z_1, Z_2])
""", setup=mysetup,
                   number=10)

cy = timeit.timeit("""
mixed, alpha_new = c_picture_worker.callHierarchicMerge([Z_new, Z_new_1, Z_new_2], [Z, Z_1, Z_2])
""", setup="""
import src.c_picture_worker as c_picture_worker
""" + mysetup,
                   number=10)

print(cy, py)
if cy < py:
    print("Cython is {}x faster than Python".format(py / cy))
else:
    print("Cython is {}x slower than Python".format(py / cy))