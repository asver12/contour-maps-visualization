import unittest
from contour_visualization import picture_cross, helper, color_schemes
import numpy as np

from contour_visualization.Gaussian import Gaussian


class PictureCrossTests(unittest.TestCase):
    # def test_get_cross(self):
    #     pass
    #
    # def test_get_line(self):
    #     pass

    def test_get_half_lines_basic(self):
        first_line, second_line = picture_cross.get_half_lines((0, 0), (0, 2), 2)
        self.assertSequenceEqual(first_line, ((0, -2), (0, 0)))
        self.assertSequenceEqual(second_line, ((0, 0), (0, 2)))
        first_line, second_line = picture_cross.get_half_lines((0, 0), (0, -2), 2)
        self.assertSequenceEqual(first_line, ((0, 2), (0, 0)))
        self.assertSequenceEqual(second_line, ((0, 0), (0, -2)))
        first_line, second_line = picture_cross.get_half_lines((0, 0), (2, 0), 2)
        self.assertSequenceEqual(first_line, ((-2, 0), (0, 0)))
        self.assertSequenceEqual(second_line, ((0, 0), (2, 0)))
        first_line, second_line = picture_cross.get_half_lines((0, 0), (-2, 0), 2)
        self.assertSequenceEqual(first_line, ((2, 0), (0, 0)))
        self.assertSequenceEqual(second_line, ((0, 0), (-2, 0)))

    def test_get_half_lines_different_directions(self):
        first_line, second_line = picture_cross.get_half_lines((0, 0), (2, 3), 3.605551275463989)
        self.assertSequenceEqual(first_line, ((-2., -3.), (0, 0)))
        self.assertSequenceEqual(second_line, ((0, 0), (2., 3.)))

    def test_get_half_lines_different_directions_and_startpoint(self):
        # different x-value
        first_line, second_line = picture_cross.get_half_lines((2, 0), (2, 3), 3.605551275463989)
        self.assertSequenceEqual(first_line, ((0., -3.), (2, 0)))
        self.assertSequenceEqual(second_line, ((2, 0), (4., 3.)))
        # different y-value
        first_line, second_line = picture_cross.get_half_lines((0, 2), (2, 3), 3.605551275463989)
        self.assertSequenceEqual(first_line, ((-2., -1.), (0, 2)))
        self.assertSequenceEqual(second_line, ((0, 2), (2., 5.)))

    def test_get_half_lines_reversed(self):
        # neg x-value
        first_line, second_line = picture_cross.get_half_lines((0, 2), (-2, 3), 3.605551275463989)
        self.assertSequenceEqual(first_line, ((2., -1.), (0, 2)))
        self.assertSequenceEqual(second_line, ((0, 2), (-2., 5.)))
        # neg y-value
        first_line, second_line = picture_cross.get_half_lines((0, 2), (2, -3), 3.605551275463989)
        self.assertSequenceEqual(first_line, ((-2., 5.), (0, 2)))
        self.assertSequenceEqual(second_line, ((0, 2), (2., -1.)))

    @classmethod
    def setUpClass(cls):
        cls.x_min = -10
        cls.x_max = 10
        cls.y_min = -10
        cls.y_max = 10
        cls.size = 100
        cls.weight = 0.5
        x_list = np.linspace(cls.x_min, cls.x_max, cls.size)
        y_list = np.linspace(cls.y_min, cls.y_max, cls.size)
        cls.x_list, cls.y_list = np.meshgrid(x_list, y_list)

    def test_generate_crosses_crosses(self):
        dist_one_first_line = [((1., -4.), (5., 4.440892098500626e-16)),
                               ((-0.181818, -2.818181),
                                (3.818181, 1.181818)),
                               ((-1., -2.),
                                (3., 2.)),
                               ((-1.818181, -1.181818),
                                (2.181818, 2.818181)),
                               ((-3., -4.440892098500626e-16), (1., 4.))]
        dist_one_second_line = [((-1.4721359549995792, -6.47213595499958),
                                 (-5.47213595499958, -2.472135954999579)),
                                ((1.1162821280456323, -3.8837178719543686),
                                 (-2.8837178719543686, 0.11628212804563232)),
                                ((3.0000000000000004, -2.0000000000000004),
                                 (-1.0000000000000004, 2.0000000000000004)),
                                ((4.883717871954369, -0.11628212804563187),
                                 (0.8837178719543681, 3.883717871954369)),
                                ((7.47213595499958, 2.472135954999579),
                                 (3.472135954999579, 6.47213595499958))]
        dist_two_first_line = [((0.0, -7.000000000000002), (8.000000000000002, 1.0)),
                               ((-2.363636363636364, -4.636363636363638),
                                (5.636363636363638, 3.363636363636364)),
                               ((-4.000000000000001, -3.000000000000001),
                                (4.000000000000001, 5.000000000000001)),
                               ((-5.636363636363638, -1.3636363636363642),
                                (2.363636363636364, 6.636363636363638)),
                               ((-8.000000000000002, 1.0), (0.0, 9.000000000000002))]
        dist_two_second_line = [((-1.6568542494923806, -8.656854249492383),
                                 (-9.656854249492383, -0.6568542494923806)),
                                ((1.6172644221835126, -5.382735577816489),
                                 (-6.382735577816489, 2.6172644221835126)),
                                ((4.000000000000001, -3.000000000000001),
                                 (-4.000000000000001, 5.000000000000001)),
                                ((6.382735577816488, -0.6172644221835135),
                                 (-1.6172644221835135, 7.382735577816488)),
                                ((9.656854249492383, 2.6568542494923806),
                                 (1.6568542494923806, 10.656854249492383))]

        dist_1 = Gaussian(means=[1, 0], cov_matrix=[[3, 2], [2, 3]], x_min=self.x_min, x_max=self.x_max,
                          y_min=self.y_min, y_max=self.y_max,
                          size=self.size, weight=self.weight)
        dist_2 = Gaussian(means=[0, 1], cov_matrix=[[6, 2], [2, 6]], x_min=self.x_min, x_max=self.x_max,
                          y_min=self.y_min, y_max=self.y_max,
                          size=self.size, weight=self.weight)
        distributions = [dist_1, dist_2]

        # unpleasant....
        limits = helper.get_limits(distributions)
        z_list = helper.generate_distribution_grids(distributions, limits=limits)
        z_min, z_max, z_sum = helper.generate_weights(z_list)
        colorschemes = color_schemes.get_colorbrewer_schemes()

        crosses = picture_cross.generate_crosses(distributions, z_list, z_min, z_max, colorschemes, method="normal",
                                                 num_of_levels=1)
        np.testing.assert_almost_equal(dist_one_first_line, crosses[0][0], decimal=5)
        np.testing.assert_almost_equal(dist_one_second_line, crosses[0][1], decimal=5)
        np.testing.assert_almost_equal(dist_two_first_line, crosses[1][0], decimal=5)
        np.testing.assert_almost_equal(dist_two_second_line, crosses[1][1], decimal=5)

    def test_generate_crosses_colors(self):
        dist_one_colors = [[0.77647059, 0.85882353, 0.9372549, 1.],
                           [0.12941176, 0.44313725, 0.70980392, 1.],
                           [.12941176, 0.44313725, 0.70980392, 1.],
                           [0.77647059, 0.85882353, 0.9372549, 1.]]

        dist_two_colors = [[0.99607843, 0.92428677, 0.85887397, 1.],
                           [0.99215686, 0.77615063, 0.57159781, 1.],
                           [0.99215686, 0.77615063, 0.57159781, 1.],
                           [0.99607843, 0.92428677, 0.85887397, 1.]]

        dist_1 = Gaussian(means=[1, 0], cov_matrix=[[3, 2], [2, 3]], x_min=self.x_min, x_max=self.x_max,
                          y_min=self.y_min, y_max=self.y_max,
                          size=self.size, weight=self.weight)
        dist_2 = Gaussian(means=[0, 1], cov_matrix=[[6, 2], [2, 6]], x_min=self.x_min, x_max=self.x_max,
                          y_min=self.y_min, y_max=self.y_max,
                          size=self.size, weight=self.weight)
        distributions = [dist_1, dist_2]

        # unpleasant....
        limits = helper.get_limits(distributions)
        z_list = helper.generate_distribution_grids(distributions, limits=limits)
        z_min, z_max, z_sum = helper.generate_weights(z_list)
        colorschemes = color_schemes.get_colorbrewer_schemes()

        crosses = picture_cross.generate_crosses(distributions, z_list, z_min, z_max, colorschemes, method="normal",
                                                 num_of_levels=1)
        np.testing.assert_almost_equal(dist_one_colors, crosses[0][2])
        np.testing.assert_almost_equal(dist_one_colors, crosses[0][3])
        np.testing.assert_almost_equal(dist_two_colors, crosses[1][2])
        np.testing.assert_almost_equal(dist_two_colors, crosses[1][3])

    #
    def test_generate_crosses_iso_lines(self):
        dist_one_iso_lines = [0.25, 0.75, 0.75, 0.25]

        dist_two_iso_lines = [0.09889134000871382,
                              0.2966740200253193,
                              0.2966740200253193,
                              0.09889134000871382]
        dist_1 = Gaussian(means=[1, 0], cov_matrix=[[3, 2], [2, 3]], x_min=self.x_min, x_max=self.x_max,
                          y_min=self.y_min, y_max=self.y_max,
                          size=self.size, weight=self.weight)
        dist_2 = Gaussian(means=[0, 1], cov_matrix=[[6, 2], [2, 6]], x_min=self.x_min, x_max=self.x_max,
                          y_min=self.y_min, y_max=self.y_max,
                          size=self.size, weight=self.weight)
        distributions = [dist_1, dist_2]

        # unpleasant....
        limits = helper.get_limits(distributions)
        z_list = helper.generate_distribution_grids(distributions, limits=limits)
        z_min, z_max, z_sum = helper.generate_weights(z_list)
        colorschemes = color_schemes.get_colorbrewer_schemes()

        crosses = picture_cross.generate_crosses(distributions, z_list, z_min, z_max, colorschemes, method="normal", num_of_levels=1)
        np.testing.assert_almost_equal(dist_one_iso_lines, crosses[0][4])
        # np.testing.assert_almost_equal(dist_two_iso_lines, crosses[1][4])

    def test_find_point_indices_value_error(self):
        # throws an error when more or less than 2 points are given
        with self.assertRaises(ValueError):
            picture_cross.find_point_indices((0, 0, 0), self.x_list, self.y_list)
        with self.assertRaises(ValueError):
            picture_cross.find_point_indices((0,), self.x_list, self.y_list)

    def test_find_point_indices_range_error(self):
        with self.assertRaises(ValueError):
            picture_cross.find_point_indices((-11, 0), self.x_list, self.y_list)
        with self.assertRaises(ValueError):
            picture_cross.find_point_indices((11, 0), self.x_list, self.y_list)
        with self.assertRaises(ValueError):
            picture_cross.find_point_indices((0, -11), self.x_list, self.y_list)
        with self.assertRaises(ValueError):
            picture_cross.find_point_indices((0, 11), self.x_list, self.y_list)

    def test_find_point_indices(self):
        index_x, index_y = picture_cross.find_point_indices((-2, 0), self.x_list, self.y_list)
        self.assertSequenceEqual([index_x, index_y], [40, 50])
        index_x, index_y = picture_cross.find_point_indices((0, 0), self.x_list, self.y_list)
        self.assertSequenceEqual([index_x, index_y], [50, 50])

    @classmethod
    def tearDownClass(cls):
        pass

    def test_map_points_range_error(self):
        # Number value and mapping points not matching
        points = np.linspace(-0.5, 3.5, 9)
        value_line = np.linspace(0, 5, 11)
        point_mapping = np.linspace(10, 15, 15)
        with self.assertRaises(ValueError):
            picture_cross.map_points(points, value_line, point_mapping)
        # no points given
        with self.assertRaises(ValueError):
            picture_cross.map_points([], value_line, point_mapping)

    def test_map_points_basic(self):
        points = np.linspace(0.5, 4.5, 9)
        value_line = np.linspace(0, 5, 11)
        point_mapping = np.linspace(10, 15, 11)
        rest = picture_cross.map_points(points, value_line, point_mapping)
        self.assertSequenceEqual(rest, [10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5])

    def test_map_points_exclude_first_and_last(self):
        points = np.linspace(0, 5, 11)
        value_line = np.linspace(0, 5, 11)
        point_mapping = np.linspace(10, 15, 11)
        rest = picture_cross.map_points(points, value_line, point_mapping)
        self.assertSequenceEqual(rest, [10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5])

    def test_map_points__points_out_of_range(self):
        # to far left
        points = np.linspace(-0.5, 3.5, 9)
        value_line = np.linspace(0, 5, 11)
        point_mapping = np.linspace(10, 15, 11)
        rest = picture_cross.map_points(points, value_line, point_mapping)
        self.assertSequenceEqual(rest, [10.5, 11, 11.5, 12, 12.5, 13, 13.5])

        # to fare right
        points = np.linspace(1.5, 5.5, 9)
        value_line = np.linspace(0, 5, 11)
        point_mapping = np.linspace(10, 15, 11)
        rest = picture_cross.map_points(points, value_line, point_mapping)
        self.assertSequenceEqual(rest, [11.5, 12, 12.5, 13, 13.5, 14, 14.5])

    def test_get_distance_basic(self):
        point_1 = (1.5, 3.1)
        point_2 = (3.3, 4.1)
        length = picture_cross.get_distance(point_1, point_2)
        self.assertAlmostEqual(length, 2.059126)

    def test_get_distance_y(self):
        point_1 = (0, 1)
        point_2 = (1, 1)
        length = picture_cross.get_distance(point_1, point_2)
        self.assertEqual(length, 1)

        length = picture_cross.get_distance(point_2, point_1)
        self.assertEqual(length, 1)

    def test_get_distance_x(self):
        point_1 = (1, 0)
        point_2 = (1, 1)
        length = picture_cross.get_distance(point_1, point_2)
        self.assertEqual(length, 1)

        length = picture_cross.get_distance(point_2, point_1)
        self.assertEqual(length, 1)

    def test_get_distance_3d(self):
        point_1 = (7, 4, 3)
        point_2 = (17, 6, 2)
        length = picture_cross.get_distance(point_1, point_2)
        self.assertAlmostEqual(length, 10.246951, places=5)

    def test_get_distance_assert_length_error(self):
        point_1 = (1, 0, 0)
        point_2 = (1, 1)
        with self.assertRaises(ValueError):
            picture_cross.get_distance(point_1, point_2)
        point_1 = (1, 1)
        point_2 = (1, 0, 0)
        with self.assertRaises(ValueError):
            picture_cross.get_distance(point_1, point_2)

    def test_get_distance_assert_type_error(self):
        point_1 = "hallo"
        point_2 = "ichbi"
        with self.assertRaises(TypeError):
            picture_cross.get_distance(point_1, point_2)

    def test_get_direction_basic(self):
        point_1 = (1.5, 3.1)
        point_2 = (3.3, 4.1)
        direction = picture_cross.get_direction(point_1, point_2)
        self.assertAlmostEqual(direction[1], 0.48564, places=4)
        self.assertAlmostEqual(direction[0], 0.87415, places=4)

    def test_get_direction_diag(self):
        point_1 = (0, 0)
        point_2 = (1, 1)
        direction = picture_cross.get_direction(point_1, point_2)
        self.assertAlmostEqual(direction[0], 0.70710, places=4)
        self.assertAlmostEqual(direction[1], 0.70710, places=4)

    def test_get_direction_x(self):
        point_1 = (1, 0)
        point_2 = (1, 1)
        direction = picture_cross.get_direction(point_1, point_2)
        self.assertSequenceEqual(list(direction), [0, 1])

    def test_get_direction_y(self):
        point_1 = (0, 1)
        point_2 = (1, 1)
        direction = picture_cross.get_direction(point_1, point_2)
        self.assertSequenceEqual(list(direction), (1, 0))

    def test_get_direction_assert_length_error(self):
        point_1 = (1, 0, 0)
        point_2 = (1, 1)
        with self.assertRaises(ValueError):
            picture_cross.get_direction(point_1, point_2)
        point_1 = (1, 1)
        point_2 = (1, 0, 0)
        with self.assertRaises(ValueError):
            picture_cross.get_direction(point_1, point_2)

    def test_get_relatie_broader_interval_basic(self):
        point_1 = (3.2, 1.5)
        point_2 = (-.5, 1.7)
        point_1_new, point_2_new = picture_cross.get_broader_interval(point_1, point_2, 1)
        results = ((6.9, 1.3), (-4.2, 1.9))
        for i, point in enumerate([point_1_new, point_2_new]):
            self.assertSequenceEqual(list(point), results[i])

    def test_get_relative_broader_interval_diag(self):
        point_1 = (0, 0)
        point_2 = (1, 1)
        point_1_new, point_2_new = picture_cross.get_broader_interval(point_1, point_2, 1)
        results = ((-1., -1.), (2., 2.))
        for i, point in enumerate([point_1_new, point_2_new]):
            self.assertSequenceEqual(list(point), results[i])

    def test_get_relative_broader_interval_diag_reversed(self):
        point_1 = (1, 1)
        point_2 = (0, 0)
        point_1_new, point_2_new = picture_cross.get_broader_interval(point_1, point_2, 1)
        results = ((2., 2.), (-1., -1.))
        for i, point in enumerate([point_1_new, point_2_new]):
            self.assertSequenceEqual(list(point), results[i])

    def test_get_relative_broader_interval_diag_2(self):
        point_1 = (0, 0)
        point_2 = (1, 1)
        point_1_new, point_2_new = picture_cross.get_broader_interval(point_1, point_2, .5)
        results = ((-.5, -.5), (1.5, 1.5))
        for i, point in enumerate([point_1_new, point_2_new]):
            self.assertSequenceEqual(list(point), results[i])

    def test_get_relative_broader_interval_x(self):
        point_1 = (0, 0)
        point_2 = (1, 0)
        point_1_new, point_2_new = picture_cross.get_broader_interval(point_1, point_2, 1)
        results = ((-1., 0.), (2., 0.))
        for i, point in enumerate([point_1_new, point_2_new]):
            self.assertSequenceEqual(list(point), results[i])

    def test_get_relative_broader_interval_x_reversed(self):
        point_1 = (1, 0)
        point_2 = (0, 0)
        point_1_new, point_2_new = picture_cross.get_broader_interval(point_1, point_2, 1)
        results = ((2., 0.), (-1., 0.))
        for i, point in enumerate([point_1_new, point_2_new]):
            self.assertSequenceEqual(list(point), results[i])

    def test_get_relative_broader_interval_y(self):
        point_1 = (0, 0)
        point_2 = (0, 1)
        point_1_new, point_2_new = picture_cross.get_broader_interval(point_1, point_2, 1)
        results = ((0., -1.), (0., 2.))
        for i, point in enumerate([point_1_new, point_2_new]):
            self.assertSequenceEqual(list(point), results[i])

    def test_get_relative_broader_interval_y_reversed(self):
        point_1 = (0, 1)
        point_2 = (0, 0)
        point_1_new, point_2_new = picture_cross.get_broader_interval(point_1, point_2, 1)
        results = ((0., 2.), (0., -1.))
        for i, point in enumerate([point_1_new, point_2_new]):
            self.assertSequenceEqual(list(point), results[i])

    def test_get_broad_total(self):
        broad = 10
        broad_short, broad_long = picture_cross.get_broad(broad, [], [], False)
        self.assertSequenceEqual([broad_short, broad_long], [10, 10])

    def test_get_broad_total_2(self):
        broad = -2
        broad_short, broad_long = picture_cross.get_broad(broad, [], [], False)
        self.assertSequenceEqual([broad_short, broad_long], [-2, -2])

    def test_get_broad_relative(self):
        broad = "100%"
        broad_short, broad_long = picture_cross.get_broad(broad, [(0, 0), (1, 1)], [(0, 0), (2, 2)], False)
        res = 0.7071067811865476
        self.assertSequenceEqual([broad_short, broad_long], [res * 2, res * 4])

    def test_get_broad_relative_rev(self):
        broad = "100%"
        broad_short, broad_long = picture_cross.get_broad(broad, [(1, 1), (0, 0)], [(2, 2), (0, 0)], False)
        res = 0.7071067811865476
        self.assertSequenceEqual([broad_short, broad_long], [res * 2, res * 4])

    def test_get_broad_relative_x(self):
        broad = "100%"
        broad_short, broad_long = picture_cross.get_broad(broad, [(0, 0), (1, 0)], [(0, 0), (2, 0)], False)
        self.assertSequenceEqual([broad_short, broad_long], [1., 2.])

    def test_get_broad_relative_y(self):
        broad = "100%"
        broad_short, broad_long = picture_cross.get_broad(broad, [(0, 0), (0, 1)], [(0, 0), (0, 2)], False)
        self.assertSequenceEqual([broad_short, broad_long], [1., 2.])

    def test_get_broad_relative_y_same_length(self):
        broad = "100%"
        broad_short, broad_long = picture_cross.get_broad(broad, [(0, 0), (0, 1)], [(0, 0), (0, 2)], True)
        self.assertSequenceEqual([broad_short, broad_long], [1., 1.])

    def test_get_broad_relative_rev_same_length(self):
        broad = "100%"
        broad_short, broad_long = picture_cross.get_broad(broad, [(1, 1), (0, 0)], [(2, 2), (0, 0)], True)
        res = 0.7071067811865476
        self.assertSequenceEqual([broad_short, broad_long], [res * 2, res * 2])

    def test_get_broad_raise_value_error(self):
        # if not in format "50.0%" raise value error
        with self.assertRaises(ValueError):
            picture_cross.get_broad("abc", [(0, 0), (0, 1)], [(0, 0), (0, 2)], False)
        with self.assertRaises(ValueError):
            picture_cross.get_broad("10%1", [(0, 0), (0, 1)], [(0, 0), (0, 2)], False)


if __name__ == '__main__':
    unittest.main()
