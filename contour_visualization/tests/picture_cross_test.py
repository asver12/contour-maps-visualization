import unittest
from contour_visualization import picture_cross
import numpy as np


class PictureCrossTest(unittest.TestCase):
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
        x_min = -10
        x_max = 10
        y_min = -10
        y_max = 10
        size = 100
        x_list = np.linspace(x_min, x_max, size)
        y_list = np.linspace(y_min, y_max, size)
        cls.x_list, cls.y_list = np.meshgrid(x_list, y_list)

    def test_find_point_indices_value_error(self):
        with self.assertRaises(ValueError):
            picture_cross.find_point_indices((0, 0, 0), self.x_list, self.y_list)

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


if __name__ == '__main__':
    unittest.main()
