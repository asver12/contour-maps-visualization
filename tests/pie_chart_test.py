import unittest

from contour_visualization import pie_chart_vis


class MyTestCase(unittest.TestCase):
    def test_sort_ratios_basic(self):
        sorting_list = [3, 5, 2, 6]
        sorted_list = [[0, 1, 2], [4, 6, 8], [5, 1, 2], [2, 8, 8]]
        sorting_list, sorted_list = pie_chart_vis.sort_ratios(sorting_list, sorted_list)
        self.assertSequenceEqual(sorting_list, [2, 3, 5, 6])
        self.assertSequenceEqual(sorted_list, [[5, 1, 2], [0, 1, 2], [4, 6, 8], [2, 8, 8]])

    def test_sort_ratios_basic_2(self):
        sorting_list = [0.3, 0.1, 12, 0.4]
        sorted_list = [[0, 1, 2], [4, 6, 8], [5, 1, 2], [2, 8, 8]]
        sorting_list, sorted_list = pie_chart_vis.sort_ratios(sorting_list, sorted_list)
        self.assertSequenceEqual(sorting_list, [0.1, 0.3, 0.4, 12])
        self.assertSequenceEqual(sorted_list, [[4, 6, 8], [0, 1, 2], [2, 8, 8], [5, 1, 2]])

    def test_sort_ratios_stable(self):
        sorting_list = [3, 6, 5, 3, 5]
        sorted_list = [[0., 1., 2], [4., 6, 8], [5., 1, 2], [2, 8., 8], [5, 5, 5]]
        sorting_list, sorted_list = pie_chart_vis.sort_ratios(sorting_list, sorted_list)
        self.assertSequenceEqual(sorted_list, [[0., 1., 2], [2, 8., 8], [5., 1, 2], [5, 5, 5], [4., 6, 8]])

    def test_sort_ratios_list_same(self):
        sorting_list = [3, 3, 5, 6, 8]
        sorted_list = [[0., 1., 2], [4., 6, 8], [5., 1, 2], [2., 8, 8], [5, 5, 5]]
        sorting_list, sorted_list = pie_chart_vis.sort_ratios(sorting_list, sorted_list)
        self.assertSequenceEqual(sorted_list, sorted_list)
        self.assertSequenceEqual(sorting_list, sorting_list)

    def test_sort_ratios_different_length(self):
        sorting_list = [3, 5, 4]
        sorted_list = [[0., 1., 2], [4., 6, 8], [5., 1, 2], [2., 8, 8], [5, 5, 5]]
        with self.assertRaises(ValueError):
            pie_chart_vis.sort_ratios(sorting_list, sorted_list)

        sorting_list = [3, 3, 5, 6, 8]
        sorted_list = [[0., 1., 2], [4., 6, 8], [5., 1, 2]]
        with self.assertRaises(ValueError):
            pie_chart_vis.sort_ratios(sorting_list, sorted_list)


if __name__ == '__main__':
    unittest.main()
