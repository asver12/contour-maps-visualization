from abc import ABC, abstractmethod


class Distribution(ABC):
    """
    Abstract class for the probability distributions which are visualized
    """

    def __init__(self, x_min=-10, x_max=10, y_min=-10, y_max=10, size=200, weight=1):
        """
        Abstract class for the probability distributions which are visualized

        :param x_min: min-x-value in which the vis is calculated
        :param x_max: max-y-value in which the vis is calculated
        :param y_min: min-y-value in which the vis is calculated
        :param y_max: max-y-value in which the vis is calculated
        :param size: number of points per axis
        :param weight: weight of the visualization
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.weight = weight
        self.size = size

    @abstractmethod
    def get_density(self, x):
        """
        Returns the density of the distribution at a given point x = (x,y)

        :param x: (x,y)-coordinates
        :return: z-coordinates
        """
        pass

    @abstractmethod
    def get_density_grid(self, size=None, x_min=None, x_max=None, y_min=None, y_max=None):
        """
        returns the density as a grid given by each endpoint. The grid is uniform.

        :param size: number of points per row and column
        :param x_min: start of x-range
        :param x_max: end of x-range
        :param y_min: start of y-range
        :param y_max: end of y-range
        :return: grid with shape size x size and in range (x_min, ... , x_max) x (y_min, ... , y_max)
        """
        pass

    def get_attributes(self):
        return [self.x_min, self.x_max, self.y_min, self.y_max, self.size, self.weight]

    def __str__(self):
        return "[x_min: {}, x_max: {}, y_min: {}, y_max: {}, weight: {}, size: {}]".format(self.x_min, self.x_max, self.y_min, self.y_max, self.weight, self.size)
