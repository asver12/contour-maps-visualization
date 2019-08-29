from abc import ABC, abstractmethod


class Distribution(ABC):
    def __init__(self, x_min=-10, x_max=10, y_min=-10, y_max=10, size=200, weight=1):
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
        returns the density on a grid given by each endpoint. The grid is uniform.

        :param size: number of points per row and column
        :param x_min: start of x-axis
        :param x_max: end of x-axis
        :param y_min: start of y-axis
        :param y_max: end of y-axis
        :return: grid with size = size*size from x_min to x_max and y_min to y_max
        """
        pass

    def get_attributes(self):
        return [self.x_min, self.x_max, self.y_min, self.y_max, self.size, self.weight]

    def __str__(self):
        return "[x_min: {}, x_max: {}, y_min: {}, y_max: {}, weight: {}, size: {}]".format(self.x_min, self.x_max, self.y_min, self.y_max, self.weight, self.size)
