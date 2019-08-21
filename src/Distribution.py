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
        pass

    @abstractmethod
    def get_density_grid(self, size=None, x_min=None, x_max=None, y_min=None, y_max=None):
        pass

    def get_attributes(self):
        return [self.x_min, self.x_max, self.y_min, self.y_max, self.size, self.weight]

    def __str__(self):
        return "[x_min: {}, x_max: {}, y_min: {}, y_max: {}, weight: {}, size: {}]".format(self.x_min, self.x_max, self.y_min, self.y_max, self.weight, self.size)
