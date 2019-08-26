import numpy as np

from src.Distribution import Distribution


class ModelbaseDistribution(Distribution):
    def __init__(self, model, used_categorical_name, categorical_attribute, attributes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.used_categorical_name = used_categorical_name
        self.model = model
        self.model.marginalize(keep=[categorical_attribute, *attributes])
        self.vec_density = np.vectorize(self.get_density_wrapper, signature="(),()->()")
        if "x_min" not in kwargs.keys():
            min_values = self.model.data[attributes].min()
            max_values = self.model.data[attributes].max()
        else:
            min_values = self.x_min, self.y_min
            max_values = self.x_max, self.y_max
            # max_values = self.model.data[self.model.data[categorical_attribute] == used_categorical_name][attributes].max()
            # min_values = self.model.data[self.model.data[categorical_attribute] == used_categorical_name][attributes].min()
        self.x_min = min_values[0] - abs(max_values[0] - min_values[0]) * 0.2
        self.x_max = max_values[0] + abs(max_values[0] - min_values[0]) * 0.2
        self.y_min = min_values[1] - abs(max_values[1] - min_values[1]) * 0.2
        self.y_max = max_values[1] + abs(max_values[1] - min_values[1]) * 0.2

    def get_density_grid(self, size=None, x_min=None, x_max=None, y_min=None, y_max=None):
        if x_min is None:
            x_min = self.x_min
        if x_max is None:
            x_max = self.x_max
        if y_min is None:
            y_min = self.y_min
        if y_max is None:
            y_max = self.y_max
        if size is None:
            size = self.size
        x_list = np.linspace(x_min, x_max, size)
        y_list = np.linspace(y_min, y_max, size)
        x, y = np.meshgrid(x_list, y_list)
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        return x_list, y_list, self.vec_density(x=x, y=y)

    def get_density_wrapper(self, x, y):
        return self.model.density([self.used_categorical_name, x, y])

    def get_density(self, x):
        return self.model.density([self.used_categorical_name, *x])
