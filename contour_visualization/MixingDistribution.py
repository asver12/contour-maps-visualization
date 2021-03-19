from contour_visualization.Distribution import Distribution


class MixingDistribution(Distribution):

    def __init__(self, gaussian_1, gaussian_2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gaussian_1 = gaussian_1
        self.gaussian_2 = gaussian_2

    def get_density_grid(self, size=None, x_min=None, x_max=None, y_min=None, y_max=None):
        x_1, y_1, weights_1 = self.gaussian_1.get_density_grid()
        x_2, y_2, weights_2 = self.gaussian_2.get_density_grid()
        return x_1 + x_2, y_1 + y_2, (weights_1 + weights_2) / 2

    def get_density(self, x):
        return self.gaussian_1.get_density(x) + self.gaussian_2.get_density(x)


