Distribution
============

A distribution is used to provide an interface for the visualization tools to work with.
Each distribution must have a density function and implement all abstract methods given in the distribution class.

.. autoclass:: contour_visualization.Distribution.Distribution
    :members: get_density, get_density_grid
    :noindex:

This package ships with two types of distributions.
A gaussian distribution and a wrapper for the `Lumen <https://github.com/lumen-org>`__ distributions.
The gaussian distribution has a covariance matrix and therefor can be used to generate cross-plots.

To generate your own distribution let it inherit from the Distribution class and implement all abstract methods.



.. code-block:: python

    from scipy.stats import multivariate_normal
    from contour_visualization.Distribution import Distribution

    class NewDistribution(Distribution):
        def __init__(self, *args, **kwargs):
            super.__init__(*args, **kwargs)
            self.gaussian = multivariate_normal()

        def get_density(self, x):
            return self.gaussian(x)

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
            return x, y, self.weight * self.gau.pdf(pos)