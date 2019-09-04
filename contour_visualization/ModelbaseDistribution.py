import copy
import pickle
import logging

logger = logging.getLogger(__name__)
import numpy as np

try:
    import mb_modelbase
except Exception as e:
    logger.warn(e)

from contour_visualization.Distribution import Distribution


class ModelbaseDistribution(Distribution):
    """
    Integrates a models from `Lumen <https://github.com/lumen-org>`__
    """

    def __init__(self, model, used_categorical_name, categorical_attribute, attributes, *args, **kwargs):
        """
        Integrates a models from `Lumen <https://github.com/lumen-org>`__

        :param model: for lumen created model or path to model
        :param used_categorical_name: categoris which are wanted to use
        :param categorical_attribute: specific name of this distribution inside of the categori
        :param attributes: other attributes which are needed. For the visualizations to work give 2
        """
        super().__init__(*args, **kwargs)
        if isinstance(model, str):
            try:
                # import mb_modelbase
                # Model.load()
                model = pickle.load(open(model, "rb"))
            except Exception as e:
                raise e
        self.categorical_attribute = categorical_attribute
        self.used_categorical_name = used_categorical_name
        self.model = copy.deepcopy(model)
        self.model.marginalize(keep=[categorical_attribute, *attributes])
        self.vec_density = np.vectorize(self.get_density_wrapper, signature="(),()->()")
        if "x_min" not in kwargs.keys():
            min_values = self.model.data[attributes].min()
            max_values = self.model.data[attributes].max()
        else:
            min_values = kwargs["x_min"], kwargs["y_min"]
            max_values = kwargs["x_max"], kwargs["y_max"]
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
        return x, y, self.vec_density(x=x, y=y)

    def get_density_wrapper(self, x, y):
        return self.model.density([self.used_categorical_name, x, y])

    def get_density(self, x):
        return self.model.density([self.used_categorical_name, *x])
