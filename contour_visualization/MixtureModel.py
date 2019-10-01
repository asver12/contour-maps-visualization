import itertools

from contour_visualization import Gaussian
from sklearn.mixture import GaussianMixture
import logging

logger = logging.getLogger(__name__)


def generate_gaussian_plots(data, columns, n_components, *args, **kwargs):
    models = []
    xlabels = []
    ylabels = []
    for i, j in itertools.combinations(columns, 2):
        models.append(MixtureModel(data[[i, j]], n_components=n_components).get_gaussians(*args, **kwargs))
        xlabels.append(i)
        ylabels.append(j)
    return models, xlabels, ylabels


class MixtureModel:
    def __init__(self, df, normalize=True, n_components=3, covariance_type="full", max_iter=100, random_state=0,
                 *args,
                 **kwargs):
        if normalize:
            data = (df - df.min()) / (df.max() - df.min())
        else:
            data = df
        self.data = data
        self.n_components = n_components
        self.estimator = GaussianMixture(n_components=n_components, covariance_type=covariance_type, max_iter=max_iter,
                                         random_state=random_state, *args, **kwargs)
        self.estimator.fit(data)

    def learn_new(self, n_components, covariance_type, max_iter=20, random_state=0, *args, **kwargs):
        self.n_components = n_components
        self.estimator = GaussianMixture(n_components=n_components, covariance_type=covariance_type, max_iter=max_iter,
                                         random_state=random_state, *args, **kwargs)
        self.estimator.fit(self.data)

    def get_gaussians(self, *args, **kwargs):
        dataset = []
        for n in range(self.n_components):
            means = self.estimator.means_[n, :2]
            cov_matrix = self.estimator.covariances_[n][:2, :2]
            weight = self.estimator.weights_[n]
            logging.info("Means[{}]: {}".format(n, means))
            logging.info("Cov-Matrix[{}]: {}".format(n, cov_matrix))
            logging.info("weights[{}]: {}".format(n, weight))
            dataset.append(
                Gaussian.Gaussian(weight=weight, means=means, cov_matrix=cov_matrix ** 1 / 2, *args, **kwargs))
        return dataset
