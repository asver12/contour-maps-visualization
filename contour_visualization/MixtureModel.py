from contour_visualization import Gaussian, helper
from sklearn.mixture import GaussianMixture

import logging

logger = logging.getLogger(__name__)


class MixtureModel:
    def __init__(self, data, n_components=3, covariance_type="full", max_iter=20, random_state=0, *args, **kwargs):
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

    def get_gaussians(self):
        datasets = []
        for n in range(self.n_components):
            means = self.estimator.means_[n, :2]
            cov_matrix = self.estimator.covariances_[n][:2, :2]
            weight = self.estimator.weights_[n]
            logging.info("Means[{}]: {}".format(n, means))
            logging.info("Cov-Matrix[{}]: {}".format(n, cov_matrix))
            logging.info("weights[{}]: {}".format(n, weight))
            datasets.append(
                Gaussian.Gaussian(weight=weight, means=means, cov_matrix=cov_matrix))
        x_min, x_max = helper.get_x_values(datasets)
        y_min, y_max = helper.get_y_values(datasets)
        return [
            Gaussian.Gaussian(means=gau.means, cov_matrix=gau.cov_matrix, weight=gau.weight, x_min=x_min, x_max=x_max,
                              y_min=y_min, y_max=y_max) for gau in datasets]
