import itertools
import numpy as np
import matplotlib.pyplot as plt
import copy
from src import picture_plot
from src.ModelbaseDistribution import ModelbaseDistribution

import logging

logger = logging.getLogger(__name__)


def create_distributions(model, categorical_var, x, y, x_min, x_max, y_min, y_max, model_type=ModelbaseDistribution,
                         size=10):
    models = []
    for i in np.unique(model.data[categorical_var]):
        next_model = model_type(copy.deepcopy(model), i, categorical_var, [x, y], x_min=x_min, x_max=x_max, y_min=y_min,
                                y_max=y_max, size=size)
        logger.info(next_model)
        models.append(next_model)
    return models


def scatterplot_matrix(model, categorical_var, names, size=20, *args, **kwargs):
    """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid."""
    numvars = len(names)
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    # Plot the data.
    xy_max = max(model.data[names].max())
    xy_min = min(model.data[names].min())
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i, j)]:
            logger.info("x: {} , y: {}".format(x, y))
            dataset = create_distributions(model, categorical_var, names[x], names[y], xy_min, xy_max, xy_min, xy_max,
                                           size=size)
            picture_plot.plot_image(axes[x, y], dataset, *args, **kwargs)

    # Label the diagonal subplots...
    for i, label in enumerate(names):
        axes[i, i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                            ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j, i].xaxis.set_visible(True)
        axes[i, j].yaxis.set_visible(True)

    return fig
