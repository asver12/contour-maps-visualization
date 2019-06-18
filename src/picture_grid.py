import math
from matplotlib import pyplot as plt

from src import picture_worker, color_operations


def plot_image(gaussians, colorschemes, levels=8,
                title="", columns=5,
                bottom=0.0,
                left=0., right=2.,
                top=2.):
    images, _, _ = picture_worker.get_image_list(gaussians, colorschemes)

    print("{}".format(["mu_x", "variance_x", "mu_y", "variance_y"]))
    colors = color_operations.get_colorcodes(colorschemes)[:len(images)]

    if len(images) == 1:
        picture_worker.plot_image(plt, images[0], gaussians[0], [], colors, False, False,
                   "", levels)
        plt.subplots_adjust(bottom=bottom, left=left, right=right, top=top)
    else:
        for i in range(math.ceil(len(images) / columns)):
            subplot = images[i * columns:(i + 1) * columns]
            fig, axes = plt.subplots(1, len(subplot), sharex='col', sharey='row')
            if len(subplot) == 1:
                plot_image(axes, subplot[0], gaussians[i * columns], "", colors,
                           False,
                           False,
                           "",
                           levels)
            else:
                for j in range(len(subplot)):
                    picture_worker.plot_image(axes[j], subplot[j], [gaussians[j + i * columns]], "", [colors[j]],
                               False,
                               False,
                               "",
                               levels)
            fig.subplots_adjust(bottom=bottom, left=left, right=right, top=top)