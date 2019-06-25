from matplotlib import pyplot as plt

def plot_images(images, gaussians, *args, **kwargs):
    for img, gaussian in zip(images, gaussians):
        plot_image(img, gaussian, *args, **kwargs)


def plot_image(images, gaussians, title="", with_axis=True):
    """

    :param with_axis: if mathematical axis is shown or not
    :param images: List of List of 2D-images in rgb to plot
    :param gaussians: [gaussian_1, ... , gaussian_n] gaussians from which the image is calculated
    :param title: title of picture
    :return:
    """
    fig, axis = plt.subplots(1, len(images), sharex='col', sharey='row')
    extent = gaussians[0][:4]
    for i, image in enumerate(images):
        axis[i].imshow(image, extent=extent, origin='lower')
        if title == "" and gaussians:
            title = ' '.join("{}".format(gaussians[i][4:-1]))
        axis[i].set_title(title)
    if not with_axis:
        axis.axis("off")
