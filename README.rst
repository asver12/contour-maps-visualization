contour-maps-visualisation
==========================

Objective
---------
Objective of this project is to find a appealing representation of multiple 2D-distribution where a third categorical variable is set. This should happen in a 2D-Cartesian coordinate system where the x- and y-axis represent the variables of the distribution. f(x,y) and the categorical variable is represented through the color. In detail it is your goal to make it optical possible to distinguish two overlapping distributions and be able to tell which one lays on top and which one lays on bottom. This should also be possible for more then two distributions overlapping each other.


Process
-------
In order to achieve this goal we are going to try a bunch of different approaches. These are evaluated by us and then further pursued or rejected. At the moment the approaches are:
* to change the mixing opertor
* to change the color space
* to indivdualize the mixing operator for each pixel

Prerequisites
-------------

Install submodules with:

.. code-block:: console

    git submodule update --init

Make sure you have python 3.5 or higher and pip3 installed.
Than install dependencies with:

.. code-block:: console

    pip3 install -r requirements.txt

`Jupyter Interactive Notebook <https://jupyter.org/>`__ should be installed with requirements. If not install it manually

Quickstart
----------

import the Modules:

.. code-block:: python3

    import example_data, picture_plot

create a list of distributions with:

.. code-block:: python3

    _, _, gaussians, _ = example_data.generate_four_moving_gaussians(size=100)

and get your first visualisation with:

.. code-block:: python3

    picture_plot.plot_images(gaussians, contours=True, contour_lines=True, contour_line_level = 2)

.. figure:: images/example_visualisation.png
    :align: center