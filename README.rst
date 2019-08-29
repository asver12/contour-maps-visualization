contour-maps-visualisation
==========================

Objective
---------
This library offers a way to visualize multiple 2D-distribution where each distribution has one categorical variable.
It tries in particular to help the viewer to achieve a better understanding of the distributions.
It ships with three types of visualizations:


.. image:: images/contour_img.png
    :scale: 50 %
    :width: 30%

.. image:: images/crosses.png
    :scale: 50 %
    :width: 30%

.. image:: images/pie_chart.png
    :scale: 50 %
    :width: 30%

where each can be used with the others as pleased. There are four tiypes of visualizations in total. Contours, contour-lines, crosses and pie-charts.

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