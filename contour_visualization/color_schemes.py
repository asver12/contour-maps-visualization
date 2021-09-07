import random
import math
from typing import List, Any, Union, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from skimage import color

import logging

logger = logging.getLogger(__name__)


def _check_constrains(min_value, max_value):
    if min_value < 0. or min_value > max_value:
        raise Exception("{} is not accepted as minimum value".format(min_value))
    if max_value > 1.:
        raise Exception("{} is not accepted as maximum value".format(max_value))


def create_monochromatic_colorscheme(startcolor, levels, min_value=0, max_value=1, lvl_white=1):
    """
    Generates a monochromatic colorscheme from a startcolor. The values of the startcolor are in rgba
    with r,g,b in [0,1] and a = 1

    :param startcolor: rgba with alpha 255
    :param levels: Array with levels at which the color change
    :param lvl_white: First x colors which are supposed to be white
    :return: colorarray with len(levels) + 1 entrys each color is of the form [r,g,b,a], r,g,b,a in [0,1]
    """
    _check_constrains(min_value, max_value)
    norm_levels = np.linspace(min_value, max_value, len(levels) + 1)
    logger.debug("Min: {} | Max: {}".format(min_value, max_value))
    color_scheme = [[float(startcolor[0]), float(startcolor[1]), float(startcolor[2]), float(i)] for i in
                    norm_levels]
    for i in range(lvl_white + 1 if lvl_white < len(levels) else len(levels) + 1):
        color_scheme[i] = np.array([1., 1., 1., 1.])
    logger.debug(color_scheme)
    return color_scheme


# blue_color_scheme = ["#f7fbff", "#eff3ff", "#deebf7", "#c6dbef", "#bdd7e7", "#9ecae1", "#6baed6", "#4292c6",
#                      "#3182bd", "#2171b5", "#08519c", "#084594", "#08306b"]
# green_color_scheme = ['#f7fcf5', '#edf8e9', '#e5f5e0', '#c7e9c0', '#bae4b3', '#a1d99b', '#74c476', '#41ab5d',
#                       '#31a354', '#238b45', '#006d2c', '#005a32', '#00441b']
# orange_color_scheme = ['#fff5eb', '#feedde', '#fee6ce', '#fdd0a2', '#fdbe85', '#fdae6b', '#fd8d3c', '#f16913',
#                        '#e6550d', '#d94801', '#a63603', '#8c2d04', '#7f2704']
# purple_color_scheme = ['#fcfbfd', '#f2f0f7', '#efedf5', '#dadaeb', '#cbc9e2', '#bcbddc', '#9e9ac8', '#807dba',
#                        '#756bb1', '#6a51a3', '#54278f', '#4a1486', '#3f007d']
# red_color_scheme = ['#fff5f0', '#fee5d9', '#fee0d2', '#fcbba1', '#fcae91', '#fc9272', '#fb6a4a', '#ef3b2c',
#                     '#de2d26', '#cb181d', '#a50f15', '#99000d', '#67000d']
# grey_color_scheme = ['#ffffff', '#f7f7f7', '#f0f0f0', '#d9d9d9', '#cccccc', '#bdbdbd', '#969696', '#737373', '#636363',
#                     '#525252', '#252525', '#000000']
blue_color_scheme = [[0.9686274509803922, 0.984313725490196, 1.0],
                     [0.9372549019607843, 0.9529411764705882, 1.0],
                     [0.8705882352941177, 0.9215686274509803, 0.9686274509803922],
                     [0.7764705882352941, 0.8588235294117647, 0.9372549019607843],
                     [0.7411764705882353, 0.8431372549019608, 0.9058823529411765],
                     [0.6196078431372549, 0.792156862745098, 0.8823529411764706],
                     [0.4196078431372549, 0.6823529411764706, 0.8392156862745098],
                     [0.25882352941176473, 0.5725490196078431, 0.7764705882352941],
                     [0.19215686274509805, 0.5098039215686274, 0.7411764705882353],
                     [0.12941176470588237, 0.44313725490196076, 0.7098039215686275],
                     [0.03137254901960784, 0.3176470588235294, 0.611764705882353],
                     [0.03137254901960784, 0.27058823529411763, 0.5803921568627451],
                     [0.03137254901960784, 0.18823529411764706, 0.4196078431372549]]
green_color_scheme = [[0.9686274509803922, 0.9882352941176471, 0.9607843137254902],
                      [0.9294117647058824, 0.9725490196078431, 0.9137254901960784],
                      [0.8980392156862745, 0.9607843137254902, 0.8784313725490196],
                      [0.7803921568627451, 0.9137254901960784, 0.7529411764705882],
                      [0.7294117647058823, 0.8941176470588236, 0.7019607843137254],
                      [0.6313725490196078, 0.8509803921568627, 0.6078431372549019],
                      [0.4549019607843137, 0.7686274509803922, 0.4627450980392157],
                      [0.2549019607843137, 0.6705882352941176, 0.36470588235294116],
                      [0.19215686274509805, 0.6392156862745098, 0.32941176470588235],
                      [0.13725490196078433, 0.5450980392156862, 0.27058823529411763],
                      [0.0, 0.42745098039215684, 0.17254901960784313],
                      [0.0, 0.35294117647058826, 0.19607843137254902],
                      [0.0, 0.26666666666666666, 0.10588235294117647]]
orange_color_scheme = [[1.0, 0.9607843137254902, 0.9215686274509803],
                       [0.996078431372549, 0.9294117647058824, 0.8705882352941177],
                       [0.996078431372549, 0.9019607843137255, 0.807843137254902],
                       [0.9921568627450981, 0.8156862745098039, 0.6352941176470588],
                       [0.9921568627450981, 0.7450980392156863, 0.5215686274509804],
                       [0.9921568627450981, 0.6823529411764706, 0.4196078431372549],
                       [0.9921568627450981, 0.5529411764705883, 0.23529411764705882],
                       [0.9450980392156862, 0.4117647058823529, 0.07450980392156863],
                       [0.9019607843137255, 0.3333333333333333, 0.050980392156862744],
                       [0.8509803921568627, 0.2823529411764706, 0.00392156862745098],
                       [0.6509803921568628, 0.21176470588235294, 0.011764705882352941],
                       [0.5490196078431373, 0.17647058823529413, 0.01568627450980392],
                       [0.4980392156862745, 0.15294117647058825, 0.01568627450980392]]
purple_color_scheme = [[0.9882352941176471, 0.984313725490196, 0.9921568627450981],
                       [0.9490196078431372, 0.9411764705882353, 0.9686274509803922],
                       [0.9372549019607843, 0.9294117647058824, 0.9607843137254902],
                       [0.8549019607843137, 0.8549019607843137, 0.9215686274509803],
                       [0.796078431372549, 0.788235294117647, 0.8862745098039215],
                       [0.7372549019607844, 0.7411764705882353, 0.8627450980392157],
                       [0.6196078431372549, 0.6039215686274509, 0.7843137254901961],
                       [0.5019607843137255, 0.49019607843137253, 0.7294117647058823],
                       [0.4588235294117647, 0.4196078431372549, 0.6941176470588235],
                       [0.41568627450980394, 0.3176470588235294, 0.6392156862745098],
                       [0.32941176470588235, 0.15294117647058825, 0.5607843137254902],
                       [0.2901960784313726, 0.0784313725490196, 0.5254901960784314],
                       [0.24705882352941178, 0.0, 0.49019607843137253]]
red_color_scheme = [[1.0, 0.9607843137254902, 0.9411764705882353],
                    [0.996078431372549, 0.8980392156862745, 0.8509803921568627],
                    [0.996078431372549, 0.8784313725490196, 0.8235294117647058],
                    [0.9882352941176471, 0.7333333333333333, 0.6313725490196078],
                    [0.9882352941176471, 0.6823529411764706, 0.5686274509803921],
                    [0.9882352941176471, 0.5725490196078431, 0.4470588235294118],
                    [0.984313725490196, 0.41568627450980394, 0.2901960784313726],
                    [0.9372549019607843, 0.23137254901960785, 0.17254901960784313],
                    [0.8705882352941177, 0.17647058823529413, 0.14901960784313725],
                    [0.796078431372549, 0.09411764705882353, 0.11372549019607843],
                    [0.6470588235294118, 0.058823529411764705, 0.08235294117647059],
                    [0.6, 0.0, 0.050980392156862744],
                    [0.403921568627451, 0.0, 0.050980392156862744]]
grey_color_scheme = [[1.0, 1.0, 1.0],
                     [0.9686274509803922, 0.9686274509803922, 0.9686274509803922],
                     [0.9411764705882353, 0.9411764705882353, 0.9411764705882353],
                     [0.8509803921568627, 0.8509803921568627, 0.8509803921568627],
                     [0.8, 0.8, 0.8],
                     [0.7411764705882353, 0.7411764705882353, 0.7411764705882353],
                     [0.5882352941176471, 0.5882352941176471, 0.5882352941176471],
                     [0.45098039215686275, 0.45098039215686275, 0.45098039215686275],
                     [0.38823529411764707, 0.38823529411764707, 0.38823529411764707],
                     [0.3215686274509804, 0.3215686274509804, 0.3215686274509804],
                     [0.1450980392156863, 0.1450980392156863, 0.1450980392156863],
                     [0.0, 0.0, 0.0]]

viridis_color_scheme = [(0.26666666666666666, 0.00392156862745098, 0.32941176470588235),
                        (0.26666666666666666, 0.00784313725490196, 0.33725490196078434),
                        (0.27058823529411763, 0.01568627450980392, 0.3411764705882353),
                        (0.27058823529411763, 0.0196078431372549, 0.34901960784313724),
                        (0.27450980392156865, 0.027450980392156862, 0.35294117647058826),
                        (0.27450980392156865, 0.03137254901960784, 0.3607843137254902),
                        (0.27450980392156865, 0.0392156862745098, 0.36470588235294116),
                        (0.27450980392156865, 0.043137254901960784, 0.3686274509803922),
                        (0.2784313725490196, 0.050980392156862744, 0.3764705882352941),
                        (0.2784313725490196, 0.054901960784313725, 0.3803921568627451),
                        (0.2784313725490196, 0.06274509803921569, 0.38823529411764707),
                        (0.2784313725490196, 0.06666666666666667, 0.39215686274509803),
                        (0.2784313725490196, 0.07450980392156863, 0.396078431372549),
                        (0.2823529411764706, 0.0784313725490196, 0.403921568627451),
                        (0.2823529411764706, 0.08627450980392157, 0.40784313725490196),
                        (0.2823529411764706, 0.09019607843137255, 0.4117647058823529),
                        (0.2823529411764706, 0.09411764705882353, 0.41568627450980394),
                        (0.2823529411764706, 0.10196078431372549, 0.4235294117647059),
                        (0.2823529411764706, 0.10588235294117647, 0.42745098039215684),
                        (0.2823529411764706, 0.10980392156862745, 0.43137254901960786),
                        (0.2823529411764706, 0.11372549019607843, 0.43529411764705883),
                        (0.2823529411764706, 0.12156862745098039, 0.4392156862745098),
                        (0.2823529411764706, 0.12549019607843137, 0.44313725490196076),
                        (0.2823529411764706, 0.12941176470588237, 0.45098039215686275),
                        (0.2823529411764706, 0.13725490196078433, 0.4549019607843137),
                        (0.2823529411764706, 0.1411764705882353, 0.4588235294117647),
                        (0.2823529411764706, 0.1450980392156863, 0.4627450980392157),
                        (0.2823529411764706, 0.14901960784313725, 0.4666666666666667),
                        (0.2823529411764706, 0.1568627450980392, 0.47058823529411764),
                        (0.2823529411764706, 0.1607843137254902, 0.4745098039215686),
                        (0.2784313725490196, 0.16470588235294117, 0.47843137254901963),
                        (0.2784313725490196, 0.17254901960784313, 0.47843137254901963),
                        (0.2784313725490196, 0.17647058823529413, 0.4823529411764706),
                        (0.2784313725490196, 0.1803921568627451, 0.48627450980392156),
                        (0.2784313725490196, 0.1843137254901961, 0.49019607843137253),
                        (0.27450980392156865, 0.18823529411764706, 0.49411764705882355),
                        (0.27450980392156865, 0.19607843137254902, 0.49411764705882355),
                        (0.27450980392156865, 0.2, 0.4980392156862745),
                        (0.27450980392156865, 0.20392156862745098, 0.5019607843137255),
                        (0.27058823529411763, 0.20784313725490197, 0.5058823529411764),
                        (0.27058823529411763, 0.21568627450980393, 0.5058823529411764),
                        (0.27058823529411763, 0.2196078431372549, 0.5098039215686274),
                        (0.26666666666666666, 0.2235294117647059, 0.5137254901960784),
                        (0.26666666666666666, 0.22745098039215686, 0.5137254901960784),
                        (0.26666666666666666, 0.23137254901960785, 0.5176470588235295),
                        (0.2627450980392157, 0.23921568627450981, 0.5176470588235295),
                        (0.2627450980392157, 0.24313725490196078, 0.5215686274509804),
                        (0.25882352941176473, 0.24705882352941178, 0.5215686274509804),
                        (0.25882352941176473, 0.25098039215686274, 0.5254901960784314),
                        (0.25882352941176473, 0.2549019607843137, 0.5254901960784314),
                        (0.2549019607843137, 0.25882352941176473, 0.5294117647058824),
                        (0.2549019607843137, 0.26666666666666666, 0.5294117647058824),
                        (0.25098039215686274, 0.27058823529411763, 0.5333333333333333),
                        (0.25098039215686274, 0.27450980392156865, 0.5333333333333333),
                        (0.24705882352941178, 0.2784313725490196, 0.5333333333333333),
                        (0.24705882352941178, 0.2823529411764706, 0.5372549019607843),
                        (0.24313725490196078, 0.28627450980392155, 0.5372549019607843),
                        (0.24313725490196078, 0.2901960784313726, 0.5372549019607843),
                        (0.24313725490196078, 0.2980392156862745, 0.5411764705882353),
                        (0.23921568627450981, 0.30196078431372547, 0.5411764705882353),
                        (0.23921568627450981, 0.3058823529411765, 0.5411764705882353),
                        (0.23529411764705882, 0.30980392156862746, 0.5411764705882353),
                        (0.23529411764705882, 0.3137254901960784, 0.5450980392156862),
                        (0.23137254901960785, 0.3176470588235294, 0.5450980392156862),
                        (0.23137254901960785, 0.3215686274509804, 0.5450980392156862),
                        (0.22745098039215686, 0.3254901960784314, 0.5450980392156862),
                        (0.22745098039215686, 0.32941176470588235, 0.5490196078431373),
                        (0.2235294117647059, 0.3333333333333333, 0.5490196078431373),
                        (0.2235294117647059, 0.33725490196078434, 0.5490196078431373),
                        (0.2196078431372549, 0.34509803921568627, 0.5490196078431373),
                        (0.2196078431372549, 0.34901960784313724, 0.5490196078431373),
                        (0.21568627450980393, 0.35294117647058826, 0.5490196078431373),
                        (0.21568627450980393, 0.3568627450980392, 0.5529411764705883),
                        (0.21176470588235294, 0.3607843137254902, 0.5529411764705883),
                        (0.21176470588235294, 0.36470588235294116, 0.5529411764705883),
                        (0.20784313725490197, 0.3686274509803922, 0.5529411764705883),
                        (0.20784313725490197, 0.37254901960784315, 0.5529411764705883),
                        (0.20392156862745098, 0.3764705882352941, 0.5529411764705883),
                        (0.20392156862745098, 0.3803921568627451, 0.5529411764705883),
                        (0.2, 0.3843137254901961, 0.5529411764705883), (0.2, 0.38823529411764707, 0.5529411764705883),
                        (0.19607843137254902, 0.39215686274509803, 0.5568627450980392),
                        (0.19607843137254902, 0.396078431372549, 0.5568627450980392),
                        (0.19215686274509805, 0.4, 0.5568627450980392),
                        (0.19215686274509805, 0.403921568627451, 0.5568627450980392),
                        (0.19215686274509805, 0.40784313725490196, 0.5568627450980392),
                        (0.18823529411764706, 0.4117647058823529, 0.5568627450980392),
                        (0.18823529411764706, 0.41568627450980394, 0.5568627450980392),
                        (0.1843137254901961, 0.4196078431372549, 0.5568627450980392),
                        (0.1843137254901961, 0.4235294117647059, 0.5568627450980392),
                        (0.1803921568627451, 0.42745098039215684, 0.5568627450980392),
                        (0.1803921568627451, 0.43137254901960786, 0.5568627450980392),
                        (0.1803921568627451, 0.43529411764705883, 0.5568627450980392),
                        (0.17647058823529413, 0.4392156862745098, 0.5568627450980392),
                        (0.17647058823529413, 0.44313725490196076, 0.5568627450980392),
                        (0.17254901960784313, 0.44313725490196076, 0.5568627450980392),
                        (0.17254901960784313, 0.4470588235294118, 0.5568627450980392),
                        (0.17254901960784313, 0.45098039215686275, 0.5568627450980392),
                        (0.16862745098039217, 0.4549019607843137, 0.5568627450980392),
                        (0.16862745098039217, 0.4588235294117647, 0.5568627450980392),
                        (0.16470588235294117, 0.4627450980392157, 0.5568627450980392),
                        (0.16470588235294117, 0.4666666666666667, 0.5568627450980392),
                        (0.16470588235294117, 0.47058823529411764, 0.5568627450980392),
                        (0.1607843137254902, 0.4745098039215686, 0.5568627450980392),
                        (0.1607843137254902, 0.47843137254901963, 0.5568627450980392),
                        (0.1607843137254902, 0.4823529411764706, 0.5568627450980392),
                        (0.1568627450980392, 0.48627450980392156, 0.5568627450980392),
                        (0.1568627450980392, 0.49019607843137253, 0.5568627450980392),
                        (0.15294117647058825, 0.49411764705882355, 0.5568627450980392),
                        (0.15294117647058825, 0.4980392156862745, 0.5568627450980392),
                        (0.15294117647058825, 0.5019607843137255, 0.5568627450980392),
                        (0.14901960784313725, 0.5058823529411764, 0.5568627450980392),
                        (0.14901960784313725, 0.5098039215686274, 0.5568627450980392),
                        (0.14901960784313725, 0.5098039215686274, 0.5568627450980392),
                        (0.1450980392156863, 0.5137254901960784, 0.5568627450980392),
                        (0.1450980392156863, 0.5176470588235295, 0.5568627450980392),
                        (0.1450980392156863, 0.5215686274509804, 0.5568627450980392),
                        (0.1411764705882353, 0.5254901960784314, 0.5568627450980392),
                        (0.1411764705882353, 0.5294117647058824, 0.5568627450980392),
                        (0.13725490196078433, 0.5333333333333333, 0.5568627450980392),
                        (0.13725490196078433, 0.5372549019607843, 0.5568627450980392),
                        (0.13725490196078433, 0.5411764705882353, 0.5529411764705883),
                        (0.13333333333333333, 0.5450980392156862, 0.5529411764705883),
                        (0.13333333333333333, 0.5490196078431373, 0.5529411764705883),
                        (0.13333333333333333, 0.5529411764705883, 0.5529411764705883),
                        (0.12941176470588237, 0.5568627450980392, 0.5529411764705883),
                        (0.12941176470588237, 0.5607843137254902, 0.5529411764705883),
                        (0.12941176470588237, 0.5647058823529412, 0.5529411764705883),
                        (0.12941176470588237, 0.5686274509803921, 0.5490196078431373),
                        (0.12549019607843137, 0.5725490196078431, 0.5490196078431373),
                        (0.12549019607843137, 0.5725490196078431, 0.5490196078431373),
                        (0.12549019607843137, 0.5764705882352941, 0.5490196078431373),
                        (0.12156862745098039, 0.5803921568627451, 0.5490196078431373),
                        (0.12156862745098039, 0.5843137254901961, 0.5450980392156862),
                        (0.12156862745098039, 0.5882352941176471, 0.5450980392156862),
                        (0.12156862745098039, 0.592156862745098, 0.5450980392156862),
                        (0.12156862745098039, 0.596078431372549, 0.5450980392156862),
                        (0.12156862745098039, 0.6, 0.5411764705882353),
                        (0.12156862745098039, 0.6039215686274509, 0.5411764705882353),
                        (0.11764705882352941, 0.6078431372549019, 0.5411764705882353),
                        (0.11764705882352941, 0.611764705882353, 0.5372549019607843),
                        (0.11764705882352941, 0.615686274509804, 0.5372549019607843),
                        (0.12156862745098039, 0.6196078431372549, 0.5372549019607843),
                        (0.12156862745098039, 0.6235294117647059, 0.5333333333333333),
                        (0.12156862745098039, 0.6274509803921569, 0.5333333333333333),
                        (0.12156862745098039, 0.6313725490196078, 0.5333333333333333),
                        (0.12156862745098039, 0.6313725490196078, 0.5294117647058824),
                        (0.12156862745098039, 0.6352941176470588, 0.5294117647058824),
                        (0.12549019607843137, 0.6392156862745098, 0.5254901960784314),
                        (0.12549019607843137, 0.6431372549019608, 0.5254901960784314),
                        (0.12941176470588237, 0.6470588235294118, 0.5215686274509804),
                        (0.12941176470588237, 0.6509803921568628, 0.5215686274509804),
                        (0.13333333333333333, 0.6549019607843137, 0.5215686274509804),
                        (0.13333333333333333, 0.6588235294117647, 0.5176470588235295),
                        (0.13725490196078433, 0.6627450980392157, 0.5137254901960784),
                        (0.1411764705882353, 0.6666666666666666, 0.5137254901960784),
                        (0.1450980392156863, 0.6705882352941176, 0.5098039215686274),
                        (0.1450980392156863, 0.6745098039215687, 0.5098039215686274),
                        (0.14901960784313725, 0.6784313725490196, 0.5058823529411764),
                        (0.15294117647058825, 0.6784313725490196, 0.5058823529411764),
                        (0.1568627450980392, 0.6823529411764706, 0.5019607843137255),
                        (0.1607843137254902, 0.6862745098039216, 0.4980392156862745),
                        (0.16470588235294117, 0.6901960784313725, 0.4980392156862745),
                        (0.17254901960784313, 0.6941176470588235, 0.49411764705882355),
                        (0.17647058823529413, 0.6980392156862745, 0.49019607843137253),
                        (0.1803921568627451, 0.7019607843137254, 0.48627450980392156),
                        (0.1843137254901961, 0.7058823529411765, 0.48627450980392156),
                        (0.19215686274509805, 0.7098039215686275, 0.4823529411764706),
                        (0.19607843137254902, 0.7137254901960784, 0.47843137254901963),
                        (0.20392156862745098, 0.7137254901960784, 0.4745098039215686),
                        (0.20784313725490197, 0.7176470588235294, 0.4745098039215686),
                        (0.21568627450980393, 0.7215686274509804, 0.47058823529411764),
                        (0.2196078431372549, 0.7254901960784313, 0.4666666666666667),
                        (0.22745098039215686, 0.7294117647058823, 0.4627450980392157),
                        (0.23137254901960785, 0.7333333333333333, 0.4588235294117647),
                        (0.23921568627450981, 0.7372549019607844, 0.4549019607843137),
                        (0.24705882352941178, 0.7372549019607844, 0.45098039215686275),
                        (0.25098039215686274, 0.7411764705882353, 0.4470588235294118),
                        (0.25882352941176473, 0.7450980392156863, 0.44313725490196076),
                        (0.26666666666666666, 0.7490196078431373, 0.4392156862745098),
                        (0.27450980392156865, 0.7529411764705882, 0.43529411764705883),
                        (0.2823529411764706, 0.7568627450980392, 0.43137254901960786),
                        (0.2901960784313726, 0.7568627450980392, 0.42745098039215684),
                        (0.2980392156862745, 0.7607843137254902, 0.4235294117647059),
                        (0.3058823529411765, 0.7647058823529411, 0.4196078431372549),
                        (0.3137254901960784, 0.7686274509803922, 0.41568627450980394),
                        (0.3215686274509804, 0.7725490196078432, 0.4117647058823529),
                        (0.32941176470588235, 0.7725490196078432, 0.40784313725490196),
                        (0.33725490196078434, 0.7764705882352941, 0.403921568627451),
                        (0.34509803921568627, 0.7803921568627451, 0.396078431372549),
                        (0.35294117647058826, 0.7843137254901961, 0.39215686274509803),
                        (0.3607843137254902, 0.7843137254901961, 0.38823529411764707),
                        (0.3686274509803922, 0.788235294117647, 0.3843137254901961),
                        (0.3764705882352941, 0.792156862745098, 0.3764705882352941),
                        (0.38823529411764707, 0.796078431372549, 0.37254901960784315),
                        (0.396078431372549, 0.796078431372549, 0.3686274509803922),
                        (0.403921568627451, 0.8, 0.3607843137254902),
                        (0.4117647058823529, 0.803921568627451, 0.3568627450980392),
                        (0.4235294117647059, 0.803921568627451, 0.35294117647058826),
                        (0.43137254901960786, 0.807843137254902, 0.34509803921568627),
                        (0.4392156862745098, 0.8117647058823529, 0.3411764705882353),
                        (0.45098039215686275, 0.8156862745098039, 0.33725490196078434),
                        (0.4588235294117647, 0.8156862745098039, 0.32941176470588235),
                        (0.4666666666666667, 0.8196078431372549, 0.3254901960784314),
                        (0.47843137254901963, 0.8196078431372549, 0.3176470588235294),
                        (0.48627450980392156, 0.8235294117647058, 0.3137254901960784),
                        (0.4980392156862745, 0.8274509803921568, 0.3058823529411765),
                        (0.5058823529411764, 0.8274509803921568, 0.30196078431372547),
                        (0.5176470588235295, 0.8313725490196079, 0.29411764705882354),
                        (0.5254901960784314, 0.8352941176470589, 0.28627450980392155),
                        (0.5372549019607843, 0.8352941176470589, 0.2823529411764706),
                        (0.5450980392156862, 0.8392156862745098, 0.27450980392156865),
                        (0.5568627450980392, 0.8392156862745098, 0.27058823529411763),
                        (0.5647058823529412, 0.8431372549019608, 0.2627450980392157),
                        (0.5764705882352941, 0.8431372549019608, 0.2549019607843137),
                        (0.5843137254901961, 0.8470588235294118, 0.25098039215686274),
                        (0.596078431372549, 0.8470588235294118, 0.24313725490196078),
                        (0.6078431372549019, 0.8509803921568627, 0.23529411764705882),
                        (0.615686274509804, 0.8509803921568627, 0.23137254901960785),
                        (0.6274509803921569, 0.8549019607843137, 0.2235294117647059),
                        (0.6352941176470588, 0.8549019607843137, 0.21568627450980393),
                        (0.6470588235294118, 0.8588235294117647, 0.21176470588235294),
                        (0.6588235294117647, 0.8588235294117647, 0.20392156862745098),
                        (0.6666666666666666, 0.8627450980392157, 0.19607843137254902),
                        (0.6784313725490196, 0.8627450980392157, 0.18823529411764706),
                        (0.6901960784313725, 0.8666666666666667, 0.1843137254901961),
                        (0.6980392156862745, 0.8666666666666667, 0.17647058823529413),
                        (0.7098039215686275, 0.8705882352941177, 0.16862745098039217),
                        (0.7215686274509804, 0.8705882352941177, 0.1607843137254902),
                        (0.7294117647058823, 0.8705882352941177, 0.1568627450980392),
                        (0.7411764705882353, 0.8745098039215686, 0.14901960784313725),
                        (0.7529411764705882, 0.8745098039215686, 0.1450980392156863),
                        (0.7607843137254902, 0.8745098039215686, 0.13725490196078433),
                        (0.7725490196078432, 0.8784313725490196, 0.12941176470588237),
                        (0.7843137254901961, 0.8784313725490196, 0.12549019607843137),
                        (0.792156862745098, 0.8823529411764706, 0.12156862745098039),
                        (0.803921568627451, 0.8823529411764706, 0.11372549019607843),
                        (0.8156862745098039, 0.8823529411764706, 0.10980392156862745),
                        (0.8235294117647058, 0.8862745098039215, 0.10588235294117647),
                        (0.8352941176470589, 0.8862745098039215, 0.10196078431372549),
                        (0.8470588235294118, 0.8862745098039215, 0.09803921568627451),
                        (0.8549019607843137, 0.8901960784313725, 0.09803921568627451),
                        (0.8666666666666667, 0.8901960784313725, 0.09411764705882353),
                        (0.8745098039215686, 0.8901960784313725, 0.09411764705882353),
                        (0.8862745098039215, 0.8941176470588236, 0.09411764705882353),
                        (0.8980392156862745, 0.8941176470588236, 0.09803921568627451),
                        (0.9058823529411765, 0.8941176470588236, 0.09803921568627451),
                        (0.9176470588235294, 0.8980392156862745, 0.10196078431372549),
                        (0.9254901960784314, 0.8980392156862745, 0.10588235294117647),
                        (0.9372549019607843, 0.8980392156862745, 0.10980392156862745),
                        (0.9450980392156862, 0.8980392156862745, 0.11372549019607843),
                        (0.9568627450980393, 0.9019607843137255, 0.11764705882352941),
                        (0.9647058823529412, 0.9019607843137255, 0.12549019607843137),
                        (0.9725490196078431, 0.9019607843137255, 0.12941176470588237),
                        (0.984313725490196, 0.9058823529411765, 0.13725490196078433),
                        (0.9921568627450981, 0.9058823529411765, 0.1450980392156863)]

_colorschemes = {"blue": blue_color_scheme, "orange": orange_color_scheme, "green": green_color_scheme,
                 "purple": purple_color_scheme, "red": red_color_scheme, "viridis": viridis_color_scheme, "grey": grey_color_scheme}
brewer_colorscheme_names = list(_colorschemes.keys())[:-1]

_colorscheme = "colorscheme"
_colorscheme_name = "colorscheme_name"


def get_colorbrewer_schemes():
    """
    returns all available colorpaletts from brewer
    :return: [{colorscheme: colorscheme_1, "colorscheme_name": colorscheme_name_1}, ... , {colorscheme: colorscheme_n, colorscheme_name: colorscheme_name_n}]
    """
    return [{_colorscheme: create_color_brewer_colorscheme, _colorscheme_name: colorscheme_name} for colorscheme_name
            in
            brewer_colorscheme_names]


def get_viridis_color_scheme():
    """

    :return:
    """
    return {_colorscheme: create_color_brewer_colorscheme, _colorscheme_name: "viridis"}


def get_background_colorbrewer_scheme():
    """
    returns a grey colorpalett from brewer
    :return: {colorscheme: create_color_brewer_colorscheme, colorscheme_name: grey}
    """
    return {_colorscheme: create_color_brewer_colorscheme, _colorscheme_name: "grey"}


def get_white():
    return np.array([1., 1., 1., 1.])


def get_main_color(colorscheme):
    """
    returns all colors in a brewer color-scheme
    :param colorscheme: {colorscheme:colorscheme , colorscheme_name:colorscheme_name}brewer-colorscheme to output
    :return: [rgb_1, ... rgb_12]
    """
    return _colorschemes[colorscheme[_colorscheme_name]]


def get_representiv_colors(colorschemes, position=-4):
    return [get_main_color(i)[position] for i in colorschemes]


def _interpolate(color_array, start, end, position):
    """
    find the exact position of a point in between two colors.

    :param color_array: colors in which the rang is
    :param start: index of the first color to use
    :param end: index of the second color to use
    :param position: position at which the new color is supposed to be
    :return: values for the new colorindex
    """
    percentage = (position - start) / (end - start)
    return (1. - percentage) * np.array(color_array[start]) + percentage * np.array(color_array[end])


def create_color_brewer_colorscheme(colorscheme_name, levels, lvl_white=1):
    """
    creates a colorscheme based on the sequential colorschemes from http://colorbrewer2.org.

    :param colorscheme_name: name of the colorscheme from http://colorbrewer2.org. blue, orange, green, purple, red, grey
    :param levels: points at which the colors are returned inbetween [0, 1] at exactly these points the colors are selected
    :param min_value: minimal value to work with. This is the lower bound of the colorscheme [0,1]
    :param max_value: maximal value to work with. This is the upper bound of the colorscheme [0,1]
    :param lvl_white: number of colors in list which are white, beginning from the first element
    :param verbose: outputs additional information
    :return:
    """
    if min(levels) < 0 or (len(levels) != 1 and min(levels) == max(levels)):
        raise ValueError(
            "Minimum of levels is suppost to be in the intervall [0 <= x < Maximum({})]. Found {}".format(min(levels),
                                                                                                          max(levels)))
    if max(levels) > 1.:
        raise ValueError(
            "Maximum of levels is suppost to be in the intervall [Minimum({}) < x <= 1]. Found {}".format(
                min(levels), max(levels)))
    _colorscheme = _colorschemes.get(colorscheme_name, blue_color_scheme)
    colors = []
    num_of_colors = len(_colorscheme) - 1
    for i in levels:
        # interpolation between the colors in the colorscheme
        start = math.floor(i * num_of_colors)
        end = math.ceil(i * num_of_colors)
        logger.debug("Indexes: {} - {} |Value: {}, Number of colors: {}".format(start, end, i, num_of_colors))
        if start == end:
            colors.append(np.append(_colorscheme[start], 1.))
        else:
            colors.append(np.append(_interpolate(_colorscheme, start, end, i * num_of_colors), 1.0))
    for i in range(lvl_white if lvl_white < len(levels) else len(levels)):
        colors[i] = get_white()
    logger.debug("{}[{}]".format(colors, len(colors)))
    return colors


def create_hsl_colorscheme(startcolor, levels, min_value=0, max_value=1, lvl_white=1):
    """

    :param startcolor:
    :param levels:
    :param min_value:
    :param max_value:
    :param lvl_white:
    :param verbose:
    :return:
    """
    _check_constrains(min_value, max_value)
    rgb = __convert_startcolor(startcolor)
    norm_levels = np.linspace(min_value, max_value, len(levels))
    logger.debug("Min: {} | Max: {}".format(min_value, max_value))
    hsv = color.rgb2hsv(rgb)
    if all([i == 0. for i in startcolor]):
        color_scheme = [color.hsv2rgb([[[float(hsv[0][0][0]), float(hsv[0][0][1]), float(1 - i)]]]) for i in
                        norm_levels]
    else:
        color_scheme = [color.hsv2rgb([[[float(hsv[0][0][0]), float(i), float(hsv[0][0][2])]]]) for i in norm_levels]
    color_scheme = [np.array([i[0][0][0], i[0][0][1], i[0][0][2], 1.]) for i in color_scheme]
    for i in range(lvl_white + 1 if lvl_white < len(levels) else len(levels)):
        color_scheme[i] = np.array([1., 1., 1., 1.])
    logger.debug(color_scheme)
    return color_scheme


def __convert_startcolor(startcolor):
    if len(startcolor) == 4:
        rgb = color.rgba2rgb([[startcolor]])
    elif len(startcolor) == 3:
        rgb = [[startcolor]]
    else:
        raise ValueError("Expected RGB or RGBa value")
    return rgb


def matplotlib_colorschemes(colorscheme_name, levels, lvl_white=1):
    """
    Generates a colorscheme from matplotlib

    :param colorscheme_name: Colorscheme as String
    :param levels: Array with levels at which the color change
    :return: colorarray with len(levels) + 1 entrys each color is of the form [r,g,b,a], r,g,b,a in [0,1]
    """
    logger.debug("colorscheme: {} | levels: {} |level white: {}".format(colorscheme_name, levels, lvl_white))
    color_scheme = [i for i in
                    plt.cm.get_cmap(colorscheme_name)(np.linspace(0, 1, len(levels)))]
    for i in range(lvl_white + 1):
        color_scheme[i] = np.array([1., 1., 1., 1.])
    logger.debug(color_scheme)
    return color_scheme


def random_matplotlib_colorschemes():
    return list(cm.cmap_d.keys())[random.randrange(len(cm.cmap_d.keys()))]
