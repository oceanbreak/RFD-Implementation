"""
Common operations with images:
    readImPatch() - reads image and converts it to grayscale float numpy array
    cropImg() - returns cropped image numpy array

"""

import numpy as np
from skimage import io
from skimage import img_as_float as imf
from skimage.color import rgb2gray
from skimage.transform import integral


def readImPatch(input_file):
    return imf(rgb2gray(io.imread(input_file)))


def cropImg(input_image, top_left_pix, bot_rght_pix):
    """
    :param input_image: input image array
    :param top_left_pix: x and y coordinaes of top left pixel of cropped area
    :param bot_rght_pix: x and y coordinaes of bottom right pixel of cropped area
    """
    return input_image[ top_left_pix[1]:bot_rght_pix[1], top_left_pix[0]:bot_rght_pix[0] ]

