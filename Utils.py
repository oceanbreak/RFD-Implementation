"""
Common operations with images:
    readImPatch() - reads image and converts it to grayscale float numpy array
    cropImg() - returns cropped image numpy array
    calcRectSum() - returns sum of values in specified rectangle

"""

import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage import img_as_float as imf
from skimage.color import rgb2gray
from skimage.transform import integral


def readImPatch(input_file, show=False):
    img = imf(rgb2gray(io.imread(input_file)))
    if show:
        io.imshow(img)
        plt.title(input_file)
        plt.show()
    return img


def cropImg(input_image, top_left_pix, bot_rght_pix):
    """
    Returns cropped image
    :param input_image: input image as numpy float array
    :param top_left_pix: x and y coordinaes of top left pixel of cropped area
    :param bot_rght_pix: x and y coordinaes of bottom right pixel of cropped area
    """
    return input_image[ top_left_pix[1]:bot_rght_pix[1], top_left_pix[0]:bot_rght_pix[0] ]




if __name__ == '__main__':
    in_img = readImPatch('data/patches0000.bmp')
    in_int_img = readImPatch('data/patches0000_integral.tif')

