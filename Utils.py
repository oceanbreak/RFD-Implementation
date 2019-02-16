"""
Common operations with images:
    readImPatch() - reads image and converts it to grayscale float numpy array
    cropImg() - returns cropped image numpy array

"""

import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage import img_as_float as imf
from skimage.color import rgb2gray
from skimage.transform import integral


def readImPatch(input_file, show=False):
    img = imf(rgb2gray(io.imread(input_file)))
    io.imshow(img)
    plt.title(input_file)
    plt.show()
    return img


def cropImg(input_image, top_left_pix, bot_rght_pix):
    """
    :param input_image: input image array
    :param top_left_pix: x and y coordinaes of top left pixel of cropped area
    :param bot_rght_pix: x and y coordinaes of bottom right pixel of cropped area
    """
    return input_image[ top_left_pix[1]:bot_rght_pix[1], top_left_pix[0]:bot_rght_pix[0] ]


def calcRectSum(input_image, top_left_pix, bot_rght_pix, integral = False):
    """
    Calculate sum in rectangle area of given image and borders
    :param input_image
    :param top_left_pix (x,y)
    :param bot_rght_pix (x,y)
    :param integral: Set True if input is an integral image
    :return: sum value
    """
    if integral:
        x1 = top_left_pix[0] - 1
        x2 = bot_rght_pix[0]
        y1 = top_left_pix[1] - 1
        y2 = bot_rght_pix[1]
        if x1<0 and y1<0:
            return input_image[y2, x2]
        elif x1>=0 and y1<0:
            return input_image[y2, x2] - input_image[y2, x1]
        elif x1<0 and y1>=0:
            return input_image[y2, x2] - input_image[y1, x2]
        else:
            return input_image[y2, x2] - input_image[y1, x2] - input_image[y2, x1] + input_image[y1, x1]

    else:
        sum = 0
        for i in range(top_left_pix[1], bot_rght_pix[1]+1):
            for j in range(top_left_pix[0], bot_rght_pix[0]+1):
                sum += input_image[i, j]
        return sum

if __name__ == '__main__':
    in_img = readImPatch('data/patches0000.bmp', True)
    in_int_img = readImPatch('data/patches0000_integral.tif', True)

    int_1 = calcRectSum(in_img, (0,200), (640,840))
    int_2  = calcRectSum(in_int_img, (0,200), (640,840), True)


    print("Integral via sum: %f" % int_1)
    print("Integral via integral: %f" % int_2)