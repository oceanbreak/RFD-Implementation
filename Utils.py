"""
Operations with images:
    readImPatch() - reads image and converts it to grayscale float numpy array
    cropImg() - returns cropped image numpy array
    getHoG() - returns array of gadients of input patch over specified number of directions
    calcRectSum() - returns
"""

import numpy as np
from matplotlib import pyplot as plt
from skimage import io
from skimage import img_as_float as imf
from skimage.color import rgb2gray
from skimage.filters import gaussian
from math import atan2, sqrt, pi


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


def getHoG(im_patch, orient_quant=8):
    """
    Calculate gradients in <orient_quant> directions of an input patch
    :param: im_patch - input image patch
    :param: orient_quant - number of directions of gradient, 8 default
    :return: gradient
    """
    # print('Calculating gradient bins ...')
    nDir = orient_quant
    GRAD_RADIUS = 1
    height, width = im_patch.shape
    sigma0 = 1.6
    smooth_patch = gaussian(im_patch, sigma0)

    hog = np.zeros(nDir * width * height)

    for y in range(GRAD_RADIUS, height - 1 - GRAD_RADIUS):
        for x in range(GRAD_RADIUS, width - 1 - GRAD_RADIUS):
            grad2 = smooth_patch[y, x + GRAD_RADIUS]
            grad1 = smooth_patch[y, x - GRAD_RADIUS]
            dx = grad2 - grad1
            grad2 = smooth_patch[y + GRAD_RADIUS, x]
            grad1 = smooth_patch[y - GRAD_RADIUS, x]
            dy = grad2 - grad1

            # Calculate direction and magnitude of gradient
            dir = atan2(dy, dx)             # Direction
            mag = sqrt(dx * dx + dy * dy)   # Magnitude

            # Assign direction to pixel
            idxDir = (dir + pi) * nDir / (2.0 * pi)
            if int(idxDir) == nDir: idxDir -= nDir

            # Soft assignment of direction with weights
            dirIdx = [0, 0]
            dirWeight = [0, 0]
            dirIdx[0] = int(idxDir)
            dirIdx[1] = (dirIdx[0] + 1) % nDir
            dirWeight[0] = 1.0 - (idxDir - dirIdx[0])
            dirWeight[1] = idxDir - dirIdx[0]

            # Calculate current position in one-dimensional array
            pos_idx = y * width + x
            hog[dirIdx[0] * width * height + pos_idx] = dirWeight[0] * mag
            hog[dirIdx[1] * width * height + pos_idx] = dirWeight[1] * mag

    gradMap = []  # Initialize gradients array
    for i in range(nDir + 1):
        gradMap.append(np.zeros((height, width)))

    for y in range(height):
        for x in range(width):
            Total = 0
            for i in range(nDir):
                gradMap[i][y, x] = hog[i * width * height + y * width + x]
                Total += hog[i * width * height + y * width + x]
            gradMap[nDir][y, x] = Total
    return gradMap


def calcRectSum(input_image, top_left_pix, bot_rght_pix, integral=False):
    """
    Calculate sum in rectangle area of given image and borders
    :param input_image as numpy float array
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
        if x1 < 0 & y1 < 0:
            return input_image[y2, x2]
        elif x1 >= 0 & y1 < 0:
            return input_image[y2, x2] - input_image[y2, x1]
        elif x1 < 0 & y1 >= 0:
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
    in_img = readImPatch('data/patches0000.bmp')
    in_int_img = readImPatch('data/patches0000_integral.tif')

