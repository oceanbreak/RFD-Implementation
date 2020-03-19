"""
<<<<<<< Updated upstream
This module provides searching for optimum threshold value
for rectangular areas of RFD
"""

import numpy as np
from RectangleCalculation import candidateRects
from sys import stdout
import struct

# Initializing dataset array, fpg, and rectangle list
DATASET_INFO = 'D:/Coding/IPPI/Datasets/Patch/Patch_Dataset_Info_FPG.npy'
DATASET_PIXEL_PATH = 'D:/Coding/IPPI/Datasets/Patch/dataset_pixels/'

dataset_info = np.load(DATASET_INFO)
rectangle_set = candidateRects(64, 64)
threshold_set = np.empty([len(rectangle_set), 8], dtype='float32')

print('Loaded rectangle set of length %s' % len(rectangle_set))


def loadPixel(y,x):
    """
    :param x:
    :param y:
    :return:
    """
    filename = DATASET_PIXEL_PATH + 'PATCH_%i_%i.npy' % (y, x)
    return np.load(filename)


def binarResponses(input_array, threshold):
    """
    Binarizing responses, where input array is array of responses for some area
    """
    return input_array > threshold


def calcOnesZeros(input_array, fpg):
    """
    Function calculates 1s and 0s responses in one group
    input array - array of Trues and Falses that are responses of RFD
    fpg - array of First Patches in Group index
    """
    ones_zeros = []
    for i in range(len(fpg)-1):
        ones = 0
        zeros = 0
        for j in range(fpg[i], fpg[i+1]):
            if input_array[j]: ones += 1
            else: zeros += 1
        ones_zeros.append( (ones, zeros) )
    return tuple(ones_zeros)


def calcTP(ones_zeros):
    """
    True Positive rate calculation for current response set
    """
    tp_up = 0
    tp_down = 0
    for i in range(len(ones_zeros)):
        x = ones_zeros[i][0]
        y = ones_zeros[i][1]
        tp_up += x*(x-1) + y*(y-1)
        tp_down += (x+y)*(x+y-1)
    return tp_up / tp_down


def calcTN(ones_zeros):
    """
    True negative rate calculation for current response set
    """
    xsum = 0
    ysum = 0
    for x, y in ones_zeros:
        xsum += x
        ysum += y
    tn_up = 0
    tn_down = 0
    for  i in range(len(ones_zeros)):
        x = ones_zeros[i][0]
        y = ones_zeros[i][1]
        tn_up += x*(ysum-y) + y*(xsum-x)
        tn_down += (xsum+ysum-x-y)*(x+y)
    return tn_up / tn_down


def writeThresholdValue(filename, ytop, xtop, ybot, xbot, channel, threshold):
    with open(filename, 'a+b') as binary_file:
        entry = struct.pack('<HHHHHf', ytop, xtop, ybot, xbot, channel, threshold)
        binary_file.write(entry)
        binary_file.flush()


def calcRectResponse(dataset, dsbegin, dsend, ytop, xtop, ybot, xbot, rect_index, rect_response):
    """
    Takes a dataset, calculates response for one specified rectangle
    and stores value in rect_response array.
    """
    stdout.write('\rCalculating rectangle %i' % rect_index)

    rect0 = loadPixel(ytop, xtop)
    rect1 = loadPixel(ybot, xbot)
    rect2 = loadPixel(ybot, xtop)
    rect3 = loadPixel(ytop, xbot)

    # zd = dataset[:, ytop, xtop, -1] + \
    #         dataset[:, ybot, xbot, -1] - \
    #         dataset[:, ybot, xtop, -1] - \
    #         dataset[:, ytop, xbot, -1]

    zd = rect0[:,:,:,-1] + rect1[:,:,:,-1] - rect2[:,:,:,-1] - rect3[:,:,:,-1]

    rect_response[rect_index, dsbegin:dsend, :] = (dataset[:(dsend-dsbegin), ytop, xtop, :-1] +
            dataset[:(dsend-dsbegin), ybot, xbot, :-1] -
            dataset[:(dsend-dsbegin), ybot, xtop, :-1] -
            dataset[:(dsend-dsbegin), ytop, xbot, :-1]) / zd[None, :(dsend-dsbegin), None]


def searchOptimaThreshold(q, response_array, begin_rect, chan_index, output_file):
    q.put(True)
    # stdout.write('\nCalculating rectangles for channel %s \n' % chan_index)

    for rect_index in range(response_array.shape[0]):
        # Initialiaze binary search
        delta = 0.25
        threshold_middle = 0.5
        bin_resp = binarResponses(response_array[rect_index, :, chan_index], threshold_middle)
        ones_zeros = calcOnesZeros(bin_resp, dataset_info)
        accur_max = 0.5 * calcTP(ones_zeros) + 0.5 * calcTN(ones_zeros)
        optima_threshold = threshold_middle
        while delta > 0.003:
            threshold_left, threshold_right = threshold_middle - delta, threshold_middle + delta
            for cur_threshold in (threshold_left, threshold_right):
                # # stdout.write('Calculating %s rectangle %s channel: threshold %s \n' %
                #              (rect_index, chan_index, cur_threshold))
                bin_resp = binarResponses(response_array[rect_index, :, chan_index], cur_threshold)
                ones_zeros = calcOnesZeros(bin_resp, dataset_info)
                cur_accuracy = 0.5 * calcTP(ones_zeros) + 0.5 * calcTN(ones_zeros)
                if cur_accuracy > accur_max:
                    accur_max = cur_accuracy
                    optima_threshold = cur_threshold
            threshold_middle = optima_threshold
            delta = delta / 2
        stdout.write('Threshold calculated for rectangle (%s, %s, %s, %s) channel %s is %s \n ' %
                     (*rectangle_set[begin_rect + rect_index], chan_index, optima_threshold))
        writeThresholdValue(output_file, *rectangle_set[begin_rect + rect_index], chan_index, optima_threshold)
    q.put(False)


if __name__ == '__main__':
    """
    Main loop: iterating through rectangles, slicing dataset
    and calculating responses, then TP and TN rate for each rectangle,
    then storing into threshold_set array.
    """
    for x in range(3):
        for y in range(4):
            loadPixel(x, y)
