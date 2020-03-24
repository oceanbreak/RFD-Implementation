"""
<<<<<<< Updated upstream
This module provides searching for optimum threshold value
for rectangular areas of RFD
"""

import numpy as np
from RectangleCalculation import candidateRects
from sys import stdout
from matplotlib import pyplot as plt
from random import randint

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


def AccuracyVrsThreshold(threshold_vector, input_array, fpg):
    tp = []
    tn = []
    accuracy = []
    for t in threshold_vector:
        ones_zeros = binarResponses(input_array, t)
        ones_zeros = calcOnesZeros(ones_zeros, fpg)
        tp_v = calcTP(ones_zeros)
        tn_v = calcTN(ones_zeros)
        tp.append(tp_v)
        tn.append(tn_v)
        accuracy.append( 0.5*tp_v + 0.5*tn_v )
    return np.column_stack((tp, tn, accuracy))


def plotAccuracyThreshold(rect, channel):
    t_vector = np.arange(0, 1, 0.05)
    responses = calcRectResponse(*rect)[:,0,0,channel]
    a_vector = AccuracyVrsThreshold(t_vector, responses, dataset_info)
    index_max = np.argmax(a_vector[:,2])
    print('Rectangle (%i,%i), (%i,%i); Channel %i THRESHOLD: %f ' % (*rect, channel, t_vector[index_max]))
    plt.plot(t_vector, a_vector[:, 0], label='TP')
    plt.plot(t_vector, a_vector[:, 1], label='TN')
    plt.plot(t_vector, a_vector[:, 2], label='(TP+TN)/2')
    plt.title('Rectangle (%i,%i), (%i,%i); Channel %i' % (*rect, channel))
    # plt.legend()
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.ion()
    plt.show()
    plt.pause(0.001)
    # input("Press [enter] to continue.")


def calcRectResponse(ytop, xtop, ybot, xbot):
    """
    Takes a dataset, calculates response for one specified rectangle
    and stores value in rect_response array.
    """
    stdout.write('\rCalculating rectangle (%i, %i), (%i, %i)' % (ytop, xtop, ybot, xbot))

    rect0 = loadPixel(ytop, xtop)
    rect1 = loadPixel(ybot, xbot)
    rect2 = loadPixel(ybot, xtop)
    rect3 = loadPixel(ytop, xbot)

    zd = rect0[:,:,:,-1] + rect1[:,:,:,-1] - rect2[:,:,:,-1] - rect3[:,:,:,-1]
    return np.nan_to_num(np.divide(rect0[:,:,:,:-1] + rect1[:,:,:,:-1] - rect2[:,:,:,:-1] - rect3[:,:,:,:-1],
                                   zd[:, :, None]) )


def searchOptimalThreshold(response_array, begin_rect, chan_index, output_file):
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



if __name__ == '__main__':
    """
    Main loop: iterating through rectangles, slicing dataset
    and calculating responses, then TP and TN rate for each rectangle,
    then storing into threshold_set array.
    """
    for i in range(50):
        x = randint(0, len(rectangle_set)-1)
        channel = randint(0,7)
        cur_rect = rectangle_set[x]
        plotAccuracyThreshold(cur_rect, channel)
    input('press any key to quit')