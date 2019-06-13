"""
<<<<<<< Updated upstream
This module provides searching for optimum threshold value
for rectangular areas of RFD
"""

import numpy as np
import psutil
from RectangleCalculation import candidateRects
from sys import stdout
import struct
from multiprocessing import Process, Queue

# Initializing dataset array, fpg, and rectangle list
DATASET = '/home/oceanbreak//Documents/IPPI/Datasets/Patch_Dataset_Integral.npy'
DATASET_INFO = '/home/oceanbreak//Documents/IPPI/Datasets/Patch_Dataset_Info_FPG.npy'

dataset_raw = np.load(DATASET, mmap_mode='r')
dataset_info = np.load(DATASET_INFO)
rectangle_set = candidateRects(*dataset_raw.shape[1:3])
threshold_set = np.empty([len(rectangle_set), 8], dtype='float32')

print('Loaded rectangle set of length %s' % len(rectangle_set))
print('Loaded dataset of shape %s' % str(dataset_raw.shape))

rect_set_slice = 99
dataset_slice = 5000

# Temrporary arrays for storing calculated responses and input dataset sliced into parts
temp_dataset_array = np.zeros([dataset_slice, 64, 64, 9], dtype='float32')
temp_response_array = np.zeros([rect_set_slice, dataset_raw.shape[0], 8], dtype='float32')

print(psutil.virtual_memory().free)

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

    zd = dataset[:, ytop, xtop, -1] + \
            dataset[:, ybot, xbot, -1] - \
            dataset[:, ybot, xtop, -1] - \
            dataset[:, ytop, xbot, -1]

    rect_response[rect_index, dsbegin:dsend, :] = (dataset[:(dsend-dsbegin), ytop, xtop, :-1] +
            dataset[:(dsend-dsbegin), ybot, xbot, :-1] -
            dataset[:(dsend-dsbegin), ybot, xtop, :-1] -
            dataset[:(dsend-dsbegin), ytop, xbot, :-1]) / zd[None, :(dsend-dsbegin), None]


def searchOptimaThreshold(q, response_array, rect_index_start, rect_index_end, output_file):
    for rect_index in range(rect_index_start, rect_index_end):
        stdout.write('\nCalculating rectangles %s to %s' % (rect_index_start, rect_index_end))
        for chan_index in range(response_array.shape[2]):
            # Initialiaze binary search
            delta = 0.25
            threshold_middle = 0.5
            bin_resp = binarResponses(response_array[rect_index, :, chan_index], threshold_middle)
            ones_zeros = calcOnesZeros(bin_resp, dataset_info)
            accur_max = 0.5 * calcTP(ones_zeros) + 0.5 * calcTN(ones_zeros)
            optima_threshold = threshold_middle
            while delta > 0.004:
                threshold_left, threshold_right = threshold_middle - delta, threshold_middle + delta
                for cur_threshold in (threshold_left, threshold_right):
                    stdout.write('\rCalculating %s rectangle %s channel: threshold %s ' %
                                 (rect_index, chan_index, cur_threshold))
                    bin_resp = binarResponses(response_array[rect_index, :, chan_index], cur_threshold)
                    ones_zeros = calcOnesZeros(bin_resp, dataset_info)
                    cur_accuracy = 0.5 * calcTP(ones_zeros) + 0.5 * calcTN(ones_zeros)
                    if cur_accuracy > accur_max:
                        accur_max = cur_accuracy
                        optima_threshold = cur_threshold
                threshold_middle = optima_threshold
                delta = delta / 2
            stdout.write('\r\n ')
            writeThresholdValue(output_file, *rectangle_set[rect_index], chan_index, optima_threshold)
        q.put(True)
    q.put(False)


if __name__ == '__main__':
    """
    Main loop: iterating through rectangles, slicing dataset
    and calculating responses, then TP and TN rate for each rectangle,
    then storing into threshold_set array.
    """

    for rect_group in range(len(rectangle_set) // rect_set_slice + 1):
        begin_rect = rect_group*rect_set_slice
        end_rect = (rect_group + 1)*rect_set_slice
        end_rect = end_rect if end_rect < len(rectangle_set) else len(rectangle_set)
        stdout.write('Calculating rectangles from %s to %s \n' % (begin_rect, end_rect))

        for set_group in range(dataset_raw.shape[0] // dataset_slice + 1):
            begin_set = set_group*dataset_slice
            end_set = (set_group + 1)*dataset_slice
            end_set = end_set if end_set < dataset_raw.shape[0] else dataset_raw.shape[0]
            temp_dataset_array[:(end_set - begin_set),:,:,:] = dataset_raw[begin_set : end_set, :, :, 1:]
            stdout.write('    Calculating patches from %s to %s \n\n ' % (begin_set, end_set))

            for rect_index in range(begin_rect, end_rect):
                calcRectResponse(temp_dataset_array, begin_set, end_set, *rectangle_set[rect_index],
                                 rect_index%rect_set_slice, temp_response_array)

        proc = [None]*3
        queue = Queue()
        for index in range(3):
            start = index*33
            end = (index+1)*33
            filename = 'Thresh_Learned_' + str(index)
            proc[index] = Process(target=searchOptimaThreshold, args=(queue, temp_response_array, start, end, filename))
            proc[index].start()
            print('\nProcess %s started\n' % index)

        result = True
        while result:
            for pr in proc:
                result = queue.get()
