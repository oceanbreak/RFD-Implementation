"""
This module provides searching for optimum threshold value
for rectangular areas of RFD
"""

import numpy as np
from RFD import RFD
from sys import stdout
import Utils

DATASET = '/home/oceanbreak/Documents/IPPI/Datasets/Patch_Dataset_Integral.npy'

dataset = np.load(DATASET, mmap_mode='r+')
print(dataset.shape)

response_set = np.zeros((dataset.shape[0], 8))
cur_threshold = 0.1
rectangle_param = (3, (2,2), (26,26))


# def calculateRfd(i):
#     img_descriptor = RFD(dataset[i], learn=True)
#     response_set[i]=(img_descriptor.rfResponseRect(*rectangle_param, cur_threshold, integral=True))

def calculateRfd(i):
    zd = Utils.calcRectSum(dataset[i, :, :, -1], rectangle_param[1], rectangle_param[2], integral=True)
    for channel_num in range(8):
        cur_channel = Utils.calcRectSum(dataset[i, :, :, channel_num+1], rectangle_param[1], rectangle_param[2], integral=True)
        response_set[i, channel_num] = cur_channel/ zd

def calculateInterval(begin, end):
    stdout.write('Calcilatinng interval (%i, %i)\n\n' % (begin, end))
    for i in range(begin, end):
        stdout.write('\r' + 'Processing %i image of %i' % (i, end-begin))
        calculateRfd(i)

calculateInterval(0, 10000)
print(response_set[:50])
