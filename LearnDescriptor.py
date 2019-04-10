"""
<<<<<<< Updated upstream
This module provides searching for optimum threshold value
for rectangular areas of RFD
"""

import numpy as np
from RFD import RFD
from sys import stdout
import Utils
from matplotlib import pyplot as plt

DATASET = '/media/sf_Share/Patch_Dataset_Integral.npy'
DATASET_INFO = '/media/sf_Share/Patch_Dataset_Info_FPG.npy'

dataset = np.load(DATASET, mmap_mode='r')
dataset_info = np.load(DATASET_INFO)
dataset = dataset[:1000]
print(dataset.shape)

response_set = np.zeros((dataset.shape[0], 8))
cur_threshold = 0.1
rectangle_param = (3, (0,0), (32,32))


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

#end = dataset.shape[0]
end = 1000
calculateInterval(0, end)

x = np.array([i for i in range(end)])
y = np.zeros(end)
for i in range(y.shape[0]):
    if i in dataset_info:
        y[i] = 1

plt.plot(x[:end], response_set[:end, 0])
plt.plot(x[:end], y[:end])

plt.show()

Utils.showImage(dataset[16,:12,:12,0])
print('\n')
print(response_set[2, 5])