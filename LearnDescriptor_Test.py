
import Utils
import numpy as np
import os
from sys import stdout
from matplotlib import pyplot as plt

# DATASET = '/media/sf_Share/Patch_Dataset_Integral.npy'
# DATASET_INFO = '/media/sf_Share/Patch_Dataset_Info_FPG.npy'
TEMP_FILE = 'temp.dat'

DATASET = '/home/oceanbreak//Documents/IPPI/Datasets/Patch_Dataset_Integral.npy'
DATASET_INFO = '/home/oceanbreak//Documents/IPPI/Datasets/Patch_Dataset_Info_FPG.npy'
dataset_cut = 10000

dataset_raw = np.load(DATASET, mmap_mode='r')
dataset_info = np.load(DATASET_INFO)


im_num = 12
im_chan = 6

# Utils.showImage(dataset[im_num,:,:,im_chan])

rectangle_param = ((2,4), (32,28))  # Choose one rectangle, top left pixel and bottom right pixel
rect_param_set = ( ((2,4), (32,26)),
                   ((4,6), (10, 42)),
                   ((0,0), (16,28)))



# initialize array of responses for all channels and one rectangle
# response_set = np.zeros((dataset.shape[0], 8))

def manualCalcRect(dataset, rectangle_param):
    print('Rectangle values:')
    pixel_values = []
    zd_values = []
    for i in range(2):
        for j in range(2):
            x = rectangle_param[i][0]
            y = rectangle_param[j][1]
            v = dataset[im_num, y, x, im_chan]
            zd = dataset[im_num, y, x, -1]
            print('x = %i, y = %i : value = %f' % (x,y,v))
            pixel_values.append(v)
            zd_values.append(zd)
    rect_sum = pixel_values[3] + pixel_values[0] - pixel_values[1] - pixel_values[2]
    w_r_sum = rect_sum / (zd_values[3] + zd_values[0] - zd_values[1] - zd_values[2])
    print('Rectangle sum: %f' % rect_sum)
    print('Weighted Sum: %f ' % w_r_sum)
    print()


def calcArraySum(dataset, rectangle_param, index):
    print('Assigning values')
    rp = []
    k = 0
    for i in range(2):
        for j in range(2):
            # pixel_coords.append((rectangle_param[i][0], rectangle_param[j][1]))
            x = rectangle_param[i][0]
            y = rectangle_param[j][1]
            rp.append((y,x))

    zd = dataset[:, rp[0][0], rp[0][1], -1] + \
            dataset[:, rp[3][0], rp[3][1], -1] - \
            dataset[:, rp[1][0], rp[1][1], -1] - \
            dataset[:, rp[2][0], rp[2][1], -1]


    value[index,:,:] = (dataset[:, rp[0][0], rp[0][1], :-1] + \
            dataset[:, rp[3][0], rp[3][1], :-1] - \
            dataset[:, rp[1][0], rp[1][1], :-1] - \
            dataset[:, rp[2][0], rp[2][1], :-1] ) / zd[None, :, None]
    print('End of calculation')
    print('Calculated throug array size:')
    print('Rectangle sum %f ' % value[index, im_num, im_chan])

    # print('Weighted sum %f ' % (value[index, im_num, im_chan] / value[index, im_num, -1]) )
    print()


dataset = dataset_raw[:dataset_cut, :, :, 1:10]
shape_arr_val = (len(rect_param_set), dataset.shape[0], dataset.shape[-1]-1)
value = np.zeros(shape_arr_val)

for i, r in enumerate(rect_param_set):
    manualCalcRect(dataset, r)
    calcArraySum(dataset, r, i)
