
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


dataset_raw = np.load(DATASET, mmap_mode='r')
dataset_info = np.load(DATASET_INFO)


im_num = 285965
im_chan = 5

rectangle_param = ((2,4), (32,28))  # Choose one rectangle, top left pixel and bottom right pixel

rect_param_set = [((32,32), (63, 63))]



# initialize array of responses for all channels and one rectangle
# response_set = np.zeros((dataset.shape[0], 8))

def manualCalcRect(dataset, im_num, im_chan, rectangle_param):
    print('Rectangle values:')
    pixel_values = []
    zd_values = []
    for i in range(2):
        for j in range(2):
            x = rectangle_param[i][0]
            y = rectangle_param[j][1]
            v = dataset[im_num, y, x, im_chan+1]
            zd = dataset[im_num, y, x, -1]
            print('x = %i, y = %i : value = %f' % (x,y,v))
            pixel_values.append(v)
            zd_values.append(zd)
    rect_sum = pixel_values[3] + pixel_values[0] - pixel_values[1] - pixel_values[2]
    return rect_sum / (zd_values[3] + zd_values[0] - zd_values[1] - zd_values[2])


def calcArraySum(dataset, ds_start, ds_end, rectangle_param, rect_index ):
    stdout.write('\rCalculating interval (%i, %i), rectangle %i' % (ds_start, ds_end, rect_index))
    rp = []
    for i in range(2):
        for j in range(2):
            x = rectangle_param[i][0]
            y = rectangle_param[j][1]
            rp.append((y,x))

    zd = dataset[:, rp[0][0], rp[0][1], -1] + \
            dataset[:, rp[3][0], rp[3][1], -1] - \
            dataset[:, rp[1][0], rp[1][1], -1] - \
            dataset[:, rp[2][0], rp[2][1], -1]

    rect_response[rect_index, ds_start:ds_end, :] = (dataset[:, rp[0][0], rp[0][1], :-1] +
            dataset[:, rp[3][0], rp[3][1], :-1] -
            dataset[:, rp[1][0], rp[1][1], :-1] -
            dataset[:, rp[2][0], rp[2][1], :-1]) / zd[None, :, None]

dataset_cut = 5000
shape_arr_val = (len(rect_param_set), dataset_raw.shape[0], dataset_raw.shape[-1]-2)
rect_response = np.memmap(TEMP_FILE, dtype=np.float32, mode='w+', shape=shape_arr_val)


for  i in range(dataset_raw.shape[0] // dataset_cut + 1):
    begin = i*dataset_cut
    if (i+1)*dataset_cut < dataset_raw.shape[0]:
        end = (i+1) * dataset_cut
    else:
        end = dataset_raw.shape[0]

    dataset = np.zeros( (end-begin, dataset_raw.shape[1], dataset_raw.shape[2], dataset_raw.shape[-1] -1 ) )
    dataset[:,:,:, :] = dataset_raw[begin:end, :, :, 1:10]

    for i, r in enumerate(rect_param_set):
        calcArraySum(dataset, begin, end, r, i, )
    stdout.write('\nCalculating finished\n')

    del dataset

np.save('Patch_rect_response.npy', rect_response)
os.remove(TEMP_FILE)
