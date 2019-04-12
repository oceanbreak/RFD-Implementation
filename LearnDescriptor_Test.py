
import Utils
import numpy as np
from sys import stdout
from matplotlib import pyplot as plt

DATASET = '/media/sf_Share/Patch_Dataset_Integral.npy'
DATASET_INFO = '/media/sf_Share/Patch_Dataset_Info_FPG.npy'
TEMP_FILE = 'temp.dat'

# DATASET = '/home/oceanbreak//Documents/IPPI/Datasets/Patch_Dataset_Integral.npy'
# DATASET_INFO = '/home/oceanbreak//Documents/IPPI/Datasets/Patch_Dataset_Info_FPG.npy'
dataset_cut = 25000

dataset_raw = np.load(DATASET, mmap_mode='r')
dataset = np.zeros((dataset_cut, 64, 64, 9))

dataset_info = np.load(DATASET_INFO)
dataset[:,:,:] = dataset_raw[:dataset_cut, :, :, 1:10]
im_num = 12
im_chan = 6
print(dataset.shape)
# Utils.showImage(dataset[im_num,:,:,im_chan])

rectangle_param = ((2,4), (32,28))  # Choose one rectangle, top left pixel and bottom right pixel
rect_param_set = ( ((2,4), (32,26)),
                   ((4,6), (10, 42)),
                   ((0,0), (16,28)))

shape_arr_val = (len(rect_param_set), dataset.shape[0], dataset.shape[-1])
# value = np.memmap(TEMP_FILE, dtype=np.float32, mode='w+', shape=shape_arr_val)
value = np.zeros(shape_arr_val)

# initialize array of responses for all channels and one rectangle
# response_set = np.zeros((dataset.shape[0], 8))

def manualCalcRect(rectangle_param):
    print('Rectangle values:')
    pixel_values = []
    for i in range(2):
        for j in range(2):
            x = rectangle_param[i][0]
            y = rectangle_param[j][1]
            v = dataset[im_num, y, x, im_chan]
            print('x = %i, y = %i : value = %f' % (x,y,v))
            pixel_values.append(v)
    rect_sum = pixel_values[3] + pixel_values[0] - pixel_values[1] - pixel_values[2]
    print('Rectangle sum: %f' % rect_sum)


def calcArraySum(rectangle_param, index):
    # pixel_values = np.zeros((dataset.shape[0], 8, 4))
    # pixel_values = np.memmap('temp.dat', dtype=np.float32, mode='w+', shape=shape_arr)
    # value = np.memmap('temp.dat', dtype=np.float32, mode='w+', shape=shape_arr)
    # print(pixel_values.shape)
    print('Assigning values')
    rp = []
    k = 0
    for i in range(2):
        for j in range(2):
            print('Iteraton %i' % k)

            # pixel_coords.append((rectangle_param[i][0], rectangle_param[j][1]))
            x = rectangle_param[i][0]
            y = rectangle_param[j][1]

            if k==0 or k==3: sign = 1
            else: sign = -1

            rp.append((y,x))

            # value = value + sign * dataset[:, y, x, 1:9]

            k += 1
    # global value
    value[index,:,:] = dataset[:, rp[0][0], rp[0][1], 0:9] + \
            dataset[:, rp[3][0], rp[3][1], 0:9] - \
            dataset[:, rp[1][0], rp[1][1], 0:9] - \
            dataset[:, rp[2][0], rp[2][1], 0:9]
    print('End of calculation')
    print('Calculated throug array size:')
    print(value.shape)
    print('Rectangle sum %f' % value[index, im_num, im_chan])


# def testNpAr():
#     arr1 = np.array([0.001*i*13%23 for i in range(3000000)])
#     arr2 = np.array([0.001*i*91%37 for i in range(3000000)])
#     print(arr1+arr2*arr1)
# testNpAr()

for i, r in enumerate(rect_param_set):
    manualCalcRect(r)
    calcArraySum(r, i)
