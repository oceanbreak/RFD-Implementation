
import Utils
import numpy as np
from sys import stdout
from matplotlib import pyplot as plt

# DATASET = '/media/sf_Share/Patch_Dataset_Integral.npy'
# DATASET_INFO = '/media/sf_Share/Patch_Dataset_Info_FPG.npy'

DATASET = '/home/oceanbreak//Documents/IPPI/Datasets/Patch_Dataset_Integral.npy'
DATASET_INFO = '/home/oceanbreak//Documents/IPPI/Datasets/Patch_Dataset_Info_FPG.npy'

dataset = np.load(DATASET, mmap_mode='r')
dataset_info = np.load(DATASET_INFO)
dataset = dataset[:50000]
im_num = 12
im_chan = 6
print(dataset.shape)
# Utils.showImage(dataset[im_num,:,:,im_chan])

rectangle_param = ((2,4), (32,28))  # Choose one rectangle, top left pixel and bottom right pixel
rect_param_set = ( ((2,4), (32,26)),
                   ((4,6), (10, 42)),
                   ((0,0), (16,28)))

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


def calcArraySum(rectangle_param):
    # pixel_values = np.zeros((dataset.shape[0], 8, 4))
    shape_arr = (dataset.shape[0], 8)
    # pixel_values = np.memmap('temp.dat', dtype=np.float32, mode='w+', shape=shape_arr)
    value = np.zeros(shape_arr)
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
    value[:,:] = dataset[:, rp[0][0], rp[0][1], 1:9] + \
            dataset[:, rp[3][0], rp[3][1], 1:9] - \
            dataset[:, rp[1][0], rp[1][1], 1:9] - \
            dataset[:, rp[2][0], rp[2][1], 1:9]
    print('End of calculation')
    print('Calculated throug array size:')
    print(value.shape)
    print('Rectangle sum %f' % value[im_num, im_chan-1])
    del value

def calcArraySum1(rectangle_param):
    # pixel_values = np.zeros((dataset.shape[0], 8, 4))
    # pixel_values = np.memmap('temp.dat', dtype=np.float32, mode='w+', shape=shape_arr)
    rec_param = []
    print('Assigning values')
    for i in range(2):
        for j in range(2):
            x = rectangle_param[i][0]
            y = rectangle_param[j][1]
            rec_param.append((y,x))

    value1 = dataset[:, rec_param[0][0], rec_param[0][1], 1:9]
    value2 = dataset[:, rec_param[1][0], rec_param[1][1], 1:9]
    value3 = dataset[:, rec_param[2][0], rec_param[2][1], 1:9]
    value4 = dataset[:, rec_param[3][0], rec_param[3][1], 1:9]
    print('Now calculating value')
    value = value1 + value4 - value2 - value3


    print('End of calculation')
    print('Calculated throug array size:')
    print(value.shape)
    print('Rectangle sum %f' % value[im_num, im_chan-1])

    del value, value1, value2, value3, value4

for r in rect_param_set:
    manualCalcRect(r)
    calcArraySum(r)



#
# def calcRectSum(input_image_vector, top_left_pix, bot_rght_pix, integral=False):
#     """
#     Calculate sum in rectangle area of given image and borders
#     :param input_image_vector as numpy float array
#     :param top_left_pix (y,x)
#     :param bot_rght_pix (y,x)
#     :param integral: Set True if input is an integral image
#     :return: sum value
#     """
#     print(input_image_vector.shape)
#     if integral:
#         x1 = top_left_pix[1] - 1
#         x2 = bot_rght_pix[1]
#         y1 = top_left_pix[0] - 1
#         y2 = bot_rght_pix[0]
#         if x1 < 0 & y1 < 0:
#             return input_image_vector[:, y2, x2]
#         elif x1 >= 0 & y1 < 0:
#             return input_image_vector[:, y2, x2] - input_image_vector[:, y2, x1]
#         elif x1 < 0 & y1 >= 0:
#             return input_image_vector[:, y2, x2] - input_image_vector[:, y1, x2]
#         else:
#             return input_image_vector[:, y2, x2] - input_image_vector[:, y1, x2] - input_image_vector[:, y2, x1] + input_image_vector[:, y1, x1]
#     else:
#         sum = 0
#         for i in range(top_left_pix[1], bot_rght_pix[1]+1):
#             for j in range(top_left_pix[0], bot_rght_pix[0]+1):
#                 sum += input_image_vector[i, j]
#         return sum
#
#
# def calculateRfd():
#     """
#     Function that calculates response for i-th patch for one rectangle and 8 directions
#     :param i:
#     :return:
#     """
#     print('Calculating Rfd\n')
#     zd = calcRectSum(dataset[:, :, :, -1], rectangle_param[0], rectangle_param[1], integral=True)   # Sum of all channels
#     for channel_num in range(8):
#         stdout.write('Processing %i channel' % channel_num)
#         cur_channel = calcRectSum(dataset[:, :, :, channel_num+1], rectangle_param[0], rectangle_param[1], integral=True)
#         cur_channel = cur_channel / zd
#         response_set[:, channel_num] = cur_channel
#
#
#
# calculateRfd()
#
# # end = 1000
# #
# # x = np.array([i for i in range(end)])
# # y = np.zeros(end)
# # for i in range(y.shape[0]):
# #     if i in dataset_info:
# #         y[i] = 1
# #
# # plt.plot(x[:end], response_set[:end, 5])
# # plt.plot(x[:end], y[:end])
# # plt.show()
#
#
# print(response_set[2,5])
