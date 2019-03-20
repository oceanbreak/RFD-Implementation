"""
Preparing patch dataset.
Reads selected input patches and calculates gradient bins and integral sum of every patch,
storing it to separate .tif file with name, corresponding to gradient channel number.
"""

import Utils
from skimage.transform import integral as intg
from skimage import io
from matplotlib import pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join

PATCHES_ARRAY_SIZE = (16, 16)
PATCH_SIZE = (64, 64)
DIRECTIONS_NUM = 8
DATASET_PATH = '/home/ocean/Documents/Patch/halfdome/'
patches_in_img = PATCHES_ARRAY_SIZE[0] * PATCHES_ARRAY_SIZE[1]

def patchPositionByIndex(index):
    pic_num = index // patches_in_img
    pic_row = (index % patches_in_img) // PATCHES_ARRAY_SIZE[0]
    pic_col = (index % patches_in_img) % PATCHES_ARRAY_SIZE[1]
    return(pic_num, pic_row, pic_col)

def getPatch(file_list, pic_num, pic_row, pic_col):
    img = Utils.readImPatch(file_list[pic_num])
    top_lft_pix = (PATCH_SIZE[0]*pic_row, PATCH_SIZE[1]*pic_col)
    bot_rgt_pix = (PATCH_SIZE[0]*pic_row + PATCH_SIZE[0], PATCH_SIZE[1]*pic_col + PATCH_SIZE[1])
    return Utils.cropImg(img, top_lft_pix, bot_rgt_pix)

def processImagesBatch():
    """
    Function asks user for input images (originally from Patch Dataset) and calculates gradients for specified
    number of direction bins and stores them to one .mpy file
    """

    # Generate array of input files
    input_file_list = [DATASET_PATH + f for f in listdir(DATASET_PATH) if  f.split('.')[-1]=='bmp']
    input_file_list.sort()

    output_file_name = '/'.join(input_file_list[0].split('/')[:-2] + [input_file_list[0].split("/")[-2] + '_dataset_grad_integr.npy'])
    output_file_name = input_file_list[0].split('.')[-2] + '.npy'
    print(output_file_name)
    print(input_file_list)


    for i in range(9, 12):
        patch_config = patchPositionByIndex(i)
        cur_patch = getPatch(input_file_list, *patch_config)
        cur_patch_hog = Utils.getHoG(cur_patch)
        print(cur_patch_hog.shape)
        # for j in range(9):
        #     io.imshow(cur_patch_hog[:,:,j])
        #     plt.title(i)
        #     plt.show()
    shape_arr = (len(input_file_list) * PATCHES_ARRAY_SIZE[0] * PATCHES_ARRAY_SIZE[1],
             *PATCH_SIZE, DIRECTIONS_NUM + 2)
    print(shape_arr)
    img_grad_integr_array = np.memmap('temp.dat', dtype=np.float32, mode='w+', shape=shape_arr)
    # img_grad_integr_array = np.zeros((len(input_file_list)*PATCHES_ARRAY_SIZE[0]*PATCHES_ARRAY_SIZE[1],
    #                                       *PATCH_SIZE, DIRECTIONS_NUM + 1))
    print(img_grad_integr_array.shape)
    # for  file_index, input_file in enumerate(input_file_list):
    #     img = Utils.readImPatch(input_file)
    #     print('Processing ' + input_file)

        # Initialize array for integral images of all gradient bins with final total gradient
        #img_int = np.zeros((*img.shape, DIRECTIONS_NUM + 1))

        # Iterate through image
        # for j in range(PATCHES_ARRAY_SIZE[0]):
        #     for i in range(PATCHES_ARRAY_SIZE[1]):
        #         im_patch = Utils.cropImg(img, (i*PATCH_SIZE[0], j*PATCH_SIZE[1]),
        #                                  (i*PATCH_SIZE[0] + PATCH_SIZE[0], j*PATCH_SIZE[1] + PATCH_SIZE[1]))
        #
        #         io.imshow(im_patch)
        #         plt.show()
        #
        #         # Calculate gradient of patch and count integral sum of every bin
        #         img_grad_array = Utils.getHoG(im_patch)
        #
        #         # for dir_idx, img_gradient in enumerate(img_grad_array):
        #         #     img_integral = intg.integral_image(img_gradient)
        #             # img_int[j*PATCH_SIZE[1] : j*PATCH_SIZE[1] + PATCH_SIZE[1],
        #             #         i*PATCH_SIZE[0] : i*PATCH_SIZE[0] + PATCH_SIZE[0], dir_idx] = img_integral
        #
        #             # print(file_index*PATCHES_ARRAY_SIZE[0]*PATCHES_ARRAY_SIZE[1] + j*PATCHES_ARRAY_SIZE[0] + i,
        #             #          j * PATCH_SIZE[1] + PATCH_SIZE[1],
        #             #          i*PATCH_SIZE[0] + PATCH_SIZE[0], dir_idx)
        #
        #
        #             # img_grad_integr_array[file_index*PATCHES_ARRAY_SIZE[0]*PATCHES_ARRAY_SIZE[1] + j*PATCHES_ARRAY_SIZE[0] + i,
        #             #         j*PATCH_SIZE[1] : j*PATCH_SIZE[1] + PATCH_SIZE[1],
        #             #         i*PATCH_SIZE[0] : i*PATCH_SIZE[0] + PATCH_SIZE[0], dir_idx] = file_index*PATCHES_ARRAY_SIZE[0]*PATCHES_ARRAY_SIZE[1] + j*PATCHES_ARRAY_SIZE[0] + i
        #
        # #print(img_grad_integr_array.shape)

if __name__ == '__main__':
    processImagesBatch()