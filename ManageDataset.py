"""
Preparing patch dataset.
Reads selected input patches and calculates gradient bins and integral sum of every patch,
storing it to separate nunpy array with 10 channels, where
1-st one is original image, others are integral sums of gradients.
"""

import Utils
from skimage.transform import integral as intg
import numpy as np
from os import listdir

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

    # Generate array of information about patches
    info_file = [DATASET_PATH + f for f in listdir(DATASET_PATH) if  f.split('.')[-1]=='txt'][0]
    patches_info_list = []

    with open(info_file, 'r') as info:
        for line in info:
            patches_info_list.append(line.split()[0])

    output_file_name = '/'.join(input_file_list[0].split('/')[:-1]) + '.npy'

    shape_arr = (len(input_file_list) * PATCHES_ARRAY_SIZE[0] * PATCHES_ARRAY_SIZE[1],
             *PATCH_SIZE, DIRECTIONS_NUM + 2)

    img_grad_integr_array = np.memmap('temp.dat', dtype=np.float32, mode='w+', shape=shape_arr)
    print('Created Temporary Array temp.dat of size ' + str(shape_arr))

    for cur_num in range(len(patches_info_list)):
        cur_patch = getPatch(input_file_list, *patchPositionByIndex(cur_num))
        cur_patch_gradients = Utils.getHoG(cur_patch)

        for i in range(DIRECTIONS_NUM + 2):
            if i==0:
                img_grad_integr_array[cur_num, :, :, i] = cur_patch
            else:
                img_grad_integr_array[cur_num, :, :, i] = intg.integral_image(cur_patch_gradients[:,:,i-1])
        print('Calculating ... %i' % cur_num)

    print('Calculated temp array')
    np.save(output_file_name, img_grad_integr_array)
    print('saved as %s' % output_file_name)


if __name__ == '__main__':
    processImagesBatch()