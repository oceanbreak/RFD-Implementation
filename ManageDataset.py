"""
Preparing patch dataset.
Reads selected input patches and calculates gradient bins and integral sum of every patch,
storing it to separate nunpy array with 10 channels, where
1-st one is original image, others are integral sums of gradients.
"""

import sys, os, time
import Utils
from skimage.transform import integral as intg
import numpy as np
from os import listdir

# Global variables
PATCHES_ARRAY_SIZE = (16, 16)
PATCH_SIZE = (64, 64)
DIRECTIONS_NUM = 8
DATASET_DIRECTORIES = []
DATASET_LENGTHS = []
patches_in_img = PATCHES_ARRAY_SIZE[0] * PATCHES_ARRAY_SIZE[1]

TEMP_FILE = '/media/ocean/DATA/temp.dat'
OUTPUT_FILE_NAME = '/media/sf_Share/IPPI/Datasets/Patch/Patch_Dataset.npy'

# Initializing work directories via locate.cfg file
print('Initializing dataset directories:')
with open('locate.cfg', 'r') as init_file:
    for line in init_file:
        DATASET_DIRECTORIES.append(line.rstrip())
        print(line.rstrip())
print('...')


def patchPositionByIndex(index):
    """
    Reads index of an image by order and generates values for looking for an image in patches files
    :param index:
    :return: array: (No of pic., containing patch,
                    row number, column number)
    """
    pic_num = index // patches_in_img
    pic_row = (index % patches_in_img) // PATCHES_ARRAY_SIZE[0]
    pic_col = (index % patches_in_img) % PATCHES_ARRAY_SIZE[1]
    return(pic_num, pic_row, pic_col)

def getPatch(file_list, pic_num, pic_row, pic_col):
    img = Utils.readImPatch(file_list[pic_num])
    top_lft_pix = (PATCH_SIZE[0]*pic_row, PATCH_SIZE[1]*pic_col)
    bot_rgt_pix = (PATCH_SIZE[0]*pic_row + PATCH_SIZE[0], PATCH_SIZE[1]*pic_col + PATCH_SIZE[1])
    return Utils.cropImg(img, top_lft_pix, bot_rgt_pix)

def calcArrayShape():
    """
    Calculating shape of total array, that will store all patches,
    also modifies DATASET_LENGTHS for offset calculation
    :return: aray shape (number of patches, patch height, patch width, number of channels)
    """
    shape = [0, *PATCH_SIZE, DIRECTIONS_NUM + 2]
    for cur_dir in DATASET_DIRECTORIES:
        try:
            info_file = [cur_dir + filename for filename in listdir(cur_dir) if filename.split('.')[-1]=='txt'][0]
        except IndexError:
            print('No info file in folder %s' % cur_dir)
            raise(ValueError)
        cur_patch_num = 0
        with open(info_file, 'r') as info_data:
            for line in info_data:
                cur_patch_num += 1
        shape[0] += cur_patch_num
        DATASET_LENGTHS.append(cur_patch_num)
    return tuple(shape)

def processDataset():
    # Create memmap array:
    shape_arr = calcArrayShape()
    dataset_array = np.memmap(TEMP_FILE, dtype=np.float32, mode='w+', shape=shape_arr)
    print('Created temporary array: %s of size %s' % (TEMP_FILE, str(shape_arr)))

    for offset_index, cur_dir in enumerate(DATASET_DIRECTORIES):
        print('Current directory: ' + cur_dir)
        input_file_list = [cur_dir + f for f in listdir(cur_dir) if f.split('.')[-1] == 'bmp']
        input_file_list.sort()

        # Calculate offset of array
        if offset_index == 0:
            offset = 0
        else:
            offset += DATASET_LENGTHS[offset_index - 1]

        # Read patches and store into array
        for cur_patch_num in range(DATASET_LENGTHS[offset_index]):
            cur_patch = getPatch(input_file_list, *patchPositionByIndex(cur_patch_num))
            cur_patch_gradients = Utils.getHoG(cur_patch)

            for i in range(DIRECTIONS_NUM + 2):
                if i == 0:
                    dataset_array[cur_patch_num + offset, :, :, i] = cur_patch
                else:
                    dataset_array[cur_patch_num + offset, :, :, i] = intg.integral_image(cur_patch_gradients[:, :, i - 1])

            sys.stdout.write('\r' + 'Calculating %i of %i patches' % (cur_patch_num, DATASET_LENGTHS[offset_index]))
        sys.stdout.write('\r' + 'Total of %i patches calculated\n' % DATASET_LENGTHS[offset_index])

    print('Calculated temp array')
    np.save(OUTPUT_FILE_NAME, dataset_array)
    print('Saved to %s' % OUTPUT_FILE_NAME)

    del dataset_array
    os.remove(TEMP_FILE)
    print('Delete temporary array: %s' % TEMP_FILE)



if __name__ == '__main__':
    processDataset()