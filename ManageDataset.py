"""
Preparing patch dataset.
Reads selected input patches and calculates gradient bins and integral sum of every patch,
storing it to separate .tif file with name, corresponding to gradient channel number.
"""

import Utils
from skimage.transform import integral as intg
from tkinter import Tk
from skimage import io
from tkinter import filedialog
from matplotlib import pyplot as plt
import numpy as np

PATCHES_ARRAY_SIZE = (16, 16)
PATCH_SIZE = (64, 64)
DIRECTIONS_NUM = 8

def processImagesBatch():
    """
    Function asks user for input images (originally from Patch Dataset) and calculates gradients for specified
    number of direction bins and stores them to .tif files.
    """
    root = Tk()
    input_file_list = filedialog.askopenfilenames()
    root.destroy()
    for  input_file in input_file_list:
        img = Utils.readImPatch(input_file)
        print('Processing ' + input_file)

        # Initialize array for integral images of all gradient bins with final total gradient
        img_int = [np.zeros(img.shape) for dir_idx in range(DIRECTIONS_NUM + 1)]

        # Iterate through image
        for j in range(PATCHES_ARRAY_SIZE[0]):
            for i in range(PATCHES_ARRAY_SIZE[1]):
                im_patch = Utils.cropImg(img, (i*PATCH_SIZE[0], j*PATCH_SIZE[1]),
                                         (i*PATCH_SIZE[0] + PATCH_SIZE[0], j*PATCH_SIZE[1] + PATCH_SIZE[1]))

                # Calculate gradient of patch and count integral sum of every bin
                img_grad_array = Utils.getHoG(im_patch)

                for dir_idx, img_gradient in enumerate(img_grad_array):
                    img_integral = intg.integral_image(img_gradient)

                    img_int[dir_idx][j*PATCH_SIZE[1] : j*PATCH_SIZE[1] + PATCH_SIZE[1],
                            i*PATCH_SIZE[0] : i*PATCH_SIZE[0] + PATCH_SIZE[0]] = img_integral

        # Save integral images to file (last Total gradient excluded)
        for dir_idx, cur_int_patch in enumerate(img_int):
            if dir_idx < DIRECTIONS_NUM:
                output_file = '.'.join(input_file.split('.')[:-1]) + 'ch%d_integral.tif' % dir_idx
                io.imsave(output_file, cur_int_patch)

if __name__ == '__main__':
    processImagesBatch()