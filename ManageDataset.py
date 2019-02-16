"""
Preparing patch dataset
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

def createIntegralImages():
    root = Tk()
    input_file_list = filedialog.askopenfilenames()
    root.destroy()
    for  input_file in input_file_list:
        img = Utils.readImPatch(input_file)
        print('Processing ' + input_file)

        # Initialize array for integral images
        img_int = np.zeros(img.shape)

        for j in range(PATCHES_ARRAY_SIZE[0]):
            for i in range(PATCHES_ARRAY_SIZE[1]):
                im_patch = Utils.cropImg(img, (i*PATCH_SIZE[0], j*PATCH_SIZE[1]),
                                         (i*PATCH_SIZE[0] + PATCH_SIZE[0], j*PATCH_SIZE[1] + PATCH_SIZE[1]))

                img_integral = intg.integral_image(im_patch)

                img_int[j*PATCH_SIZE[1] : j*PATCH_SIZE[1] + PATCH_SIZE[1],
                        i*PATCH_SIZE[0] : i*PATCH_SIZE[0] + PATCH_SIZE[0]] = img_integral


        output_file = '.'.join(input_file.split('.')[:-1]) + '_integral.tif'
        io.imsave(output_file, img_int)

if __name__ == '__main__':
    createIntegralImages()