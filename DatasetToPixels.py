"""
Module for converting dataset of images (64x64) into 64*64 files
for each pixel for faster loading from disk
"""

import numpy as np
from sys import stdout

DATASET = '/home/oceanbreak//Documents/IPPI/Datasets/Patch_Dataset_Integral.npy'
DIST = '/home/oceanbreak//Documents/IPPI/Datasets/dataset_pixels/'

dataset_raw = np.load(DATASET, mmap_mode='r')
temp_dataset = np.zeros((dataset_raw.shape[0],1,1, 9), dtype='float32')
dataset_slice = 5000

for i in range(6, 64):
    for j in range(31, 64):
        current_file_name = 'PATCH_%s_%s.npy' % (i,j)
        stdout.write('Processing file "%s" \n' % current_file_name)
        for slicer in range(dataset_raw.shape[0] // dataset_slice + 1):
            begin = slicer * 5000
            end = begin + dataset_slice if begin + dataset_slice < dataset_raw.shape[0] else dataset_raw.shape[0]
            temp_dataset[begin:end, 0, 0, :] = dataset_raw[begin:end, i, j, 1:]
            stdout.write('\rCalculating pixel (%s, %s) for patches from %s to %s of %s'
                         % (i,j,begin,end, dataset_raw.shape[0]))
        np.save(DIST + current_file_name, temp_dataset)
        # stdout.write('\nFile written\n')