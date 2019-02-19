"""
version 0.1.1
"""

import sys
import Utils
from skimage import io
from matplotlib import pyplot as plt
import numpy as np

class RFD:
    def __init__(self, input_img, show_steps = False, learn = False):
        self._input_img = input_img
        self._im_height, self._im_width = self._input_img.shape
        self._show_steps = show_steps
        self._orient_quant = 8  # Number of orientations for gradient to assign

        if not learn:
            self._gradMap = Utils.getHoG(self._input_img, self._orient_quant)      # Initialize gradients array
        else:
            self._gradMap = []

    def calculateDescriptor(self):
        for i, array in enumerate(self._gradMap):
            self.show(array, "Gradient dir " + str(i))

    def show(self, img, name):
        if self._show_steps:
            io.imshow(img)
            plt.title(name)
            plt.show()

    def receptiveFieldResponseRect(self, channel_num, top_left_pix, bot_rght_pix, threshold):
        pass

    def calcNormFactor(self, top_left_pix, bot_rght_pix):
        pass

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: python %s <inputfile> \n' % sys.argv[0])
        raise SystemExit(1)
    input_img = Utils.readImPatch(sys.argv[1])
    input_img = Utils.cropImg(input_img, (0,0), (64,64))
    img_to_process = RFD(input_img, True)
    img_to_process.calculateDescriptor()