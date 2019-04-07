"""
version 0.1.1
"""

import Utils
from skimage import io
from matplotlib import pyplot as plt
import numpy as np

class RFD:
    def __init__(self, input_img, show_steps = False, learn = False):
        self._input_img = input_img
        self._show_steps = show_steps
        self._orient_quant = 8  # Number of orientations for gradient to assign
        self._is_learning = learn

        if not self._is_learning:
            self._gradMap = Utils.getHoG(self._input_img, self._orient_quant)      # Initialize gradients array
        else:
            self._gradMap = self._input_img[:,:,1:self._orient_quant+2]

    def calculateDescriptor(self):
        pass

    def show(self, img, name):
        if self._show_steps:
            io.imshow(img)
            plt.title(name)
            plt.show()

    def rfResponseRect(self, channel_num, top_left_pix, bot_rght_pix, integral=False):
        # Calculation of rectangular field response
        cur_channel = Utils.calcRectSum(self._gradMap[:,:,channel_num], top_left_pix, bot_rght_pix, integral)
        zr = Utils.calcRectSum(self._gradMap[:,:,-1], top_left_pix, bot_rght_pix, integral)
        return cur_channel / zr

    def binarResponse(self, response, threshold):
        return int(response > threshold)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: python %s <inputfile> \n' % sys.argv[0])
        raise SystemExit(1)
    input_img = Utils.readImPatch(sys.argv[1])
    input_img = Utils.cropImg(input_img, (0,0), (64,64))
    img_to_process = RFD(input_img, True)
    img_to_process.calculateDescriptor()