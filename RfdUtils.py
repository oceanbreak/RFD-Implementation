"""
version 0.1.1
"""

import sys
import Utils
from skimage import io
from skimage.filters import gaussian
from matplotlib import pyplot as plt
import numpy as np
from math import atan2, sqrt, pi

class RFD:
    def __init__(self, input_img, show_steps = False):
        self._input_img = input_img
        self._im_height, self._im_width = self._input_img.shape
        self._show_steps = show_steps

        self._orient_quant = 8  # Number of orientations for gradient to assign
        self._gradMap = []      # Initialize gradients array
        for i in range(self._orient_quant + 1):
            self._gradMap.append(np.zeros((self._im_height, self._im_width)))

    def show(self, img, name):
        if self._show_steps:
            io.imshow(img)
            plt.title(name)
            plt.show()

    def getHoG(self):
        print('Calculating gradient bins ...')
        nDir = self._orient_quant
        GRAD_RADIUS = 1
        height = self._im_height
        width = self._im_width
        sigma0 = 1.6
        smooth_patch = gaussian(self._input_img, sigma0)

        hog = np.zeros(nDir * width * height)

        for y in range(GRAD_RADIUS, height - 1 - GRAD_RADIUS):
            for x in range(GRAD_RADIUS, width - 1 - GRAD_RADIUS):
                grad2 = smooth_patch[y, x + GRAD_RADIUS]
                grad1 = smooth_patch[y, x - GRAD_RADIUS]
                dx = grad2 - grad1
                grad2 = smooth_patch[y + GRAD_RADIUS, x]
                grad1 = smooth_patch[y - GRAD_RADIUS, x]
                dy = grad2 - grad1

                # Calculate direction and magnitude of gradient
                dir = atan2(dy, dx)             # Direction
                mag = sqrt(dx * dx + dy * dy)   # Magnitude

                # Assign direction to pixel
                idxDir = (dir + pi) * nDir / (2.0 * pi)
                if int(idxDir) == nDir: idxDir -= nDir

                # Soft assignment of direction with weights
                dirIdx = [0, 0]
                dirWeight = [0, 0]
                dirIdx[0] = int(idxDir)
                dirIdx[1] = (dirIdx[0] + 1) % nDir
                dirWeight[0] = 1.0 - (idxDir - dirIdx[0])
                dirWeight[1] = idxDir - dirIdx[0]

                # Calculate current position in one-dimensional array
                pos_idx = y * width + x
                hog[dirIdx[0] * width * height + pos_idx] = dirWeight[0] * mag
                hog[dirIdx[1] * width * height + pos_idx] = dirWeight[1] * mag

        for y in range(height):
            for x in range(width):
                Total = 0
                for i in range(nDir):
                    self._gradMap[i][y, x] = hog[i * width * height + y * width + x]
                    Total += hog[i * width * height + y * width + x]
                self._gradMap[nDir][y, x] = Total

        for i, array in enumerate(self._gradMap):
            self.show(array, str(i))



if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: python %s <inputfile> \n' % sys.argv[0])
        raise SystemExit(1)
    input_img = Utils.readImPatch(sys.argv[1])
    input_img = Utils.cropImg(input_img, (0,0), (64,64))
    img_to_process = RFD(input_img, True)
    img_to_process.getHoG()