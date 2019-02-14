"""
version 0.1.0
"""

import sys
from skimage import io
from skimage import img_as_float as imf
from skimage.color import rgb2gray
from skimage.filters import gaussian
from matplotlib import pyplot as plt
import numpy as np
from math import atan2, sqrt, pi


class RfdUtils:
    # Class that reads image and seprates it into set of gradients in 8 directions
    # Input: RGB or Grayscale image
    def __init__(self, input_image, show_steps = False):
        if len(input_image) > 20:
            self._im_name = input_image[:3] + "..." + input_image[-20:]
        else:
            self._im_name = input_image
        self._input_image = io.imread(input_image)
        self._show_steps = show_steps    # Enables to plot each step of program

        # Convert to grayscale and then to float array
        self._in_gray_img = rgb2gray(self._input_image)
        self._in_float_img = imf(self._in_gray_img)

        # Calculate height and width
        self._im_height = self._input_image.shape[0]
        self._im_width = self._input_image.shape[1]

        self._orient_quant = 8  # Quantization of gradient directions
        self._gradMap = []      # Initialize output gradients array
        for i in range(self._orient_quant+1):
            self._gradMap.append(np.zeros((self._im_height, self._im_width)))

        self.show(self._in_float_img, 'Grayscaled input image "%s"' % self._im_name)

    def show(self, img, name):
        if self._show_steps:
            io.imshow(img)
            plt.title(name)
            plt.show()

    def showGradients(self):
        f = plt.figure()
        for array, num in zip(self._gradMap, range(1, len(self._gradMap))):
            ax = f.add_subplot(2, self._orient_quant/2, num)
            ax.margins(0)
            io.imshow(array)
            ax.set_title(str(num))
        f.tight_layout()
        plt.show()

    # This function is almost identical copy of exactly the same function in RFD source code
    def getHoG(self):
        print('Calculating gradient bins ...')
        nDir = self._orient_quant
        GRAD_RADIUS = 1
        height = self._im_height
        width = self._im_width
        sigma0 = 1.6
        smooth_patch = gaussian(self._in_gray_img, sigma0)

        hog = np.zeros(nDir * width * height)

        for y in range(GRAD_RADIUS, height-1-GRAD_RADIUS):
            for x in range(GRAD_RADIUS, width-1-GRAD_RADIUS):
                gray2 = smooth_patch[y, x+GRAD_RADIUS]
                gray1 = smooth_patch[y, x-GRAD_RADIUS]
                dx = gray2 - gray1
                gray2 = smooth_patch[y+GRAD_RADIUS, x]
                gray1 = smooth_patch[y-GRAD_RADIUS, x]
                dy = gray2 - gray1

                # Calculate direction and magnitude of gradient
                dir = atan2(dy, dx)
                mag = sqrt(dx * dx + dy * dy)

                # Assign direction to pixel
                idxDir = (dir + pi) * nDir / (2.0 * pi)
                if int(idxDir) == nDir: idxDir -= nDir

                # Soft assignment of direction with weights
                dirIdx = [0,0]
                dirWeight = [0,0]
                dirIdx[0] = int(idxDir)
                dirIdx[1] = (dirIdx[0] + 1) % nDir
                dirWeight[0] = 1.0 - (idxDir - dirIdx[0])
                dirWeight[1] = idxDir - dirIdx[0]

                # Calculate current position in one-dimensional array
                pos_idx = y * width + x
                hog[dirIdx[0]*width*height + pos_idx] = dirWeight[0] * mag
                hog[dirIdx[1] * width * height + pos_idx] = dirWeight[1] * mag

        for y in range(height):
            for x in range(width):
                Total = 0
                for i in range(nDir):
                    self._gradMap[i][y, x] = hog[i * width * height + y * width + x]
                    Total += hog[i * width * height + y * width + x]
                self._gradMap[nDir][y,x] = Total




if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: python %s <inputfile> \n' % sys.argv[0])
        raise SystemExit(1)
    img_to_process = RfdUtils(sys.argv[1], True)
    img_to_process.getHoG()
    img_to_process.showGradients()
