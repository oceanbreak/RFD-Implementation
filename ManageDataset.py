import Utils

from skimage.transform import integral as intg
from tkinter import Tk
from skimage import io
from skimage.color import rgb2gray
from skimage import img_as_float as imf
from tkinter import filedialog
from matplotlib import pyplot as plt
import numpy as np

root = Tk()
input_file = filedialog.askopenfilename()
root.destroy()
img = Utils.readImPatch(input_file)
#img = Utils.cropImg(img, (0,0), (20,3))
io.imshow(img)
plt.title('Input')
plt.show()

height, width  = img.shape
print(img.shape)

img_integral = intg.integral_image(img)
plt.imshow(img_integral)
plt.colorbar()
plt.show()

output_file = '.'.join(input_file.split('.')[:-1]) + '_integral.tif'
io.imsave(output_file, img_integral)

