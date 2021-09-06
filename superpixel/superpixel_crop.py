from skimage import io
from skimage.segmentation import slic, watershed
from skimage.filters import sobel
from skimage.color import rgb2gray
import numpy as np


img = io.imread("input1.png")
slic_segments = slic(img, start_label=1)
slic_array = np.array(slic_segments)
slic_flatten = slic_array.flatten()
slic_max = slic_flatten[np.argmax(slic_flatten)]

mask_array = np.empty((0,1), bool)

for i in range(slic_max+1):
    temp = slic_array == i
    mask_array = np.append(mask_array, [temp])
    print(mask_array)