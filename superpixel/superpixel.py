from skimage import io
from skimage.segmentation import felzenszwalb, quickshift, slic, watershed, mark_boundaries
from skimage.filters import sobel
from skimage.color import rgb2gray
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt


"""
1. superpixel適用
2. 分割マスクデータ出力
3. 実際の結果出力
"""

img = io.imread("algae.jpg")

felzen_segments = felzenszwalb(img)
quick_segments = quickshift(img)
slic_segments = slic(img, start_label=1)
water_segments = watershed(sobel(rgb2gray(img)), markers=250)

plt.figure(figsize=(10, 10))
plt.rcParams["font.size"] = 15

plt.subplot(2, 2, 1)
plt.title("Felzenszwalb")
plt.imshow(mark_boundaries(img,felzen_segments))

plt.subplot(2, 2, 2)
plt.title("quickshift")
plt.imshow(mark_boundaries(img,quick_segments))

plt.subplot(2, 2, 3)
plt.title("slic")
plt.imshow(mark_boundaries(img,slic_segments))

plt.subplot(2, 2, 4)
plt.title("watershed")
plt.imshow(mark_boundaries(img,water_segments))

plt.savefig('algae_result.png')
plt.show()