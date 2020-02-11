import os

from PIL import Image, ImageDraw, ImageFilter

from .Directory import MASK_CROP_DATA_DIRECTORY, BRUSH_CROP_DATA_DIRECTORY, FEATHERING_DATA_DIRECTORY

cwd = os.getcwd()
target_brush = cwd + '/kmeans_hsv_algae/brush_crop_data'
target_mask = cwd + '/kmeans_hsv_algae/mask_crop_data'
brush_list = os.listdir(target_brush)
mask_list = os.listdir(target_mask)
brush_list.sort()
mask_list.sort()

for index, (brush, mask) in enumerate(zip(brush_list, mask_list)):
    im_rgba = Image.open(target_brush + '/' + brush)
    im_back = Image.new('RGBA', im_rgba.size, (0, 0, 0, 0))
    im_a = Image.open(target_mask + '/' + mask).convert('L')
    im_a_blur = im_a.filter(ImageFilter.GaussianBlur(7))

    im = Image.composite(im_rgba, im_back, im_a_blur)
    im.save('./kmeans_hsv_algae/feathering/result'+ str(index) +'.png')

def applyFeatheringProcessing(kernel_size):
    brush_list = os.listdir(BRUSH_CROP_DATA_DIRECTORY)
    mask_list = os.listdir(MASK_CROP_DATA_DIRECTORY)
    brush_list.sort()
    mask_list.sort()

    for index, (brush, mask) in enumerate(zip(brush_list, mask_list)):
        im_rgba = Image.open(BRUSH_CROP_DATA_DIRECTORY + '/' + brush)
        im_back = Image.new('RGBA', im_rgba.size, (0, 0, 0, 0))
        im_a = Image.open(MASK_CROP_DATA_DIRECTORY + '/' + mask).convert('L')
        im_a_blur = im_a.filter(ImageFilter.GaussianBlur(kernel_size))

        im = Image.composite(im_rgba, im_back, im_a_blur)
        im.save(FEATHERING_DATA_DIRECTORY + '/result'+ str(index) +'.png')